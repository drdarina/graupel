import logging
from datetime import datetime, timezone, timedelta, tzinfo
from typing import List, Dict, Any, Optional

from .models import (
    MeteogramConfig,
    Location,
    WeatherModel,
    RawForecastData,
    DataPoint,
    HarmonizedForecastResponse,
    VerticalCloudLevel,
    VerticalCloudProfile,
    DetailedCloudForecast,
    VerticalCloudTransition,
    SunPeriod,
    VariableTimelineCoverage,
    ModelTimelineDiagnostics,
)
from .open_meteo import (
    DEFAULT_VARIABLES,
    VARIABLE_TO_OPENMETEO,
    OpenMeteoClient,
    to_open_meteo_model,
    get_model_catalog,
    CONVECTIVE_VARIABLES,
    WEATHER_ICON_VARIABLES,
    resolve_vertical_cloud_source,
)
from .harmonizer import Harmonizer
from .storage import Storage
from .timeline import (
    HOUR,
    build_hourly_timeline,
    canonical_timestamp,
    get_user_timezone,
    get_user_timezone_name,
    model_end_timestamp,
    parse_timestamp,
    validate_timestamp_sequence,
)

DEFAULT_PRESSURE_LEVELS = [
    1000,
    950,
    925,
    900,
    850,
    800,
    700,
    600,
    500,
    400,
    300,
    250,
    200,
    150,
    100,
]

# Cached vertical cloud profiles older than this are refetched. The regional
# models used for cloud chains publish new runs every 1-3 hours.
CLOUD_CACHE_MAX_AGE = timedelta(hours=1)

logger = logging.getLogger(__name__)


def estimate_altitude_m_asl(pressure_hpa: float) -> float:
    """Estimates altitude in meters ASL from pressure using the standard barometric formula."""
    return round(44330.0 * (1.0 - (pressure_hpa / 1013.25) ** 0.1903), 1)


import math


def get_local_tz(location: Location) -> timezone:
    """Resolve tzinfo for a location based on its timezone or longitude offset."""
    tz_str = getattr(location, "timezone", None)
    if tz_str and tz_str != "auto":
        try:
            from .timeline import resolve_timezone

            return resolve_timezone(tz_str)
        except Exception:
            pass

    if getattr(location, "longitude", None) is not None:
        approx_offset_hours = round(location.longitude / 15.0)
        return timezone(timedelta(hours=approx_offset_hours))

    return timezone.utc


def get_local_now(location: Location) -> datetime:
    """Resolve current local time for a location based on its timezone or longitude offset."""
    loc_tz = get_local_tz(location)
    return datetime.now(loc_tz)


def get_yesterday_midnight_utc(
    location: Optional[Location] = None,
    reference_time: Optional[datetime] = None,
    target_tz: Optional[tzinfo] = None,
) -> datetime:
    """Return the aware UTC datetime corresponding to 00:00 local time of the previous calendar day."""
    if reference_time is None:
        reference_time = datetime.now(timezone.utc)

    if target_tz is not None:
        loc_tz = target_tz
    elif (
        location is not None
        and getattr(location, "timezone", None)
        and location.timezone != "auto"
    ):
        loc_tz = get_local_tz(location)
    else:
        loc_tz = get_user_timezone()

    now_local = reference_time.astimezone(loc_tz)
    today_midnight_local = now_local.replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    yesterday_date = today_midnight_local.date() - timedelta(days=1)
    yesterday_midnight_local = datetime(
        yesterday_date.year,
        yesterday_date.month,
        yesterday_date.day,
        0,
        0,
        0,
        tzinfo=loc_tz,
    )
    return yesterday_midnight_local.astimezone(timezone.utc)


def get_hours_since_midnight(
    location: Optional[Location] = None,
    reference_time: Optional[datetime] = None,
    target_tz: Optional[tzinfo] = None,
) -> int:
    """Return absolute elapsed hours since local midnight of TODAY, including DST days."""
    if reference_time is None:
        reference_time = datetime.now(timezone.utc)
    if target_tz is not None:
        loc_tz = target_tz
    elif (
        location is not None
        and getattr(location, "timezone", None)
        and location.timezone != "auto"
    ):
        loc_tz = get_local_tz(location)
    else:
        loc_tz = get_user_timezone()
    now_local = reference_time.astimezone(loc_tz)
    today_midnight_local = now_local.replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    elapsed = (
        reference_time - today_midnight_local.astimezone(timezone.utc)
    ).total_seconds() / 3600.0
    return max(0, min(25, int(elapsed)))


def get_past_hours_for_yesterday(
    location: Optional[Location] = None,
    reference_time: Optional[datetime] = None,
    target_tz: Optional[tzinfo] = None,
) -> int:
    """Return past_hours count needed for API requests to cover back to yesterday 00:00 local time."""
    if reference_time is None:
        reference_time = datetime.now(timezone.utc)
    yesterday_midnight_utc = get_yesterday_midnight_utc(
        location, reference_time, target_tz=target_tz
    )
    elapsed = (
        reference_time - yesterday_midnight_utc
    ).total_seconds() / 3600.0
    return max(24, math.ceil(elapsed) + 2)


class ForecastService:
    def __init__(
        self,
        client: OpenMeteoClient = None,
        storage: Storage = None,
        cloud_cache_max_age: timedelta = CLOUD_CACHE_MAX_AGE,
    ):
        self.client = client or OpenMeteoClient()
        self.storage = storage or Storage()
        self.cloud_cache_max_age = cloud_cache_max_age

    def _is_cloud_cache_fresh(self, fetched_at: Optional[datetime]) -> bool:
        if fetched_at is None:
            return False
        age = datetime.now(timezone.utc) - fetched_at
        # A fetch time in the future (e.g. after a clock change) is stale.
        return timedelta(0) <= age < self.cloud_cache_max_age

    @staticmethod
    def _normalize_cached_cloud_forecast(
        cached: DetailedCloudForecast, model: WeatherModel
    ) -> DetailedCloudForecast:
        cached.profiles = sorted(
            [
                profile.model_copy(
                    update={
                        "timestamp": canonical_timestamp(profile.timestamp)
                    }
                )
                for profile in cached.profiles
            ],
            key=lambda profile: parse_timestamp(profile.timestamp),
        )
        validate_timestamp_sequence(
            [profile.timestamp for profile in cached.profiles],
            dataset_name=f"{model.name} cached vertical cloud timeline",
        )
        return cached

    def _extract_sun_periods(
        self, raw_response: Dict[str, Any]
    ) -> List[SunPeriod]:
        daily = raw_response.get("daily", {})
        times = daily.get("time", [])
        sunrises = daily.get("sunrise", [])
        sunsets = daily.get("sunset", [])
        periods: List[SunPeriod] = []
        response_timezone = raw_response.get("timezone") or "UTC"
        for day_str, sr, ss in zip(times, sunrises, sunsets):
            if sr and ss:
                periods.append(
                    SunPeriod(
                        day=day_str,
                        sunrise=canonical_timestamp(sr, response_timezone),
                        sunset=canonical_timestamp(ss, response_timezone),
                    )
                )
        return periods

    async def fetch_sun_phases(
        self,
        location: Location,
        forecast_days: int = 16,
        timezone_str: Optional[str] = None,
    ) -> List[SunPeriod]:
        """Fetch sunrise/sunset including yesterday via the existing best-match endpoint."""
        params = {
            "latitude": location.latitude,
            "longitude": location.longitude,
            "daily": "sunrise,sunset",
            # Two UTC past days also cover yesterday in every location timezone.
            "past_days": 2,
            # All data transport uses UTC. Location timezones are display-only.
            "timezone": "GMT",
            "forecast_days": min(16, max(1, forecast_days)),
            "models": "best_match",
        }
        raw = await self.client._get(params)
        return self._extract_sun_periods(raw)

    async def _fetch_raw_model_datasets(
        self,
        config: MeteogramConfig,
        model_name: str,
        horizon: int,
        variables: List[str],
        past_hours: Optional[int] = None,
    ) -> List[RawForecastData]:
        """Fetch raw data for specified variables for a specific model."""
        catalog_model = next(
            (m for m in get_model_catalog()
             if (m.id or to_open_meteo_model(m.name)) == to_open_meteo_model(model_name)),
            None,
        )
        openmeteo_vars = [
            VARIABLE_TO_OPENMETEO.get(v, v) for v in variables
            if Harmonizer.normalize_variable(v) not in CONVECTIVE_VARIABLES
            or (catalog_model and Harmonizer.normalize_variable(v) in catalog_model.supported_variables)
        ]
        if past_hours is None:
            user_tz = get_user_timezone()
            past_hours = get_past_hours_for_yesterday(target_tz=user_tz)

        params = {
            "latitude": config.location.latitude,
            "longitude": config.location.longitude,
            "hourly": ",".join(openmeteo_vars),
            "models": to_open_meteo_model(model_name),
            "forecast_hours": horizon,
            "past_hours": past_hours,
            # Avoid browser/local-time ambiguity and DST duplicate hours.
            "timezone": "GMT",
        }

        raw_response = await self.client._get(params)

        hourly = raw_response.get("hourly", {})
        hourly_units = raw_response.get("hourly_units", {})
        fetch_time = canonical_timestamp(datetime.now(timezone.utc))
        response_timezone = raw_response.get("timezone") or "UTC"
        timestamps = [
            canonical_timestamp(timestamp, response_timezone)
            for timestamp in hourly.get("time", [])
        ]
        validate_timestamp_sequence(
            timestamps,
            dataset_name=f"{model_name} API timeline",
        )

        raw_datasets: List[RawForecastData] = []
        # Preserve a requested variable even when the upstream response omits its
        # value array. The harmonizer will represent it as timestamped nulls.
        keys_to_process = list(dict.fromkeys(openmeteo_vars))

        for raw_var in keys_to_process:
            vals = hourly.get(raw_var, [])
            dataset = RawForecastData(
                model_name=model_name,
                fetch_time=fetch_time,
                forecast_run_time=None,
                timestamps=timestamps,
                values=vals,
                unit=hourly_units.get(raw_var, ""),
                variable=raw_var,
                max_horizon_hours=horizon,
            )
            raw_datasets.append(dataset)

        return raw_datasets

    async def _fetch_raw_model_data(
        self,
        config: MeteogramConfig,
        model_name: str,
        horizon: int,
        variable: str = "temperature",
        past_hours: Optional[int] = None,
    ) -> RawForecastData:
        """Fetch single raw dataset (for backward compatibility)."""
        datasets = await self._fetch_raw_model_datasets(
            config, model_name, horizon, [variable], past_hours=past_hours
        )
        if datasets:
            return datasets[0]
        return RawForecastData(
            model_name=model_name,
            fetch_time=canonical_timestamp(datetime.now(timezone.utc)),
            timestamps=[],
            values=[],
            unit="",
            variable=variable,
            max_horizon_hours=horizon,
        )

    async def get_merged_forecast(
        self,
        config: MeteogramConfig,
        variables: Optional[List[str]] = None,
        variable: Optional[str] = None,
        reference_time: Optional[datetime] = None,
    ) -> HarmonizedForecastResponse:
        """
        Fetches all models individually for requested variables, harmonizes raw data into standardized DataPoints,
        and returns both individual model series and a combined forecast series where transitions
        happen based on max_forecast_horizon_hours in the configuration.
        """
        if not config.model_chain:
            raise ValueError("Configuration contains no forecast models.")

        if variables is not None:
            target_variables = variables
        elif variable is not None:
            target_variables = [variable]
        else:
            target_variables = DEFAULT_VARIABLES + CONVECTIVE_VARIABLES + WEATHER_ICON_VARIABLES

        raw_datasets: List[RawForecastData] = []
        model_forecasts: Dict[str, List[DataPoint]] = {}
        fetch_errors: Dict[str, Exception] = {}

        # Fetch astronomical sunrise and sunset periods directly for up to 16 days
        sun_phases: List[SunPeriod] = []
        if config.location:
            try:
                sun_phases = await self.fetch_sun_phases(
                    config.location, forecast_days=16
                )
            except Exception as e:
                import logging

                logging.getLogger(__name__).warning(
                    f"Failed to fetch sun phases: {e}"
                )

        now_utc = (
            parse_timestamp(reference_time)
            if reference_time is not None
            else datetime.now(timezone.utc)
        )
        user_tz = get_user_timezone()
        user_tz_name = get_user_timezone_name()
        yesterday_midnight_utc = get_yesterday_midnight_utc(
            config.location, now_utc, target_tz=user_tz
        )
        past_hours = get_past_hours_for_yesterday(
            config.location, now_utc, target_tz=user_tz
        )

        # Fetch and harmonize each model
        for weather_model in config.model_chain:
            try:
                model_raw_list = await self._fetch_raw_model_datasets(
                    config,
                    weather_model.name,
                    weather_model.max_forecast_horizon_hours,
                    variables=target_variables,
                    past_hours=past_hours,
                )
                raw_datasets.extend(model_raw_list)

                harmonized_pts: List[DataPoint] = []
                for raw_data in model_raw_list:
                    pts = Harmonizer.harmonize_raw_data(raw_data)
                    harmonized_pts.extend(pts)

                model_forecasts[weather_model.name] = harmonized_pts
            except Exception as e:
                import logging

                logging.getLogger(__name__).warning(
                    f"Failed to fetch forecast data for model '{weather_model.name}': {e}"
                )
                fetch_errors[weather_model.name] = e
                model_forecasts[weather_model.name] = []

        if len(fetch_errors) == len(config.model_chain) and not raw_datasets:
            first_err = next(iter(fetch_errors.values()))
            raise first_err

        if not raw_datasets:
            return HarmonizedForecastResponse(
                combined_forecast=[],
                model_forecasts=model_forecasts,
                raw_data=[],
                sun_phases=sun_phases,
                user_timezone=user_tz_name,
            )

        # Establish one absolute timeline from the shared request origin and the
        # configured model horizon. Source array lengths never define the window.
        all_timestamps_set: set[str] = set()
        for d in raw_datasets:
            all_timestamps_set.update(d.timestamps)

        source_timestamps = sorted(
            all_timestamps_set, key=lambda value: parse_timestamp(value)
        )

        if not source_timestamps:
            return HarmonizedForecastResponse(
                combined_forecast=[],
                model_forecasts=model_forecasts,
                raw_data=raw_datasets,
                sun_phases=sun_phases,
                user_timezone=user_tz_name,
            )

        # The visible timeline always begins at local midnight of the previous calendar day
        timeline_start_dt = yesterday_midnight_utc
        t_now = now_utc.replace(minute=0, second=0, microsecond=0)
        t_now_iso = canonical_timestamp(t_now)
        max_chain_horizon = max(
            (m.max_forecast_horizon_hours for m in config.model_chain),
            default=0,
        )
        timeline_end_iso = model_end_timestamp(t_now, max_chain_horizon)
        all_timestamps = build_hourly_timeline(
            canonical_timestamp(timeline_start_dt), timeline_end_iso
        )

        failed_models = [
            m
            for m in config.model_chain
            if m.name in fetch_errors
            or not any(
                p.value is not None for p in model_forecasts.get(m.name, [])
            )
        ]
        # Determine normalized target variable names to look for
        normalized_vars = list(
            dict.fromkeys(
                [Harmonizer.normalize_variable(v) for v in target_variables]
            )
        )

        # Track default or discovered units per normalized variable
        DEFAULT_UNITS = {
            "temperature": "°C",
            "apparent_temperature": "°C",
            "wind_speed": "km/h",
            "wind_gusts": "km/h",
            "wind_direction": "°",
            "cloud_cover": "%",
            "precipitation": "mm",
            "precipitation_probability": "%",
            "cape": "J/kg",
            "convective_inhibition": "J/kg",
            "lightning_potential": "J/kg",
        }
        var_units: Dict[str, str] = {}
        for d in raw_datasets:
            norm_v = Harmonizer.normalize_variable(d.variable)
            if d.unit and norm_v not in var_units:
                var_units[norm_v] = "J/kg" if norm_v in CONVECTIVE_VARIABLES else d.unit

        # Clip each model to its configured absolute half-open interval and
        # record actual variable availability separately from that model range.
        timeline_diagnostics: Dict[str, ModelTimelineDiagnostics] = {}
        for weather_model in config.model_chain:
            expected_end = parse_timestamp(
                model_end_timestamp(
                    t_now, weather_model.max_forecast_horizon_hours
                )
            )
            model_points = [
                point
                for point in model_forecasts.get(weather_model.name, [])
                if timeline_start_dt
                <= parse_timestamp(point.timestamp)
                < expected_end
            ]
            model_points.sort(
                key=lambda point: (
                    parse_timestamp(point.timestamp),
                    point.variable,
                )
            )
            model_forecasts[weather_model.name] = model_points

            variable_diagnostics: Dict[str, VariableTimelineCoverage] = {}
            expected_last = (
                expected_end - HOUR
                if weather_model.max_forecast_horizon_hours > 0
                else None
            )
            for norm_var in normalized_vars:
                variable_points = [
                    point
                    for point in model_points
                    if point.variable == norm_var
                    and parse_timestamp(point.timestamp) >= t_now
                ]
                variable_points.sort(
                    key=lambda point: parse_timestamp(point.timestamp)
                )
                valid_points = [
                    point
                    for point in variable_points
                    if point.value is not None
                ]
                variable_times = [
                    point.timestamp for point in variable_points
                ]
                spacing_anomalies = validate_timestamp_sequence(
                    variable_times,
                    dataset_name=f"{weather_model.name} / {norm_var}",
                )
                valid_time_set = {
                    parse_timestamp(point.timestamp) for point in valid_points
                }
                expected_sample_count = max(
                    0, weather_model.max_forecast_horizon_hours
                )
                complete = bool(
                    expected_last is None
                    or (
                        expected_last in valid_time_set
                        and len(valid_time_set) == expected_sample_count
                        and not spacing_anomalies
                    )
                )

                coverage = VariableTimelineCoverage(
                    variable=norm_var,
                    sample_count=len(variable_points),
                    valid_sample_count=len(valid_points),
                    first_timestamp=(
                        variable_points[0].timestamp
                        if variable_points
                        else None
                    ),
                    last_timestamp=(
                        variable_points[-1].timestamp
                        if variable_points
                        else None
                    ),
                    first_valid_timestamp=(
                        valid_points[0].timestamp if valid_points else None
                    ),
                    last_valid_timestamp=(
                        valid_points[-1].timestamp if valid_points else None
                    ),
                    expected_start=t_now_iso,
                    expected_end_exclusive=canonical_timestamp(expected_end),
                    expected_last_timestamp=(
                        canonical_timestamp(expected_last)
                        if expected_last is not None
                        else None
                    ),
                    complete=complete,
                    spacing_anomalies=spacing_anomalies,
                )
                variable_diagnostics[norm_var] = coverage

                if not coverage.complete and norm_var not in WEATHER_ICON_VARIABLES and not (norm_var in CONVECTIVE_VARIABLES and not variable_points):
                    import logging

                    logging.getLogger(__name__).warning(
                        "%s / %s: expected hourly data through %s "
                        "(model boundary %s exclusive), received valid data "
                        "only through %s",
                        weather_model.name,
                        norm_var,
                        coverage.expected_last_timestamp,
                        coverage.expected_end_exclusive,
                        coverage.last_valid_timestamp,
                    )

            timeline_diagnostics[weather_model.name] = (
                ModelTimelineDiagnostics(
                    model=weather_model.name,
                    expected_start=t_now_iso,
                    expected_end_exclusive=canonical_timestamp(expected_end),
                    variables=variable_diagnostics,
                )
            )

        # Populate empty models in model_forecasts so every model in chain is represented
        for weather_model in config.model_chain:
            if not model_forecasts.get(weather_model.name):
                empty_pts: List[DataPoint] = []
                for ts in all_timestamps:
                    current_time = parse_timestamp(ts)
                    hrs = (
                        0.0
                        if current_time < t_now
                        else (current_time - t_now).total_seconds() / 3600.0
                    )
                    if hrs < weather_model.max_forecast_horizon_hours:
                        for norm_var in normalized_vars:
                            unit = var_units.get(
                                norm_var, DEFAULT_UNITS.get(norm_var, "")
                            )
                            empty_pts.append(
                                DataPoint(
                                    timestamp=ts,
                                    value=None,
                                    unit=unit,
                                    variable=norm_var,
                                    model=weather_model.name,
                                )
                            )
                model_forecasts[weather_model.name] = empty_pts

        # Determine active models in chain (span > 0)
        active_chain: List[WeatherModel] = []
        for idx, m in enumerate(config.model_chain):
            prev_h = (
                config.model_chain[idx - 1].max_forecast_horizon_hours
                if idx > 0
                else 0
            )
            if m.max_forecast_horizon_hours > prev_h:
                active_chain.append(m)

        effective_chain = active_chain if active_chain else config.model_chain

        forecast_lookup: Dict[str, Dict[tuple[str, str], DataPoint]] = {}
        valid_times_by_model: Dict[str, set[str]] = {}
        valid_variables_by_model: Dict[str, set[str]] = {}
        for weather_model in effective_chain:
            points = model_forecasts.get(weather_model.name, [])
            point_lookup: Dict[tuple[str, str], DataPoint] = {}
            valid_times: set[str] = set()
            valid_variables: set[str] = set()
            for point in points:
                canonical_time = canonical_timestamp(point.timestamp)
                point_lookup[(canonical_time, point.variable)] = point
                if point.value is not None:
                    valid_times.add(canonical_time)
                    valid_variables.add(point.variable)
            forecast_lookup[weather_model.name] = point_lookup
            valid_times_by_model[weather_model.name] = valid_times
            valid_variables_by_model[weather_model.name] = valid_variables

        combined_forecast: List[DataPoint] = []

        for ts in all_timestamps:
            current_time = parse_timestamp(ts)

            # Determine primary model for this horizon:
            # Past hours before t_now are assigned to the primary model (effective_chain[0])
            if current_time < t_now:
                primary_model = effective_chain[0]
            else:
                hours_from_now = (
                    current_time - t_now
                ).total_seconds() / 3600.0
                primary_model = None
                for weather_model in effective_chain:
                    if (
                        hours_from_now
                        < weather_model.max_forecast_horizon_hours
                    ):
                        primary_model = weather_model
                        break
                if primary_model is None and effective_chain:
                    primary_model = effective_chain[-1]

            primary_has_data_at_ts = bool(
                primary_model
                and ts in valid_times_by_model.get(primary_model.name, set())
            )

            # If the selected primary model has ended its forecast run or has no valid data at this timestamp
            # (and is not an outright failed model whose slice must remain empty), advance to the next
            # available model in effective_chain that has valid data at this timestamp.
            if (
                current_time >= t_now
                and primary_model
                and not primary_has_data_at_ts
                and primary_model not in failed_models
            ):
                primary_idx = (
                    effective_chain.index(primary_model)
                    if primary_model in effective_chain
                    else -1
                )
                candidates = (
                    effective_chain[primary_idx + 1 :]
                    + effective_chain[:primary_idx]
                )
                for candidate in candidates:
                    if candidate in failed_models:
                        continue
                    if ts in valid_times_by_model.get(candidate.name, set()):
                        primary_model = candidate
                        break

            for norm_var in normalized_vars:
                selected_point = None

                if primary_model:
                    pt = forecast_lookup.get(primary_model.name, {}).get(
                        (ts, norm_var)
                    )
                    if pt is not None and pt.value is not None:
                        selected_point = pt

                # Fallback check:
                # If primary model has valid data overall for this variable, but is missing a point
                # (e.g. boundary handover between adjacent models), allow fallback.
                # However, if the primary model has NO valid data for this variable (or was empty/failed),
                # do NOT fallback — leave this model's specified range empty!
                if selected_point is None and primary_model and current_time >= t_now and norm_var not in CONVECTIVE_VARIABLES + WEATHER_ICON_VARIABLES:
                    primary_has_var_data = norm_var in valid_variables_by_model.get(
                        primary_model.name, set()
                    )
                    if primary_has_var_data:
                        for weather_model in effective_chain:
                            if weather_model.name == primary_model.name:
                                continue
                            pt = forecast_lookup.get(weather_model.name, {}).get(
                                (ts, norm_var)
                            )
                            if pt is not None and pt.value is not None:
                                selected_point = pt
                                break

                # If still no point, emit an empty DataPoint with value=None for the primary model
                if selected_point is None and primary_model:
                    unit = var_units.get(
                        norm_var, DEFAULT_UNITS.get(norm_var, "")
                    )
                    selected_point = DataPoint(
                        timestamp=ts,
                        value=None,
                        unit=unit,
                        variable=norm_var,
                        model=primary_model.name,
                    )

                if selected_point is not None:
                    combined_forecast.append(selected_point)

        # Vertical Cloud Profile Source Chain Execution
        vertical_cloud_forecast = None
        vertical_cloud_transitions: List[VerticalCloudTransition] = []
        vertical_cloud_model_forecasts: Dict[str, DetailedCloudForecast] = {}

        cloud_chain = config.cloud_model_chain
        if cloud_chain is None:
            from .open_meteo import get_default_cloud_model

            cloud_chain = [get_default_cloud_model()]

        if cloud_chain and len(cloud_chain) > 0 and all_timestamps:
            try:
                # 1. Fetch each model in cloud_chain
                for c_model in cloud_chain:
                    cid = c_model.id or to_open_meteo_model(c_model.name)
                    c_forecast = (
                        await self.fetch_vertical_cloud_model_forecast(
                            config.location, c_model
                        )
                    )
                    vertical_cloud_model_forecasts[c_model.name] = c_forecast
                    if cid:
                        vertical_cloud_model_forecasts[cid] = c_forecast

                # 2. Filter active models (non-zero span)
                active_cloud_chain = [
                    m
                    for idx, m in enumerate(cloud_chain)
                    if m.max_forecast_horizon_hours
                    > (
                        cloud_chain[idx - 1].max_forecast_horizon_hours
                        if idx > 0
                        else 0
                    )
                ]
                effective_cloud_chain = (
                    active_cloud_chain if active_cloud_chain else cloud_chain
                )

                composite_profiles: List[VerticalCloudProfile] = []
                last_c_model_id: Optional[str] = None

                cloud_profile_lookup: Dict[str, Dict[str, VerticalCloudProfile]] = {}
                for key, forecast in vertical_cloud_model_forecasts.items():
                    cloud_profile_lookup[key] = {
                        canonical_timestamp(profile.timestamp): profile
                        for profile in forecast.profiles
                    }

                for ts in all_timestamps:
                    current_dt = parse_timestamp(ts)
                    if current_dt < t_now:
                        primary_c_model = effective_cloud_chain[0]
                    else:
                        hours_from_now = (
                            current_dt - t_now
                        ).total_seconds() / 3600.0
                        primary_c_model = None
                        for c_m in effective_cloud_chain:
                            if (
                                hours_from_now
                                < c_m.max_forecast_horizon_hours
                            ):
                                primary_c_model = c_m
                                break

                    if primary_c_model is not None:
                        cid = primary_c_model.id or to_open_meteo_model(
                            primary_c_model.name
                        )
                        matching_prof = (
                            cloud_profile_lookup.get(primary_c_model.name, {}).get(ts)
                            or cloud_profile_lookup.get(cid, {}).get(ts)
                        )
                        if matching_prof:
                            composite_profiles.append(
                                VerticalCloudProfile(
                                    timestamp=ts,
                                    source_model_id=cid,
                                    source_model_name=primary_c_model.name,
                                    source_run=matching_prof.source_run,
                                    levels=matching_prof.levels,
                                )
                            )
                            if (
                                last_c_model_id is not None
                                and last_c_model_id != cid
                            ):
                                vertical_cloud_transitions.append(
                                    VerticalCloudTransition(
                                        timestamp=ts,
                                        from_model=last_c_model_id,
                                        to_model=cid,
                                    )
                                )
                            last_c_model_id = cid
                        else:
                            last_c_model_id = None
                    else:
                        last_c_model_id = None

                vertical_cloud_forecast = DetailedCloudForecast(
                    location=config.location, profiles=composite_profiles
                )
            except Exception as e:
                import logging

                logging.getLogger(__name__).warning(
                    f"Failed to resolve vertical cloud profile: {e}"
                )

        return HarmonizedForecastResponse(
            combined_forecast=combined_forecast,
            model_forecasts=model_forecasts,
            raw_data=raw_datasets,
            vertical_cloud_forecast=vertical_cloud_forecast,
            vertical_cloud_transitions=vertical_cloud_transitions,
            vertical_cloud_model_forecasts=vertical_cloud_model_forecasts,
            sun_phases=sun_phases,
            forecast_start_time=t_now_iso,
            timeline_start=canonical_timestamp(timeline_start_dt),
            timeline_end=timeline_end_iso,
            user_timezone=user_tz_name,
            timeline_diagnostics=timeline_diagnostics,
        )

    async def fetch_vertical_cloud_model_forecast(
        self, location: Location, model: WeatherModel
    ) -> DetailedCloudForecast:
        """
        Fetches the vertical cloud pressure-level forecast for a single model
        across its full horizon.

        A cached copy is reused while it is younger than
        ``cloud_cache_max_age``. Older copies are refetched; if that fetch
        fails or returns nothing, the stale copy is returned instead.
        """
        model_id = model.id or to_open_meteo_model(model.name)
        entry = self.storage.read_vertical_cloud_forecast_entry(
            location, model_id
        )
        cached, fetched_at = entry if entry else (None, None)
        if cached is not None and self._is_cloud_cache_fresh(fetched_at):
            return self._normalize_cached_cloud_forecast(cached, model)

        levels_to_fetch = model.pressure_levels_hpa or DEFAULT_PRESSURE_LEVELS
        horizon = (
            model.max_forecast_horizon_hours or model.max_forecast_hours or 72
        )

        var_list = []
        for p in levels_to_fetch:
            var_list.append(f"cloud_cover_{p}hPa")
            var_list.append(f"geopotential_height_{p}hPa")

        user_tz = get_user_timezone()
        past_hours = get_past_hours_for_yesterday(target_tz=user_tz)
        params = {
            "latitude": location.latitude,
            "longitude": location.longitude,
            "hourly": ",".join(var_list),
            "models": to_open_meteo_model(model.name),
            "forecast_hours": horizon,
            "past_hours": past_hours,
            "timezone": "GMT",
        }

        try:
            raw_response = await self.client._get(params)
        except Exception as error:
            logger.warning(
                "Failed to fetch vertical cloud profile for %s: %s",
                model.name,
                error,
            )
            raw_response = {}

        hourly = raw_response.get("hourly", {})
        response_timezone = raw_response.get("timezone") or "UTC"
        timestamps = [
            canonical_timestamp(timestamp, response_timezone)
            for timestamp in hourly.get("time", [])
        ]
        validate_timestamp_sequence(
            timestamps,
            dataset_name=f"{model.name} vertical cloud timeline",
        )

        profiles: List[VerticalCloudProfile] = []

        for idx, ts in enumerate(timestamps):
            level_objs: List[VerticalCloudLevel] = []
            for p in levels_to_fetch:
                cc_vals = hourly.get(f"cloud_cover_{p}hPa", [])
                gh_vals = hourly.get(f"geopotential_height_{p}hPa", [])

                cc_val = cc_vals[idx] if idx < len(cc_vals) else None
                gh_val = gh_vals[idx] if idx < len(gh_vals) else None

                alt_asl = (
                    float(gh_val)
                    if gh_val is not None
                    else estimate_altitude_m_asl(p)
                )
                cc_percent = float(cc_val) if cc_val is not None else None

                level_objs.append(
                    VerticalCloudLevel(
                        pressure_hpa=float(p),
                        altitude_m_asl=alt_asl,
                        cloud_cover_percent=cc_percent,
                    )
                )

            # Sort levels by altitude ascending
            level_objs.sort(
                key=lambda l: (
                    l.altitude_m_asl if l.altitude_m_asl is not None else 0.0
                )
            )

            profiles.append(
                VerticalCloudProfile(
                    timestamp=ts,
                    source_model_id=model_id,
                    source_model_name=model.name,
                    levels=level_objs,
                )
            )

        forecast_result = DetailedCloudForecast(
            location=location, profiles=profiles
        )
        if profiles:
            self.storage.save_vertical_cloud_forecast(
                location, model_id, forecast_result
            )
            return forecast_result

        if cached is not None:
            logger.warning(
                "Using stale cached vertical cloud profile for %s "
                "(fetched %s)",
                model.name,
                fetched_at.isoformat() if fetched_at else "at unknown time",
            )
            return self._normalize_cached_cloud_forecast(cached, model)

        return forecast_result

    async def get_vertical_cloud_forecast(
        self,
        location: Location,
        timestamps: List[str],
        available_models: List[WeatherModel],
        start_time: Optional[str] = None,
    ) -> tuple[DetailedCloudForecast, List[VerticalCloudTransition]]:
        """
        Constructs a continuous vertical cloud forecast across requested timestamps by automatically
        selecting the optimal vertical cloud model for each timestamp.
        """
        if not timestamps:
            return DetailedCloudForecast(location=location, profiles=[]), []

        canonical_timestamps = sorted(
            {canonical_timestamp(timestamp) for timestamp in timestamps},
            key=lambda timestamp: parse_timestamp(timestamp),
        )
        validate_timestamp_sequence(
            canonical_timestamps,
            dataset_name="requested vertical cloud timeline",
        )
        ref_start_time = canonical_timestamp(
            start_time or canonical_timestamps[0]
        )
        model_forecast_cache: Dict[str, DetailedCloudForecast] = {}

        composite_profiles: List[VerticalCloudProfile] = []
        transitions: List[VerticalCloudTransition] = []
        last_model_id: Optional[str] = None

        for ts in canonical_timestamps:
            best_model = resolve_vertical_cloud_source(
                location, ts, available_models, start_time=ref_start_time
            )
            if best_model is None:
                # No detailed model available for this timestamp
                last_model_id = None
                continue

            best_model_id = best_model.id or to_open_meteo_model(
                best_model.name
            )

            if best_model_id not in model_forecast_cache:
                m_forecast = await self.fetch_vertical_cloud_model_forecast(
                    location, best_model
                )
                model_forecast_cache[best_model_id] = m_forecast

            m_forecast = model_forecast_cache[best_model_id]
            matching_profile = next(
                (
                    p
                    for p in m_forecast.profiles
                    if canonical_timestamp(p.timestamp) == ts
                ),
                None,
            )

            if matching_profile:
                # Clone profile ensuring source_model_id and source_model_name are preserved
                prof_copy = VerticalCloudProfile(
                    timestamp=ts,
                    source_model_id=best_model_id,
                    source_model_name=best_model.name,
                    source_run=matching_profile.source_run,
                    levels=matching_profile.levels,
                )
                composite_profiles.append(prof_copy)

                if (
                    last_model_id is not None
                    and last_model_id != best_model_id
                ):
                    transitions.append(
                        VerticalCloudTransition(
                            timestamp=ts,
                            from_model=last_model_id,
                            to_model=best_model_id,
                        )
                    )
                last_model_id = best_model_id
            else:
                last_model_id = None

        return (
            DetailedCloudForecast(
                location=location, profiles=composite_profiles
            ),
            transitions,
        )
