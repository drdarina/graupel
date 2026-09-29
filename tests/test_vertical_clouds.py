import pytest
from datetime import datetime, timezone
from graupel.data.models import (
    Location,
    WeatherModel,
    DetailedCloudForecast,
    VerticalCloudProfile,
    VerticalCloudLevel,
)
from graupel.data.open_meteo import (
    resolve_vertical_cloud_source,
    is_location_in_model_coverage,
)
from graupel.data.service import ForecastService, estimate_altitude_m_asl
from graupel.data.storage import Storage


def test_location_coverage_checks():
    ch_loc = Location(name="Zurich", latitude=47.376, longitude=8.541)
    berlin_loc = Location(name="Berlin", latitude=52.52, longitude=13.405)
    ny_loc = Location(name="New York", latitude=40.712, longitude=-74.006)

    m_ch = WeatherModel(
        name="ICON-CH1",
        id="meteoswiss_icon_ch1",
        region="Switzerland / Alps",
        supports_vertical_cloud_profile=True,
    )
    m_eu = WeatherModel(
        name="ICON-EU",
        id="icon_eu",
        region="Europe",
        supports_vertical_cloud_profile=True,
    )
    m_global = WeatherModel(
        name="GFS Seamless",
        id="gfs_seamless",
        region="Global",
        supports_vertical_cloud_profile=True,
    )

    assert is_location_in_model_coverage(m_ch, ch_loc) is True
    assert is_location_in_model_coverage(m_ch, berlin_loc) is False

    assert is_location_in_model_coverage(m_eu, ch_loc) is True
    assert is_location_in_model_coverage(m_eu, berlin_loc) is True
    assert is_location_in_model_coverage(m_eu, ny_loc) is False

    assert is_location_in_model_coverage(m_global, ch_loc) is True
    assert is_location_in_model_coverage(m_global, ny_loc) is True


def test_resolve_vertical_cloud_source():
    ch_loc = Location(name="Zurich", latitude=47.376, longitude=8.541)
    berlin_loc = Location(name="Berlin", latitude=52.52, longitude=13.405)

    std_10_levels = [1000, 950, 925, 900, 850, 800, 700, 600, 500, 400]
    m_ch1 = WeatherModel(
        name="ICON-CH1",
        id="meteoswiss_icon_ch1",
        region="Switzerland / Alps",
        supports_vertical_cloud_profile=True,
        spatial_resolution_km=1.1,
        max_forecast_horizon_hours=33,
        pressure_levels_hpa=std_10_levels,
    )
    m_ch2 = WeatherModel(
        name="ICON-CH2",
        id="meteoswiss_icon_ch2",
        region="Switzerland / Alps",
        supports_vertical_cloud_profile=True,
        spatial_resolution_km=2.1,
        max_forecast_horizon_hours=120,
        pressure_levels_hpa=std_10_levels,
    )
    m_global = WeatherModel(
        name="GFS Seamless",
        id="gfs_seamless",
        region="Global",
        supports_vertical_cloud_profile=True,
        spatial_resolution_km=13.0,
        max_forecast_horizon_hours=384,
        pressure_levels_hpa=std_10_levels,
    )

    models = [m_ch1, m_ch2, m_global]
    start_time = "2026-09-11T00:00:00Z"

    # 1. Regional detailed model available for Switzerland at +10h
    best_0_33 = resolve_vertical_cloud_source(
        ch_loc, "2026-09-11T10:00:00Z", models, start_time=start_time
    )
    assert best_0_33 is not None
    assert best_0_33.id == "meteoswiss_icon_ch1"

    # 2. Regional horizon exceeded (+40h) -> falls back to next regional ICON-CH2
    best_40 = resolve_vertical_cloud_source(
        ch_loc, "2026-09-12T16:00:00Z", models, start_time=start_time
    )
    assert best_40 is not None
    assert best_40.id == "meteoswiss_icon_ch2"

    # 3. Horizon exceeded for all regional (+150h) -> falls back to global
    best_150 = resolve_vertical_cloud_source(
        ch_loc, "2026-09-17T06:00:00Z", models, start_time=start_time
    )
    assert best_150 is not None
    assert best_150.id == "gfs_seamless"

    # 4. Regional model outside geographical coverage (Berlin) -> skips ICON-CH1/CH2, picks global
    best_berlin = resolve_vertical_cloud_source(
        berlin_loc, "2026-09-11T10:00:00Z", models, start_time=start_time
    )
    assert best_berlin is not None
    assert best_berlin.id == "gfs_seamless"

    # 5. No detailed model available
    no_vcloud_models = [
        WeatherModel(
            name="Custom", id="c1", supports_vertical_cloud_profile=False
        )
    ]
    assert (
        resolve_vertical_cloud_source(
            ch_loc,
            "2026-09-11T10:00:00Z",
            no_vcloud_models,
            start_time=start_time,
        )
        is None
    )


def test_barometric_altitude_estimate():
    # 1000 hPa ~ 110m
    alt1000 = estimate_altitude_m_asl(1000)
    assert 50 <= alt1000 <= 200

    # 500 hPa ~ 5570m
    alt500 = estimate_altitude_m_asl(500)
    assert 5000 <= alt500 <= 6000


@pytest.mark.asyncio
async def test_vertical_cloud_storage_and_service():
    storage = Storage(":memory:")
    service = ForecastService(storage=storage)

    loc = Location(
        name="Zurich", latitude=47.376, longitude=8.541, elevation=408
    )

    # Save mock vertical cloud forecast to storage
    mock_forecast = DetailedCloudForecast(
        location=loc,
        profiles=[
            VerticalCloudProfile(
                timestamp="2026-09-11T00:00:00Z",
                source_model_id="meteoswiss_icon_ch1",
                source_model_name="ICON-CH1",
                levels=[
                    VerticalCloudLevel(
                        pressure_hpa=1000,
                        altitude_m_asl=110,
                        cloud_cover_percent=20,
                    ),
                    VerticalCloudLevel(
                        pressure_hpa=850,
                        altitude_m_asl=1450,
                        cloud_cover_percent=80,
                    ),
                    VerticalCloudLevel(
                        pressure_hpa=500,
                        altitude_m_asl=5570,
                        cloud_cover_percent=40,
                    ),
                ],
            )
        ],
    )

    storage.save_vertical_cloud_forecast(
        loc, "meteoswiss_icon_ch1", mock_forecast
    )

    read_back = storage.read_vertical_cloud_forecast(
        loc, "meteoswiss_icon_ch1"
    )
    assert read_back is not None
    assert len(read_back.profiles) == 1
    assert read_back.profiles[0].levels[1].cloud_cover_percent == 80


@pytest.mark.asyncio
async def test_get_vertical_cloud_forecast_source_transitions():
    storage = Storage(":memory:")
    service = ForecastService(storage=storage)

    loc = Location(
        name="Zurich", latitude=47.376, longitude=8.541, elevation=408
    )

    std_10_levels = [1000, 950, 925, 900, 850, 800, 700, 600, 500, 400]
    m_ch1 = WeatherModel(
        name="ICON-CH1",
        id="meteoswiss_icon_ch1",
        region="Switzerland / Alps",
        supports_vertical_cloud_profile=True,
        spatial_resolution_km=1.1,
        max_forecast_horizon_hours=2,
        pressure_levels_hpa=std_10_levels,
    )
    m_global = WeatherModel(
        name="GFS Seamless",
        id="gfs_seamless",
        region="Global",
        supports_vertical_cloud_profile=True,
        spatial_resolution_km=13.0,
        max_forecast_horizon_hours=10,
        pressure_levels_hpa=std_10_levels,
    )

    # Seed mock storage forecasts for both models
    storage.save_vertical_cloud_forecast(
        loc,
        "meteoswiss_icon_ch1",
        DetailedCloudForecast(
            location=loc,
            profiles=[
                VerticalCloudProfile(
                    timestamp="2026-09-11T00:00:00Z",
                    source_model_id="meteoswiss_icon_ch1",
                    source_model_name="ICON-CH1",
                    levels=[
                        VerticalCloudLevel(
                            pressure_hpa=850,
                            altitude_m_asl=1450,
                            cloud_cover_percent=50,
                        )
                    ],
                ),
                VerticalCloudProfile(
                    timestamp="2026-09-11T01:00:00Z",
                    source_model_id="meteoswiss_icon_ch1",
                    source_model_name="ICON-CH1",
                    levels=[
                        VerticalCloudLevel(
                            pressure_hpa=850,
                            altitude_m_asl=1450,
                            cloud_cover_percent=60,
                        )
                    ],
                ),
            ],
        ),
    )

    storage.save_vertical_cloud_forecast(
        loc,
        "gfs_seamless",
        DetailedCloudForecast(
            location=loc,
            profiles=[
                VerticalCloudProfile(
                    timestamp="2026-09-11T03:00:00Z",
                    source_model_id="gfs_seamless",
                    source_model_name="GFS Seamless",
                    levels=[
                        VerticalCloudLevel(
                            pressure_hpa=850,
                            altitude_m_asl=1450,
                            cloud_cover_percent=30,
                        )
                    ],
                ),
            ],
        ),
    )

    timestamps = [
        "2026-09-11T00:00:00Z",
        "2026-09-11T01:00:00Z",
        "2026-09-11T03:00:00Z",
    ]
    vforecast, transitions = await service.get_vertical_cloud_forecast(
        loc, timestamps, [m_ch1, m_global]
    )

    assert len(vforecast.profiles) == 3
    assert vforecast.profiles[0].source_model_id == "meteoswiss_icon_ch1"
    assert vforecast.profiles[2].source_model_id == "gfs_seamless"
    assert len(transitions) == 1
    assert transitions[0].from_model == "meteoswiss_icon_ch1"
    assert transitions[0].to_model == "gfs_seamless"


# ---------------------------------------------------------------------------
# Vertical cloud cache expiry
# ---------------------------------------------------------------------------

from datetime import timedelta  # noqa: E402
import sqlite3  # noqa: E402

from graupel.data.service import CLOUD_CACHE_MAX_AGE  # noqa: E402

CACHE_LOC = Location(name="Zugspitze", latitude=47.4211, longitude=10.9853)
CACHE_MODEL = WeatherModel(
    name="ICON-D2",
    id="icon_d2",
    max_forecast_horizon_hours=2,
    pressure_levels_hpa=[850],
)


def _cloud_forecast(cover: float) -> DetailedCloudForecast:
    return DetailedCloudForecast(
        location=CACHE_LOC,
        profiles=[
            VerticalCloudProfile(
                timestamp="2026-09-11T00:00:00Z",
                source_model_id="icon_d2",
                source_model_name="ICON-D2",
                levels=[
                    VerticalCloudLevel(
                        pressure_hpa=850,
                        altitude_m_asl=1450,
                        cloud_cover_percent=cover,
                    )
                ],
            )
        ],
    )


class _CloudClient:
    """Fake Open-Meteo client returning one hour of 850 hPa cloud cover."""

    def __init__(self, cover: float = 90.0, fail: bool = False):
        self.cover = cover
        self.fail = fail
        self.calls = 0

    async def _get(self, params):
        self.calls += 1
        if self.fail:
            raise RuntimeError("network down")
        return {
            "timezone": "GMT",
            "hourly": {
                "time": ["2026-09-11T00:00"],
                "cloud_cover_850hPa": [self.cover],
                "geopotential_height_850hPa": [1500.0],
            },
        }


def _cover(forecast: DetailedCloudForecast) -> float:
    return forecast.profiles[0].levels[0].cloud_cover_percent


def test_default_cloud_cache_max_age_is_one_hour():
    assert CLOUD_CACHE_MAX_AGE == timedelta(hours=1)


def test_cloud_cache_entry_records_fetch_time():
    storage = Storage(":memory:")
    fetched = datetime(2026, 9, 11, 6, 0, tzinfo=timezone.utc)
    storage.save_vertical_cloud_forecast(
        CACHE_LOC, "icon_d2", _cloud_forecast(10), fetched_at=fetched
    )

    forecast, fetched_at = storage.read_vertical_cloud_forecast_entry(
        CACHE_LOC, "icon_d2"
    )
    assert _cover(forecast) == 10
    assert fetched_at == fetched


@pytest.mark.asyncio
async def test_fresh_cloud_cache_is_reused_without_fetching():
    storage = Storage(":memory:")
    storage.save_vertical_cloud_forecast(
        CACHE_LOC, "icon_d2", _cloud_forecast(10)
    )
    client = _CloudClient(cover=90)
    service = ForecastService(client=client, storage=storage)

    result = await service.fetch_vertical_cloud_model_forecast(
        CACHE_LOC, CACHE_MODEL
    )

    assert client.calls == 0
    assert _cover(result) == 10


@pytest.mark.asyncio
async def test_stale_cloud_cache_is_refetched_and_replaced():
    storage = Storage(":memory:")
    stale_time = datetime.now(timezone.utc) - CLOUD_CACHE_MAX_AGE - timedelta(
        minutes=1
    )
    storage.save_vertical_cloud_forecast(
        CACHE_LOC, "icon_d2", _cloud_forecast(10), fetched_at=stale_time
    )
    client = _CloudClient(cover=90)
    service = ForecastService(client=client, storage=storage)

    result = await service.fetch_vertical_cloud_model_forecast(
        CACHE_LOC, CACHE_MODEL
    )

    assert client.calls == 1
    assert _cover(result) == 90
    forecast, fetched_at = storage.read_vertical_cloud_forecast_entry(
        CACHE_LOC, "icon_d2"
    )
    assert _cover(forecast) == 90
    assert fetched_at > stale_time

    # The refreshed entry is fresh again, so a second view does not refetch.
    await service.fetch_vertical_cloud_model_forecast(CACHE_LOC, CACHE_MODEL)
    assert client.calls == 1


@pytest.mark.asyncio
async def test_cloud_cache_respects_custom_max_age():
    storage = Storage(":memory:")
    storage.save_vertical_cloud_forecast(
        CACHE_LOC,
        "icon_d2",
        _cloud_forecast(10),
        fetched_at=datetime.now(timezone.utc) - timedelta(minutes=10),
    )
    client = _CloudClient(cover=90)
    service = ForecastService(
        client=client,
        storage=storage,
        cloud_cache_max_age=timedelta(minutes=5),
    )

    result = await service.fetch_vertical_cloud_model_forecast(
        CACHE_LOC, CACHE_MODEL
    )

    assert client.calls == 1
    assert _cover(result) == 90


@pytest.mark.asyncio
async def test_cloud_cache_with_future_fetch_time_is_refetched():
    storage = Storage(":memory:")
    storage.save_vertical_cloud_forecast(
        CACHE_LOC,
        "icon_d2",
        _cloud_forecast(10),
        fetched_at=datetime.now(timezone.utc) + timedelta(hours=2),
    )
    client = _CloudClient(cover=90)
    service = ForecastService(client=client, storage=storage)

    result = await service.fetch_vertical_cloud_model_forecast(
        CACHE_LOC, CACHE_MODEL
    )

    assert client.calls == 1
    assert _cover(result) == 90


@pytest.mark.asyncio
async def test_stale_cloud_cache_is_kept_when_refetch_fails():
    storage = Storage(":memory:")
    storage.save_vertical_cloud_forecast(
        CACHE_LOC,
        "icon_d2",
        _cloud_forecast(10),
        fetched_at=datetime.now(timezone.utc) - timedelta(hours=5),
    )
    client = _CloudClient(fail=True)
    service = ForecastService(client=client, storage=storage)

    result = await service.fetch_vertical_cloud_model_forecast(
        CACHE_LOC, CACHE_MODEL
    )

    assert client.calls == 1
    assert _cover(result) == 10


@pytest.mark.asyncio
async def test_failed_fetch_without_cache_returns_empty_forecast():
    storage = Storage(":memory:")
    client = _CloudClient(fail=True)
    service = ForecastService(client=client, storage=storage)

    result = await service.fetch_vertical_cloud_model_forecast(
        CACHE_LOC, CACHE_MODEL
    )

    assert client.calls == 1
    assert result.profiles == []
    assert storage.read_vertical_cloud_forecast(CACHE_LOC, "icon_d2") is None


@pytest.mark.asyncio
async def test_legacy_database_rows_are_migrated_and_refetched(tmp_path):
    db_path = tmp_path / "legacy.db"
    loc_key = f"{CACHE_LOC.latitude:.4f}_{CACHE_LOC.longitude:.4f}"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE vertical_cloud_forecasts (
                id TEXT PRIMARY KEY,
                location_key TEXT NOT NULL,
                model_id TEXT NOT NULL,
                data TEXT NOT NULL
            )
            """
        )
        conn.execute(
            "INSERT INTO vertical_cloud_forecasts VALUES (?, ?, ?, ?)",
            (
                f"{loc_key}_icon_d2",
                loc_key,
                "icon_d2",
                _cloud_forecast(10).model_dump_json(),
            ),
        )
    conn.close()

    storage = Storage(str(db_path))
    forecast, fetched_at = storage.read_vertical_cloud_forecast_entry(
        CACHE_LOC, "icon_d2"
    )
    assert _cover(forecast) == 10
    assert fetched_at is None

    client = _CloudClient(cover=90)
    service = ForecastService(client=client, storage=storage)
    result = await service.fetch_vertical_cloud_model_forecast(
        CACHE_LOC, CACHE_MODEL
    )

    assert client.calls == 1
    assert _cover(result) == 90
    _, fetched_at = storage.read_vertical_cloud_forecast_entry(
        CACHE_LOC, "icon_d2"
    )
    assert fetched_at is not None
