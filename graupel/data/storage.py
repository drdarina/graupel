import sqlite3
import json
from datetime import datetime, timezone
from typing import List, Optional, Tuple
from .models import MeteogramConfig, Location, DetailedCloudForecast


class Storage:
    def __init__(self, db_path: str = ":memory:"):
        self.db_path = db_path
        self._shared_memory_conn = None
        if "?mode=memory" in self.db_path or self.db_path == ":memory:":
            # Keep a persistent connection for in-memory databases
            # so the database isn't destroyed when the connection closes.
            self._shared_memory_conn = self._get_connection()
        self._init_db()

    def _get_connection(self):
        if self._shared_memory_conn is not None:
            return self._shared_memory_conn
        if "mode=memory" in self.db_path or "cache=shared" in self.db_path:
            return sqlite3.connect(self.db_path, uri=True)
        return sqlite3.connect(self.db_path)

    from contextlib import contextmanager

    @contextmanager
    def _db_connection(self):
        conn = self._get_connection()
        try:
            yield conn
        finally:
            if conn != self._shared_memory_conn:
                conn.close()

    def close(self):
        if self._shared_memory_conn is not None:
            self._shared_memory_conn.close()
            self._shared_memory_conn = None

    def _init_db(self):
        with self._db_connection() as conn:
            with conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS meteogram_configs (
                        id TEXT PRIMARY KEY,
                        data TEXT NOT NULL,
                        position INTEGER DEFAULT 0
                    )
                    """)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS vertical_cloud_forecasts (
                        id TEXT PRIMARY KEY,
                        location_key TEXT NOT NULL,
                        model_id TEXT NOT NULL,
                        data TEXT NOT NULL,
                        fetched_at TEXT
                    )
                    """)
                # Older databases predate fetched_at. Their rows keep NULL and
                # are treated as stale, so they are refetched on next use.
                cloud_columns = [
                    row[1]
                    for row in conn.execute(
                        "PRAGMA table_info(vertical_cloud_forecasts)"
                    ).fetchall()
                ]
                if "fetched_at" not in cloud_columns:
                    conn.execute(
                        "ALTER TABLE vertical_cloud_forecasts "
                        "ADD COLUMN fetched_at TEXT"
                    )
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS location_history (
                        id TEXT PRIMARY KEY,
                        name TEXT NOT NULL,
                        latitude REAL NOT NULL,
                        longitude REAL NOT NULL,
                        elevation REAL,
                        country TEXT,
                        admin1 TEXT,
                        timezone TEXT,
                        selection_count INTEGER DEFAULT 1,
                        last_selected_at TEXT NOT NULL
                    )
                    """)
                cursor = conn.execute("PRAGMA table_info(meteogram_configs)")
                columns = [row[1] for row in cursor.fetchall()]
                if "position" not in columns:
                    conn.execute(
                        "ALTER TABLE meteogram_configs ADD COLUMN position INTEGER DEFAULT 0"
                    )
                    conn.execute(
                        "UPDATE meteogram_configs SET position = (SELECT COUNT(*) FROM meteogram_configs c2 WHERE c2.rowid <= meteogram_configs.rowid) - 1"
                    )

    def create(
        self, config: MeteogramConfig, position: Optional[int] = None
    ) -> MeteogramConfig:
        if config.id is None:
            import uuid

            config.id = str(uuid.uuid4())

        with self._db_connection() as conn:
            with conn:
                if position is None:
                    cursor = conn.execute(
                        "SELECT COALESCE(MAX(position), -1) + 1 FROM meteogram_configs"
                    )
                    pos = cursor.fetchone()[0]
                else:
                    pos = position
                    conn.execute(
                        "UPDATE meteogram_configs SET position = position + 1 WHERE position >= ?",
                        (pos,),
                    )
                config.position = pos
                conn.execute(
                    "INSERT INTO meteogram_configs (id, data, position) VALUES (?, ?, ?)",
                    (config.id, config.model_dump_json(), pos),
                )
        return config

    def read(self, config_id: str) -> Optional[MeteogramConfig]:
        with self._db_connection() as conn:
            cursor = conn.execute(
                "SELECT data, position FROM meteogram_configs WHERE id = ?",
                (config_id,),
            )
            row = cursor.fetchone()
            if row:
                cfg = MeteogramConfig.model_validate_json(row[0])
                cfg.position = row[1]
                return cfg
            return None

    def update(self, config: MeteogramConfig) -> MeteogramConfig:
        if config.id is None:
            raise ValueError("Config ID cannot be None for update")
        with self._db_connection() as conn:
            with conn:
                if config.position is not None:
                    cursor = conn.execute(
                        "UPDATE meteogram_configs SET data = ?, position = ? WHERE id = ?",
                        (config.model_dump_json(), config.position, config.id),
                    )
                else:
                    cursor = conn.execute(
                        "UPDATE meteogram_configs SET data = ? WHERE id = ?",
                        (config.model_dump_json(), config.id),
                    )
                if cursor.rowcount == 0:
                    raise KeyError(f"Config with id {config.id} not found")
        return config

    def delete(self, config_id: str) -> None:
        with self._db_connection() as conn:
            with conn:
                cursor = conn.execute(
                    "SELECT position FROM meteogram_configs WHERE id = ?",
                    (config_id,),
                )
                row = cursor.fetchone()
                if not row:
                    raise KeyError(f"Config with id {config_id} not found")
                deleted_pos = row[0]
                conn.execute(
                    "DELETE FROM meteogram_configs WHERE id = ?", (config_id,)
                )
                if deleted_pos is not None:
                    conn.execute(
                        "UPDATE meteogram_configs SET position = position - 1 WHERE position > ?",
                        (deleted_pos,),
                    )

    def reorder(self, ordered_ids: List[str]) -> None:
        with self._db_connection() as conn:
            with conn:
                for idx, cfg_id in enumerate(ordered_ids):
                    cursor = conn.execute(
                        "SELECT data FROM meteogram_configs WHERE id = ?",
                        (cfg_id,),
                    )
                    row = cursor.fetchone()
                    if row:
                        try:
                            cfg = MeteogramConfig.model_validate_json(row[0])
                            cfg.position = idx
                            conn.execute(
                                "UPDATE meteogram_configs SET position = ?, data = ? WHERE id = ?",
                                (idx, cfg.model_dump_json(), cfg_id),
                            )
                        except Exception:
                            conn.execute(
                                "UPDATE meteogram_configs SET position = ? WHERE id = ?",
                                (idx, cfg_id),
                            )

    def list_all(self) -> List[MeteogramConfig]:
        with self._db_connection() as conn:
            cursor = conn.execute(
                "SELECT data, position FROM meteogram_configs ORDER BY position ASC, rowid ASC"
            )
            configs = []
            for row in cursor.fetchall():
                cfg = MeteogramConfig.model_validate_json(row[0])
                cfg.position = row[1]
                configs.append(cfg)
            return configs

    def save_vertical_cloud_forecast(
        self,
        location: Location,
        model_id: str,
        detailed_forecast: DetailedCloudForecast,
        fetched_at: Optional[datetime] = None,
    ) -> None:
        loc_key = f"{location.latitude:.4f}_{location.longitude:.4f}"
        rec_id = f"{loc_key}_{model_id.lower()}"
        if fetched_at is None:
            fetched_at = datetime.now(timezone.utc)
        with self._db_connection() as conn:
            with conn:
                conn.execute(
                    """
                    INSERT INTO vertical_cloud_forecasts
                        (id, location_key, model_id, data, fetched_at)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        data = excluded.data,
                        fetched_at = excluded.fetched_at
                    """,
                    (
                        rec_id,
                        loc_key,
                        model_id.lower(),
                        detailed_forecast.model_dump_json(),
                        fetched_at.astimezone(timezone.utc).isoformat(),
                    ),
                )

    def read_vertical_cloud_forecast_entry(
        self, location: Location, model_id: str
    ) -> Optional[Tuple[DetailedCloudForecast, Optional[datetime]]]:
        """Return the cached forecast and when it was fetched (UTC).

        The fetch time is None for rows written before it was recorded.
        """
        loc_key = f"{location.latitude:.4f}_{location.longitude:.4f}"
        rec_id = f"{loc_key}_{model_id.lower()}"
        with self._db_connection() as conn:
            cursor = conn.execute(
                "SELECT data, fetched_at FROM vertical_cloud_forecasts "
                "WHERE id = ?",
                (rec_id,),
            )
            row = cursor.fetchone()
            if not row:
                return None
            forecast = DetailedCloudForecast.model_validate_json(row[0])
            fetched_at = None
            if row[1]:
                try:
                    fetched_at = datetime.fromisoformat(row[1])
                except ValueError:
                    fetched_at = None
                if fetched_at is not None and fetched_at.tzinfo is None:
                    fetched_at = fetched_at.replace(tzinfo=timezone.utc)
            return forecast, fetched_at

    def read_vertical_cloud_forecast(
        self, location: Location, model_id: str
    ) -> Optional[DetailedCloudForecast]:
        entry = self.read_vertical_cloud_forecast_entry(location, model_id)
        return entry[0] if entry else None

    def record_location_selection(self, location: Location) -> None:
        import datetime
        loc_id = f"{location.latitude:.4f}_{location.longitude:.4f}"
        loc_id = f"{round(location.latitude, 4):.4f}_{round(location.longitude, 4):.4f}"
        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        with self._db_connection() as conn:
            with conn:
                conn.execute(
                    """
                    INSERT INTO location_history (
                        id, name, latitude, longitude, elevation, country, admin1, timezone, selection_count, last_selected_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        selection_count = selection_count + 1,
                        last_selected_at = excluded.last_selected_at,
                        name = excluded.name,
                        elevation = COALESCE(excluded.elevation, location_history.elevation),
                        country = COALESCE(excluded.country, location_history.country),
                        admin1 = COALESCE(excluded.admin1, location_history.admin1),
                        timezone = COALESCE(excluded.timezone, location_history.timezone)
                    """,
                    (
                        loc_id,
                        location.name,
                        round(location.latitude, 4),
                        round(location.longitude, 4),
                        location.elevation,
                        location.country,
                        location.admin1,
                        location.timezone,
                        now_iso,
                    ),
                )

    def get_most_used_locations(self, limit: int = 10) -> List[Location]:
        with self._db_connection() as conn:
            cursor = conn.execute(
                """
                SELECT name, latitude, longitude, elevation, country, admin1, timezone
                FROM location_history
                ORDER BY selection_count DESC, last_selected_at DESC
                LIMIT ?
                """,
                (limit,),
            )
            rows = cursor.fetchall()
            return [
                Location(
                    name=row[0],
                    latitude=row[1],
                    longitude=row[2],
                    elevation=row[3],
                    country=row[4],
                    admin1=row[5],
                    timezone=row[6],
                )
                for row in rows
            ]

