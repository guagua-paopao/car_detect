BEGIN;

CREATE TABLE IF NOT EXISTS camera_schema_version (
  version INTEGER PRIMARY KEY,
  applied_at_ms BIGINT NOT NULL
);
CREATE TABLE IF NOT EXISTS camera_tasks (
  task_id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  camera_profile TEXT NOT NULL,
  enabled SMALLINT NOT NULL CHECK (enabled IN (0,1)),
  frame_interval_ms INTEGER NOT NULL,
  output_mode TEXT NOT NULL CHECK (output_mode IN ('latest','archive','both')),
  jpeg_quality INTEGER NOT NULL,
  max_width INTEGER NOT NULL,
  max_height INTEGER NOT NULL,
  retention_days INTEGER NOT NULL,
  max_saved_frames INTEGER NOT NULL,
  version INTEGER NOT NULL,
  created_at_ms BIGINT NOT NULL,
  updated_at_ms BIGINT NOT NULL,
  deleted_at_ms BIGINT
);
CREATE INDEX IF NOT EXISTS idx_camera_tasks_updated ON camera_tasks(deleted_at_ms,updated_at_ms DESC);
CREATE TABLE IF NOT EXISTS camera_task_runs (
  run_id TEXT PRIMARY KEY,
  task_id TEXT NOT NULL REFERENCES camera_tasks(task_id),
  definition_version INTEGER NOT NULL,
  definition_json TEXT NOT NULL,
  status TEXT NOT NULL,
  camera_profile TEXT NOT NULL,
  hub_instance_id TEXT,
  create_time_ms BIGINT NOT NULL,
  start_time_ms BIGINT,
  stop_time_ms BIGINT,
  last_update_ms BIGINT NOT NULL,
  worker_consumer TEXT,
  capture_backend TEXT,
  capture_fps DOUBLE PRECISION NOT NULL DEFAULT 0,
  save_fps DOUBLE PRECISION NOT NULL DEFAULT 0,
  consumed_frames BIGINT NOT NULL DEFAULT 0,
  saved_frames BIGINT NOT NULL DEFAULT 0,
  skipped_frames BIGINT NOT NULL DEFAULT 0,
  dropped_frames BIGINT NOT NULL DEFAULT 0,
  last_source_sequence BIGINT NOT NULL DEFAULT 0,
  last_frame_time_ms BIGINT,
  width INTEGER NOT NULL DEFAULT 0,
  height INTEGER NOT NULL DEFAULT 0,
  stop_reason TEXT,
  error_code TEXT,
  error_message TEXT
);
CREATE UNIQUE INDEX IF NOT EXISTS uq_camera_task_active_run ON camera_task_runs(task_id)
  WHERE status IN ('queued','starting','running','reconnecting');
CREATE INDEX IF NOT EXISTS idx_camera_runs_task_time ON camera_task_runs(task_id,create_time_ms DESC);
CREATE TABLE IF NOT EXISTS camera_frames (
  frame_id TEXT PRIMARY KEY,
  task_id TEXT NOT NULL REFERENCES camera_tasks(task_id),
  run_id TEXT NOT NULL REFERENCES camera_task_runs(run_id),
  source_sequence BIGINT NOT NULL,
  capture_time_ms BIGINT NOT NULL,
  save_time_ms BIGINT NOT NULL,
  relative_path TEXT NOT NULL UNIQUE,
  width INTEGER NOT NULL,
  height INTEGER NOT NULL,
  size_bytes BIGINT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_camera_frames_task_time ON camera_frames(task_id,capture_time_ms DESC);
CREATE INDEX IF NOT EXISTS idx_camera_frames_run_sequence ON camera_frames(run_id,source_sequence);

INSERT INTO camera_schema_version(version,applied_at_ms)
VALUES(1,(EXTRACT(EPOCH FROM clock_timestamp())*1000)::BIGINT)
ON CONFLICT(version) DO NOTHING;
COMMIT;
