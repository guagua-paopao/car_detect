BEGIN;

CREATE TABLE IF NOT EXISTS vision_events (
  event_id TEXT PRIMARY KEY,
  event_kind TEXT NOT NULL,
  task_id TEXT NOT NULL REFERENCES camera_tasks(task_id),
  run_id TEXT NOT NULL REFERENCES camera_task_runs(run_id),
  camera_profile TEXT NOT NULL,
  occurred_at_ms BIGINT NOT NULL,
  payload_json JSONB NOT NULL,
  evidence_frame_id TEXT,
  created_at_ms BIGINT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_vision_events_task_kind_time
  ON vision_events(task_id,event_kind,occurred_at_ms DESC,event_id DESC);

INSERT INTO vision_events(
  event_id,event_kind,task_id,run_id,camera_profile,occurred_at_ms,
  payload_json,evidence_frame_id,created_at_ms)
SELECT
  e.event_id,'algorithm_alert',e.task_id,e.run_id,e.camera_profile,
  e.occurred_at_ms,
  jsonb_build_object(
    'schema_version','1.0',
    'event_kind','algorithm_alert',
    'event_id',e.event_id,
    'task_id',e.task_id,
    'camera_id',e.task_id,
    'run_id',e.run_id,
    'camera_profile',e.camera_profile,
    'event_type',e.event_type,
    'category',e.category,
    'severity',e.severity,
    'confidence',e.confidence,
    'track_id',e.track_id,
    'occurred_at_ms',e.occurred_at_ms,
    'algorithm',jsonb_build_object(
      'profile',e.algorithm_profile,
      'model',e.model_name,
      'config_version',e.config_version,
      'demo_classifier',e.demo_classifier <> 0),
    'payload',e.payload_json,
    'created_at_ms',e.created_at_ms
  ) || CASE
    WHEN e.evidence_frame_id IS NULL THEN '{}'::jsonb
    ELSE jsonb_build_object(
      'evidence',jsonb_build_object('frame_id',e.evidence_frame_id))
  END,
  e.evidence_frame_id,e.created_at_ms
FROM security_alert_events e
ON CONFLICT(event_id) DO NOTHING;

DO $migration$
BEGIN
  IF NOT EXISTS (
    SELECT 1
    FROM pg_constraint
    WHERE conname='security_alert_events_vision_event_fkey'
  ) THEN
    ALTER TABLE security_alert_events
      ADD CONSTRAINT security_alert_events_vision_event_fkey
      FOREIGN KEY(event_id) REFERENCES vision_events(event_id) NOT VALID;
    ALTER TABLE security_alert_events
      VALIDATE CONSTRAINT security_alert_events_vision_event_fkey;
  END IF;

  IF EXISTS (
    SELECT 1
    FROM pg_constraint
    WHERE conname='callback_outbox_event_id_fkey'
      AND confrelid='security_alert_events'::regclass
  ) THEN
    ALTER TABLE callback_outbox
      DROP CONSTRAINT callback_outbox_event_id_fkey;
  END IF;
  IF NOT EXISTS (
    SELECT 1
    FROM pg_constraint
    WHERE conname='callback_outbox_event_id_fkey'
  ) THEN
    ALTER TABLE callback_outbox
      ADD CONSTRAINT callback_outbox_event_id_fkey
      FOREIGN KEY(event_id) REFERENCES vision_events(event_id);
  END IF;
END
$migration$;

CREATE TABLE IF NOT EXISTS vehicle_track_results (
  event_id TEXT PRIMARY KEY REFERENCES vision_events(event_id),
  task_id TEXT NOT NULL REFERENCES camera_tasks(task_id),
  run_id TEXT NOT NULL REFERENCES camera_task_runs(run_id),
  camera_id TEXT NOT NULL,
  track_id BIGINT NOT NULL,
  first_seen_at_ms BIGINT NOT NULL,
  last_seen_at_ms BIGINT NOT NULL,
  occurred_at_ms BIGINT NOT NULL,
  vehicle_class TEXT NOT NULL,
  vehicle_class_confidence DOUBLE PRECISION NOT NULL,
  body_type TEXT NOT NULL,
  body_type_confidence DOUBLE PRECISION NOT NULL,
  body_type_stable SMALLINT NOT NULL CHECK(body_type_stable IN (0,1)),
  body_type_samples_used INTEGER NOT NULL,
  color TEXT NOT NULL,
  color_confidence DOUBLE PRECISION NOT NULL,
  color_stable SMALLINT NOT NULL CHECK(color_stable IN (0,1)),
  color_samples_used INTEGER NOT NULL,
  detector_artifact TEXT NOT NULL,
  attribute_artifact TEXT NOT NULL,
  labels_version TEXT NOT NULL,
  config_version TEXT NOT NULL,
  snapshot_relative_path TEXT,
  evidence_frame_id TEXT,
  crop_quality DOUBLE PRECISION NOT NULL,
  finalized_reason TEXT NOT NULL
    CHECK(finalized_reason IN ('stable','track_exit','run_stop')),
  created_at_ms BIGINT NOT NULL,
  UNIQUE(run_id,track_id)
);
CREATE INDEX IF NOT EXISTS idx_vehicle_results_camera_time
  ON vehicle_track_results(camera_id,occurred_at_ms DESC,event_id DESC);
CREATE INDEX IF NOT EXISTS idx_vehicle_results_run_track
  ON vehicle_track_results(run_id,track_id);

CREATE TABLE IF NOT EXISTS vehicle_attribute_observations (
  observation_id TEXT PRIMARY KEY,
  task_id TEXT NOT NULL REFERENCES camera_tasks(task_id),
  run_id TEXT NOT NULL REFERENCES camera_task_runs(run_id),
  track_id BIGINT NOT NULL,
  crop_sequence BIGINT NOT NULL,
  observed_at_ms BIGINT NOT NULL,
  quality_score DOUBLE PRECISION NOT NULL,
  body_type TEXT NOT NULL,
  body_type_confidence DOUBLE PRECISION NOT NULL,
  color TEXT NOT NULL,
  color_confidence DOUBLE PRECISION NOT NULL,
  attribute_artifact TEXT NOT NULL,
  labels_version TEXT NOT NULL,
  expires_at_ms BIGINT NOT NULL,
  UNIQUE(run_id,track_id,crop_sequence)
);
CREATE INDEX IF NOT EXISTS idx_vehicle_observations_expiry
  ON vehicle_attribute_observations(expires_at_ms,observation_id);

INSERT INTO camera_schema_version(version,applied_at_ms)
VALUES(4,(EXTRACT(EPOCH FROM clock_timestamp())*1000)::BIGINT)
ON CONFLICT(version) DO NOTHING;

COMMIT;
