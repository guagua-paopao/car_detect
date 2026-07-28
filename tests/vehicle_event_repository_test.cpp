#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "business/camera_task_repository.h"
#include "business/postgres_client.h"
#include "business/vehicle_event_publisher.h"
#include "postgres_test_guard.h"

namespace {

using namespace yolo11_server;
using json = nlohmann::json;

void require(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

VehicleTrackResultRecord vehicleEvent(
    const std::string& event_id,
    long long track_id,
    long long stamp) {
    VehicleTrackResultRecord result;
    result.event_id = event_id;
    result.task_id = "vehicle_camera";
    result.camera_id = "vehicle_camera";
    result.run_id = "vehicle_run";
    result.track_id = track_id;
    result.first_seen_at_ms = stamp;
    result.last_seen_at_ms = stamp + 100;
    result.occurred_at_ms = stamp + 100;
    result.vehicle_class = "car";
    result.vehicle_class_confidence = 0.96;
    result.body_type = "suv";
    result.body_type_confidence = 0.91;
    result.body_type_stable = true;
    result.body_type_samples_used = 4;
    result.color = "white";
    result.color_confidence = 0.88;
    result.color_stable = true;
    result.color_samples_used = 4;
    result.detector_artifact = "vehicle-det-v1";
    result.attribute_artifact = "vehicle-attr-v1";
    result.labels_version = "vehicle-labels-v1";
    result.config_version = "vehicle-analytics-m4";
    result.snapshot_relative_path =
        "vehicle_camera/vehicle_run/ve_vehicle_42.jpg";
    result.evidence_frame_id = "vf_vehicle_42";
    result.crop_quality = 0.93;
    result.finalized_reason = "stable";
    result.created_at_ms = stamp + 120;
    return result;
}

}  // namespace

int main() {
    if (const int guard = requireDisposablePostgresTestDatabase()) {
        return guard;
    }

    CameraTasksSection storage;
    storage.postgres_dsn_env = "YOLO11_TEST_POSTGRES_DSN";
    PostgresConnection database;
    std::string error;
    require(
        database.openFromEnvironment(storage.postgres_dsn_env, error),
        "test PostgreSQL connection must open: " + error);
    require(
        database.exec(
            "DROP TABLE IF EXISTS callback_outbox,vehicle_attribute_observations,"
            "vehicle_track_results,security_alert_events,vision_events,"
            "camera_idempotency_keys,camera_frames,camera_run_analysis_results,"
            "camera_task_runs,camera_tasks,camera_schema_version CASCADE;",
            error),
        "vehicle schema reset must succeed: " + error);

    auto repository = std::make_shared<CameraTaskRepository>(storage);
    require(repository->initialize(error), "repository must initialize: " + error);

    constexpr long long stamp = 1785200000000LL;
    CameraTaskDefinition task;
    task.task_id = "vehicle_camera";
    task.name = "Vehicle event repository test";
    task.camera_profile = "gate_01";
    task.enabled = true;
    task.desired_state = "running";
    task.analysis_enabled = true;
    task.target_infer_fps = 8.0;
    task.algorithm_profile = "vehicle_default";
    task.algorithms = {"vehicle_detection", "vehicle_attribute"};
    task.callback_profile = "backend_primary";
    task.created_at_ms = stamp;
    task.updated_at_ms = stamp;
    std::string code;
    require(
        repository->createTask(task, code, error),
        "vehicle camera must persist: " + error);

    CameraTaskRunRecord run;
    run.run_id = "vehicle_run";
    run.task_id = task.task_id;
    run.definition_version = 1;
    run.definition_json = "{}";
    run.status = "queued";
    run.camera_profile = task.camera_profile;
    run.create_time_ms = stamp;
    run.last_update_ms = stamp;
    require(
        repository->createRun(run, code, error),
        "vehicle run must persist: " + error);

    VehicleTrackSnapshot track;
    track.key = {"vehicle_camera", "vehicle_run", 7, 42};
    track.state = VehicleTrackState::AttributeStable;
    track.vehicle_class = "car";
    track.detection_confidence = 0.96f;
    track.first_seen_ms = stamp + 1000;
    track.last_seen_ms = stamp + 1100;
    VehicleTrackAttributeSnapshot attributes;
    attributes.key = track.key;
    attributes.artifact_id = "vehicle-attr-v1";
    attributes.labels_version = "vehicle-labels-v1";
    attributes.body_type = {"suv", 0.91f, true, 4};
    attributes.color = {"white", 0.88f, true, 4};
    attributes.observation_count = 4;
    VehicleEventPublishContext publish_context;
    publish_context.task_id = "vehicle_camera";
    publish_context.camera_profile = task.camera_profile;
    publish_context.callback_profile = task.callback_profile;
    publish_context.detector_artifact = "vehicle-det-v1";
    publish_context.attribute_artifact = "vehicle-attr-v1";
    publish_context.labels_version = "vehicle-labels-v1";
    publish_context.config_version = "vehicle-analytics-m4";
    publish_context.snapshot_relative_path =
        "vehicle_camera/vehicle_run/ve_vehicle_42.jpg";
    publish_context.evidence_frame_id = "vf_vehicle_42";
    publish_context.crop_quality = 0.93f;
    publish_context.finalized_reason = "stable";
    publish_context.published_at_ms = stamp + 1120;
    VehicleEventPublisher publisher(repository);
    VehicleTrackResultRecord event;
    bool duplicate = false;
    require(
        publisher.publish(
            track,
            attributes,
            publish_context,
            event,
            duplicate,
            error) &&
            !duplicate &&
            event.event_id == VehicleEventPublisher::eventId(track.key),
        "M3 track must map to a deterministic persisted vehicle event: " +
            error);

    VehicleTrackResultRecord loaded;
    bool found = false;
    require(
        repository->getVehicleEvent(event.event_id, loaded, found, error) &&
            found &&
            loaded.track_id == 42 &&
            loaded.body_type == "suv" &&
            loaded.color == "white" &&
            loaded.delivery_status == "pending",
        "vehicle event query must return typed projection and delivery state");

    VisionEventRecord immutable;
    require(
        repository->getVisionEvent(event.event_id, immutable, found, error) &&
            found &&
            immutable.event_kind == "vehicle_passage",
        "vehicle event must create an immutable vision event");
    const auto payload = json::parse(immutable.payload_json);
    require(
        payload["event_id"] == event.event_id &&
            payload["event_kind"] == "vehicle_passage" &&
            payload["attributes"]["body_type"]["label"] == "suv" &&
            !payload.contains("delivery"),
        "immutable callback payload must match vehicle event schema");

    std::vector<VehicleTrackResultRecord> listed;
    require(
        repository->listVehicleEvents(
            task.task_id,
            event.occurred_at_ms,
            event.occurred_at_ms,
            20,
            0,
            listed,
            error) &&
            listed.size() == 1 &&
            listed.front().event_id == event.event_id,
        "camera/time query must return the vehicle event");

    require(
        !repository->publishVehicleEvent(
            event, task.camera_profile, task.callback_profile, code, error) &&
            code == "VEHICLE_EVENT_ALREADY_EXISTS",
        "same event must be idempotently rejected");
    auto duplicate_track =
        vehicleEvent("ve_vehicle_duplicate_track", 42, stamp + 2000);
    require(
        !repository->publishVehicleEvent(
            duplicate_track,
            task.camera_profile,
            task.callback_profile,
            code,
            error) &&
            code == "VEHICLE_EVENT_ALREADY_EXISTS",
        "one run/track must publish at most one event");

    VehicleAttributeObservationRecord observation;
    observation.observation_id = "vo_vehicle_42_1";
    observation.task_id = task.task_id;
    observation.run_id = run.run_id;
    observation.track_id = 42;
    observation.crop_sequence = 1;
    observation.observed_at_ms = stamp + 1050;
    observation.quality_score = 0.93;
    observation.body_type = "suv";
    observation.body_type_confidence = 0.91;
    observation.color = "white";
    observation.color_confidence = 0.88;
    observation.attribute_artifact = "vehicle-attr-v1";
    observation.labels_version = "vehicle-labels-v1";
    observation.expires_at_ms = stamp + 2000;
    require(
        repository->insertVehicleAttributeObservation(
            observation, code, error),
        "debug observation must persist: " + error);
    require(
        !repository->insertVehicleAttributeObservation(
            observation, code, error) &&
            code == "VEHICLE_OBSERVATION_ALREADY_EXISTS",
        "duplicate crop observation must be rejected");
    int deleted = 0;
    require(
        repository->deleteExpiredVehicleAttributeObservations(
            stamp + 2000, 100, deleted, error) &&
            deleted == 1,
        "expired debug observation must be removed");

    CameraTaskRepositoryStats stats;
    require(
        repository->stats(stats, error) &&
            stats.vision_events_total == 1 &&
            stats.vehicle_events_total == 1 &&
            stats.vehicle_observations_total == 0 &&
            stats.callbacks_pending == 1,
        "repository metrics must expose vehicle storage and outbox state");

    std::cout
        << "PASS: vehicle event storage, idempotency, observations, and outbox\n";
    return 0;
}
