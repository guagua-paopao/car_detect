#include "business/vehicle_event_publisher.h"

#include <algorithm>
#include <iomanip>
#include <sstream>
#include <utility>

#include <openssl/sha.h>

namespace yolo11_server {

namespace {

std::string sha256Prefix(const std::string& value, std::size_t bytes) {
    unsigned char digest[SHA256_DIGEST_LENGTH]{};
    SHA256(
        reinterpret_cast<const unsigned char*>(value.data()),
        value.size(),
        digest);
    std::ostringstream output;
    output << std::hex << std::setfill('0');
    for (std::size_t index = 0;
         index < std::min(bytes, sizeof(digest));
         ++index) {
        output << std::setw(2) << static_cast<int>(digest[index]);
    }
    return output.str();
}

}  // namespace

VehicleEventPublisher::VehicleEventPublisher(
    std::shared_ptr<CameraTaskRepository> repository)
    : repository_(std::move(repository)) {
}

bool VehicleEventPublisher::publish(
    const VehicleTrackSnapshot& track,
    const std::optional<VehicleTrackAttributeSnapshot>& attributes,
    const VehicleEventPublishContext& context,
    VehicleTrackResultRecord& persisted,
    bool& duplicate,
    std::string& error) {
    persisted = {};
    duplicate = false;
    error.clear();
    if (!repository_) {
        error = "vehicle event repository is unavailable";
        return false;
    }

    VehicleTrackResultRecord result;
    result.event_id = eventId(track.key);
    result.task_id = context.task_id;
    result.camera_id = track.key.camera_id;
    result.run_id = track.key.run_id;
    result.track_id = track.key.track_id;
    result.first_seen_at_ms = track.first_seen_ms;
    result.last_seen_at_ms = track.last_seen_ms;
    result.occurred_at_ms = track.last_seen_ms;
    result.vehicle_class = track.vehicle_class;
    result.vehicle_class_confidence = track.detection_confidence;
    result.detector_artifact = context.detector_artifact;
    result.attribute_artifact = context.attribute_artifact;
    result.labels_version = context.labels_version;
    if (attributes) {
        result.body_type = attributes->body_type.label;
        result.body_type_confidence = attributes->body_type.confidence;
        result.body_type_stable = attributes->body_type.stable;
        result.body_type_samples_used =
            static_cast<int>(attributes->body_type.samples_used);
        result.color = attributes->color.label;
        result.color_confidence = attributes->color.confidence;
        result.color_stable = attributes->color.stable;
        result.color_samples_used =
            static_cast<int>(attributes->color.samples_used);
        if (!attributes->artifact_id.empty()) {
            result.attribute_artifact = attributes->artifact_id;
        }
        if (!attributes->labels_version.empty()) {
            result.labels_version = attributes->labels_version;
        }
    }
    result.config_version = context.config_version;
    result.snapshot_relative_path = context.snapshot_relative_path;
    result.evidence_frame_id = context.evidence_frame_id;
    result.crop_quality = context.crop_quality;
    result.finalized_reason = context.finalized_reason;
    result.created_at_ms = context.published_at_ms;

    std::string error_code;
    if (!repository_->publishVehicleEvent(
            result,
            context.camera_profile,
            context.callback_profile,
            error_code,
            error)) {
        duplicate = error_code == "VEHICLE_EVENT_ALREADY_EXISTS";
        if (!duplicate) return false;
        bool found = false;
        if (!repository_->getVehicleEvent(
                result.event_id, persisted, found, error)) {
            return false;
        }
        if (!found) {
            error = "duplicate vehicle event projection is missing";
            return false;
        }
        return true;
    }
    persisted = std::move(result);
    return true;
}

bool VehicleEventPublisher::persistObservation(
    const VehicleAttributeResult& result,
    long long observed_at_ms,
    int retention_days,
    bool& duplicate,
    std::string& error) {
    duplicate = false;
    error.clear();
    if (!repository_) {
        error = "vehicle event repository is unavailable";
        return false;
    }
    const VehicleTrackKey key{
        result.camera_id,
        result.run_id,
        result.metadata.run_generation,
        result.track_id
    };
    VehicleAttributeObservationRecord observation;
    observation.observation_id =
        "vo_" + sha256Prefix(
            eventId(key) + ":" + std::to_string(result.crop_sequence),
            16);
    observation.task_id = result.camera_id;
    observation.run_id = result.run_id;
    observation.track_id = result.track_id;
    observation.crop_sequence = result.crop_sequence;
    observation.observed_at_ms = observed_at_ms;
    observation.quality_score = result.quality_score;
    observation.body_type = result.body_type.label;
    observation.body_type_confidence = result.body_type.confidence;
    observation.color = result.color.label;
    observation.color_confidence = result.color.confidence;
    observation.attribute_artifact = result.metadata.artifact_id;
    observation.labels_version = result.metadata.labels_version;
    const long long days = std::clamp(retention_days, 1, 365);
    observation.expires_at_ms =
        observed_at_ms + days * 24LL * 60LL * 60LL * 1000LL;
    std::string error_code;
    if (!repository_->insertVehicleAttributeObservation(
            observation, error_code, error)) {
        duplicate =
            error_code == "VEHICLE_OBSERVATION_ALREADY_EXISTS";
        return duplicate;
    }
    return true;
}

std::string VehicleEventPublisher::eventId(const VehicleTrackKey& key) {
    const std::string identity =
        key.camera_id + "\n" +
        key.run_id + "\n" +
        std::to_string(key.run_generation) + "\n" +
        std::to_string(key.track_id);
    return "ve_" + sha256Prefix(identity, 20);
}

}  // namespace yolo11_server
