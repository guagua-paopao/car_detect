#pragma once

#include <memory>
#include <optional>
#include <string>

#include "business/camera_task_repository.h"
#include "business/vehicle_cascade_runtime.h"

namespace yolo11_server {

struct VehicleEventPublishContext {
    std::string task_id;
    std::string camera_profile;
    std::string callback_profile;
    std::string detector_artifact;
    std::string attribute_artifact;
    std::string labels_version;
    std::string config_version;
    std::string snapshot_relative_path;
    std::string evidence_frame_id;
    float crop_quality = 0.0f;
    std::string finalized_reason = "track_exit";
    long long published_at_ms = 0;
};

class VehicleEventPublisher final {
public:
    explicit VehicleEventPublisher(
        std::shared_ptr<CameraTaskRepository> repository);

    bool publish(
        const VehicleTrackSnapshot& track,
        const std::optional<VehicleTrackAttributeSnapshot>& attributes,
        const VehicleEventPublishContext& context,
        VehicleTrackResultRecord& persisted,
        bool& duplicate,
        std::string& error);

    bool persistObservation(
        const VehicleAttributeResult& result,
        long long observed_at_ms,
        int retention_days,
        bool& duplicate,
        std::string& error);

    static std::string eventId(const VehicleTrackKey& key);

private:
    std::shared_ptr<CameraTaskRepository> repository_;
};

}  // namespace yolo11_server
