#pragma once

#include <atomic>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "business/camera_task_repository.h"
#include "business/camera_task_runtime_control.h"
#include "server/app_config.h"
#include "server/analysis_snapshot_writer.h"
#include "server/camera_inference_pool.h"
#include "server/vehicle_attribute_batch_scheduler.h"

namespace yolo11_server {

class TensorRtVehicleAttributeRunner;
class VehicleEventPublisher;
struct VehicleRuntimeAssets;

struct CameraAlgorithmProcessorSnapshot {
    std::size_t active_sessions = 0;
    long long processed_frames = 0;
    long long persisted_alerts = 0;
    long long duplicate_alerts = 0;
    long long failed_frames = 0;
    VehicleAttributeBatchSchedulerSnapshot attribute_scheduler;
    AnalysisSnapshotWriterMetrics snapshot_writer;
};

// Stateful per-camera analytics behind the fixed inference pool. Sessions are
// keyed by stable camera id and replaced only when run_id changes.
class CameraAlgorithmProcessor final : public ICameraInferenceResultHandler {
public:
    CameraAlgorithmProcessor(
        AppConfig config,
        std::shared_ptr<CameraTaskRepository> repository,
        std::shared_ptr<ICameraAnalysisStatusSink> status_sink = {}
    );
    ~CameraAlgorithmProcessor() noexcept override;

    bool start(std::string& error);
    void stop() noexcept;

    bool handle(const CameraInferenceResult& result, std::string& error) override;
    void detachCamera(
        const std::string& task_id,
        const std::string& run_id) noexcept override;

    CameraAlgorithmProcessorSnapshot snapshot() const;

private:
    struct VehicleSession;
    struct VehicleAttributeRunnerSlot;
    bool handleVehicle(
        const CameraInferenceResult& result,
        std::string& error);

    AppConfig config_;
    std::shared_ptr<CameraTaskRepository> repository_;
    std::shared_ptr<ICameraAnalysisStatusSink> status_sink_;
    mutable std::mutex vehicle_sessions_mutex_;
    std::map<std::string, std::shared_ptr<VehicleSession>> vehicle_sessions_;
    std::unique_ptr<VehicleRuntimeAssets> vehicle_assets_;
    std::unique_ptr<VehicleAttributeBatchScheduler>
        vehicle_attribute_scheduler_;
    std::vector<std::unique_ptr<VehicleAttributeRunnerSlot>>
        vehicle_attribute_runners_;
    std::unique_ptr<VehicleEventPublisher> vehicle_event_publisher_;
    std::unique_ptr<AnalysisSnapshotWriter> snapshot_writer_;
    bool vehicle_enabled_ = false;
    std::atomic<bool> running_{ false };
    std::atomic<long long> processed_frames_{ 0 };
    std::atomic<long long> persisted_alerts_{ 0 };
    std::atomic<long long> duplicate_alerts_{ 0 };
    std::atomic<long long> failed_frames_{ 0 };
};

}  // namespace yolo11_server
