#pragma once

#include <memory>
#include <string>

#include <crow.h>

#include "business/camera_task_repository.h"
#include "server/app_config.h"
#include "server/camera_profile_registry.h"
#include "server/camera_task_http_controller.h"
#include "server/camera_task_queue.h"
#include "server/redis_task_queue.h"
#include "server/unified_camera_application_service.h"

namespace yolo11_server {

// HTTP composition root for the vehicle detection service. TensorRT remains
// isolated in the worker process; this layer owns only APIs and persistence.
class VisionHttpServer final {
public:
    explicit VisionHttpServer(const AppConfig& config);
    ~VisionHttpServer() noexcept = default;

    bool initialize(std::string& error);
    void registerRoutes(crow::SimpleApp& app);

private:
    crow::response health() const;
    crow::response ready() const;
    crow::response adminAsset(
        const std::string& file_name,
        const std::string& content_type) const;
    bool readAlgorithmRuntime(
        AlgorithmRuntimeSnapshot& runtime,
        std::string& error) const;

    AppConfig config_;
    mutable RedisTaskQueue redis_;
    std::shared_ptr<CameraTaskRepository> camera_task_repository_;
    std::shared_ptr<ICameraTaskApiControl> camera_task_control_;
    std::shared_ptr<CameraProfileRegistry> camera_profile_registry_;
    std::shared_ptr<UnifiedCameraApplicationService>
        unified_camera_application_service_;
    std::unique_ptr<CameraTaskHttpController> camera_task_controller_;
};

}  // namespace yolo11_server
