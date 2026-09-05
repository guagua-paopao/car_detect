#include "server/vision_http_server.h"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "server/worker_runtime_readiness.h"

namespace yolo11_server {

namespace {
using json = nlohmann::json;

long long nowMs() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

crow::response jsonResponse(int code, const json& body) {
    crow::response response(code, body.dump());
    response.set_header("Content-Type", "application/json; charset=utf-8");
    response.set_header("Cache-Control", "no-store");
    return response;
}

bool readFile(const std::filesystem::path& path, std::string& bytes) {
    std::ifstream input(path, std::ios::binary);
    if (!input) return false;
    bytes.assign(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
    return !bytes.empty();
}

}  // namespace

VisionHttpServer::VisionHttpServer(const AppConfig& config)
    : config_(config), redis_(config.redis) {
    if (config_.camera_tasks.enabled) {
        camera_task_repository_ =
            std::make_shared<CameraTaskRepository>(config_.camera_tasks);
        camera_task_control_ = std::make_shared<CameraTaskQueue>(
            config_.redis, config_.camera_tasks, "camera_task_http");
        if (!config_.stream.camera_profiles_path.empty()) {
            camera_profile_registry_ = std::make_shared<CameraProfileRegistry>(
                config_.stream.camera_profiles_path);
        }
        unified_camera_application_service_ =
            std::make_shared<UnifiedCameraApplicationService>(
                config_,
                camera_task_repository_,
                camera_task_control_,
                camera_profile_registry_);
        camera_task_controller_ = std::make_unique<CameraTaskHttpController>(
            config_,
            camera_task_repository_,
            camera_task_control_,
            std::string{},
            camera_profile_registry_,
            [this](AlgorithmRuntimeSnapshot& runtime, std::string& error) {
                return readAlgorithmRuntime(runtime, error);
            },
            unified_camera_application_service_,
            [this](
                const std::string& camera_id,
                std::vector<VehicleRealtimeTrackRecord>& tracks,
                long long& generated_at_ms,
                std::string& error) {
                tracks.clear();
                generated_at_ms = 0;
                error.clear();

                CameraTaskDefinition task;
                bool task_found = false;
                if (!camera_task_repository_->getTask(
                        camera_id, false, task, task_found, error)) {
                    return false;
                }
                if (!task_found) {
                    error = "TASK_NOT_FOUND";
                    return false;
                }

                std::vector<CameraTaskRunRecord> runs;
                if (!camera_task_repository_->listRuns(
                        camera_id, 20, 0, runs, error)) {
                    return false;
                }
                CameraTaskRunRecord active_run;
                bool active_found = false;
                for (const auto& run : runs) {
                    if (run.status == "queued" || run.status == "starting" ||
                        run.status == "running" || run.status == "reconnecting" ||
                        run.status == "stopping") {
                        active_run = run;
                        active_found = true;
                        break;
                    }
                }
                if (!active_found) return true;

                CameraTaskRunHotStatus hot;
                if (!camera_task_control_->getRunStatus(
                        active_run.run_id, hot, error)) {
                    return false;
                }
                if (!hot.found) {
                    error.clear();
                    return true;
                }
                generated_at_ms = hot.analysis_last_update_ms > 0
                    ? hot.analysis_last_update_ms : hot.last_update_ms;

                auto state = json::parse(
                    hot.analysis_state_json, nullptr, false);
                if (state.is_discarded() || !state.is_object() ||
                    state.value("mode", "") != "vehicle_cascade") {
                    return true;
                }
                const auto items = state.value("items", json::array());
                if (!items.is_array()) return true;

                for (const auto& item : items) {
                    if (!item.is_object()) continue;
                    VehicleRealtimeTrackRecord track;
                    track.camera_id = camera_id;
                    track.run_id = hot.run_id.empty()
                        ? active_run.run_id : hot.run_id;
                    track.run_generation = item.value("run_generation", 0ULL);
                    track.track_id = item.value("track_id", 0LL);
                    track.state = item.value("state", "unknown");
                    track.last_seen_at_ms = item.value(
                        "last_seen_at_ms", generated_at_ms);

                    const auto vehicle_it = item.find("vehicle_class");
                    if (vehicle_it != item.end() && vehicle_it->is_object()) {
                        track.vehicle_class = vehicle_it->value(
                            "label", std::string("unknown"));
                        track.vehicle_class_confidence = vehicle_it->value(
                            "confidence", 0.0);
                    }
                    else {
                        track.vehicle_class = item.value(
                            "vehicle_class", std::string("unknown"));
                        track.vehicle_class_confidence = item.value(
                            "vehicle_class_confidence", 0.0);
                    }

                    const auto attributes_it = item.find("attributes");
                    const auto readAttribute = [&item, attributes_it](
                            const char* name,
                            const char* flat_label,
                            const char* flat_confidence,
                            const char* flat_stable,
                            const char* flat_samples,
                            std::string& label,
                            double& confidence,
                            bool& stable,
                            int& samples) {
                        if (attributes_it != item.end() &&
                            attributes_it->is_object()) {
                            const auto it = attributes_it->find(name);
                            if (it != attributes_it->end() && it->is_object()) {
                                label = it->value("label", std::string("unknown"));
                                confidence = it->value("confidence", 0.0);
                                stable = it->value("stable", false);
                                samples = it->value("samples_used", 0);
                                return;
                            }
                        }
                        label = item.value(flat_label, std::string("unknown"));
                        confidence = item.value(flat_confidence, 0.0);
                        stable = item.value(flat_stable, false);
                        samples = item.value(flat_samples, 0);
                    };
                    readAttribute(
                        "body_type", "body_type", "body_type_confidence",
                        "body_type_stable", "body_type_samples_used",
                        track.body_type, track.body_type_confidence,
                        track.body_type_stable, track.body_type_samples_used);
                    readAttribute(
                        "color", "color", "color_confidence", "color_stable",
                        "color_samples_used", track.color,
                        track.color_confidence, track.color_stable,
                        track.color_samples_used);
                    tracks.push_back(std::move(track));
                }
                return true;
            });
    }
}

bool VisionHttpServer::initialize(std::string& error) {
    if (!config_.redis.enabled) {
        error = "redis.enabled must be true";
        return false;
    }
    if (!config_.camera_tasks.enabled || !camera_task_controller_) {
        error = "camera_tasks.enabled must be true";
        return false;
    }
    if (!redis_.connect(error)) return false;
    return camera_task_controller_->initialize(error);
}

void VisionHttpServer::registerRoutes(crow::SimpleApp& app) {
    CROW_ROUTE(app, "/camera-admin")([this]() {
        return adminAsset("index.html", "text/html; charset=utf-8");
    });
    CROW_ROUTE(app, "/camera-admin/app.js")([this]() {
        return adminAsset("app.js", "application/javascript; charset=utf-8");
    });
    CROW_ROUTE(app, "/camera-admin/styles.css")([this]() {
        return adminAsset("styles.css", "text/css; charset=utf-8");
    });
    CROW_ROUTE(app, "/api/v1/health")([this]() { return health(); });
    CROW_ROUTE(app, "/api/v1/ready")([this]() { return ready(); });
    camera_task_controller_->registerRoutes(app);
}

crow::response VisionHttpServer::adminAsset(
    const std::string& file_name,
    const std::string& content_type
) const {
    std::string bytes;
    const auto root = std::filesystem::absolute(
        std::filesystem::u8path(config_.camera_tasks.admin_ui_dir));
    if (!readFile(root / std::filesystem::u8path(file_name), bytes)) {
        return crow::response(404, "Camera admin asset was not found");
    }
    crow::response response(200, std::move(bytes));
    response.set_header("Content-Type", content_type);
    response.set_header("Cache-Control", file_name == "index.html" ? "no-store" : "public, max-age=300");
    response.set_header("X-Content-Type-Options", "nosniff");
    response.set_header("Content-Security-Policy",
        "default-src 'self'; base-uri 'none'; object-src 'none'; "
        "frame-ancestors 'none'; form-action 'self'; "
        "img-src 'self' blob:; style-src 'self'; "
        "script-src 'self'; connect-src 'self'");
    response.set_header("Referrer-Policy", "no-referrer");
    response.set_header(
        "Permissions-Policy", "camera=(), microphone=(), geolocation=()");
    return response;
}

crow::response VisionHttpServer::health() const {
    std::string error;
    const bool redis_ok = redis_.ping(error);
    const auto camera = camera_task_controller_->health();
    const bool camera_ok = camera.initialized && camera.storage_ok &&
        camera.output_root_writable && camera.worker_num_valid &&
        camera.callback_config_valid;
    const bool healthy = redis_ok && camera_ok;
    return jsonResponse(healthy ? 200 : 503, {
        {"success", healthy}, {"service", "car-detect"},
        {"redis", {{"ok", redis_ok}, {"error", error}}},
        {"camera_tasks", {
            {"enabled", camera.enabled}, {"initialized", camera.initialized},
            {"storage_ok", camera.storage_ok},
            {"output_root_writable", camera.output_root_writable},
            {"token_configured", camera.token_configured},
            {"worker_num_valid", camera.worker_num_valid},
            {"callback_config_valid", camera.callback_config_valid},
            {"analysis_enabled", config_.analysis.enabled},
            {"callbacks_enabled", config_.callbacks.enabled}
        }}
    });
}

crow::response VisionHttpServer::ready() const {
    std::string redis_error;
    const bool redis_ok = redis_.ping(redis_error);
    std::vector<WorkerHeartbeatRecord> workers;
    std::string worker_error;
    const bool workers_ok = redis_ok && redis_.getWorkerHeartbeats(
        config_.worker.consumer_name_prefix, config_.worker.worker_num,
        workers, worker_error);
    int alive = 0;
    json worker_items = json::array();
    for (const auto& worker : workers) {
        if (worker.alive) ++alive;
        worker_items.push_back({
            {"consumer_name", worker.consumer_name}, {"alive", worker.alive},
            {"status", worker.status},
            {"runner_model_type", worker.runner_model_type},
            {"worker_group", worker.worker_group},
            {"runtime_mode", worker.runtime_mode},
            {"worker_generation", worker.worker_generation},
            {"camera_task_manager_running", worker.camera_task_manager_running},
            {"hub_registry_ready", worker.hub_registry_ready},
            {"coordination_healthy", worker.coordination_healthy},
            {"last_error", worker.last_error}
        });
    }
    const auto camera = camera_task_controller_->health();
    const auto worker_readiness = evaluateWorkerRuntimeReadiness(
        workers, camera.enabled);
    const auto& algorithm_runtime = worker_readiness.algorithm_runtime;
    const bool algorithm_runtime_available =
        !camera.enabled || algorithm_runtime.generated_at_ms > 0;
    const bool algorithm_runtime_fresh = !camera.enabled ||
        (algorithm_runtime_available &&
            nowMs() - algorithm_runtime.generated_at_ms <=
                std::max(5000, config_.worker.heartbeat_interval_ms * 3));
    const bool inference_pool_ready =
        !camera.enabled || !config_.analysis.enabled ||
        (algorithm_runtime.inference_running &&
            algorithm_runtime.processor_running &&
            algorithm_runtime.inference_workers_configured ==
                config_.analysis.inference_workers &&
            algorithm_runtime.inference_workers_ready ==
                config_.analysis.inference_workers);
    const bool callback_delivery_ready =
        !camera.enabled || !config_.callbacks.enabled ||
        (algorithm_runtime.callback_running &&
            algorithm_runtime.callback_profiles_ready > 0);
    const bool camera_ready = !camera.enabled ||
        (camera.initialized && camera.token_configured && camera.storage_ok &&
            camera.output_root_writable && camera.worker_num_valid &&
            camera.callback_config_valid &&
            worker_readiness.camera_role_alive &&
            worker_readiness.single_vision_worker &&
            worker_readiness.mode_consistent &&
            worker_readiness.coordination_healthy &&
            worker_readiness.camera_task_manager_running &&
            worker_readiness.hub_registry_ready &&
            algorithm_runtime_available && algorithm_runtime_fresh &&
            algorithm_runtime.host_running && inference_pool_ready &&
            callback_delivery_ready);
    const bool is_ready = redis_ok && workers_ok &&
        alive >= config_.worker.min_alive_workers && camera_ready;
    return jsonResponse(is_ready ? 200 : 503, {
        {"success", is_ready}, {"ready", is_ready},
        {"redis_ok", redis_ok}, {"alive_workers", alive},
        {"required_workers", config_.worker.min_alive_workers},
        {"worker_error", worker_error}, {"workers", worker_items},
        {"camera_tasks_ready", camera_ready},
        {"camera_frame_role_alive", worker_readiness.camera_role_alive},
        {"expected_runtime_mode", "vehicle_camera_pipeline"},
        {"worker_mode_consistent", worker_readiness.mode_consistent},
        {"single_vision_worker", worker_readiness.single_vision_worker},
        {"worker_coordination_healthy", worker_readiness.coordination_healthy},
        {"camera_task_manager_running", worker_readiness.camera_task_manager_running},
        {"hub_registry_ready", worker_readiness.hub_registry_ready},
        {"algorithm_runtime_available", algorithm_runtime_available},
        {"algorithm_runtime_fresh", algorithm_runtime_fresh},
        {"algorithm_runtime_generated_at_ms", algorithm_runtime.generated_at_ms},
        {"inference_pool_ready", inference_pool_ready},
        {"callback_delivery_ready", callback_delivery_ready}
    });
}

bool VisionHttpServer::readAlgorithmRuntime(
    AlgorithmRuntimeSnapshot& runtime,
    std::string& error
) const {
    runtime = {};
    std::vector<WorkerHeartbeatRecord> workers;
    if (!redis_.getWorkerHeartbeats(
            config_.worker.consumer_name_prefix,
            config_.worker.worker_num,
            workers,
            error)) {
        return false;
    }
    for (const auto& worker : workers) {
        if (!worker.alive || worker.worker_kind != "vision_host") continue;
        if (worker.algorithm_runtime.generated_at_ms > runtime.generated_at_ms) {
            runtime = worker.algorithm_runtime;
        }
    }
    if (runtime.generated_at_ms <= 0) {
        error = "ALGORITHM_RUNTIME_UNAVAILABLE";
        return false;
    }
    error.clear();
    return true;
}

}  // namespace yolo11_server
