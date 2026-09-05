#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "server/worker_runtime_readiness.h"

namespace {

using namespace yolo11_server;

void require(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

WorkerHeartbeatRecord vehicleWorker() {
    WorkerHeartbeatRecord worker;
    worker.alive = true;
    worker.worker_kind = "vision_host";
    worker.task_kind = "camera_pipeline";
    worker.runtime_mode = "vehicle_camera_pipeline";
    worker.worker_generation = "200:1234";
    worker.camera_task_manager_running = true;
    worker.hub_registry_ready = true;
    worker.coordination_healthy = true;
    worker.algorithm_runtime.generated_at_ms = 1235;
    worker.algorithm_runtime.host_running = true;
    return worker;
}

}  // namespace

int main() {
    const auto healthy = evaluateWorkerRuntimeReadiness(
        { vehicleWorker() }, true);
    require(healthy.camera_role_alive &&
            healthy.single_vision_worker &&
            healthy.mode_consistent &&
            healthy.coordination_healthy &&
            healthy.camera_task_manager_running &&
            healthy.hub_registry_ready &&
            healthy.algorithm_runtime.host_running,
        "one fenced vehicle Camera Worker must satisfy readiness");

    auto incompatible = vehicleWorker();
    incompatible.task_kind = "camera_frame";
    incompatible.runtime_mode = "unknown";
    const auto mismatch = evaluateWorkerRuntimeReadiness(
        { incompatible }, true);
    require(!mismatch.camera_role_alive &&
            mismatch.single_vision_worker &&
            !mismatch.mode_consistent,
        "/ready must reject an incompatible Worker");

    auto duplicate = vehicleWorker();
    duplicate.worker_generation = "201:1236";
    const auto double_consumer = evaluateWorkerRuntimeReadiness(
        { vehicleWorker(), duplicate }, true);
    require(double_consumer.camera_role_alive &&
            !double_consumer.single_vision_worker,
        "/ready must reject two live Vision Worker generations");

    auto unfenced = vehicleWorker();
    unfenced.coordination_healthy = false;
    const auto lease_lost = evaluateWorkerRuntimeReadiness(
        { unfenced }, true);
    require(!lease_lost.coordination_healthy,
        "/ready must reject a Worker that lost process ownership");

    std::cout << "Worker runtime readiness tests passed\n";
    return 0;
}
