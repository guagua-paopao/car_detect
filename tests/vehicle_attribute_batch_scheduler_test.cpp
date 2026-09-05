#include "server/vehicle_attribute_batch_scheduler.h"

#include <atomic>
#include <condition_variable>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using namespace yolo11_server;

namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

class RecordingRunner final : public IVehicleAttributeRunner {
public:
    bool initialize(const ModelArtifactDescriptor&, std::string&) override {
        return true;
    }

    std::vector<VehicleAttributeResult> inferBatch(
        const std::vector<VehicleAttributeCrop>& crops) override {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            batches_.push_back(crops.size());
        }
        std::vector<VehicleAttributeResult> results;
        for (const auto& crop : crops) {
            VehicleAttributeResult result;
            result.camera_id = crop.camera_id;
            result.run_id = crop.run_id;
            result.track_id = crop.track_id;
            result.crop_sequence = crop.crop_sequence;
            result.metadata.run_generation = crop.run_generation;
            result.body_type = {"sedan", 0.9f};
            result.color = {"blue", 0.8f};
            results.push_back(std::move(result));
        }
        return results;
    }

    void release() noexcept override {}

    std::vector<std::size_t> batches() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return batches_;
    }

private:
    mutable std::mutex mutex_;
    std::vector<std::size_t> batches_;
};

VehicleAttributeCrop crop(std::string camera, std::int64_t track) {
    auto image = std::make_shared<OwnedImage>();
    image->width = 2;
    image->height = 2;
    image->channels = 3;
    image->row_stride_bytes = 6;
    image->pixels.assign(12, 127);
    VehicleAttributeCrop result;
    result.crop = std::move(image);
    result.camera_id = std::move(camera);
    result.run_id = "run";
    result.run_generation = 1;
    result.track_id = track;
    result.crop_sequence = 1;
    result.quality_score = 1.0f;
    return result;
}

}  // namespace

int main() {
    try {
        auto runner = std::make_unique<RecordingRunner>();
        auto* recording = runner.get();
        VehicleAttributeBatchSchedulerConfig config;
        config.max_batch = 16;
        config.max_pending_requests = 4;
        config.max_wait = std::chrono::milliseconds(50);
        VehicleAttributeBatchScheduler scheduler(config, std::move(runner));
        std::string error;
        require(scheduler.start(error), "scheduler must start");

        std::mutex gate_mutex;
        std::condition_variable gate;
        int waiting = 0;
        bool release = false;
        std::atomic<int> completed{0};
        std::vector<std::thread> callers;
        for (int camera = 0; camera < 3; ++camera) {
            callers.emplace_back([&, camera]() {
                {
                    std::unique_lock<std::mutex> lock(gate_mutex);
                    ++waiting;
                    gate.notify_all();
                    gate.wait(lock, [&]() { return release; });
                }
                std::vector<VehicleAttributeCrop> crops;
                crops.push_back(crop("camera-" + std::to_string(camera), 1));
                crops.push_back(crop("camera-" + std::to_string(camera), 2));
                std::vector<VehicleAttributeResult> results;
                std::string request_error;
                require(scheduler.infer(
                    std::move(crops), results, request_error),
                    "batched request must succeed");
                require(results.size() == 2,
                    "request must receive exactly its own results");
                require(results[0].camera_id ==
                    "camera-" + std::to_string(camera),
                    "camera result routing must be preserved");
                ++completed;
            });
        }
        {
            std::unique_lock<std::mutex> lock(gate_mutex);
            gate.wait(lock, [&]() { return waiting == 3; });
            release = true;
        }
        gate.notify_all();
        for (auto& caller : callers) caller.join();

        const auto snapshot = scheduler.snapshot();
        const auto batches = recording->batches();
        require(completed.load() == 3 && snapshot.requests == 3 &&
                snapshot.completed_requests == 3 &&
                snapshot.failed_requests == 0 && snapshot.crops == 6,
            "all bounded scheduler requests must complete");
        require(batches.size() == 1 && batches.front() == 6 &&
                snapshot.batch_histogram[6] == 1,
            "concurrent camera requests must aggregate into one batch");
        require(snapshot.maximum_pending_requests <=
                config.max_pending_requests &&
                snapshot.maximum_pending_crops <=
                    config.max_pending_requests * 2 &&
                snapshot.pending_requests == 0 && snapshot.pending_crops == 0,
            "scheduler queue must remain bounded");
        require(snapshot.mean_queue_wait_ms > 0.0 &&
                snapshot.p95_queue_wait_ms <=
                    snapshot.maximum_queue_wait_ms + 0.5 &&
                snapshot.p99_queue_wait_ms <=
                    snapshot.maximum_queue_wait_ms + 0.5,
            "bounded scheduler queue latency percentiles must be observable");
        scheduler.stop();
        std::cout << "PASS: bounded multi-camera attribute batching and routing\n";
        return EXIT_SUCCESS;
    }
    catch (const std::exception& exception) {
        std::cerr << "FAIL: " << exception.what() << '\n';
        return EXIT_FAILURE;
    }
}
