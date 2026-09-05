#pragma once

#include <array>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "server/vehicle_model_contract.h"

namespace yolo11_server {

struct VehicleAttributeBatchSchedulerConfig {
    std::size_t max_batch = 16;
    std::size_t max_pending_requests = 64;
    std::chrono::milliseconds max_wait{2};
};

struct VehicleAttributeBatchSchedulerSnapshot {
    bool running = false;
    std::uint64_t requests = 0;
    std::uint64_t completed_requests = 0;
    std::uint64_t failed_requests = 0;
    std::uint64_t batches = 0;
    std::uint64_t crops = 0;
    std::size_t pending_requests = 0;
    std::size_t maximum_pending_requests = 0;
    std::size_t pending_crops = 0;
    std::size_t maximum_pending_crops = 0;
    double mean_queue_wait_ms = 0.0;
    double p95_queue_wait_ms = 0.0;
    double p99_queue_wait_ms = 0.0;
    double maximum_queue_wait_ms = 0.0;
    std::array<std::uint64_t, 17> batch_histogram{};
};

// A single bounded FIFO worker aggregates synchronous callers from multiple
// cameras for at most max_wait. Each camera request remains ordered and no
// request is discarded to form a larger batch.
class VehicleAttributeBatchScheduler {
public:
    VehicleAttributeBatchScheduler(
        VehicleAttributeBatchSchedulerConfig config,
        VehicleAttributeRunnerPtr runner);
    ~VehicleAttributeBatchScheduler() noexcept;

    VehicleAttributeBatchScheduler(
        const VehicleAttributeBatchScheduler&) = delete;
    VehicleAttributeBatchScheduler& operator=(
        const VehicleAttributeBatchScheduler&) = delete;

    bool start(std::string& error);
    void stop() noexcept;
    bool infer(
        std::vector<VehicleAttributeCrop> crops,
        std::vector<VehicleAttributeResult>& results,
        std::string& error);
    VehicleAttributeBatchSchedulerSnapshot snapshot() const;

private:
    struct Request;
    void workerLoop() noexcept;

    VehicleAttributeBatchSchedulerConfig config_;
    VehicleAttributeRunnerPtr runner_;
    mutable std::mutex mutex_;
    std::condition_variable ready_;
    std::condition_variable capacity_available_;
    std::deque<std::shared_ptr<Request>> pending_;
    std::thread worker_;
    bool running_ = false;
    bool stopping_ = false;
    VehicleAttributeBatchSchedulerSnapshot metrics_;
    double queue_wait_sum_ms_ = 0.0;
    std::uint64_t queue_wait_samples_ = 0;
    std::size_t pending_crops_ = 0;
    static constexpr std::size_t kQueueWaitHistogramBuckets = 202;
    std::array<std::uint64_t, kQueueWaitHistogramBuckets>
        queue_wait_histogram_{};
};

}  // namespace yolo11_server
