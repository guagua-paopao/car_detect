#include "server/vehicle_attribute_batch_scheduler.h"

#include <algorithm>
#include <cmath>
#include <exception>
#include <stdexcept>
#include <utility>

namespace yolo11_server {

namespace {

constexpr double kQueueWaitBucketWidthMs = 0.5;

double queueWaitPercentile(
    const std::array<std::uint64_t, 202>& histogram,
    std::uint64_t samples,
    double quantile) {
    if (samples == 0) return 0.0;
    const auto target = static_cast<std::uint64_t>(
        std::ceil(static_cast<double>(samples) * quantile));
    std::uint64_t cumulative = 0;
    for (std::size_t index = 0; index < histogram.size(); ++index) {
        cumulative += histogram[index];
        if (cumulative >= target) {
            return static_cast<double>(index) * kQueueWaitBucketWidthMs;
        }
    }
    return static_cast<double>(histogram.size() - 1) *
        kQueueWaitBucketWidthMs;
}

}  // namespace

struct VehicleAttributeBatchScheduler::Request {
    std::vector<VehicleAttributeCrop> crops;
    std::vector<VehicleAttributeResult> results;
    std::string error;
    std::chrono::steady_clock::time_point queued_at;
    std::mutex mutex;
    std::condition_variable completed;
    bool done = false;
};

VehicleAttributeBatchScheduler::VehicleAttributeBatchScheduler(
    VehicleAttributeBatchSchedulerConfig config,
    VehicleAttributeRunnerPtr runner)
    : config_(config), runner_(std::move(runner)) {
    config_.max_batch = std::clamp<std::size_t>(config_.max_batch, 1, 16);
    config_.max_pending_requests =
        std::max<std::size_t>(1, config_.max_pending_requests);
    config_.max_wait = std::clamp(
        config_.max_wait,
        std::chrono::milliseconds(0),
        std::chrono::milliseconds(100));
}

VehicleAttributeBatchScheduler::~VehicleAttributeBatchScheduler() noexcept {
    stop();
}

bool VehicleAttributeBatchScheduler::start(std::string& error) {
    error.clear();
    std::lock_guard<std::mutex> lock(mutex_);
    if (running_) return true;
    if (!runner_) {
        error = "attribute batch scheduler runner is unavailable";
        return false;
    }
    if (worker_.joinable() || !pending_.empty()) {
        error = "attribute batch scheduler has incomplete prior state";
        return false;
    }
    stopping_ = false;
    running_ = true;
    metrics_ = {};
    metrics_.running = true;
    queue_wait_sum_ms_ = 0.0;
    queue_wait_samples_ = 0;
    pending_crops_ = 0;
    queue_wait_histogram_.fill(0);
    try {
        worker_ = std::thread([this]() { workerLoop(); });
    }
    catch (const std::exception& exception) {
        running_ = false;
        metrics_.running = false;
        error = std::string("could not start attribute batch scheduler: ") +
            exception.what();
        return false;
    }
    return true;
}

void VehicleAttributeBatchScheduler::stop() noexcept {
    try {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!running_ && !worker_.joinable()) return;
            stopping_ = true;
            running_ = false;
            metrics_.running = false;
        }
        ready_.notify_all();
        capacity_available_.notify_all();
        if (worker_.joinable()) worker_.join();
        if (runner_) runner_->release();
    }
    catch (...) {
    }
}

bool VehicleAttributeBatchScheduler::infer(
    std::vector<VehicleAttributeCrop> crops,
    std::vector<VehicleAttributeResult>& results,
    std::string& error) {
    results.clear();
    error.clear();
    if (crops.empty() || crops.size() > config_.max_batch) {
        error = "attribute scheduler request batch is out of range";
        return false;
    }
    auto request = std::make_shared<Request>();
    request->crops = std::move(crops);
    request->queued_at = std::chrono::steady_clock::now();
    {
        std::unique_lock<std::mutex> lock(mutex_);
        capacity_available_.wait(lock, [&]() {
            return stopping_ || !running_ ||
                pending_.size() < config_.max_pending_requests;
        });
        if (stopping_ || !running_) {
            error = "attribute batch scheduler is not running";
            return false;
        }
        pending_.push_back(request);
        pending_crops_ += request->crops.size();
        ++metrics_.requests;
        metrics_.maximum_pending_requests = std::max(
            metrics_.maximum_pending_requests, pending_.size());
        metrics_.maximum_pending_crops = std::max(
            metrics_.maximum_pending_crops, pending_crops_);
    }
    ready_.notify_one();

    std::unique_lock<std::mutex> request_lock(request->mutex);
    request->completed.wait(request_lock, [&]() { return request->done; });
    error = std::move(request->error);
    results = std::move(request->results);
    return error.empty();
}

VehicleAttributeBatchSchedulerSnapshot
VehicleAttributeBatchScheduler::snapshot() const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto result = metrics_;
    result.pending_requests = pending_.size();
    result.pending_crops = pending_crops_;
    result.mean_queue_wait_ms = queue_wait_samples_ == 0
        ? 0.0 : queue_wait_sum_ms_ / static_cast<double>(queue_wait_samples_);
    result.p95_queue_wait_ms = queueWaitPercentile(
        queue_wait_histogram_, queue_wait_samples_, 0.95);
    result.p99_queue_wait_ms = queueWaitPercentile(
        queue_wait_histogram_, queue_wait_samples_, 0.99);
    return result;
}

void VehicleAttributeBatchScheduler::workerLoop() noexcept {
    while (true) {
        std::vector<std::shared_ptr<Request>> requests;
        std::vector<VehicleAttributeCrop> batch;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            ready_.wait(lock, [&]() { return stopping_ || !pending_.empty(); });
            if (stopping_ && pending_.empty()) break;
            const auto deadline = pending_.front()->queued_at + config_.max_wait;
            auto pending_crops = [&]() {
                std::size_t count = 0;
                for (const auto& request : pending_) {
                    if (count + request->crops.size() > config_.max_batch) break;
                    count += request->crops.size();
                }
                return count;
            };
            while (!stopping_ && pending_crops() < config_.max_batch &&
                   std::chrono::steady_clock::now() < deadline) {
                ready_.wait_until(lock, deadline);
            }
            std::size_t count = 0;
            while (!pending_.empty()) {
                const auto& request = pending_.front();
                if (!requests.empty() &&
                    count + request->crops.size() > config_.max_batch) break;
                if (request->crops.size() > config_.max_batch) {
                    request->error = "attribute scheduler internal batch overflow";
                    requests.push_back(request);
                    pending_.pop_front();
                    break;
                }
                count += request->crops.size();
                pending_crops_ -= request->crops.size();
                requests.push_back(request);
                pending_.pop_front();
                if (count == config_.max_batch) break;
            }
            const auto now = std::chrono::steady_clock::now();
            for (const auto& request : requests) {
                const double wait_ms = std::chrono::duration<double, std::milli>(
                    now - request->queued_at).count();
                queue_wait_sum_ms_ += wait_ms;
                ++queue_wait_samples_;
                const auto bucket = std::min(
                    queue_wait_histogram_.size() - 1,
                    static_cast<std::size_t>(
                        std::max(0.0, wait_ms) / kQueueWaitBucketWidthMs));
                ++queue_wait_histogram_[bucket];
                metrics_.maximum_queue_wait_ms = std::max(
                    metrics_.maximum_queue_wait_ms, wait_ms);
                batch.insert(
                    batch.end(), request->crops.begin(), request->crops.end());
            }
            ++metrics_.batches;
            metrics_.crops += batch.size();
            if (batch.size() < metrics_.batch_histogram.size()) {
                ++metrics_.batch_histogram[batch.size()];
            }
        }
        capacity_available_.notify_all();

        std::vector<VehicleAttributeResult> combined;
        std::string failure;
        try {
            combined = runner_->inferBatch(batch);
            if (combined.size() != batch.size()) {
                failure = "attribute scheduler runner result count mismatch";
            }
        }
        catch (const std::exception& exception) {
            failure = exception.what();
        }
        catch (...) {
            failure = "unknown attribute scheduler runner failure";
        }

        {
            std::lock_guard<std::mutex> lock(mutex_);
            metrics_.completed_requests += requests.size();
            if (!failure.empty()) metrics_.failed_requests += requests.size();
        }

        std::size_t offset = 0;
        for (const auto& request : requests) {
            {
                std::lock_guard<std::mutex> request_lock(request->mutex);
                request->error = failure;
                if (failure.empty()) {
                    const auto count = request->crops.size();
                    request->results.assign(
                        combined.begin() + static_cast<std::ptrdiff_t>(offset),
                        combined.begin() + static_cast<std::ptrdiff_t>(offset + count));
                    offset += count;
                }
                request->done = true;
            }
            request->completed.notify_one();
        }
    }
}

}  // namespace yolo11_server
