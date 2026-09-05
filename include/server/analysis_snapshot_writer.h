#pragma once

#include <condition_variable>
#include <cstddef>
#include <deque>
#include <filesystem>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/core.hpp>

namespace yolo11_server {

struct AnalysisSnapshotWriteResult {
    bool success = false;
    std::size_t encoded_bytes = 0;
    std::string error;
};

struct AnalysisSnapshotWriterMetrics {
    bool running = false;
    std::size_t queue_depth = 0;
    std::size_t active_jobs = 0;
    std::size_t maximum_queue_depth = 0;
    unsigned long long submitted_jobs = 0;
    unsigned long long completed_jobs = 0;
    unsigned long long failed_jobs = 0;
    double mean_enqueue_wait_ms = 0.0;
    double maximum_enqueue_wait_ms = 0.0;
};

struct AnalysisSnapshotJob {
    cv::Mat image;
    std::filesystem::path output_path;
    int jpeg_quality = 90;
};

using AnalysisSnapshotCompletion =
    std::function<void(const AnalysisSnapshotWriteResult&)>;

// A fixed-size CPU encoder/writer pool. Producers block when its bounded queue
// is full, so snapshot work can overlap GPU inference without silent loss.
class AnalysisSnapshotWriter final {
public:
    AnalysisSnapshotWriter() = default;
    ~AnalysisSnapshotWriter() noexcept;

    AnalysisSnapshotWriter(const AnalysisSnapshotWriter&) = delete;
    AnalysisSnapshotWriter& operator=(const AnalysisSnapshotWriter&) = delete;

    bool start(int worker_count, std::size_t queue_capacity, std::string& error);
    void stop() noexcept;
    bool enqueue(
        AnalysisSnapshotJob job,
        AnalysisSnapshotCompletion completion,
        std::string& error);
    bool waitIdle(int timeout_ms);
    AnalysisSnapshotWriterMetrics metrics() const;

private:
    struct PendingJob {
        AnalysisSnapshotJob job;
        AnalysisSnapshotCompletion completion;
    };

    void workerLoop() noexcept;
    static AnalysisSnapshotWriteResult write(
        const AnalysisSnapshotJob& job) noexcept;

    mutable std::mutex mutex_;
    std::condition_variable work_cv_;
    std::condition_variable space_cv_;
    std::condition_variable idle_cv_;
    std::deque<PendingJob> queue_;
    std::vector<std::thread> workers_;
    std::size_t capacity_ = 1;
    std::size_t active_jobs_ = 0;
    std::size_t maximum_queue_depth_ = 0;
    unsigned long long submitted_jobs_ = 0;
    unsigned long long completed_jobs_ = 0;
    unsigned long long failed_jobs_ = 0;
    double enqueue_wait_sum_ms_ = 0.0;
    double maximum_enqueue_wait_ms_ = 0.0;
    bool accepting_ = false;
    bool stopping_ = false;
};

}  // namespace yolo11_server
