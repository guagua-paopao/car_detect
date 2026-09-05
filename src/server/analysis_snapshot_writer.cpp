#include "server/analysis_snapshot_writer.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <fstream>
#include <system_error>
#include <utility>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

#include <opencv2/imgcodecs.hpp>

namespace yolo11_server {

namespace {

std::atomic<unsigned long long> temporary_counter{0};

bool publishEncoded(
    const std::filesystem::path& output_path,
    const std::vector<unsigned char>& encoded,
    std::string& error) {
    if (output_path.empty()) return true;
    std::error_code fs_error;
    std::filesystem::create_directories(output_path.parent_path(), fs_error);
    if (fs_error) {
        error = fs_error.message();
        return false;
    }
    const auto suffix = temporary_counter.fetch_add(1) + 1;
    auto temporary = output_path;
    temporary += ".snapshot-" + std::to_string(suffix) + ".tmp";
    {
        std::ofstream stream(temporary, std::ios::binary | std::ios::trunc);
        if (!stream) {
            error = "could not create temporary snapshot";
            return false;
        }
        stream.write(
            reinterpret_cast<const char*>(encoded.data()),
            static_cast<std::streamsize>(encoded.size()));
        stream.flush();
        if (!stream) {
            error = "could not write temporary snapshot";
            stream.close();
            std::filesystem::remove(temporary, fs_error);
            return false;
        }
    }
#ifdef _WIN32
    if (!MoveFileExW(
            temporary.c_str(), output_path.c_str(),
            MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
        error = "could not atomically publish snapshot, win32_error=" +
            std::to_string(GetLastError());
        std::filesystem::remove(temporary, fs_error);
        return false;
    }
#else
    std::filesystem::rename(temporary, output_path, fs_error);
    if (fs_error) {
        error = fs_error.message();
        std::filesystem::remove(temporary, fs_error);
        return false;
    }
#endif
    return true;
}

}  // namespace

AnalysisSnapshotWriter::~AnalysisSnapshotWriter() noexcept {
    stop();
}

bool AnalysisSnapshotWriter::start(
    int worker_count,
    std::size_t queue_capacity,
    std::string& error) {
    error.clear();
    std::lock_guard<std::mutex> lock(mutex_);
    if (accepting_) return true;
    if (!workers_.empty()) {
        error = "snapshot writer has not stopped cleanly";
        return false;
    }
    capacity_ = std::max<std::size_t>(1, queue_capacity);
    accepting_ = true;
    stopping_ = false;
    try {
        for (int index = 0; index < std::max(1, worker_count); ++index) {
            workers_.emplace_back([this]() { workerLoop(); });
        }
    }
    catch (const std::exception& exception) {
        accepting_ = false;
        stopping_ = true;
        work_cv_.notify_all();
        error = exception.what();
        return false;
    }
    return true;
}

void AnalysisSnapshotWriter::stop() noexcept {
    try {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!accepting_ && workers_.empty()) return;
            accepting_ = false;
            stopping_ = true;
        }
        work_cv_.notify_all();
        space_cv_.notify_all();
        for (auto& worker : workers_) {
            if (worker.joinable()) worker.join();
        }
        workers_.clear();
        std::lock_guard<std::mutex> lock(mutex_);
        queue_.clear();
        active_jobs_ = 0;
        stopping_ = false;
        idle_cv_.notify_all();
    }
    catch (...) {
    }
}

bool AnalysisSnapshotWriter::enqueue(
    AnalysisSnapshotJob job,
    AnalysisSnapshotCompletion completion,
    std::string& error) {
    error.clear();
    if (job.image.empty()) {
        error = "snapshot image is empty";
        return false;
    }
    const auto started = std::chrono::steady_clock::now();
    std::unique_lock<std::mutex> lock(mutex_);
    space_cv_.wait(lock, [&]() {
        return !accepting_ || queue_.size() < capacity_;
    });
    const double waited = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - started).count();
    if (!accepting_) {
        error = "snapshot writer is not accepting jobs";
        return false;
    }
    enqueue_wait_sum_ms_ += waited;
    maximum_enqueue_wait_ms_ = std::max(maximum_enqueue_wait_ms_, waited);
    ++submitted_jobs_;
    queue_.push_back({std::move(job), std::move(completion)});
    maximum_queue_depth_ = std::max(maximum_queue_depth_, queue_.size());
    work_cv_.notify_one();
    return true;
}

bool AnalysisSnapshotWriter::waitIdle(int timeout_ms) {
    std::unique_lock<std::mutex> lock(mutex_);
    return idle_cv_.wait_for(
        lock, std::chrono::milliseconds(std::max(0, timeout_ms)),
        [&]() { return queue_.empty() && active_jobs_ == 0; });
}

AnalysisSnapshotWriterMetrics AnalysisSnapshotWriter::metrics() const {
    std::lock_guard<std::mutex> lock(mutex_);
    AnalysisSnapshotWriterMetrics result;
    result.running = accepting_;
    result.queue_depth = queue_.size();
    result.active_jobs = active_jobs_;
    result.maximum_queue_depth = maximum_queue_depth_;
    result.submitted_jobs = submitted_jobs_;
    result.completed_jobs = completed_jobs_;
    result.failed_jobs = failed_jobs_;
    result.mean_enqueue_wait_ms = submitted_jobs_ == 0 ? 0.0 :
        enqueue_wait_sum_ms_ / static_cast<double>(submitted_jobs_);
    result.maximum_enqueue_wait_ms = maximum_enqueue_wait_ms_;
    return result;
}

void AnalysisSnapshotWriter::workerLoop() noexcept {
    for (;;) {
        PendingJob pending;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            work_cv_.wait(lock, [&]() {
                return stopping_ || !queue_.empty();
            });
            if (queue_.empty() && stopping_) return;
            pending = std::move(queue_.front());
            queue_.pop_front();
            ++active_jobs_;
            space_cv_.notify_one();
        }
        const auto result = write(pending.job);
        try {
            if (pending.completion) pending.completion(result);
        }
        catch (...) {
        }
        {
            std::lock_guard<std::mutex> lock(mutex_);
            --active_jobs_;
            if (result.success) ++completed_jobs_;
            else ++failed_jobs_;
            if (queue_.empty() && active_jobs_ == 0) idle_cv_.notify_all();
        }
    }
}

AnalysisSnapshotWriteResult AnalysisSnapshotWriter::write(
    const AnalysisSnapshotJob& job) noexcept {
    AnalysisSnapshotWriteResult result;
    try {
        std::vector<unsigned char> encoded;
        if (!cv::imencode(
                ".jpg", job.image, encoded,
                {cv::IMWRITE_JPEG_QUALITY,
                    std::clamp(job.jpeg_quality, 1, 100)})) {
            result.error = "OpenCV JPEG encoder returned false";
            return result;
        }
        result.encoded_bytes = encoded.size();
        if (!publishEncoded(job.output_path, encoded, result.error)) {
            return result;
        }
        result.success = true;
        return result;
    }
    catch (const std::exception& exception) {
        result.error = exception.what();
        return result;
    }
    catch (...) {
        result.error = "unknown snapshot encoder error";
        return result;
    }
}

}  // namespace yolo11_server
