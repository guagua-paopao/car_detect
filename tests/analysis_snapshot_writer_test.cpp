#include "server/analysis_snapshot_writer.h"

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>

#include <opencv2/core.hpp>

using namespace yolo11_server;

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

}  // namespace

int main() {
    const auto unique = std::chrono::steady_clock::now()
        .time_since_epoch().count();
    const auto root = std::filesystem::temp_directory_path() /
        ("vcas-analysis-snapshot-writer-" + std::to_string(unique));
    const auto output = root / "latest.jpg";
    std::error_code fs_error;
    std::filesystem::create_directories(root, fs_error);
    require(!fs_error, "temporary directory must be created");

    AnalysisSnapshotWriter writer;
    std::string error;
    require(writer.start(1, 2, error), "writer must start: " + error);
    std::atomic<int> completions{0};
    std::atomic<int> failures{0};
    constexpr int kJobs = 12;
    for (int index = 0; index < kJobs; ++index) {
        AnalysisSnapshotJob job;
        job.image = cv::Mat(
            360, 640, CV_8UC3,
            cv::Scalar(index * 7 % 255, index * 11 % 255, index * 13 % 255))
                .clone();
        job.output_path = output;
        job.jpeg_quality = 88;
        require(writer.enqueue(
            std::move(job),
            [&](const AnalysisSnapshotWriteResult& result) {
                ++completions;
                if (!result.success || result.encoded_bytes == 0) ++failures;
            },
            error), "writer must accept every bounded job: " + error);
    }
    require(writer.waitIdle(30000), "writer must drain every queued job");
    const auto metrics = writer.metrics();
    require(completions.load() == kJobs && failures.load() == 0,
        "every completion must succeed");
    require(metrics.submitted_jobs == kJobs &&
            metrics.completed_jobs == kJobs &&
            metrics.failed_jobs == 0 &&
            metrics.queue_depth == 0 &&
            metrics.active_jobs == 0 &&
            metrics.maximum_queue_depth <= 2,
        "writer metrics must prove bounded lossless completion");
    require(std::filesystem::is_regular_file(output, fs_error) &&
            !fs_error && std::filesystem::file_size(output, fs_error) > 0,
        "latest JPEG must be atomically published");
    writer.stop();
    std::filesystem::remove_all(root, fs_error);
    require(!fs_error, "temporary snapshot directory must be removed");
    std::cout << "PASS: bounded analysis snapshot writer completed "
              << kJobs << " jobs without loss\n";
    return 0;
}
