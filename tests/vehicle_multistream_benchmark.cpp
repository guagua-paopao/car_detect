#include "server/vehicle_tensorrt_adapters.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <nlohmann/json.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

using namespace yolo11_server;

namespace {

using Clock = std::chrono::steady_clock;

const std::vector<std::string> kVehicleClasses = {
    "car", "bus", "truck", "motorcycle", "vehicle", "other"};

ModelArtifactDescriptor detectionArtifact() {
    ModelArtifactDescriptor artifact;
    artifact.artifact_id = "vehicle-det-v1";
    artifact.role = VehicleModelRole::Detection;
    artifact.delivery_status = ModelDeliveryStatus::Deployed;
    artifact.labels_version = "vehicle-labels-v1";
    artifact.onnx_path = "models/vehicle-det-v1.onnx";
    artifact.engine_path = "engines/vehicle-det-v1.engine";
    artifact.onnx_sha256 =
        "a2655b27c9c1905f1ffbc3e16cb08e9800cca549d74cecc01d1b0bf47c504b93";
    artifact.engine_sha256 =
        "bdba13a99bea49c1bac448665fa102572bddf791a51c929b642020f586d53fb8";
    artifact.input_width = 960;
    artifact.input_height = 960;
    artifact.max_batch = 1;
    artifact.output_names = {"vehicle_detections"};
    return artifact;
}

struct I420Frame {
    std::vector<std::uint8_t> bytes;
    int width = 0;
    int height = 0;

    I420ImageView view() const {
        const std::size_t y_bytes = static_cast<std::size_t>(width) * height;
        const std::size_t chroma_bytes = y_bytes / 4U;
        return {
            bytes.data(),
            bytes.data() + y_bytes,
            bytes.data() + y_bytes + chroma_bytes,
            width,
            height,
            static_cast<std::size_t>(width),
            static_cast<std::size_t>(width / 2),
            static_cast<std::size_t>(width / 2),
        };
    }
};

I420Frame loadI420(const std::string& video) {
    cv::VideoCapture capture(video);
    cv::Mat bgr;
    if (!capture.isOpened() || !capture.read(bgr) || bgr.empty()) {
        throw std::runtime_error("could not decode multistream benchmark video");
    }
    if ((bgr.cols & 1) || (bgr.rows & 1)) {
        throw std::runtime_error("multistream benchmark requires even frame dimensions");
    }
    cv::Mat yuv;
    cv::cvtColor(bgr, yuv, cv::COLOR_BGR2YUV_I420);
    if (!yuv.isContinuous()) yuv = yuv.clone();
    I420Frame frame;
    frame.width = bgr.cols;
    frame.height = bgr.rows;
    frame.bytes.assign(yuv.data, yuv.data + yuv.total());
    return frame;
}

std::unique_ptr<TensorRtVehicleDetectionRunner> makeRunner(
    const std::string& project_root) {
    TensorRtDetectionOptions options;
    options.artifact_root = project_root;
    options.vehicle_classes = kVehicleClasses;
    auto runner = std::make_unique<TensorRtVehicleDetectionRunner>(
        std::move(options));
    std::string error;
    if (!runner->initialize(detectionArtifact(), error)) {
        throw std::runtime_error("detection runner initialization failed: " + error);
    }
    return runner;
}

nlohmann::json runMode(
    const std::string& project_root,
    const I420Frame& frame,
    int stream_count,
    int iterations,
    int warmup,
    int runs) {
    std::vector<std::unique_ptr<TensorRtVehicleDetectionRunner>> runners;
    for (int index = 0; index < stream_count; ++index) {
        runners.push_back(makeRunner(project_root));
    }
    for (auto& runner : runners) {
        for (int index = 0; index < warmup; ++index) {
            VehicleDetectionRequest request;
            request.i420_frame = frame.view();
            request.frame_sequence = index + 1;
            (void)runner->infer(request);
        }
        const auto timing = runner->lastTiming();
        if (!timing.cuda_graph_used || timing.cuda_graph_fallback) {
            throw std::runtime_error("CUDA Graph was not active before multistream measurement");
        }
    }

    nlohmann::json summaries = nlohmann::json::array();
    for (int run = 0; run < runs; ++run) {
        std::atomic<int> ready{0};
        std::atomic<bool> go{false};
        std::atomic<std::size_t> detections{0};
        std::atomic<int> failures{0};
        std::vector<std::thread> threads;
        threads.reserve(static_cast<std::size_t>(stream_count));
        for (int stream = 0; stream < stream_count; ++stream) {
            threads.emplace_back([&, stream]() {
                ++ready;
                while (!go.load(std::memory_order_acquire)) {
                    std::this_thread::yield();
                }
                try {
                    std::size_t local_detections = 0;
                    for (int index = 0; index < iterations; ++index) {
                        VehicleDetectionRequest request;
                        request.i420_frame = frame.view();
                        request.camera_id = "stream-" + std::to_string(stream);
                        request.run_id = "multistream";
                        request.frame_sequence =
                            static_cast<std::int64_t>(run) * iterations + index + 1;
                        local_detections += runners[stream]->infer(request).detections.size();
                    }
                    detections.fetch_add(local_detections);
                }
                catch (...) {
                    ++failures;
                }
            });
        }
        while (ready.load(std::memory_order_acquire) != stream_count) {
            std::this_thread::yield();
        }
        const auto started = Clock::now();
        go.store(true, std::memory_order_release);
        for (auto& thread : threads) thread.join();
        const double seconds =
            std::chrono::duration<double>(Clock::now() - started).count();
        if (failures.load() != 0) {
            throw std::runtime_error("multistream inference worker failed");
        }
        const std::size_t frames =
            static_cast<std::size_t>(stream_count) * iterations;
        summaries.push_back({
            {"run", run + 1},
            {"stream_count", stream_count},
            {"frames", frames},
            {"wall_seconds", seconds},
            {"aggregate_fps", static_cast<double>(frames) / seconds},
            {"detections", detections.load()},
            {"silent_drops", 0},
        });
    }
    return summaries;
}

double medianMetric(const nlohmann::json& runs, const char* key) {
    std::vector<double> values;
    for (const auto& run : runs) values.push_back(run.at(key).get<double>());
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3 || argc > 7) {
        std::cerr << "Usage: vehicle_multistream_benchmark <project-root> <video> "
                     "[iterations=2000] [warmup=200] [runs=3] [report.json]\n";
        return 2;
    }
    try {
        const std::string project_root = argv[1];
        const int iterations = argc >= 4 ? std::stoi(argv[3]) : 2000;
        const int warmup = argc >= 5 ? std::stoi(argv[4]) : 200;
        const int runs = argc >= 6 ? std::stoi(argv[5]) : 3;
        if (iterations <= 0 || warmup <= 0 || runs <= 0) {
            throw std::invalid_argument("benchmark counts must be positive");
        }
        const auto frame = loadI420(argv[2]);
        nlohmann::json single = nlohmann::json::array();
        nlohmann::json dual = nlohmann::json::array();
        for (int run = 0; run < runs; ++run) {
            if ((run & 1) == 0) {
                single.push_back(runMode(
                    project_root, frame, 1, iterations, warmup, 1).front());
                dual.push_back(runMode(
                    project_root, frame, 2, iterations, warmup, 1).front());
            }
            else {
                dual.push_back(runMode(
                    project_root, frame, 2, iterations, warmup, 1).front());
                single.push_back(runMode(
                    project_root, frame, 1, iterations, warmup, 1).front());
            }
            single.back()["run"] = run + 1;
            dual.back()["run"] = run + 1;
        }
        const double single_fps = medianMetric(single, "aggregate_fps");
        const double dual_fps = medianMetric(dual, "aggregate_fps");
        nlohmann::json report = {
            {"schema_version", "1.0"},
            {"source", argv[2]},
            {"buffering", "bounded per-runner persistent buffers; two independent CUDA streams"},
            {"iterations_per_stream_per_run", iterations},
            {"warmup_frames_per_stream", warmup},
            {"runs_requested", runs},
            {"run_order", "alternating single/dual; odd-numbered pairs start with single"},
            {"single_stream", single},
            {"dual_stream", dual},
            {"single_stream_median_fps", single_fps},
            {"dual_stream_median_fps", dual_fps},
            {"dual_stream_throughput_change_percent",
                single_fps > 0.0 ? (dual_fps / single_fps - 1.0) * 100.0 : 0.0},
        };
        const std::string serialized = report.dump(2);
        if (argc >= 7) {
            const std::filesystem::path path(argv[6]);
            if (!path.parent_path().empty()) {
                std::filesystem::create_directories(path.parent_path());
            }
            std::ofstream stream(path, std::ios::binary | std::ios::trunc);
            if (!stream) throw std::runtime_error("could not create report file");
            stream << serialized << '\n';
        }
        std::cout << serialized << '\n';
        return 0;
    }
    catch (const std::exception& exception) {
        std::cerr << "FAIL: " << exception.what() << '\n';
        return 1;
    }
}
