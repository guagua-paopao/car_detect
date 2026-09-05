#include "server/vehicle_tensorrt_adapters.h"

#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

using namespace yolo11_server;

namespace {

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
    const char* candidate_path = std::getenv("VCAS_DETECTION_ENGINE_PATH");
    const char* candidate_hash = std::getenv("VCAS_DETECTION_ENGINE_SHA256");
    if (candidate_path && candidate_hash) {
        artifact.engine_path = candidate_path;
        artifact.engine_sha256 = candidate_hash;
    }
    artifact.input_width = 960;
    artifact.input_height = 960;
    artifact.max_batch = 1;
    artifact.output_names = {"vehicle_detections"};
    return artifact;
}

float iou(const VehicleBox& first, const VehicleBox& second) {
    const float left = std::max(first.x1, second.x1);
    const float top = std::max(first.y1, second.y1);
    const float right = std::min(first.x2, second.x2);
    const float bottom = std::min(first.y2, second.y2);
    const float intersection =
        std::max(0.0f, right - left) * std::max(0.0f, bottom - top);
    const float first_area = (first.x2 - first.x1) * (first.y2 - first.y1);
    const float second_area = (second.x2 - second.x1) * (second.y2 - second.y1);
    const float denominator = first_area + second_area - intersection;
    return denominator > 0.0f ? intersection / denominator : 0.0f;
}

VehicleDetectionRequest requestFor(const cv::Mat& frame, std::int64_t sequence) {
    VehicleDetectionRequest request;
    request.frame = {
        frame.data, frame.cols, frame.rows, frame.channels(), frame.step,
        ImagePixelFormat::Bgr8};
    request.camera_id = "gpu-parity";
    request.run_id = "gpu-parity";
    request.run_generation = 1;
    request.frame_sequence = sequence;
    return request;
}

VehicleDetectionRequest requestForI420(
    const cv::Mat& packed,
    int width,
    int height,
    std::int64_t sequence) {
    const std::size_t y_bytes = static_cast<std::size_t>(width) * height;
    const std::size_t chroma_bytes = y_bytes / 4U;
    VehicleDetectionRequest request;
    request.i420_frame = {
        packed.data,
        packed.data + y_bytes,
        packed.data + y_bytes + chroma_bytes,
        width,
        height,
        static_cast<std::size_t>(width),
        static_cast<std::size_t>(width / 2),
        static_cast<std::size_t>(width / 2),
    };
    request.camera_id = "gpu-parity";
    request.run_id = "gpu-parity";
    request.run_generation = 1;
    request.frame_sequence = sequence;
    return request;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: vehicle_gpu_pipeline_parity_test <project-root> <video>\n";
        return 2;
    }
    try {
        TensorRtDetectionOptions cpu_options;
        cpu_options.artifact_root = argv[1];
        cpu_options.vehicle_classes = kVehicleClasses;
        cpu_options.use_gpu_preprocess = false;
        cpu_options.use_gpu_postprocess = false;
        TensorRtDetectionOptions gpu_options = cpu_options;
        gpu_options.use_gpu_preprocess = true;
        gpu_options.use_gpu_postprocess = true;
        TensorRtVehicleDetectionRunner cpu(std::move(cpu_options));
        TensorRtVehicleDetectionRunner gpu(std::move(gpu_options));
        std::string error;
        if (!cpu.initialize(detectionArtifact(), error)) {
            throw std::runtime_error("CPU-reference runner initialization failed: " + error);
        }
        if (!gpu.initialize(detectionArtifact(), error)) {
            throw std::runtime_error("GPU runner initialization failed: " + error);
        }

        cv::VideoCapture capture(argv[2]);
        if (!capture.isOpened()) throw std::runtime_error("could not open parity video");
        constexpr int kFrames = 200;
        constexpr int kStride = 7;
        std::size_t matched_detections = 0;
        float minimum_iou = 1.0f;
        float maximum_score_difference = 0.0f;
        float maximum_coordinate_delta_pixels = 0.0f;
        for (int frame_index = 0; frame_index < kFrames; ++frame_index) {
            cv::Mat frame;
            for (int skip = 0; skip < kStride; ++skip) {
                if (!capture.read(frame) || frame.empty()) {
                    capture.set(cv::CAP_PROP_POS_FRAMES, 0.0);
                    if (!capture.read(frame) || frame.empty()) {
                        throw std::runtime_error("could not decode parity frame");
                    }
                }
            }
            cv::Mat i420;
            cv::cvtColor(frame, i420, cv::COLOR_BGR2YUV_I420);
            cv::Mat reference_bgr;
            cv::cvtColor(i420, reference_bgr, cv::COLOR_YUV2BGR_I420);
            const auto expected = cpu.infer(requestFor(reference_bgr, frame_index + 1));
            const auto actual = gpu.infer(requestForI420(
                i420, frame.cols, frame.rows, frame_index + 1));
            if (expected.detections.size() != actual.detections.size()) {
                throw std::runtime_error(
                    "detection count mismatch at frame " + std::to_string(frame_index));
            }
            std::vector<bool> used(actual.detections.size(), false);
            for (const auto& reference : expected.detections) {
                std::size_t best = actual.detections.size();
                float best_iou = -1.0f;
                for (std::size_t index = 0; index < actual.detections.size(); ++index) {
                    if (used[index] || actual.detections[index].vehicle_class !=
                            reference.vehicle_class) continue;
                    const float value = iou(reference.box, actual.detections[index].box);
                    if (value > best_iou) {
                        best_iou = value;
                        best = index;
                    }
                }
                float coordinate_delta_pixels = std::numeric_limits<float>::infinity();
                if (best < actual.detections.size()) {
                    const auto& candidate = actual.detections[best].box;
                    coordinate_delta_pixels = std::max({
                        std::abs(reference.box.x1 - candidate.x1) * frame.cols,
                        std::abs(reference.box.x2 - candidate.x2) * frame.cols,
                        std::abs(reference.box.y1 - candidate.y1) * frame.rows,
                        std::abs(reference.box.y2 - candidate.y2) * frame.rows,
                    });
                }
                if (best == actual.detections.size() ||
                    (best_iou < 0.99f && coordinate_delta_pixels > 2.0f)) {
                    std::ostringstream detail;
                    detail << "box parity failed at frame " << frame_index
                           << " class=" << reference.vehicle_class
                           << " score=" << reference.confidence
                           << " reference=[" << reference.box.x1 << ','
                           << reference.box.y1 << ',' << reference.box.x2 << ','
                           << reference.box.y2 << "] best_iou=" << best_iou;
                    if (best < actual.detections.size()) {
                        detail << " actual_score=" << actual.detections[best].confidence
                               << " actual=[" << actual.detections[best].box.x1 << ','
                               << actual.detections[best].box.y1 << ','
                               << actual.detections[best].box.x2 << ','
                               << actual.detections[best].box.y2 << ']';
                    }
                    throw std::runtime_error(detail.str());
                }
                used[best] = true;
                minimum_iou = std::min(minimum_iou, best_iou);
                maximum_coordinate_delta_pixels = std::max(
                    maximum_coordinate_delta_pixels, coordinate_delta_pixels);
                maximum_score_difference = std::max(
                    maximum_score_difference,
                    std::abs(reference.confidence - actual.detections[best].confidence));
                ++matched_detections;
            }
        }
        const auto cpu_timing = cpu.lastTiming();
        const auto gpu_timing = gpu.lastTiming();
        if (gpu_timing.h2d_bytes >= cpu_timing.h2d_bytes ||
            (!gpu_timing.cuda_graph_used && gpu_timing.gpu_preprocess_ms <= 0.0) ||
            (gpu_timing.cuda_graph_used && gpu_timing.cuda_graph_ms <= 0.0)) {
            throw std::runtime_error("GPU preprocessing telemetry invariant failed");
        }
        if (!gpu_timing.cuda_graph_used || gpu_timing.cuda_graph_fallback) {
            throw std::runtime_error("CUDA Graph was not active on the steady-state parity frame");
        }
        if (maximum_score_difference > 1e-5f) {
            throw std::runtime_error("score parity exceeded 1e-5");
        }
        std::cout << "PASS: CUDA preprocessing and postprocessing matched CPU reference across "
                  << kFrames << " frames; detections=" << matched_detections
                  << " min_iou=" << minimum_iou
                  << " max_coordinate_delta_px=" << maximum_coordinate_delta_pixels
                  << " max_score_diff=" << maximum_score_difference
                  << " cpu_h2d_bytes=" << cpu_timing.h2d_bytes
                  << " gpu_h2d_bytes=" << gpu_timing.h2d_bytes
                  << " cuda_graph_used=" << std::boolalpha
                  << gpu_timing.cuda_graph_used << '\n';
        return 0;
    }
    catch (const std::exception& exception) {
        std::cerr << "FAIL: " << exception.what() << '\n';
        return 1;
    }
}
