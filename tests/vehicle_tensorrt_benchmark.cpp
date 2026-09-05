#include "business/ffmpeg_process_capture_reader.h"
#include "business/vehicle_cascade_runtime.h"
#include "server/analysis_snapshot_writer.h"
#include "server/vehicle_tensorrt_adapters.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <nlohmann/json.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

using namespace yolo11_server;

namespace {

using Clock = std::chrono::steady_clock;

struct Sample {
    TensorRtStageTiming stage;
    TensorRtStageTiming attribute_stage;
    double source_wait_ms = 0.0;
    double wall_ms = 0.0;
    double input_prepare_ms = 0.0;
    double attribute_crop_ms = 0.0;
    double tracking_ms = 0.0;
    double result_output_ms = 0.0;
    double snapshot_ms = 0.0;
    double capture_to_result_ms = 0.0;
    std::size_t detections = 0;
    std::size_t attributes = 0;
};

const std::vector<std::string> kVehicleClasses = {
    "car", "bus", "truck", "motorcycle", "vehicle", "other"};
const std::vector<std::string> kBodyTypes = {
    "sedan", "suv", "mpv", "van", "pickup", "bus",
    "light_truck", "heavy_truck", "other", "unknown"};
const std::vector<std::string> kColors = {
    "black", "white", "silver_gray", "red", "blue", "green",
    "yellow_orange", "brown_beige", "other", "unknown"};

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

ModelArtifactDescriptor productionAttributeArtifact() {
    ModelArtifactDescriptor artifact;
    artifact.artifact_id = "vehicle-attr-best-components-256-r1";
    artifact.role = VehicleModelRole::Attributes;
    artifact.delivery_status = ModelDeliveryStatus::Deployed;
    artifact.labels_version = "vehicle-labels-v1";
    artifact.onnx_path =
        "models/candidates/vehicle-attr-best-components-256-r1/"
        "vehicle-attr-best-components-256-r1.onnx";
    artifact.engine_path =
        "models/candidates/vehicle-attr-best-components-256-r1/"
        "vehicle-attr-best-components-256-r1.engine";
    artifact.onnx_sha256 =
        "b470c013acff8cfbe9a7cf7ead8184f8a1516998507cfce4ba342e4ecf4ed86e";
    artifact.engine_sha256 =
        "eab83cc6e66133af5d61672e121465311ef914423dea54a2c32a6041ed98f5d6";
    artifact.input_width = 256;
    artifact.input_height = 256;
    artifact.max_batch = 16;
    artifact.output_names = {"body_type", "color"};
    return artifact;
}

ModelArtifactDescriptor attributeArtifact() {
    auto artifact = productionAttributeArtifact();
    const char* candidate_path = std::getenv("VCAS_ATTRIBUTE_ENGINE_PATH");
    const char* candidate_hash = std::getenv("VCAS_ATTRIBUTE_ENGINE_SHA256");
    if (candidate_path && candidate_hash) {
        artifact.engine_path = candidate_path;
        artifact.engine_sha256 = candidate_hash;
    }
    return artifact;
}

bool environmentFlag(const char* name) {
    const char* value = std::getenv(name);
    return value && std::string(value) != "0" && std::string(value) != "false";
}

double elapsedMs(Clock::time_point started) {
    return std::chrono::duration<double, std::milli>(Clock::now() - started).count();
}

long long unixTimeMs() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

double mean(const std::vector<double>& values) {
    return values.empty() ? 0.0 :
        std::accumulate(values.begin(), values.end(), 0.0) /
            static_cast<double>(values.size());
}

double percentile(std::vector<double> values, double quantile) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    const double position = quantile * static_cast<double>(values.size() - 1);
    const auto lower = static_cast<std::size_t>(std::floor(position));
    const auto upper = static_cast<std::size_t>(std::ceil(position));
    if (lower == upper) return values[lower];
    const double fraction = position - static_cast<double>(lower);
    return values[lower] * (1.0 - fraction) + values[upper] * fraction;
}

nlohmann::json summarize(const std::vector<double>& values) {
    return {
        {"mean", mean(values)},
        {"p50", percentile(values, 0.50)},
        {"p95", percentile(values, 0.95)},
        {"p99", percentile(values, 0.99)},
        {"min", values.empty() ? 0.0 : *std::min_element(values.begin(), values.end())},
        {"max", values.empty() ? 0.0 : *std::max_element(values.begin(), values.end())},
    };
}

nlohmann::json summarizeRun(
    const std::vector<Sample>& samples,
    double run_wall_seconds) {
    std::vector<double> wall;
    std::vector<double> source_wait;
    std::vector<double> preprocess;
    std::vector<double> host_staging;
    std::vector<double> gpu_preprocess;
    std::vector<double> allocation;
    std::vector<double> h2d;
    std::vector<double> inference;
    std::vector<double> d2h;
    std::vector<double> postprocess;
    std::vector<double> gpu_postprocess;
    std::vector<double> cuda_graph;
    std::vector<double> device_frame_copy;
    std::vector<double> input_prepare;
    std::vector<double> attribute_crop;
    std::vector<double> attribute_preprocess;
    std::vector<double> attribute_h2d;
    std::vector<double> attribute_inference;
    std::vector<double> attribute_postprocess;
    std::vector<double> attribute_cuda_graph;
    std::vector<double> attribute_total;
    std::vector<double> tracking;
    std::vector<double> result_output;
    std::vector<double> snapshot;
    std::vector<double> total;
    std::vector<double> capture_to_result;
    std::size_t detections = 0;
    std::size_t attributes = 0;
    std::size_t h2d_bytes = 0;
    std::size_t d2h_bytes = 0;
    std::size_t attribute_h2d_bytes = 0;
    std::size_t cuda_graph_frames = 0;
    std::size_t cuda_graph_fallback_frames = 0;
    std::size_t device_frame_retained_frames = 0;
    std::size_t device_frame_retain_fallback_frames = 0;
    std::size_t device_frame_copy_bytes = 0;
    std::array<std::size_t, 17> attribute_batch_histogram{};
    std::array<std::vector<double>, 17> attribute_batch_preprocess;
    std::array<std::vector<double>, 17> attribute_batch_inference;
    std::array<std::vector<double>, 17> attribute_batch_cuda_graph;
    std::array<std::vector<double>, 17> attribute_batch_total;
    std::array<std::size_t, 17> attribute_batch_graph_hits{};
    std::array<std::size_t, 17> attribute_batch_graph_builds{};
    std::array<std::size_t, 17> attribute_batch_graph_fallbacks{};
    std::size_t attribute_cuda_graph_frames = 0;
    std::size_t attribute_cuda_graph_builds = 0;
    std::size_t attribute_cuda_graph_fallback_frames = 0;
    for (const auto& sample : samples) {
        wall.push_back(sample.wall_ms);
        source_wait.push_back(sample.source_wait_ms);
        preprocess.push_back(sample.stage.preprocess_ms);
        host_staging.push_back(sample.stage.host_staging_ms);
        gpu_preprocess.push_back(sample.stage.gpu_preprocess_ms);
        allocation.push_back(sample.stage.allocation_ms);
        h2d.push_back(sample.stage.h2d_ms);
        inference.push_back(sample.stage.inference_ms);
        d2h.push_back(sample.stage.d2h_ms);
        postprocess.push_back(sample.stage.postprocess_ms);
        gpu_postprocess.push_back(sample.stage.gpu_postprocess_ms);
        cuda_graph.push_back(sample.stage.cuda_graph_ms);
        device_frame_copy.push_back(sample.stage.device_frame_copy_ms);
        input_prepare.push_back(sample.input_prepare_ms);
        attribute_crop.push_back(sample.attribute_crop_ms);
        attribute_preprocess.push_back(sample.attribute_stage.preprocess_ms);
        attribute_h2d.push_back(sample.attribute_stage.h2d_ms);
        attribute_inference.push_back(sample.attribute_stage.inference_ms);
        attribute_postprocess.push_back(sample.attribute_stage.postprocess_ms);
        attribute_cuda_graph.push_back(sample.attribute_stage.cuda_graph_ms);
        attribute_total.push_back(sample.attribute_stage.total_ms);
        tracking.push_back(sample.tracking_ms);
        result_output.push_back(sample.result_output_ms);
        snapshot.push_back(sample.snapshot_ms);
        total.push_back(sample.wall_ms);
        if (sample.capture_to_result_ms > 0.0) {
            capture_to_result.push_back(sample.capture_to_result_ms);
        }
        detections += sample.detections;
        attributes += sample.attributes;
        if (sample.attributes < attribute_batch_histogram.size()) {
            ++attribute_batch_histogram[sample.attributes];
            attribute_batch_preprocess[sample.attributes].push_back(
                sample.attribute_stage.preprocess_ms);
            attribute_batch_inference[sample.attributes].push_back(
                sample.attribute_stage.inference_ms);
            attribute_batch_cuda_graph[sample.attributes].push_back(
                sample.attribute_stage.cuda_graph_ms);
            attribute_batch_total[sample.attributes].push_back(
                sample.attribute_stage.total_ms);
            if (sample.attribute_stage.cuda_graph_used) {
                ++attribute_batch_graph_hits[sample.attributes];
            }
            if (sample.attribute_stage.cuda_graph_built) {
                ++attribute_batch_graph_builds[sample.attributes];
            }
            if (sample.attribute_stage.cuda_graph_fallback) {
                ++attribute_batch_graph_fallbacks[sample.attributes];
            }
        }
        if (sample.attribute_stage.cuda_graph_used) {
            ++attribute_cuda_graph_frames;
        }
        if (sample.attribute_stage.cuda_graph_built) {
            ++attribute_cuda_graph_builds;
        }
        if (sample.attribute_stage.cuda_graph_fallback) {
            ++attribute_cuda_graph_fallback_frames;
        }
        h2d_bytes = sample.stage.h2d_bytes;
        d2h_bytes = sample.stage.d2h_bytes;
        attribute_h2d_bytes = sample.attribute_stage.h2d_bytes;
        if (sample.stage.cuda_graph_used) ++cuda_graph_frames;
        if (sample.stage.cuda_graph_fallback) ++cuda_graph_fallback_frames;
        if (sample.stage.device_frame_retained) ++device_frame_retained_frames;
        if (sample.stage.device_frame_retain_fallback) {
            ++device_frame_retain_fallback_frames;
        }
        device_frame_copy_bytes = sample.stage.device_frame_copy_bytes;
    }
    nlohmann::json attribute_batch_counts = nlohmann::json::object();
    nlohmann::json attribute_batch_timing = nlohmann::json::object();
    for (std::size_t batch = 0; batch < attribute_batch_histogram.size(); ++batch) {
        attribute_batch_counts[std::to_string(batch)] =
            attribute_batch_histogram[batch];
        if (attribute_batch_histogram[batch] > 0) {
            attribute_batch_timing[std::to_string(batch)] = {
                {"count", attribute_batch_histogram[batch]},
                {"preprocess_ms", summarize(attribute_batch_preprocess[batch])},
                {"inference_ms", summarize(attribute_batch_inference[batch])},
                {"cuda_graph_ms", summarize(
                    attribute_batch_cuda_graph[batch])},
                {"total_ms", summarize(attribute_batch_total[batch])},
                {"cuda_graph_hits", attribute_batch_graph_hits[batch]},
                {"cuda_graph_builds", attribute_batch_graph_builds[batch]},
                {"cuda_graph_fallbacks",
                    attribute_batch_graph_fallbacks[batch]},
            };
        }
    }
    return {
        {"samples", samples.size()},
        {"run_wall_seconds", run_wall_seconds},
        {"sustained_fps", run_wall_seconds > 0.0 ?
            static_cast<double>(samples.size()) / run_wall_seconds : 0.0},
        {"wall_ms", summarize(wall)},
        {"rtsp_pull_decode_wait_ms", summarize(source_wait)},
        {"preprocess_ms", summarize(preprocess)},
        {"host_staging_ms", summarize(host_staging)},
        {"gpu_preprocess_ms", summarize(gpu_preprocess)},
        {"allocation_ms", summarize(allocation)},
        {"h2d_ms", summarize(h2d)},
        {"inference_ms", summarize(inference)},
        {"d2h_ms", summarize(d2h)},
        {"postprocess_ms", summarize(postprocess)},
        {"gpu_postprocess_ms", summarize(gpu_postprocess)},
        {"cuda_graph_ms", summarize(cuda_graph)},
        {"device_frame_copy_ms", summarize(device_frame_copy)},
        {"input_prepare_ms", summarize(input_prepare)},
        {"attribute_crop_ms", summarize(attribute_crop)},
        {"attribute_preprocess_ms", summarize(attribute_preprocess)},
        {"attribute_h2d_ms", summarize(attribute_h2d)},
        {"attribute_inference_ms", summarize(attribute_inference)},
        {"attribute_postprocess_ms", summarize(attribute_postprocess)},
        {"attribute_cuda_graph_ms", summarize(attribute_cuda_graph)},
        {"attribute_total_ms", summarize(attribute_total)},
        {"tracking_ms", summarize(tracking)},
        {"result_output_ms", summarize(result_output)},
        {"snapshot_ms", summarize(snapshot)},
        {"total_ms", summarize(total)},
        {"capture_to_result_ms", summarize(capture_to_result)},
        {"detections_total", detections},
        {"attributes_total", attributes},
        {"attribute_batch_histogram", std::move(attribute_batch_counts)},
        {"attribute_batch_timing", std::move(attribute_batch_timing)},
        {"h2d_bytes_per_frame", h2d_bytes},
        {"d2h_bytes_per_frame", d2h_bytes},
        {"attribute_h2d_bytes_per_batch", attribute_h2d_bytes},
        {"cuda_graph_frames", cuda_graph_frames},
        {"cuda_graph_fallback_frames", cuda_graph_fallback_frames},
        {"attribute_cuda_graph_frames", attribute_cuda_graph_frames},
        {"attribute_cuda_graph_builds", attribute_cuda_graph_builds},
        {"attribute_cuda_graph_fallback_frames",
            attribute_cuda_graph_fallback_frames},
        {"device_frame_retained_frames", device_frame_retained_frames},
        {"device_frame_retain_fallback_frames",
            device_frame_retain_fallback_frames},
        {"device_frame_copy_bytes_per_frame", device_frame_copy_bytes},
    };
}

int parsePositive(const char* text, const char* name, int maximum) {
    const int value = std::stoi(text);
    if (value <= 0 || value > maximum) {
        throw std::invalid_argument(
            std::string(name) + " must be in [1, " + std::to_string(maximum) + "]");
    }
    return value;
}

bool isRtsp(const std::string& source) {
    return source.rfind("rtsp://", 0) == 0 || source.rfind("rtsps://", 0) == 0;
}

void setDetectionInput(
    const FrameEnvelope& frame,
    VehicleDetectionRequest& request) {
    if (frame.i420.valid()) {
        const auto* bytes = frame.i420.bytes->data();
        request.i420_frame = {
            bytes + frame.i420.y_offset,
            bytes + frame.i420.u_offset,
            bytes + frame.i420.v_offset,
            frame.i420.width,
            frame.i420.height,
            frame.i420.y_stride_bytes,
            frame.i420.u_stride_bytes,
            frame.i420.v_stride_bytes,
        };
        return;
    }
    const auto& image = frame.bgrImage();
    request.frame = {
        image.data, image.cols, image.rows,
        image.channels(), image.step, ImagePixelFormat::Bgr8};
}

void setCompatibilityDetectionInput(
    const FrameEnvelope& frame, VehicleDetectionRequest& request) {
    const auto& image = frame.bgrImage();
    request.frame = {
        image.data, image.cols, image.rows,
        image.channels(), image.step, ImagePixelFormat::Bgr8};
}

std::vector<VehicleAttributeCrop> makeAttributeCrops(
    const FrameEnvelope& frame,
    const VehicleDetectionResult& detections,
    std::size_t maximum,
    bool use_i420_roi,
    const std::vector<std::int64_t>* track_ids = nullptr) {
    std::vector<VehicleAttributeCrop> crops;
    crops.reserve(std::min(maximum, detections.detections.size()));
    std::shared_ptr<OwnedI420Image> i420_frame;
    if (use_i420_roi && frame.i420.valid()) {
        i420_frame = std::make_shared<OwnedI420Image>();
        i420_frame->bytes = frame.i420.bytes;
        i420_frame->width = frame.i420.width;
        i420_frame->height = frame.i420.height;
        i420_frame->y_offset = frame.i420.y_offset;
        i420_frame->u_offset = frame.i420.u_offset;
        i420_frame->v_offset = frame.i420.v_offset;
        i420_frame->y_stride_bytes = frame.i420.y_stride_bytes;
        i420_frame->u_stride_bytes = frame.i420.u_stride_bytes;
        i420_frame->v_stride_bytes = frame.i420.v_stride_bytes;
    }
    const cv::Mat* image = nullptr;
    if (!i420_frame) image = &frame.bgrImage();
    for (std::size_t detection_index = 0;
         detection_index < detections.detections.size(); ++detection_index) {
        if (crops.size() >= maximum) break;
        const auto& detection = detections.detections[detection_index];
        VehicleAttributeCrop crop;
        if (i420_frame) {
            crop.i420_frame = i420_frame;
            if (detections.device_i420_frame &&
                detections.device_i420_frame->valid()) {
                crop.device_i420_frame = detections.device_i420_frame;
            }
            crop.source_box = detection.box;
        }
        else {
        const int left = std::clamp(
            static_cast<int>(std::floor(detection.box.x1 * image->cols)),
            0, image->cols - 1);
        const int top = std::clamp(
            static_cast<int>(std::floor(detection.box.y1 * image->rows)),
            0, image->rows - 1);
        const int right = std::clamp(
            static_cast<int>(std::ceil(detection.box.x2 * image->cols)),
            left + 1, image->cols);
        const int bottom = std::clamp(
            static_cast<int>(std::ceil(detection.box.y2 * image->rows)),
            top + 1, image->rows);
        auto owned = std::make_shared<OwnedImage>();
        owned->width = right - left;
        owned->height = bottom - top;
        owned->channels = 3;
        owned->row_stride_bytes = static_cast<std::size_t>(owned->width) * 3U;
        owned->pixel_format = ImagePixelFormat::Bgr8;
        owned->pixels.resize(
            owned->row_stride_bytes * static_cast<std::size_t>(owned->height));
        for (int row = 0; row < owned->height; ++row) {
            std::memcpy(
                owned->pixels.data() + static_cast<std::size_t>(row) *
                    owned->row_stride_bytes,
                image->ptr(top + row) + static_cast<std::size_t>(left) * 3U,
                owned->row_stride_bytes);
        }
        crop.crop = std::move(owned);
        }
        crop.camera_id = detections.camera_id;
        crop.run_id = detections.run_id;
        crop.run_generation = detections.metadata.run_generation;
        crop.track_id = track_ids && detection_index < track_ids->size()
            ? track_ids->at(detection_index)
            : static_cast<std::int64_t>(crops.size() + 1);
        crop.crop_sequence = static_cast<std::uint64_t>(detections.frame_sequence);
        crop.quality_score = detection.confidence;
        crops.push_back(std::move(crop));
    }
    return crops;
}

float boxIou(const VehicleBox& first, const VehicleBox& second) {
    const float left = std::max(first.x1, second.x1);
    const float top = std::max(first.y1, second.y1);
    const float right = std::min(first.x2, second.x2);
    const float bottom = std::min(first.y2, second.y2);
    const float intersection =
        std::max(0.0f, right - left) * std::max(0.0f, bottom - top);
    const float first_area =
        std::max(0.0f, first.x2 - first.x1) *
        std::max(0.0f, first.y2 - first.y1);
    const float second_area =
        std::max(0.0f, second.x2 - second.x1) *
        std::max(0.0f, second.y2 - second.y1);
    const float denominator = first_area + second_area - intersection;
    return denominator > 0.0f ? intersection / denominator : 0.0f;
}

class BusinessChain {
public:
    BusinessChain(std::string run_id, bool async_snapshots)
        : run_id_(std::move(run_id)),
          async_snapshots_(async_snapshots) {
        tracker_.startRun("benchmark", run_id_, 1);
        if (async_snapshots_) {
            snapshot_writer_ = std::make_unique<AnalysisSnapshotWriter>();
            std::string error;
            if (!snapshot_writer_->start(1, 2, error)) {
                throw std::runtime_error(
                    "could not start bounded snapshot writer: " + error);
            }
        }
    }

    ~BusinessChain() {
        if (snapshot_writer_) snapshot_writer_->stop();
    }

    VehicleTrackingUpdate track(
        const VehicleDetectionResult& detections,
        std::vector<std::int64_t>& track_ids,
        double& tracking_ms) {
        const auto started = Clock::now();
        VehicleTrackingUpdate update;
        std::string error;
        if (!tracker_.update(detections, update, error)) {
            throw std::runtime_error("business tracker failed: " + error);
        }
        track_ids.assign(detections.detections.size(), 0);
        std::vector<bool> used(update.active.size(), false);
        for (std::size_t detection_index = 0;
             detection_index < detections.detections.size(); ++detection_index) {
            const auto& detection = detections.detections[detection_index];
            std::size_t best = update.active.size();
            float best_iou = -1.0f;
            for (std::size_t track_index = 0;
                 track_index < update.active.size(); ++track_index) {
                if (used[track_index] ||
                    update.active[track_index].vehicle_class !=
                        detection.vehicle_class) {
                    continue;
                }
                const float candidate = boxIou(
                    detection.box, update.active[track_index].box);
                if (candidate > best_iou) {
                    best = track_index;
                    best_iou = candidate;
                }
            }
            if (best == update.active.size() || best_iou < 0.99f) {
                throw std::runtime_error(
                    "business detection-to-track association failed");
            }
            used[best] = true;
            track_ids[detection_index] = update.active[best].key.track_id;
        }
        tracking_ms = elapsedMs(started);
        return update;
    }

    void output(
        const FrameEnvelope& frame,
        const VehicleDetectionResult& detections,
        const VehicleTrackingUpdate& tracking,
        const std::vector<VehicleAttributeResult>& attributes,
        double& result_output_ms,
        double& snapshot_ms) {
        const auto output_started = Clock::now();
        std::string error;
        for (const auto& result : attributes) {
            if (!aggregator_.add(result, error)) {
                throw std::runtime_error(
                    "business attribute association failed: " + error);
            }
        }
        nlohmann::json items = nlohmann::json::array();
        for (const auto& track : tracking.active) {
            const auto attributes_for_track = aggregator_.snapshot(track.key);
            items.push_back({
                {"track_id", track.key.track_id},
                {"frame_sequence", track.last_frame_sequence},
                {"vehicle_class", track.vehicle_class},
                {"vehicle_class_confidence", track.detection_confidence},
                {"body_type", attributes_for_track
                    ? attributes_for_track->body_type.label : "unknown"},
                {"color", attributes_for_track
                    ? attributes_for_track->color.label : "unknown"},
            });
        }
        const nlohmann::json result = {
            {"success", true},
            {"mode", "vehicle_cascade_benchmark"},
            {"camera_id", detections.camera_id},
            {"run_id", detections.run_id},
            {"timestamp_ms", detections.captured_at_ms},
            {"frame_sequence", detections.frame_sequence},
            {"detection_count", detections.detections.size()},
            {"items", std::move(items)},
        };
        const auto serialized = result.dump();
        if (result["camera_id"] != "benchmark" ||
            result["run_id"] != run_id_ ||
            result["frame_sequence"] != detections.frame_sequence ||
            result["timestamp_ms"] != detections.captured_at_ms) {
            throw std::runtime_error("business result identity mismatch");
        }
        serialized_bytes_ += serialized.size();
        result_output_ms = elapsedMs(output_started);

        ++frames_;
        if ((frames_ - 1U) % kSnapshotIntervalFrames != 0U) return;
        const auto snapshot_started = Clock::now();
        const auto& image = frame.bgrImage();
        cv::Mat rendered = image.clone();
        for (const auto& track : tracking.active) {
            const int left = std::clamp(
                static_cast<int>(track.box.x1 * image.cols), 0,
                std::max(0, image.cols - 1));
            const int top = std::clamp(
                static_cast<int>(track.box.y1 * image.rows), 0,
                std::max(0, image.rows - 1));
            const int right = std::clamp(
                static_cast<int>(track.box.x2 * image.cols), left + 1,
                image.cols);
            const int bottom = std::clamp(
                static_cast<int>(track.box.y2 * image.rows), top + 1,
                image.rows);
            cv::rectangle(rendered,
                cv::Rect(left, top, right - left, bottom - top),
                cv::Scalar(40, 220, 40), 2, cv::LINE_AA);
        }
        if (snapshot_writer_) {
            AnalysisSnapshotJob job;
            job.image = std::move(rendered);
            job.jpeg_quality = 90;
            std::string enqueue_error;
            if (!snapshot_writer_->enqueue(
                    std::move(job),
                    [this](const AnalysisSnapshotWriteResult& result) {
                        if (!result.success) {
                            ++snapshot_failures_;
                            return;
                        }
                        snapshot_bytes_.fetch_add(result.encoded_bytes);
                        ++snapshots_;
                    },
                    enqueue_error)) {
                throw std::runtime_error(
                    "business snapshot enqueue failed: " + enqueue_error);
            }
        }
        else {
            std::vector<std::uint8_t> encoded;
            if (!cv::imencode(".jpg", rendered, encoded,
                    {cv::IMWRITE_JPEG_QUALITY, 90})) {
                throw std::runtime_error("business snapshot JPEG encode failed");
            }
            snapshot_bytes_.fetch_add(encoded.size());
            ++snapshots_;
        }
        snapshot_ms = elapsedMs(snapshot_started);
    }

    void finish() {
        if (!snapshot_writer_) return;
        if (!snapshot_writer_->waitIdle(60000)) {
            throw std::runtime_error("bounded snapshot writer did not drain");
        }
        snapshot_writer_->stop();
        if (snapshot_failures_.load() != 0) {
            throw std::runtime_error("bounded snapshot writer failed a job");
        }
    }

    nlohmann::json metrics() const {
        const auto& tracker = tracker_.metrics();
        const auto& aggregator = aggregator_.metrics();
        nlohmann::json writer = nullptr;
        if (snapshot_writer_) {
            const auto value = snapshot_writer_->metrics();
            writer = {
                {"queue_depth", value.queue_depth},
                {"active_jobs", value.active_jobs},
                {"maximum_queue_depth", value.maximum_queue_depth},
                {"submitted_jobs", value.submitted_jobs},
                {"completed_jobs", value.completed_jobs},
                {"failed_jobs", value.failed_jobs},
                {"mean_enqueue_wait_ms", value.mean_enqueue_wait_ms},
                {"maximum_enqueue_wait_ms", value.maximum_enqueue_wait_ms},
            };
        }
        return {
            {"frames", frames_},
            {"serialized_result_bytes", serialized_bytes_},
            {"snapshot_interval_frames", kSnapshotIntervalFrames},
            {"async_snapshots", async_snapshots_},
            {"snapshots_encoded", snapshots_.load()},
            {"snapshot_jpeg_bytes", snapshot_bytes_.load()},
            {"snapshot_failures", snapshot_failures_.load()},
            {"snapshot_writer", std::move(writer)},
            {"tracks_created", tracker.tracks_created},
            {"tracks_confirmed", tracker.tracks_confirmed},
            {"tracks_exited", tracker.tracks_exited},
            {"stale_results_rejected", tracker.stale_results_rejected},
            {"attribute_results_accepted", aggregator.accepted},
            {"attribute_stale_rejected", aggregator.stale_rejected},
            {"attribute_version_rejected", aggregator.version_rejected},
        };
    }

private:
    static constexpr std::uint64_t kSnapshotIntervalFrames = 12;
    std::string run_id_;
    bool async_snapshots_ = false;
    VehicleTracker tracker_;
    VehicleTrackAttributeAggregator aggregator_;
    std::unique_ptr<AnalysisSnapshotWriter> snapshot_writer_;
    std::uint64_t frames_ = 0;
    std::atomic<std::uint64_t> snapshots_{0};
    std::uint64_t serialized_bytes_ = 0;
    std::atomic<std::uint64_t> snapshot_bytes_{0};
    std::atomic<std::uint64_t> snapshot_failures_{0};
};

class FrameSource {
public:
    explicit FrameSource(const std::string& source)
        : source_(source), rtsp_(isRtsp(source)) {
        if (rtsp_) {
            CaptureSection config;
            config.backend = "ffmpeg";
            config.transport = "tcp";
            config.warmup_frames = 0;
            config.latest_frame_only = true;
            config.open_timeout_ms = 15000;
            config.read_timeout_ms = 5000;
            config.reconnect_max_attempts = 3;
            reader_ = std::make_unique<FfmpegProcessCaptureReader>(config);
            std::string error;
            if (!reader_->start(source, "rtsp://127.0.0.1/benchmark", "benchmark", error)) {
                throw std::runtime_error("RTSP reader start failed: " + error);
            }
        }
        else {
            file_capture_.open(source);
            if (!file_capture_.isOpened()) {
                throw std::runtime_error("could not decode benchmark video: " + source);
            }
            file_fps_ = file_capture_.get(cv::CAP_PROP_FPS);
            if (!std::isfinite(file_fps_) || file_fps_ <= 0.0) {
                file_fps_ = 25.0;
            }
            file_base_time_ms_ = unixTimeMs();
        }
    }

    ~FrameSource() {
        if (reader_) reader_->stop();
    }

    SharedCameraFrame next() {
        if (!rtsp_) {
            if (!file_capture_.read(file_frame_) || file_frame_.empty()) {
                ++file_loops_;
                file_capture_.set(cv::CAP_PROP_POS_FRAMES, 0.0);
                if (!file_capture_.read(file_frame_) || file_frame_.empty()) {
                    file_capture_.release();
                    file_capture_.open(source_);
                    if (!file_capture_.isOpened() ||
                        !file_capture_.read(file_frame_) || file_frame_.empty()) {
                        throw std::runtime_error(
                            "could not loop benchmark video: " + source_);
                    }
                }
            }
            auto frame = std::make_shared<FrameEnvelope>();
            frame->image = file_frame_;
            frame->sequence = ++file_sequence_;
            frame->capture_time_ms = file_base_time_ms_ +
                static_cast<long long>(std::llround(
                    static_cast<double>(file_sequence_ - 1U) *
                    1000.0 / file_fps_));
            frame->publish_time = Clock::now();
            ++consumed_frames_;
            return frame;
        }
        const auto deadline = Clock::now() + std::chrono::seconds(15);
        while (Clock::now() < deadline) {
            auto frame = reader_->getLatestFrameShared(last_sequence_);
            if (frame) {
                if (last_sequence_ != 0 && frame->sequence > last_sequence_ + 1) {
                    sequence_gaps_ += frame->sequence - last_sequence_ - 1;
                }
                last_sequence_ = frame->sequence;
                ++consumed_frames_;
                return frame;
            }
            const auto metrics = reader_->metrics();
            if (metrics.state == "failed") {
                throw std::runtime_error("RTSP reader failed: " + metrics.last_error);
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        throw std::runtime_error("timed out waiting for a new RTSP frame");
    }

    nlohmann::json metrics() const {
        if (!reader_) {
            return {
                {"state", "running"},
                {"backend", "opencv-video-file"},
                {"source_fps", file_fps_},
                {"synthetic_capture_timestamps", true},
                {"consumer_frames", consumed_frames_},
                {"file_loops", file_loops_},
                {"width", file_frame_.cols},
                {"height", file_frame_.rows},
                {"reconnect_count", 0},
            };
        }
        const auto value = reader_->metrics();
        return {
            {"state", value.state},
            {"backend", value.backend_name},
            {"capture_fps", value.capture_fps},
            {"source_fps", value.source_fps},
            {"dropped_frames", value.dropped_frames},
            {"consumer_sequence_gaps", sequence_gaps_},
            {"consumer_frames", consumed_frames_},
            {"reconnect_count", value.reconnect_count},
            {"open_count", value.open_count},
            {"width", value.width},
            {"height", value.height},
        };
    }

    bool rtsp() const noexcept { return rtsp_; }
    std::uint64_t sequenceGaps() const noexcept { return sequence_gaps_; }

private:
    std::string source_;
    bool rtsp_ = false;
    cv::VideoCapture file_capture_;
    cv::Mat file_frame_;
    double file_fps_ = 25.0;
    long long file_base_time_ms_ = 0;
    std::uint64_t file_sequence_ = 0;
    std::uint64_t file_loops_ = 0;
    std::uint64_t last_sequence_ = 0;
    std::uint64_t sequence_gaps_ = 0;
    std::uint64_t consumed_frames_ = 0;
    std::unique_ptr<FfmpegProcessCaptureReader> reader_;
};

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3 || argc > 7) {
        std::cerr << "Usage: vehicle_tensorrt_benchmark <project-root> <video-or-rtsp> "
                     "[iterations=2000] [warmup=200] [runs=3] [report.json]\n";
        return 2;
    }
    try {
        const std::string project_root = argv[1];
        const std::string source = argv[2];
        const int iterations = argc >= 4 ? parsePositive(argv[3], "iterations", 100000) : 2000;
        const int warmup = argc >= 5 ? parsePositive(argv[4], "warmup", 10000) : 200;
        const int runs = argc >= 6 ? parsePositive(argv[5], "runs", 20) : 3;
        const bool attributes_enabled = environmentFlag("VCAS_BENCHMARK_ATTRIBUTES");
        const bool compatibility_mode = environmentFlag("VCAS_BENCHMARK_COMPATIBILITY");
        const bool attribute_i420_roi = !compatibility_mode &&
            !environmentFlag("VCAS_DISABLE_ATTRIBUTE_I420_ROI");
        const bool attribute_device_i420 = attribute_i420_roi &&
            !environmentFlag("VCAS_DISABLE_ATTRIBUTE_DEVICE_I420");
        const bool compare_production_attribute =
            environmentFlag("VCAS_COMPARE_PRODUCTION_ATTRIBUTE");
        const bool attribute_cuda_graph = !compatibility_mode &&
            !environmentFlag("VCAS_DISABLE_ATTRIBUTE_CUDA_GRAPH");
        const bool business_chain_enabled =
            environmentFlag("VCAS_BENCHMARK_BUSINESS_CHAIN");
        const bool async_snapshots = business_chain_enabled &&
            environmentFlag("VCAS_BENCHMARK_ASYNC_SNAPSHOTS");

        TensorRtDetectionOptions options;
        options.artifact_root = project_root;
        options.vehicle_classes = kVehicleClasses;
        options.use_cuda_graph = !compatibility_mode;
        options.retain_i420_device_frame = attribute_device_i420;
        TensorRtVehicleDetectionRunner detector(std::move(options));
        std::string error;
        if (!detector.initialize(detectionArtifact(), error)) {
            throw std::runtime_error("detector initialization failed: " + error);
        }
        std::unique_ptr<TensorRtVehicleAttributeRunner> attribute_runner;
        std::unique_ptr<TensorRtVehicleAttributeRunner>
            production_attribute_reference;
        std::size_t attribute_parity_batches = 0;
        std::size_t attribute_parity_rois = 0;
        std::size_t attribute_parity_body_top1_mismatches = 0;
        std::size_t attribute_parity_color_top1_mismatches = 0;
        double attribute_parity_maximum_confidence_delta = 0.0;
        nlohmann::json attribute_parity_mismatch_examples =
            nlohmann::json::array();
        if (attributes_enabled) {
            TensorRtAttributeOptions attribute_options;
            attribute_options.artifact_root = project_root;
            attribute_options.body_types = kBodyTypes;
            attribute_options.colors = kColors;
            attribute_options.use_gpu_preprocess = !compatibility_mode;
            attribute_options.use_cuda_graph = attribute_cuda_graph;
            attribute_runner = std::make_unique<TensorRtVehicleAttributeRunner>(
                std::move(attribute_options));
            if (!attribute_runner->initialize(attributeArtifact(), error)) {
                throw std::runtime_error("attribute runner initialization failed: " + error);
            }
            if (compare_production_attribute) {
                TensorRtAttributeOptions reference_options;
                reference_options.artifact_root = project_root;
                reference_options.body_types = kBodyTypes;
                reference_options.colors = kColors;
                reference_options.use_gpu_preprocess = !compatibility_mode;
                reference_options.use_cuda_graph = false;
                production_attribute_reference =
                    std::make_unique<TensorRtVehicleAttributeRunner>(
                        std::move(reference_options));
                if (!production_attribute_reference->initialize(
                        productionAttributeArtifact(), error)) {
                    throw std::runtime_error(
                        "production attribute reference initialization failed: " +
                        error);
                }
            }
        }
        FrameSource frames(source);
        const auto infer_pipeline = [&](
            const SharedCameraFrame& frame,
            const std::string& run_id,
            BusinessChain* business_chain,
            Sample* sample) {
            const auto pipeline_started = Clock::now();
            VehicleDetectionRequest request;
            const auto input_prepare_started = Clock::now();
            if (compatibility_mode) {
                setCompatibilityDetectionInput(*frame, request);
            }
            else {
                setDetectionInput(*frame, request);
            }
            const double input_prepare_ms = elapsedMs(input_prepare_started);
            request.camera_id = "benchmark";
            request.run_id = run_id;
            request.run_generation = 1;
            request.frame_sequence = frame->sequence;
            request.captured_at_ms = frame->capture_time_ms;
            const auto result = detector.infer(request);
            double tracking_ms = 0.0;
            std::vector<std::int64_t> track_ids;
            VehicleTrackingUpdate tracking;
            if (business_chain) {
                tracking = business_chain->track(
                    result, track_ids, tracking_ms);
            }
            TensorRtStageTiming attribute_timing;
            double attribute_crop_ms = 0.0;
            std::size_t attribute_count = 0;
            std::vector<VehicleAttributeResult> attribute_results;
            if (attribute_runner) {
                const auto crop_started = Clock::now();
                auto crops = makeAttributeCrops(
                    *frame, result,
                    static_cast<std::size_t>(attributeArtifact().max_batch),
                    attribute_i420_roi,
                    business_chain ? &track_ids : nullptr);
                attribute_crop_ms = elapsedMs(crop_started);
                if (!crops.empty()) {
                    attribute_results = attribute_runner->inferBatch(crops);
                    attribute_count = attribute_results.size();
                    attribute_timing = attribute_runner->lastTiming();
                    if (production_attribute_reference) {
                        const auto reference =
                            production_attribute_reference->inferBatch(crops);
                        if (reference.size() != attribute_results.size()) {
                            throw std::runtime_error(
                                "attribute engine parity result count mismatch");
                        }
                        for (std::size_t index = 0;
                             index < attribute_results.size(); ++index) {
                            const bool body_mismatch =
                                reference[index].body_type.label !=
                                attribute_results[index].body_type.label;
                            const bool color_mismatch =
                                reference[index].color.label !=
                                attribute_results[index].color.label;
                            if (body_mismatch) {
                                ++attribute_parity_body_top1_mismatches;
                            }
                            if (color_mismatch) {
                                ++attribute_parity_color_top1_mismatches;
                            }
                            if ((body_mismatch || color_mismatch) &&
                                attribute_parity_mismatch_examples.size() < 20) {
                                attribute_parity_mismatch_examples.push_back({
                                    {"frame_sequence", frame->sequence},
                                    {"roi_index", index},
                                    {"body_reference",
                                        reference[index].body_type.label},
                                    {"body_candidate",
                                        attribute_results[index].body_type.label},
                                    {"body_reference_confidence",
                                        reference[index].body_type.confidence},
                                    {"body_candidate_confidence",
                                        attribute_results[index].body_type.confidence},
                                    {"color_reference",
                                        reference[index].color.label},
                                    {"color_candidate",
                                        attribute_results[index].color.label},
                                    {"color_reference_confidence",
                                        reference[index].color.confidence},
                                    {"color_candidate_confidence",
                                        attribute_results[index].color.confidence},
                                });
                            }
                            attribute_parity_maximum_confidence_delta = std::max(
                                attribute_parity_maximum_confidence_delta,
                                static_cast<double>(std::abs(
                                    reference[index].body_type.confidence -
                                    attribute_results[index].body_type.confidence)));
                            attribute_parity_maximum_confidence_delta = std::max(
                                attribute_parity_maximum_confidence_delta,
                                static_cast<double>(std::abs(
                                    reference[index].color.confidence -
                                    attribute_results[index].color.confidence)));
                        }
                        ++attribute_parity_batches;
                        attribute_parity_rois += attribute_results.size();
                    }
                }
            }
            double result_output_ms = 0.0;
            double snapshot_ms = 0.0;
            if (business_chain) {
                business_chain->output(
                    *frame, result, tracking, attribute_results,
                    result_output_ms, snapshot_ms);
            }
            if (sample) {
                sample->wall_ms = elapsedMs(pipeline_started);
                sample->input_prepare_ms = input_prepare_ms;
                sample->attribute_crop_ms = attribute_crop_ms;
                sample->tracking_ms = tracking_ms;
                sample->result_output_ms = result_output_ms;
                sample->snapshot_ms = snapshot_ms;
                sample->stage = detector.lastTiming();
                sample->attribute_stage = attribute_timing;
                sample->detections = result.detections.size();
                sample->attributes = attribute_count;
            }
        };
        std::unique_ptr<BusinessChain> warmup_business;
        if (business_chain_enabled) {
            warmup_business = std::make_unique<BusinessChain>(
                "benchmark-warmup", async_snapshots);
        }
        for (int index = 0; index < warmup; ++index) {
            const auto frame = frames.next();
            infer_pipeline(
                frame, "benchmark-warmup", warmup_business.get(), nullptr);
        }
        if (warmup_business) warmup_business->finish();

        nlohmann::json report = {
            {"schema_version", "1.0"},
            {"source", source},
            {"source_kind", frames.rtsp() ? "rtsp" : "file"},
            {"warmup_frames", warmup},
            {"iterations_per_run", iterations},
            {"runs_requested", runs},
            {"runs", nlohmann::json::array()},
            {"engine_path", detectionArtifact().engine_path},
            {"engine_sha256", detectionArtifact().engine_sha256},
            {"attributes_enabled", attributes_enabled},
            {"attribute_engine_path", attributes_enabled
                ? attributeArtifact().engine_path : ""},
            {"attribute_engine_sha256", attributes_enabled
                ? attributeArtifact().engine_sha256 : ""},
            {"compatibility_mode", compatibility_mode},
            {"attribute_cuda_graph", attribute_cuda_graph},
            {"business_chain_enabled", business_chain_enabled},
            {"async_snapshots", async_snapshots},
            {"attribute_input", attribute_i420_roi
                ? (attribute_device_i420
                    ? "device-i420-roi-batch" : "owned-i420-roi-batch")
                : "owned-bgr-crops"},
            {"attribute_engine_parity", {
                {"enabled", compare_production_attribute},
                {"reference_engine_path",
                    productionAttributeArtifact().engine_path},
                {"reference_engine_sha256",
                    productionAttributeArtifact().engine_sha256},
                {"confidence_tolerance", 5e-3},
            }},
        };
        std::vector<double> run_total_means;
        std::vector<double> run_p95;
        std::vector<double> run_fps;
        std::uint64_t measured_sequence_gaps = 0;
        for (int run = 0; run < runs; ++run) {
            const std::string run_id =
                "benchmark-" + std::to_string(run + 1);
            std::unique_ptr<BusinessChain> business_chain;
            if (business_chain_enabled) {
                business_chain = std::make_unique<BusinessChain>(
                    run_id, async_snapshots);
            }
            std::vector<Sample> samples;
            samples.reserve(static_cast<std::size_t>(iterations));
            const auto sequence_gaps_before_run = frames.sequenceGaps();
            const auto run_started = Clock::now();
            for (int index = 0; index < iterations; ++index) {
                const auto source_wait_started = Clock::now();
                const auto frame = frames.next();
                const double source_wait_ms = elapsedMs(source_wait_started);
                Sample sample;
                sample.source_wait_ms = source_wait_ms;
                infer_pipeline(
                    frame, run_id, business_chain.get(), &sample);
                sample.capture_to_result_ms = frames.rtsp()
                    ? static_cast<double>(unixTimeMs() - frame->capture_time_ms) : 0.0;
                samples.push_back(sample);
            }
            if (business_chain) business_chain->finish();
            const double run_seconds =
                std::chrono::duration<double>(Clock::now() - run_started).count();
            auto summary = summarizeRun(samples, run_seconds);
            summary["run"] = run + 1;
            const auto run_sequence_gaps =
                frames.sequenceGaps() - sequence_gaps_before_run;
            summary["consumer_sequence_gaps"] = run_sequence_gaps;
            if (business_chain) {
                summary["business_chain"] = business_chain->metrics();
            }
            measured_sequence_gaps += run_sequence_gaps;
            run_total_means.push_back(summary["total_ms"]["mean"].get<double>());
            run_p95.push_back(summary["total_ms"]["p95"].get<double>());
            run_fps.push_back(summary["sustained_fps"].get<double>());
            report["runs"].push_back(std::move(summary));
        }
        report["capture"] = frames.metrics();
        report["median_run_total_mean_ms"] = percentile(run_total_means, 0.50);
        report["median_run_total_p95_ms"] = percentile(run_p95, 0.50);
        report["median_sustained_fps"] = percentile(run_fps, 0.50);
        report["measured_consumer_sequence_gaps"] = measured_sequence_gaps;
        report["completed_at_unix_ms"] = unixTimeMs();
        report["attribute_engine_parity"]["batches"] =
            attribute_parity_batches;
        report["attribute_engine_parity"]["rois"] =
            attribute_parity_rois;
        report["attribute_engine_parity"]["maximum_confidence_delta"] =
            attribute_parity_maximum_confidence_delta;
        report["attribute_engine_parity"]["body_top1_mismatches"] =
            attribute_parity_body_top1_mismatches;
        report["attribute_engine_parity"]["color_top1_mismatches"] =
            attribute_parity_color_top1_mismatches;
        report["attribute_engine_parity"]["mismatch_examples"] =
            std::move(attribute_parity_mismatch_examples);
        const bool attribute_parity_top1_match =
            attribute_parity_body_top1_mismatches == 0 &&
            attribute_parity_color_top1_mismatches == 0;
        const bool attribute_parity_passed =
            !compare_production_attribute ||
            (attribute_parity_rois >= 200 &&
             attribute_parity_top1_match &&
             attribute_parity_maximum_confidence_delta <= 5e-3);
        report["attribute_engine_parity"]["top1_match"] =
            attribute_parity_top1_match;
        report["attribute_engine_parity"]["passed"] =
            attribute_parity_passed;

        const std::string serialized = report.dump(2);
        if (argc >= 7) {
            const std::filesystem::path report_path(argv[6]);
            if (!report_path.parent_path().empty()) {
                std::filesystem::create_directories(report_path.parent_path());
            }
            std::ofstream stream(report_path, std::ios::binary | std::ios::trunc);
            if (!stream) throw std::runtime_error("could not create report file");
            stream << serialized << '\n';
        }
        std::cout << serialized << '\n';
        if (!attribute_parity_passed) {
            std::cerr << "FAIL: attribute engine real-ROI parity gate failed\n";
            return 1;
        }
        return 0;
    }
    catch (const std::exception& exception) {
        std::cerr << "FAIL: " << exception.what() << '\n';
        return 1;
    }
}
