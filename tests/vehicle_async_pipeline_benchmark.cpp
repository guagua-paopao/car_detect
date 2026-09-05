#include "business/ffmpeg_process_capture_reader.h"
#include "server/camera_inference_pool.h"
#include "server/vehicle_attribute_batch_scheduler.h"
#include "server/vehicle_tensorrt_adapters.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>
#include <opencv2/core.hpp>

using namespace yolo11_server;

namespace {

using Clock = std::chrono::steady_clock;

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
    artifact.input_width = 960;
    artifact.input_height = 960;
    artifact.max_batch = 1;
    artifact.output_names = {"vehicle_detections"};
    return artifact;
}

ModelArtifactDescriptor attributeArtifact() {
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

double elapsedMs(Clock::time_point started) {
    return std::chrono::duration<double, std::milli>(
        Clock::now() - started).count();
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
        {"min", values.empty() ? 0.0 :
            *std::min_element(values.begin(), values.end())},
        {"max", values.empty() ? 0.0 :
            *std::max_element(values.begin(), values.end())},
    };
}

class BenchmarkVehicleRunner final : public IModelRunner {
public:
    explicit BenchmarkVehicleRunner(std::string project_root)
        : project_root_(std::move(project_root)) {}

    std::string modelType() const override { return "vehicle_detection"; }

    bool init(const AppConfig& config, std::string& error) override {
        TensorRtDetectionOptions options;
        options.gpu_id = config.model.gpu_id;
        options.artifact_root = project_root_;
        options.vehicle_classes = kVehicleClasses;
        options.use_gpu_preprocess = true;
        options.use_gpu_postprocess = true;
        options.use_cuda_graph = true;
        options.retain_i420_device_frame = true;
        detector_ = std::make_unique<TensorRtVehicleDetectionRunner>(
            std::move(options));
        if (!detector_->initialize(detectionArtifact(), error)) {
            detector_.reset();
            return false;
        }
        return true;
    }

    ModelOutput infer(const cv::Mat& image) override {
        if (image.empty() || image.type() != CV_8UC3) {
            throw std::invalid_argument("benchmark runner requires CV_8UC3");
        }
        VehicleDetectionRequest request;
        request.frame = {
            image.data, image.cols, image.rows, image.channels(), image.step,
            ImagePixelFormat::Bgr8};
        return inferRequest(request);
    }

    ModelOutput infer(const FrameEnvelope& frame) override {
        if (!frame.i420.valid()) return infer(frame.bgrImage());
        const auto* bytes = frame.i420.bytes->data();
        VehicleDetectionRequest request;
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
        return inferRequest(request);
    }

    cv::Mat draw(const cv::Mat& image, const ModelOutput&) override {
        return image.clone();
    }

    void release() noexcept override {
        if (detector_) detector_->release();
        detector_.reset();
    }

private:
    ModelOutput inferRequest(const VehicleDetectionRequest& request) {
        if (!detector_) throw std::runtime_error("runner is not initialized");
        ModelOutput output;
        output.model_type = "vehicle_detection";
        output.has_vehicle_detection = true;
        output.vehicle_detection = detector_->infer(request);
        return output;
    }

    std::string project_root_;
    std::unique_ptr<TensorRtVehicleDetectionRunner> detector_;
};

struct AttributeBatchSample {
    std::size_t batch_size = 0;
    double wall_ms = 0.0;
    TensorRtStageTiming stage;
};

struct AttributeTimingState {
    std::mutex mutex;
    std::vector<AttributeBatchSample> samples;
};

class MeasuringAttributeRunner final : public IVehicleAttributeRunner {
public:
    MeasuringAttributeRunner(
        std::string project_root,
        std::shared_ptr<AttributeTimingState> timing)
        : project_root_(std::move(project_root)), timing_(std::move(timing)) {}

    bool initialize(
        const ModelArtifactDescriptor& artifact,
        std::string& error) override {
        TensorRtAttributeOptions options;
        options.artifact_root = project_root_;
        options.body_types = kBodyTypes;
        options.colors = kColors;
        options.use_gpu_preprocess = true;
        options.use_cuda_graph = true;
        runner_ = std::make_unique<TensorRtVehicleAttributeRunner>(
            std::move(options));
        return runner_->initialize(artifact, error);
    }

    std::vector<VehicleAttributeResult> inferBatch(
        const std::vector<VehicleAttributeCrop>& crops) override {
        const auto started = Clock::now();
        auto result = runner_->inferBatch(crops);
        AttributeBatchSample sample;
        sample.batch_size = crops.size();
        sample.wall_ms = elapsedMs(started);
        sample.stage = runner_->lastTiming();
        {
            std::lock_guard<std::mutex> lock(timing_->mutex);
            timing_->samples.push_back(sample);
        }
        return result;
    }

    void release() noexcept override {
        if (runner_) runner_->release();
        runner_.reset();
    }

private:
    std::string project_root_;
    std::shared_ptr<AttributeTimingState> timing_;
    std::unique_ptr<TensorRtVehicleAttributeRunner> runner_;
};

std::vector<VehicleAttributeCrop> makeAttributeCrops(
    const CameraInferenceResult& inference) {
    const auto& frame = *inference.job.frame;
    auto detections = inference.output.vehicle_detection;
    detections.camera_id = inference.job.task_id;
    detections.run_id = inference.job.run_id;
    detections.frame_sequence = static_cast<std::int64_t>(
        inference.job.source_sequence);
    detections.captured_at_ms = inference.job.capture_time_ms;
    detections.metadata.run_generation = 1;

    std::shared_ptr<OwnedI420Image> i420_frame;
    if (frame.i420.valid()) {
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

    const std::size_t maximum = static_cast<std::size_t>(
        attributeArtifact().max_batch);
    std::vector<VehicleAttributeCrop> crops;
    crops.reserve(std::min(maximum, detections.detections.size()));
    for (const auto& detection : detections.detections) {
        if (crops.size() == maximum) break;
        VehicleAttributeCrop crop;
        crop.i420_frame = i420_frame;
        crop.device_i420_frame = detections.device_i420_frame;
        crop.source_box = detection.box;
        crop.camera_id = detections.camera_id;
        crop.run_id = detections.run_id;
        crop.run_generation = 1;
        crop.track_id = static_cast<std::int64_t>(crops.size() + 1);
        crop.crop_sequence = inference.job.source_sequence;
        crop.quality_score = detection.confidence;
        crops.push_back(std::move(crop));
    }
    return crops;
}

struct FrameSample {
    double inference_ms = 0.0;
    double result_handler_ms = 0.0;
    double submit_to_complete_ms = 0.0;
    double capture_to_complete_ms = 0.0;
    std::size_t detections = 0;
    std::size_t attributes = 0;
};

enum class AttributeExecutionMode {
    SharedScheduler,
    SchedulerPool,
    ContextPool,
    PerRoute,
};

const char* attributeExecutionName(AttributeExecutionMode mode) {
    switch (mode) {
        case AttributeExecutionMode::SharedScheduler:
            return "shared_scheduler_single_context";
        case AttributeExecutionMode::SchedulerPool:
            return "sharded_scheduler_pool_2";
        case AttributeExecutionMode::ContextPool:
            return "bounded_context_pool_2";
        case AttributeExecutionMode::PerRoute:
            return "independent_context_per_route";
    }
    return "unknown";
}

class BenchmarkResultHandler final : public ICameraInferenceResultHandler {
public:
    BenchmarkResultHandler(
        std::string project_root,
        AttributeExecutionMode attribute_mode,
        int route_count)
        : project_root_(std::move(project_root)),
          attribute_mode_(attribute_mode),
          route_count_(route_count),
          attribute_timing_(std::make_shared<AttributeTimingState>()) {}

    bool start(std::string& error) {
        if (attribute_mode_ == AttributeExecutionMode::SchedulerPool) {
            for (int index = 0; index < 2; ++index) {
                auto runner = std::make_unique<MeasuringAttributeRunner>(
                    project_root_, attribute_timing_);
                if (!runner->initialize(attributeArtifact(), error)) {
                    stop();
                    return false;
                }
                VehicleAttributeBatchSchedulerConfig config;
                config.max_batch = static_cast<std::size_t>(
                    attributeArtifact().max_batch);
                config.max_pending_requests = 32;
                config.max_wait = std::chrono::milliseconds(2);
                auto scheduler =
                    std::make_unique<VehicleAttributeBatchScheduler>(
                        config, std::move(runner));
                if (!scheduler->start(error)) {
                    stop();
                    return false;
                }
                scheduler_pool_.push_back(std::move(scheduler));
            }
            return true;
        }
        if (attribute_mode_ != AttributeExecutionMode::SharedScheduler) {
            const int runner_count = attribute_mode_ ==
                    AttributeExecutionMode::ContextPool
                ? 2 : route_count_;
            for (int index = 0; index < runner_count; ++index) {
                auto slot = std::make_unique<DirectRunnerSlot>();
                slot->runner = std::make_unique<MeasuringAttributeRunner>(
                    project_root_, attribute_timing_);
                if (!slot->runner->initialize(attributeArtifact(), error)) {
                    stop();
                    return false;
                }
                direct_runners_.push_back(std::move(slot));
            }
            return true;
        }
        auto runner = std::make_unique<MeasuringAttributeRunner>(
            project_root_, attribute_timing_);
        if (!runner->initialize(attributeArtifact(), error)) return false;
        VehicleAttributeBatchSchedulerConfig config;
        config.max_batch = static_cast<std::size_t>(
            attributeArtifact().max_batch);
        config.max_pending_requests = 64;
        config.max_wait = std::chrono::milliseconds(2);
        scheduler_ = std::make_unique<VehicleAttributeBatchScheduler>(
            config, std::move(runner));
        return scheduler_->start(error);
    }

    void stop() noexcept {
        if (scheduler_) scheduler_->stop();
        scheduler_.reset();
        for (auto& scheduler : scheduler_pool_) scheduler->stop();
        scheduler_pool_.clear();
        for (auto& slot : direct_runners_) {
            if (slot && slot->runner) slot->runner->release();
        }
        direct_runners_.clear();
    }

    void noteSubmission(
        const std::string& camera,
        std::uint64_t sequence,
        Clock::time_point submitted) {
        std::lock_guard<std::mutex> lock(mutex_);
        submissions_[{camera, sequence}] = submitted;
    }

    bool handle(
        const CameraInferenceResult& inference,
        std::string& error) override {
        error.clear();
        const auto started = Clock::now();
        auto crops = makeAttributeCrops(inference);
        std::vector<VehicleAttributeResult> attributes;
        if (!crops.empty()) {
            if (scheduler_) {
                if (!scheduler_->infer(std::move(crops), attributes, error)) {
                    complete(inference, started, attributes.size(), error);
                    return false;
                }
            }
            else if (!scheduler_pool_.empty()) {
                const std::size_t index = shardedIndex(
                    inference.job.task_id, scheduler_pool_.size());
                if (!scheduler_pool_[index]->infer(
                        std::move(crops), attributes, error)) {
                    complete(inference, started, attributes.size(), error);
                    return false;
                }
            }
            else {
                const std::size_t index = directRunnerIndex(
                    inference.job.task_id);
                auto& slot = *direct_runners_.at(index);
                try {
                    std::lock_guard<std::mutex> lock(slot.mutex);
                    attributes = slot.runner->inferBatch(crops);
                }
                catch (const std::exception& exception) {
                    error = exception.what();
                    complete(inference, started, attributes.size(), error);
                    return false;
                }
            }
        }
        complete(inference, started, attributes.size(), {});
        return true;
    }

    void detachCamera(const std::string&, const std::string&) noexcept override {}

    bool waitFor(
        const std::string& camera,
        std::uint64_t sequence,
        std::chrono::seconds timeout,
        std::string& error) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (!completed_.wait_for(lock, timeout, [&]() {
                const auto found = completed_sequences_.find(camera);
                return found != completed_sequences_.end() &&
                    found->second >= sequence;
            })) {
            error = "timed out waiting for result " + camera + "/" +
                std::to_string(sequence);
            return false;
        }
        if (!failure_.empty()) {
            error = failure_;
            return false;
        }
        return true;
    }

    void clearMeasurementSamples() {
        std::lock_guard<std::mutex> lock(mutex_);
        frame_samples_.clear();
        failure_.clear();
        std::lock_guard<std::mutex> timing_lock(attribute_timing_->mutex);
        attribute_timing_->samples.clear();
    }

    VehicleAttributeBatchSchedulerSnapshot schedulerSnapshot() const {
        if (scheduler_) return scheduler_->snapshot();
        VehicleAttributeBatchSchedulerSnapshot combined;
        for (const auto& scheduler : scheduler_pool_) {
            const auto value = scheduler->snapshot();
            const double previous_wait_sum = combined.mean_queue_wait_ms *
                static_cast<double>(combined.completed_requests);
            const double value_wait_sum = value.mean_queue_wait_ms *
                static_cast<double>(value.completed_requests);
            combined.running = combined.running || value.running;
            combined.requests += value.requests;
            combined.completed_requests += value.completed_requests;
            combined.failed_requests += value.failed_requests;
            combined.batches += value.batches;
            combined.crops += value.crops;
            combined.pending_requests += value.pending_requests;
            combined.maximum_pending_requests += value.maximum_pending_requests;
            combined.pending_crops += value.pending_crops;
            combined.maximum_pending_crops += value.maximum_pending_crops;
            combined.mean_queue_wait_ms = combined.completed_requests == 0 ?
                0.0 : (previous_wait_sum + value_wait_sum) /
                    static_cast<double>(combined.completed_requests);
            combined.p95_queue_wait_ms = std::max(
                combined.p95_queue_wait_ms, value.p95_queue_wait_ms);
            combined.p99_queue_wait_ms = std::max(
                combined.p99_queue_wait_ms, value.p99_queue_wait_ms);
            combined.maximum_queue_wait_ms = std::max(
                combined.maximum_queue_wait_ms, value.maximum_queue_wait_ms);
            for (std::size_t index = 0;
                 index < combined.batch_histogram.size(); ++index) {
                combined.batch_histogram[index] += value.batch_histogram[index];
            }
        }
        return combined;
    }

    nlohmann::json measurementSummary() const {
        std::vector<FrameSample> frames;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            frames = frame_samples_;
        }
        std::vector<AttributeBatchSample> batches;
        {
            std::lock_guard<std::mutex> lock(attribute_timing_->mutex);
            batches = attribute_timing_->samples;
        }
        std::vector<double> inference;
        std::vector<double> handler;
        std::vector<double> submit_to_complete;
        std::vector<double> capture_to_complete;
        std::vector<double> attribute_wall;
        std::vector<double> attribute_preprocess;
        std::vector<double> attribute_inference;
        std::vector<double> attribute_postprocess;
        std::vector<double> attribute_total;
        std::array<std::uint64_t, 17> batch_histogram{};
        std::uint64_t detections = 0;
        std::uint64_t attributes = 0;
        for (const auto& frame : frames) {
            inference.push_back(frame.inference_ms);
            handler.push_back(frame.result_handler_ms);
            submit_to_complete.push_back(frame.submit_to_complete_ms);
            capture_to_complete.push_back(frame.capture_to_complete_ms);
            detections += frame.detections;
            attributes += frame.attributes;
        }
        for (const auto& batch : batches) {
            attribute_wall.push_back(batch.wall_ms);
            attribute_preprocess.push_back(batch.stage.preprocess_ms);
            attribute_inference.push_back(batch.stage.inference_ms);
            attribute_postprocess.push_back(batch.stage.postprocess_ms);
            attribute_total.push_back(batch.stage.total_ms);
            if (batch.batch_size < batch_histogram.size()) {
                ++batch_histogram[batch.batch_size];
            }
        }
        nlohmann::json histogram = nlohmann::json::object();
        for (std::size_t index = 0; index < batch_histogram.size(); ++index) {
            histogram[std::to_string(index)] = batch_histogram[index];
        }
        return {
            {"completed_frames", frames.size()},
            {"detections", detections},
            {"attributes", attributes},
            {"detection_inference_ms", summarize(inference)},
            {"result_handler_ms", summarize(handler)},
            {"submit_to_complete_ms", summarize(submit_to_complete)},
            {"capture_to_complete_ms", summarize(capture_to_complete)},
            {"attribute_batches", batches.size()},
            {"attribute_batch_histogram", std::move(histogram)},
            {"attribute_wall_ms", summarize(attribute_wall)},
            {"attribute_preprocess_ms", summarize(attribute_preprocess)},
            {"attribute_inference_ms", summarize(attribute_inference)},
            {"attribute_postprocess_ms", summarize(attribute_postprocess)},
            {"attribute_total_ms", summarize(attribute_total)},
        };
    }

private:
    struct DirectRunnerSlot {
        std::mutex mutex;
        std::unique_ptr<MeasuringAttributeRunner> runner;
    };

    std::size_t directRunnerIndex(const std::string& camera) const {
        if (direct_runners_.empty()) {
            throw std::runtime_error("direct attribute runner pool is empty");
        }
        if (attribute_mode_ == AttributeExecutionMode::PerRoute) {
            return shardedIndex(camera, direct_runners_.size());
        }
        return std::hash<std::string>{}(camera) % direct_runners_.size();
    }

    static std::size_t shardedIndex(
        const std::string& camera,
        std::size_t shard_count) {
        if (shard_count == 0) {
            throw std::runtime_error("attribute shard count is zero");
        }
        {
            const auto separator = camera.rfind('-');
            if (separator != std::string::npos) {
                const int route = std::stoi(camera.substr(separator + 1));
                if (route > 0) {
                    return static_cast<std::size_t>(route - 1) %
                        shard_count;
                }
            }
        }
        return std::hash<std::string>{}(camera) % shard_count;
    }

    void complete(
        const CameraInferenceResult& inference,
        Clock::time_point started,
        std::size_t attribute_count,
        const std::string& failure) {
        const auto now = Clock::now();
        FrameSample sample;
        sample.inference_ms = inference.inference_ms;
        sample.result_handler_ms = std::chrono::duration<double, std::milli>(
            now - started).count();
        sample.capture_to_complete_ms =
            std::chrono::duration<double, std::milli>(
                now - inference.job.frame->publish_time).count();
        sample.detections =
            inference.output.vehicle_detection.detections.size();
        sample.attributes = attribute_count;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            const auto key = std::make_pair(
                inference.job.task_id,
                static_cast<std::uint64_t>(inference.job.source_sequence));
            const auto submitted = submissions_.find(key);
            if (submitted != submissions_.end()) {
                sample.submit_to_complete_ms =
                    std::chrono::duration<double, std::milli>(
                        now - submitted->second).count();
                submissions_.erase(submitted);
            }
            frame_samples_.push_back(sample);
            completed_sequences_[inference.job.task_id] =
                inference.job.source_sequence;
            if (!failure.empty() && failure_.empty()) failure_ = failure;
        }
        completed_.notify_all();
    }

    std::string project_root_;
    AttributeExecutionMode attribute_mode_ =
        AttributeExecutionMode::SharedScheduler;
    int route_count_ = 1;
    std::unique_ptr<VehicleAttributeBatchScheduler> scheduler_;
    std::vector<std::unique_ptr<VehicleAttributeBatchScheduler>>
        scheduler_pool_;
    std::vector<std::unique_ptr<DirectRunnerSlot>> direct_runners_;
    std::shared_ptr<AttributeTimingState> attribute_timing_;
    mutable std::mutex mutex_;
    std::condition_variable completed_;
    std::map<std::pair<std::string, std::uint64_t>, Clock::time_point>
        submissions_;
    std::map<std::string, std::uint64_t> completed_sequences_;
    std::vector<FrameSample> frame_samples_;
    std::string failure_;
};

class RtspFrameSource {
public:
    RtspFrameSource(const std::string& uri, int route_count)
        : last_sequences_(static_cast<std::size_t>(route_count), 0),
          sequence_gaps_(static_cast<std::size_t>(route_count), 0),
          consumed_(static_cast<std::size_t>(route_count), 0) {
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
        const std::string profile = "async-benchmark-shared-source";
        if (!reader_->start(uri, "rtsp://benchmark/source", profile, error)) {
            throw std::runtime_error("RTSP reader start failed: " + error);
        }
    }

    ~RtspFrameSource() { reader_->stop(); }

    SharedCameraFrame next(int route) {
        const auto route_index = static_cast<std::size_t>(route);
        const auto deadline = Clock::now() + std::chrono::seconds(15);
        while (Clock::now() < deadline) {
            auto frame = reader_->getLatestFrameShared(
                last_sequences_.at(route_index));
            if (frame) {
                if (last_sequences_[route_index] != 0 &&
                    frame->sequence > last_sequences_[route_index] + 1) {
                    sequence_gaps_[route_index] += frame->sequence -
                        last_sequences_[route_index] - 1;
                }
                last_sequences_[route_index] = frame->sequence;
                ++consumed_[route_index];
                return frame;
            }
            const auto metrics = reader_->metrics();
            if (metrics.state == "failed") {
                throw std::runtime_error("RTSP reader failed: " +
                    metrics.last_error);
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        throw std::runtime_error("timed out waiting for a new RTSP frame");
    }

    std::uint64_t gaps(int route) const {
        return sequence_gaps_.at(static_cast<std::size_t>(route));
    }
    std::uint64_t consumed(int route) const {
        return consumed_.at(static_cast<std::size_t>(route));
    }

    nlohmann::json metrics() const {
        const auto value = reader_->metrics();
        nlohmann::json routes = nlohmann::json::array();
        for (std::size_t index = 0; index < consumed_.size(); ++index) {
            routes.push_back({
                {"route", index + 1},
                {"consumer_sequence_gaps", sequence_gaps_[index]},
                {"consumer_frames", consumed_[index]},
            });
        }
        return {
            {"state", value.state},
            {"backend", value.backend_name},
            {"capture_fps", value.capture_fps},
            {"source_fps", value.source_fps},
            {"reader_dropped_frames", value.dropped_frames},
            {"reconnect_count", value.reconnect_count},
            {"open_count", value.open_count},
            {"width", value.width},
            {"height", value.height},
            {"routes", std::move(routes)},
        };
    }

private:
    std::unique_ptr<FfmpegProcessCaptureReader> reader_;
    std::vector<std::uint64_t> last_sequences_;
    std::vector<std::uint64_t> sequence_gaps_;
    std::vector<std::uint64_t> consumed_;
};

nlohmann::json poolSnapshot(const CameraInferencePoolSnapshot& value) {
    return {
        {"running", value.running},
        {"workers_configured", value.workers_configured},
        {"workers_ready", value.workers_ready},
        {"active_cameras", value.active_cameras},
        {"pending_cameras", value.pending_cameras},
        {"submitted_jobs", value.submitted_jobs},
        {"replaced_jobs", value.replaced_jobs},
        {"processed_jobs", value.processed_jobs},
        {"failed_jobs", value.failed_jobs},
        {"stale_results", value.stale_results},
        {"pending_result_jobs", value.pending_result_jobs},
        {"maximum_pending_result_jobs", value.maximum_pending_result_jobs},
        {"handled_result_jobs", value.handled_result_jobs},
    };
}

nlohmann::json schedulerDelta(
    const VehicleAttributeBatchSchedulerSnapshot& before,
    const VehicleAttributeBatchSchedulerSnapshot& after) {
    nlohmann::json histogram = nlohmann::json::object();
    for (std::size_t index = 0; index < after.batch_histogram.size(); ++index) {
        histogram[std::to_string(index)] =
            after.batch_histogram[index] - before.batch_histogram[index];
    }
    const auto completed = after.completed_requests - before.completed_requests;
    const double before_sum = before.mean_queue_wait_ms *
        static_cast<double>(before.completed_requests);
    const double after_sum = after.mean_queue_wait_ms *
        static_cast<double>(after.completed_requests);
    return {
        {"requests", after.requests - before.requests},
        {"completed_requests", completed},
        {"failed_requests", after.failed_requests - before.failed_requests},
        {"batches", after.batches - before.batches},
        {"crops", after.crops - before.crops},
        {"pending_requests", after.pending_requests},
        {"pending_crops", after.pending_crops},
        {"maximum_pending_requests_lifetime",
            after.maximum_pending_requests},
        {"maximum_pending_crops_lifetime", after.maximum_pending_crops},
        {"mean_queue_wait_ms", completed == 0 ? 0.0 :
            (after_sum - before_sum) / static_cast<double>(completed)},
        {"p95_queue_wait_ms_lifetime", after.p95_queue_wait_ms},
        {"p99_queue_wait_ms_lifetime", after.p99_queue_wait_ms},
        {"maximum_queue_wait_ms_lifetime", after.maximum_queue_wait_ms},
        {"batch_histogram", std::move(histogram)},
    };
}

void runJobs(
    CameraInferencePool& pool,
    const std::shared_ptr<BenchmarkResultHandler>& handler,
    RtspFrameSource& source,
    int streams,
    std::vector<std::uint64_t>& job_sequences,
    int total_jobs) {
    std::atomic<bool> failed{false};
    std::mutex failure_mutex;
    std::string failure;
    std::vector<std::thread> producers;
    producers.reserve(static_cast<std::size_t>(streams));
    for (int stream = 0; stream < streams; ++stream) {
        const int count = total_jobs / streams +
            (stream < total_jobs % streams ? 1 : 0);
        producers.emplace_back([&, stream, count]() {
            try {
                for (int index = 0; index < count && !failed.load(); ++index) {
                    auto frame = source.next(stream);
                    CameraFrameJob job;
                    job.task_id = "async-benchmark-camera-" +
                        std::to_string(stream + 1);
                    job.run_id = "async-benchmark-run";
                    job.camera_profile = job.task_id;
                    job.source_sequence = ++job_sequences[stream];
                    job.capture_time_ms = frame->capture_time_ms;
                    job.algorithm_profile = "vehicle-production-core";
                    job.algorithms = {
                        "vehicle_detection", "vehicle_attribute"};
                    job.frame = std::move(frame);
                    handler->noteSubmission(
                        job.task_id, job.source_sequence, Clock::now());
                    CameraFrameJobSubmitResult submit;
                    std::string error;
                    if (!pool.submitLatest(std::move(job), submit, error) ||
                        !submit.accepted || submit.dropped_backlog != 0) {
                        throw std::runtime_error(
                            "inference submission failed or replaced: " + error);
                    }
                    if (!handler->waitFor(
                            "async-benchmark-camera-" +
                                std::to_string(stream + 1),
                            job_sequences[stream],
                            std::chrono::seconds(120), error)) {
                        throw std::runtime_error(error);
                    }
                }
            }
            catch (const std::exception& exception) {
                failed.store(true);
                std::lock_guard<std::mutex> lock(failure_mutex);
                if (failure.empty()) failure = exception.what();
            }
        });
    }
    for (auto& producer : producers) producer.join();
    if (failed.load()) throw std::runtime_error(failure);
}

nlohmann::json runMode(
    const std::string& project_root,
    const std::string& source,
    int streams,
    int measurement_jobs,
    int warmup_jobs,
    bool asynchronous,
    AttributeExecutionMode attribute_mode =
        AttributeExecutionMode::SharedScheduler) {
    const std::string mode_name = std::string(
        asynchronous ? "async" : "sync") + "/" +
        attributeExecutionName(attribute_mode);
    std::cerr << "[benchmark] " << mode_name << ": initializing attribute scheduler\n";
    AppConfig config;
    config.model.gpu_id = 0;
    config.analysis.enabled = true;
    config.analysis.inference_workers = 2;
    config.analysis.model_init_timeout_ms = 120000;
    config.analysis.async_result_dispatch = asynchronous;
    config.analysis.result_queue_capacity = 2;

    auto handler = std::make_shared<BenchmarkResultHandler>(
        project_root, attribute_mode, streams);
    std::string error;
    if (!handler->start(error)) {
        throw std::runtime_error("attribute scheduler start failed: " + error);
    }
    CameraInferencePool pool(
        config,
        [project_root](int) {
            return std::make_unique<BenchmarkVehicleRunner>(project_root);
        },
        handler);
    std::cerr << "[benchmark] " << mode_name << ": initializing inference pool\n";
    if (!pool.start(error)) {
        handler->stop();
        throw std::runtime_error("inference pool start failed: " + error);
    }

    std::vector<std::uint64_t> job_sequences(
        static_cast<std::size_t>(streams), 0);
    try {
        RtspFrameSource frame_source(source, streams);
        std::cerr << "[benchmark] " << mode_name
                  << ": shared RTSP source for " << streams
                  << " routes started\n";
        std::cerr << "[benchmark] " << mode_name << ": warming "
                  << warmup_jobs << " jobs\n";
        runJobs(
            pool, handler, frame_source, streams, job_sequences, warmup_jobs);
        handler->clearMeasurementSamples();
        const auto scheduler_before = handler->schedulerSnapshot();
        std::vector<std::uint64_t> gaps_before;
        std::vector<std::uint64_t> consumed_before;
        for (int route = 0; route < streams; ++route) {
            gaps_before.push_back(frame_source.gaps(route));
            consumed_before.push_back(frame_source.consumed(route));
        }

        const auto started = Clock::now();
        std::cerr << "[benchmark] " << mode_name << ": measuring "
                  << measurement_jobs << " jobs\n";
        runJobs(
            pool, handler, frame_source, streams, job_sequences,
            measurement_jobs);
        const double wall_seconds =
            std::chrono::duration<double>(Clock::now() - started).count();
        const auto pool_after = pool.snapshot();
        const auto scheduler_after = handler->schedulerSnapshot();
        auto summary = handler->measurementSummary();

        std::uint64_t source_gaps = 0;
        std::uint64_t consumed = 0;
        for (int route = 0; route < streams; ++route) {
            const auto index = static_cast<std::size_t>(route);
            source_gaps += frame_source.gaps(route) - gaps_before[index];
            consumed += frame_source.consumed(route) - consumed_before[index];
        }
        summary["mode"] = asynchronous ? "async_result_dispatch" :
            "synchronous_result_dispatch";
        summary["attribute_execution"] =
            attributeExecutionName(attribute_mode);
        summary["wall_seconds"] = wall_seconds;
        summary["aggregate_fps"] =
            static_cast<double>(measurement_jobs) / wall_seconds;
        summary["measurement_jobs_requested"] = measurement_jobs;
        summary["measurement_source_frames_consumed"] = consumed;
        summary["inference_job_replacements"] = pool_after.replaced_jobs;
        summary["inference_silent_drops"] = 0;
        summary["source_latest_frame_sequence_gaps"] = source_gaps;
        summary["pool"] = poolSnapshot(pool_after);
        summary["attribute_scheduler"] = schedulerDelta(
            scheduler_before, scheduler_after);
        summary["source"] = frame_source.metrics();
        std::cerr << "[benchmark] " << mode_name << ": stopping\n";
        pool.stop();
        handler->stop();
        std::cerr << "[benchmark] " << mode_name << ": complete\n";
        return summary;
    }
    catch (...) {
        pool.stop();
        handler->stop();
        throw;
    }
}

double medianMetric(const nlohmann::json& runs, const char* key) {
    std::vector<double> values;
    for (const auto& run : runs) values.push_back(run.at(key).get<double>());
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}

int parsePositive(const char* value, const char* name, int maximum) {
    const int parsed = std::stoi(value);
    if (parsed <= 0 || parsed > maximum) {
        throw std::invalid_argument(std::string(name) + " is out of range");
    }
    return parsed;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3 || argc > 9) {
        std::cerr <<
            "Usage: vehicle_async_pipeline_benchmark <project-root> <rtsp> "
            "[streams=4] [measurement-jobs=2000] [warmup-jobs=200] "
            "[runs=3] [report.json] [suite=async|scheduler]\n";
        return 2;
    }
    try {
        const std::string project_root = argv[1];
        const std::string source = argv[2];
        if (source.rfind("rtsp://", 0) != 0 &&
            source.rfind("rtsps://", 0) != 0) {
            throw std::invalid_argument("benchmark source must be RTSP");
        }
        const int streams = argc >= 4 ?
            parsePositive(argv[3], "streams", 16) : 4;
        const int measurement_jobs = argc >= 5 ?
            parsePositive(argv[4], "measurement jobs", 100000) : 2000;
        const int warmup_jobs = argc >= 6 ?
            parsePositive(argv[5], "warmup jobs", 10000) : 200;
        const int runs = argc >= 7 ?
            parsePositive(argv[6], "runs", 20) : 3;
        const std::string suite = argc >= 9 ? argv[8] : "async";
        nlohmann::json report;
        if (suite == "async") {
            nlohmann::json synchronous = nlohmann::json::array();
            nlohmann::json asynchronous = nlohmann::json::array();
            for (int run = 0; run < runs; ++run) {
                if ((run & 1) == 0) {
                    synchronous.push_back(runMode(
                        project_root, source, streams, measurement_jobs,
                        warmup_jobs, false));
                    asynchronous.push_back(runMode(
                        project_root, source, streams, measurement_jobs,
                        warmup_jobs, true));
                }
                else {
                    asynchronous.push_back(runMode(
                        project_root, source, streams, measurement_jobs,
                        warmup_jobs, true));
                    synchronous.push_back(runMode(
                        project_root, source, streams, measurement_jobs,
                        warmup_jobs, false));
                }
                synchronous.back()["run"] = run + 1;
                asynchronous.back()["run"] = run + 1;
            }
            const double synchronous_fps = medianMetric(
                synchronous, "aggregate_fps");
            const double asynchronous_fps = medianMetric(
                asynchronous, "aggregate_fps");
            report = {
                {"schema_version", "1.0"},
                {"benchmark", "bounded cross-frame result dispatch"},
                {"source_kind", "real_rtsp"},
                {"source", source},
                {"stream_count", streams},
                {"inference_workers", 2},
                {"result_queue_capacity_per_worker", 2},
                {"measurement_jobs_per_run", measurement_jobs},
                {"warmup_jobs_per_mode_per_run", warmup_jobs},
                {"runs_requested", runs},
                {"run_order", "alternating sync/async pairs"},
                {"fixed_completion_count", true},
                {"synchronous", synchronous},
                {"asynchronous", asynchronous},
                {"synchronous_median_aggregate_fps", synchronous_fps},
                {"asynchronous_median_aggregate_fps", asynchronous_fps},
                {"throughput_change_percent", synchronous_fps > 0.0 ?
                    (asynchronous_fps / synchronous_fps - 1.0) * 100.0 : 0.0},
            };
        }
        else if (suite == "scheduler") {
            nlohmann::json independent = nlohmann::json::array();
            nlohmann::json context_pool = nlohmann::json::array();
            nlohmann::json scheduler_pool = nlohmann::json::array();
            nlohmann::json shared = nlohmann::json::array();
            const std::array<AttributeExecutionMode, 4> modes = {
                AttributeExecutionMode::PerRoute,
                AttributeExecutionMode::ContextPool,
                AttributeExecutionMode::SchedulerPool,
                AttributeExecutionMode::SharedScheduler};
            for (int run = 0; run < runs; ++run) {
                for (int offset = 0; offset < 4; ++offset) {
                    const auto mode = modes[static_cast<std::size_t>(
                        (run + offset) % 4)];
                    auto value = runMode(
                        project_root, source, streams, measurement_jobs,
                        warmup_jobs, false, mode);
                    value["run"] = run + 1;
                    if (mode == AttributeExecutionMode::PerRoute) {
                        independent.push_back(std::move(value));
                    }
                    else if (mode == AttributeExecutionMode::ContextPool) {
                        context_pool.push_back(std::move(value));
                    }
                    else if (mode == AttributeExecutionMode::SchedulerPool) {
                        scheduler_pool.push_back(std::move(value));
                    }
                    else {
                        shared.push_back(std::move(value));
                    }
                }
            }
            const double independent_fps = medianMetric(
                independent, "aggregate_fps");
            const double context_pool_fps = medianMetric(
                context_pool, "aggregate_fps");
            const double scheduler_pool_fps = medianMetric(
                scheduler_pool, "aggregate_fps");
            const double shared_fps = medianMetric(shared, "aggregate_fps");
            report = {
                {"schema_version", "1.0"},
                {"benchmark", "multi-route attribute execution matrix"},
                {"source_kind", "real_rtsp"},
                {"source", source},
                {"stream_count", streams},
                {"inference_workers", 2},
                {"measurement_jobs_per_run", measurement_jobs},
                {"warmup_jobs_per_mode_per_run", warmup_jobs},
                {"runs_requested", runs},
                {"run_order", "rotating independent/context-pool/scheduler-pool/shared"},
                {"fixed_completion_count", true},
                {"independent_context_per_route", independent},
                {"bounded_context_pool_2", context_pool},
                {"sharded_scheduler_pool_2", scheduler_pool},
                {"shared_scheduler_single_context", shared},
                {"median_aggregate_fps", {
                    {"independent_context_per_route", independent_fps},
                    {"bounded_context_pool_2", context_pool_fps},
                    {"sharded_scheduler_pool_2", scheduler_pool_fps},
                    {"shared_scheduler_single_context", shared_fps}}},
                {"shared_vs_independent_throughput_change_percent",
                    independent_fps > 0.0 ?
                        (shared_fps / independent_fps - 1.0) * 100.0 : 0.0},
                {"shared_vs_context_pool_throughput_change_percent",
                    context_pool_fps > 0.0 ?
                        (shared_fps / context_pool_fps - 1.0) * 100.0 : 0.0},
                {"scheduler_pool_vs_shared_throughput_change_percent",
                    shared_fps > 0.0 ?
                        (scheduler_pool_fps / shared_fps - 1.0) * 100.0 : 0.0},
            };
        }
        else {
            throw std::invalid_argument(
                "suite must be either async or scheduler");
        }
        const std::string serialized = report.dump(2);
        if (argc >= 8) {
            const std::filesystem::path path(argv[7]);
            if (!path.parent_path().empty()) {
                std::filesystem::create_directories(path.parent_path());
            }
            std::ofstream output(path, std::ios::binary | std::ios::trunc);
            if (!output) throw std::runtime_error("could not create report");
            output << serialized << '\n';
        }
        std::cout << serialized << '\n';
        return 0;
    }
    catch (const std::exception& exception) {
        std::cerr << "FAIL: " << exception.what() << '\n';
        return 1;
    }
}
