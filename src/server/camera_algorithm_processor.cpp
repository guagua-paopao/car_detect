#include "server/camera_algorithm_processor.h"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iomanip>
#include <set>
#include <sstream>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "business/vehicle_event_publisher.h"
#include "server/camera_task_api_control.h"
#include "server/vehicle_runtime_loader.h"
#include "server/vehicle_tensorrt_adapters.h"
#include "server/vehicle_attribute_batch_scheduler.h"

namespace yolo11_server {

namespace {

using json = nlohmann::json;

long long wallNowMs() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}


}  // namespace


struct CameraAlgorithmProcessor::VehicleSession {
    VehicleSession(
        const AppConfig& config,
        const CameraFrameJob& job,
        const VehicleRuntimeAssets& assets,
        std::size_t attribute_runner_index_value)
        : run_id(job.run_id),
          camera_profile(job.camera_profile),
          callback_profile(job.callback_profile),
          config_version(assets.config_version),
          attribute_runner_index(attribute_runner_index_value),
          runtime(assets.cascade) {
        runtime.startRun(job.task_id, job.run_id, 1);
        const int snapshot_fps = job.snapshot_fps > 0
            ? job.snapshot_fps : config.vehicle_analytics.snapshot_fps;
        snapshot_interval_frames = std::max(1, static_cast<int>(
            std::max(0.1, job.target_infer_fps) /
            std::max(1, snapshot_fps)));
        const auto relative = std::filesystem::path(job.task_id) /
            job.run_id / "analysis" / "latest.jpg";
        snapshot_relative_path = relative.generic_string();
        snapshot_path =
            std::filesystem::u8path(config.camera_tasks.output_dir) / relative;
        std::error_code directory_error;
        std::filesystem::create_directories(
            snapshot_path.parent_path(), directory_error);
        snapshot_degraded = static_cast<bool>(directory_error);
    }

    std::string run_id;
    std::string camera_profile;
    std::string callback_profile;
    std::string config_version;
    std::size_t attribute_runner_index = 0;
    VehicleCascadeRuntime runtime;
    std::filesystem::path snapshot_path;
    std::string snapshot_relative_path;
    int snapshot_interval_frames = 1;
    long long last_snapshot_frame = 0;
    long long frame_count = 0;
    long long last_persist_ms = 0;
    long long infer_window_start_ms = 0;
    long long infer_window_frames = 0;
    double infer_fps = 0.0;
    bool snapshot_degraded = false;
    bool snapshot_pending = false;
    bool storage_degraded = false;
    bool active = true;
    bool has_snapshot = false;
    CameraRunAnalysisResultRecord latest_result;
    CameraTaskRunHotStatus latest_hot;
    std::map<std::int64_t, VehicleAttributeResult> latest_candidates;
    std::mutex mutex;
};

struct CameraAlgorithmProcessor::VehicleAttributeRunnerSlot {
    std::unique_ptr<TensorRtVehicleAttributeRunner> runner;
    std::size_t active_sessions = 0;
    std::mutex mutex;
};

CameraAlgorithmProcessor::CameraAlgorithmProcessor(
    AppConfig config,
    std::shared_ptr<CameraTaskRepository> repository,
    std::shared_ptr<ICameraAnalysisStatusSink> status_sink
) : config_(std::move(config)),
    repository_(std::move(repository)),
    status_sink_(std::move(status_sink)) {
}

CameraAlgorithmProcessor::~CameraAlgorithmProcessor() noexcept {
    stop();
}

bool CameraAlgorithmProcessor::start(std::string& error) {
    error.clear();
    if (running_.load()) return true;
    if (!repository_) {
        error = "camera algorithm repository is unavailable";
        return false;
    }
    if (!repository_->initialize(error)) return false;
    const std::string model_type = config_.model.type;
    vehicle_enabled_ = config_.vehicle_analytics.enabled &&
        (model_type == "vehicle" || model_type == "vehicle_detection");
    if (vehicle_enabled_) {
        vehicle_assets_ = std::make_unique<VehicleRuntimeAssets>();
        if (!loadVehicleRuntimeAssets(config_, *vehicle_assets_, error)) {
            vehicle_assets_.reset();
            return false;
        }
        const auto create_attribute_runner = [&]() {
            TensorRtAttributeOptions options;
            options.gpu_id = config_.model.gpu_id;
            options.artifact_root = ".";
            options.body_types = vehicle_assets_->body_types;
            options.colors = vehicle_assets_->colors;
            return std::make_unique<TensorRtVehicleAttributeRunner>(
                std::move(options));
        };
        if (config_.vehicle_analytics.dynamic_batching) {
            auto attribute_runner = create_attribute_runner();
            if (!attribute_runner->initialize(
                    vehicle_assets_->attributes, error)) {
                vehicle_assets_.reset();
                return false;
            }
            VehicleAttributeBatchSchedulerConfig scheduler_config;
            scheduler_config.max_batch = vehicle_assets_->attribute_max_batch;
            scheduler_config.max_pending_requests = 64;
            scheduler_config.max_wait = std::chrono::milliseconds(2);
            vehicle_attribute_scheduler_ =
                std::make_unique<VehicleAttributeBatchScheduler>(
                    scheduler_config, std::move(attribute_runner));
            if (!vehicle_attribute_scheduler_->start(error)) {
                vehicle_attribute_scheduler_.reset();
                vehicle_assets_.reset();
                return false;
            }
        }
        else {
            for (int index = 0;
                 index < config_.vehicle_analytics.attribute_contexts;
                 ++index) {
                auto slot = std::make_unique<VehicleAttributeRunnerSlot>();
                slot->runner = create_attribute_runner();
                if (!slot->runner->initialize(
                        vehicle_assets_->attributes, error)) {
                    for (auto& created : vehicle_attribute_runners_) {
                        created->runner->release();
                    }
                    vehicle_attribute_runners_.clear();
                    vehicle_assets_.reset();
                    return false;
                }
                vehicle_attribute_runners_.push_back(std::move(slot));
            }
        }
        vehicle_event_publisher_ =
            std::make_unique<VehicleEventPublisher>(repository_);
        if (config_.vehicle_analytics.async_snapshots) {
            snapshot_writer_ = std::make_unique<AnalysisSnapshotWriter>();
            if (!snapshot_writer_->start(
                    config_.vehicle_analytics.snapshot_writer_threads,
                    static_cast<std::size_t>(
                        config_.vehicle_analytics.snapshot_queue_capacity),
                    error)) {
                snapshot_writer_.reset();
                vehicle_event_publisher_.reset();
                if (vehicle_attribute_scheduler_) {
                    vehicle_attribute_scheduler_->stop();
                }
                vehicle_attribute_scheduler_.reset();
                for (auto& slot : vehicle_attribute_runners_) {
                    if (slot && slot->runner) slot->runner->release();
                }
                vehicle_attribute_runners_.clear();
                vehicle_assets_.reset();
                vehicle_enabled_ = false;
                return false;
            }
        }
    }
    running_.store(true);
    return true;
}

void CameraAlgorithmProcessor::stop() noexcept {
    running_.store(false);
    try {
        if (snapshot_writer_) snapshot_writer_->stop();
        snapshot_writer_.reset();
        {
            std::lock_guard<std::mutex> vehicle_lock(vehicle_sessions_mutex_);
            for (const auto& entry : vehicle_sessions_) {
                std::lock_guard<std::mutex> session_lock(entry.second->mutex);
                entry.second->active = false;
            }
            vehicle_sessions_.clear();
        }
        if (vehicle_attribute_scheduler_) vehicle_attribute_scheduler_->stop();
        vehicle_attribute_scheduler_.reset();
        for (auto& slot : vehicle_attribute_runners_) {
            if (slot && slot->runner) slot->runner->release();
        }
        vehicle_attribute_runners_.clear();
        vehicle_event_publisher_.reset();
        vehicle_assets_.reset();
        vehicle_enabled_ = false;
    }
    catch (...) {
    }
}


bool CameraAlgorithmProcessor::handleVehicle(
    const CameraInferenceResult& inference,
    std::string& error) {
    error.clear();
    if (!vehicle_enabled_ || !vehicle_assets_ ||
        (!vehicle_attribute_scheduler_ && vehicle_attribute_runners_.empty()) ||
        !vehicle_event_publisher_) {
        error = "VEHICLE_ANALYTICS_NOT_INITIALIZED";
        return false;
    }

    std::shared_ptr<VehicleSession> session;
    {
        std::lock_guard<std::mutex> lock(vehicle_sessions_mutex_);
        auto found = vehicle_sessions_.find(inference.job.task_id);
        if (found != vehicle_sessions_.end() &&
            found->second->run_id == inference.job.run_id) {
            session = found->second;
        }
        else {
            if (found != vehicle_sessions_.end()) {
                std::lock_guard<std::mutex> old_lock(found->second->mutex);
                found->second->active = false;
                const auto old_index = found->second->attribute_runner_index;
                if (old_index < vehicle_attribute_runners_.size() &&
                    vehicle_attribute_runners_[old_index]->active_sessions > 0) {
                    --vehicle_attribute_runners_[old_index]->active_sessions;
                }
                vehicle_sessions_.erase(found);
            }
            std::size_t attribute_runner_index = 0;
            if (!vehicle_attribute_runners_.empty()) {
                for (std::size_t index = 1;
                     index < vehicle_attribute_runners_.size(); ++index) {
                    if (vehicle_attribute_runners_[index]->active_sessions <
                        vehicle_attribute_runners_[attribute_runner_index]
                            ->active_sessions) {
                        attribute_runner_index = index;
                    }
                }
                ++vehicle_attribute_runners_[attribute_runner_index]
                    ->active_sessions;
            }
            session = std::make_shared<VehicleSession>(
                config_, inference.job, *vehicle_assets_,
                attribute_runner_index);
            vehicle_sessions_[inference.job.task_id] = session;
        }
    }

    CameraRunAnalysisResultRecord analysis_result;
    CameraTaskRunHotStatus hot;
    std::vector<VehicleTrackSnapshot> exited_tracks;
    bool persist_analysis = false;
    {
        std::lock_guard<std::mutex> lock(session->mutex);
        if (!session->active || session->run_id != inference.job.run_id) {
            error = "STALE_VEHICLE_ANALYSIS_SESSION";
            return false;
        }

        VehicleDetectionResult detections =
            inference.output.vehicle_detection;
        detections.camera_id = inference.job.task_id;
        detections.run_id = inference.job.run_id;
        detections.frame_sequence = static_cast<std::int64_t>(
            inference.job.source_sequence);
        detections.captured_at_ms = inference.job.capture_time_ms;
        detections.metadata.run_generation = 1;

        VehicleTrackingUpdate tracking;
        if (!session->runtime.onDetections(detections, tracking, error)) {
            return false;
        }
        exited_tracks = tracking.exited;

        std::shared_ptr<OwnedI420Image> i420_frame;
        if (inference.job.frame->i420.valid()) {
            const auto& source = inference.job.frame->i420;
            i420_frame = std::make_shared<OwnedI420Image>();
            i420_frame->bytes = source.bytes;
            i420_frame->width = source.width;
            i420_frame->height = source.height;
            i420_frame->y_offset = source.y_offset;
            i420_frame->u_offset = source.u_offset;
            i420_frame->v_offset = source.v_offset;
            i420_frame->y_stride_bytes = source.y_stride_bytes;
            i420_frame->u_stride_bytes = source.u_stride_bytes;
            i420_frame->v_stride_bytes = source.v_stride_bytes;
        }
        ImageView frame_view;
        if (!i420_frame) {
            const auto& image = inference.job.frame->bgrImage();
            frame_view = {
                image.data,
                image.cols,
                image.rows,
                image.channels(),
                image.step,
                ImagePixelFormat::Bgr8
            };
        }
        for (const auto& track : tracking.active) {
            if (track.state != VehicleTrackState::Confirmed &&
                track.state != VehicleTrackState::AttributeCollecting) {
                continue;
            }
            CropQualityAssessment assessment;
            std::string crop_error;
            const bool truncated = track.box.x1 <= 0.002f ||
                track.box.y1 <= 0.002f || track.box.x2 >= 0.998f ||
                track.box.y2 >= 0.998f;
            if (i420_frame) {
                session->runtime.queueCrop(
                    i420_frame,
                    detections.device_i420_frame,
                    track.key.track_id,
                    inference.job.source_sequence,
                    0.0f,
                    truncated,
                    assessment,
                    crop_error);
            }
            else {
                session->runtime.queueCrop(
                    frame_view,
                    track.key.track_id,
                    inference.job.source_sequence,
                    0.0f,
                    truncated,
                    assessment,
                    crop_error);
            }
        }

        auto crops = session->runtime.takeAttributeBatch(
            vehicle_assets_->attribute_max_batch);
        if (!crops.empty()) {
            std::vector<VehicleAttributeResult> attributes;
            if (vehicle_attribute_scheduler_) {
                if (!vehicle_attribute_scheduler_->infer(
                        std::move(crops), attributes, error)) return false;
            }
            else {
                const auto index = session->attribute_runner_index;
                if (index >= vehicle_attribute_runners_.size()) {
                    error = "VEHICLE_ATTRIBUTE_CONTEXT_UNAVAILABLE";
                    return false;
                }
                auto& slot = *vehicle_attribute_runners_[index];
                try {
                    std::lock_guard<std::mutex> runner_lock(slot.mutex);
                    attributes = slot.runner->inferBatch(crops);
                }
                catch (const std::exception& exception) {
                    error = exception.what();
                    return false;
                }
            }
            if (!session->runtime.onAttributeResults(attributes, error)) {
                return false;
            }
            for (const auto& item : attributes) {
                session->latest_candidates[item.track_id] = item;
                bool duplicate = false;
                std::string observation_error;
                vehicle_event_publisher_->persistObservation(
                    item,
                    inference.job.capture_time_ms,
                    config_.vehicle_analytics.observation_retention_days,
                    duplicate,
                    observation_error);
            }
        }

        ++session->frame_count;
        ++session->infer_window_frames;
        if (session->infer_window_start_ms <= 0) {
            session->infer_window_start_ms = inference.job.capture_time_ms;
        }
        const auto infer_elapsed = inference.job.capture_time_ms -
            session->infer_window_start_ms;
        if (infer_elapsed >= 1000) {
            session->infer_fps =
                static_cast<double>(session->infer_window_frames) * 1000.0 /
                std::max(1LL, infer_elapsed);
            session->infer_window_frames = 0;
            session->infer_window_start_ms = inference.job.capture_time_ms;
        }

        const bool snapshot_due = !session->snapshot_pending &&
            (!session->has_snapshot ||
             session->frame_count - session->last_snapshot_frame >=
                session->snapshot_interval_frames);
        const cv::Mat* snapshot_image = nullptr;
        cv::Mat rendered;
        if (snapshot_due) {
            const auto& image = inference.job.frame->bgrImage();
            snapshot_image = &image;
            rendered = image.clone();
        }
        json realtime_items = json::array();
        int confirmed_count = 0;
        for (const auto& tracked : tracking.active) {
            const auto current = session->runtime.trackSnapshot(
                tracked.key.track_id);
            const auto& track = current ? *current : tracked;
            const auto attributes = session->runtime.attributeSnapshot(
                track.key.track_id);
            const auto candidate = session->latest_candidates.find(
                track.key.track_id);
            if (track.state == VehicleTrackState::Confirmed ||
                track.state == VehicleTrackState::AttributeCollecting ||
                track.state == VehicleTrackState::AttributeStable) {
                ++confirmed_count;
            }
            if (snapshot_image) {
                const auto& image = *snapshot_image;
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
                const cv::Scalar color(40, 220, 40);
                cv::rectangle(
                    rendered, cv::Rect(left, top, right - left, bottom - top),
                    color, 2, cv::LINE_AA);
                std::ostringstream label;
                label << "#" << track.key.track_id << " "
                      << track.vehicle_class << " " << std::fixed
                      << std::setprecision(2) << track.detection_confidence;
                if (attributes && attributes->stable()) {
                    label << " | " << attributes->body_type.label
                          << " | " << attributes->color.label;
                }
                else if (candidate != session->latest_candidates.end()) {
                    label << " | " << candidate->second.body_type.label << "?"
                          << " | " << candidate->second.color.label << "?";
                }
                else {
                    label << " | unknown | unknown";
                }
                int baseline = 0;
                const auto text_size = cv::getTextSize(
                    label.str(), cv::FONT_HERSHEY_SIMPLEX, 0.52, 1,
                    &baseline);
                const int text_top = std::max(
                    0, top - text_size.height - 8);
                cv::rectangle(
                    rendered,
                    cv::Rect(
                        left,
                        text_top,
                        std::min(text_size.width + 8, image.cols - left),
                        text_size.height + 8),
                    cv::Scalar(20, 20, 20),
                    cv::FILLED);
                cv::putText(
                    rendered,
                    label.str(),
                    cv::Point(left + 4, text_top + text_size.height + 2),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.52,
                    color,
                    1,
                    cv::LINE_AA);
            }

            realtime_items.push_back({
                {"track_id", track.key.track_id},
                {"run_generation", track.key.run_generation},
                {"state", std::string(toString(track.state))},
                {"last_seen_at_ms", track.last_seen_ms},
                {"vehicle_class", track.vehicle_class},
                {"vehicle_class_confidence", track.detection_confidence},
                {"body_type", attributes
                    ? attributes->body_type.label : "unknown"},
                {"body_type_confidence", attributes
                    ? attributes->body_type.confidence : 0.0f},
                {"body_type_stable", attributes
                    ? attributes->body_type.stable : false},
                {"body_type_samples_used", attributes
                    ? attributes->body_type.samples_used : 0},
                {"color", attributes
                    ? attributes->color.label : "unknown"},
                {"color_confidence", attributes
                    ? attributes->color.confidence : 0.0f},
                {"color_stable", attributes
                    ? attributes->color.stable : false},
                {"color_samples_used", attributes
                    ? attributes->color.samples_used : 0},
                {"attribute_candidate", candidate == session->latest_candidates.end()
                    ? json(nullptr) : json{
                        {"body_type", candidate->second.body_type.label},
                        {"body_type_confidence", candidate->second.body_type.confidence},
                        {"color", candidate->second.color.label},
                        {"color_confidence", candidate->second.color.confidence},
                        {"published_as", "unknown_until_stable"}
                    }}
            });
        }

        if (snapshot_image) {
            cv::rectangle(
                rendered, cv::Rect(0, 0, rendered.cols, 34),
                cv::Scalar(15, 15, 15), cv::FILLED);
            const std::string panel =
                "VCAS Vehicle AI | det=" +
                vehicle_assets_->detector.artifact_id +
                " attr=" + vehicle_assets_->attributes.artifact_id +
                " | vehicles=" + std::to_string(confirmed_count);
            cv::putText(
                rendered, panel, cv::Point(12, 23),
                cv::FONT_HERSHEY_SIMPLEX, 0.58,
                cv::Scalar(0, 220, 255), 2, cv::LINE_AA);

            if (snapshot_writer_) {
                AnalysisSnapshotJob snapshot_job;
                snapshot_job.image = std::move(rendered);
                snapshot_job.output_path = session->snapshot_path;
                snapshot_job.jpeg_quality = std::clamp(
                    config_.vehicle_analytics.jpeg_quality, 1, 100);
                const auto scheduled_frame = session->frame_count;
                const auto scheduled_run = session->run_id;
                std::weak_ptr<VehicleSession> weak_session = session;
                session->snapshot_pending = true;
                std::string snapshot_error;
                if (!snapshot_writer_->enqueue(
                        std::move(snapshot_job),
                        [weak_session, scheduled_frame, scheduled_run](
                            const AnalysisSnapshotWriteResult& result) {
                            const auto current = weak_session.lock();
                            if (!current) return;
                            std::lock_guard<std::mutex> lock(current->mutex);
                            current->snapshot_pending = false;
                            if (!current->active ||
                                current->run_id != scheduled_run) {
                                return;
                            }
                            if (result.success) {
                                current->last_snapshot_frame =
                                    scheduled_frame;
                                current->snapshot_degraded = false;
                                current->has_snapshot = true;
                            }
                            else {
                                current->snapshot_degraded = true;
                            }
                        },
                        snapshot_error)) {
                    session->snapshot_pending = false;
                    session->snapshot_degraded = true;
                }
            }
            else {
                try {
                    const std::vector<int> parameters{
                        cv::IMWRITE_JPEG_QUALITY,
                        std::clamp(config_.vehicle_analytics.jpeg_quality, 1, 100)
                    };
                    if (cv::imwrite(
                            session->snapshot_path.string(), rendered,
                            parameters)) {
                        session->last_snapshot_frame = session->frame_count;
                        session->snapshot_degraded = false;
                        session->has_snapshot = true;
                    }
                    else {
                        session->snapshot_degraded = true;
                    }
                }
                catch (...) {
                    session->snapshot_degraded = true;
                }
            }
        }

        const json vehicle_state = {
            {"success", true},
            {"mode", "vehicle_cascade"},
            {"camera_id", inference.job.task_id},
            {"run_id", inference.job.run_id},
            {"timestamp_ms", inference.job.capture_time_ms},
            {"detector_artifact", vehicle_assets_->detector.artifact_id},
            {"attribute_artifact", vehicle_assets_->attributes.artifact_id},
            {"labels_version", vehicle_assets_->labels_version},
            {"detection_count", detections.detections.size()},
            {"confirmed_count", confirmed_count},
            {"items", std::move(realtime_items)}
        };
        analysis_result.run_id = inference.job.run_id;
        analysis_result.task_id = inference.job.task_id;
        analysis_result.analysis_state_json = vehicle_state.dump();
        analysis_result.snapshot_relative_path =
            session->has_snapshot
                ? session->snapshot_relative_path : std::string{};
        analysis_result.storage_degraded = session->storage_degraded;
        analysis_result.snapshot_degraded = session->snapshot_degraded;
        analysis_result.last_update_ms = inference.job.capture_time_ms;

        hot.found = true;
        hot.run_id = inference.job.run_id;
        hot.task_id = inference.job.task_id;
        hot.analysis_config_version = session->config_version;
        hot.infer_fps = session->infer_fps;
        hot.last_inference_ms = inference.inference_ms;
        hot.analysis_frame_count = session->frame_count;
        hot.analysis_state_json = analysis_result.analysis_state_json;
        hot.analysis_snapshot_relative_path =
            analysis_result.snapshot_relative_path;
        hot.analysis_storage_degraded = session->storage_degraded;
        hot.analysis_snapshot_degraded = session->snapshot_degraded;
        hot.analysis_last_update_ms = inference.job.capture_time_ms;
        persist_analysis = session->last_persist_ms <= 0 ||
            inference.job.capture_time_ms - session->last_persist_ms >= 1000 ||
            !exited_tracks.empty();
        session->latest_result = analysis_result;
        session->latest_hot = hot;
    }

    for (const auto& track : exited_tracks) {
        const auto attributes = session->runtime.finalizeTrack(
            track.key.track_id);
        VehicleEventPublishContext context;
        context.task_id = inference.job.task_id;
        context.camera_profile = inference.job.camera_profile;
        context.callback_profile = inference.job.callback_profile;
        context.detector_artifact = vehicle_assets_->detector.artifact_id;
        context.attribute_artifact = vehicle_assets_->attributes.artifact_id;
        context.labels_version = vehicle_assets_->labels_version;
        context.config_version = vehicle_assets_->config_version;
        context.snapshot_relative_path = session->snapshot_relative_path;
        context.published_at_ms = wallNowMs();
        VehicleTrackResultRecord persisted;
        bool duplicate = false;
        std::string publish_error;
        if (!vehicle_event_publisher_->publish(
                track, attributes, context, persisted, duplicate,
                publish_error) && !duplicate) {
            error = publish_error;
            return false;
        }
    }

    if (persist_analysis) {
        std::string repository_error;
        const bool saved = repository_->upsertRunAnalysisResult(
            analysis_result, repository_error);
        std::lock_guard<std::mutex> lock(session->mutex);
        session->storage_degraded = !saved;
        if (saved) session->last_persist_ms = analysis_result.last_update_ms;
        hot.analysis_storage_degraded = !saved;
    }
    if (status_sink_) {
        std::string status_error;
        status_sink_->updateAnalysisStatus(hot, status_error);
    }
    ++processed_frames_;
    return true;
}

bool CameraAlgorithmProcessor::handle(
    const CameraInferenceResult& inference,
    std::string& error
) {
    error.clear();
    if (!running_.load()) {
        error = "ALGORITHM_PROCESSOR_NOT_RUNNING";
        return false;
    }
    if (!inference.output.has_vehicle_detection) {
        error = "UNSUPPORTED_NON_VEHICLE_OUTPUT";
        ++failed_frames_;
        return false;
    }
    if (!handleVehicle(inference, error)) ++failed_frames_;
    return error.empty();
}

void CameraAlgorithmProcessor::detachCamera(
    const std::string& task_id,
    const std::string& run_id
) noexcept {
    try {
        std::shared_ptr<VehicleSession> vehicle_session;
        {
            std::lock_guard<std::mutex> vehicle_lock(vehicle_sessions_mutex_);
            const auto found = vehicle_sessions_.find(task_id);
            if (found != vehicle_sessions_.end() &&
                found->second->run_id == run_id) {
                vehicle_session = found->second;
                const auto index = vehicle_session->attribute_runner_index;
                if (index < vehicle_attribute_runners_.size() &&
                    vehicle_attribute_runners_[index]->active_sessions > 0) {
                    --vehicle_attribute_runners_[index]->active_sessions;
                }
                vehicle_sessions_.erase(found);
            }
        }
        if (vehicle_session) {
            CameraRunAnalysisResultRecord final_result;
            CameraTaskRunHotStatus final_hot;
            std::vector<VehicleTrackSnapshot> exited;
            {
                std::lock_guard<std::mutex> lock(vehicle_session->mutex);
                vehicle_session->active = false;
                exited = vehicle_session->runtime.stop(wallNowMs());
                final_result = vehicle_session->latest_result;
                final_result.finalized_at_ms = wallNowMs();
                final_result.last_update_ms = std::max(
                    final_result.last_update_ms, final_result.finalized_at_ms);
                final_hot = vehicle_session->latest_hot;
                final_hot.analysis_last_update_ms = final_result.last_update_ms;
            }
            if (vehicle_event_publisher_ && vehicle_assets_) {
                for (const auto& track : exited) {
                    const auto attributes =
                        vehicle_session->runtime.finalizeTrack(track.key.track_id);
                    VehicleEventPublishContext context;
                    context.task_id = task_id;
                    context.camera_profile = vehicle_session->camera_profile;
                    context.callback_profile = vehicle_session->callback_profile;
                    context.detector_artifact =
                        vehicle_assets_->detector.artifact_id;
                    context.attribute_artifact =
                        vehicle_assets_->attributes.artifact_id;
                    context.labels_version = vehicle_assets_->labels_version;
                    context.config_version = vehicle_assets_->config_version;
                    context.snapshot_relative_path =
                        vehicle_session->snapshot_relative_path;
                    context.finalized_reason = "camera_stop";
                    context.published_at_ms = wallNowMs();
                    VehicleTrackResultRecord persisted;
                    bool duplicate = false;
                    std::string ignored;
                    vehicle_event_publisher_->publish(
                        track, attributes, context, persisted, duplicate, ignored);
                }
            }
            std::string ignored;
            if (vehicle_session->has_snapshot) {
                repository_->upsertRunAnalysisResult(final_result, ignored);
                if (status_sink_) {
                    status_sink_->updateAnalysisStatus(final_hot, ignored);
                }
            }
            return;
        }

    }
    catch (...) {
    }
}

CameraAlgorithmProcessorSnapshot CameraAlgorithmProcessor::snapshot() const {
    CameraAlgorithmProcessorSnapshot result;
    {
        std::lock_guard<std::mutex> lock(vehicle_sessions_mutex_);
        result.active_sessions = vehicle_sessions_.size();
    }
    result.processed_frames = processed_frames_.load();
    result.persisted_alerts = persisted_alerts_.load();
    result.duplicate_alerts = duplicate_alerts_.load();
    result.failed_frames = failed_frames_.load();
    if (vehicle_attribute_scheduler_) {
        result.attribute_scheduler = vehicle_attribute_scheduler_->snapshot();
    }
    if (snapshot_writer_) {
        result.snapshot_writer = snapshot_writer_->metrics();
    }
    return result;
}

}  // namespace yolo11_server
