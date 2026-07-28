#include "business/vehicle_cascade_runtime.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <tuple>
#include <utility>

namespace yolo11_server {

namespace {

float clamp01(float value) {
    return std::max(0.0f, std::min(1.0f, value));
}

float intersectionOverUnion(const VehicleBox& first, const VehicleBox& second) {
    const float left = std::max(first.x1, second.x1);
    const float top = std::max(first.y1, second.y1);
    const float right = std::min(first.x2, second.x2);
    const float bottom = std::min(first.y2, second.y2);
    const float width = std::max(0.0f, right - left);
    const float height = std::max(0.0f, bottom - top);
    const float intersection = width * height;
    const float first_area = (first.x2 - first.x1) * (first.y2 - first.y1);
    const float second_area = (second.x2 - second.x1) * (second.y2 - second.y1);
    const float denominator = first_area + second_area - intersection;
    return denominator > 0.0f ? intersection / denominator : 0.0f;
}

VehicleTrackKey keyFromResult(const VehicleAttributeResult& result) {
    return {
        result.camera_id,
        result.run_id,
        result.metadata.run_generation,
        result.track_id,
    };
}

bool validPrediction(const AttributePrediction& value) {
    return !value.label.empty() &&
        value.confidence >= 0.0f &&
        value.confidence <= 1.0f;
}

FusedAttribute fuseHead(
    const std::vector<VehicleAttributeResult>& items,
    bool body_type,
    const TrackAttributeAggregatorConfig& config) {
    std::map<std::string, float> weights;
    std::map<std::string, std::size_t> counts;
    float total_quality = 0.0f;
    for (const auto& item : items) {
        const auto& prediction = body_type ? item.body_type : item.color;
        const float quality = clamp01(item.quality_score);
        total_quality += quality;
        if (prediction.label == "unknown") continue;
        weights[prediction.label] += quality * clamp01(prediction.confidence);
        counts[prediction.label] += 1;
    }

    FusedAttribute fused;
    if (weights.empty() || total_quality <= 0.0f) return fused;
    const auto best = std::max_element(
        weights.begin(),
        weights.end(),
        [](const auto& left, const auto& right) {
            if (left.second != right.second) return left.second < right.second;
            return left.first > right.first;
        });
    fused.confidence = clamp01(best->second / total_quality);
    fused.samples_used = counts[best->first];
    const float threshold =
        body_type ? config.body_type_threshold : config.color_threshold;
    fused.stable =
        fused.samples_used >= config.min_samples &&
        fused.confidence >= threshold;
    if (fused.stable) fused.label = best->first;
    return fused;
}

}  // namespace

std::string_view toString(VehicleTrackState state) noexcept {
    switch (state) {
    case VehicleTrackState::New: return "NEW";
    case VehicleTrackState::Tentative: return "TENTATIVE";
    case VehicleTrackState::Confirmed: return "CONFIRMED";
    case VehicleTrackState::AttributeCollecting: return "ATTR_COLLECTING";
    case VehicleTrackState::AttributeStable: return "ATTR_STABLE";
    case VehicleTrackState::Exited: return "EXITED";
    }
    return "EXITED";
}

bool VehicleTrackKey::operator<(const VehicleTrackKey& other) const noexcept {
    return std::tie(camera_id, run_id, run_generation, track_id) <
        std::tie(other.camera_id, other.run_id, other.run_generation, other.track_id);
}

bool VehicleTrackKey::operator==(const VehicleTrackKey& other) const noexcept {
    return camera_id == other.camera_id &&
        run_id == other.run_id &&
        run_generation == other.run_generation &&
        track_id == other.track_id;
}

VehicleTracker::VehicleTracker(VehicleTrackerConfig config)
    : config_(config) {
    config_.min_confirm_hits = std::max(1, config_.min_confirm_hits);
    config_.track_timeout_ms = std::max<std::int64_t>(1, config_.track_timeout_ms);
    config_.match_iou_threshold = clamp01(config_.match_iou_threshold);
}

void VehicleTracker::startRun(
    std::string camera_id,
    std::string run_id,
    std::uint64_t run_generation) {
    camera_id_ = std::move(camera_id);
    run_id_ = std::move(run_id);
    run_generation_ = run_generation;
    next_track_id_ = 1;
    last_frame_sequence_ = -1;
    tracks_.clear();
    metrics_ = {};
}

bool VehicleTracker::update(
    const VehicleDetectionResult& result,
    VehicleTrackingUpdate& update,
    std::string& error) {
    error.clear();
    update = {};
    if (camera_id_.empty() || run_id_.empty()) {
        error = "tracker Run is not initialized";
        return false;
    }
    if (result.camera_id != camera_id_ ||
        result.run_id != run_id_ ||
        result.metadata.run_generation != run_generation_) {
        metrics_.stale_results_rejected += 1;
        error = "stale or foreign detection result";
        return false;
    }
    if (result.frame_sequence <= last_frame_sequence_) {
        metrics_.stale_results_rejected += 1;
        error = "detection frame sequence must be strictly increasing";
        return false;
    }
    for (const auto& detection : result.detections) {
        if (!detection.box.valid() ||
            detection.vehicle_class.empty() ||
            detection.confidence < 0.0f ||
            detection.confidence > 1.0f) {
            error = "detection result contains an invalid vehicle";
            return false;
        }
    }

    struct Candidate {
        std::size_t track_index = 0;
        std::size_t detection_index = 0;
        float iou = 0.0f;
    };
    std::vector<Candidate> candidates;
    for (std::size_t ti = 0; ti < tracks_.size(); ++ti) {
        for (std::size_t di = 0; di < result.detections.size(); ++di) {
            const float iou = intersectionOverUnion(
                tracks_[ti].box,
                result.detections[di].box);
            if (iou >= config_.match_iou_threshold) {
                candidates.push_back({ti, di, iou});
            }
        }
    }
    std::sort(
        candidates.begin(),
        candidates.end(),
        [](const Candidate& left, const Candidate& right) {
            return left.iou > right.iou;
        });

    std::vector<bool> track_matched(tracks_.size(), false);
    std::vector<bool> detection_matched(result.detections.size(), false);
    for (const auto& candidate : candidates) {
        if (track_matched[candidate.track_index] ||
            detection_matched[candidate.detection_index]) {
            continue;
        }
        auto& track = tracks_[candidate.track_index];
        const auto& detection = result.detections[candidate.detection_index];
        track.box = detection.box;
        track.vehicle_class = detection.vehicle_class;
        track.detection_confidence = detection.confidence;
        track.hits += 1;
        track.missed_updates = 0;
        track.last_seen_ms = result.captured_at_ms;
        track.last_frame_sequence = result.frame_sequence;
        if (track.state == VehicleTrackState::Tentative &&
            track.hits >= config_.min_confirm_hits) {
            track.state = VehicleTrackState::Confirmed;
            metrics_.tracks_confirmed += 1;
        }
        track_matched[candidate.track_index] = true;
        detection_matched[candidate.detection_index] = true;
    }

    for (std::size_t index = 0; index < tracks_.size(); ++index) {
        if (!track_matched[index]) tracks_[index].missed_updates += 1;
    }
    for (std::size_t index = 0; index < result.detections.size(); ++index) {
        if (detection_matched[index]) continue;
        const auto& detection = result.detections[index];
        VehicleTrackSnapshot track;
        track.key = {camera_id_, run_id_, run_generation_, next_track_id_++};
        track.state = config_.min_confirm_hits == 1
            ? VehicleTrackState::Confirmed
            : VehicleTrackState::Tentative;
        track.box = detection.box;
        track.vehicle_class = detection.vehicle_class;
        track.detection_confidence = detection.confidence;
        track.hits = 1;
        track.first_seen_ms = result.captured_at_ms;
        track.last_seen_ms = result.captured_at_ms;
        track.last_frame_sequence = result.frame_sequence;
        tracks_.push_back(track);
        metrics_.tracks_created += 1;
        if (track.state == VehicleTrackState::Confirmed) {
            metrics_.tracks_confirmed += 1;
        }
    }

    std::vector<VehicleTrackSnapshot> retained;
    retained.reserve(tracks_.size());
    for (auto track : tracks_) {
        if (result.captured_at_ms - track.last_seen_ms >= config_.track_timeout_ms) {
            track.state = VehicleTrackState::Exited;
            update.exited.push_back(track);
            metrics_.tracks_exited += 1;
        }
        else {
            retained.push_back(track);
            update.active.push_back(track);
        }
    }
    tracks_ = std::move(retained);
    last_frame_sequence_ = result.frame_sequence;
    metrics_.frames += 1;
    return true;
}

bool VehicleTracker::markAttributeCollecting(
    std::int64_t track_id,
    std::string& error) {
    for (auto& track : tracks_) {
        if (track.key.track_id != track_id) continue;
        if (track.state == VehicleTrackState::Confirmed ||
            track.state == VehicleTrackState::AttributeCollecting) {
            track.state = VehicleTrackState::AttributeCollecting;
            error.clear();
            return true;
        }
        error = "track must be confirmed before attribute collection";
        return false;
    }
    error = "track_id is not active";
    return false;
}

bool VehicleTracker::markAttributeStable(
    std::int64_t track_id,
    std::string& error) {
    for (auto& track : tracks_) {
        if (track.key.track_id != track_id) continue;
        if (track.state == VehicleTrackState::AttributeCollecting ||
            track.state == VehicleTrackState::AttributeStable) {
            track.state = VehicleTrackState::AttributeStable;
            error.clear();
            return true;
        }
        error = "track must collect attributes before becoming stable";
        return false;
    }
    error = "track_id is not active";
    return false;
}

std::optional<VehicleTrackSnapshot> VehicleTracker::find(
    std::int64_t track_id) const {
    const auto it = std::find_if(
        tracks_.begin(),
        tracks_.end(),
        [track_id](const VehicleTrackSnapshot& track) {
            return track.key.track_id == track_id;
        });
    if (it == tracks_.end()) return std::nullopt;
    return *it;
}

std::vector<VehicleTrackSnapshot> VehicleTracker::stop(
    std::int64_t stopped_at_ms) {
    std::vector<VehicleTrackSnapshot> exited;
    exited.reserve(tracks_.size());
    for (auto track : tracks_) {
        track.state = VehicleTrackState::Exited;
        track.last_seen_ms = std::max(track.last_seen_ms, stopped_at_ms);
        exited.push_back(std::move(track));
        metrics_.tracks_exited += 1;
    }
    tracks_.clear();
    return exited;
}

const VehicleTrackerMetrics& VehicleTracker::metrics() const noexcept {
    return metrics_;
}

VehicleCropQualityGate::VehicleCropQualityGate(CropQualityConfig config)
    : config_(config) {
    config_.min_width_px = std::max(1, config_.min_width_px);
    config_.min_height_px = std::max(1, config_.min_height_px);
    config_.min_sharpness = clamp01(config_.min_sharpness);
    config_.min_exposure = clamp01(config_.min_exposure);
    config_.max_exposure = clamp01(config_.max_exposure);
    config_.max_occlusion = clamp01(config_.max_occlusion);
}

CropQualityAssessment VehicleCropQualityGate::assess(
    const ImageView& frame,
    const VehicleBox& box,
    float occlusion_fraction,
    bool truncated) const {
    CropQualityAssessment result;
    if (!frame.valid() || !box.valid()) {
        result.reason = "invalid_input";
        return result;
    }
    occlusion_fraction = clamp01(occlusion_fraction);
    const int left = std::max(
        0,
        static_cast<int>(std::floor(box.x1 * static_cast<float>(frame.width))));
    const int top = std::max(
        0,
        static_cast<int>(std::floor(box.y1 * static_cast<float>(frame.height))));
    const int right = std::min(
        frame.width,
        static_cast<int>(std::ceil(box.x2 * static_cast<float>(frame.width))));
    const int bottom = std::min(
        frame.height,
        static_cast<int>(std::ceil(box.y2 * static_cast<float>(frame.height))));
    result.crop_width_px = right - left;
    result.crop_height_px = bottom - top;
    if (result.crop_width_px <= 0 || result.crop_height_px <= 0) {
        result.reason = "empty_crop";
        return result;
    }

    double intensity_sum = 0.0;
    double gradient_sum = 0.0;
    std::size_t intensity_count = 0;
    std::size_t gradient_count = 0;
    for (int y = top; y < bottom; ++y) {
        const auto* row = frame.data +
            static_cast<std::size_t>(y) * frame.row_stride_bytes;
        for (int x = left; x < right; ++x) {
            const auto offset =
                static_cast<std::size_t>(x) * static_cast<std::size_t>(frame.channels);
            float current = 0.0f;
            for (int channel = 0; channel < frame.channels; ++channel) {
                current += static_cast<float>(row[offset + channel]);
            }
            current /= static_cast<float>(frame.channels);
            intensity_sum += current;
            intensity_count += 1;
            if (x > left) {
                const auto previous_offset = offset -
                    static_cast<std::size_t>(frame.channels);
                float previous = 0.0f;
                for (int channel = 0; channel < frame.channels; ++channel) {
                    previous += static_cast<float>(row[previous_offset + channel]);
                }
                previous /= static_cast<float>(frame.channels);
                gradient_sum += std::abs(current - previous);
                gradient_count += 1;
            }
        }
    }
    result.exposure = intensity_count == 0
        ? 0.0f
        : static_cast<float>(intensity_sum / intensity_count / 255.0);
    result.sharpness = gradient_count == 0
        ? 0.0f
        : static_cast<float>(gradient_sum / gradient_count / 255.0);

    const float width_score = std::min(
        1.0f,
        static_cast<float>(result.crop_width_px) /
            static_cast<float>(config_.min_width_px));
    const float height_score = std::min(
        1.0f,
        static_cast<float>(result.crop_height_px) /
            static_cast<float>(config_.min_height_px));
    const float size_score = std::min(width_score, height_score);
    const float sharpness_score = config_.min_sharpness <= 0.0f
        ? 1.0f
        : std::min(1.0f, result.sharpness / config_.min_sharpness);
    const float exposure_score =
        clamp01(1.0f - std::abs(result.exposure - 0.5f) / 0.5f);
    const float visibility_score = 1.0f - occlusion_fraction;
    const float truncation_score = truncated ? 0.0f : 1.0f;
    result.quality_score = clamp01(
        0.25f * size_score +
        0.25f * sharpness_score +
        0.20f * exposure_score +
        0.20f * visibility_score +
        0.10f * truncation_score);

    if (result.crop_width_px < config_.min_width_px ||
        result.crop_height_px < config_.min_height_px) {
        result.reason = "crop_too_small";
    }
    else if (result.sharpness < config_.min_sharpness) {
        result.reason = "crop_too_blurry";
    }
    else if (result.exposure < config_.min_exposure ||
        result.exposure > config_.max_exposure) {
        result.reason = "exposure_out_of_range";
    }
    else if (occlusion_fraction > config_.max_occlusion) {
        result.reason = "too_occluded";
    }
    else if (truncated && config_.reject_truncated) {
        result.reason = "truncated";
    }
    else {
        result.eligible = true;
        result.reason = "accepted";
    }
    return result;
}

std::shared_ptr<const OwnedImage> VehicleCropQualityGate::copyCrop(
    const ImageView& frame,
    const VehicleBox& box,
    std::string& error) const {
    error.clear();
    if (!frame.valid() || !box.valid()) {
        error = "cannot copy invalid frame or box";
        return nullptr;
    }
    const int left = std::max(
        0,
        static_cast<int>(std::floor(box.x1 * static_cast<float>(frame.width))));
    const int top = std::max(
        0,
        static_cast<int>(std::floor(box.y1 * static_cast<float>(frame.height))));
    const int right = std::min(
        frame.width,
        static_cast<int>(std::ceil(box.x2 * static_cast<float>(frame.width))));
    const int bottom = std::min(
        frame.height,
        static_cast<int>(std::ceil(box.y2 * static_cast<float>(frame.height))));
    const int width = right - left;
    const int height = bottom - top;
    if (width <= 0 || height <= 0) {
        error = "crop is empty";
        return nullptr;
    }
    auto crop = std::make_shared<OwnedImage>();
    crop->width = width;
    crop->height = height;
    crop->channels = frame.channels;
    crop->pixel_format = frame.pixel_format;
    crop->row_stride_bytes =
        static_cast<std::size_t>(width) * static_cast<std::size_t>(frame.channels);
    crop->pixels.resize(
        crop->row_stride_bytes * static_cast<std::size_t>(height));
    for (int y = 0; y < height; ++y) {
        const auto* source = frame.data +
            static_cast<std::size_t>(top + y) * frame.row_stride_bytes +
            static_cast<std::size_t>(left) *
                static_cast<std::size_t>(frame.channels);
        auto* destination = crop->pixels.data() +
            static_cast<std::size_t>(y) * crop->row_stride_bytes;
        std::copy(
            source,
            source + crop->row_stride_bytes,
            destination);
    }
    if (!crop->valid()) {
        error = "owned crop failed validation";
        return nullptr;
    }
    return crop;
}

VehicleAttributeCandidateQueue::VehicleAttributeCandidateQueue(
    AttributeQueueConfig config)
    : config_(config) {
    config_.max_pending = std::max<std::size_t>(1, config_.max_pending);
}

bool VehicleAttributeCandidateQueue::submit(
    VehicleAttributeCrop crop,
    std::string& error) {
    error.clear();
    if (!crop.crop || !crop.crop->valid() ||
        crop.camera_id.empty() ||
        crop.run_id.empty() ||
        crop.track_id <= 0 ||
        crop.crop_sequence == 0 ||
        crop.quality_score < 0.0f ||
        crop.quality_score > 1.0f) {
        error = "attribute crop is invalid";
        return false;
    }
    VehicleTrackKey key{
        crop.camera_id,
        crop.run_id,
        crop.run_generation,
        crop.track_id,
    };
    const auto existing = pending_.find(key);
    if (existing != pending_.end()) {
        if (crop.crop_sequence <= existing->second.crop_sequence) {
            metrics_.stale_dropped += 1;
            error = "stale crop sequence";
            return false;
        }
        if (crop.quality_score <= existing->second.quality_score) {
            metrics_.duplicate_dropped += 1;
            error = "pending track already has an equal or better crop";
            return false;
        }
        existing->second = std::move(crop);
        metrics_.replaced += 1;
        return true;
    }

    if (pending_.size() >= config_.max_pending) {
        const auto lowest = std::min_element(
            pending_.begin(),
            pending_.end(),
            [](const auto& left, const auto& right) {
                return left.second.quality_score < right.second.quality_score;
            });
        if (lowest != pending_.end() &&
            crop.quality_score <= lowest->second.quality_score) {
            metrics_.overflow_dropped += 1;
            error = "attribute queue is full and crop quality is not competitive";
            return false;
        }
        if (lowest != pending_.end()) {
            pending_.erase(lowest);
            metrics_.overflow_evicted += 1;
        }
    }
    pending_.emplace(std::move(key), std::move(crop));
    metrics_.accepted += 1;
    return true;
}

std::vector<VehicleAttributeCrop> VehicleAttributeCandidateQueue::takeBatch(
    std::size_t max_batch) {
    std::vector<std::pair<VehicleTrackKey, VehicleAttributeCrop>> ordered(
        pending_.begin(),
        pending_.end());
    std::sort(
        ordered.begin(),
        ordered.end(),
        [](const auto& left, const auto& right) {
            if (left.second.quality_score != right.second.quality_score) {
                return left.second.quality_score > right.second.quality_score;
            }
            return left.second.crop_sequence < right.second.crop_sequence;
        });
    const auto count = std::min(max_batch, ordered.size());
    std::vector<VehicleAttributeCrop> batch;
    batch.reserve(count);
    for (std::size_t index = 0; index < count; ++index) {
        batch.push_back(std::move(ordered[index].second));
        pending_.erase(ordered[index].first);
    }
    return batch;
}

void VehicleAttributeCandidateQueue::clear() noexcept {
    pending_.clear();
    metrics_ = {};
}

std::size_t VehicleAttributeCandidateQueue::size() const noexcept {
    return pending_.size();
}

const AttributeQueueMetrics& VehicleAttributeCandidateQueue::metrics() const noexcept {
    return metrics_;
}

VehicleTrackAttributeAggregator::VehicleTrackAttributeAggregator(
    TrackAttributeAggregatorConfig config)
    : config_(config) {
    config_.min_samples = std::max<std::size_t>(1, config_.min_samples);
    config_.max_observations = std::max(
        config_.min_samples,
        config_.max_observations);
    config_.body_type_threshold = clamp01(config_.body_type_threshold);
    config_.color_threshold = clamp01(config_.color_threshold);
}

bool VehicleTrackAttributeAggregator::add(
    const VehicleAttributeResult& result,
    std::string& error) {
    error.clear();
    if (result.camera_id.empty() ||
        result.run_id.empty() ||
        result.track_id <= 0 ||
        result.crop_sequence == 0 ||
        result.metadata.artifact_id.empty() ||
        result.metadata.labels_version.empty() ||
        result.quality_score < 0.0f ||
        result.quality_score > 1.0f ||
        !validPrediction(result.body_type) ||
        !validPrediction(result.color)) {
        error = "attribute result is invalid";
        return false;
    }
    const auto key = keyFromResult(result);
    auto& track = observations_[key];
    if (track.items.empty()) {
        track.artifact_id = result.metadata.artifact_id;
        track.labels_version = result.metadata.labels_version;
    }
    else if (track.artifact_id != result.metadata.artifact_id ||
        track.labels_version != result.metadata.labels_version) {
        metrics_.version_rejected += 1;
        error = "attribute model or labels version changed within a track";
        return false;
    }
    if (result.crop_sequence <= track.last_crop_sequence) {
        metrics_.stale_rejected += 1;
        error = "attribute crop sequence must be strictly increasing";
        return false;
    }
    track.last_crop_sequence = result.crop_sequence;
    track.items.push_back(result);
    std::sort(
        track.items.begin(),
        track.items.end(),
        [](const VehicleAttributeResult& left, const VehicleAttributeResult& right) {
            if (left.quality_score != right.quality_score) {
                return left.quality_score > right.quality_score;
            }
            return left.crop_sequence > right.crop_sequence;
        });
    if (track.items.size() > config_.max_observations) {
        track.items.resize(config_.max_observations);
    }
    metrics_.accepted += 1;
    return true;
}

std::optional<VehicleTrackAttributeSnapshot>
VehicleTrackAttributeAggregator::snapshot(const VehicleTrackKey& key) const {
    const auto it = observations_.find(key);
    if (it == observations_.end()) return std::nullopt;
    VehicleTrackAttributeSnapshot result;
    result.key = key;
    result.artifact_id = it->second.artifact_id;
    result.labels_version = it->second.labels_version;
    result.observation_count = it->second.items.size();
    result.body_type = fuseHead(it->second.items, true, config_);
    result.color = fuseHead(it->second.items, false, config_);
    return result;
}

std::optional<VehicleTrackAttributeSnapshot>
VehicleTrackAttributeAggregator::finalize(const VehicleTrackKey& key) {
    auto result = snapshot(key);
    if (!result) return std::nullopt;
    observations_.erase(key);
    metrics_.finalized += 1;
    return result;
}

void VehicleTrackAttributeAggregator::clear() noexcept {
    observations_.clear();
    metrics_ = {};
}

const TrackAttributeAggregatorMetrics&
VehicleTrackAttributeAggregator::metrics() const noexcept {
    return metrics_;
}

VehicleCascadeRuntime::VehicleCascadeRuntime(VehicleCascadeRuntimeConfig config)
    : tracker_(config.tracker),
      crop_quality_(config.crop_quality),
      attribute_queue_(config.attribute_queue),
      aggregator_(config.aggregator) {
}

void VehicleCascadeRuntime::startRun(
    std::string camera_id,
    std::string run_id,
    std::uint64_t run_generation) {
    camera_id_ = std::move(camera_id);
    run_id_ = std::move(run_id);
    run_generation_ = run_generation;
    tracker_.startRun(camera_id_, run_id_, run_generation_);
    attribute_queue_.clear();
    aggregator_.clear();
}

bool VehicleCascadeRuntime::onDetections(
    const VehicleDetectionResult& result,
    VehicleTrackingUpdate& update,
    std::string& error) {
    return tracker_.update(result, update, error);
}

bool VehicleCascadeRuntime::queueCrop(
    const ImageView& frame,
    std::int64_t track_id,
    std::uint64_t crop_sequence,
    float occlusion_fraction,
    bool truncated,
    CropQualityAssessment& assessment,
    std::string& error) {
    const auto track = tracker_.find(track_id);
    if (!track) {
        error = "track_id is not active";
        return false;
    }
    if (track->state != VehicleTrackState::Confirmed &&
        track->state != VehicleTrackState::AttributeCollecting) {
        error = "track is not eligible for attribute collection";
        return false;
    }
    assessment = crop_quality_.assess(
        frame,
        track->box,
        occlusion_fraction,
        truncated);
    if (!assessment.eligible) {
        error = "crop rejected: " + assessment.reason;
        return false;
    }
    auto crop_image = crop_quality_.copyCrop(frame, track->box, error);
    if (!crop_image) return false;
    VehicleAttributeCrop crop;
    crop.crop = std::move(crop_image);
    crop.camera_id = camera_id_;
    crop.run_id = run_id_;
    crop.run_generation = run_generation_;
    crop.track_id = track_id;
    crop.crop_sequence = crop_sequence;
    crop.quality_score = assessment.quality_score;
    if (!attribute_queue_.submit(std::move(crop), error)) return false;
    return tracker_.markAttributeCollecting(track_id, error);
}

std::vector<VehicleAttributeCrop> VehicleCascadeRuntime::takeAttributeBatch(
    std::size_t max_batch) {
    return attribute_queue_.takeBatch(max_batch);
}

bool VehicleCascadeRuntime::onAttributeResults(
    const std::vector<VehicleAttributeResult>& results,
    std::string& error) {
    for (const auto& result : results) {
        if (result.camera_id != camera_id_ ||
            result.run_id != run_id_ ||
            result.metadata.run_generation != run_generation_) {
            error = "stale or foreign attribute result";
            return false;
        }
        if (!aggregator_.add(result, error)) return false;
        const auto snapshot = aggregator_.snapshot(keyFromResult(result));
        if (snapshot && snapshot->stable()) {
            if (!tracker_.markAttributeStable(result.track_id, error)) return false;
        }
    }
    error.clear();
    return true;
}

std::optional<VehicleTrackSnapshot>
VehicleCascadeRuntime::trackSnapshot(std::int64_t track_id) const {
    return tracker_.find(track_id);
}

std::optional<VehicleTrackAttributeSnapshot>
VehicleCascadeRuntime::attributeSnapshot(std::int64_t track_id) const {
    return aggregator_.snapshot(keyFor(track_id));
}

std::optional<VehicleTrackAttributeSnapshot>
VehicleCascadeRuntime::finalizeTrack(std::int64_t track_id) {
    return aggregator_.finalize(keyFor(track_id));
}

std::vector<VehicleTrackSnapshot> VehicleCascadeRuntime::stop(
    std::int64_t stopped_at_ms) {
    attribute_queue_.clear();
    return tracker_.stop(stopped_at_ms);
}

const VehicleTrackerMetrics&
VehicleCascadeRuntime::trackerMetrics() const noexcept {
    return tracker_.metrics();
}

const AttributeQueueMetrics&
VehicleCascadeRuntime::attributeQueueMetrics() const noexcept {
    return attribute_queue_.metrics();
}

const TrackAttributeAggregatorMetrics&
VehicleCascadeRuntime::aggregatorMetrics() const noexcept {
    return aggregator_.metrics();
}

VehicleTrackKey VehicleCascadeRuntime::keyFor(
    std::int64_t track_id) const {
    return {camera_id_, run_id_, run_generation_, track_id};
}

}  // namespace yolo11_server
