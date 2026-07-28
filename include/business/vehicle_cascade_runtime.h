#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include "server/vehicle_model_contract.h"

namespace yolo11_server {

enum class VehicleTrackState {
    New,
    Tentative,
    Confirmed,
    AttributeCollecting,
    AttributeStable,
    Exited,
};

std::string_view toString(VehicleTrackState state) noexcept;

struct VehicleTrackKey {
    std::string camera_id;
    std::string run_id;
    std::uint64_t run_generation = 0;
    std::int64_t track_id = 0;

    bool operator<(const VehicleTrackKey& other) const noexcept;
    bool operator==(const VehicleTrackKey& other) const noexcept;
};

struct VehicleTrackerConfig {
    int min_confirm_hits = 3;
    std::int64_t track_timeout_ms = 1500;
    float match_iou_threshold = 0.30f;
};

struct VehicleTrackSnapshot {
    VehicleTrackKey key;
    VehicleTrackState state = VehicleTrackState::New;
    VehicleBox box;
    std::string vehicle_class;
    float detection_confidence = 0.0f;
    int hits = 0;
    int missed_updates = 0;
    std::int64_t first_seen_ms = 0;
    std::int64_t last_seen_ms = 0;
    std::int64_t last_frame_sequence = 0;
};

struct VehicleTrackingUpdate {
    std::vector<VehicleTrackSnapshot> active;
    std::vector<VehicleTrackSnapshot> exited;
};

struct VehicleTrackerMetrics {
    std::uint64_t frames = 0;
    std::uint64_t tracks_created = 0;
    std::uint64_t tracks_confirmed = 0;
    std::uint64_t tracks_exited = 0;
    std::uint64_t stale_results_rejected = 0;
};

class VehicleTracker {
public:
    explicit VehicleTracker(VehicleTrackerConfig config = {});

    void startRun(
        std::string camera_id,
        std::string run_id,
        std::uint64_t run_generation);
    bool update(
        const VehicleDetectionResult& result,
        VehicleTrackingUpdate& update,
        std::string& error);
    bool markAttributeCollecting(std::int64_t track_id, std::string& error);
    bool markAttributeStable(std::int64_t track_id, std::string& error);
    std::optional<VehicleTrackSnapshot> find(std::int64_t track_id) const;
    std::vector<VehicleTrackSnapshot> stop(std::int64_t stopped_at_ms);
    const VehicleTrackerMetrics& metrics() const noexcept;

private:
    VehicleTrackerConfig config_;
    std::string camera_id_;
    std::string run_id_;
    std::uint64_t run_generation_ = 0;
    std::int64_t next_track_id_ = 1;
    std::int64_t last_frame_sequence_ = -1;
    std::vector<VehicleTrackSnapshot> tracks_;
    VehicleTrackerMetrics metrics_;
};

struct CropQualityConfig {
    int min_width_px = 96;
    int min_height_px = 64;
    float min_sharpness = 0.02f;
    float min_exposure = 0.08f;
    float max_exposure = 0.95f;
    float max_occlusion = 0.50f;
    bool reject_truncated = true;
};

struct CropQualityAssessment {
    bool eligible = false;
    float quality_score = 0.0f;
    float sharpness = 0.0f;
    float exposure = 0.0f;
    int crop_width_px = 0;
    int crop_height_px = 0;
    std::string reason;
};

class VehicleCropQualityGate {
public:
    explicit VehicleCropQualityGate(CropQualityConfig config = {});

    CropQualityAssessment assess(
        const ImageView& frame,
        const VehicleBox& box,
        float occlusion_fraction,
        bool truncated) const;
    std::shared_ptr<const OwnedImage> copyCrop(
        const ImageView& frame,
        const VehicleBox& box,
        std::string& error) const;

private:
    CropQualityConfig config_;
};

struct AttributeQueueConfig {
    std::size_t max_pending = 128;
};

struct AttributeQueueMetrics {
    std::uint64_t accepted = 0;
    std::uint64_t replaced = 0;
    std::uint64_t duplicate_dropped = 0;
    std::uint64_t stale_dropped = 0;
    std::uint64_t overflow_dropped = 0;
    std::uint64_t overflow_evicted = 0;
};

class VehicleAttributeCandidateQueue {
public:
    explicit VehicleAttributeCandidateQueue(AttributeQueueConfig config = {});

    bool submit(VehicleAttributeCrop crop, std::string& error);
    std::vector<VehicleAttributeCrop> takeBatch(std::size_t max_batch);
    void clear() noexcept;
    std::size_t size() const noexcept;
    const AttributeQueueMetrics& metrics() const noexcept;

private:
    AttributeQueueConfig config_;
    std::map<VehicleTrackKey, VehicleAttributeCrop> pending_;
    AttributeQueueMetrics metrics_;
};

struct TrackAttributeAggregatorConfig {
    std::size_t min_samples = 3;
    std::size_t max_observations = 5;
    float body_type_threshold = 0.75f;
    float color_threshold = 0.70f;
};

struct FusedAttribute {
    std::string label = "unknown";
    float confidence = 0.0f;
    bool stable = false;
    std::size_t samples_used = 0;
};

struct VehicleTrackAttributeSnapshot {
    VehicleTrackKey key;
    std::string artifact_id;
    std::string labels_version;
    FusedAttribute body_type;
    FusedAttribute color;
    std::size_t observation_count = 0;

    bool stable() const noexcept {
        return body_type.stable && color.stable;
    }
};

struct TrackAttributeAggregatorMetrics {
    std::uint64_t accepted = 0;
    std::uint64_t stale_rejected = 0;
    std::uint64_t version_rejected = 0;
    std::uint64_t finalized = 0;
};

class VehicleTrackAttributeAggregator {
public:
    explicit VehicleTrackAttributeAggregator(
        TrackAttributeAggregatorConfig config = {});

    bool add(const VehicleAttributeResult& result, std::string& error);
    std::optional<VehicleTrackAttributeSnapshot> snapshot(
        const VehicleTrackKey& key) const;
    std::optional<VehicleTrackAttributeSnapshot> finalize(
        const VehicleTrackKey& key);
    void clear() noexcept;
    const TrackAttributeAggregatorMetrics& metrics() const noexcept;

private:
    struct TrackObservations {
        std::string artifact_id;
        std::string labels_version;
        std::uint64_t last_crop_sequence = 0;
        std::vector<VehicleAttributeResult> items;
    };

    TrackAttributeAggregatorConfig config_;
    std::map<VehicleTrackKey, TrackObservations> observations_;
    TrackAttributeAggregatorMetrics metrics_;
};

struct VehicleCascadeRuntimeConfig {
    VehicleTrackerConfig tracker;
    CropQualityConfig crop_quality;
    AttributeQueueConfig attribute_queue;
    TrackAttributeAggregatorConfig aggregator;
};

class VehicleCascadeRuntime {
public:
    explicit VehicleCascadeRuntime(VehicleCascadeRuntimeConfig config = {});

    void startRun(
        std::string camera_id,
        std::string run_id,
        std::uint64_t run_generation);
    bool onDetections(
        const VehicleDetectionResult& result,
        VehicleTrackingUpdate& update,
        std::string& error);
    bool queueCrop(
        const ImageView& frame,
        std::int64_t track_id,
        std::uint64_t crop_sequence,
        float occlusion_fraction,
        bool truncated,
        CropQualityAssessment& assessment,
        std::string& error);
    std::vector<VehicleAttributeCrop> takeAttributeBatch(std::size_t max_batch);
    bool onAttributeResults(
        const std::vector<VehicleAttributeResult>& results,
        std::string& error);
    std::optional<VehicleTrackSnapshot> trackSnapshot(
        std::int64_t track_id) const;
    std::optional<VehicleTrackAttributeSnapshot> attributeSnapshot(
        std::int64_t track_id) const;
    std::optional<VehicleTrackAttributeSnapshot> finalizeTrack(
        std::int64_t track_id);
    std::vector<VehicleTrackSnapshot> stop(std::int64_t stopped_at_ms);
    const VehicleTrackerMetrics& trackerMetrics() const noexcept;
    const AttributeQueueMetrics& attributeQueueMetrics() const noexcept;
    const TrackAttributeAggregatorMetrics& aggregatorMetrics() const noexcept;

private:
    VehicleTrackKey keyFor(std::int64_t track_id) const;

    std::string camera_id_;
    std::string run_id_;
    std::uint64_t run_generation_ = 0;
    VehicleTracker tracker_;
    VehicleCropQualityGate crop_quality_;
    VehicleAttributeCandidateQueue attribute_queue_;
    VehicleTrackAttributeAggregator aggregator_;
};

}  // namespace yolo11_server
