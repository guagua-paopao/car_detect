#include "business/vehicle_cascade_runtime.h"

#include <cassert>
#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

using namespace yolo11_server;

namespace {

std::shared_ptr<OwnedImage> checkerImage(int width, int height) {
    auto image = std::make_shared<OwnedImage>();
    image->width = width;
    image->height = height;
    image->channels = 3;
    image->row_stride_bytes = static_cast<std::size_t>(width) * 3;
    image->pixels.resize(image->row_stride_bytes * static_cast<std::size_t>(height));
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const std::uint8_t value = (x + y) % 2 == 0 ? 32 : 224;
            const auto offset =
                static_cast<std::size_t>(y) * image->row_stride_bytes +
                static_cast<std::size_t>(x) * 3;
            image->pixels[offset] = value;
            image->pixels[offset + 1] = value;
            image->pixels[offset + 2] = value;
        }
    }
    assert(image->valid());
    return image;
}

VehicleDetectionResult detectionFrame(
    std::uint64_t generation,
    std::int64_t sequence,
    std::int64_t timestamp,
    bool with_vehicle = true) {
    VehicleDetectionResult result;
    result.metadata.artifact_id = "vehicle-det-v1";
    result.metadata.labels_version = "vehicle-labels-v1";
    result.metadata.run_generation = generation;
    result.camera_id = "gate_01";
    result.run_id = "run_01";
    result.frame_sequence = sequence;
    result.captured_at_ms = timestamp;
    if (with_vehicle) {
        result.detections.push_back({
            {0.10f, 0.10f, 0.90f, 0.90f},
            "car",
            0.96f,
        });
    }
    return result;
}

VehicleAttributeCrop crop(
    const std::shared_ptr<const OwnedImage>& image,
    std::int64_t track_id,
    std::uint64_t sequence,
    float quality) {
    VehicleAttributeCrop result;
    result.crop = image;
    result.camera_id = "gate_01";
    result.run_id = "run_01";
    result.run_generation = 7;
    result.track_id = track_id;
    result.crop_sequence = sequence;
    result.quality_score = quality;
    return result;
}

VehicleAttributeResult attributeResult(
    std::int64_t track_id,
    std::uint64_t sequence,
    float quality,
    std::string body_type,
    std::string color) {
    VehicleAttributeResult result;
    result.metadata.artifact_id = "vehicle-attr-v1";
    result.metadata.labels_version = "vehicle-labels-v1";
    result.metadata.run_generation = 7;
    result.camera_id = "gate_01";
    result.run_id = "run_01";
    result.track_id = track_id;
    result.crop_sequence = sequence;
    result.quality_score = quality;
    result.body_type = {std::move(body_type), 0.95f};
    result.color = {std::move(color), 0.95f};
    return result;
}

void testTracker() {
    VehicleTracker tracker({2, 100, 0.30f});
    tracker.startRun("gate_01", "run_01", 7);
    VehicleTrackingUpdate update;
    std::string error;

    assert(tracker.update(detectionFrame(7, 1, 0), update, error));
    assert(update.active.size() == 1);
    assert(update.active.front().state == VehicleTrackState::Tentative);
    const auto track_id = update.active.front().key.track_id;

    assert(tracker.update(detectionFrame(7, 2, 10), update, error));
    assert(update.active.front().key.track_id == track_id);
    assert(update.active.front().state == VehicleTrackState::Confirmed);

    assert(!tracker.update(detectionFrame(6, 3, 20), update, error));
    assert(error.find("stale") != std::string::npos);
    assert(tracker.metrics().stale_results_rejected == 1);

    assert(tracker.update(detectionFrame(7, 3, 200, false), update, error));
    assert(update.active.empty());
    assert(update.exited.size() == 1);
    assert(update.exited.front().state == VehicleTrackState::Exited);
}

void testQualityGate() {
    auto image = checkerImage(200, 120);
    VehicleCropQualityGate gate({96, 64, 0.02f, 0.08f, 0.95f, 0.50f, true});
    const VehicleBox good_box{0.10f, 0.10f, 0.90f, 0.90f};
    const auto accepted = gate.assess(image->view(), good_box, 0.10f, false);
    assert(accepted.eligible);
    assert(accepted.reason == "accepted");
    assert(accepted.crop_width_px == 160);
    assert(accepted.crop_height_px == 96);

    std::string error;
    const auto copied = gate.copyCrop(image->view(), good_box, error);
    assert(copied && copied->valid());
    assert(copied->width == 160 && copied->height == 96);

    const auto small = gate.assess(
        image->view(),
        {0.10f, 0.10f, 0.30f, 0.40f},
        0.0f,
        false);
    assert(!small.eligible);
    assert(small.reason == "crop_too_small");

    auto dark = std::make_shared<OwnedImage>();
    dark->width = 200;
    dark->height = 120;
    dark->channels = 3;
    dark->row_stride_bytes = 600;
    dark->pixels.resize(600 * 120);
    assert(!gate.assess(dark->view(), good_box, 0.0f, false).eligible);
    assert(!gate.assess(image->view(), good_box, 0.0f, true).eligible);
}

void testAttributeQueue() {
    const auto image = checkerImage(160, 96);
    VehicleAttributeCandidateQueue queue({2});
    std::string error;

    assert(queue.submit(crop(image, 1, 1, 0.40f), error));
    assert(queue.submit(crop(image, 1, 2, 0.80f), error));
    assert(queue.metrics().replaced == 1);
    assert(!queue.submit(crop(image, 1, 1, 0.90f), error));
    assert(queue.metrics().stale_dropped == 1);
    assert(queue.submit(crop(image, 2, 1, 0.50f), error));
    assert(queue.submit(crop(image, 3, 1, 0.90f), error));
    assert(queue.metrics().overflow_evicted == 1);
    assert(queue.size() == 2);

    const auto batch = queue.takeBatch(2);
    assert(batch.size() == 2);
    assert(batch[0].track_id == 3);
    assert(batch[1].track_id == 1);
    assert(queue.size() == 0);
}

void testIndependentFusion() {
    VehicleTrackAttributeAggregator aggregator({3, 5, 0.75f, 0.70f});
    std::string error;
    const VehicleTrackKey key{"gate_01", "run_01", 7, 42};

    assert(aggregator.add(attributeResult(42, 1, 1.0f, "suv", "black"), error));
    assert(aggregator.add(attributeResult(42, 2, 1.0f, "suv", "white"), error));
    assert(aggregator.add(attributeResult(42, 3, 1.0f, "suv", "white"), error));
    auto snapshot = aggregator.snapshot(key);
    assert(snapshot);
    assert(snapshot->body_type.stable);
    assert(snapshot->body_type.label == "suv");
    assert(!snapshot->color.stable);
    assert(snapshot->color.label == "unknown");

    assert(aggregator.add(attributeResult(42, 4, 1.0f, "suv", "white"), error));
    snapshot = aggregator.snapshot(key);
    assert(snapshot && snapshot->stable());
    assert(snapshot->color.label == "white");

    auto stale = attributeResult(42, 4, 1.0f, "suv", "white");
    assert(!aggregator.add(stale, error));
    assert(error.find("strictly increasing") != std::string::npos);

    const auto final = aggregator.finalize(key);
    assert(final && final->stable());
    assert(!aggregator.snapshot(key));
}

void testCascadeRuntime() {
    VehicleCascadeRuntimeConfig config;
    config.tracker = {3, 1500, 0.30f};
    config.crop_quality = {96, 64, 0.02f, 0.08f, 0.95f, 0.50f, true};
    config.attribute_queue = {8};
    config.aggregator = {3, 5, 0.75f, 0.70f};
    VehicleCascadeRuntime runtime(config);
    runtime.startRun("gate_01", "run_01", 7);

    VehicleTrackingUpdate update;
    std::string error;
    assert(runtime.onDetections(detectionFrame(7, 1, 0), update, error));
    assert(runtime.onDetections(detectionFrame(7, 2, 100), update, error));
    assert(runtime.onDetections(detectionFrame(7, 3, 200), update, error));
    assert(update.active.size() == 1);
    assert(update.active.front().state == VehicleTrackState::Confirmed);
    const auto track_id = update.active.front().key.track_id;

    const auto image = checkerImage(200, 120);
    CropQualityAssessment assessment;
    assert(runtime.queueCrop(
        image->view(),
        track_id,
        1,
        0.10f,
        false,
        assessment,
        error));
    assert(assessment.eligible);
    const auto batch = runtime.takeAttributeBatch(16);
    assert(batch.size() == 1);
    assert(batch.front().crop && batch.front().crop->valid());

    std::vector<VehicleAttributeResult> results;
    results.push_back(attributeResult(track_id, 1, 0.90f, "suv", "white"));
    results.push_back(attributeResult(track_id, 2, 0.90f, "suv", "white"));
    results.push_back(attributeResult(track_id, 3, 0.90f, "suv", "white"));
    assert(runtime.onAttributeResults(results, error));
    const auto fused = runtime.attributeSnapshot(track_id);
    assert(fused && fused->stable());
    const auto track = runtime.trackSnapshot(track_id);
    assert(track);
    assert(track->state == VehicleTrackState::AttributeStable);
    assert(runtime.trackerMetrics().tracks_confirmed == 1);
    assert(runtime.attributeQueueMetrics().accepted == 1);
    assert(runtime.aggregatorMetrics().accepted == 3);

    const auto final = runtime.finalizeTrack(track_id);
    assert(final && final->stable());
    const auto stopped = runtime.stop(1000);
    assert(stopped.size() == 1);
    assert(stopped.front().state == VehicleTrackState::Exited);
}

}  // namespace

int main() {
    testTracker();
    testQualityGate();
    testAttributeQueue();
    testIndependentFusion();
    testCascadeRuntime();
    std::cout << "PASS: M3 vehicle tracking, quality, queue, fusion, and cascade runtime\n";
    return 0;
}
