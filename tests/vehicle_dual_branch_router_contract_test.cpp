#include "server/vehicle_tensorrt_adapters.h"

#include <cassert>
#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

using namespace yolo11_server;

namespace {

const std::vector<std::string> kBodyTypes = {
    "sedan", "suv", "mpv", "van", "pickup", "bus",
    "light_truck", "heavy_truck", "other", "unknown"};
const std::vector<std::string> kColors = {
    "black", "white", "silver_gray", "red", "blue", "green",
    "yellow_orange", "brown_beige", "other", "unknown"};

ModelArtifactDescriptor attributeArtifact(
    std::string id,
    std::string onnx,
    std::string onnx_sha,
    std::string engine,
    std::string engine_sha) {
    ModelArtifactDescriptor artifact;
    artifact.artifact_id = std::move(id);
    artifact.role = VehicleModelRole::Attributes;
    artifact.delivery_status = ModelDeliveryStatus::EngineValidated;
    artifact.labels_version = "vehicle-labels-v1";
    artifact.onnx_path = std::move(onnx);
    artifact.engine_path = std::move(engine);
    artifact.onnx_sha256 = std::move(onnx_sha);
    artifact.engine_sha256 = std::move(engine_sha);
    artifact.input_width = 224;
    artifact.input_height = 224;
    artifact.max_batch = 16;
    artifact.output_names = {"body_type", "color"};
    return artifact;
}

std::shared_ptr<OwnedImage> probeImage() {
    auto image = std::make_shared<OwnedImage>();
    image->width = 320;
    image->height = 180;
    image->channels = 3;
    image->row_stride_bytes = 320u * 3u;
    image->pixel_format = ImagePixelFormat::Bgr8;
    image->pixels.resize(image->row_stride_bytes * 180u, 96u);
    assert(image->valid());
    return image;
}

VehicleAttributeResult routeColor(
    const VehicleAttributeResult& candidate,
    const VehicleAttributeResult& baseline) {
    constexpr float kBaseGate = 0.8f;
    constexpr float kCandidateGate = 0.9f;
    constexpr float kMargin = 0.05f;
    VehicleAttributeResult routed = baseline;
    if (baseline.color.confidence < kBaseGate &&
        candidate.color.confidence >= kCandidateGate &&
        candidate.color.confidence >= baseline.color.confidence + kMargin) {
        routed.color = candidate.color;
    }
    else if (baseline.color.confidence < kBaseGate) {
        routed.color = {"unknown", 0.0f};
    }
    return routed;
}

}  // namespace

int main(int argc, char** argv) {
    assert(argc == 2);
    const std::string root = argv[1];

    TensorRtAttributeOptions candidate_options;
    candidate_options.artifact_root = root;
    candidate_options.body_types = kBodyTypes;
    candidate_options.colors = kColors;
    TensorRtVehicleAttributeRunner candidate(std::move(candidate_options));

    TensorRtAttributeOptions baseline_options;
    baseline_options.artifact_root = root;
    baseline_options.body_types = kBodyTypes;
    baseline_options.colors = kColors;
    TensorRtVehicleAttributeRunner baseline(std::move(baseline_options));

    std::string error;
    const auto candidate_artifact = attributeArtifact(
        "vehicle-attr-v32-color-candidate",
        "color-candidate.onnx",
        "b9c5149fd42b0a4cb6d281a34b0d50becc5aeebda73696ab7216d91a3bc3b506",
        "body-candidate.engine",
        "65a0bcca6cf94ab52e173fb606406c4ce63e7d95ff65894220be4045d787a0b1");
    const auto baseline_artifact = attributeArtifact(
        "vehicle-attr-e-224-color-baseline",
        "color-baseline.onnx",
        "874267440af4fb5512eaa137eeead883d28b9877bdaab635ed5f2f8085a091a9",
        "color-baseline.engine",
        "f21642d91dc9bc458e8a26d064e2140a5c3368c7311f2b7accb12e2f6e3b52b4");
    if (!candidate.initialize(candidate_artifact, error)) {
        std::cerr << "candidate initialize failed: " << error << '\n';
        return 2;
    }
    if (!baseline.initialize(baseline_artifact, error)) {
        std::cerr << "baseline initialize failed: " << error << '\n';
        return 3;
    }

    VehicleAttributeCrop crop;
    crop.crop = probeImage();
    crop.camera_id = "v32-contract-camera";
    crop.run_id = "v32-contract-run";
    crop.run_generation = 1;
    crop.track_id = 7;
    crop.crop_sequence = 1;
    crop.quality_score = 1.0f;
    const auto candidate_result = candidate.inferBatch({crop}).front();
    const auto baseline_result = baseline.inferBatch({crop}).front();
    const auto routed = routeColor(candidate_result, baseline_result);
    assert(!routed.body_type.label.empty());
    assert(!routed.color.label.empty());
    assert(routed.color.confidence >= 0.0f && routed.color.confidence <= 1.0f);
    assert(routed.metadata.artifact_id == baseline_result.metadata.artifact_id);

    candidate.release();
    baseline.release();
    std::cout << "PASS: v32 dual-branch TensorRT candidate contract and conservative color route\n";
    return 0;
}
