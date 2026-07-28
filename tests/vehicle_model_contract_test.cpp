#include "server/vehicle_model_contract.h"

#include <cassert>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>

using namespace yolo11_server;

namespace {

ModelArtifactDescriptor detectionArtifact() {
    ModelArtifactDescriptor artifact;
    artifact.artifact_id = "vehicle-det-v1";
    artifact.role = VehicleModelRole::Detection;
    artifact.labels_version = "vehicle-labels-v1";
    artifact.onnx_path = "models/vehicle-det-v1.onnx";
    artifact.engine_path = "engines/vehicle-det-v1.engine";
    artifact.input_width = 960;
    artifact.input_height = 960;
    artifact.max_batch = 1;
    artifact.output_names = {"vehicle_detections"};
    return artifact;
}

ModelArtifactDescriptor attributeArtifact() {
    ModelArtifactDescriptor artifact;
    artifact.artifact_id = "vehicle-attr-v1";
    artifact.role = VehicleModelRole::Attributes;
    artifact.labels_version = "vehicle-labels-v1";
    artifact.onnx_path = "models/vehicle-attr-v1.onnx";
    artifact.engine_path = "engines/vehicle-attr-v1.engine";
    artifact.input_width = 224;
    artifact.input_height = 224;
    artifact.max_batch = 16;
    artifact.output_names = {"body_type", "color"};
    return artifact;
}

class FakeDetectionRunner final : public IVehicleDetectionRunner {
public:
    bool initialize(
        const ModelArtifactDescriptor& artifact,
        std::string& error) override {
        if (artifact.role != VehicleModelRole::Detection) {
            error = "wrong role";
            return false;
        }
        artifact_id_ = artifact.artifact_id;
        labels_version_ = artifact.labels_version;
        return true;
    }

    VehicleDetectionResult infer(
        const VehicleDetectionRequest& request) override {
        if (!request.frame.valid()) throw std::invalid_argument("invalid frame");
        VehicleDetectionResult result;
        result.metadata.artifact_id = artifact_id_;
        result.metadata.labels_version = labels_version_;
        result.metadata.run_generation = request.run_generation;
        result.camera_id = request.camera_id;
        result.run_id = request.run_id;
        result.frame_sequence = request.frame_sequence;
        result.captured_at_ms = request.captured_at_ms;
        result.detections.push_back({
            {0.1f, 0.2f, 0.8f, 0.9f}, "car", 0.95f});
        return result;
    }

    void release() noexcept override {
        artifact_id_.clear();
        labels_version_.clear();
    }

private:
    std::string artifact_id_;
    std::string labels_version_;
};

class FakeAttributeRunner final : public IVehicleAttributeRunner {
public:
    bool initialize(
        const ModelArtifactDescriptor& artifact,
        std::string& error) override {
        if (artifact.role != VehicleModelRole::Attributes) {
            error = "wrong role";
            return false;
        }
        artifact_id_ = artifact.artifact_id;
        labels_version_ = artifact.labels_version;
        return true;
    }

    std::vector<VehicleAttributeResult> inferBatch(
        const std::vector<VehicleAttributeCrop>& crops) override {
        std::vector<VehicleAttributeResult> results;
        for (const auto& crop : crops) {
            if (!crop.crop || !crop.crop->valid()) {
                throw std::invalid_argument("invalid crop");
            }
            VehicleAttributeResult result;
            result.metadata.artifact_id = artifact_id_;
            result.metadata.labels_version = labels_version_;
            result.metadata.run_generation = crop.run_generation;
            result.camera_id = crop.camera_id;
            result.run_id = crop.run_id;
            result.track_id = crop.track_id;
            result.crop_sequence = crop.crop_sequence;
            result.quality_score = crop.quality_score;
            result.body_type = {"suv", 0.91f};
            result.color = {"white", 0.88f};
            results.push_back(std::move(result));
        }
        return results;
    }

    void release() noexcept override {
        artifact_id_.clear();
        labels_version_.clear();
    }

private:
    std::string artifact_id_;
    std::string labels_version_;
};

}  // namespace

int main() {
    std::string error;
    VehicleModelRegistry registry;
    assert(registry.add(detectionArtifact(), error));
    assert(registry.add(attributeArtifact(), error));
    assert(registry.size() == 2);
    assert(registry.find(VehicleModelRole::Detection)->artifact_id == "vehicle-det-v1");
    assert(registry.find("vehicle-attr-v1")->max_batch == 16);

    auto duplicate = detectionArtifact();
    duplicate.artifact_id = "vehicle-det-v2";
    assert(!registry.add(duplicate, error));
    assert(error.find("duplicate active role") != std::string::npos);

    auto unsafe = attributeArtifact();
    unsafe.engine_path = "../private/vehicle-attr-v1.engine";
    assert(!validateModelArtifact(unsafe, error));
    assert(error.find("safe relative") != std::string::npos);

    auto falsely_deployed = attributeArtifact();
    falsely_deployed.delivery_status = ModelDeliveryStatus::Deployed;
    assert(!validateModelArtifact(falsely_deployed, error));
    assert(error.find("onnx_sha256") != std::string::npos);

    std::vector<std::uint8_t> pixels(16 * 16 * 3);
    ImageView image{pixels.data(), 16, 16, 3, 16 * 3};
    assert(image.valid());
    auto owned = std::make_shared<OwnedImage>();
    owned->pixels = pixels;
    owned->width = 16;
    owned->height = 16;
    owned->channels = 3;
    owned->row_stride_bytes = 16 * 3;
    assert(owned->valid());
    assert(owned->view().valid());

    FakeDetectionRunner detector;
    assert(detector.initialize(*registry.find(VehicleModelRole::Detection), error));
    VehicleDetectionRequest frame;
    frame.frame = image;
    frame.camera_id = "gate_01";
    frame.run_id = "run_01";
    frame.run_generation = 7;
    frame.frame_sequence = 42;
    const auto detection = detector.infer(frame);
    assert(detection.metadata.artifact_id == "vehicle-det-v1");
    assert(detection.metadata.run_generation == 7);
    assert(detection.detections.size() == 1);
    assert(detection.detections.front().box.valid());

    FakeAttributeRunner attributes;
    assert(attributes.initialize(
        *registry.find(VehicleModelRole::Attributes), error));
    VehicleAttributeCrop crop;
    crop.crop = owned;
    crop.camera_id = "gate_01";
    crop.run_id = "run_01";
    crop.run_generation = 7;
    crop.track_id = 101;
    crop.crop_sequence = 3;
    crop.quality_score = 0.93f;
    const auto predictions = attributes.inferBatch({crop});
    assert(predictions.size() == 1);
    assert(predictions.front().track_id == 101);
    assert(predictions.front().crop_sequence == 3);
    assert(predictions.front().metadata.run_generation == 7);
    assert(predictions.front().body_type.label == "suv");
    assert(predictions.front().color.label == "white");

    std::cout << "PASS: vehicle detection/attribute runner and registry contracts\n";
    return 0;
}
