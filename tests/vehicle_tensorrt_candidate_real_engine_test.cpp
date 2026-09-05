#include "server/vehicle_tensorrt_adapters.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

using namespace yolo11_server;

namespace {

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
    artifact.delivery_status = ModelDeliveryStatus::EngineValidated;
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
    artifact.artifact_id = "vehicle-attr-v28-merged-input224";
    artifact.role = VehicleModelRole::Attributes;
    artifact.delivery_status = ModelDeliveryStatus::EngineValidated;
    artifact.labels_version = "vehicle-labels-v1";
    artifact.onnx_path = "models/vehicle-attr-v1.onnx";
    artifact.engine_path = "engines/vehicle-attr-v1.engine";
    artifact.onnx_sha256 =
        "af2b9be478c0ace3fa6f9d1fc49bade33e7e1bd87668ce5ca6c08b6b5db6ad2e";
    artifact.engine_sha256 =
        "9710429c2db9bc74bda0912176a782d8efe1acce236220f0591389deb10d11c6";
    artifact.input_width = 224;
    artifact.input_height = 224;
    artifact.max_batch = 16;
    artifact.output_names = {"body_type", "color"};
    return artifact;
}

std::shared_ptr<OwnedImage> gradientImage(int width, int height) {
    auto image = std::make_shared<OwnedImage>();
    image->width = width;
    image->height = height;
    image->channels = 3;
    image->row_stride_bytes = static_cast<std::size_t>(width * 3);
    image->pixel_format = ImagePixelFormat::Bgr8;
    image->pixels.resize(
        static_cast<std::size_t>(height) * image->row_stride_bytes);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const auto offset =
                static_cast<std::size_t>(y) * image->row_stride_bytes +
                static_cast<std::size_t>(x * 3);
            image->pixels[offset] = static_cast<std::uint8_t>((x + y) % 256);
            image->pixels[offset + 1] = static_cast<std::uint8_t>((2 * x) % 256);
            image->pixels[offset + 2] = static_cast<std::uint8_t>((2 * y) % 256);
        }
    }
    assert(image->valid());
    return image;
}

bool contains(
    const std::vector<std::string>& values,
    const std::string& value) {
    return std::find(values.begin(), values.end(), value) != values.end();
}

}  // namespace

int main(int argc, char** argv) {
    assert(argc == 2);
    const std::string project_root = argv[1];

    TensorRtDetectionOptions detection_options;
    detection_options.artifact_root = project_root;
    detection_options.vehicle_classes = kVehicleClasses;
    TensorRtVehicleDetectionRunner detection(std::move(detection_options));

    std::string error;
    assert(detection.initialize(detectionArtifact(), error));
    auto frame = gradientImage(640, 384);
    VehicleDetectionRequest request;
    request.frame = frame->view();
    request.camera_id = "engine-smoke-camera";
    request.run_id = "engine-smoke-run";
    request.run_generation = 1;
    request.frame_sequence = 1;
    const auto detection_result = detection.infer(request);
    assert(detection_result.metadata.artifact_id == "vehicle-det-v1");
    assert(detection_result.metadata.labels_version == "vehicle-labels-v1");
    assert(detection_result.metadata.inference_time_us > 0);
    for (const auto& item : detection_result.detections) {
        assert(item.box.valid());
        assert(contains(kVehicleClasses, item.vehicle_class));
        assert(item.confidence >= 0.0f && item.confidence <= 1.0f);
    }

    TensorRtAttributeOptions attribute_options;
    attribute_options.artifact_root = project_root;
    attribute_options.body_types = kBodyTypes;
    attribute_options.colors = kColors;
    TensorRtVehicleAttributeRunner attributes(std::move(attribute_options));
    assert(attributes.initialize(attributeArtifact(), error));
    VehicleAttributeCrop crop;
    crop.crop = gradientImage(320, 180);
    crop.camera_id = "engine-smoke-camera";
    crop.run_id = "engine-smoke-run";
    crop.run_generation = 1;
    crop.track_id = 1;
    crop.crop_sequence = 1;
    crop.quality_score = 1.0f;
    const auto attribute_results = attributes.inferBatch({crop});
    assert(attribute_results.size() == 1);
    const auto& attribute_result = attribute_results.front();
    assert(attribute_result.metadata.artifact_id == "vehicle-attr-v28-merged-input224");
    assert(attribute_result.metadata.labels_version == "vehicle-labels-v1");
    assert(attribute_result.metadata.inference_time_us > 0);
    assert(contains(kBodyTypes, attribute_result.body_type.label));
    assert(contains(kColors, attribute_result.color.label));
    assert(
        attribute_result.body_type.confidence >= 0.0f &&
        attribute_result.body_type.confidence <= 1.0f);
    assert(
        attribute_result.color.confidence >= 0.0f &&
        attribute_result.color.confidence <= 1.0f);

    detection.release();
    attributes.release();
    std::cout
        << "PASS: project TensorRT adapters loaded both release engines and "
        << "completed GPU inference; detections="
        << detection_result.detections.size()
        << " body_type=" << attribute_result.body_type.label
        << " color=" << attribute_result.color.label << '\n';
    return 0;
}
