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

ModelArtifactDescriptor detectorArtifact() {
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
    artifact.delivery_status = ModelDeliveryStatus::EngineValidated;
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

std::shared_ptr<OwnedImage> gradientImage(int width, int height) {
    auto image = std::make_shared<OwnedImage>();
    image->width = width;
    image->height = height;
    image->channels = 3;
    image->row_stride_bytes = static_cast<std::size_t>(width * 3);
    image->pixel_format = ImagePixelFormat::Bgr8;
    image->pixels.resize(static_cast<std::size_t>(height) * image->row_stride_bytes);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const auto offset = static_cast<std::size_t>(y) * image->row_stride_bytes +
                static_cast<std::size_t>(x * 3);
            image->pixels[offset] = static_cast<std::uint8_t>((x + y) % 256);
            image->pixels[offset + 1] = static_cast<std::uint8_t>((2 * x) % 256);
            image->pixels[offset + 2] = static_cast<std::uint8_t>((2 * y) % 256);
        }
    }
    assert(image->valid());
    return image;
}

bool contains(const std::vector<std::string>& values, const std::string& value) {
    return std::find(values.begin(), values.end(), value) != values.end();
}

}  // namespace

int main(int argc, char** argv) {
    assert(argc == 2);
    const std::string project_root = argv[1];
    std::string error;

    TensorRtDetectionOptions detection_options;
    detection_options.artifact_root = project_root;
    detection_options.vehicle_classes = kVehicleClasses;
    TensorRtVehicleDetectionRunner detector(std::move(detection_options));
    assert(detector.initialize(detectorArtifact(), error));
    VehicleDetectionRequest request;
    request.frame = gradientImage(640, 384)->view();
    request.camera_id = "best-model-smoke";
    request.run_id = "best-model-smoke";
    request.run_generation = 1;
    request.frame_sequence = 1;
    const auto detections = detector.infer(request);
    assert(detections.metadata.artifact_id == "vehicle-det-v1");
    for (const auto& detection : detections.detections) {
        assert(detection.box.valid());
        assert(contains(kVehicleClasses, detection.vehicle_class));
    }

    TensorRtAttributeOptions attribute_options;
    attribute_options.artifact_root = project_root;
    attribute_options.body_types = kBodyTypes;
    attribute_options.colors = kColors;
    TensorRtVehicleAttributeRunner attributes(std::move(attribute_options));
    assert(attributes.initialize(attributeArtifact(), error));
    VehicleAttributeCrop crop;
    crop.crop = gradientImage(320, 180);
    crop.camera_id = "best-model-smoke";
    crop.run_id = "best-model-smoke";
    crop.run_generation = 1;
    crop.track_id = 1;
    crop.crop_sequence = 1;
    crop.quality_score = 1.0f;
    const auto results = attributes.inferBatch({crop});
    assert(results.size() == 1);
    assert(results.front().metadata.artifact_id ==
        "vehicle-attr-best-components-256-r1");
    assert(contains(kBodyTypes, results.front().body_type.label));
    assert(contains(kColors, results.front().color.label));
    assert(results.front().body_type.confidence >= 0.0f &&
        results.front().body_type.confidence <= 1.0f);
    assert(results.front().color.confidence >= 0.0f &&
        results.front().color.confidence <= 1.0f);

    std::cout << "PASS: best detector and attribute TensorRT engines completed "
              << "real GPU inference; detections=" << detections.detections.size()
              << " body_type=" << results.front().body_type.label
              << " color=" << results.front().color.label << '\n';
    return 0;
}
