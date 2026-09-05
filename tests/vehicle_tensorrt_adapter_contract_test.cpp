#include "server/vehicle_tensorrt_adapters.h"

#include <cassert>
#include <iostream>
#include <string>

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

}  // namespace

int main() {
    TensorRtDetectionOptions detection_options;
    detection_options.vehicle_classes = {
        "car", "bus", "truck", "motorcycle", "vehicle", "other"};
    TensorRtVehicleDetectionRunner detection(std::move(detection_options));

    TensorRtAttributeOptions attribute_options;
    attribute_options.body_types = {
        "sedan", "suv", "mpv", "van", "pickup", "bus",
        "light_truck", "heavy_truck", "other", "unknown"};
    attribute_options.colors = {
        "black", "white", "silver_gray", "red", "blue", "green",
        "yellow_orange", "brown_beige", "other", "unknown"};
    TensorRtVehicleAttributeRunner attributes(std::move(attribute_options));

    std::string error;
    auto detection_artifact = detectionArtifact();
    assert(!detection.initialize(detection_artifact, error));
    assert(error.find("engine_validated or deployed") != std::string::npos);

    detection_artifact.delivery_status = ModelDeliveryStatus::EngineValidated;
    detection_artifact.onnx_sha256 = std::string(64, '1');
    detection_artifact.engine_sha256 = std::string(64, '2');
    assert(!detection.initialize(detection_artifact, error));
    assert(error.find("engine file not found") != std::string::npos);

    auto attribute_artifact = attributeArtifact();
    assert(!attributes.initialize(attribute_artifact, error));
    assert(error.find("engine_validated or deployed") != std::string::npos);

    attribute_artifact.delivery_status = ModelDeliveryStatus::EngineValidated;
    attribute_artifact.onnx_sha256 = std::string(64, '3');
    attribute_artifact.engine_sha256 = std::string(64, '4');
    assert(!attributes.initialize(attribute_artifact, error));
    assert(error.find("engine file not found") != std::string::npos);

    detection.release();
    attributes.release();
    std::cout << "PASS: real TensorRT adapters enforce status, hashes, labels, and files\n";
    return 0;
}
