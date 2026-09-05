#include "server/vehicle_tensorrt_adapters.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdlib>
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

bool contains(
    const std::vector<std::string>& values,
    const std::string& value) {
    return std::find(values.begin(), values.end(), value) != values.end();
}

int positiveInteger(const char* value, const char* label) {
    const int parsed = std::atoi(value);
    if (parsed != 224 && parsed != 256) {
        std::cerr << label << " must be 224 or 256\n";
        std::exit(64);
    }
    return parsed;
}

ModelArtifactDescriptor attributeArtifact(
    std::string id,
    std::string onnx,
    std::string onnx_sha,
    std::string engine,
    std::string engine_sha,
    int input_size) {
    ModelArtifactDescriptor artifact;
    artifact.artifact_id = std::move(id);
    artifact.role = VehicleModelRole::Attributes;
    artifact.delivery_status = ModelDeliveryStatus::EngineValidated;
    artifact.labels_version = "vehicle-labels-v1";
    artifact.onnx_path = std::move(onnx);
    artifact.engine_path = std::move(engine);
    artifact.onnx_sha256 = std::move(onnx_sha);
    artifact.engine_sha256 = std::move(engine_sha);
    artifact.input_width = input_size;
    artifact.input_height = input_size;
    artifact.max_batch = 16;
    artifact.output_names = {"body_type", "color"};
    return artifact;
}

std::shared_ptr<OwnedImage> gradientImage(
    int width,
    int height,
    int phase) {
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
            image->pixels[offset] =
                static_cast<std::uint8_t>((x + y + phase) % 256);
            image->pixels[offset + 1] =
                static_cast<std::uint8_t>((2 * x + phase) % 256);
            image->pixels[offset + 2] =
                static_cast<std::uint8_t>((2 * y + phase) % 256);
        }
    }
    assert(image->valid());
    return image;
}

std::vector<VehicleAttributeCrop> probeCrops() {
    std::vector<VehicleAttributeCrop> crops;
    for (int index = 0; index < 2; ++index) {
        VehicleAttributeCrop crop;
        crop.crop = gradientImage(320 + index * 16, 180 + index * 8, index * 37);
        crop.camera_id = "stage71-contract-camera";
        crop.run_id = "stage71-contract-run";
        crop.run_generation = 1;
        crop.track_id = 100 + index;
        crop.crop_sequence = static_cast<std::uint64_t>(index + 1);
        crop.quality_score = 1.0f - static_cast<float>(index) * 0.1f;
        crops.push_back(std::move(crop));
    }
    return crops;
}

void validateResult(
    const VehicleAttributeResult& result,
    const std::string& artifact_id) {
    assert(result.metadata.artifact_id == artifact_id);
    assert(result.metadata.labels_version == "vehicle-labels-v1");
    assert(result.metadata.inference_time_us > 0);
    assert(contains(kBodyTypes, result.body_type.label));
    assert(contains(kColors, result.color.label));
    assert(result.body_type.confidence >= 0.0f);
    assert(result.body_type.confidence <= 1.0f);
    assert(result.color.confidence >= 0.0f);
    assert(result.color.confidence <= 1.0f);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 12) {
        std::cerr
            << "usage: vehicle_stage71_decoupled_real_engine_test "
            << "ROOT BODY_ONNX BODY_ONNX_SHA BODY_ENGINE BODY_ENGINE_SHA "
            << "BODY_SIZE COLOR_ONNX COLOR_ONNX_SHA COLOR_ENGINE "
            << "COLOR_ENGINE_SHA COLOR_SIZE\n";
        return 64;
    }
    const std::string root = argv[1];
    const int body_size = positiveInteger(argv[6], "body input size");
    const int color_size = positiveInteger(argv[11], "color input size");
    const auto body_artifact = attributeArtifact(
        "vehicle-attr-stage71-body-candidate",
        argv[2], argv[3], argv[4], argv[5], body_size);
    const auto color_artifact = attributeArtifact(
        "vehicle-attr-stage71-color-candidate",
        argv[7], argv[8], argv[9], argv[10], color_size);

    TensorRtAttributeOptions body_options;
    body_options.artifact_root = root;
    body_options.body_types = kBodyTypes;
    body_options.colors = kColors;
    TensorRtVehicleAttributeRunner body_runner(std::move(body_options));

    TensorRtAttributeOptions color_options;
    color_options.artifact_root = root;
    color_options.body_types = kBodyTypes;
    color_options.colors = kColors;
    TensorRtVehicleAttributeRunner color_runner(std::move(color_options));

    std::string error;
    if (!body_runner.initialize(body_artifact, error)) {
        std::cerr << "body specialist initialize failed: " << error << '\n';
        return 2;
    }
    if (!color_runner.initialize(color_artifact, error)) {
        std::cerr << "color specialist initialize failed: " << error << '\n';
        body_runner.release();
        return 3;
    }

    const auto crops = probeCrops();
    const auto body_results = body_runner.inferBatch(crops);
    const auto color_results = color_runner.inferBatch(crops);
    assert(body_results.size() == crops.size());
    assert(color_results.size() == crops.size());
    for (std::size_t index = 0; index < crops.size(); ++index) {
        validateResult(body_results[index], body_artifact.artifact_id);
        validateResult(color_results[index], color_artifact.artifact_id);
        assert(body_results[index].track_id == color_results[index].track_id);
        assert(body_results[index].crop_sequence == color_results[index].crop_sequence);
        // Runtime routing contract: body comes only from the body specialist
        // and color comes only from the color specialist. The other two
        // logits are intentionally ignored rather than blended.
        const auto& routed_body = body_results[index].body_type;
        const auto& routed_color = color_results[index].color;
        assert(contains(kBodyTypes, routed_body.label));
        assert(contains(kColors, routed_color.label));
    }

    body_runner.release();
    color_runner.release();
    std::cout
        << "PASS: Stage71 decoupled TensorRT engines loaded and completed "
        << "real adapter inference for " << crops.size() << " crops; "
        << "body=" << body_results.front().body_type.label
        << " color=" << color_results.front().color.label << '\n';
    return 0;
}
