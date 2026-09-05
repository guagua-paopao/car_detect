#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace yolo11_server {

enum class ImagePixelFormat {
    Bgr8,
    Rgb8,
};

enum class VehicleModelRole {
    Unknown,
    Detection,
    Attributes,
};

enum class ModelDeliveryStatus {
    Planned,
    OnnxValidated,
    EngineValidated,
    Deployed,
    Blocked,
};

std::string_view toString(VehicleModelRole role) noexcept;
std::string_view toString(ModelDeliveryStatus status) noexcept;

struct ModelArtifactDescriptor {
    std::string artifact_id;
    VehicleModelRole role = VehicleModelRole::Unknown;
    ModelDeliveryStatus delivery_status = ModelDeliveryStatus::Planned;
    std::string labels_version;
    std::string backend = "tensorrt";
    std::string precision = "fp16";
    std::string onnx_path;
    std::string engine_path;
    std::string onnx_sha256;
    std::string engine_sha256;
    int input_width = 0;
    int input_height = 0;
    int input_channels = 3;
    int max_batch = 1;
    std::vector<std::string> output_names;
};

bool validateModelArtifact(
    const ModelArtifactDescriptor& artifact,
    std::string& error);

class VehicleModelRegistry {
public:
    bool add(ModelArtifactDescriptor artifact, std::string& error);
    const ModelArtifactDescriptor* find(VehicleModelRole role) const noexcept;
    const ModelArtifactDescriptor* find(std::string_view artifact_id) const noexcept;
    std::size_t size() const noexcept;

private:
    std::vector<ModelArtifactDescriptor> artifacts_;
};

struct ImageView {
    const std::uint8_t* data = nullptr;
    int width = 0;
    int height = 0;
    int channels = 0;
    std::size_t row_stride_bytes = 0;
    ImagePixelFormat pixel_format = ImagePixelFormat::Bgr8;

    bool valid() const noexcept;
};

struct I420ImageView {
    const std::uint8_t* y_plane = nullptr;
    const std::uint8_t* u_plane = nullptr;
    const std::uint8_t* v_plane = nullptr;
    int width = 0;
    int height = 0;
    std::size_t y_stride_bytes = 0;
    std::size_t u_stride_bytes = 0;
    std::size_t v_stride_bytes = 0;

    bool valid() const noexcept;
};

struct OwnedI420Image {
    std::shared_ptr<const std::vector<std::uint8_t>> bytes;
    int width = 0;
    int height = 0;
    std::size_t y_offset = 0;
    std::size_t u_offset = 0;
    std::size_t v_offset = 0;
    std::size_t y_stride_bytes = 0;
    std::size_t u_stride_bytes = 0;
    std::size_t v_stride_bytes = 0;

    bool valid() const noexcept;
    I420ImageView view() const noexcept;
};

// CUDA-resident I420 planes retained by a bounded producer-owned lease.  The
// pointers are device addresses and must never be dereferenced by host code.
struct DeviceI420Image {
    const std::uint8_t* y_plane = nullptr;
    const std::uint8_t* u_plane = nullptr;
    const std::uint8_t* v_plane = nullptr;
    int width = 0;
    int height = 0;
    std::size_t y_stride_bytes = 0;
    std::size_t u_stride_bytes = 0;
    std::size_t v_stride_bytes = 0;
    std::shared_ptr<void> lease;

    bool valid() const noexcept;
};

struct OwnedImage {
    std::vector<std::uint8_t> pixels;
    int width = 0;
    int height = 0;
    int channels = 0;
    std::size_t row_stride_bytes = 0;
    ImagePixelFormat pixel_format = ImagePixelFormat::Bgr8;

    bool valid() const noexcept;
    ImageView view() const noexcept;
};

struct VehicleBox {
    float x1 = 0.0f;
    float y1 = 0.0f;
    float x2 = 0.0f;
    float y2 = 0.0f;

    bool valid() const noexcept;
};

struct VehicleDetection {
    VehicleBox box;
    std::string vehicle_class;
    float confidence = 0.0f;
};

struct ModelResultMetadata {
    std::string artifact_id;
    std::string labels_version;
    std::uint64_t run_generation = 0;
    std::int64_t inference_time_us = 0;
};

struct VehicleDetectionRequest {
    ImageView frame;
    I420ImageView i420_frame;
    std::string camera_id;
    std::string run_id;
    std::uint64_t run_generation = 0;
    std::int64_t frame_sequence = 0;
    std::int64_t captured_at_ms = 0;
};

struct VehicleDetectionResult {
    ModelResultMetadata metadata;
    std::string camera_id;
    std::string run_id;
    std::int64_t frame_sequence = 0;
    std::int64_t captured_at_ms = 0;
    std::vector<VehicleDetection> detections;
    std::shared_ptr<const DeviceI420Image> device_i420_frame;
};

struct VehicleAttributeCrop {
    std::shared_ptr<const OwnedImage> crop;
    std::shared_ptr<const OwnedI420Image> i420_frame;
    std::shared_ptr<const DeviceI420Image> device_i420_frame;
    VehicleBox source_box;
    std::string camera_id;
    std::string run_id;
    std::uint64_t run_generation = 0;
    std::int64_t track_id = 0;
    std::uint64_t crop_sequence = 0;
    float quality_score = 0.0f;
};

struct AttributePrediction {
    std::string label = "unknown";
    float confidence = 0.0f;
};

struct VehicleAttributeResult {
    ModelResultMetadata metadata;
    std::string camera_id;
    std::string run_id;
    std::int64_t track_id = 0;
    std::uint64_t crop_sequence = 0;
    float quality_score = 0.0f;
    AttributePrediction body_type;
    AttributePrediction color;
};

class IVehicleDetectionRunner {
public:
    virtual ~IVehicleDetectionRunner() = default;
    virtual bool initialize(
        const ModelArtifactDescriptor& artifact,
        std::string& error) = 0;
    virtual VehicleDetectionResult infer(
        const VehicleDetectionRequest& request) = 0;
    virtual void release() noexcept = 0;
};

class IVehicleAttributeRunner {
public:
    virtual ~IVehicleAttributeRunner() = default;
    virtual bool initialize(
        const ModelArtifactDescriptor& artifact,
        std::string& error) = 0;
    virtual std::vector<VehicleAttributeResult> inferBatch(
        const std::vector<VehicleAttributeCrop>& crops) = 0;
    virtual void release() noexcept = 0;
};

using VehicleDetectionRunnerPtr = std::unique_ptr<IVehicleDetectionRunner>;
using VehicleAttributeRunnerPtr = std::unique_ptr<IVehicleAttributeRunner>;

}  // namespace yolo11_server
