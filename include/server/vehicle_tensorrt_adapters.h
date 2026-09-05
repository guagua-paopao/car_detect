#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "server/vehicle_model_contract.h"

namespace yolo11_server {

struct TensorRtStageTiming {
    double preprocess_ms = 0.0;
    double host_staging_ms = 0.0;
    double gpu_preprocess_ms = 0.0;
    double allocation_ms = 0.0;
    double h2d_ms = 0.0;
    double inference_ms = 0.0;
    double d2h_ms = 0.0;
    double postprocess_ms = 0.0;
    double gpu_postprocess_ms = 0.0;
    double cuda_graph_ms = 0.0;
    double device_frame_copy_ms = 0.0;
    double total_ms = 0.0;
    std::size_t h2d_bytes = 0;
    std::size_t d2h_bytes = 0;
    std::size_t device_frame_copy_bytes = 0;
    bool cuda_graph_used = false;
    bool cuda_graph_fallback = false;
    bool cuda_graph_built = false;
    bool device_frame_retained = false;
    bool device_frame_retain_fallback = false;
};

struct TensorRtDetectionOptions {
    int gpu_id = 0;
    std::string artifact_root = ".";
    std::string input_tensor_name = "images";
    std::string output_tensor_name = "output0";
    std::vector<std::string> vehicle_classes;
    float confidence_threshold = 0.25f;
    float nms_threshold = 0.45f;
    bool use_gpu_preprocess = true;
    bool use_gpu_postprocess = true;
    bool use_cuda_graph = true;
    bool retain_i420_device_frame = true;
};

struct TensorRtAttributeOptions {
    int gpu_id = 0;
    std::string artifact_root = ".";
    std::string input_tensor_name = "images";
    std::string body_type_output_name = "body_type";
    std::string color_output_name = "color";
    std::vector<std::string> body_types;
    std::vector<std::string> colors;
    bool use_gpu_preprocess = true;
    bool use_cuda_graph = true;
};

class TensorRtVehicleDetectionRunner final : public IVehicleDetectionRunner {
public:
    explicit TensorRtVehicleDetectionRunner(TensorRtDetectionOptions options);
    ~TensorRtVehicleDetectionRunner() override;

    bool initialize(
        const ModelArtifactDescriptor& artifact,
        std::string& error) override;
    VehicleDetectionResult infer(
        const VehicleDetectionRequest& request) override;
    TensorRtStageTiming lastTiming() const noexcept;
    void release() noexcept override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

class TensorRtVehicleAttributeRunner final : public IVehicleAttributeRunner {
public:
    explicit TensorRtVehicleAttributeRunner(TensorRtAttributeOptions options);
    ~TensorRtVehicleAttributeRunner() override;

    bool initialize(
        const ModelArtifactDescriptor& artifact,
        std::string& error) override;
    std::vector<VehicleAttributeResult> inferBatch(
        const std::vector<VehicleAttributeCrop>& crops) override;
    TensorRtStageTiming lastTiming() const noexcept;
    void release() noexcept override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace yolo11_server
