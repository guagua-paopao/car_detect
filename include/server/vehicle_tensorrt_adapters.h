#pragma once

#include <memory>
#include <string>
#include <vector>

#include "server/vehicle_model_contract.h"

namespace yolo11_server {

struct TensorRtDetectionOptions {
    int gpu_id = 0;
    std::string artifact_root = ".";
    std::string input_tensor_name = "images";
    std::string output_tensor_name = "output0";
    std::vector<std::string> vehicle_classes;
    float confidence_threshold = 0.25f;
    float nms_threshold = 0.45f;
};

struct TensorRtAttributeOptions {
    int gpu_id = 0;
    std::string artifact_root = ".";
    std::string input_tensor_name = "images";
    std::string body_type_output_name = "body_type";
    std::string color_output_name = "color";
    std::vector<std::string> body_types;
    std::vector<std::string> colors;
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
    void release() noexcept override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace yolo11_server
