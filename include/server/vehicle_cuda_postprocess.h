#pragma once

#include <memory>
#include <string>
#include <vector>

#include <NvInferRuntime.h>
#include <cuda_runtime_api.h>

#include "server/vehicle_cuda_preprocess.h"
#include "server/vehicle_model_contract.h"

namespace yolo11_server {

struct VehicleCudaPostprocessTiming {
    double allocation_ms = 0.0;
    double kernel_ms = 0.0;
    double d2h_ms = 0.0;
    std::size_t d2h_bytes = 0;
};

class VehicleCudaPostprocessor {
public:
    VehicleCudaPostprocessor();
    ~VehicleCudaPostprocessor();

    VehicleCudaPostprocessor(const VehicleCudaPostprocessor&) = delete;
    VehicleCudaPostprocessor& operator=(const VehicleCudaPostprocessor&) = delete;

    bool enqueue(
        const float* output,
        const nvinfer1::Dims& shape,
        const VehicleCudaPreprocessGeometry& geometry,
        int source_width,
        int source_height,
        int class_count,
        float confidence_threshold,
        float nms_threshold,
        cudaStream_t stream,
        VehicleCudaPostprocessTiming& timing,
        std::string& error,
        bool record_timing = true);

    bool collect(
        const std::vector<std::string>& labels,
        std::vector<VehicleDetection>& detections,
        VehicleCudaPostprocessTiming& timing,
        std::string& error,
        bool collect_timing = true) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace yolo11_server
