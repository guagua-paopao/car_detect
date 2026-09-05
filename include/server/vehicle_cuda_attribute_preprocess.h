#pragma once

#include <memory>
#include <string>
#include <vector>

#include <cuda_runtime_api.h>

#include "server/vehicle_cuda_preprocess.h"
#include "server/vehicle_model_contract.h"

namespace yolo11_server {

class VehicleCudaAttributePreprocessor {
public:
    VehicleCudaAttributePreprocessor();
    ~VehicleCudaAttributePreprocessor();

    VehicleCudaAttributePreprocessor(const VehicleCudaAttributePreprocessor&) = delete;
    VehicleCudaAttributePreprocessor& operator=(
        const VehicleCudaAttributePreprocessor&) = delete;

    bool enqueueResizeBatch(
        const std::vector<VehicleAttributeCrop>& crops,
        float* destination,
        int destination_width,
        int destination_height,
        cudaStream_t stream,
        VehicleCudaPreprocessTiming& timing,
        std::string& error);

    // Staging is deliberately separated from enqueue so the stable pinned
    // descriptor slab -> CUDA crop/resize sequence can be captured together
    // with TensorRT and compact output D2H in one CUDA Graph.
    bool stageResizeBatch(
        const std::vector<VehicleAttributeCrop>& crops,
        float* destination,
        int destination_width,
        int destination_height,
        VehicleCudaPreprocessTiming& timing,
        std::string& error);

    bool enqueueStaged(
        cudaStream_t stream,
        VehicleCudaPreprocessTiming& timing,
        std::string& error,
        bool record_timing = true);

    bool collectTiming(
        VehicleCudaPreprocessTiming& timing,
        std::string& error) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace yolo11_server
