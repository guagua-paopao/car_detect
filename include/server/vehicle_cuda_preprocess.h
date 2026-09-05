#pragma once

#include <memory>
#include <string>

#include <cuda_runtime_api.h>

#include "server/vehicle_model_contract.h"

namespace yolo11_server {

struct VehicleCudaPreprocessGeometry {
    float scale = 1.0f;
    float pad_x = 0.0f;
    float pad_y = 0.0f;
};

struct VehicleCudaPreprocessTiming {
    double host_staging_ms = 0.0;
    double h2d_ms = 0.0;
    double kernel_ms = 0.0;
    double allocation_ms = 0.0;
    double device_frame_copy_ms = 0.0;
    std::size_t h2d_bytes = 0;
    std::size_t device_frame_copy_bytes = 0;
};

class VehicleCudaPreprocessor {
public:
    VehicleCudaPreprocessor();
    ~VehicleCudaPreprocessor();

    VehicleCudaPreprocessor(const VehicleCudaPreprocessor&) = delete;
    VehicleCudaPreprocessor& operator=(const VehicleCudaPreprocessor&) = delete;

    bool enqueueLetterbox(
        const ImageView& source,
        float* destination,
        int destination_width,
        int destination_height,
        cudaStream_t stream,
        VehicleCudaPreprocessGeometry& geometry,
        VehicleCudaPreprocessTiming& timing,
        std::string& error);

    bool enqueueI420Letterbox(
        const I420ImageView& source,
        float* destination,
        int destination_width,
        int destination_height,
        cudaStream_t stream,
        VehicleCudaPreprocessGeometry& geometry,
        VehicleCudaPreprocessTiming& timing,
        std::string& error);

    bool stageLetterbox(
        const ImageView& source,
        int destination_width,
        int destination_height,
        VehicleCudaPreprocessGeometry& geometry,
        VehicleCudaPreprocessTiming& timing,
        std::string& error);

    bool stageI420Letterbox(
        const I420ImageView& source,
        int destination_width,
        int destination_height,
        VehicleCudaPreprocessGeometry& geometry,
        VehicleCudaPreprocessTiming& timing,
        std::string& error);

    bool enqueueStaged(
        float* destination,
        cudaStream_t stream,
        VehicleCudaPreprocessTiming& timing,
        std::string& error,
        bool record_timing = true);

    bool retainStagedI420(
        cudaStream_t stream,
        std::shared_ptr<const DeviceI420Image>& frame,
        VehicleCudaPreprocessTiming& timing,
        std::string& error);

    bool collectTiming(
        VehicleCudaPreprocessTiming& timing,
        std::string& error) const;

    bool collectRetainedI420Timing(
        VehicleCudaPreprocessTiming& timing,
        std::string& error) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace yolo11_server
