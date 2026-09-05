#pragma once

#include <memory>
#include <string>

#include <opencv2/opencv.hpp>

#include "business/camera_frame_types.h"
#include "server/app_config.h"
#include "server/model_output.h"

namespace yolo11_server {

class TensorRtVehicleDetectionRunner;

class IModelRunner {
public:
    virtual ~IModelRunner() = default;
    virtual std::string modelType() const = 0;
    virtual bool init(const AppConfig& config, std::string& error) = 0;
    virtual ModelOutput infer(const cv::Mat& image) = 0;
    virtual ModelOutput infer(const FrameEnvelope& frame) {
        return infer(frame.bgrImage());
    }
    virtual cv::Mat draw(const cv::Mat& image, const ModelOutput& output) = 0;
    virtual void release() noexcept = 0;
};

class VehicleCameraModelRunner final : public IModelRunner {
public:
    ~VehicleCameraModelRunner() override;
    std::string modelType() const override;
    bool init(const AppConfig& config, std::string& error) override;
    ModelOutput infer(const cv::Mat& image) override;
    ModelOutput infer(const FrameEnvelope& frame) override;
    cv::Mat draw(const cv::Mat& image, const ModelOutput& output) override;
    void release() noexcept override;

private:
    std::unique_ptr<TensorRtVehicleDetectionRunner> detector_;
};

std::unique_ptr<IModelRunner> createModelRunner(const std::string& model_type);

}  // namespace yolo11_server
