#include "server/model_runner.h"

#include <algorithm>
#include <cctype>
#include <stdexcept>

#include <opencv2/imgproc.hpp>

#include "server/vehicle_runtime_loader.h"
#include "server/vehicle_tensorrt_adapters.h"

namespace yolo11_server {

namespace {
std::string lower(std::string text) {
    std::transform(text.begin(), text.end(), text.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return text;
}
}  // namespace

VehicleCameraModelRunner::~VehicleCameraModelRunner() = default;

std::string VehicleCameraModelRunner::modelType() const {
    return "vehicle_detection";
}

bool VehicleCameraModelRunner::init(
    const AppConfig& config,
    std::string& error) {
    VehicleRuntimeAssets assets;
    if (!loadVehicleRuntimeAssets(config, assets, error)) return false;
    TensorRtDetectionOptions options;
    options.gpu_id = config.model.gpu_id;
    options.artifact_root = ".";
    options.vehicle_classes = assets.vehicle_classes;
    options.confidence_threshold = assets.detection_confidence_threshold;
    detector_ = std::make_unique<TensorRtVehicleDetectionRunner>(
        std::move(options));
    if (!detector_->initialize(assets.detector, error)) {
        detector_.reset();
        return false;
    }
    return true;
}

ModelOutput VehicleCameraModelRunner::infer(const cv::Mat& image) {
    if (!detector_) {
        throw std::runtime_error("VehicleCameraModelRunner is not initialized");
    }
    if (image.empty() || image.type() != CV_8UC3) {
        throw std::invalid_argument(
            "VehicleCameraModelRunner requires an 8-bit BGR image");
    }
    VehicleDetectionRequest request;
    request.frame = {
        image.data,
        image.cols,
        image.rows,
        image.channels(),
        image.step,
        ImagePixelFormat::Bgr8
    };
    ModelOutput output;
    output.model_type = "vehicle_detection";
    output.has_vehicle_detection = true;
    output.vehicle_detection = detector_->infer(request);
    return output;
}

ModelOutput VehicleCameraModelRunner::infer(const FrameEnvelope& frame) {
    if (!detector_) {
        throw std::runtime_error("VehicleCameraModelRunner is not initialized");
    }
    if (!frame.i420.valid()) return infer(frame.bgrImage());
    const auto* bytes = frame.i420.bytes->data();
    VehicleDetectionRequest request;
    request.i420_frame = {
        bytes + frame.i420.y_offset,
        bytes + frame.i420.u_offset,
        bytes + frame.i420.v_offset,
        frame.i420.width,
        frame.i420.height,
        frame.i420.y_stride_bytes,
        frame.i420.u_stride_bytes,
        frame.i420.v_stride_bytes,
    };
    ModelOutput output;
    output.model_type = "vehicle_detection";
    output.has_vehicle_detection = true;
    output.vehicle_detection = detector_->infer(request);
    return output;
}

cv::Mat VehicleCameraModelRunner::draw(
    const cv::Mat& image,
    const ModelOutput& output) {
    cv::Mat rendered = image.clone();
    if (!output.has_vehicle_detection) return rendered;
    for (const auto& item : output.vehicle_detection.detections) {
        const cv::Rect box(
            cv::Point(
                static_cast<int>(item.box.x1 * image.cols),
                static_cast<int>(item.box.y1 * image.rows)),
            cv::Point(
                static_cast<int>(item.box.x2 * image.cols),
                static_cast<int>(item.box.y2 * image.rows)));
        cv::rectangle(rendered, box, cv::Scalar(0, 220, 255), 2);
        cv::putText(
            rendered,
            item.vehicle_class + " " + std::to_string(item.confidence).substr(0, 4),
            cv::Point(box.x, std::max(18, box.y - 5)),
            cv::FONT_HERSHEY_SIMPLEX,
            0.55,
            cv::Scalar(0, 220, 255),
            2,
            cv::LINE_AA);
    }
    return rendered;
}

void VehicleCameraModelRunner::release() noexcept {
    if (detector_) detector_->release();
    detector_.reset();
}

std::unique_ptr<IModelRunner> createModelRunner(const std::string& model_type) {
    const auto type = lower(model_type);
    if (type == "vehicle" || type == "vehicle_detection") {
        return std::make_unique<VehicleCameraModelRunner>();
    }
    return nullptr;
}

}  // namespace yolo11_server
