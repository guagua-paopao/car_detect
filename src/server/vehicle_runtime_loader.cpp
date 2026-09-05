#include "server/vehicle_runtime_loader.h"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <utility>

#include <nlohmann/json.hpp>

namespace yolo11_server {

namespace {

using json = nlohmann::json;

json readJson(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("file not found: " + path.string());
    }
    json value;
    input >> value;
    return value;
}

std::filesystem::path absolutePath(const std::string& value) {
    return std::filesystem::absolute(
        std::filesystem::u8path(value.empty() ? "." : value)).lexically_normal();
}

ModelDeliveryStatus deliveryStatus(const std::string& value) {
    if (value == "engine_validated") return ModelDeliveryStatus::EngineValidated;
    if (value == "deployed") return ModelDeliveryStatus::Deployed;
    if (value == "onnx_validated") return ModelDeliveryStatus::OnnxValidated;
    if (value == "blocked") return ModelDeliveryStatus::Blocked;
    return ModelDeliveryStatus::Planned;
}

VehicleModelRole modelRole(const std::string& value) {
    if (value == "detection") return VehicleModelRole::Detection;
    if (value == "attributes") return VehicleModelRole::Attributes;
    return VehicleModelRole::Unknown;
}

ModelArtifactDescriptor descriptor(const json& item) {
    ModelArtifactDescriptor result;
    result.artifact_id = item.value("artifact_id", "");
    result.role = modelRole(item.value("role", ""));
    result.delivery_status = deliveryStatus(item.value("delivery_status", ""));
    result.labels_version = item.value("labels_version", "");
    const auto input = item.value("input", json::object());
    result.input_width = input.value("width", 0);
    result.input_height = input.value("height", 0);
    result.input_channels = input.value("channels", 3);
    const auto deployment = item.value("deployment", json::object());
    result.backend = deployment.value("backend", "tensorrt");
    result.precision = deployment.value("precision", "fp16");
    result.max_batch = deployment.value("max_batch", 1);
    const auto files = item.value("files", json::object());
    result.onnx_path = files.value("onnx_path", "");
    result.engine_path = files.value("engine_path", "");
    result.onnx_sha256 = files.value("onnx_sha256", "");
    result.engine_sha256 = files.value("engine_sha256", "");
    if (item.contains("output_names")) {
        result.output_names = item.at("output_names").get<std::vector<std::string>>();
    }
    return result;
}

template <typename T>
void assignIfPresent(const json& value, const char* key, T& target) {
    if (value.contains(key)) target = value.at(key).get<T>();
}

}  // namespace

bool loadVehicleRuntimeAssets(
    const AppConfig& config,
    VehicleRuntimeAssets& assets,
    std::string& error) {
    assets = {};
    error.clear();
    try {
        const auto analytics_path = absolutePath(config.vehicle_analytics.config_path);
        const auto analytics_root = readJson(analytics_path);
        const auto analytics = analytics_root.at("vehicle_analytics");
        assets.config_version = analytics_root.value("config_version", "vehicle-analytics-v1");
        assets.labels_version = analytics_root.value("labels_version", "");

        auto labels_path = std::filesystem::u8path(
            analytics_root.value("labels_path", "./config/vehicle_labels.v1.json"));
        if (labels_path.is_relative()) {
            // Runtime paths in the shipped config are project-root-relative.
            labels_path = std::filesystem::current_path() / labels_path;
        }
        const auto labels = readJson(labels_path.lexically_normal());
        assets.vehicle_classes = labels.at("vehicle_classes").get<std::vector<std::string>>();
        assets.body_types = labels.at("body_types").get<std::vector<std::string>>();
        assets.colors = labels.at("colors").get<std::vector<std::string>>();
        if (assets.labels_version.empty()) {
            assets.labels_version = labels.value("labels_version", "");
        }

        const auto registry = readJson(absolutePath(config.vehicle_analytics.model_registry_path));
        for (const auto& item : registry.at("artifacts")) {
            auto artifact = descriptor(item);
            if (artifact.role == VehicleModelRole::Detection) {
                assets.detector = std::move(artifact);
            }
            else if (artifact.role == VehicleModelRole::Attributes) {
                assets.attributes = std::move(artifact);
            }
        }
        if (assets.detector.artifact_id.empty() ||
            assets.attributes.artifact_id.empty()) {
            throw std::runtime_error(
                "vehicle registry must contain detection and attributes artifacts");
        }
        if (assets.detector.labels_version != assets.labels_version ||
            assets.attributes.labels_version != assets.labels_version) {
            throw std::runtime_error("vehicle model and label versions do not match");
        }
        std::string validation_error;
        if (!validateModelArtifact(assets.detector, validation_error) ||
            !validateModelArtifact(assets.attributes, validation_error)) {
            throw std::runtime_error(validation_error);
        }

        assignIfPresent(analytics, "min_confirm_hits", assets.cascade.tracker.min_confirm_hits);
        assignIfPresent(
            analytics,
            "detection_confidence_threshold",
            assets.detection_confidence_threshold);
        assignIfPresent(analytics, "track_timeout_ms", assets.cascade.tracker.track_timeout_ms);
        assignIfPresent(analytics, "tracking_iou_threshold", assets.cascade.tracker.match_iou_threshold);
        assignIfPresent(analytics, "min_crop_width_px", assets.cascade.crop_quality.min_width_px);
        assignIfPresent(analytics, "min_crop_height_px", assets.cascade.crop_quality.min_height_px);
        assignIfPresent(analytics, "min_crop_sharpness", assets.cascade.crop_quality.min_sharpness);
        assignIfPresent(analytics, "min_crop_exposure", assets.cascade.crop_quality.min_exposure);
        assignIfPresent(analytics, "max_crop_exposure", assets.cascade.crop_quality.max_exposure);
        assignIfPresent(analytics, "max_crop_occlusion", assets.cascade.crop_quality.max_occlusion);
        assignIfPresent(analytics, "reject_truncated_crops", assets.cascade.crop_quality.reject_truncated);
        const auto queues = analytics_root.value("queues", json::object());
        if (queues.contains("attribute_max_pending")) {
            assets.cascade.attribute_queue.max_pending =
                queues.at("attribute_max_pending").get<std::size_t>();
        }
        assignIfPresent(analytics, "attribute_vote_samples", assets.cascade.aggregator.min_samples);
        assignIfPresent(analytics, "attribute_max_observations", assets.cascade.aggregator.max_observations);
        assignIfPresent(analytics, "type_threshold", assets.cascade.aggregator.body_type_threshold);
        assignIfPresent(analytics, "color_threshold", assets.cascade.aggregator.color_threshold);
        const char* demo_vehicle =
            std::getenv("YOLO11_DEMO_VEHICLE_ANALYSIS");
        if (demo_vehicle && std::string(demo_vehicle) == "1") {
            // The public 60-second demo contains deliberately small, blurred
            // and edge-truncated vehicles.  Let the attribute model inspect
            // those crops, while the unchanged fusion thresholds still map
            // uncertain predictions to the contract value `unknown`.
            assets.cascade.tracker.min_confirm_hits =
                std::min(2, assets.cascade.tracker.min_confirm_hits);
            assets.cascade.tracker.match_iou_threshold =
                std::min(0.20f, assets.cascade.tracker.match_iou_threshold);
            assets.cascade.crop_quality.min_width_px =
                std::min(48, assets.cascade.crop_quality.min_width_px);
            assets.cascade.crop_quality.min_height_px =
                std::min(32, assets.cascade.crop_quality.min_height_px);
            assets.cascade.crop_quality.min_sharpness =
                std::min(0.005f, assets.cascade.crop_quality.min_sharpness);
            assets.cascade.crop_quality.min_exposure =
                std::min(0.02f, assets.cascade.crop_quality.min_exposure);
            assets.cascade.crop_quality.max_exposure =
                std::max(0.98f, assets.cascade.crop_quality.max_exposure);
            assets.cascade.crop_quality.reject_truncated = false;
        }
        assets.attribute_max_batch = static_cast<std::size_t>(
            std::max(1, assets.attributes.max_batch));
        assets.detection_confidence_threshold = std::clamp(
            assets.detection_confidence_threshold, 0.0f, 1.0f);
        return true;
    }
    catch (const std::exception& exception) {
        error = exception.what();
        assets = {};
        return false;
    }
}

}  // namespace yolo11_server
