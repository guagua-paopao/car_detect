#include "server/vehicle_model_contract.h"

#include <algorithm>
#include <cctype>
#include <utility>

namespace yolo11_server {

namespace {

bool isLowerHexSha256(const std::string& value) {
    return value.size() == 64 &&
        std::all_of(value.begin(), value.end(), [](unsigned char ch) {
            return std::isdigit(ch) || (ch >= 'a' && ch <= 'f');
        });
}

bool isSafeRelativePath(const std::string& value, std::string_view extension) {
    if (value.empty()) return false;
    if (value.front() == '/' || value.front() == '\\' ||
        value.find('\\') != std::string::npos) {
        return false;
    }
    if (value.size() >= 2 &&
        std::isalpha(static_cast<unsigned char>(value[0])) &&
        value[1] == ':') {
        return false;
    }
    std::size_t start = 0;
    while (start <= value.size()) {
        const auto separator = value.find('/', start);
        const auto length = separator == std::string::npos
            ? value.size() - start
            : separator - start;
        const auto part = value.substr(start, length);
        if (part.empty() || part == "." || part == "..") return false;
        if (separator == std::string::npos) break;
        start = separator + 1;
    }
    return value.size() > extension.size() &&
        value.compare(value.size() - extension.size(), extension.size(), extension) == 0;
}

bool containsExactly(
    const std::vector<std::string>& values,
    std::initializer_list<std::string_view> expected) {
    if (values.size() != expected.size()) return false;
    for (const auto item : expected) {
        if (std::find(values.begin(), values.end(), item) == values.end()) {
            return false;
        }
    }
    return true;
}

}  // namespace

std::string_view toString(VehicleModelRole role) noexcept {
    switch (role) {
    case VehicleModelRole::Detection: return "detection";
    case VehicleModelRole::Attributes: return "attributes";
    default: return "unknown";
    }
}

std::string_view toString(ModelDeliveryStatus status) noexcept {
    switch (status) {
    case ModelDeliveryStatus::Planned: return "planned";
    case ModelDeliveryStatus::OnnxValidated: return "onnx_validated";
    case ModelDeliveryStatus::EngineValidated: return "engine_validated";
    case ModelDeliveryStatus::Deployed: return "deployed";
    case ModelDeliveryStatus::Blocked: return "blocked";
    }
    return "blocked";
}

bool validateModelArtifact(
    const ModelArtifactDescriptor& artifact,
    std::string& error) {
    error.clear();
    if (artifact.artifact_id.empty()) {
        error = "artifact_id is required";
    }
    else if (artifact.role == VehicleModelRole::Unknown) {
        error = "model role must be detection or attributes";
    }
    else if (artifact.labels_version.empty()) {
        error = "labels_version is required";
    }
    else if (artifact.backend != "tensorrt") {
        error = "M2 supports only the tensorrt deployment backend";
    }
    else if (artifact.precision != "fp16" &&
        artifact.precision != "fp32" &&
        artifact.precision != "int8") {
        error = "precision must be fp32, fp16, or int8";
    }
    else if (!isSafeRelativePath(artifact.onnx_path, ".onnx")) {
        error = "onnx_path must be a safe relative .onnx path";
    }
    else if (!isSafeRelativePath(artifact.engine_path, ".engine")) {
        error = "engine_path must be a safe relative .engine path";
    }
    else if (artifact.input_width <= 0 ||
        artifact.input_height <= 0 ||
        artifact.input_channels != 3) {
        error = "model input must have positive dimensions and three channels";
    }
    else if (artifact.max_batch <= 0 || artifact.max_batch > 128) {
        error = "max_batch must be in [1, 128]";
    }
    else if (artifact.role == VehicleModelRole::Detection &&
        !containsExactly(artifact.output_names, {"vehicle_detections"})) {
        error = "detection model must expose vehicle_detections";
    }
    else if (artifact.role == VehicleModelRole::Attributes &&
        !containsExactly(artifact.output_names, {"body_type", "color"})) {
        error = "attribute model must expose body_type and color";
    }
    else if (!artifact.onnx_sha256.empty() &&
        !isLowerHexSha256(artifact.onnx_sha256)) {
        error = "onnx_sha256 must be 64 lowercase hexadecimal characters";
    }
    else if (!artifact.engine_sha256.empty() &&
        !isLowerHexSha256(artifact.engine_sha256)) {
        error = "engine_sha256 must be 64 lowercase hexadecimal characters";
    }
    else if (artifact.delivery_status != ModelDeliveryStatus::Planned &&
        artifact.delivery_status != ModelDeliveryStatus::Blocked &&
        artifact.onnx_sha256.empty()) {
        error = "validated model status requires onnx_sha256";
    }
    else if ((artifact.delivery_status == ModelDeliveryStatus::EngineValidated ||
        artifact.delivery_status == ModelDeliveryStatus::Deployed) &&
        artifact.engine_sha256.empty()) {
        error = "engine-validated or deployed status requires engine_sha256";
    }
    return error.empty();
}

bool VehicleModelRegistry::add(
    ModelArtifactDescriptor artifact,
    std::string& error) {
    if (!validateModelArtifact(artifact, error)) return false;
    if (find(artifact.artifact_id) != nullptr) {
        error = "duplicate artifact_id: " + artifact.artifact_id;
        return false;
    }
    if (find(artifact.role) != nullptr) {
        error = "duplicate active role: " + std::string(toString(artifact.role));
        return false;
    }
    artifacts_.push_back(std::move(artifact));
    error.clear();
    return true;
}

const ModelArtifactDescriptor* VehicleModelRegistry::find(
    VehicleModelRole role) const noexcept {
    const auto it = std::find_if(
        artifacts_.begin(),
        artifacts_.end(),
        [role](const ModelArtifactDescriptor& item) {
            return item.role == role;
        });
    return it == artifacts_.end() ? nullptr : &*it;
}

const ModelArtifactDescriptor* VehicleModelRegistry::find(
    std::string_view artifact_id) const noexcept {
    const auto it = std::find_if(
        artifacts_.begin(),
        artifacts_.end(),
        [artifact_id](const ModelArtifactDescriptor& item) {
            return item.artifact_id == artifact_id;
        });
    return it == artifacts_.end() ? nullptr : &*it;
}

std::size_t VehicleModelRegistry::size() const noexcept {
    return artifacts_.size();
}

bool ImageView::valid() const noexcept {
    if (data == nullptr || width <= 0 || height <= 0 || channels <= 0) {
        return false;
    }
    const auto minimum_stride =
        static_cast<std::size_t>(width) * static_cast<std::size_t>(channels);
    return row_stride_bytes >= minimum_stride;
}

bool VehicleBox::valid() const noexcept {
    return x1 >= 0.0f && y1 >= 0.0f &&
        x1 < x2 && y1 < y2 &&
        x2 <= 1.0f && y2 <= 1.0f;
}

}  // namespace yolo11_server
