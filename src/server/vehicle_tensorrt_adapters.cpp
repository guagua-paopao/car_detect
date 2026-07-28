#include "server/vehicle_tensorrt_adapters.h"

#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <cuda_runtime_api.h>
#include <openssl/evp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace yolo11_server {

namespace {

class TensorRtLogger final : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* message) noexcept override {
        if (severity <= Severity::kWARNING) last_message_ = message ? message : "";
    }

    const std::string& lastMessage() const noexcept {
        return last_message_;
    }

private:
    std::string last_message_;
};

TensorRtLogger g_logger;

std::string resolveArtifactPath(
    const std::string& artifact_root,
    const std::string& relative_path) {
    const auto root = std::filesystem::absolute(
        artifact_root.empty() ? "." : artifact_root);
    return (root / std::filesystem::path(relative_path))
        .lexically_normal()
        .string();
}

bool cudaOk(cudaError_t status, std::string_view action, std::string& error) {
    if (status == cudaSuccess) return true;
    error = std::string(action) + ": " + cudaGetErrorString(status);
    return false;
}

std::string sha256File(const std::string& path, std::string& error) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        error = "engine file not found: " + path;
        return {};
    }
    EVP_MD_CTX* context = EVP_MD_CTX_new();
    if (!context) {
        error = "EVP_MD_CTX_new failed";
        return {};
    }
    std::string result;
    if (EVP_DigestInit_ex(context, EVP_sha256(), nullptr) != 1) {
        error = "EVP_DigestInit_ex failed";
    }
    else {
        std::vector<char> buffer(1024 * 1024);
        while (stream) {
            stream.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
            const auto count = stream.gcount();
            if (count > 0 &&
                EVP_DigestUpdate(
                    context,
                    buffer.data(),
                    static_cast<std::size_t>(count)) != 1) {
                error = "EVP_DigestUpdate failed";
                break;
            }
        }
        if (error.empty()) {
            unsigned char digest[EVP_MAX_MD_SIZE] = {};
            unsigned int length = 0;
            if (EVP_DigestFinal_ex(context, digest, &length) != 1) {
                error = "EVP_DigestFinal_ex failed";
            }
            else {
                std::ostringstream text;
                text << std::hex << std::setfill('0');
                for (unsigned int index = 0; index < length; ++index) {
                    text << std::setw(2) << static_cast<int>(digest[index]);
                }
                result = text.str();
            }
        }
    }
    EVP_MD_CTX_free(context);
    return result;
}

std::vector<char> readBinaryFile(const std::string& path, std::string& error) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) {
        error = "engine file not found: " + path;
        return {};
    }
    const auto size = stream.tellg();
    if (size <= 0) {
        error = "engine file is empty: " + path;
        return {};
    }
    stream.seekg(0, std::ios::beg);
    std::vector<char> bytes(static_cast<std::size_t>(size));
    if (!stream.read(bytes.data(), static_cast<std::streamsize>(size))) {
        error = "failed to read engine file: " + path;
        return {};
    }
    return bytes;
}

std::size_t volume(const nvinfer1::Dims& dims) {
    if (dims.nbDims <= 0) return 0;
    std::size_t result = 1;
    for (int index = 0; index < dims.nbDims; ++index) {
        if (dims.d[index] <= 0) return 0;
        result *= static_cast<std::size_t>(dims.d[index]);
    }
    return result;
}

class TensorRtEngine {
public:
    ~TensorRtEngine() {
        release();
    }

    bool load(
        const ModelArtifactDescriptor& artifact,
        const std::string& artifact_root,
        int gpu_id,
        const std::vector<std::string>& tensor_names,
        std::string& error) {
        release();
        gpu_id_ = gpu_id;
        if (!cudaOk(cudaSetDevice(gpu_id_), "cudaSetDevice", error)) return false;
        const auto engine_path =
            resolveArtifactPath(artifact_root, artifact.engine_path);
        const auto actual_hash = sha256File(engine_path, error);
        if (actual_hash.empty()) return false;
        if (actual_hash != artifact.engine_sha256) {
            error = "engine SHA256 does not match model registry";
            return false;
        }
        auto bytes = readBinaryFile(engine_path, error);
        if (bytes.empty()) return false;
        initLibNvInferPlugins(&g_logger, "");
        runtime_ = nvinfer1::createInferRuntime(g_logger);
        if (!runtime_) {
            error = "TensorRT createInferRuntime failed: " + g_logger.lastMessage();
            return false;
        }
        engine_ = runtime_->deserializeCudaEngine(bytes.data(), bytes.size());
        if (!engine_) {
            error = "TensorRT deserializeCudaEngine failed: " + g_logger.lastMessage();
            release();
            return false;
        }
        context_ = engine_->createExecutionContext();
        if (!context_) {
            error = "TensorRT createExecutionContext failed";
            release();
            return false;
        }
        if (!cudaOk(cudaStreamCreate(&stream_), "cudaStreamCreate", error)) {
            release();
            return false;
        }
        for (std::size_t index = 0; index < tensor_names.size(); ++index) {
            const auto& name = tensor_names[index];
            if (name.empty() || engine_->getTensorIOMode(name.c_str()) ==
                nvinfer1::TensorIOMode::kNONE) {
                error = "TensorRT engine is missing tensor: " + name;
                release();
                return false;
            }
            const auto expected_mode = index == 0
                ? nvinfer1::TensorIOMode::kINPUT
                : nvinfer1::TensorIOMode::kOUTPUT;
            if (engine_->getTensorIOMode(name.c_str()) != expected_mode) {
                error = "TensorRT tensor has unexpected input/output mode: " + name;
                release();
                return false;
            }
            if (engine_->getTensorDataType(name.c_str()) != nvinfer1::DataType::kFLOAT) {
                error = "TensorRT adapter currently requires float tensors: " + name;
                release();
                return false;
            }
        }
        return true;
    }

    bool execute(
        const std::string& input_name,
        const nvinfer1::Dims& input_shape,
        const std::vector<float>& input,
        const std::vector<std::string>& output_names,
        std::vector<std::vector<float>>& outputs,
        std::vector<nvinfer1::Dims>& output_shapes,
        std::string& error) {
        outputs.clear();
        output_shapes.clear();
        if (!engine_ || !context_ || !stream_) {
            error = "TensorRT engine is not initialized";
            return false;
        }
        if (!cudaOk(cudaSetDevice(gpu_id_), "cudaSetDevice", error)) return false;
        if (!context_->setInputShape(input_name.c_str(), input_shape)) {
            error = "TensorRT rejected input shape";
            return false;
        }
        if (input.size() != volume(input_shape)) {
            error = "preprocessed input volume does not match TensorRT shape";
            return false;
        }

        void* input_device = nullptr;
        std::vector<void*> output_devices(output_names.size(), nullptr);
        const auto cleanup = [&]() {
            if (input_device) cudaFree(input_device);
            for (void* pointer : output_devices) {
                if (pointer) cudaFree(pointer);
            }
        };

        if (!cudaOk(
            cudaMalloc(&input_device, input.size() * sizeof(float)),
            "cudaMalloc input",
            error)) {
            return false;
        }
        if (!cudaOk(
            cudaMemcpyAsync(
                input_device,
                input.data(),
                input.size() * sizeof(float),
                cudaMemcpyHostToDevice,
                stream_),
            "cudaMemcpyAsync input",
            error)) {
            cleanup();
            return false;
        }
        if (!context_->setInputTensorAddress(input_name.c_str(), input_device)) {
            error = "setInputTensorAddress failed";
            cleanup();
            return false;
        }

        outputs.resize(output_names.size());
        output_shapes.resize(output_names.size());
        for (std::size_t index = 0; index < output_names.size(); ++index) {
            output_shapes[index] = context_->getTensorShape(output_names[index].c_str());
            const auto count = volume(output_shapes[index]);
            if (count == 0) {
                error = "TensorRT output shape is unresolved: " + output_names[index];
                cleanup();
                return false;
            }
            outputs[index].resize(count);
            if (!cudaOk(
                cudaMalloc(&output_devices[index], count * sizeof(float)),
                "cudaMalloc output",
                error)) {
                cleanup();
                return false;
            }
            if (!context_->setOutputTensorAddress(
                output_names[index].c_str(),
                output_devices[index])) {
                error = "setOutputTensorAddress failed: " + output_names[index];
                cleanup();
                return false;
            }
        }
        if (!context_->enqueueV3(stream_)) {
            error = "TensorRT enqueueV3 failed: " + g_logger.lastMessage();
            cleanup();
            return false;
        }
        for (std::size_t index = 0; index < outputs.size(); ++index) {
            if (!cudaOk(
                cudaMemcpyAsync(
                    outputs[index].data(),
                    output_devices[index],
                    outputs[index].size() * sizeof(float),
                    cudaMemcpyDeviceToHost,
                    stream_),
                "cudaMemcpyAsync output",
                error)) {
                cleanup();
                return false;
            }
        }
        const bool synchronized =
            cudaOk(cudaStreamSynchronize(stream_), "cudaStreamSynchronize", error);
        cleanup();
        return synchronized;
    }

    void release() noexcept {
        if (stream_) {
            cudaStreamDestroy(stream_);
            stream_ = nullptr;
        }
        delete context_;
        context_ = nullptr;
        delete engine_;
        engine_ = nullptr;
        delete runtime_;
        runtime_ = nullptr;
    }

private:
    int gpu_id_ = 0;
    nvinfer1::IRuntime* runtime_ = nullptr;
    nvinfer1::ICudaEngine* engine_ = nullptr;
    nvinfer1::IExecutionContext* context_ = nullptr;
    cudaStream_t stream_ = nullptr;
};

struct PreparedImage {
    std::vector<float> tensor;
    float scale = 1.0f;
    float pad_x = 0.0f;
    float pad_y = 0.0f;
};

float sourceChannel(const ImageView& image, int x, int y, int rgb_channel) {
    const auto* pixel = image.data +
        static_cast<std::size_t>(y) * image.row_stride_bytes +
        static_cast<std::size_t>(x) * static_cast<std::size_t>(image.channels);
    int source_channel = rgb_channel;
    if (image.pixel_format == ImagePixelFormat::Bgr8) {
        source_channel = 2 - rgb_channel;
    }
    return static_cast<float>(pixel[source_channel]) / 255.0f;
}

PreparedImage letterbox(
    const ImageView& image,
    int target_width,
    int target_height) {
    PreparedImage prepared;
    prepared.scale = std::min(
        static_cast<float>(target_width) / static_cast<float>(image.width),
        static_cast<float>(target_height) / static_cast<float>(image.height));
    const int resized_width = std::max(
        1,
        static_cast<int>(std::round(image.width * prepared.scale)));
    const int resized_height = std::max(
        1,
        static_cast<int>(std::round(image.height * prepared.scale)));
    prepared.pad_x = static_cast<float>(target_width - resized_width) / 2.0f;
    prepared.pad_y = static_cast<float>(target_height - resized_height) / 2.0f;
    const auto plane =
        static_cast<std::size_t>(target_width) * static_cast<std::size_t>(target_height);
    prepared.tensor.assign(3 * plane, 114.0f / 255.0f);
    const int pad_left = static_cast<int>(std::floor(prepared.pad_x));
    const int pad_top = static_cast<int>(std::floor(prepared.pad_y));
    for (int y = 0; y < resized_height; ++y) {
        const int source_y = std::min(
            image.height - 1,
            static_cast<int>(static_cast<float>(y) / prepared.scale));
        for (int x = 0; x < resized_width; ++x) {
            const int source_x = std::min(
                image.width - 1,
                static_cast<int>(static_cast<float>(x) / prepared.scale));
            const auto destination =
                static_cast<std::size_t>(pad_top + y) *
                    static_cast<std::size_t>(target_width) +
                static_cast<std::size_t>(pad_left + x);
            for (int channel = 0; channel < 3; ++channel) {
                prepared.tensor[
                    static_cast<std::size_t>(channel) * plane + destination] =
                    sourceChannel(image, source_x, source_y, channel);
            }
        }
    }
    return prepared;
}

std::vector<float> resizeBatch(
    const std::vector<VehicleAttributeCrop>& crops,
    int target_width,
    int target_height) {
    const auto plane =
        static_cast<std::size_t>(target_width) * static_cast<std::size_t>(target_height);
    std::vector<float> tensor(crops.size() * 3 * plane);
    for (std::size_t batch = 0; batch < crops.size(); ++batch) {
        const auto image = crops[batch].crop->view();
        for (int y = 0; y < target_height; ++y) {
            const int source_y = std::min(
                image.height - 1,
                y * image.height / target_height);
            for (int x = 0; x < target_width; ++x) {
                const int source_x = std::min(
                    image.width - 1,
                    x * image.width / target_width);
                const auto destination =
                    static_cast<std::size_t>(y) *
                        static_cast<std::size_t>(target_width) +
                    static_cast<std::size_t>(x);
                for (int channel = 0; channel < 3; ++channel) {
                    tensor[
                        batch * 3 * plane +
                        static_cast<std::size_t>(channel) * plane +
                        destination] =
                        sourceChannel(image, source_x, source_y, channel);
                }
            }
        }
    }
    return tensor;
}

AttributePrediction softmaxTop1(
    const float* logits,
    std::size_t count,
    const std::vector<std::string>& labels) {
    if (count != labels.size() || count == 0) {
        throw std::runtime_error("attribute tensor size does not match labels");
    }
    const float maximum = *std::max_element(logits, logits + count);
    double denominator = 0.0;
    std::size_t best_index = 0;
    float best_logit = logits[0];
    for (std::size_t index = 0; index < count; ++index) {
        denominator += std::exp(static_cast<double>(logits[index] - maximum));
        if (logits[index] > best_logit) {
            best_logit = logits[index];
            best_index = index;
        }
    }
    const float confidence = denominator <= 0.0
        ? 0.0f
        : static_cast<float>(
            std::exp(static_cast<double>(best_logit - maximum)) / denominator);
    return {labels[best_index], confidence};
}

float iou(const VehicleBox& first, const VehicleBox& second) {
    const float left = std::max(first.x1, second.x1);
    const float top = std::max(first.y1, second.y1);
    const float right = std::min(first.x2, second.x2);
    const float bottom = std::min(first.y2, second.y2);
    const float intersection =
        std::max(0.0f, right - left) * std::max(0.0f, bottom - top);
    const float first_area = (first.x2 - first.x1) * (first.y2 - first.y1);
    const float second_area = (second.x2 - second.x1) * (second.y2 - second.y1);
    const float total = first_area + second_area - intersection;
    return total > 0.0f ? intersection / total : 0.0f;
}

std::vector<VehicleDetection> decodeYolo(
    const std::vector<float>& output,
    const nvinfer1::Dims& shape,
    const PreparedImage& prepared,
    const ImageView& source,
    int input_width,
    int input_height,
    const TensorRtDetectionOptions& options) {
    if (shape.nbDims != 3 || shape.d[0] != 1) {
        throw std::runtime_error("detection output must have shape [1,F,A] or [1,A,F]");
    }
    const int features = 4 + static_cast<int>(options.vehicle_classes.size());
    bool feature_first = false;
    int anchors = 0;
    if (shape.d[1] == features) {
        feature_first = true;
        anchors = shape.d[2];
    }
    else if (shape.d[2] == features) {
        anchors = shape.d[1];
    }
    else {
        throw std::runtime_error("detection output feature count does not match labels");
    }
    const auto at = [&](int anchor, int feature) {
        return feature_first
            ? output[static_cast<std::size_t>(feature) *
                static_cast<std::size_t>(anchors) +
                static_cast<std::size_t>(anchor)]
            : output[static_cast<std::size_t>(anchor) *
                static_cast<std::size_t>(features) +
                static_cast<std::size_t>(feature)];
    };
    std::vector<VehicleDetection> candidates;
    for (int anchor = 0; anchor < anchors; ++anchor) {
        int best_class = 0;
        float confidence = at(anchor, 4);
        for (int index = 1; index < static_cast<int>(options.vehicle_classes.size()); ++index) {
            if (at(anchor, 4 + index) > confidence) {
                confidence = at(anchor, 4 + index);
                best_class = index;
            }
        }
        if (confidence < options.confidence_threshold) continue;
        const float center_x = at(anchor, 0);
        const float center_y = at(anchor, 1);
        const float width = at(anchor, 2);
        const float height = at(anchor, 3);
        VehicleBox box;
        box.x1 = std::max(
            0.0f,
            (center_x - width / 2.0f - prepared.pad_x) /
                prepared.scale / static_cast<float>(source.width));
        box.y1 = std::max(
            0.0f,
            (center_y - height / 2.0f - prepared.pad_y) /
                prepared.scale / static_cast<float>(source.height));
        box.x2 = std::min(
            1.0f,
            (center_x + width / 2.0f - prepared.pad_x) /
                prepared.scale / static_cast<float>(source.width));
        box.y2 = std::min(
            1.0f,
            (center_y + height / 2.0f - prepared.pad_y) /
                prepared.scale / static_cast<float>(source.height));
        if (!box.valid()) continue;
        candidates.push_back({
            box,
            options.vehicle_classes[static_cast<std::size_t>(best_class)],
            confidence,
        });
    }
    std::sort(
        candidates.begin(),
        candidates.end(),
        [](const VehicleDetection& left, const VehicleDetection& right) {
            return left.confidence > right.confidence;
        });
    std::vector<VehicleDetection> kept;
    for (const auto& candidate : candidates) {
        const bool suppressed = std::any_of(
            kept.begin(),
            kept.end(),
            [&](const VehicleDetection& existing) {
                return existing.vehicle_class == candidate.vehicle_class &&
                    iou(existing.box, candidate.box) > options.nms_threshold;
            });
        if (!suppressed) kept.push_back(candidate);
    }
    (void)input_width;
    (void)input_height;
    return kept;
}

}  // namespace

struct TensorRtVehicleDetectionRunner::Impl {
    explicit Impl(TensorRtDetectionOptions value)
        : options(std::move(value)) {
    }

    TensorRtDetectionOptions options;
    ModelArtifactDescriptor artifact;
    TensorRtEngine engine;
    bool initialized = false;
};

TensorRtVehicleDetectionRunner::TensorRtVehicleDetectionRunner(
    TensorRtDetectionOptions options)
    : impl_(std::make_unique<Impl>(std::move(options))) {
}

TensorRtVehicleDetectionRunner::~TensorRtVehicleDetectionRunner() = default;

bool TensorRtVehicleDetectionRunner::initialize(
    const ModelArtifactDescriptor& artifact,
    std::string& error) {
    release();
    if (!validateModelArtifact(artifact, error)) return false;
    if (artifact.role != VehicleModelRole::Detection) {
        error = "detection adapter requires detection artifact";
        return false;
    }
    if (artifact.delivery_status != ModelDeliveryStatus::EngineValidated &&
        artifact.delivery_status != ModelDeliveryStatus::Deployed) {
        error = "detection artifact must be engine_validated or deployed";
        return false;
    }
    if (impl_->options.vehicle_classes.empty()) {
        error = "detection labels are required";
        return false;
    }
    if (!impl_->engine.load(
        artifact,
        impl_->options.artifact_root,
        impl_->options.gpu_id,
        {impl_->options.input_tensor_name, impl_->options.output_tensor_name},
        error)) {
        return false;
    }
    impl_->artifact = artifact;
    impl_->initialized = true;
    return true;
}

VehicleDetectionResult TensorRtVehicleDetectionRunner::infer(
    const VehicleDetectionRequest& request) {
    if (!impl_->initialized) {
        throw std::runtime_error("TensorRtVehicleDetectionRunner is not initialized");
    }
    if (!request.frame.valid() || request.frame.channels != 3) {
        throw std::invalid_argument("detection request requires a valid 3-channel frame");
    }
    const auto started = std::chrono::steady_clock::now();
    const auto prepared = letterbox(
        request.frame,
        impl_->artifact.input_width,
        impl_->artifact.input_height);
    std::vector<std::vector<float>> outputs;
    std::vector<nvinfer1::Dims> shapes;
    std::string error;
    if (!impl_->engine.execute(
        impl_->options.input_tensor_name,
        nvinfer1::Dims4{
            1,
            3,
            impl_->artifact.input_height,
            impl_->artifact.input_width},
        prepared.tensor,
        {impl_->options.output_tensor_name},
        outputs,
        shapes,
        error)) {
        throw std::runtime_error(error);
    }
    VehicleDetectionResult result;
    result.metadata.artifact_id = impl_->artifact.artifact_id;
    result.metadata.labels_version = impl_->artifact.labels_version;
    result.metadata.run_generation = request.run_generation;
    result.camera_id = request.camera_id;
    result.run_id = request.run_id;
    result.frame_sequence = request.frame_sequence;
    result.captured_at_ms = request.captured_at_ms;
    result.detections = decodeYolo(
        outputs.front(),
        shapes.front(),
        prepared,
        request.frame,
        impl_->artifact.input_width,
        impl_->artifact.input_height,
        impl_->options);
    result.metadata.inference_time_us =
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - started).count();
    return result;
}

void TensorRtVehicleDetectionRunner::release() noexcept {
    if (!impl_) return;
    impl_->engine.release();
    impl_->initialized = false;
    impl_->artifact = {};
}

struct TensorRtVehicleAttributeRunner::Impl {
    explicit Impl(TensorRtAttributeOptions value)
        : options(std::move(value)) {
    }

    TensorRtAttributeOptions options;
    ModelArtifactDescriptor artifact;
    TensorRtEngine engine;
    bool initialized = false;
};

TensorRtVehicleAttributeRunner::TensorRtVehicleAttributeRunner(
    TensorRtAttributeOptions options)
    : impl_(std::make_unique<Impl>(std::move(options))) {
}

TensorRtVehicleAttributeRunner::~TensorRtVehicleAttributeRunner() = default;

bool TensorRtVehicleAttributeRunner::initialize(
    const ModelArtifactDescriptor& artifact,
    std::string& error) {
    release();
    if (!validateModelArtifact(artifact, error)) return false;
    if (artifact.role != VehicleModelRole::Attributes) {
        error = "attribute adapter requires attributes artifact";
        return false;
    }
    if (artifact.delivery_status != ModelDeliveryStatus::EngineValidated &&
        artifact.delivery_status != ModelDeliveryStatus::Deployed) {
        error = "attribute artifact must be engine_validated or deployed";
        return false;
    }
    if (impl_->options.body_types.empty() || impl_->options.colors.empty()) {
        error = "attribute label sets are required";
        return false;
    }
    if (!impl_->engine.load(
        artifact,
        impl_->options.artifact_root,
        impl_->options.gpu_id,
        {
            impl_->options.input_tensor_name,
            impl_->options.body_type_output_name,
            impl_->options.color_output_name,
        },
        error)) {
        return false;
    }
    impl_->artifact = artifact;
    impl_->initialized = true;
    return true;
}

std::vector<VehicleAttributeResult>
TensorRtVehicleAttributeRunner::inferBatch(
    const std::vector<VehicleAttributeCrop>& crops) {
    if (!impl_->initialized) {
        throw std::runtime_error("TensorRtVehicleAttributeRunner is not initialized");
    }
    if (crops.empty()) return {};
    if (crops.size() > static_cast<std::size_t>(impl_->artifact.max_batch)) {
        throw std::invalid_argument("attribute batch exceeds artifact max_batch");
    }
    for (const auto& crop : crops) {
        if (!crop.crop || !crop.crop->valid() || crop.crop->channels != 3) {
            throw std::invalid_argument("attribute batch contains an invalid crop");
        }
    }
    const auto started = std::chrono::steady_clock::now();
    const auto input = resizeBatch(
        crops,
        impl_->artifact.input_width,
        impl_->artifact.input_height);
    std::vector<std::vector<float>> outputs;
    std::vector<nvinfer1::Dims> shapes;
    std::string error;
    if (!impl_->engine.execute(
        impl_->options.input_tensor_name,
        nvinfer1::Dims4{
            static_cast<int>(crops.size()),
            3,
            impl_->artifact.input_height,
            impl_->artifact.input_width},
        input,
        {
            impl_->options.body_type_output_name,
            impl_->options.color_output_name,
        },
        outputs,
        shapes,
        error)) {
        throw std::runtime_error(error);
    }
    const auto batch = crops.size();
    if (outputs[0].size() != batch * impl_->options.body_types.size() ||
        outputs[1].size() != batch * impl_->options.colors.size()) {
        throw std::runtime_error("attribute output volumes do not match batch and labels");
    }
    const auto elapsed =
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - started).count();
    std::vector<VehicleAttributeResult> results;
    results.reserve(batch);
    for (std::size_t index = 0; index < batch; ++index) {
        VehicleAttributeResult result;
        result.metadata.artifact_id = impl_->artifact.artifact_id;
        result.metadata.labels_version = impl_->artifact.labels_version;
        result.metadata.run_generation = crops[index].run_generation;
        result.metadata.inference_time_us = elapsed;
        result.camera_id = crops[index].camera_id;
        result.run_id = crops[index].run_id;
        result.track_id = crops[index].track_id;
        result.crop_sequence = crops[index].crop_sequence;
        result.quality_score = crops[index].quality_score;
        result.body_type = softmaxTop1(
            outputs[0].data() + index * impl_->options.body_types.size(),
            impl_->options.body_types.size(),
            impl_->options.body_types);
        result.color = softmaxTop1(
            outputs[1].data() + index * impl_->options.colors.size(),
            impl_->options.colors.size(),
            impl_->options.colors);
        results.push_back(std::move(result));
    }
    return results;
}

void TensorRtVehicleAttributeRunner::release() noexcept {
    if (!impl_) return;
    impl_->engine.release();
    impl_->initialized = false;
    impl_->artifact = {};
}

}  // namespace yolo11_server
