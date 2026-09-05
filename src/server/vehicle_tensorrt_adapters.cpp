#include "server/vehicle_tensorrt_adapters.h"
#include "server/vehicle_cuda_attribute_preprocess.h"
#include "server/vehicle_cuda_preprocess.h"
#include "server/vehicle_cuda_postprocess.h"

#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <cuda_runtime_api.h>
#include <openssl/evp.h>

#include <algorithm>
#include <array>
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
        if (!cudaOk(cudaEventCreate(&h2d_started_), "cudaEventCreate h2d start", error) ||
            !cudaOk(cudaEventCreate(&h2d_finished_), "cudaEventCreate h2d finish", error) ||
            !cudaOk(cudaEventCreate(&inference_started_), "cudaEventCreate inference start", error) ||
            !cudaOk(cudaEventCreate(&inference_finished_), "cudaEventCreate inference finish", error) ||
            !cudaOk(cudaEventCreate(&d2h_finished_), "cudaEventCreate d2h finish", error) ||
            !cudaOk(cudaEventCreate(&graph_started_), "cudaEventCreate graph start", error) ||
            !cudaOk(cudaEventCreate(&graph_finished_), "cudaEventCreate graph finish", error)) {
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
        const ImageView* gpu_source,
        const I420ImageView* gpu_i420_source,
        VehicleCudaPreprocessor* gpu_preprocessor,
        VehicleCudaPreprocessGeometry* gpu_geometry,
        VehicleCudaAttributePreprocessor* gpu_attribute_preprocessor,
        const std::vector<VehicleAttributeCrop>* gpu_attribute_crops,
        bool use_cuda_graph,
        const std::vector<std::string>& output_names,
        std::vector<std::vector<float>>& outputs,
        std::vector<nvinfer1::Dims>& output_shapes,
        TensorRtStageTiming* timing,
        std::string& error) {
        if (timing) *timing = {};
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
        const bool has_gpu_source = gpu_source || gpu_i420_source;
        const bool use_gpu_preprocess =
            has_gpu_source && gpu_preprocessor && gpu_geometry;
        const bool incomplete_gpu_preprocess =
            (has_gpu_source || gpu_preprocessor || gpu_geometry) && !use_gpu_preprocess;
        if (incomplete_gpu_preprocess) {
            error = "incomplete CUDA preprocessing configuration";
            return false;
        }
        if (gpu_source && gpu_i420_source) {
            error = "multiple CUDA preprocessing sources were provided";
            return false;
        }
        const bool has_gpu_attribute_input =
            gpu_attribute_preprocessor || gpu_attribute_crops;
        const bool use_gpu_attribute_preprocess =
            gpu_attribute_preprocessor && gpu_attribute_crops;
        if (has_gpu_attribute_input && !use_gpu_attribute_preprocess) {
            error = "incomplete CUDA attribute preprocessing configuration";
            return false;
        }
        if (use_gpu_preprocess && use_gpu_attribute_preprocess) {
            error = "multiple CUDA preprocessing modes were provided";
            return false;
        }
        const bool use_any_gpu_preprocess =
            use_gpu_preprocess || use_gpu_attribute_preprocess;
        const std::size_t input_elements = volume(input_shape);
        if (!use_any_gpu_preprocess && input.size() != input_elements) {
            error = "preprocessed input volume does not match TensorRT shape";
            return false;
        }

        const auto allocation_started = std::chrono::steady_clock::now();
        const std::size_t input_bytes = input_elements * sizeof(float);
        const auto* previous_input_device = input_device_;
        if (!reserveDevice(input_device_, input_capacity_bytes_, input_bytes,
                "input", error)) return false;
        if (previous_input_device != input_device_) destroyAttributeGraphs();
        VehicleCudaPreprocessTiming gpu_preprocess_timing;
        if (use_gpu_attribute_preprocess) {
            if (input_shape.nbDims != 4 || input_shape.d[0] <= 0 ||
                input_shape.d[1] != 3 ||
                static_cast<std::size_t>(input_shape.d[0]) !=
                    gpu_attribute_crops->size()) {
                error = "CUDA attribute resize requires matching NCHW batch input";
                return false;
            }
            if (!gpu_attribute_preprocessor->stageResizeBatch(
                    *gpu_attribute_crops,
                    static_cast<float*>(input_device_),
                    input_shape.d[3],
                    input_shape.d[2],
                    gpu_preprocess_timing,
                    error)) return false;
            if (!gpu_attribute_preprocessor->enqueueStaged(
                    stream_, gpu_preprocess_timing, error)) return false;
        }
        else if (use_gpu_preprocess) {
            if (input_shape.nbDims != 4 || input_shape.d[0] != 1 ||
                input_shape.d[1] != 3) {
                error = "CUDA letterbox requires NCHW input with batch size 1";
                return false;
            }
            const bool enqueued = gpu_i420_source
                ? gpu_preprocessor->enqueueI420Letterbox(
                    *gpu_i420_source, static_cast<float*>(input_device_),
                    input_shape.d[3], input_shape.d[2], stream_, *gpu_geometry,
                    gpu_preprocess_timing, error)
                : gpu_preprocessor->enqueueLetterbox(
                    *gpu_source, static_cast<float*>(input_device_),
                    input_shape.d[3], input_shape.d[2], stream_, *gpu_geometry,
                    gpu_preprocess_timing, error);
            if (!enqueued) return false;
        }
        else {
            if (!cudaOk(
                cudaEventRecord(h2d_started_, stream_),
                "cudaEventRecord h2d start",
                error)) return false;
            if (!cudaOk(
                cudaMemcpyAsync(
                    input_device_,
                    input.data(),
                    input_bytes,
                    cudaMemcpyHostToDevice,
                    stream_),
                "cudaMemcpyAsync input",
                error)) return false;
            if (!cudaOk(
                cudaEventRecord(h2d_finished_, stream_),
                "cudaEventRecord h2d finish",
                error)) return false;
        }
        if (!context_->setInputTensorAddress(input_name.c_str(), input_device_)) {
            error = "setInputTensorAddress failed";
            return false;
        }

        outputs.resize(output_names.size());
        output_shapes.resize(output_names.size());
        output_devices_.resize(output_names.size(), nullptr);
        output_capacity_bytes_.resize(output_names.size(), 0);
        output_hosts_.resize(output_names.size(), nullptr);
        output_host_capacity_bytes_.resize(output_names.size(), 0);
        for (std::size_t index = 0; index < output_names.size(); ++index) {
            output_shapes[index] = context_->getTensorShape(output_names[index].c_str());
            const auto count = volume(output_shapes[index]);
            if (count == 0) {
                error = "TensorRT output shape is unresolved: " + output_names[index];
                return false;
            }
            outputs[index].resize(count);
            const std::size_t output_bytes = count * sizeof(float);
            const auto* previous_output_device = output_devices_[index];
            const auto* previous_output_host = output_hosts_[index];
            if (!reserveDevice(
                    output_devices_[index], output_capacity_bytes_[index],
                    output_bytes, "output", error) ||
                !reserveHost(
                    output_hosts_[index], output_host_capacity_bytes_[index],
                    output_bytes, "output", error)) return false;
            if (previous_output_device != output_devices_[index] ||
                previous_output_host != output_hosts_[index]) {
                destroyAttributeGraphs();
            }
            if (!context_->setOutputTensorAddress(
                output_names[index].c_str(),
                output_devices_[index])) {
                error = "setOutputTensorAddress failed: " + output_names[index];
                return false;
            }
        }
        if (timing) {
            timing->allocation_ms =
                std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - allocation_started).count();
            timing->h2d_bytes = use_any_gpu_preprocess
                ? gpu_preprocess_timing.h2d_bytes : input_bytes;
            for (const auto& output : outputs) {
                timing->d2h_bytes += output.size() * sizeof(float);
            }
        }
        bool graph_used = false;
        bool graph_fallback = false;
        bool graph_built = false;
        const int graph_batch = input_shape.nbDims == 4 ? input_shape.d[0] : 0;
        auto enqueue_output = [&](bool record_timing) {
            for (std::size_t index = 0; index < outputs.size(); ++index) {
                if (!cudaOk(
                    cudaMemcpyAsync(
                        output_hosts_[index],
                        output_devices_[index],
                        outputs[index].size() * sizeof(float),
                        cudaMemcpyDeviceToHost,
                        stream_),
                    "cudaMemcpyAsync output",
                    error)) return false;
            }
            return !record_timing || cudaOk(
                cudaEventRecord(d2h_finished_, stream_),
                "cudaEventRecord d2h finish", error);
        };
        auto enqueue_inference = [&](bool record_timing, bool copy_output) {
            if (record_timing && !cudaOk(
                    cudaEventRecord(inference_started_, stream_),
                    "cudaEventRecord inference start", error)) return false;
            if (!context_->enqueueV3(stream_)) {
                error = "TensorRT enqueueV3 failed: " + g_logger.lastMessage();
                return false;
            }
            if (record_timing && !cudaOk(
                    cudaEventRecord(inference_finished_, stream_),
                    "cudaEventRecord inference finish", error)) return false;
            return !copy_output || enqueue_output(record_timing);
        };
        auto launch_attribute_graph = [&](cudaGraphExec_t graph_exec) {
            return cudaOk(cudaEventRecord(graph_started_, stream_),
                    "cudaEventRecord attribute graph start", error) &&
                cudaOk(cudaGraphLaunch(graph_exec, stream_),
                    "cudaGraphLaunch attribute", error) &&
                cudaOk(cudaEventRecord(graph_finished_, stream_),
                    "cudaEventRecord attribute graph finish", error);
        };

        if (use_cuda_graph && graph_batch > 0 && graph_batch <
                static_cast<int>(attribute_graphs_.size())) {
            auto& graph = attribute_graphs_[graph_batch];
            if (graph.exec) {
                if (!launch_attribute_graph(graph.exec)) {
                    resetAttributeGraph(graph);
                    graph.disabled = true;
                    graph_fallback = true;
                    cudaGetLastError();
                    error.clear();
                    if (!enqueue_inference(true, true)) return false;
                }
                else {
                    graph_used = true;
                    if (!enqueue_output(false)) return false;
                }
            }
            else if (graph.warmed && !graph.disabled) {
                cudaGraph_t captured = nullptr;
                const cudaError_t begin = cudaStreamBeginCapture(
                    stream_, cudaStreamCaptureModeThreadLocal);
                bool capture_ok = begin == cudaSuccess &&
                    enqueue_inference(false, false);
                const cudaError_t end = begin == cudaSuccess
                    ? cudaStreamEndCapture(stream_, &captured) : begin;
                capture_ok = capture_ok && end == cudaSuccess && captured;
                if (capture_ok) {
                    capture_ok = cudaGraphInstantiate(&graph.exec, captured, 0) ==
                        cudaSuccess && graph.exec;
                }
                if (captured) cudaGraphDestroy(captured);
                if (capture_ok && launch_attribute_graph(graph.exec)) {
                    graph_used = true;
                    graph_built = true;
                }
                else {
                    resetAttributeGraph(graph);
                    graph.disabled = true;
                    graph_fallback = true;
                    cudaGetLastError();
                    error.clear();
                    if (!enqueue_inference(true, true)) return false;
                }
                if (graph_used && !enqueue_output(false)) return false;
            }
            else {
                if (!enqueue_inference(true, true)) return false;
                graph.warmed = true;
            }
        }
        else if (!enqueue_inference(true, true)) {
            return false;
        }
        const bool synchronized =
            cudaOk(cudaStreamSynchronize(stream_), "cudaStreamSynchronize", error);
        if (synchronized && timing) {
            float elapsed = 0.0f;
            if (use_any_gpu_preprocess) {
                const bool collected = use_gpu_attribute_preprocess
                    ? gpu_attribute_preprocessor->collectTiming(
                        gpu_preprocess_timing, error)
                    : gpu_preprocessor->collectTiming(
                        gpu_preprocess_timing, error);
                if (!collected) return false;
                timing->host_staging_ms = gpu_preprocess_timing.host_staging_ms;
                timing->gpu_preprocess_ms = gpu_preprocess_timing.kernel_ms;
                timing->preprocess_ms =
                    gpu_preprocess_timing.host_staging_ms +
                    gpu_preprocess_timing.kernel_ms;
                timing->allocation_ms += gpu_preprocess_timing.allocation_ms;
                timing->h2d_ms = gpu_preprocess_timing.h2d_ms;
            }
            else {
                if (!cudaOk(
                    cudaEventElapsedTime(&elapsed, h2d_started_, h2d_finished_),
                    "cudaEventElapsedTime h2d",
                    error)) return false;
                timing->h2d_ms = elapsed;
            }
            if (graph_used) {
                if (!cudaOk(cudaEventElapsedTime(
                        &elapsed, graph_started_, graph_finished_),
                        "cudaEventElapsedTime attribute graph", error)) {
                    return false;
                }
                timing->cuda_graph_ms = elapsed;
            }
            else {
                if (!cudaOk(
                    cudaEventElapsedTime(
                        &elapsed, inference_started_, inference_finished_),
                    "cudaEventElapsedTime inference",
                    error)) return false;
                timing->inference_ms = elapsed;
                if (!cudaOk(
                    cudaEventElapsedTime(
                        &elapsed, inference_finished_, d2h_finished_),
                    "cudaEventElapsedTime d2h",
                    error)) return false;
                timing->d2h_ms = elapsed;
            }
            timing->cuda_graph_used = graph_used;
            timing->cuda_graph_fallback = graph_fallback;
            timing->cuda_graph_built = graph_built;
        }
        if (synchronized) {
            for (std::size_t index = 0; index < outputs.size(); ++index) {
                std::copy_n(
                    output_hosts_[index], outputs[index].size(),
                    outputs[index].data());
            }
        }
        return synchronized;
    }

    bool executeDetection(
        const std::string& input_name,
        const nvinfer1::Dims& input_shape,
        const ImageView& source,
        const I420ImageView* i420_source,
        VehicleCudaPreprocessor& preprocessor,
        VehicleCudaPreprocessGeometry& geometry,
        const std::string& output_name,
        VehicleCudaPostprocessor& postprocessor,
        const TensorRtDetectionOptions& options,
        std::vector<VehicleDetection>& detections,
        std::shared_ptr<const DeviceI420Image>* retained_i420_frame,
        TensorRtStageTiming* timing,
        std::string& error) {
        if (timing) *timing = {};
        if (retained_i420_frame) retained_i420_frame->reset();
        if (!engine_ || !context_ || !stream_) {
            error = "TensorRT engine is not initialized";
            return false;
        }
        if (!cudaOk(cudaSetDevice(gpu_id_), "cudaSetDevice", error)) return false;
        if (!context_->setInputShape(input_name.c_str(), input_shape)) {
            error = "TensorRT rejected input shape";
            return false;
        }
        const std::size_t input_elements = volume(input_shape);
        if (input_shape.nbDims != 4 || input_shape.d[0] != 1 ||
            input_shape.d[1] != 3 || input_elements == 0) {
            error = "CUDA detection requires resolved NCHW input with batch size 1";
            return false;
        }
        const int source_width = i420_source ? i420_source->width : source.width;
        const int source_height = i420_source ? i420_source->height : source.height;
        const bool source_is_i420 = i420_source != nullptr;
        if (graph_exec_ && !graphMatches(
                input_elements, source_width, source_height, source_is_i420)) {
            destroyGraph();
            graph_warmed_ = false;
            graph_capture_disabled_ = false;
        }

        const auto allocation_started = std::chrono::steady_clock::now();
        const std::size_t input_bytes = input_elements * sizeof(float);
        if (!reserveDevice(input_device_, input_capacity_bytes_, input_bytes,
                "input", error)) return false;
        output_devices_.resize(1, nullptr);
        output_capacity_bytes_.resize(1, 0);
        const nvinfer1::Dims output_shape = context_->getTensorShape(output_name.c_str());
        const std::size_t output_elements = volume(output_shape);
        if (output_elements == 0) {
            error = "TensorRT output shape is unresolved: " + output_name;
            return false;
        }
        if (!reserveDevice(
                output_devices_[0], output_capacity_bytes_[0],
                output_elements * sizeof(float), "output", error)) return false;
        if (!context_->setInputTensorAddress(input_name.c_str(), input_device_) ||
            !context_->setOutputTensorAddress(output_name.c_str(), output_devices_[0])) {
            error = "failed to set TensorRT detection tensor addresses";
            return false;
        }
        if (timing) {
            timing->allocation_ms =
                std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - allocation_started).count();
        }

        VehicleCudaPreprocessTiming preprocess_timing;
        VehicleCudaPostprocessTiming postprocess_timing;
        const bool staged = i420_source
            ? preprocessor.stageI420Letterbox(
                *i420_source, input_shape.d[3], input_shape.d[2], geometry,
                preprocess_timing, error)
            : preprocessor.stageLetterbox(
                source, input_shape.d[3], input_shape.d[2], geometry,
                preprocess_timing, error);
        if (!staged) return false;

        const auto enqueue_pipeline = [&](
            bool record_timing, std::string& pipeline_error) {
            if (!preprocessor.enqueueStaged(
                    static_cast<float*>(input_device_), stream_,
                    preprocess_timing, pipeline_error, record_timing)) {
                return false;
            }
            if (record_timing &&
                !cudaOk(cudaEventRecord(inference_started_, stream_),
                    "cudaEventRecord inference start", pipeline_error)) return false;
            if (!context_->enqueueV3(stream_)) {
                pipeline_error =
                    "TensorRT enqueueV3 failed: " + g_logger.lastMessage();
                return false;
            }
            if (record_timing && !cudaOk(cudaEventRecord(inference_finished_, stream_),
                    "cudaEventRecord inference finish", pipeline_error)) {
                return false;
            }
            return postprocessor.enqueue(
                static_cast<const float*>(output_devices_[0]),
                output_shape,
                geometry,
                source_width,
                source_height,
                static_cast<int>(options.vehicle_classes.size()),
                options.confidence_threshold,
                options.nms_threshold,
                stream_,
                postprocess_timing,
                pipeline_error,
                record_timing);
        };

        const auto launch_graph = [&](std::string& launch_error) {
            return cudaOk(cudaEventRecord(graph_started_, stream_),
                    "cudaEventRecord graph start", launch_error) &&
                cudaOk(cudaGraphLaunch(graph_exec_, stream_),
                    "cudaGraphLaunch detection", launch_error) &&
                cudaOk(cudaEventRecord(graph_finished_, stream_),
                    "cudaEventRecord graph finish", launch_error);
        };

        bool graph_used = false;
        bool graph_fallback = false;
        if (options.use_cuda_graph && graph_exec_) {
            if (launch_graph(error)) {
                graph_used = true;
            }
            else {
                destroyGraph();
                graph_capture_disabled_ = true;
                graph_fallback = true;
                cudaGetLastError();
                error.clear();
                if (!enqueue_pipeline(true, error)) return false;
            }
        }
        else if (options.use_cuda_graph && graph_warmed_ &&
                 !graph_capture_disabled_) {
            cudaGraph_t captured_graph = nullptr;
            const cudaError_t begin_status = cudaStreamBeginCapture(
                stream_, cudaStreamCaptureModeThreadLocal);
            bool capture_ok = begin_status == cudaSuccess;
            std::string capture_error;
            if (capture_ok) capture_ok = enqueue_pipeline(false, capture_error);
            const cudaError_t end_status = begin_status == cudaSuccess
                ? cudaStreamEndCapture(stream_, &captured_graph)
                : begin_status;
            capture_ok = capture_ok && end_status == cudaSuccess && captured_graph;
            if (capture_ok) {
                const cudaError_t instantiate_status = cudaGraphInstantiate(
                    &graph_exec_, captured_graph, 0);
                capture_ok = instantiate_status == cudaSuccess && graph_exec_;
            }
            if (captured_graph) cudaGraphDestroy(captured_graph);
            if (capture_ok) {
                graph_input_elements_ = input_elements;
                graph_source_width_ = source_width;
                graph_source_height_ = source_height;
                graph_source_is_i420_ = source_is_i420;
                if (!launch_graph(error)) {
                    destroyGraph();
                    return false;
                }
                graph_used = true;
            }
            else {
                destroyGraph();
                graph_capture_disabled_ = true;
                graph_fallback = true;
                cudaGetLastError();
                if (!enqueue_pipeline(true, error)) return false;
            }
        }
        else {
            if (!enqueue_pipeline(true, error)) return false;
        }
        if (source_is_i420 && retained_i420_frame) {
            std::string retain_error;
            if (!preprocessor.retainStagedI420(
                    stream_, *retained_i420_frame,
                    preprocess_timing, retain_error)) {
                retained_i420_frame->reset();
                if (timing) timing->device_frame_retain_fallback = true;
                cudaGetLastError();
            }
            else if (timing) {
                timing->device_frame_retained = true;
            }
        }
        if (!cudaOk(cudaStreamSynchronize(stream_),
                "cudaStreamSynchronize", error)) return false;
        if (options.use_cuda_graph && !graph_used && !graph_capture_disabled_) {
            graph_warmed_ = true;
            graph_input_elements_ = input_elements;
            graph_source_width_ = source_width;
            graph_source_height_ = source_height;
            graph_source_is_i420_ = source_is_i420;
        }
        if (!graph_used && !preprocessor.collectTiming(preprocess_timing, error)) {
            return false;
        }
        if (graph_used && !preprocessor.collectRetainedI420Timing(
                preprocess_timing, error)) return false;
        const auto collect_started = std::chrono::steady_clock::now();
        if (!postprocessor.collect(
                options.vehicle_classes, detections, postprocess_timing, error,
                !graph_used)) {
            return false;
        }
        const double host_collect_ms =
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - collect_started).count();

        if (timing) {
            float elapsed = 0.0f;
            if (graph_used) {
                if (!cudaOk(cudaEventElapsedTime(
                        &elapsed, graph_started_, graph_finished_),
                        "cudaEventElapsedTime graph", error)) return false;
                timing->cuda_graph_ms = elapsed;
            }
            else {
                if (!cudaOk(cudaEventElapsedTime(
                        &elapsed, inference_started_, inference_finished_),
                        "cudaEventElapsedTime inference", error)) return false;
                timing->inference_ms = elapsed;
            }
            timing->host_staging_ms = preprocess_timing.host_staging_ms;
            timing->gpu_preprocess_ms = preprocess_timing.kernel_ms;
            timing->preprocess_ms =
                preprocess_timing.host_staging_ms + preprocess_timing.kernel_ms;
            timing->h2d_ms = preprocess_timing.h2d_ms;
            timing->h2d_bytes = preprocess_timing.h2d_bytes;
            timing->device_frame_copy_ms =
                preprocess_timing.device_frame_copy_ms;
            timing->device_frame_copy_bytes =
                preprocess_timing.device_frame_copy_bytes;
            timing->gpu_postprocess_ms = postprocess_timing.kernel_ms;
            timing->postprocess_ms = postprocess_timing.kernel_ms + host_collect_ms;
            timing->d2h_ms = postprocess_timing.d2h_ms;
            timing->d2h_bytes = postprocess_timing.d2h_bytes;
            timing->cuda_graph_used = graph_used;
            timing->cuda_graph_fallback = graph_fallback;
            timing->allocation_ms +=
                preprocess_timing.allocation_ms + postprocess_timing.allocation_ms;
        }
        return true;
    }

    void release() noexcept {
        destroyGraph();
        destroyAttributeGraphs();
        if (stream_) {
            cudaStreamSynchronize(stream_);
            cudaStreamDestroy(stream_);
            stream_ = nullptr;
        }
        if (input_device_) cudaFree(input_device_);
        input_device_ = nullptr;
        input_capacity_bytes_ = 0;
        for (void* pointer : output_devices_) {
            if (pointer) cudaFree(pointer);
        }
        output_devices_.clear();
        output_capacity_bytes_.clear();
        for (float* pointer : output_hosts_) {
            if (pointer) cudaFreeHost(pointer);
        }
        output_hosts_.clear();
        output_host_capacity_bytes_.clear();
        if (h2d_started_) {
            cudaEventDestroy(h2d_started_);
            h2d_started_ = nullptr;
        }
        if (h2d_finished_) {
            cudaEventDestroy(h2d_finished_);
            h2d_finished_ = nullptr;
        }
        if (inference_finished_) {
            cudaEventDestroy(inference_finished_);
            inference_finished_ = nullptr;
        }
        if (inference_started_) {
            cudaEventDestroy(inference_started_);
            inference_started_ = nullptr;
        }
        if (d2h_finished_) {
            cudaEventDestroy(d2h_finished_);
            d2h_finished_ = nullptr;
        }
        if (graph_finished_) {
            cudaEventDestroy(graph_finished_);
            graph_finished_ = nullptr;
        }
        if (graph_started_) {
            cudaEventDestroy(graph_started_);
            graph_started_ = nullptr;
        }
        delete context_;
        context_ = nullptr;
        delete engine_;
        engine_ = nullptr;
        delete runtime_;
        runtime_ = nullptr;
    }

private:
    struct AttributeGraphEntry {
        cudaGraphExec_t exec = nullptr;
        bool warmed = false;
        bool disabled = false;
    };

    void resetAttributeGraph(AttributeGraphEntry& graph) noexcept {
        if (graph.exec) cudaGraphExecDestroy(graph.exec);
        graph = {};
    }

    void destroyAttributeGraphs() noexcept {
        for (auto& graph : attribute_graphs_) {
            resetAttributeGraph(graph);
        }
    }

    bool graphMatches(
        std::size_t input_elements,
        int source_width,
        int source_height,
        bool source_is_i420) const noexcept {
        return graph_exec_ && graph_input_elements_ == input_elements &&
            graph_source_width_ == source_width &&
            graph_source_height_ == source_height &&
            graph_source_is_i420_ == source_is_i420;
    }

    void destroyGraph() noexcept {
        if (graph_exec_) {
            cudaGraphExecDestroy(graph_exec_);
            graph_exec_ = nullptr;
        }
    }

    bool reserveDevice(
        void*& pointer,
        std::size_t& capacity,
        std::size_t required,
        std::string_view label,
        std::string& error) {
        if (capacity >= required && pointer) return true;
        if (pointer) {
            cudaFree(pointer);
            pointer = nullptr;
            capacity = 0;
        }
        if (!cudaOk(cudaMalloc(&pointer, required),
                std::string("cudaMalloc ") + std::string(label), error)) {
            return false;
        }
        capacity = required;
        return true;
    }

    bool reserveHost(
        float*& pointer,
        std::size_t& capacity,
        std::size_t required,
        std::string_view label,
        std::string& error) {
        if (capacity >= required && pointer) return true;
        if (pointer) {
            cudaFreeHost(pointer);
            pointer = nullptr;
            capacity = 0;
        }
        if (!cudaOk(cudaMallocHost(reinterpret_cast<void**>(&pointer), required),
                std::string("cudaMallocHost ") + std::string(label), error)) {
            return false;
        }
        capacity = required;
        return true;
    }

    int gpu_id_ = 0;
    nvinfer1::IRuntime* runtime_ = nullptr;
    nvinfer1::ICudaEngine* engine_ = nullptr;
    nvinfer1::IExecutionContext* context_ = nullptr;
    cudaStream_t stream_ = nullptr;
    cudaEvent_t h2d_started_ = nullptr;
    cudaEvent_t h2d_finished_ = nullptr;
    cudaEvent_t inference_started_ = nullptr;
    cudaEvent_t inference_finished_ = nullptr;
    cudaEvent_t d2h_finished_ = nullptr;
    cudaEvent_t graph_started_ = nullptr;
    cudaEvent_t graph_finished_ = nullptr;
    void* input_device_ = nullptr;
    std::size_t input_capacity_bytes_ = 0;
    std::vector<void*> output_devices_;
    std::vector<std::size_t> output_capacity_bytes_;
    std::vector<float*> output_hosts_;
    std::vector<std::size_t> output_host_capacity_bytes_;
    cudaGraphExec_t graph_exec_ = nullptr;
    std::size_t graph_input_elements_ = 0;
    int graph_source_width_ = 0;
    int graph_source_height_ = 0;
    bool graph_source_is_i420_ = false;
    bool graph_warmed_ = false;
    bool graph_capture_disabled_ = false;
    std::array<AttributeGraphEntry, 17> attribute_graphs_{};
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
    // Preserve original anchor order for equal scores. CUB radix sort is stable,
    // so this also defines one deterministic tie rule across CPU and GPU NMS.
    std::stable_sort(
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
    VehicleCudaPreprocessor preprocessor;
    VehicleCudaPostprocessor postprocessor;
    std::vector<std::vector<float>> outputs;
    std::vector<nvinfer1::Dims> shapes;
    TensorRtStageTiming last_timing;
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
    const bool has_i420 = request.i420_frame.valid();
    const bool has_rgb = request.frame.valid() && request.frame.channels == 3;
    if (!has_i420 && !has_rgb) {
        throw std::invalid_argument(
            "detection request requires a valid 3-channel or I420 frame");
    }
    const auto started = std::chrono::steady_clock::now();
    PreparedImage prepared;
    VehicleCudaPreprocessGeometry gpu_geometry;
    const ImageView* gpu_source = nullptr;
    const I420ImageView* gpu_i420_source = nullptr;
    VehicleCudaPreprocessor* gpu_preprocessor = nullptr;
    VehicleCudaPreprocessGeometry* geometry_output = nullptr;
    double cpu_preprocess_ms = 0.0;
    if (impl_->options.use_gpu_preprocess) {
        if (has_i420) gpu_i420_source = &request.i420_frame;
        else gpu_source = &request.frame;
        gpu_preprocessor = &impl_->preprocessor;
        geometry_output = &gpu_geometry;
    }
    else {
        if (!has_rgb) {
            throw std::invalid_argument(
                "CPU preprocessing requires a valid 3-channel fallback frame");
        }
        const auto preprocess_started = std::chrono::steady_clock::now();
        prepared = letterbox(
            request.frame,
            impl_->artifact.input_width,
            impl_->artifact.input_height);
        cpu_preprocess_ms =
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - preprocess_started).count();
    }
    VehicleDetectionResult result;
    result.metadata.artifact_id = impl_->artifact.artifact_id;
    result.metadata.labels_version = impl_->artifact.labels_version;
    result.metadata.run_generation = request.run_generation;
    result.camera_id = request.camera_id;
    result.run_id = request.run_id;
    result.frame_sequence = request.frame_sequence;
    result.captured_at_ms = request.captured_at_ms;
    std::string error;
    const nvinfer1::Dims4 input_shape{
        1,
        3,
        impl_->artifact.input_height,
        impl_->artifact.input_width};
    if (impl_->options.use_gpu_postprocess) {
        if (!impl_->options.use_gpu_preprocess) {
            throw std::runtime_error(
                "GPU postprocessing currently requires GPU preprocessing");
        }
        if (!impl_->engine.executeDetection(
                impl_->options.input_tensor_name,
                input_shape,
                request.frame,
                has_i420 ? &request.i420_frame : nullptr,
                impl_->preprocessor,
                gpu_geometry,
                impl_->options.output_tensor_name,
                impl_->postprocessor,
                impl_->options,
                result.detections,
                impl_->options.retain_i420_device_frame
                    ? &result.device_i420_frame : nullptr,
                &impl_->last_timing,
                error)) {
            throw std::runtime_error(error);
        }
    }
    else {
        if (!impl_->engine.execute(
                impl_->options.input_tensor_name,
                input_shape,
                prepared.tensor,
                gpu_source,
                gpu_i420_source,
                gpu_preprocessor,
                geometry_output,
                nullptr,
                nullptr,
                false,
                {impl_->options.output_tensor_name},
                impl_->outputs,
                impl_->shapes,
                &impl_->last_timing,
                error)) {
            throw std::runtime_error(error);
        }
        if (impl_->options.use_gpu_preprocess) {
            prepared.scale = gpu_geometry.scale;
            prepared.pad_x = gpu_geometry.pad_x;
            prepared.pad_y = gpu_geometry.pad_y;
        }
        else {
            impl_->last_timing.preprocess_ms = cpu_preprocess_ms;
        }
        const auto postprocess_started = std::chrono::steady_clock::now();
        result.detections = decodeYolo(
            impl_->outputs.front(),
            impl_->shapes.front(),
            prepared,
            has_i420 ? ImageView{
                nullptr,
                request.i420_frame.width,
                request.i420_frame.height,
                0,
                0,
                ImagePixelFormat::Bgr8} : request.frame,
            impl_->artifact.input_width,
            impl_->artifact.input_height,
            impl_->options);
        impl_->last_timing.postprocess_ms =
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - postprocess_started).count();
    }
    impl_->last_timing.total_ms =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - started).count();
    result.metadata.inference_time_us =
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - started).count();
    return result;
}

TensorRtStageTiming TensorRtVehicleDetectionRunner::lastTiming() const noexcept {
    return impl_ ? impl_->last_timing : TensorRtStageTiming{};
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
    VehicleCudaAttributePreprocessor preprocessor;
    TensorRtStageTiming last_timing;
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
        const bool valid_pixels = crop.crop && crop.crop->valid() &&
            crop.crop->channels == 3;
        const bool valid_i420_roi = crop.i420_frame &&
            crop.i420_frame->valid() && crop.source_box.valid();
        const bool valid_device_i420_roi = crop.device_i420_frame &&
            crop.device_i420_frame->valid() && crop.source_box.valid();
        if ((!impl_->options.use_gpu_preprocess && !valid_pixels) ||
            (impl_->options.use_gpu_preprocess &&
                !valid_pixels && !valid_i420_roi &&
                !valid_device_i420_roi)) {
            throw std::invalid_argument("attribute batch contains an invalid crop");
        }
    }
    const auto started = std::chrono::steady_clock::now();
    std::vector<float> input;
    double cpu_preprocess_ms = 0.0;
    if (!impl_->options.use_gpu_preprocess) {
        const auto preprocess_started = std::chrono::steady_clock::now();
        input = resizeBatch(
            crops,
            impl_->artifact.input_width,
            impl_->artifact.input_height);
        cpu_preprocess_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - preprocess_started).count();
    }
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
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        impl_->options.use_gpu_preprocess ? &impl_->preprocessor : nullptr,
        impl_->options.use_gpu_preprocess ? &crops : nullptr,
        impl_->options.use_cuda_graph,
        {
            impl_->options.body_type_output_name,
            impl_->options.color_output_name,
        },
        outputs,
        shapes,
        &impl_->last_timing,
        error)) {
        throw std::runtime_error(error);
    }
    if (!impl_->options.use_gpu_preprocess) {
        impl_->last_timing.preprocess_ms = cpu_preprocess_ms;
    }
    const auto batch = crops.size();
    if (outputs[0].size() != batch * impl_->options.body_types.size() ||
        outputs[1].size() != batch * impl_->options.colors.size()) {
        throw std::runtime_error("attribute output volumes do not match batch and labels");
    }
    const auto elapsed =
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - started).count();
    const auto postprocess_started = std::chrono::steady_clock::now();
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
    impl_->last_timing.postprocess_ms =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - postprocess_started).count();
    impl_->last_timing.total_ms =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - started).count();
    return results;
}

TensorRtStageTiming TensorRtVehicleAttributeRunner::lastTiming() const noexcept {
    return impl_ ? impl_->last_timing : TensorRtStageTiming{};
}

void TensorRtVehicleAttributeRunner::release() noexcept {
    if (!impl_) return;
    impl_->engine.release();
    impl_->initialized = false;
    impl_->artifact = {};
}

}  // namespace yolo11_server
