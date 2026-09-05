#include "server/vehicle_cuda_postprocess.h"

#include <algorithm>
#include <chrono>
#include <string_view>

#include <cub/device/device_radix_sort.cuh>

namespace yolo11_server {

namespace {

struct GpuDetection {
    float x1;
    float y1;
    float x2;
    float y2;
    float confidence;
    int class_id;
};

bool cudaOk(cudaError_t status, std::string_view action, std::string& error) {
    if (status == cudaSuccess) return true;
    error = std::string(action) + ": " + cudaGetErrorString(status);
    return false;
}

__global__ void decodeKernel(
    const float* output,
    GpuDetection* candidates,
    float* scores,
    int anchors,
    int features,
    bool feature_first,
    int class_count,
    float confidence_threshold,
    float scale,
    float pad_x,
    float pad_y,
    int source_width,
    int source_height) {
    const int anchor = blockIdx.x * blockDim.x + threadIdx.x;
    if (anchor >= anchors) return;
    const auto at = [&](int feature) {
        return feature_first
            ? output[feature * anchors + anchor]
            : output[anchor * features + feature];
    };
    int best_class = 0;
    float confidence = at(4);
    for (int index = 1; index < class_count; ++index) {
        const float value = at(4 + index);
        if (value > confidence) {
            confidence = value;
            best_class = index;
        }
    }
    GpuDetection candidate{};
    candidate.confidence = -1.0f;
    candidate.class_id = best_class;
    if (confidence >= confidence_threshold) {
        const float center_x = at(0);
        const float center_y = at(1);
        const float width = at(2);
        const float height = at(3);
        candidate.x1 = fmaxf(
            0.0f,
            (center_x - width / 2.0f - pad_x) /
                scale / static_cast<float>(source_width));
        candidate.y1 = fmaxf(
            0.0f,
            (center_y - height / 2.0f - pad_y) /
                scale / static_cast<float>(source_height));
        candidate.x2 = fminf(
            1.0f,
            (center_x + width / 2.0f - pad_x) /
                scale / static_cast<float>(source_width));
        candidate.y2 = fminf(
            1.0f,
            (center_y + height / 2.0f - pad_y) /
                scale / static_cast<float>(source_height));
        if (candidate.x2 > candidate.x1 && candidate.y2 > candidate.y1) {
            candidate.confidence = confidence;
        }
    }
    candidates[anchor] = candidate;
    scores[anchor] = candidate.confidence;
}

__device__ float overlap(const GpuDetection& first, const GpuDetection& second) {
    const float left = fmaxf(first.x1, second.x1);
    const float top = fmaxf(first.y1, second.y1);
    const float right = fminf(first.x2, second.x2);
    const float bottom = fminf(first.y2, second.y2);
    const float intersection =
        fmaxf(0.0f, right - left) * fmaxf(0.0f, bottom - top);
    const float first_area = (first.x2 - first.x1) * (first.y2 - first.y1);
    const float second_area = (second.x2 - second.x1) * (second.y2 - second.y1);
    const float denominator = first_area + second_area - intersection;
    return denominator > 0.0f ? intersection / denominator : 0.0f;
}

__global__ void greedyNmsKernel(
    const GpuDetection* sorted,
    GpuDetection* kept,
    int* kept_count,
    int anchors,
    float confidence_threshold,
    float nms_threshold) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    int count = 0;
    for (int index = 0; index < anchors; ++index) {
        const GpuDetection candidate = sorted[index];
        if (candidate.confidence < confidence_threshold) break;
        bool suppressed = false;
        for (int kept_index = 0; kept_index < count; ++kept_index) {
            if (kept[kept_index].class_id == candidate.class_id &&
                overlap(kept[kept_index], candidate) > nms_threshold) {
                suppressed = true;
                break;
            }
        }
        if (!suppressed) kept[count++] = candidate;
    }
    *kept_count = count;
}

__global__ void copyCompactToMappedHostKernel(
    const GpuDetection* kept,
    const int* kept_count,
    GpuDetection* mapped_host_kept,
    int* mapped_host_count,
    int capacity) {
    const int count = *kept_count;
    if (threadIdx.x == 0) *mapped_host_count = count;
    for (int index = threadIdx.x;
         index < count && index < capacity;
         index += blockDim.x) {
        mapped_host_kept[index] = kept[index];
    }
}

}  // namespace

struct VehicleCudaPostprocessor::Impl {
    ~Impl() {
        if (candidates_in) cudaFree(candidates_in);
        if (candidates_out) cudaFree(candidates_out);
        if (scores_in) cudaFree(scores_in);
        if (scores_out) cudaFree(scores_out);
        if (kept) cudaFree(kept);
        if (kept_count) cudaFree(kept_count);
        if (sort_scratch) cudaFree(sort_scratch);
        if (host_kept) cudaFreeHost(host_kept);
        if (host_kept_count) cudaFreeHost(host_kept_count);
        if (postprocess_started) cudaEventDestroy(postprocess_started);
        if (postprocess_finished) cudaEventDestroy(postprocess_finished);
        if (d2h_started) cudaEventDestroy(d2h_started);
        if (d2h_finished) cudaEventDestroy(d2h_finished);
    }

    bool ensureEvents(std::string& error) {
        if (postprocess_started && postprocess_finished &&
            d2h_started && d2h_finished) return true;
        return cudaOk(cudaEventCreate(&postprocess_started),
                "cudaEventCreate postprocess start", error) &&
            cudaOk(cudaEventCreate(&postprocess_finished),
                "cudaEventCreate postprocess finish", error) &&
            cudaOk(cudaEventCreate(&d2h_started),
                "cudaEventCreate postprocess d2h start", error) &&
            cudaOk(cudaEventCreate(&d2h_finished),
                "cudaEventCreate postprocess d2h finish", error);
    }

    bool reserve(int required, std::string& error) {
        if (capacity >= required && candidates_in && candidates_out &&
            scores_in && scores_out && kept && kept_count && sort_scratch &&
            host_kept && host_kept_count && mapped_host_kept &&
            mapped_host_kept_count) return true;
        if (candidates_in) cudaFree(candidates_in);
        if (candidates_out) cudaFree(candidates_out);
        if (scores_in) cudaFree(scores_in);
        if (scores_out) cudaFree(scores_out);
        if (kept) cudaFree(kept);
        if (kept_count) cudaFree(kept_count);
        if (sort_scratch) cudaFree(sort_scratch);
        if (host_kept) cudaFreeHost(host_kept);
        if (host_kept_count) cudaFreeHost(host_kept_count);
        candidates_in = nullptr;
        candidates_out = nullptr;
        scores_in = nullptr;
        scores_out = nullptr;
        kept = nullptr;
        kept_count = nullptr;
        sort_scratch = nullptr;
        sort_scratch_bytes = 0;
        host_kept = nullptr;
        host_kept_count = nullptr;
        mapped_host_kept = nullptr;
        mapped_host_kept_count = nullptr;
        capacity = 0;
        const std::size_t bytes = static_cast<std::size_t>(required) * sizeof(GpuDetection);
        if (!cudaOk(cudaMalloc(reinterpret_cast<void**>(&candidates_in), bytes),
                "cudaMalloc postprocess candidates input", error) ||
            !cudaOk(cudaMalloc(reinterpret_cast<void**>(&candidates_out), bytes),
                "cudaMalloc postprocess candidates output", error) ||
            !cudaOk(cudaMalloc(reinterpret_cast<void**>(&scores_in),
                static_cast<std::size_t>(required) * sizeof(float)),
                "cudaMalloc postprocess scores input", error) ||
            !cudaOk(cudaMalloc(reinterpret_cast<void**>(&scores_out),
                static_cast<std::size_t>(required) * sizeof(float)),
                "cudaMalloc postprocess scores output", error) ||
            !cudaOk(cudaMalloc(reinterpret_cast<void**>(&kept), bytes),
                "cudaMalloc postprocess kept", error) ||
            !cudaOk(cudaMalloc(reinterpret_cast<void**>(&kept_count), sizeof(int)),
                "cudaMalloc postprocess count", error) ||
            !cudaOk(cudaHostAlloc(reinterpret_cast<void**>(&host_kept), bytes,
                cudaHostAllocMapped), "cudaHostAlloc mapped postprocess kept", error) ||
            !cudaOk(cudaHostAlloc(reinterpret_cast<void**>(&host_kept_count), sizeof(int),
                cudaHostAllocMapped), "cudaHostAlloc mapped postprocess count", error) ||
            !cudaOk(cudaHostGetDevicePointer(
                reinterpret_cast<void**>(&mapped_host_kept), host_kept, 0),
                "cudaHostGetDevicePointer postprocess kept", error) ||
            !cudaOk(cudaHostGetDevicePointer(
                reinterpret_cast<void**>(&mapped_host_kept_count), host_kept_count, 0),
                "cudaHostGetDevicePointer postprocess count", error)) {
            return false;
        }
        if (!cudaOk(cub::DeviceRadixSort::SortPairsDescending(
                nullptr,
                sort_scratch_bytes,
                scores_in,
                scores_out,
                candidates_in,
                candidates_out,
                required), "query CUB detection sort storage", error) ||
            !cudaOk(cudaMalloc(&sort_scratch, sort_scratch_bytes),
                "cudaMalloc CUB detection sort storage", error)) {
            return false;
        }
        capacity = required;
        return true;
    }

    GpuDetection* candidates_in = nullptr;
    GpuDetection* candidates_out = nullptr;
    float* scores_in = nullptr;
    float* scores_out = nullptr;
    GpuDetection* kept = nullptr;
    int* kept_count = nullptr;
    void* sort_scratch = nullptr;
    std::size_t sort_scratch_bytes = 0;
    GpuDetection* host_kept = nullptr;
    int* host_kept_count = nullptr;
    GpuDetection* mapped_host_kept = nullptr;
    int* mapped_host_kept_count = nullptr;
    int capacity = 0;
    int active_anchors = 0;
    cudaEvent_t postprocess_started = nullptr;
    cudaEvent_t postprocess_finished = nullptr;
    cudaEvent_t d2h_started = nullptr;
    cudaEvent_t d2h_finished = nullptr;
};

VehicleCudaPostprocessor::VehicleCudaPostprocessor()
    : impl_(std::make_unique<Impl>()) {
}

VehicleCudaPostprocessor::~VehicleCudaPostprocessor() = default;

bool VehicleCudaPostprocessor::enqueue(
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
    bool record_timing) {
    timing = {};
    if (!output || !stream || shape.nbDims != 3 || shape.d[0] != 1 ||
        source_width <= 0 || source_height <= 0 || class_count <= 0) {
        error = "invalid CUDA detection postprocess input";
        return false;
    }
    const int features = 4 + class_count;
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
        error = "CUDA detection output feature count does not match labels";
        return false;
    }
    const auto allocation_started = std::chrono::steady_clock::now();
    if (!impl_->ensureEvents(error) || !impl_->reserve(anchors, error)) return false;
    timing.allocation_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - allocation_started).count();
    impl_->active_anchors = anchors;

    if (record_timing && !cudaOk(cudaEventRecord(impl_->postprocess_started, stream),
            "cudaEventRecord postprocess start", error)) return false;
    constexpr int threads = 256;
    const int blocks = (anchors + threads - 1) / threads;
    decodeKernel<<<blocks, threads, 0, stream>>>(
        output, impl_->candidates_in, impl_->scores_in,
        anchors, features, feature_first, class_count,
        confidence_threshold, geometry.scale, geometry.pad_x, geometry.pad_y,
        source_width, source_height);
    if (!cudaOk(cudaGetLastError(), "decodeKernel launch", error)) return false;
    if (!cudaOk(cub::DeviceRadixSort::SortPairsDescending(
            impl_->sort_scratch,
            impl_->sort_scratch_bytes,
            impl_->scores_in,
            impl_->scores_out,
            impl_->candidates_in,
            impl_->candidates_out,
            anchors,
            0,
            sizeof(float) * 8,
            stream), "CUB detection sort", error)) return false;
    greedyNmsKernel<<<1, 1, 0, stream>>>(
        impl_->candidates_out, impl_->kept, impl_->kept_count, anchors,
        confidence_threshold, nms_threshold);
    if (!cudaOk(cudaGetLastError(), "greedyNmsKernel launch", error)) {
        return false;
    }
    if (record_timing &&
        (!cudaOk(cudaEventRecord(impl_->postprocess_finished, stream),
            "cudaEventRecord postprocess finish", error) ||
         !cudaOk(cudaEventRecord(impl_->d2h_started, stream),
            "cudaEventRecord postprocess d2h start", error))) return false;
    copyCompactToMappedHostKernel<<<1, threads, 0, stream>>>(
        impl_->kept,
        impl_->kept_count,
        impl_->mapped_host_kept,
        impl_->mapped_host_kept_count,
        anchors);
    if (!cudaOk(cudaGetLastError(),
            "copyCompactToMappedHostKernel launch", error)) return false;
    if (record_timing && !cudaOk(cudaEventRecord(impl_->d2h_finished, stream),
            "cudaEventRecord postprocess d2h finish", error)) return false;
    return true;
}

bool VehicleCudaPostprocessor::collect(
    const std::vector<std::string>& labels,
    std::vector<VehicleDetection>& detections,
    VehicleCudaPostprocessTiming& timing,
    std::string& error,
    bool collect_timing) const {
    if (collect_timing) {
        float elapsed = 0.0f;
        if (!cudaOk(cudaEventElapsedTime(
                &elapsed, impl_->postprocess_started, impl_->postprocess_finished),
                "cudaEventElapsedTime postprocess", error)) return false;
        timing.kernel_ms = elapsed;
        if (!cudaOk(cudaEventElapsedTime(
                &elapsed, impl_->d2h_started, impl_->d2h_finished),
                "cudaEventElapsedTime postprocess d2h", error)) return false;
        timing.d2h_ms = elapsed;
    }
    const int count = *impl_->host_kept_count;
    if (count < 0 || count > impl_->active_anchors) {
        error = "CUDA postprocess returned an invalid detection count";
        return false;
    }
    timing.d2h_bytes = sizeof(int) +
        static_cast<std::size_t>(count) * sizeof(GpuDetection);
    detections.clear();
    detections.reserve(static_cast<std::size_t>(count));
    for (int index = 0; index < count; ++index) {
        const auto& source = impl_->host_kept[index];
        if (source.class_id < 0 ||
            source.class_id >= static_cast<int>(labels.size())) {
            error = "CUDA postprocess returned an invalid class id";
            return false;
        }
        detections.push_back({
            {source.x1, source.y1, source.x2, source.y2},
            labels[static_cast<std::size_t>(source.class_id)],
            source.confidence,
        });
    }
    return true;
}

}  // namespace yolo11_server
