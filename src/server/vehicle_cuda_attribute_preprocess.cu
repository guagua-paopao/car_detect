#include "server/vehicle_cuda_attribute_preprocess.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>

namespace yolo11_server {
namespace {

bool cudaOk(cudaError_t status, const char* operation, std::string& error) {
    if (status == cudaSuccess) return true;
    error = std::string(operation) + ": " + cudaGetErrorString(status);
    return false;
}

struct DeviceCropDescriptor {
    std::size_t y_offset = 0;
    std::size_t u_offset = 0;
    std::size_t v_offset = 0;
    int width = 0;
    int height = 0;
    int y_stride_bytes = 0;
    int u_stride_bytes = 0;
    int v_stride_bytes = 0;
    int format = 1;
    int left_parity = 0;
    int top_parity = 0;
    const std::uint8_t* device_y_plane = nullptr;
    const std::uint8_t* device_u_plane = nullptr;
    const std::uint8_t* device_v_plane = nullptr;
};

constexpr int kBgr8 = 1;
constexpr int kRgb8 = 2;
constexpr int kI420 = 3;
constexpr int kDeviceI420 = 4;

__device__ __forceinline__ std::uint8_t saturateToByte(int value) {
    return static_cast<std::uint8_t>(value < 0 ? 0 : (value > 255 ? 255 : value));
}

__global__ void resizeAttributeBatchKernel(
    const std::uint8_t* source,
    const DeviceCropDescriptor* descriptors,
    int batch_size,
    float* destination,
    int destination_width,
    int destination_height) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int plane = destination_width * destination_height;
    const int pixels = batch_size * plane;
    if (index >= pixels) return;
    const int batch = index / plane;
    const int pixel = index - batch * plane;
    const int y = pixel / destination_width;
    const int x = pixel - y * destination_width;
    const DeviceCropDescriptor crop = descriptors[batch];
    const int source_x = min(crop.width - 1, x * crop.width / destination_width);
    const int source_y = min(crop.height - 1, y * crop.height / destination_height);
    const std::size_t output =
        static_cast<std::size_t>(batch) * 3U * plane + pixel;
    if (crop.format == kI420 || crop.format == kDeviceI420) {
        const auto* y_plane = crop.format == kDeviceI420
            ? crop.device_y_plane : source + crop.y_offset;
        const auto* u_plane = crop.format == kDeviceI420
            ? crop.device_u_plane : source + crop.u_offset;
        const auto* v_plane = crop.format == kDeviceI420
            ? crop.device_v_plane : source + crop.v_offset;
        const auto y_value = max(
            0,
            static_cast<int>(y_plane[
                static_cast<std::size_t>(source_y) *
                    crop.y_stride_bytes + source_x]) - 16);
        const int chroma_x = (source_x + crop.left_parity) / 2;
        const int chroma_y = (source_y + crop.top_parity) / 2;
        const auto u_offset = static_cast<std::size_t>(chroma_y) *
            crop.u_stride_bytes + chroma_x;
        const auto v_offset = static_cast<std::size_t>(chroma_y) *
            crop.v_stride_bytes + chroma_x;
        const int u_value =
            static_cast<int>(u_plane[u_offset]) - 128;
        const int v_value =
            static_cast<int>(v_plane[v_offset]) - 128;
        constexpr int shift = 20;
        constexpr int rounding = 1 << (shift - 1);
        const int scaled_y = y_value * 1220542;
        const auto red = saturateToByte(
            (scaled_y + 1673527 * v_value + rounding) >> shift);
        const auto green = saturateToByte(
            (scaled_y - 409993 * u_value - 852492 * v_value + rounding) >> shift);
        const auto blue = saturateToByte(
            (scaled_y + 2116026 * u_value + rounding) >> shift);
        destination[output] = static_cast<float>(red) / 255.0f;
        destination[output + plane] = static_cast<float>(green) / 255.0f;
        destination[output + 2U * plane] = static_cast<float>(blue) / 255.0f;
        return;
    }
    const std::uint8_t* value = source + crop.y_offset +
        static_cast<std::size_t>(source_y) * crop.y_stride_bytes +
        static_cast<std::size_t>(source_x) * 3U;
    if (crop.format == kBgr8) {
        destination[output] = static_cast<float>(value[2]) / 255.0f;
        destination[output + plane] = static_cast<float>(value[1]) / 255.0f;
        destination[output + 2U * plane] = static_cast<float>(value[0]) / 255.0f;
    }
    else {
        destination[output] = static_cast<float>(value[0]) / 255.0f;
        destination[output + plane] = static_cast<float>(value[1]) / 255.0f;
        destination[output + 2U * plane] = static_cast<float>(value[2]) / 255.0f;
    }
}

}  // namespace

struct VehicleCudaAttributePreprocessor::Impl {
    ~Impl() {
        if (device_pixels) cudaFree(device_pixels);
        if (host_pixels) cudaFreeHost(host_pixels);
        if (device_descriptors) cudaFree(device_descriptors);
        if (host_descriptors) cudaFreeHost(host_descriptors);
        if (h2d_started) cudaEventDestroy(h2d_started);
        if (h2d_finished) cudaEventDestroy(h2d_finished);
        if (kernel_finished) cudaEventDestroy(kernel_finished);
    }

    bool ensureEvents(std::string& error) {
        if (h2d_started && h2d_finished && kernel_finished) return true;
        return cudaOk(cudaEventCreate(&h2d_started),
                   "cudaEventCreate attribute h2d start", error) &&
            cudaOk(cudaEventCreate(&h2d_finished),
                "cudaEventCreate attribute h2d finish", error) &&
            cudaOk(cudaEventCreate(&kernel_finished),
                "cudaEventCreate attribute kernel finish", error);
    }

    bool reservePixels(std::size_t required, std::string& error) {
        if (pixel_capacity >= required && host_pixels && device_pixels) return true;
        if (device_pixels) cudaFree(device_pixels);
        if (host_pixels) cudaFreeHost(host_pixels);
        device_pixels = nullptr;
        host_pixels = nullptr;
        pixel_capacity = 0;
        if (!cudaOk(cudaMallocHost(reinterpret_cast<void**>(&host_pixels), required),
                "cudaMallocHost attribute pixels", error) ||
            !cudaOk(cudaMalloc(reinterpret_cast<void**>(&device_pixels), required),
                "cudaMalloc attribute pixels", error)) return false;
        pixel_capacity = required;
        return true;
    }

    bool reserveDescriptors(std::size_t required, std::string& error) {
        if (descriptor_capacity >= required && host_descriptors &&
            device_descriptors) return true;
        if (device_descriptors) cudaFree(device_descriptors);
        if (host_descriptors) cudaFreeHost(host_descriptors);
        device_descriptors = nullptr;
        host_descriptors = nullptr;
        descriptor_capacity = 0;
        const std::size_t bytes = required * sizeof(DeviceCropDescriptor);
        if (!cudaOk(cudaMallocHost(
                reinterpret_cast<void**>(&host_descriptors), bytes),
                "cudaMallocHost attribute descriptors", error) ||
            !cudaOk(cudaMalloc(
                reinterpret_cast<void**>(&device_descriptors), bytes),
                "cudaMalloc attribute descriptors", error)) return false;
        descriptor_capacity = required;
        return true;
    }

    std::uint8_t* host_pixels = nullptr;
    std::uint8_t* device_pixels = nullptr;
    std::size_t pixel_capacity = 0;
    DeviceCropDescriptor* host_descriptors = nullptr;
    DeviceCropDescriptor* device_descriptors = nullptr;
    std::size_t descriptor_capacity = 0;
    cudaEvent_t h2d_started = nullptr;
    cudaEvent_t h2d_finished = nullptr;
    cudaEvent_t kernel_finished = nullptr;
    std::size_t staged_packed_bytes = 0;
    std::size_t staged_descriptor_bytes = 0;
    int staged_batch_size = 0;
    int staged_destination_width = 0;
    int staged_destination_height = 0;
    float* staged_destination = nullptr;
    bool staged = false;
};

VehicleCudaAttributePreprocessor::VehicleCudaAttributePreprocessor()
    : impl_(std::make_unique<Impl>()) {
}

VehicleCudaAttributePreprocessor::~VehicleCudaAttributePreprocessor() = default;

bool VehicleCudaAttributePreprocessor::enqueueResizeBatch(
    const std::vector<VehicleAttributeCrop>& crops,
    float* destination,
    int destination_width,
    int destination_height,
    cudaStream_t stream,
    VehicleCudaPreprocessTiming& timing,
    std::string& error) {
    if (!stageResizeBatch(
            crops, destination, destination_width, destination_height,
            timing, error)) return false;
    return enqueueStaged(stream, timing, error);
}

bool VehicleCudaAttributePreprocessor::stageResizeBatch(
    const std::vector<VehicleAttributeCrop>& crops,
    float* destination,
    int destination_width,
    int destination_height,
    VehicleCudaPreprocessTiming& timing,
    std::string& error) {
    timing = {};
    impl_->staged = false;
    if (crops.empty() || !destination ||
        destination_width <= 0 || destination_height <= 0) {
        error = "invalid CUDA attribute preprocessing input";
        return false;
    }
    std::size_t packed_bytes = 0;
    for (const auto& item : crops) {
        if (item.device_i420_frame && item.device_i420_frame->valid() &&
            item.source_box.valid()) {
            continue;
        }
        if (item.i420_frame && item.i420_frame->valid() &&
            item.source_box.valid()) {
            const auto source = item.i420_frame->view();
            const int left = std::clamp(
                static_cast<int>(std::floor(item.source_box.x1 * source.width)),
                0, source.width - 1);
            const int top = std::clamp(
                static_cast<int>(std::floor(item.source_box.y1 * source.height)),
                0, source.height - 1);
            const int right = std::clamp(
                static_cast<int>(std::ceil(item.source_box.x2 * source.width)),
                left + 1, source.width);
            const int bottom = std::clamp(
                static_cast<int>(std::ceil(item.source_box.y2 * source.height)),
                top + 1, source.height);
            const std::size_t width = static_cast<std::size_t>(right - left);
            const std::size_t height = static_cast<std::size_t>(bottom - top);
            const std::size_t chroma_width = static_cast<std::size_t>(
                (right + 1) / 2 - left / 2);
            const std::size_t chroma_height = static_cast<std::size_t>(
                (bottom + 1) / 2 - top / 2);
            packed_bytes += width * height +
                2U * chroma_width * chroma_height;
            continue;
        }
        if (!item.crop || !item.crop->valid() || item.crop->channels != 3) {
            error = "CUDA attribute batch contains an invalid crop";
            return false;
        }
        packed_bytes += static_cast<std::size_t>(item.crop->width) * 3U *
            static_cast<std::size_t>(item.crop->height);
    }

    const auto allocation_started = std::chrono::steady_clock::now();
    if (!impl_->ensureEvents(error) ||
        (packed_bytes > 0 && !impl_->reservePixels(packed_bytes, error)) ||
        !impl_->reserveDescriptors(crops.size(), error)) return false;
    timing.allocation_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - allocation_started).count();

    const auto staging_started = std::chrono::steady_clock::now();
    std::size_t offset = 0;
    for (std::size_t batch = 0; batch < crops.size(); ++batch) {
        const auto& item = crops[batch];
        if (item.device_i420_frame && item.device_i420_frame->valid() &&
            item.source_box.valid()) {
            const auto& source = *item.device_i420_frame;
            const int left = std::clamp(
                static_cast<int>(std::floor(item.source_box.x1 * source.width)),
                0, source.width - 1);
            const int top = std::clamp(
                static_cast<int>(std::floor(item.source_box.y1 * source.height)),
                0, source.height - 1);
            const int right = std::clamp(
                static_cast<int>(std::ceil(item.source_box.x2 * source.width)),
                left + 1, source.width);
            const int bottom = std::clamp(
                static_cast<int>(std::ceil(item.source_box.y2 * source.height)),
                top + 1, source.height);
            impl_->host_descriptors[batch] = {
                0,
                0,
                0,
                right - left,
                bottom - top,
                static_cast<int>(source.y_stride_bytes),
                static_cast<int>(source.u_stride_bytes),
                static_cast<int>(source.v_stride_bytes),
                kDeviceI420,
                left & 1,
                top & 1,
                source.y_plane + static_cast<std::size_t>(top) *
                    source.y_stride_bytes + left,
                source.u_plane + static_cast<std::size_t>(top / 2) *
                    source.u_stride_bytes + left / 2,
                source.v_plane + static_cast<std::size_t>(top / 2) *
                    source.v_stride_bytes + left / 2,
            };
            continue;
        }
        if (item.i420_frame && item.i420_frame->valid() &&
            item.source_box.valid()) {
            const auto source = item.i420_frame->view();
            const int left = std::clamp(
                static_cast<int>(std::floor(item.source_box.x1 * source.width)),
                0, source.width - 1);
            const int top = std::clamp(
                static_cast<int>(std::floor(item.source_box.y1 * source.height)),
                0, source.height - 1);
            const int right = std::clamp(
                static_cast<int>(std::ceil(item.source_box.x2 * source.width)),
                left + 1, source.width);
            const int bottom = std::clamp(
                static_cast<int>(std::ceil(item.source_box.y2 * source.height)),
                top + 1, source.height);
            const int width = right - left;
            const int height = bottom - top;
            const int chroma_left = left / 2;
            const int chroma_top = top / 2;
            const int chroma_width = (right + 1) / 2 - chroma_left;
            const int chroma_height = (bottom + 1) / 2 - chroma_top;
            const std::size_t y_offset = offset;
            for (int row = 0; row < height; ++row) {
                std::memcpy(
                    impl_->host_pixels + offset +
                        static_cast<std::size_t>(row) * width,
                    source.y_plane + static_cast<std::size_t>(top + row) *
                        source.y_stride_bytes + left,
                    static_cast<std::size_t>(width));
            }
            offset += static_cast<std::size_t>(width) * height;
            const std::size_t u_offset = offset;
            for (int row = 0; row < chroma_height; ++row) {
                std::memcpy(
                    impl_->host_pixels + offset +
                        static_cast<std::size_t>(row) * chroma_width,
                    source.u_plane + static_cast<std::size_t>(chroma_top + row) *
                        source.u_stride_bytes + chroma_left,
                    static_cast<std::size_t>(chroma_width));
            }
            offset += static_cast<std::size_t>(chroma_width) * chroma_height;
            const std::size_t v_offset = offset;
            for (int row = 0; row < chroma_height; ++row) {
                std::memcpy(
                    impl_->host_pixels + offset +
                        static_cast<std::size_t>(row) * chroma_width,
                    source.v_plane + static_cast<std::size_t>(chroma_top + row) *
                        source.v_stride_bytes + chroma_left,
                    static_cast<std::size_t>(chroma_width));
            }
            offset += static_cast<std::size_t>(chroma_width) * chroma_height;
            impl_->host_descriptors[batch] = {
                y_offset,
                u_offset,
                v_offset,
                width,
                height,
                width,
                chroma_width,
                chroma_width,
                kI420,
                left & 1,
                top & 1,
            };
            continue;
        }
        const auto image = crops[batch].crop->view();
        const std::size_t row_bytes = static_cast<std::size_t>(image.width) * 3U;
        for (int row = 0; row < image.height; ++row) {
            std::memcpy(
                impl_->host_pixels + offset + static_cast<std::size_t>(row) * row_bytes,
                image.data + static_cast<std::size_t>(row) * image.row_stride_bytes,
                row_bytes);
        }
        impl_->host_descriptors[batch] = {
            offset,
            0,
            0,
            image.width,
            image.height,
            static_cast<int>(row_bytes),
            0,
            0,
            image.pixel_format == ImagePixelFormat::Bgr8 ? kBgr8 : kRgb8,
            0,
            0,
        };
        offset += row_bytes * static_cast<std::size_t>(image.height);
    }
    timing.host_staging_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - staging_started).count();
    const std::size_t descriptor_bytes =
        crops.size() * sizeof(DeviceCropDescriptor);
    timing.h2d_bytes = packed_bytes + descriptor_bytes;

    impl_->staged_packed_bytes = packed_bytes;
    impl_->staged_descriptor_bytes = descriptor_bytes;
    impl_->staged_batch_size = static_cast<int>(crops.size());
    impl_->staged_destination_width = destination_width;
    impl_->staged_destination_height = destination_height;
    impl_->staged_destination = destination;
    impl_->staged = true;
    return true;
}

bool VehicleCudaAttributePreprocessor::enqueueStaged(
    cudaStream_t stream,
    VehicleCudaPreprocessTiming& timing,
    std::string& error,
    bool record_timing) {
    if (!impl_->staged || !stream || !impl_->staged_destination ||
        impl_->staged_batch_size <= 0) {
        error = "CUDA attribute preprocessing has no staged batch";
        return false;
    }

    if ((record_timing && !cudaOk(cudaEventRecord(impl_->h2d_started, stream),
            "cudaEventRecord attribute h2d start", error)) ||
        (impl_->staged_packed_bytes > 0 && !cudaOk(cudaMemcpyAsync(
            impl_->device_pixels, impl_->host_pixels,
            impl_->staged_packed_bytes,
            cudaMemcpyHostToDevice, stream),
            "cudaMemcpyAsync attribute pixels", error)) ||
        !cudaOk(cudaMemcpyAsync(impl_->device_descriptors, impl_->host_descriptors,
            impl_->staged_descriptor_bytes, cudaMemcpyHostToDevice, stream),
            "cudaMemcpyAsync attribute descriptors", error) ||
        (record_timing && !cudaOk(cudaEventRecord(impl_->h2d_finished, stream),
            "cudaEventRecord attribute h2d finish", error))) return false;

    const int pixels = impl_->staged_batch_size *
        impl_->staged_destination_width * impl_->staged_destination_height;
    constexpr int threads = 256;
    resizeAttributeBatchKernel<<<(pixels + threads - 1) / threads, threads, 0, stream>>>(
        impl_->device_pixels,
        impl_->device_descriptors,
        impl_->staged_batch_size,
        impl_->staged_destination,
        impl_->staged_destination_width,
        impl_->staged_destination_height);
    if (!cudaOk(cudaGetLastError(), "resizeAttributeBatchKernel launch", error) ||
        (record_timing && !cudaOk(cudaEventRecord(impl_->kernel_finished, stream),
            "cudaEventRecord attribute kernel finish", error))) return false;
    return true;
}

bool VehicleCudaAttributePreprocessor::collectTiming(
    VehicleCudaPreprocessTiming& timing,
    std::string& error) const {
    float elapsed = 0.0f;
    if (!cudaOk(cudaEventElapsedTime(
            &elapsed, impl_->h2d_started, impl_->h2d_finished),
            "cudaEventElapsedTime attribute h2d", error)) return false;
    timing.h2d_ms = elapsed;
    if (!cudaOk(cudaEventElapsedTime(
            &elapsed, impl_->h2d_finished, impl_->kernel_finished),
            "cudaEventElapsedTime attribute kernel", error)) return false;
    timing.kernel_ms = elapsed;
    return true;
}

}  // namespace yolo11_server
