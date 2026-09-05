#include "server/vehicle_cuda_preprocess.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <string_view>
#include <vector>

namespace yolo11_server {

namespace {

bool cudaOk(cudaError_t status, std::string_view action, std::string& error) {
    if (status == cudaSuccess) return true;
    error = std::string(action) + ": " + cudaGetErrorString(status);
    return false;
}

__global__ void nearestLetterboxPixelKernel(
    const std::uint8_t* source,
    int source_width,
    int source_height,
    bool source_is_bgr,
    float* destination,
    int destination_width,
    int destination_height,
    int resized_width,
    int resized_height,
    int pad_left,
    int pad_top,
    float scale) {
    const int pixel = blockIdx.x * blockDim.x + threadIdx.x;
    const int plane = destination_width * destination_height;
    if (pixel >= plane) return;
    const int x = pixel % destination_width;
    const int y = pixel / destination_width;
    const int resized_x = x - pad_left;
    const int resized_y = y - pad_top;
    if (resized_x < 0 || resized_y < 0 ||
        resized_x >= resized_width || resized_y >= resized_height) {
        constexpr float padding = 114.0f / 255.0f;
        destination[pixel] = padding;
        destination[plane + pixel] = padding;
        destination[2 * plane + pixel] = padding;
        return;
    }
    int source_x = static_cast<int>(static_cast<float>(resized_x) / scale);
    int source_y = static_cast<int>(static_cast<float>(resized_y) / scale);
    source_x = source_x < source_width ? source_x : source_width - 1;
    source_y = source_y < source_height ? source_y : source_height - 1;
    const int source_index = (source_y * source_width + source_x) * 3;
    const float first = static_cast<float>(source[source_index]) / 255.0f;
    const float green = static_cast<float>(source[source_index + 1]) / 255.0f;
    const float third = static_cast<float>(source[source_index + 2]) / 255.0f;
    destination[pixel] = source_is_bgr ? third : first;
    destination[plane + pixel] = green;
    destination[2 * plane + pixel] = source_is_bgr ? first : third;
}

__device__ __forceinline__ std::uint8_t saturateToByte(int value) {
    return static_cast<std::uint8_t>(value < 0 ? 0 : (value > 255 ? 255 : value));
}

__global__ void i420LetterboxPixelKernel(
    const std::uint8_t* source,
    int source_width,
    int source_height,
    float* destination,
    int destination_width,
    int destination_height,
    int resized_width,
    int resized_height,
    int pad_left,
    int pad_top,
    float scale) {
    const int pixel = blockIdx.x * blockDim.x + threadIdx.x;
    const int plane = destination_width * destination_height;
    if (pixel >= plane) return;
    const int x = pixel % destination_width;
    const int y = pixel / destination_width;
    const int resized_x = x - pad_left;
    const int resized_y = y - pad_top;
    if (resized_x < 0 || resized_y < 0 ||
        resized_x >= resized_width || resized_y >= resized_height) {
        constexpr float padding = 114.0f / 255.0f;
        destination[pixel] = padding;
        destination[plane + pixel] = padding;
        destination[2 * plane + pixel] = padding;
        return;
    }
    int source_x = static_cast<int>(static_cast<float>(resized_x) / scale);
    int source_y = static_cast<int>(static_cast<float>(resized_y) / scale);
    source_x = source_x < source_width ? source_x : source_width - 1;
    source_y = source_y < source_height ? source_y : source_height - 1;
    const int y_plane_bytes = source_width * source_height;
    const int chroma_width = source_width / 2;
    const int chroma_index = (source_y / 2) * chroma_width + source_x / 2;
    const int y_value = max(0, static_cast<int>(source[source_y * source_width + source_x]) - 16);
    const int u_value = static_cast<int>(source[y_plane_bytes + chroma_index]) - 128;
    const int v_value = static_cast<int>(
        source[y_plane_bytes + y_plane_bytes / 4 + chroma_index]) - 128;
    constexpr int shift = 20;
    constexpr int rounding = 1 << (shift - 1);
    const int scaled_y = y_value * 1220542;
    const std::uint8_t red = saturateToByte(
        (scaled_y + 1673527 * v_value + rounding) >> shift);
    const std::uint8_t green = saturateToByte(
        (scaled_y - 409993 * u_value - 852492 * v_value + rounding) >> shift);
    const std::uint8_t blue = saturateToByte(
        (scaled_y + 2116026 * u_value + rounding) >> shift);
    destination[pixel] = static_cast<float>(red) / 255.0f;
    destination[plane + pixel] = static_cast<float>(green) / 255.0f;
    destination[2 * plane + pixel] = static_cast<float>(blue) / 255.0f;
}

}  // namespace

struct VehicleCudaPreprocessor::Impl {
    struct DeviceFramePool : public std::enable_shared_from_this<DeviceFramePool> {
        struct Slot {
            std::uint8_t* data = nullptr;
            std::size_t capacity = 0;
            bool busy = false;
        };

        ~DeviceFramePool() {
            for (auto& slot : slots) {
                if (slot.data) cudaFree(slot.data);
            }
        }

        std::shared_ptr<void> acquire(
            std::size_t required,
            double& allocation_ms,
            std::string& error) {
            constexpr std::size_t kMaximumSlots = 3;
            const auto started = std::chrono::steady_clock::now();
            std::unique_lock<std::mutex> lock(mutex);
            auto available = [&]() {
                return std::find_if(slots.begin(), slots.end(),
                    [&](const Slot& slot) {
                        return !slot.busy && slot.capacity >= required;
                    });
            };
            auto idle = [&]() {
                return std::find_if(slots.begin(), slots.end(),
                    [](const Slot& slot) { return !slot.busy; });
            };
            auto found = available();
            if (found == slots.end() && slots.size() < kMaximumSlots) {
                Slot slot;
                if (!cudaOk(cudaMalloc(
                        reinterpret_cast<void**>(&slot.data), required),
                        "cudaMalloc retained I420 frame", error)) {
                    return {};
                }
                slot.capacity = required;
                slots.push_back(slot);
                found = std::prev(slots.end());
            }
            if (found == slots.end()) {
                auto reusable = idle();
                if (reusable == slots.end()) {
                    const bool released = condition.wait_for(
                        lock, std::chrono::milliseconds(25),
                        [&]() { return idle() != slots.end(); });
                    if (!released) {
                        error = "bounded retained I420 frame pool is exhausted";
                        return {};
                    }
                    reusable = idle();
                }
                found = available();
                if (found == slots.end()) {
                    std::uint8_t* replacement = nullptr;
                    if (!cudaOk(cudaMalloc(
                            reinterpret_cast<void**>(&replacement), required),
                            "cudaMalloc resized retained I420 frame", error)) {
                        return {};
                    }
                    if (reusable->data) cudaFree(reusable->data);
                    reusable->data = replacement;
                    reusable->capacity = required;
                    found = reusable;
                }
            }
            found->busy = true;
            const std::size_t index = static_cast<std::size_t>(
                std::distance(slots.begin(), found));
            auto* pointer = found->data;
            allocation_ms += std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - started).count();
            auto self = shared_from_this();
            return std::shared_ptr<void>(pointer, [self, index](void*) {
                std::lock_guard<std::mutex> guard(self->mutex);
                self->slots[index].busy = false;
                self->condition.notify_one();
            });
        }

        std::mutex mutex;
        std::condition_variable condition;
        std::vector<Slot> slots;

    };

    Impl() {
        frame_pool = std::make_shared<DeviceFramePool>();
    }

    enum class StagedFormat {
        None,
        Bgr8,
        Rgb8,
        I420,
    };

    ~Impl() {
        if (device_buffer) cudaFree(device_buffer);
        if (host_buffer) cudaFreeHost(host_buffer);
        if (h2d_started) cudaEventDestroy(h2d_started);
        if (h2d_finished) cudaEventDestroy(h2d_finished);
        if (kernel_finished) cudaEventDestroy(kernel_finished);
        if (device_copy_started) cudaEventDestroy(device_copy_started);
        if (device_copy_finished) cudaEventDestroy(device_copy_finished);
    }

    bool ensureEvents(std::string& error) {
        if (h2d_started && h2d_finished && kernel_finished &&
            device_copy_started && device_copy_finished) return true;
        if (!cudaOk(cudaEventCreate(&h2d_started), "cudaEventCreate preprocess h2d start", error) ||
            !cudaOk(cudaEventCreate(&h2d_finished), "cudaEventCreate preprocess h2d finish", error) ||
            !cudaOk(cudaEventCreate(&kernel_finished), "cudaEventCreate preprocess kernel finish", error) ||
            !cudaOk(cudaEventCreate(&device_copy_started),
                "cudaEventCreate retained I420 copy start", error) ||
            !cudaOk(cudaEventCreate(&device_copy_finished),
                "cudaEventCreate retained I420 copy finish", error)) {
            return false;
        }
        return true;
    }

    bool reserve(std::size_t required, std::string& error) {
        if (capacity >= required && host_buffer && device_buffer) return true;
        if (device_buffer) cudaFree(device_buffer);
        if (host_buffer) cudaFreeHost(host_buffer);
        device_buffer = nullptr;
        host_buffer = nullptr;
        capacity = 0;
        if (!cudaOk(cudaMallocHost(reinterpret_cast<void**>(&host_buffer), required),
                "cudaMallocHost preprocess staging", error) ||
            !cudaOk(cudaMalloc(reinterpret_cast<void**>(&device_buffer), required),
                "cudaMalloc preprocess source", error)) {
            return false;
        }
        capacity = required;
        return true;
    }

    std::uint8_t* host_buffer = nullptr;
    std::uint8_t* device_buffer = nullptr;
    std::size_t capacity = 0;
    std::size_t staged_bytes = 0;
    StagedFormat staged_format = StagedFormat::None;
    int source_width = 0;
    int source_height = 0;
    int destination_width = 0;
    int destination_height = 0;
    int resized_width = 0;
    int resized_height = 0;
    int pad_left = 0;
    int pad_top = 0;
    float scale = 1.0f;
    cudaEvent_t h2d_started = nullptr;
    cudaEvent_t h2d_finished = nullptr;
    cudaEvent_t kernel_finished = nullptr;
    cudaEvent_t device_copy_started = nullptr;
    cudaEvent_t device_copy_finished = nullptr;
    bool device_copy_pending = false;
    std::shared_ptr<DeviceFramePool> frame_pool;
};

VehicleCudaPreprocessor::VehicleCudaPreprocessor()
    : impl_(std::make_unique<Impl>()) {
}

VehicleCudaPreprocessor::~VehicleCudaPreprocessor() = default;

bool VehicleCudaPreprocessor::enqueueLetterbox(
    const ImageView& source,
    float* destination,
    int destination_width,
    int destination_height,
    cudaStream_t stream,
    VehicleCudaPreprocessGeometry& geometry,
    VehicleCudaPreprocessTiming& timing,
    std::string& error) {
    return stageLetterbox(
            source, destination_width, destination_height,
            geometry, timing, error) &&
        enqueueStaged(destination, stream, timing, error);
}

bool VehicleCudaPreprocessor::enqueueI420Letterbox(
    const I420ImageView& source,
    float* destination,
    int destination_width,
    int destination_height,
    cudaStream_t stream,
    VehicleCudaPreprocessGeometry& geometry,
    VehicleCudaPreprocessTiming& timing,
    std::string& error) {
    return stageI420Letterbox(
            source, destination_width, destination_height,
            geometry, timing, error) &&
        enqueueStaged(destination, stream, timing, error);
}

bool VehicleCudaPreprocessor::stageLetterbox(
    const ImageView& source,
    int destination_width,
    int destination_height,
    VehicleCudaPreprocessGeometry& geometry,
    VehicleCudaPreprocessTiming& timing,
    std::string& error) {
    timing = {};
    impl_->device_copy_pending = false;
    if (!source.valid() || source.channels != 3 ||
        destination_width <= 0 || destination_height <= 0) {
        error = "invalid CUDA letterbox staging input";
        return false;
    }
    const auto allocation_started = std::chrono::steady_clock::now();
    const std::size_t row_bytes = static_cast<std::size_t>(source.width) * 3U;
    const std::size_t source_bytes = row_bytes * static_cast<std::size_t>(source.height);
    if (!impl_->ensureEvents(error) || !impl_->reserve(source_bytes, error)) return false;
    timing.allocation_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - allocation_started).count();

    const auto staging_started = std::chrono::steady_clock::now();
    if (source.row_stride_bytes == row_bytes) {
        std::memcpy(impl_->host_buffer, source.data, source_bytes);
    }
    else {
        for (int row = 0; row < source.height; ++row) {
            std::memcpy(
                impl_->host_buffer + static_cast<std::size_t>(row) * row_bytes,
                source.data + static_cast<std::size_t>(row) * source.row_stride_bytes,
                row_bytes);
        }
    }
    timing.host_staging_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - staging_started).count();
    timing.h2d_bytes = source_bytes;

    geometry.scale = std::min(
        static_cast<float>(destination_width) / static_cast<float>(source.width),
        static_cast<float>(destination_height) / static_cast<float>(source.height));
    impl_->resized_width = std::max(
        1, static_cast<int>(std::round(source.width * geometry.scale)));
    impl_->resized_height = std::max(
        1, static_cast<int>(std::round(source.height * geometry.scale)));
    geometry.pad_x = static_cast<float>(destination_width - impl_->resized_width) / 2.0f;
    geometry.pad_y = static_cast<float>(destination_height - impl_->resized_height) / 2.0f;
    impl_->staged_bytes = source_bytes;
    impl_->staged_format = source.pixel_format == ImagePixelFormat::Bgr8
        ? Impl::StagedFormat::Bgr8 : Impl::StagedFormat::Rgb8;
    impl_->source_width = source.width;
    impl_->source_height = source.height;
    impl_->destination_width = destination_width;
    impl_->destination_height = destination_height;
    impl_->pad_left = static_cast<int>(std::floor(geometry.pad_x));
    impl_->pad_top = static_cast<int>(std::floor(geometry.pad_y));
    impl_->scale = geometry.scale;
    return true;
}

bool VehicleCudaPreprocessor::stageI420Letterbox(
    const I420ImageView& source,
    int destination_width,
    int destination_height,
    VehicleCudaPreprocessGeometry& geometry,
    VehicleCudaPreprocessTiming& timing,
    std::string& error) {
    timing = {};
    impl_->device_copy_pending = false;
    if (!source.valid() || destination_width <= 0 || destination_height <= 0) {
        error = "invalid CUDA I420 letterbox staging input";
        return false;
    }
    const auto allocation_started = std::chrono::steady_clock::now();
    const std::size_t y_bytes = static_cast<std::size_t>(source.width) * source.height;
    const std::size_t chroma_bytes = y_bytes / 4U;
    const std::size_t source_bytes = y_bytes + 2U * chroma_bytes;
    if (!impl_->ensureEvents(error) || !impl_->reserve(source_bytes, error)) return false;
    timing.allocation_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - allocation_started).count();

    const auto staging_started = std::chrono::steady_clock::now();
    const auto copyPlane = [](std::uint8_t* destination_plane,
                              std::size_t destination_stride,
                              const std::uint8_t* source_plane,
                              std::size_t source_stride,
                              int rows,
                              std::size_t row_bytes) {
        for (int row = 0; row < rows; ++row) {
            std::memcpy(
                destination_plane + static_cast<std::size_t>(row) * destination_stride,
                source_plane + static_cast<std::size_t>(row) * source_stride,
                row_bytes);
        }
    };
    const bool tightly_packed =
        source.y_stride_bytes == static_cast<std::size_t>(source.width) &&
        source.u_stride_bytes == static_cast<std::size_t>(source.width / 2) &&
        source.v_stride_bytes == static_cast<std::size_t>(source.width / 2) &&
        source.u_plane == source.y_plane + y_bytes &&
        source.v_plane == source.u_plane + chroma_bytes;
    if (tightly_packed) {
        std::memcpy(impl_->host_buffer, source.y_plane, source_bytes);
    }
    else {
        copyPlane(impl_->host_buffer, static_cast<std::size_t>(source.width),
            source.y_plane, source.y_stride_bytes, source.height,
            static_cast<std::size_t>(source.width));
        copyPlane(impl_->host_buffer + y_bytes,
            static_cast<std::size_t>(source.width / 2), source.u_plane,
            source.u_stride_bytes, source.height / 2,
            static_cast<std::size_t>(source.width / 2));
        copyPlane(impl_->host_buffer + y_bytes + chroma_bytes,
            static_cast<std::size_t>(source.width / 2), source.v_plane,
            source.v_stride_bytes, source.height / 2,
            static_cast<std::size_t>(source.width / 2));
    }
    timing.host_staging_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - staging_started).count();
    timing.h2d_bytes = source_bytes;

    geometry.scale = std::min(
        static_cast<float>(destination_width) / static_cast<float>(source.width),
        static_cast<float>(destination_height) / static_cast<float>(source.height));
    impl_->resized_width = std::max(
        1, static_cast<int>(std::round(source.width * geometry.scale)));
    impl_->resized_height = std::max(
        1, static_cast<int>(std::round(source.height * geometry.scale)));
    geometry.pad_x = static_cast<float>(destination_width - impl_->resized_width) / 2.0f;
    geometry.pad_y = static_cast<float>(destination_height - impl_->resized_height) / 2.0f;

    impl_->staged_bytes = source_bytes;
    impl_->staged_format = Impl::StagedFormat::I420;
    impl_->source_width = source.width;
    impl_->source_height = source.height;
    impl_->destination_width = destination_width;
    impl_->destination_height = destination_height;
    impl_->pad_left = static_cast<int>(std::floor(geometry.pad_x));
    impl_->pad_top = static_cast<int>(std::floor(geometry.pad_y));
    impl_->scale = geometry.scale;
    return true;
}

bool VehicleCudaPreprocessor::enqueueStaged(
    float* destination,
    cudaStream_t stream,
    VehicleCudaPreprocessTiming& timing,
    std::string& error,
    bool record_timing) {
    if (!destination || !stream || impl_->staged_format == Impl::StagedFormat::None ||
        impl_->staged_bytes == 0) {
        error = "CUDA letterbox input has not been staged";
        return false;
    }
    if (record_timing && !cudaOk(cudaEventRecord(impl_->h2d_started, stream),
            "cudaEventRecord preprocess h2d start", error)) {
        return false;
    }
    if (!cudaOk(cudaMemcpyAsync(
            impl_->device_buffer, impl_->host_buffer, impl_->staged_bytes,
            cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync preprocess source", error)) {
        return false;
    }
    if (record_timing && !cudaOk(cudaEventRecord(impl_->h2d_finished, stream),
            "cudaEventRecord preprocess h2d finish", error)) return false;
    const int pixels = impl_->destination_width * impl_->destination_height;
    constexpr int threads = 256;
    const int blocks = (pixels + threads - 1) / threads;
    if (impl_->staged_format == Impl::StagedFormat::I420) {
        i420LetterboxPixelKernel<<<blocks, threads, 0, stream>>>(
            impl_->device_buffer, impl_->source_width, impl_->source_height,
            destination, impl_->destination_width, impl_->destination_height,
            impl_->resized_width, impl_->resized_height,
            impl_->pad_left, impl_->pad_top, impl_->scale);
    }
    else {
        nearestLetterboxPixelKernel<<<blocks, threads, 0, stream>>>(
            impl_->device_buffer, impl_->source_width, impl_->source_height,
            impl_->staged_format == Impl::StagedFormat::Bgr8,
            destination, impl_->destination_width, impl_->destination_height,
            impl_->resized_width, impl_->resized_height,
            impl_->pad_left, impl_->pad_top, impl_->scale);
    }
    if (!cudaOk(cudaGetLastError(), "letterbox pixel kernel launch", error)) {
        return false;
    }
    if (record_timing && !cudaOk(cudaEventRecord(impl_->kernel_finished, stream),
            "cudaEventRecord preprocess kernel finish", error)) return false;
    return true;
}

bool VehicleCudaPreprocessor::retainStagedI420(
    cudaStream_t stream,
    std::shared_ptr<const DeviceI420Image>& frame,
    VehicleCudaPreprocessTiming& timing,
    std::string& error) {
    frame.reset();
    if (!stream || impl_->staged_format != Impl::StagedFormat::I420 ||
        !impl_->device_buffer || impl_->staged_bytes == 0 ||
        impl_->source_width <= 0 || impl_->source_height <= 0) {
        error = "no staged CUDA I420 frame is available to retain";
        return false;
    }
    auto lease = impl_->frame_pool->acquire(
        impl_->staged_bytes, timing.allocation_ms, error);
    if (!lease) return false;
    auto* destination = static_cast<std::uint8_t*>(lease.get());
    if (!cudaOk(cudaEventRecord(impl_->device_copy_started, stream),
            "cudaEventRecord retained I420 copy start", error) ||
        !cudaOk(cudaMemcpyAsync(
            destination,
            impl_->device_buffer,
            impl_->staged_bytes,
            cudaMemcpyDeviceToDevice,
            stream), "cudaMemcpyAsync retained I420 frame", error) ||
        !cudaOk(cudaEventRecord(impl_->device_copy_finished, stream),
            "cudaEventRecord retained I420 copy finish", error)) {
        lease.reset();
        return false;
    }
    const std::size_t y_bytes = static_cast<std::size_t>(
        impl_->source_width) * impl_->source_height;
    const std::size_t chroma_bytes = y_bytes / 4U;
    auto retained = std::make_shared<DeviceI420Image>();
    retained->lease = std::move(lease);
    retained->y_plane = destination;
    retained->u_plane = destination + y_bytes;
    retained->v_plane = destination + y_bytes + chroma_bytes;
    retained->width = impl_->source_width;
    retained->height = impl_->source_height;
    retained->y_stride_bytes = static_cast<std::size_t>(impl_->source_width);
    retained->u_stride_bytes = static_cast<std::size_t>(impl_->source_width / 2);
    retained->v_stride_bytes = static_cast<std::size_t>(impl_->source_width / 2);
    timing.device_frame_copy_bytes = impl_->staged_bytes;
    impl_->device_copy_pending = true;
    frame = std::move(retained);
    return true;
}

bool VehicleCudaPreprocessor::collectTiming(
    VehicleCudaPreprocessTiming& timing,
    std::string& error) const {
    float elapsed = 0.0f;
    if (!cudaOk(cudaEventElapsedTime(
            &elapsed, impl_->h2d_started, impl_->h2d_finished),
            "cudaEventElapsedTime preprocess h2d", error)) return false;
    timing.h2d_ms = elapsed;
    if (!cudaOk(cudaEventElapsedTime(
            &elapsed, impl_->h2d_finished, impl_->kernel_finished),
            "cudaEventElapsedTime preprocess kernel", error)) return false;
    timing.kernel_ms = elapsed;
    return collectRetainedI420Timing(timing, error);
}

bool VehicleCudaPreprocessor::collectRetainedI420Timing(
    VehicleCudaPreprocessTiming& timing,
    std::string& error) const {
    if (!impl_->device_copy_pending) return true;
    float elapsed = 0.0f;
    if (!cudaOk(cudaEventElapsedTime(
            &elapsed, impl_->device_copy_started, impl_->device_copy_finished),
            "cudaEventElapsedTime retained I420 copy", error)) return false;
    timing.device_frame_copy_ms = elapsed;
    return true;
}

}  // namespace yolo11_server
