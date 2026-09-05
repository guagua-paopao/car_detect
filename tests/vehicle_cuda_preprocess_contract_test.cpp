#include "server/vehicle_cuda_preprocess.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime_api.h>

using namespace yolo11_server;

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

void requireCuda(cudaError_t status, const std::string& operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(
            operation + ": " + cudaGetErrorString(status));
    }
}

struct PaddedI420 {
    int width = 0;
    int height = 0;
    std::size_t y_stride = 0;
    std::size_t u_stride = 0;
    std::size_t v_stride = 0;
    std::vector<std::uint8_t> y;
    std::vector<std::uint8_t> u;
    std::vector<std::uint8_t> v;

    I420ImageView view() const {
        return {
            y.data(), u.data(), v.data(), width, height,
            y_stride, u_stride, v_stride};
    }
};

PaddedI420 makeFrame(int width, int height, int seed) {
    PaddedI420 frame;
    frame.width = width;
    frame.height = height;
    frame.y_stride = static_cast<std::size_t>(width + 13);
    frame.u_stride = static_cast<std::size_t>(width / 2 + 5);
    frame.v_stride = static_cast<std::size_t>(width / 2 + 11);
    frame.y.assign(frame.y_stride * height, 0xee);
    frame.u.assign(frame.u_stride * (height / 2), 0xdd);
    frame.v.assign(frame.v_stride * (height / 2), 0xcc);
    for (int row = 0; row < height; ++row) {
        for (int column = 0; column < width; ++column) {
            frame.y[static_cast<std::size_t>(row) * frame.y_stride + column] =
                static_cast<std::uint8_t>((seed + row * 7 + column * 3) & 0xff);
        }
    }
    for (int row = 0; row < height / 2; ++row) {
        for (int column = 0; column < width / 2; ++column) {
            frame.u[static_cast<std::size_t>(row) * frame.u_stride + column] =
                static_cast<std::uint8_t>((seed + 41 + row * 5 + column) & 0xff);
            frame.v[static_cast<std::size_t>(row) * frame.v_stride + column] =
                static_cast<std::uint8_t>((seed + 97 + row * 3 + column * 2) & 0xff);
        }
    }
    return frame;
}

void verifyPlane(
    const std::uint8_t* device,
    std::size_t device_stride,
    const std::vector<std::uint8_t>& expected,
    std::size_t expected_stride,
    int rows,
    int columns,
    const std::string& name) {
    std::vector<std::uint8_t> actual(
        static_cast<std::size_t>(rows) * static_cast<std::size_t>(columns));
    requireCuda(cudaMemcpy2D(
        actual.data(), static_cast<std::size_t>(columns),
        device, device_stride, static_cast<std::size_t>(columns), rows,
        cudaMemcpyDeviceToHost), "cudaMemcpy2D " + name);
    for (int row = 0; row < rows; ++row) {
        const auto* expected_row = expected.data() +
            static_cast<std::size_t>(row) * expected_stride;
        const auto* actual_row = actual.data() +
            static_cast<std::size_t>(row) * static_cast<std::size_t>(columns);
        require(std::equal(
            expected_row, expected_row + columns, actual_row),
            name + " plane differs at row " + std::to_string(row));
    }
}

void verifyFrame(
    const DeviceI420Image& retained,
    const PaddedI420& expected) {
    require(retained.valid(), "retained DeviceI420Image is invalid");
    require(retained.width == expected.width &&
            retained.height == expected.height,
        "retained DeviceI420Image geometry differs");
    verifyPlane(retained.y_plane, retained.y_stride_bytes,
        expected.y, expected.y_stride, expected.height, expected.width, "Y");
    verifyPlane(retained.u_plane, retained.u_stride_bytes,
        expected.u, expected.u_stride, expected.height / 2,
        expected.width / 2, "U");
    verifyPlane(retained.v_plane, retained.v_stride_bytes,
        expected.v, expected.v_stride, expected.height / 2,
        expected.width / 2, "V");
}

}  // namespace

int main() {
    cudaStream_t stream = nullptr;
    float* destination = nullptr;
    try {
        int device_count = 0;
        requireCuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
        require(device_count > 0, "an NVIDIA CUDA device is required");
        requireCuda(cudaStreamCreate(&stream), "cudaStreamCreate");
        constexpr int kDestinationWidth = 256;
        constexpr int kDestinationHeight = 256;
        requireCuda(cudaMalloc(
            reinterpret_cast<void**>(&destination),
            3U * kDestinationWidth * kDestinationHeight * sizeof(float)),
            "cudaMalloc destination");

        VehicleCudaPreprocessor preprocessor;
        const auto retain = [&](const PaddedI420& source) {
            VehicleCudaPreprocessGeometry geometry;
            VehicleCudaPreprocessTiming timing;
            std::string error;
            require(preprocessor.stageI420Letterbox(
                source.view(), kDestinationWidth, kDestinationHeight,
                geometry, timing, error), "stageI420Letterbox: " + error);
            require(preprocessor.enqueueStaged(
                destination, stream, timing, error),
                "enqueueStaged: " + error);
            std::shared_ptr<const DeviceI420Image> retained;
            require(preprocessor.retainStagedI420(
                stream, retained, timing, error),
                "retainStagedI420: " + error);
            requireCuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
            require(timing.device_frame_copy_bytes ==
                    static_cast<std::size_t>(source.width) * source.height * 3U / 2U,
                "retained copy byte count differs");
            verifyFrame(*retained, source);
            return retained;
        };

        const auto small_a = makeFrame(64, 48, 3);
        const auto small_b = makeFrame(64, 48, 19);
        const auto small_c = makeFrame(64, 48, 37);
        std::vector<std::shared_ptr<const DeviceI420Image>> held;
        held.push_back(retain(small_a));
        held.push_back(retain(small_b));
        held.push_back(retain(small_c));

        // Fill all three bounded slots, release one undersized slot, then grow
        // it while the other two are still leased. This is the exact pool
        // resize path that previously risked selecting the wrong iterator.
        held[1].reset();
        const auto large = makeFrame(128, 96, 71);
        const auto grown = retain(large);
        verifyFrame(*grown, large);

        requireCuda(cudaFree(destination), "cudaFree destination");
        destination = nullptr;
        requireCuda(cudaStreamDestroy(stream), "cudaStreamDestroy");
        stream = nullptr;
        std::cout
            << "PASS: padded I420 Y/U/V strides and bounded retained-frame pool growth matched exactly\n";
        return 0;
    }
    catch (const std::exception& exception) {
        if (destination) cudaFree(destination);
        if (stream) cudaStreamDestroy(stream);
        std::cerr << "FAIL: " << exception.what() << '\n';
        return 1;
    }
}
