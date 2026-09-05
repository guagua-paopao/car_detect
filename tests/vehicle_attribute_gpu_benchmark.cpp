#include "server/vehicle_tensorrt_adapters.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>
#include <opencv2/imgproc.hpp>

using namespace yolo11_server;

namespace {

using Clock = std::chrono::steady_clock;

const std::vector<std::string> kBodyTypes = {
    "sedan", "suv", "mpv", "van", "pickup", "bus",
    "light_truck", "heavy_truck", "other", "unknown"};
const std::vector<std::string> kColors = {
    "black", "white", "silver_gray", "red", "blue", "green",
    "yellow_orange", "brown_beige", "other", "unknown"};

ModelArtifactDescriptor attributeArtifact() {
    ModelArtifactDescriptor artifact;
    artifact.artifact_id = "vehicle-attr-best-components-256-r1";
    artifact.role = VehicleModelRole::Attributes;
    artifact.delivery_status = ModelDeliveryStatus::Deployed;
    artifact.labels_version = "vehicle-labels-v1";
    artifact.onnx_path =
        "models/candidates/vehicle-attr-best-components-256-r1/"
        "vehicle-attr-best-components-256-r1.onnx";
    artifact.engine_path =
        "models/candidates/vehicle-attr-best-components-256-r1/"
        "vehicle-attr-best-components-256-r1.engine";
    artifact.onnx_sha256 =
        "b470c013acff8cfbe9a7cf7ead8184f8a1516998507cfce4ba342e4ecf4ed86e";
    artifact.engine_sha256 =
        "eab83cc6e66133af5d61672e121465311ef914423dea54a2c32a6041ed98f5d6";
    artifact.input_width = 256;
    artifact.input_height = 256;
    artifact.max_batch = 16;
    artifact.output_names = {"body_type", "color"};
    return artifact;
}

std::shared_ptr<OwnedImage> gradientImage(int width, int height, int seed) {
    auto image = std::make_shared<OwnedImage>();
    image->width = width;
    image->height = height;
    image->channels = 3;
    image->row_stride_bytes = static_cast<std::size_t>(width) * 3U;
    image->pixel_format = ImagePixelFormat::Bgr8;
    image->pixels.resize(image->row_stride_bytes * static_cast<std::size_t>(height));
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const std::size_t offset = static_cast<std::size_t>(y) *
                image->row_stride_bytes + static_cast<std::size_t>(x) * 3U;
            image->pixels[offset] = static_cast<std::uint8_t>((x + y + seed) & 255);
            image->pixels[offset + 1] =
                static_cast<std::uint8_t>((2 * x + 3 * seed) & 255);
            image->pixels[offset + 2] =
                static_cast<std::uint8_t>((2 * y + 5 * seed) & 255);
        }
    }
    return image;
}

std::vector<VehicleAttributeCrop> makeBatch(int batch_size, int seed) {
    std::vector<VehicleAttributeCrop> crops;
    crops.reserve(static_cast<std::size_t>(batch_size));
    for (int index = 0; index < batch_size; ++index) {
        VehicleAttributeCrop crop;
        crop.crop = gradientImage(
            240 + ((seed + index) % 5) * 23,
            140 + ((seed + 2 * index) % 7) * 17,
            seed * 19 + index);
        crop.camera_id = "attribute-gpu-benchmark";
        crop.run_id = "attribute-gpu-benchmark";
        crop.run_generation = 1;
        crop.track_id = index + 1;
        crop.crop_sequence = static_cast<std::uint64_t>(seed + 1);
        crop.quality_score = 1.0f;
        crops.push_back(std::move(crop));
    }
    return crops;
}

struct I420ParityBatch {
    std::vector<VehicleAttributeCrop> bgr;
    std::vector<VehicleAttributeCrop> i420;
};

I420ParityBatch makeI420ParityBatch(int batch_size, int seed) {
    constexpr int width = 640;
    constexpr int height = 384;
    auto generated = gradientImage(width, height, seed * 31);
    cv::Mat source(
        height,
        width,
        CV_8UC3,
        generated->pixels.data(),
        generated->row_stride_bytes);
    cv::Mat packed_i420;
    cv::cvtColor(source, packed_i420, cv::COLOR_BGR2YUV_I420);
    cv::Mat reference_bgr;
    cv::cvtColor(packed_i420, reference_bgr, cv::COLOR_YUV2BGR_I420);

    auto bytes = std::make_shared<std::vector<std::uint8_t>>(
        packed_i420.datastart, packed_i420.dataend);
    auto owned_i420 = std::make_shared<OwnedI420Image>();
    owned_i420->bytes = bytes;
    owned_i420->width = width;
    owned_i420->height = height;
    owned_i420->y_offset = 0;
    owned_i420->u_offset = static_cast<std::size_t>(width) * height;
    owned_i420->v_offset = owned_i420->u_offset +
        static_cast<std::size_t>(width) * height / 4U;
    owned_i420->y_stride_bytes = width;
    owned_i420->u_stride_bytes = width / 2;
    owned_i420->v_stride_bytes = width / 2;

    I420ParityBatch result;
    result.bgr.reserve(static_cast<std::size_t>(batch_size));
    result.i420.reserve(static_cast<std::size_t>(batch_size));
    for (int index = 0; index < batch_size; ++index) {
        const int left = 1 + (seed * 13 + index * 37) % 220;
        const int top = 1 + (seed * 7 + index * 29) % 120;
        const int crop_width = 180 + (seed + index * 11) % 180;
        const int crop_height = 110 + (seed * 3 + index * 17) % 130;
        const int right = std::min(width, left + crop_width);
        const int bottom = std::min(height, top + crop_height);

        auto owned_bgr = std::make_shared<OwnedImage>();
        owned_bgr->width = right - left;
        owned_bgr->height = bottom - top;
        owned_bgr->channels = 3;
        owned_bgr->row_stride_bytes =
            static_cast<std::size_t>(owned_bgr->width) * 3U;
        owned_bgr->pixel_format = ImagePixelFormat::Bgr8;
        owned_bgr->pixels.resize(
            owned_bgr->row_stride_bytes * owned_bgr->height);
        for (int row = 0; row < owned_bgr->height; ++row) {
            std::memcpy(
                owned_bgr->pixels.data() + static_cast<std::size_t>(row) *
                    owned_bgr->row_stride_bytes,
                reference_bgr.ptr(top + row) + static_cast<std::size_t>(left) * 3U,
                owned_bgr->row_stride_bytes);
        }

        VehicleAttributeCrop legacy;
        legacy.crop = std::move(owned_bgr);
        legacy.camera_id = "attribute-i420-parity";
        legacy.run_id = "attribute-i420-parity";
        legacy.run_generation = 1;
        legacy.track_id = index + 1;
        legacy.crop_sequence = static_cast<std::uint64_t>(seed + 1);
        legacy.quality_score = 1.0f;
        result.bgr.push_back(legacy);

        VehicleAttributeCrop direct = legacy;
        direct.crop.reset();
        direct.i420_frame = owned_i420;
        direct.source_box = {
            (static_cast<float>(left) + 0.25f) / width,
            (static_cast<float>(top) + 0.25f) / height,
            (static_cast<float>(right) - 0.25f) / width,
            (static_cast<float>(bottom) - 0.25f) / height,
        };
        result.i420.push_back(std::move(direct));
    }
    return result;
}

double percentile(std::vector<double> values, double quantile) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    const double position = quantile * static_cast<double>(values.size() - 1);
    const auto lower = static_cast<std::size_t>(std::floor(position));
    const auto upper = static_cast<std::size_t>(std::ceil(position));
    if (lower == upper) return values[lower];
    const double fraction = position - static_cast<double>(lower);
    return values[lower] * (1.0 - fraction) + values[upper] * fraction;
}

nlohmann::json summarize(const std::vector<double>& values) {
    if (values.empty()) return nlohmann::json::object();
    return {
        {"mean", std::accumulate(values.begin(), values.end(), 0.0) /
            static_cast<double>(values.size())},
        {"p50", percentile(values, 0.50)},
        {"p95", percentile(values, 0.95)},
        {"p99", percentile(values, 0.99)},
        {"min", *std::min_element(values.begin(), values.end())},
        {"max", *std::max_element(values.begin(), values.end())},
    };
}

std::unique_ptr<TensorRtVehicleAttributeRunner> makeRunner(
    const std::string& project_root,
    bool gpu_preprocess,
    const ModelArtifactDescriptor& artifact) {
    TensorRtAttributeOptions options;
    options.artifact_root = project_root;
    options.body_types = kBodyTypes;
    options.colors = kColors;
    options.use_gpu_preprocess = gpu_preprocess;
    auto runner = std::make_unique<TensorRtVehicleAttributeRunner>(
        std::move(options));
    std::string error;
    if (!runner->initialize(artifact, error)) {
        throw std::runtime_error("attribute runner initialization failed: " + error);
    }
    return runner;
}

void requireParity(
    const std::vector<VehicleAttributeResult>& cpu,
    const std::vector<VehicleAttributeResult>& gpu,
    double& maximum_confidence_delta) {
    if (cpu.size() != gpu.size()) throw std::runtime_error("attribute result count mismatch");
    for (std::size_t index = 0; index < cpu.size(); ++index) {
        if (cpu[index].body_type.label != gpu[index].body_type.label ||
            cpu[index].color.label != gpu[index].color.label) {
            throw std::runtime_error("attribute top-1 mismatch");
        }
        maximum_confidence_delta = std::max(maximum_confidence_delta,
            static_cast<double>(std::abs(
                cpu[index].body_type.confidence - gpu[index].body_type.confidence)));
        maximum_confidence_delta = std::max(maximum_confidence_delta,
            static_cast<double>(std::abs(
                cpu[index].color.confidence - gpu[index].color.confidence)));
    }
}

nlohmann::json benchmark(
    TensorRtVehicleAttributeRunner& runner,
    const std::vector<VehicleAttributeCrop>& crops,
    int iterations,
    int runs) {
    nlohmann::json result = nlohmann::json::array();
    for (int run = 0; run < runs; ++run) {
        std::vector<double> preprocess;
        std::vector<double> h2d;
        std::vector<double> inference;
        std::vector<double> postprocess;
        std::vector<double> total;
        preprocess.reserve(iterations);
        h2d.reserve(iterations);
        inference.reserve(iterations);
        postprocess.reserve(iterations);
        total.reserve(iterations);
        const auto started = Clock::now();
        for (int index = 0; index < iterations; ++index) {
            (void)runner.inferBatch(crops);
            const auto timing = runner.lastTiming();
            preprocess.push_back(timing.preprocess_ms);
            h2d.push_back(timing.h2d_ms);
            inference.push_back(timing.inference_ms);
            postprocess.push_back(timing.postprocess_ms);
            total.push_back(timing.total_ms);
        }
        const double wall_seconds =
            std::chrono::duration<double>(Clock::now() - started).count();
        result.push_back({
            {"run", run + 1},
            {"samples", iterations},
            {"batch_size", crops.size()},
            {"batches_per_second", static_cast<double>(iterations) / wall_seconds},
            {"images_per_second", static_cast<double>(iterations) * crops.size() /
                wall_seconds},
            {"preprocess_ms", summarize(preprocess)},
            {"h2d_ms", summarize(h2d)},
            {"inference_ms", summarize(inference)},
            {"postprocess_ms", summarize(postprocess)},
            {"total_ms", summarize(total)},
            {"h2d_bytes_per_batch", runner.lastTiming().h2d_bytes},
        });
    }
    return result;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2 || argc > 6) {
        std::cerr << "Usage: vehicle_attribute_gpu_benchmark <project-root> "
                     "[iterations=200] [warmup=50] [runs=3] [report.json]\n";
        return 2;
    }
    try {
        const std::string project_root = argv[1];
        const int iterations = argc >= 3 ? std::stoi(argv[2]) : 200;
        const int warmup = argc >= 4 ? std::stoi(argv[3]) : 50;
        const int runs = argc >= 5 ? std::stoi(argv[4]) : 3;
        if (iterations <= 0 || warmup <= 0 || runs <= 0) {
            throw std::invalid_argument("benchmark counts must be positive");
        }
        const auto production_artifact = attributeArtifact();
        auto cpu = makeRunner(project_root, false, production_artifact);
        auto gpu = makeRunner(project_root, true, production_artifact);

        double maximum_confidence_delta = 0.0;
        for (int frame = 0; frame < 200; ++frame) {
            auto crops = makeBatch(1 + frame % 16, frame);
            requireParity(
                cpu->inferBatch(crops), gpu->inferBatch(crops),
                maximum_confidence_delta);
        }
        if (maximum_confidence_delta > 1e-5) {
            throw std::runtime_error("attribute confidence parity exceeded 1e-5");
        }

        double i420_roi_maximum_confidence_delta = 0.0;
        for (int frame = 0; frame < 200; ++frame) {
            auto batches = makeI420ParityBatch(1 + frame % 16, frame);
            requireParity(
                cpu->inferBatch(batches.bgr),
                gpu->inferBatch(batches.i420),
                i420_roi_maximum_confidence_delta);
        }
        if (i420_roi_maximum_confidence_delta > 1e-5) {
            throw std::runtime_error(
                "attribute I420 ROI confidence parity exceeded 1e-5");
        }

        int candidate_parity_frames = 0;
        double candidate_maximum_confidence_delta = 0.0;
        constexpr double candidate_confidence_tolerance = 5e-3;
        std::string candidate_engine_path;
        std::string candidate_engine_sha256;
        const char* candidate_path =
            std::getenv("VCAS_CANDIDATE_ATTRIBUTE_ENGINE_PATH");
        const char* candidate_hash =
            std::getenv("VCAS_CANDIDATE_ATTRIBUTE_ENGINE_SHA256");
        if (candidate_path != nullptr && candidate_path[0] != '\0') {
            if (candidate_hash == nullptr || candidate_hash[0] == '\0') {
                throw std::runtime_error(
                    "VCAS_CANDIDATE_ATTRIBUTE_ENGINE_SHA256 is required with "
                    "VCAS_CANDIDATE_ATTRIBUTE_ENGINE_PATH");
            }
            auto candidate_artifact = production_artifact;
            candidate_artifact.artifact_id += "-candidate";
            candidate_artifact.engine_path = candidate_path;
            candidate_artifact.engine_sha256 = candidate_hash;
            auto candidate = makeRunner(project_root, true, candidate_artifact);
            for (int frame = 0; frame < 200; ++frame) {
                auto crops = makeBatch(1 + frame % 16, frame);
                requireParity(
                    gpu->inferBatch(crops), candidate->inferBatch(crops),
                    candidate_maximum_confidence_delta);
            }
            candidate_parity_frames = 200;
            candidate_engine_path = candidate_path;
            candidate_engine_sha256 = candidate_hash;
            if (candidate_maximum_confidence_delta >
                candidate_confidence_tolerance) {
                throw std::runtime_error(
                    "candidate attribute engine confidence parity exceeded 5e-3: " +
                    std::to_string(candidate_maximum_confidence_delta));
            }
        }

        const auto crops = makeBatch(16, 1000);
        for (int index = 0; index < warmup; ++index) {
            (void)cpu->inferBatch(crops);
            (void)gpu->inferBatch(crops);
        }
        nlohmann::json cpu_runs = nlohmann::json::array();
        nlohmann::json gpu_runs = nlohmann::json::array();
        for (int run = 0; run < runs; ++run) {
            if ((run & 1) == 0) {
                cpu_runs.push_back(benchmark(*cpu, crops, iterations, 1).front());
                gpu_runs.push_back(benchmark(*gpu, crops, iterations, 1).front());
            }
            else {
                gpu_runs.push_back(benchmark(*gpu, crops, iterations, 1).front());
                cpu_runs.push_back(benchmark(*cpu, crops, iterations, 1).front());
            }
            cpu_runs.back()["run"] = run + 1;
            gpu_runs.back()["run"] = run + 1;
        }
        nlohmann::json report = {
            {"schema_version", "1.0"},
            {"artifact_id", attributeArtifact().artifact_id},
            {"parity_frames", 200},
            {"maximum_confidence_delta", maximum_confidence_delta},
            {"i420_roi_parity_frames", 200},
            {"i420_roi_maximum_confidence_delta",
                i420_roi_maximum_confidence_delta},
            {"candidate_engine_path", candidate_engine_path},
            {"candidate_engine_sha256", candidate_engine_sha256},
            {"candidate_parity_frames", candidate_parity_frames},
            {"candidate_maximum_confidence_delta",
                candidate_maximum_confidence_delta},
            {"candidate_confidence_tolerance",
                candidate_confidence_tolerance},
            {"batch_size", crops.size()},
            {"iterations_per_run", iterations},
            {"warmup_batches_per_mode", warmup},
            {"runs_requested", runs},
            {"run_order", "alternating cpu/gpu; odd-numbered pairs start with cpu"},
            {"cpu_preprocess", std::move(cpu_runs)},
            {"gpu_preprocess", std::move(gpu_runs)},
        };
        const std::string serialized = report.dump(2);
        if (argc >= 6) {
            const std::filesystem::path path(argv[5]);
            if (!path.parent_path().empty()) {
                std::filesystem::create_directories(path.parent_path());
            }
            std::ofstream stream(path, std::ios::binary | std::ios::trunc);
            if (!stream) throw std::runtime_error("could not create report file");
            stream << serialized << '\n';
        }
        std::cout << serialized << '\n';
        return 0;
    }
    catch (const std::exception& exception) {
        std::cerr << "FAIL: " << exception.what() << '\n';
        return 1;
    }
}
