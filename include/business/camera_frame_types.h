#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "business/camera_hub_status.h"

namespace yolo11_server {

struct I420FrameStorage {
    std::shared_ptr<const std::vector<std::uint8_t>> bytes;
    int width = 0;
    int height = 0;
    std::size_t y_offset = 0;
    std::size_t u_offset = 0;
    std::size_t v_offset = 0;
    std::size_t y_stride_bytes = 0;
    std::size_t u_stride_bytes = 0;
    std::size_t v_stride_bytes = 0;

    bool valid() const noexcept {
        if (!bytes || width <= 0 || height <= 0 ||
            (width & 1) != 0 || (height & 1) != 0 ||
            y_stride_bytes < static_cast<std::size_t>(width) ||
            u_stride_bytes < static_cast<std::size_t>(width / 2) ||
            v_stride_bytes < static_cast<std::size_t>(width / 2)) {
            return false;
        }
        const auto planeFits = [&](std::size_t offset, std::size_t stride,
                                   int rows, std::size_t row_bytes) {
            if (rows <= 0 || offset > bytes->size()) return false;
            const std::size_t tail = static_cast<std::size_t>(rows - 1) * stride;
            return tail <= bytes->size() - offset &&
                row_bytes <= bytes->size() - offset - tail;
        };
        return planeFits(y_offset, y_stride_bytes, height,
                static_cast<std::size_t>(width)) &&
            planeFits(u_offset, u_stride_bytes, height / 2,
                static_cast<std::size_t>(width / 2)) &&
            planeFits(v_offset, v_stride_bytes, height / 2,
                static_cast<std::size_t>(width / 2));
    }

    bool tightlyPacked() const noexcept {
        const std::size_t y_bytes = static_cast<std::size_t>(width) * height;
        const std::size_t chroma_bytes = y_bytes / 4U;
        return valid() && y_offset == 0 && u_offset == y_bytes &&
            v_offset == y_bytes + chroma_bytes &&
            y_stride_bytes == static_cast<std::size_t>(width) &&
            u_stride_bytes == static_cast<std::size_t>(width / 2) &&
            v_stride_bytes == static_cast<std::size_t>(width / 2);
    }
};

struct FrameEnvelope {
    mutable cv::Mat image;
    I420FrameStorage i420;
    std::uint64_t sequence = 0;
    long long capture_time_ms = 0;
    std::chrono::steady_clock::time_point publish_time{};
    bool resolution_changed = false;

    bool valid() const noexcept {
        return !image.empty() || i420.valid();
    }

    const cv::Mat& bgrImage() const {
        if (!i420.valid()) return image;
        std::lock_guard<std::mutex> lock(*image_mutex_);
        if (!image.empty()) return image;
        cv::Mat packed;
        if (i420.tightlyPacked()) {
            packed = cv::Mat(
                i420.height * 3 / 2,
                i420.width,
                CV_8UC1,
                const_cast<std::uint8_t*>(i420.bytes->data()));
        }
        else {
            packed.create(i420.height * 3 / 2, i420.width, CV_8UC1);
            const auto copyPlane = [](const std::uint8_t* source,
                                      std::size_t source_stride,
                                      std::uint8_t* destination,
                                      std::size_t destination_stride,
                                      int rows,
                                      std::size_t row_bytes) {
                for (int row = 0; row < rows; ++row) {
                    std::memcpy(
                        destination + static_cast<std::size_t>(row) * destination_stride,
                        source + static_cast<std::size_t>(row) * source_stride,
                        row_bytes);
                }
            };
            auto* destination = packed.data;
            const std::size_t y_bytes = static_cast<std::size_t>(i420.width) * i420.height;
            const std::size_t chroma_bytes = y_bytes / 4U;
            copyPlane(i420.bytes->data() + i420.y_offset, i420.y_stride_bytes,
                destination, static_cast<std::size_t>(i420.width),
                i420.height, static_cast<std::size_t>(i420.width));
            copyPlane(i420.bytes->data() + i420.u_offset, i420.u_stride_bytes,
                destination + y_bytes, static_cast<std::size_t>(i420.width / 2),
                i420.height / 2, static_cast<std::size_t>(i420.width / 2));
            copyPlane(i420.bytes->data() + i420.v_offset, i420.v_stride_bytes,
                destination + y_bytes + chroma_bytes,
                static_cast<std::size_t>(i420.width / 2),
                i420.height / 2, static_cast<std::size_t>(i420.width / 2));
        }
        cv::cvtColor(packed, image, cv::COLOR_YUV2BGR_I420);
        return image;
    }

private:
    mutable std::shared_ptr<std::mutex> image_mutex_ =
        std::make_shared<std::mutex>();
};

using SharedCameraFrame = std::shared_ptr<const FrameEnvelope>;

struct SubscriberDescriptor {
    std::string subscriber_id;
    std::string subscriber_type;
};

struct SubscriptionMetrics {
    std::string subscriber_id;
    std::string subscriber_type;
    std::uint64_t last_seen_sequence = 0;
    long long consumed_frames = 0;
    long long skipped_frames = 0;
    long long last_consume_time_ms = 0;
};

struct FrameReadResult {
    SharedCameraFrame frame;
    std::uint64_t skipped_since_last_read = 0;
};

}  // namespace yolo11_server
