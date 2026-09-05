#pragma once

#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "types.h"
#include "perf_metrics.h"
#include "server/vehicle_model_contract.h"

namespace yolo11_server {

    // Phase 17: unified model output container.
    // Detect/OBB fill detections; CLS fills classifications.
    // Segmentation can extend this container without forcing every model
    // into std::vector<Detection>.
    struct ClassificationItem {
        int class_id = -1;
        std::string class_name;
        float confidence = 0.0f;
    };

    struct SegmentationItem {
        Detection detection;

        // Letterboxed model-input mask from YOLO segmentation head.
        // It is mapped back to original image pixels in ResultSerializer.
        cv::Mat mask;
    };

    struct ModelOutput {
        std::string model_type = "detect";
        std::vector<Detection> detections;
        std::vector<ClassificationItem> classifications;
        std::vector<SegmentationItem> segmentations;
        // Vehicle camera pipeline payload.  The generic Detection type cannot
        // carry the detector artifact/labels contract needed by the attribute
        // cascade, so keep the typed result intact across the inference pool.
        bool has_vehicle_detection = false;
        VehicleDetectionResult vehicle_detection;

        // Phase P0: split performance metrics. Filled by model API / worker when available.
        PerfMetrics perf;

        bool empty() const {
            return detections.empty() && classifications.empty() &&
                segmentations.empty() &&
                (!has_vehicle_detection || vehicle_detection.detections.empty());
        }
    };

}  // namespace yolo11_server
