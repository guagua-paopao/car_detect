#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "business/vehicle_cascade_runtime.h"
#include "server/app_config.h"
#include "server/vehicle_model_contract.h"

namespace yolo11_server {

struct VehicleRuntimeAssets {
    ModelArtifactDescriptor detector;
    ModelArtifactDescriptor attributes;
    std::vector<std::string> vehicle_classes;
    std::vector<std::string> body_types;
    std::vector<std::string> colors;
    VehicleCascadeRuntimeConfig cascade;
    float detection_confidence_threshold = 0.25f;
    std::string config_version;
    std::string labels_version;
    std::size_t attribute_max_batch = 16;
};

bool loadVehicleRuntimeAssets(
    const AppConfig& config,
    VehicleRuntimeAssets& assets,
    std::string& error);

}  // namespace yolo11_server
