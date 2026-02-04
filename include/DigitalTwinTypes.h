#pragma once

#include "core/Core.h"
#include "core/Handle.h"

namespace DigitalTwin
{
    class Log;
    class FileSystem;

    enum class GPUType
    {
        DEFAULT,
        DISCRETE,
        INTEGRATED,
    };

    struct DigitalTwinConfig
    {
        GPUType     gpuType       = GPUType::DEFAULT;
        bool_t      headless      = true;
        const char* rootDirectory = nullptr;
        bool_t      debugMode     = false;
        uint32_t    windowWidth   = 1280;
        uint32_t    windowHeight  = 720;
        const char* windowTitle   = "Digital Twin Application";
    };

} // namespace DigitalTwin