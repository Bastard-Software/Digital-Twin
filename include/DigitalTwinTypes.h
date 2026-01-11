#pragma once

#include "core/Core.h"

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
    };

} // namespace DigitalTwin