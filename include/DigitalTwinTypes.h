#pragma once

#include "core/Core.h"
#include "core/Handle.h"

namespace DigitalTwin
{
    class Log;
    class FileSystem;
    class Window;

    DEFINE_HANDLE( WindowHandle );

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