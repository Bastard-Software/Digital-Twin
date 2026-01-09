#pragma once

#include "core/Core.h"

namespace DigitalTwin
{
    class Log;
    class FileSystem;

    struct DigitalTwinConfig
    {
        bool_t      headless      = true;
        const char* rootDirectory = nullptr;
    };

} // namespace DigitalTwin