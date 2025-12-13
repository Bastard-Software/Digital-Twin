#pragma once
#include "renderer/Camera.hpp"
#include "rhi/Buffer.hpp"
#include <glm/glm.hpp>

namespace DigitalTwin
{

    struct Scene
    {
        Camera* camera = nullptr;

        // Raw GPU Data
        Ref<Buffer> instanceBuffer = nullptr;
        uint32_t    instanceCount  = 0;
    };
} // namespace DigitalTwin