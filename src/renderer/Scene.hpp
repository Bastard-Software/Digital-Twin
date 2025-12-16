#pragma once
#include "renderer/Camera.hpp"
#include "resources/GPUMesh.hpp"
#include "rhi/Buffer.hpp"
#include <glm/glm.hpp>

namespace DigitalTwin
{

    struct Scene
    {
        Camera* camera = nullptr;

        // Instanced Rendering Data
        Ref<Buffer> instanceBuffer = nullptr;
        uint32_t    instanceCount  = 0;

        // Geometry to render
        Ref<GPUMesh> mesh = nullptr;
    };
} // namespace DigitalTwin