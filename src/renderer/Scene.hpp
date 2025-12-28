#pragma once
#include "renderer/Camera.hpp"
#include "resources/GPUMesh.hpp"
#include "rhi/Buffer.hpp"
#include <glm/glm.hpp>
#include <vector>

namespace DigitalTwin
{
    /**
     * @brief Describes the scene to be rendered in the current frame.
     */
    struct Scene
    {
        Camera* camera = nullptr;

        // Instanced Rendering Data
        Ref<Buffer> instanceBuffer  = nullptr;
        Ref<Buffer> activeInstances = nullptr; // Atomic Counter

        // Fallback count if atomic counter is not used for Indirect Draw
        uint32_t instanceCount = 0;

        // Geometry to render
        std::vector<AssetID> activeMeshIDs;
    };
} // namespace DigitalTwin