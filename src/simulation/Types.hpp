#pragma once
#include <cstdint>
#include <glm/glm.hpp>

namespace DigitalTwin
{
    /**
     * @brief High-level biological entity representation compatible with GPU (std430).
     * Represents a single agent/cell in the simulation.
     */
    struct Cell
    {
        glm::vec4 position; // xyz = position, w = radius/size
        glm::vec4 velocity; // xyz = velocity, w = state/type
        glm::vec4 color;    // rgba = visualization color (phenotype expression)
        // --- Metadata (16 bytes aligned) ---
        uint32_t meshID; // Which mesh to render?
        uint32_t _pad0;
        uint32_t _pad1;
        uint32_t _pad2; // Padding to align struct to 16 bytes multiple (total 64 bytes)
    };

    /**
     * @brief Global environment parameters passed to shaders as Uniforms.
     */
    struct EnvironmentParams
    {
        float viscosity;
        float nutrientDensity;
        float gravity;
        float _padding; // Alignment padding
    };
} // namespace DigitalTwin