#pragma once
#include <glm/glm.hpp>
#include <string>
#include <vector>

namespace DigitalTwin
{
    /**
     * @brief Standard Vertex format for Compute-Centric rendering.
     * Uses vec4 for all attributes to ensure 16-byte alignment in SSBOs (std430).
     */
    struct Vertex
    {
        glm::vec4 position; // xyz = world pos, w = 1.0 (or radius/padding)
        glm::vec4 normal;   // xyz = normal vector, w = 0.0 (padding)
        glm::vec4 color;    // rgba
    };

    /**
     * @brief CPU-side representation of a mesh.
     * Contains raw data used to generate GPU buffers.
     */
    struct Mesh
    {
        std::vector<Vertex>   vertices;
        std::vector<uint32_t> indices;
        std::string           name;
    };
} // namespace DigitalTwin