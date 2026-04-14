#pragma once
#include <glm/glm.hpp>
#include <vector>

namespace DigitalTwin
{
    /**
     * @brief Standard vertex format for the simulation's visual representation.
     */
    struct Vertex
    {
        glm::vec4 pos;
        glm::vec4 normal;
    };

    /**
     * @brief Container for the physical shape/morphology of an agent.
     */
    struct MorphologyData
    {
        std::vector<Vertex>   vertices;
        std::vector<uint32_t> indices;

        // Contact hull for rigid body dynamics (model-space sample points).
        // xyz = offset from cell centre, w = sub-sphere radius.
        // Empty → point-particle fallback (no torque, existing JKR behaviour).
        // Maximum 16 points uploaded to GPU (MAX_HULL_POINTS in jkr_forces.comp).
        std::vector<glm::vec4> contactHull;

        // Model-space half-extents for oriented steric repulsion (box support function).
        // Z = model-Z half-extent, Y = model-Y half-extent (thickness / normal direction).
        // Model-X half-extent is derived on the GPU from max |hull.points[h].x|.
        // Both 0.0 → steric disabled (backward-compatible for non-tile morphologies).
        float hullExtentZ = 0.0f;
        float hullExtentY = 0.0f;

        // Reserved scalar for morphology-specific shader parameters. Currently unused.
        float edgeAlignStrength = 0.0f;
    };

} // namespace DigitalTwin