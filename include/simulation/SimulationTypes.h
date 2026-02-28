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
    };

} // namespace DigitalTwin