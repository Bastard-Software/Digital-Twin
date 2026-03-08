#pragma once
#include "core/Core.h"
#include <glm/glm.hpp>
#include <vector>

namespace DigitalTwin
{
    /**
     * @brief Mathematical strategies for spatial distribution (seeding) of agents in the simulation volume.
     */
    class DT_API SpatialDistribution
    {
    public:
        /**
         * @brief Distributes agents uniformly inside a 3D spherical volume.
         * @param count Number of agents to seed.
         * @param radius The radius of the bounding sphere.
         * @param center The central coordinate of the distribution.
         * @return std::vector<glm::vec4> (xyz = position, w = 1.0f status flag)
         */
        static std::vector<glm::vec4> UniformInSphere( uint32_t count, float radius, const glm::vec3& center = glm::vec3( 0.0f ) );

        /**
         * @brief Distributes agents uniformly inside a 3D bounding box.
         * @param count Number of agents to seed.
         * @param extents Half-dimensions of the box (width, height, depth).
         * @param center The central coordinate of the distribution.
         */
        static std::vector<glm::vec4> UniformInBox( uint32_t count, const glm::vec3& extents, const glm::vec3& center = glm::vec3( 0.0f ) );
    };

} // namespace DigitalTwin