#include "simulation/SpatialDistribution.h"

#include <glm/gtc/constants.hpp>
#include <random>

namespace DigitalTwin
{
    std::vector<glm::vec4> SpatialDistribution::UniformInSphere( uint32_t count, float radius, const glm::vec3& center )
    {
        std::vector<glm::vec4> positions;
        positions.reserve( count );

        std::random_device                    rd;
        std::mt19937                          gen( rd() );
        std::uniform_real_distribution<float> dis( 0.0f, 1.0f );

        for( uint32_t i = 0; i < count; ++i )
        {
            // Spherical coordinates for uniform distribution inside a volume
            float u = dis( gen );
            float v = dis( gen );
            float w = dis( gen );

            float theta = u * 2.0f * glm::pi<float>();
            float phi   = glm::acos( 2.0f * v - 1.0f );
            float r     = radius * std::cbrt( w );

            float x = center.x + r * glm::sin( phi ) * glm::cos( theta );
            float y = center.y + r * glm::sin( phi ) * glm::sin( theta );
            float z = center.z + r * glm::cos( phi );

            // w = 1.0f represents an "alive" agent by default
            positions.push_back( { x, y, z, 1.0f } );
        }

        return positions;
    }

    std::vector<glm::vec4> SpatialDistribution::UniformInBox( uint32_t count, const glm::vec3& extents, const glm::vec3& center )
    {
        std::vector<glm::vec4> positions;
        positions.reserve( count );

        std::random_device                    rd;
        std::mt19937                          gen( rd() );
        std::uniform_real_distribution<float> disX( -extents.x, extents.x );
        std::uniform_real_distribution<float> disY( -extents.y, extents.y );
        std::uniform_real_distribution<float> disZ( -extents.z, extents.z );

        for( uint32_t i = 0; i < count; ++i )
        {
            float x = center.x + disX( gen );
            float y = center.y + disY( gen );
            float z = center.z + disZ( gen );

            positions.push_back( { x, y, z, 1.0f } );
        }

        return positions;
    }

} // namespace DigitalTwin