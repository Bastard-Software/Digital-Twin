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

    std::vector<glm::vec4> SpatialDistribution::VesselLine( uint32_t count, const glm::vec3& start, const glm::vec3& end, float spacing )
    {
        std::vector<glm::vec4> positions;
        if( count == 0 )
            return positions;

        positions.reserve( count );

        if( count == 1 )
        {
            glm::vec3 mid = ( start + end ) * 0.5f;
            positions.push_back( { mid.x, mid.y, mid.z, 1.0f } );
            return positions;
        }

        glm::vec3 dir        = end - start;
        float     lineLength = glm::length( dir );

        if( spacing > 0.0f && lineLength > 0.0f )
        {
            glm::vec3 step = glm::normalize( dir ) * spacing;
            for( uint32_t i = 0; i < count; ++i )
            {
                float dist = spacing * static_cast<float>( i );
                if( dist > lineLength )
                    break;
                glm::vec3 p = start + step * static_cast<float>( i );
                positions.push_back( { p.x, p.y, p.z, 1.0f } );
            }
        }
        else
        {
            // Even spacing: step = (end - start) / (count - 1)
            glm::vec3 step = dir / static_cast<float>( count - 1 );
            for( uint32_t i = 0; i < count; ++i )
            {
                glm::vec3 p = start + step * static_cast<float>( i );
                positions.push_back( { p.x, p.y, p.z, 1.0f } );
            }
        }

        return positions;
    }

    std::vector<glm::vec4> SpatialDistribution::LatticeInSphere( float spacing, float radius, const glm::vec3& center )
    {
        std::vector<glm::vec4> positions;

        for( float x = -radius; x <= radius; x += spacing )
            for( float y = -radius; y <= radius; y += spacing )
                for( float z = -radius; z <= radius; z += spacing )
                    if( x * x + y * y + z * z <= radius * radius )
                        positions.push_back( { center.x + x, center.y + y, center.z + z, 1.0f } );

        return positions;
    }

} // namespace DigitalTwin