#include "simulation/MorphologyGenerator.h"

#include <cmath>
#include <glm/gtc/constants.hpp>

namespace DigitalTwin
{
    MorphologyData MorphologyGenerator::CreateCube( float size )
    {
        MorphologyData data;
        float          hs = size * 0.5f;

        // Flat-shaded cube requires 24 vertices (4 per face)
        data.vertices = { // Front - Normal (0, 0, 1)
                          { { -hs, -hs, hs, 1.0f }, { 0.0f, 0.0f, 1.0f, 0.0f } },
                          { { hs, -hs, hs, 1.0f }, { 0.0f, 0.0f, 1.0f, 0.0f } },
                          { { hs, hs, hs, 1.0f }, { 0.0f, 0.0f, 1.0f, 0.0f } },
                          { { -hs, hs, hs, 1.0f }, { 0.0f, 0.0f, 1.0f, 0.0f } },
                          // Back - Normal (0, 0, -1)
                          { { hs, -hs, -hs, 1.0f }, { 0.0f, 0.0f, -1.0f, 0.0f } },
                          { { -hs, -hs, -hs, 1.0f }, { 0.0f, 0.0f, -1.0f, 0.0f } },
                          { { -hs, hs, -hs, 1.0f }, { 0.0f, 0.0f, -1.0f, 0.0f } },
                          { { hs, hs, -hs, 1.0f }, { 0.0f, 0.0f, -1.0f, 0.0f } },
                          // Left - Normal (-1, 0, 0)
                          { { -hs, -hs, -hs, 1.0f }, { -1.0f, 0.0f, 0.0f, 0.0f } },
                          { { -hs, -hs, hs, 1.0f }, { -1.0f, 0.0f, 0.0f, 0.0f } },
                          { { -hs, hs, hs, 1.0f }, { -1.0f, 0.0f, 0.0f, 0.0f } },
                          { { -hs, hs, -hs, 1.0f }, { -1.0f, 0.0f, 0.0f, 0.0f } },
                          // Right - Normal (1, 0, 0)
                          { { hs, -hs, hs, 1.0f }, { 1.0f, 0.0f, 0.0f, 0.0f } },
                          { { hs, -hs, -hs, 1.0f }, { 1.0f, 0.0f, 0.0f, 0.0f } },
                          { { hs, hs, -hs, 1.0f }, { 1.0f, 0.0f, 0.0f, 0.0f } },
                          { { hs, hs, hs, 1.0f }, { 1.0f, 0.0f, 0.0f, 0.0f } },
                          // Top - Normal (0, 1, 0)
                          { { -hs, hs, hs, 1.0f }, { 0.0f, 1.0f, 0.0f, 0.0f } },
                          { { hs, hs, hs, 1.0f }, { 0.0f, 1.0f, 0.0f, 0.0f } },
                          { { hs, hs, -hs, 1.0f }, { 0.0f, 1.0f, 0.0f, 0.0f } },
                          { { -hs, hs, -hs, 1.0f }, { 0.0f, 1.0f, 0.0f, 0.0f } },
                          // Bottom - Normal (0, -1, 0)
                          { { -hs, -hs, -hs, 1.0f }, { 0.0f, -1.0f, 0.0f, 0.0f } },
                          { { hs, -hs, -hs, 1.0f }, { 0.0f, -1.0f, 0.0f, 0.0f } },
                          { { hs, -hs, hs, 1.0f }, { 0.0f, -1.0f, 0.0f, 0.0f } },
                          { { -hs, -hs, hs, 1.0f }, { 0.0f, -1.0f, 0.0f, 0.0f } }
        };

        data.indices = {
            0,  1,  2,  2,  3,  0,  // Front
            4,  5,  6,  6,  7,  4,  // Back
            8,  9,  10, 10, 11, 8,  // Left
            12, 13, 14, 14, 15, 12, // Right
            16, 17, 18, 18, 19, 16, // Top
            20, 21, 22, 22, 23, 20  // Bottom
        };

        return data;
    }

    MorphologyData MorphologyGenerator::CreateSphere( float radius, uint32_t sectors, uint32_t stacks )
    {
        MorphologyData data;

        float lengthInv  = 1.0f / radius;
        float sectorStep = 2.0f * glm::pi<float>() / sectors;
        float stackStep  = glm::pi<float>() / stacks;

        for( uint32_t i = 0; i <= stacks; ++i )
        {
            float stackAngle = glm::pi<float>() / 2.0f - i * stackStep; // from pi/2 to -pi/2
            float xy         = radius * cosf( stackAngle );
            float z          = radius * sinf( stackAngle );

            for( uint32_t j = 0; j <= sectors; ++j )
            {
                float sectorAngle = j * sectorStep;

                float x = xy * cosf( sectorAngle );
                float y = xy * sinf( sectorAngle );

                Vertex vertex;
                vertex.pos = glm::vec4( x, y, z, 1.0f );
                // Normalized normal vector
                vertex.normal = glm::vec4( x * lengthInv, y * lengthInv, z * lengthInv, 0.0f );

                data.vertices.push_back( vertex );
            }
        }

        for( uint32_t i = 0; i < stacks; ++i )
        {
            uint32_t k1 = i * ( sectors + 1 );
            uint32_t k2 = k1 + sectors + 1;

            for( uint32_t j = 0; j < sectors; ++j, ++k1, ++k2 )
            {
                if( i != 0 )
                {
                    data.indices.push_back( k1 );
                    data.indices.push_back( k2 );
                    data.indices.push_back( k1 + 1 );
                }
                if( i != ( stacks - 1 ) )
                {
                    data.indices.push_back( k1 + 1 );
                    data.indices.push_back( k2 );
                    data.indices.push_back( k2 + 1 );
                }
            }
        }

        return data;
    }
} // namespace DigitalTwin