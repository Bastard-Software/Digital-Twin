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

    MorphologyData MorphologyGenerator::CreateCylinder( float radius, float height, uint32_t sectors )
    {
        MorphologyData data;
        float          hh   = height * 0.5f;
        float          step = 2.0f * glm::pi<float>() / sectors;

        // Side body: interleaved top/bottom ring vertices (smooth normals in XZ plane)
        for( uint32_t i = 0; i <= sectors; ++i )
        {
            float angle = i * step;
            float c     = cosf( angle );
            float s     = sinf( angle );
            data.vertices.push_back( { glm::vec4( radius * c, hh, radius * s, 1.0f ), glm::vec4( c, 0.0f, s, 0.0f ) } );
            data.vertices.push_back( { glm::vec4( radius * c, -hh, radius * s, 1.0f ), glm::vec4( c, 0.0f, s, 0.0f ) } );
        }
        for( uint32_t i = 0; i < sectors; ++i )
        {
            uint32_t t0 = i * 2, b0 = i * 2 + 1, t1 = ( i + 1 ) * 2, b1 = ( i + 1 ) * 2 + 1;
            data.indices.push_back( t0 ); data.indices.push_back( t1 ); data.indices.push_back( b0 );
            data.indices.push_back( t1 ); data.indices.push_back( b1 ); data.indices.push_back( b0 );
        }

        // Top cap (normal +Y, flat)
        uint32_t topCenter = static_cast<uint32_t>( data.vertices.size() );
        data.vertices.push_back( { glm::vec4( 0.0f, hh, 0.0f, 1.0f ), glm::vec4( 0.0f, 1.0f, 0.0f, 0.0f ) } );
        uint32_t topRim = static_cast<uint32_t>( data.vertices.size() );
        for( uint32_t i = 0; i <= sectors; ++i )
        {
            float angle = i * step;
            data.vertices.push_back( { glm::vec4( radius * cosf( angle ), hh, radius * sinf( angle ), 1.0f ),
                                       glm::vec4( 0.0f, 1.0f, 0.0f, 0.0f ) } );
        }
        for( uint32_t i = 0; i < sectors; ++i )
        {
            data.indices.push_back( topCenter );
            data.indices.push_back( topRim + i + 1 );
            data.indices.push_back( topRim + i );
        }

        // Bottom cap (normal -Y, flat, reversed winding)
        uint32_t botCenter = static_cast<uint32_t>( data.vertices.size() );
        data.vertices.push_back( { glm::vec4( 0.0f, -hh, 0.0f, 1.0f ), glm::vec4( 0.0f, -1.0f, 0.0f, 0.0f ) } );
        uint32_t botRim = static_cast<uint32_t>( data.vertices.size() );
        for( uint32_t i = 0; i <= sectors; ++i )
        {
            float angle = i * step;
            data.vertices.push_back( { glm::vec4( radius * cosf( angle ), -hh, radius * sinf( angle ), 1.0f ),
                                       glm::vec4( 0.0f, -1.0f, 0.0f, 0.0f ) } );
        }
        for( uint32_t i = 0; i < sectors; ++i )
        {
            data.indices.push_back( botCenter );
            data.indices.push_back( botRim + i );
            data.indices.push_back( botRim + i + 1 );
        }

        return data;
    }

    MorphologyData MorphologyGenerator::CreateSpikySphere( float radius, float spikeScale, uint32_t sectors, uint32_t stacks )
    {
        MorphologyData data;
        float          sectorStep = 2.0f * glm::pi<float>() / sectors;
        float          stackStep  = glm::pi<float>() / stacks;
        float          lengthInv  = 1.0f / radius;

        for( uint32_t i = 0; i <= stacks; ++i )
        {
            float stackAngle = glm::pi<float>() / 2.0f - i * stackStep;
            float xy         = radius * cosf( stackAngle );
            float z          = radius * sinf( stackAngle );

            for( uint32_t j = 0; j <= sectors; ++j )
            {
                float sectorAngle = j * sectorStep;
                float x           = xy * cosf( sectorAngle );
                float y           = xy * sinf( sectorAngle );

                // Alternate vertex positions outward to form spikes
                float scale = ( ( i + j ) % 2 == 0 ) ? spikeScale : 1.0f;

                Vertex vertex;
                vertex.pos    = glm::vec4( x * scale, y * scale, z * scale, 1.0f );
                vertex.normal = glm::vec4( x * lengthInv, y * lengthInv, z * lengthInv, 0.0f );
                data.vertices.push_back( vertex );
            }
        }

        // The checkerboard spike pattern creates concave valley faces whose projected winding
        // flips depending on viewing angle. Generate each triangle in both winding orders
        // (double-sided) so concave valley faces are never culled. Triangle count doubles
        // but is negligible for a handful of TipCells.
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
                    data.indices.push_back( k1 );
                    data.indices.push_back( k1 + 1 );
                    data.indices.push_back( k2 );
                }
                if( i != ( stacks - 1 ) )
                {
                    data.indices.push_back( k1 + 1 );
                    data.indices.push_back( k2 );
                    data.indices.push_back( k2 + 1 );
                    data.indices.push_back( k1 + 1 );
                    data.indices.push_back( k2 + 1 );
                    data.indices.push_back( k2 );
                }
            }
        }

        return data;
    }
} // namespace DigitalTwin