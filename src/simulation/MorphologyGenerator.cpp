#include "simulation/MorphologyGenerator.h"

#include <cmath>
#include <glm/gtc/constants.hpp>
#include <glm/trigonometric.hpp>

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

        // Contact hull: 12 points — 8 corners + 4 equatorial edge midpoints, sub-sphere radius = size/4
        const float r = size * 0.25f;
        data.contactHull = {
            // 8 corners
            { -hs, -hs, -hs, r }, { +hs, -hs, -hs, r },
            { -hs, +hs, -hs, r }, { +hs, +hs, -hs, r },
            { -hs, -hs, +hs, r }, { +hs, -hs, +hs, r },
            { -hs, +hs, +hs, r }, { +hs, +hs, +hs, r },
            // 4 equatorial edge midpoints (Y=0 plane)
            { +hs, 0.0f, 0.0f, r },
            { -hs, 0.0f, 0.0f, r },
            { 0.0f, 0.0f, +hs, r },
            { 0.0f, 0.0f, -hs, r },
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

        // Contact hull: 4 top-rim + 4 bottom-rim points at 90° intervals, sub-sphere radius = radius/3
        const float r = radius / 3.0f;
        for( int i = 0; i < 4; ++i )
        {
            float angle = i * glm::pi<float>() * 0.5f;
            float c = cosf( angle ), s = sinf( angle );
            data.contactHull.push_back( { radius * c, +hh, radius * s, r } );
            data.contactHull.push_back( { radius * c, -hh, radius * s, r } );
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

        // Contact hull: 8 spike-tip points at cardinal + diagonal directions.
        // Sub-sphere radius = R * 0.25 (same ratio as CreateCube).
        // The checkerboard spike pattern means any axis direction hits a spike tip
        // or comes very close to one — 8 points cover all meaningful contact directions.
        const float R = radius * spikeScale;
        const float r = R * 0.25f;
        const float d = R * 0.7071f; // R / sqrt(2) for diagonal directions
        data.contactHull = {
            {  +R,  0.0f, 0.0f, r }, // +X
            {  -R,  0.0f, 0.0f, r }, // -X
            { 0.0f,  +R,  0.0f, r }, // +Y
            { 0.0f,  -R,  0.0f, r }, // -Y
            {  +d,   +d,  0.0f, r }, // +XY diagonal
            {  -d,   -d,  0.0f, r }, // -XY diagonal
            { 0.0f,  +d,   +d,  r }, // +YZ diagonal
            { 0.0f,  -d,   -d,  r }, // -YZ diagonal
        };

        return data;
    }

    MorphologyData MorphologyGenerator::CreateDisc( float radius, float thickness, uint32_t sectors )
    {
        MorphologyData data;
        float          hh   = thickness * 0.5f;
        float          step = 2.0f * glm::pi<float>() / sectors;

        // Top face — normal +Y, fan from center
        uint32_t topCenter = static_cast<uint32_t>( data.vertices.size() );
        data.vertices.push_back( { glm::vec4( 0.0f, hh, 0.0f, 1.0f ), glm::vec4( 0.0f, 1.0f, 0.0f, 0.0f ) } );
        uint32_t topRim = static_cast<uint32_t>( data.vertices.size() );
        for( uint32_t i = 0; i <= sectors; ++i )
        {
            float angle = i * step;
            float c = cosf( angle ), s = sinf( angle );
            data.vertices.push_back( { glm::vec4( radius * c, hh, radius * s, 1.0f ),
                                       glm::vec4( 0.0f, 1.0f, 0.0f, 0.0f ) } );
        }
        for( uint32_t i = 0; i < sectors; ++i )
        {
            data.indices.push_back( topCenter );
            data.indices.push_back( topRim + i + 1 );
            data.indices.push_back( topRim + i );
        }

        // Bottom face — normal -Y, reversed winding
        uint32_t botCenter = static_cast<uint32_t>( data.vertices.size() );
        data.vertices.push_back( { glm::vec4( 0.0f, -hh, 0.0f, 1.0f ), glm::vec4( 0.0f, -1.0f, 0.0f, 0.0f ) } );
        uint32_t botRim = static_cast<uint32_t>( data.vertices.size() );
        for( uint32_t i = 0; i <= sectors; ++i )
        {
            float angle = i * step;
            float c = cosf( angle ), s = sinf( angle );
            data.vertices.push_back( { glm::vec4( radius * c, -hh, radius * s, 1.0f ),
                                       glm::vec4( 0.0f, -1.0f, 0.0f, 0.0f ) } );
        }
        for( uint32_t i = 0; i < sectors; ++i )
        {
            data.indices.push_back( botCenter );
            data.indices.push_back( botRim + i );
            data.indices.push_back( botRim + i + 1 );
        }

        // Edge band — outward radial normals connecting top and bottom rims
        uint32_t edgeBase = static_cast<uint32_t>( data.vertices.size() );
        for( uint32_t i = 0; i <= sectors; ++i )
        {
            float angle = i * step;
            float c = cosf( angle ), s = sinf( angle );
            data.vertices.push_back( { glm::vec4( radius * c, hh,  radius * s, 1.0f ), glm::vec4( c, 0.0f, s, 0.0f ) } );
            data.vertices.push_back( { glm::vec4( radius * c, -hh, radius * s, 1.0f ), glm::vec4( c, 0.0f, s, 0.0f ) } );
        }
        for( uint32_t i = 0; i < sectors; ++i )
        {
            uint32_t t0 = edgeBase + i * 2,     b0 = edgeBase + i * 2 + 1;
            uint32_t t1 = edgeBase + ( i + 1 ) * 2, b1 = edgeBase + ( i + 1 ) * 2 + 1;
            data.indices.push_back( t0 ); data.indices.push_back( t1 ); data.indices.push_back( b0 );
            data.indices.push_back( t1 ); data.indices.push_back( b1 ); data.indices.push_back( b0 );
        }

        // Contact hull: 8 circumference points at 45° intervals, sub-sphere radius = thickness/2
        const float r = hh; // = thickness/2
        for( int i = 0; i < 8; ++i )
        {
            float angle = i * glm::pi<float>() * 0.25f;
            data.contactHull.push_back( { radius * cosf( angle ), 0.0f, radius * sinf( angle ), r } );
        }

        return data;
    }

    MorphologyData MorphologyGenerator::CreateTile( float width, float height, float thickness )
    {
        MorphologyData data;
        const float hw = width     * 0.5f; // half-width  (X)
        const float hh = height    * 0.5f; // half-height (Z)
        const float ht = thickness * 0.5f; // half-thickness (Y)

        // 24 vertices (4 per face), same layout as CreateCube but with independent hw/hh/ht.
        data.vertices = {
            // Top face  — normal +Y (outward from vessel surface)
            { { -hw, +ht, -hh, 1.0f }, { 0.0f,  1.0f, 0.0f, 0.0f } },
            { { +hw, +ht, -hh, 1.0f }, { 0.0f,  1.0f, 0.0f, 0.0f } },
            { { +hw, +ht, +hh, 1.0f }, { 0.0f,  1.0f, 0.0f, 0.0f } },
            { { -hw, +ht, +hh, 1.0f }, { 0.0f,  1.0f, 0.0f, 0.0f } },
            // Bottom face — normal -Y (toward vessel lumen)
            { { -hw, -ht, +hh, 1.0f }, { 0.0f, -1.0f, 0.0f, 0.0f } },
            { { +hw, -ht, +hh, 1.0f }, { 0.0f, -1.0f, 0.0f, 0.0f } },
            { { +hw, -ht, -hh, 1.0f }, { 0.0f, -1.0f, 0.0f, 0.0f } },
            { { -hw, -ht, -hh, 1.0f }, { 0.0f, -1.0f, 0.0f, 0.0f } },
            // Front face — normal +Z (one axial side)
            { { -hw, -ht, +hh, 1.0f }, { 0.0f, 0.0f,  1.0f, 0.0f } },
            { { +hw, -ht, +hh, 1.0f }, { 0.0f, 0.0f,  1.0f, 0.0f } },
            { { +hw, +ht, +hh, 1.0f }, { 0.0f, 0.0f,  1.0f, 0.0f } },
            { { -hw, +ht, +hh, 1.0f }, { 0.0f, 0.0f,  1.0f, 0.0f } },
            // Back face  — normal -Z (other axial side)
            { { +hw, -ht, -hh, 1.0f }, { 0.0f, 0.0f, -1.0f, 0.0f } },
            { { -hw, -ht, -hh, 1.0f }, { 0.0f, 0.0f, -1.0f, 0.0f } },
            { { -hw, +ht, -hh, 1.0f }, { 0.0f, 0.0f, -1.0f, 0.0f } },
            { { +hw, +ht, -hh, 1.0f }, { 0.0f, 0.0f, -1.0f, 0.0f } },
            // Right face — normal +X (one circumferential side)
            { { +hw, -ht, +hh, 1.0f }, { 1.0f, 0.0f, 0.0f, 0.0f } },
            { { +hw, -ht, -hh, 1.0f }, { 1.0f, 0.0f, 0.0f, 0.0f } },
            { { +hw, +ht, -hh, 1.0f }, { 1.0f, 0.0f, 0.0f, 0.0f } },
            { { +hw, +ht, +hh, 1.0f }, { 1.0f, 0.0f, 0.0f, 0.0f } },
            // Left face  — normal -X (other circumferential side)
            { { -hw, -ht, -hh, 1.0f }, { -1.0f, 0.0f, 0.0f, 0.0f } },
            { { -hw, -ht, +hh, 1.0f }, { -1.0f, 0.0f, 0.0f, 0.0f } },
            { { -hw, +ht, +hh, 1.0f }, { -1.0f, 0.0f, 0.0f, 0.0f } },
            { { -hw, +ht, -hh, 1.0f }, { -1.0f, 0.0f, 0.0f, 0.0f } },
        };

        data.indices = {
             2,  1,  0,  0,  3,  2,  // Top    (reversed: was CW from above, now CCW)
             6,  5,  4,  4,  7,  6,  // Bottom (reversed: was CCW from above, now CW)
             8,  9, 10, 10, 11,  8,  // Front  (original — already correct)
            12, 13, 14, 14, 15, 12,  // Back   (original — already correct)
            16, 17, 18, 18, 19, 16,  // Right  (original — already correct)
            20, 21, 22, 22, 23, 20,  // Left   (original — already correct)
        };

        // Contact hull: 8 points — 2 per edge at ±1/2 edge-length positions (Y=0 mid-plane).
        // Two sub-spheres per edge define each cadherin-bearing interface; this guarantees a
        // non-zero lever arm (and thus a non-zero restoring torque) at all rotation angles,
        // and avoids single-point null-torque configurations.  No corner points — corners
        // cannot generate useful alignment torques and can lock the tile at off-zero angles.
        const float hullR = ht * 2.0f; // cadherin contact zone radius (~VE-cadherin plaque width)
        const float hx    = hw * 0.5f; // half of half-width  — lateral offset on front/back edges
        const float hz    = hh * 0.5f; // half of half-height — axial offset on right/left edges
        data.contactHull = {
            { +hw, 0.0f, +hz, hullR },  // right edge, front point
            { +hw, 0.0f, -hz, hullR },  // right edge, back point
            { -hw, 0.0f, +hz, hullR },  // left edge,  front point
            { -hw, 0.0f, -hz, hullR },  // left edge,  back point
            { +hx, 0.0f, +hh, hullR },  // front edge, right point
            { -hx, 0.0f, +hh, hullR },  // front edge, left point
            { +hx, 0.0f, -hh, hullR },  // back edge,  right point
            { -hx, 0.0f, -hh, hullR },  // back edge,  left point
        };
        data.hullExtentZ = hh * 0.5f; // reduced steric Z-extent keeps tiles inside interactDist at all demo angles
        data.hullExtentY = ht;        // model-Y half-extent (thickness / 2)

        return data;
    }

    MorphologyData MorphologyGenerator::CreateCurvedTile(
        float arcAngleDeg, float height, float thickness, float innerRadius, uint32_t sectors )
    {
        MorphologyData data;
        const float halfArc = glm::radians( arcAngleDeg ) * 0.5f;
        const float arcStep = glm::radians( arcAngleDeg ) / static_cast<float>( sectors );
        const float R_in    = innerRadius;
        const float R_out   = innerRadius + thickness;
        const float R_mid   = innerRadius + thickness * 0.5f;
        const float hh      = height * 0.5f;

        // Arc is in the YZ plane. At θ=0 the outer face normal = +Y (orientation convention).
        // Local X = vessel axial direction (height).
        // Local Z = circumferential direction (arc extent, maps to tangent in world after rotation).
        auto outerPt = [&]( float x, float t ) -> glm::vec4 {
            return { x, R_out * cosf( t ) - R_mid, R_out * sinf( t ), 1.0f };
        };
        auto innerPt = [&]( float x, float t ) -> glm::vec4 {
            return { x, R_in * cosf( t ) - R_mid, R_in * sinf( t ), 1.0f };
        };

        // Per-arc-segment vertex layout: L (x=-hh) then R (x=+hh).

        // ---- 1. Outer face (outward radial normal) ----
        uint32_t outerBase = static_cast<uint32_t>( data.vertices.size() );
        for( uint32_t i = 0; i <= sectors; ++i )
        {
            float t = -halfArc + i * arcStep;
            glm::vec4 nrm{ 0.0f, cosf( t ), sinf( t ), 0.0f };
            data.vertices.push_back( { outerPt( -hh, t ), nrm } );  // L
            data.vertices.push_back( { outerPt( +hh, t ), nrm } );  // R
        }
        for( uint32_t i = 0; i < sectors; ++i )
        {
            uint32_t L0 = outerBase + i * 2,           L1 = outerBase + ( i + 1 ) * 2;
            uint32_t R0 = outerBase + i * 2 + 1,       R1 = outerBase + ( i + 1 ) * 2 + 1;
            data.indices.push_back( L0 ); data.indices.push_back( L1 ); data.indices.push_back( R1 );
            data.indices.push_back( L0 ); data.indices.push_back( R1 ); data.indices.push_back( R0 );
        }

        // ---- 2. Inner face (inward radial normal) ----
        uint32_t innerBase = static_cast<uint32_t>( data.vertices.size() );
        for( uint32_t i = 0; i <= sectors; ++i )
        {
            float t = -halfArc + i * arcStep;
            glm::vec4 nrm{ 0.0f, -cosf( t ), -sinf( t ), 0.0f };
            data.vertices.push_back( { innerPt( -hh, t ), nrm } );
            data.vertices.push_back( { innerPt( +hh, t ), nrm } );
        }
        for( uint32_t i = 0; i < sectors; ++i )
        {
            uint32_t L0 = innerBase + i * 2,           L1 = innerBase + ( i + 1 ) * 2;
            uint32_t R0 = innerBase + i * 2 + 1,       R1 = innerBase + ( i + 1 ) * 2 + 1;
            data.indices.push_back( L0 ); data.indices.push_back( R1 ); data.indices.push_back( L1 );
            data.indices.push_back( L0 ); data.indices.push_back( R0 ); data.indices.push_back( R1 );
        }

        // ---- 3. Left axial cap (x = -hh, normal -X = vessel axial) ----
        uint32_t leftCapBase = static_cast<uint32_t>( data.vertices.size() );
        for( uint32_t i = 0; i <= sectors; ++i )
        {
            float t = -halfArc + i * arcStep;
            glm::vec4 nrm{ -1.0f, 0.0f, 0.0f, 0.0f };
            data.vertices.push_back( { outerPt( -hh, t ), nrm } );  // outer
            data.vertices.push_back( { innerPt( -hh, t ), nrm } );  // inner
        }
        for( uint32_t i = 0; i < sectors; ++i )
        {
            uint32_t o0 = leftCapBase + i * 2,         o1 = leftCapBase + ( i + 1 ) * 2;
            uint32_t n0 = leftCapBase + i * 2 + 1,     n1 = leftCapBase + ( i + 1 ) * 2 + 1;
            data.indices.push_back( o0 ); data.indices.push_back( n0 ); data.indices.push_back( n1 );
            data.indices.push_back( o0 ); data.indices.push_back( n1 ); data.indices.push_back( o1 );
        }

        // ---- 4. Right axial cap (x = +hh, normal +X) ----
        uint32_t rightCapBase = static_cast<uint32_t>( data.vertices.size() );
        for( uint32_t i = 0; i <= sectors; ++i )
        {
            float t = -halfArc + i * arcStep;
            glm::vec4 nrm{ 1.0f, 0.0f, 0.0f, 0.0f };
            data.vertices.push_back( { outerPt( +hh, t ), nrm } );
            data.vertices.push_back( { innerPt( +hh, t ), nrm } );
        }
        for( uint32_t i = 0; i < sectors; ++i )
        {
            uint32_t o0 = rightCapBase + i * 2,         o1 = rightCapBase + ( i + 1 ) * 2;
            uint32_t n0 = rightCapBase + i * 2 + 1,     n1 = rightCapBase + ( i + 1 ) * 2 + 1;
            data.indices.push_back( o0 ); data.indices.push_back( n1 ); data.indices.push_back( n0 );
            data.indices.push_back( o0 ); data.indices.push_back( o1 ); data.indices.push_back( n1 );
        }

        // ---- 5. Arc-end edge at θ = +halfArc (normal = tangent = (0, -sin(h), cos(h))) ----
        {
            float     t   = +halfArc;
            glm::vec4 nrm = { 0.0f, -sinf( halfArc ), cosf( halfArc ), 0.0f };
            uint32_t  ab  = static_cast<uint32_t>( data.vertices.size() );
            data.vertices.push_back( { outerPt( -hh, t ), nrm } );  // 0: outer L
            data.vertices.push_back( { outerPt( +hh, t ), nrm } );  // 1: outer R
            data.vertices.push_back( { innerPt( -hh, t ), nrm } );  // 2: inner L
            data.vertices.push_back( { innerPt( +hh, t ), nrm } );  // 3: inner R
            data.indices.push_back( ab );     data.indices.push_back( ab + 2 ); data.indices.push_back( ab + 1 );
            data.indices.push_back( ab + 2 ); data.indices.push_back( ab + 3 ); data.indices.push_back( ab + 1 );
        }

        // ---- 6. Arc-end edge at θ = -halfArc (normal = -tangent = (0, -sin(h), -cos(h))) ----
        {
            float     t   = -halfArc;
            glm::vec4 nrm = { 0.0f, -sinf( halfArc ), -cosf( halfArc ), 0.0f };
            uint32_t  ab  = static_cast<uint32_t>( data.vertices.size() );
            data.vertices.push_back( { outerPt( -hh, t ), nrm } );  // 0: outer L
            data.vertices.push_back( { outerPt( +hh, t ), nrm } );  // 1: outer R
            data.vertices.push_back( { innerPt( -hh, t ), nrm } );  // 2: inner L
            data.vertices.push_back( { innerPt( +hh, t ), nrm } );  // 3: inner R
            data.indices.push_back( ab );     data.indices.push_back( ab + 1 ); data.indices.push_back( ab + 2 );
            data.indices.push_back( ab + 2 ); data.indices.push_back( ab + 1 ); data.indices.push_back( ab + 3 );
        }

        // Contact hull: 8 points — 4 corners + 4 edge midpoints, on the mid-radius surface.
        // X = axial (±hh), arc angle = ±halfArc or 0, Y/Z from mid-radius at that angle.
        // Sub-sphere radius = thickness/2.
        {
            const float subR = ( R_out - R_in ) * 0.5f; // = thickness/2
            // 4 corners
            for( float axial : { -hh, +hh } )
                for( float t : { -halfArc, +halfArc } )
                    data.contactHull.push_back( { axial,
                                                  R_mid * cosf( t ) - R_mid,
                                                  R_mid * sinf( t ),
                                                  subR } );
            // 4 edge midpoints: axial midpoints at each arc-end, arc-center at each axial end
            data.contactHull.push_back( { 0.0f, R_mid * cosf( -halfArc ) - R_mid, R_mid * sinf( -halfArc ), subR } );
            data.contactHull.push_back( { 0.0f, R_mid * cosf( +halfArc ) - R_mid, R_mid * sinf( +halfArc ), subR } );
            data.contactHull.push_back( {  -hh, R_mid * cosf( 0.0f )    - R_mid, R_mid * sinf( 0.0f ),     subR } );
            data.contactHull.push_back( {  +hh, R_mid * cosf( 0.0f )    - R_mid, R_mid * sinf( 0.0f ),     subR } );
            data.hullExtentZ = R_mid * sinf( halfArc ); // circumferential half-width
            data.hullExtentY = subR;                     // thickness / 2
        }

        return data;
    }

    MorphologyData MorphologyGenerator::CreateEllipsoid( float radiusXZ, float radiusY,
                                                          uint32_t sectors, uint32_t stacks )
    {
        MorphologyData data;
        float sectorStep = 2.0f * glm::pi<float>() / sectors;
        float stackStep  = glm::pi<float>() / stacks;
        float invXZ2     = 1.0f / ( radiusXZ * radiusXZ );
        float invY2      = 1.0f / ( radiusY * radiusY );

        for( uint32_t i = 0; i <= stacks; ++i )
        {
            float phi = glm::pi<float>() / 2.0f - i * stackStep; // latitude: pi/2 (north) to -pi/2 (south)
            float cp  = cosf( phi );
            float sp  = sinf( phi );

            for( uint32_t j = 0; j <= sectors; ++j )
            {
                float theta = j * sectorStep;
                float x     = radiusXZ * cp * cosf( theta );
                float y     = radiusY  * sp;
                float z     = radiusXZ * cp * sinf( theta );

                // Outward normal = gradient of the ellipsoid implicit function F = (x/a)^2 + (y/b)^2 + (z/a)^2 - 1
                glm::vec3 n = glm::normalize( glm::vec3( x * invXZ2, y * invY2, z * invXZ2 ) );

                data.vertices.push_back( { glm::vec4( x, y, z, 1.0f ), glm::vec4( n, 0.0f ) } );
            }
        }

        // Index generation — ellipsoid uses Y as the height axis (sphere uses Z), so the
        // vertex winding is mirrored relative to CreateSphere.  Swap the two non-pivot
        // indices in each triangle to restore outward-facing normals.
        for( uint32_t i = 0; i < stacks; ++i )
        {
            uint32_t k1 = i * ( sectors + 1 );
            uint32_t k2 = k1 + sectors + 1;

            for( uint32_t j = 0; j < sectors; ++j, ++k1, ++k2 )
            {
                if( i != 0 )
                {
                    data.indices.push_back( k1 );
                    data.indices.push_back( k1 + 1 );
                    data.indices.push_back( k2 );
                }
                if( i != ( stacks - 1 ) )
                {
                    data.indices.push_back( k1 + 1 );
                    data.indices.push_back( k2 + 1 );
                    data.indices.push_back( k2 );
                }
            }
        }

        // Contact hull: 8 points — 2 poles + 4 equatorial + 2 mid-latitude.
        // Sub-sphere radius = min(rXZ, rY) / 3 so sub-spheres don't overlap at the surface.
        const float subR = std::min( radiusXZ, radiusY ) / 3.0f;
        const float c45  = 0.7071f; // cos(45 deg)
        const float s45  = 0.7071f; // sin(45 deg)
        data.contactHull = {
            { 0.0f,       +radiusY,           0.0f,            subR }, // north pole
            { 0.0f,       -radiusY,           0.0f,            subR }, // south pole
            { +radiusXZ,   0.0f,              0.0f,            subR }, // equator +X
            { -radiusXZ,   0.0f,              0.0f,            subR }, // equator -X
            { 0.0f,        0.0f,             +radiusXZ,        subR }, // equator +Z
            { 0.0f,        0.0f,             -radiusXZ,        subR }, // equator -Z
            { radiusXZ * c45,  radiusY * s45, 0.0f,            subR }, // 45 deg lat, 0 deg lon
            { 0.0f,        radiusY * s45,     radiusXZ * c45,  subR }, // 45 deg lat, 90 deg lon
        };

        return data;
    }
} // namespace DigitalTwin