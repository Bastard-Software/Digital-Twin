#include "resources/ShapeGenerator.hpp"

#include <cmath>
#include <glm/gtc/constants.hpp>

namespace DigitalTwin
{
    Mesh ShapeGenerator::CreateCube()
    {
        Mesh mesh;
        mesh.name = "Cube";

        // Pre-allocate memory to avoid reallocations.
        // A cube has 6 faces. Each face has 4 vertices.
        // Total vertices = 6 * 4 = 24.
        mesh.vertices.reserve( 24 );

        // Each face has 2 triangles (6 indices).
        // Total indices = 6 * 6 = 36.
        mesh.indices.reserve( 36 );

        // Helper to push a face (4 vertices + 2 triangles)
        auto addFace = [ & ]( glm::vec3 normal, glm::vec3 v0, glm::vec3 v1, glm::vec3 v2, glm::vec3 v3 ) {
            uint32_t baseIndex = ( uint32_t )mesh.vertices.size();

            // Vertices (using vec4(1.0) for white color default)
            mesh.vertices.push_back( { glm::vec4( v0, 1.0f ), glm::vec4( normal, 0.0f ), glm::vec4( 1.0f ) } );
            mesh.vertices.push_back( { glm::vec4( v1, 1.0f ), glm::vec4( normal, 0.0f ), glm::vec4( 1.0f ) } );
            mesh.vertices.push_back( { glm::vec4( v2, 1.0f ), glm::vec4( normal, 0.0f ), glm::vec4( 1.0f ) } );
            mesh.vertices.push_back( { glm::vec4( v3, 1.0f ), glm::vec4( normal, 0.0f ), glm::vec4( 1.0f ) } );

            // Indices (CCW Winding)
            mesh.indices.push_back( baseIndex + 0 );
            mesh.indices.push_back( baseIndex + 1 );
            mesh.indices.push_back( baseIndex + 2 );
            mesh.indices.push_back( baseIndex + 2 );
            mesh.indices.push_back( baseIndex + 3 );
            mesh.indices.push_back( baseIndex + 0 );
        };

        // Half-size for unit cube centered at 0
        float     s = 0.5f;
        glm::vec3 p0( -s, -s, s );
        glm::vec3 p1( s, -s, s );
        glm::vec3 p2( s, s, s );
        glm::vec3 p3( -s, s, s );
        glm::vec3 p4( -s, -s, -s );
        glm::vec3 p5( s, -s, -s );
        glm::vec3 p6( s, s, -s );
        glm::vec3 p7( -s, s, -s );

        // Faces: Front, Back, Right, Left, Top, Bottom
        addFace( { 0, 0, 1 }, p0, p1, p2, p3 );
        addFace( { 0, 0, -1 }, p5, p4, p7, p6 );
        addFace( { 1, 0, 0 }, p1, p5, p6, p2 );
        addFace( { -1, 0, 0 }, p4, p0, p3, p7 );
        addFace( { 0, 1, 0 }, p3, p2, p6, p7 );
        addFace( { 0, -1, 0 }, p4, p5, p1, p0 );

        return mesh;
    }

    Mesh ShapeGenerator::CreateSphere( float radius, int stacks, int slices )
    {
        Mesh mesh;
        mesh.name = "Sphere";

        // Calculate exact sizes for pre-allocation
        size_t vertexCount = ( stacks + 1 ) * ( slices + 1 );
        size_t indexCount  = stacks * slices * 6;

        mesh.vertices.reserve( vertexCount );
        mesh.indices.reserve( indexCount );

        // 1. Generate Vertices
        for( int i = 0; i <= stacks; ++i )
        {
            float phi = glm::pi<float>() * float( i ) / float( stacks ); // 0 to PI
            float y   = cos( phi );
            float r   = sin( phi );

            for( int j = 0; j <= slices; ++j )
            {
                float theta = 2.0f * glm::pi<float>() * float( j ) / float( slices ); // 0 to 2PI
                float x     = r * cos( theta );
                float z     = r * sin( theta );

                glm::vec3 normal( x, y, z );
                glm::vec3 pos = normal * radius;

                mesh.vertices.push_back( {
                    glm::vec4( pos, 1.0f ), glm::vec4( normal, 0.0f ),
                    glm::vec4( 1.0f ) // White Color
                } );
            }
        }

        // 2. Generate Indices
        for( int i = 0; i < stacks; ++i )
        {
            for( int j = 0; j < slices; ++j )
            {
                int first  = ( i * ( slices + 1 ) ) + j;
                int second = first + slices + 1;

                mesh.indices.push_back( first );
                mesh.indices.push_back( second );
                mesh.indices.push_back( first + 1 );

                mesh.indices.push_back( second );
                mesh.indices.push_back( second + 1 );
                mesh.indices.push_back( first + 1 );
            }
        }
        return mesh;
    }
} // namespace DigitalTwin