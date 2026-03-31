#include "simulation/VesselTreeGenerator.h"

#include <algorithm>
#include <cmath>
#include <glm/gtc/matrix_transform.hpp>

namespace DigitalTwin
{
    // -------------------------------------------------------------------------
    // Static factory
    // -------------------------------------------------------------------------

    VesselTreeGenerator VesselTreeGenerator::BranchingTree()
    {
        return VesselTreeGenerator{};
    }

    // -------------------------------------------------------------------------
    // Builder setters
    // -------------------------------------------------------------------------

    VesselTreeGenerator& VesselTreeGenerator::SetOrigin( glm::vec3 v )
    {
        m_origin = v;
        return *this;
    }
    VesselTreeGenerator& VesselTreeGenerator::SetDirection( glm::vec3 v )
    {
        m_direction = glm::normalize( v );
        return *this;
    }
    VesselTreeGenerator& VesselTreeGenerator::SetLength( float v )
    {
        m_length = v;
        return *this;
    }
    VesselTreeGenerator& VesselTreeGenerator::SetCellSpacing( float v )
    {
        m_cellSpacing = v;
        return *this;
    }
    VesselTreeGenerator& VesselTreeGenerator::SetRingSize( uint32_t v )
    {
        m_ringSize = v;
        return *this;
    }
    VesselTreeGenerator& VesselTreeGenerator::SetTubeRadius( float v )
    {
        m_tubeRadius = v;
        return *this;
    }
    VesselTreeGenerator& VesselTreeGenerator::SetBranchingAngle( float v )
    {
        m_branchingAngle = v;
        return *this;
    }
    VesselTreeGenerator& VesselTreeGenerator::SetBranchingDepth( uint32_t v )
    {
        m_branchingDepth = v;
        return *this;
    }
    VesselTreeGenerator& VesselTreeGenerator::SetLengthFalloff( float v )
    {
        m_lengthFalloff = v;
        return *this;
    }
    VesselTreeGenerator& VesselTreeGenerator::SetAngleJitter( float v )
    {
        m_angleJitter = v;
        return *this;
    }
    VesselTreeGenerator& VesselTreeGenerator::SetBranchProbability( float v )
    {
        m_branchProb = v;
        return *this;
    }
    VesselTreeGenerator& VesselTreeGenerator::SetSeed( uint32_t v )
    {
        m_seed = v;
        return *this;
    }
    VesselTreeGenerator& VesselTreeGenerator::SetTubeRadiusFalloff( float v )
    {
        m_tubeRadiusFalloff = v;
        return *this;
    }
    VesselTreeGenerator& VesselTreeGenerator::SetCellWidth( float v )
    {
        m_cellWidth = v;
        return *this;
    }
    VesselTreeGenerator& VesselTreeGenerator::SetCurvature( float v )
    {
        m_curvature = v;
        return *this;
    }
    VesselTreeGenerator& VesselTreeGenerator::SetBranchTwist( float v )
    {
        m_branchTwist = v;
        return *this;
    }

    // -------------------------------------------------------------------------
    // Build
    // -------------------------------------------------------------------------

    VesselTreeResult VesselTreeGenerator::Build()
    {
        m_rng.seed( m_seed );

        // Auto-derive cell width from trunk geometry so SetRingSize() keeps working as before.
        if( m_cellWidth <= 0.0f )
            m_cellWidth = 2.0f * glm::pi<float>() * m_tubeRadius / static_cast<float>( m_ringSize );

        VesselTreeResult result;

        BranchJob trunk{};
        trunk.origin     = m_origin;
        trunk.direction  = glm::normalize( m_direction );
        trunk.perp1      = perp1From( trunk.direction );
        trunk.length     = m_length;
        trunk.tubeRadius = m_tubeRadius;
        trunk.depth      = m_branchingDepth;

        buildBranch( trunk, result );

        result.totalCells = static_cast<uint32_t>( result.positions.size() );
        return result;
    }

    // -------------------------------------------------------------------------
    // Branch builder
    // -------------------------------------------------------------------------

    void VesselTreeGenerator::buildBranch( BranchJob job, VesselTreeResult& result )
    {
        const glm::vec3 dir = glm::normalize( job.direction );
        const glm::vec3 p1  = job.perp1;
        const glm::vec3 p2  = glm::normalize( glm::cross( dir, p1 ) );

        // Adaptive ring size: keep cells biologically uniform in size, vary count per ring.
        const uint32_t ringSize = std::max( 3u, static_cast<uint32_t>( std::round( 2.0f * glm::pi<float>() * job.tubeRadius / m_cellWidth ) ) );
        const uint32_t numRings = std::max( 1u, static_cast<uint32_t>( job.length / m_cellSpacing ) + 1 );
        const float    dAngle   = 2.0f * glm::pi<float>() / static_cast<float>( ringSize );

        const uint32_t cellBase = static_cast<uint32_t>( result.positions.size() );

        // ---- Quadratic Bezier: P0=origin, P2=endpoint, P1=midpoint+random offset ----
        const glm::vec3 P0 = job.origin;
        const glm::vec3 P2 = job.origin + job.length * dir;
        glm::vec3       P1 = ( P0 + P2 ) * 0.5f;
        if( m_curvature > 0.0f )
        {
            std::uniform_real_distribution<float> curvDist( -1.0f, 1.0f );
            float                                 offsetMag = m_curvature * job.length;
            P1 += ( curvDist( m_rng ) * p1 + curvDist( m_rng ) * p2 ) * offsetMag;
        }

        // Initial tangent for parallel transport (at t=0)
        glm::vec3 rawTangent0  = 2.0f * ( P1 - P0 );
        glm::vec3 prevTangent  = ( glm::length( rawTangent0 ) > 1e-4f ) ? glm::normalize( rawTangent0 ) : dir;
        glm::vec3 currentPerp1 = p1;
        glm::vec3 endTangent   = dir;
        const float twistPerRing = glm::radians( m_branchTwist );

        // ---- Place cells ----
        for( uint32_t r = 0; r < numRings; ++r )
        {
            float t  = ( numRings > 1 ) ? static_cast<float>( r ) / static_cast<float>( numRings - 1 ) : 0.0f;
            float mt = 1.0f - t;

            glm::vec3 ringCenter = mt * mt * P0 + 2.0f * mt * t * P1 + t * t * P2;
            glm::vec3 rawTangent = 2.0f * mt * ( P1 - P0 ) + 2.0f * t * ( P2 - P1 );
            glm::vec3 tangent    = ( glm::length( rawTangent ) > 1e-4f ) ? glm::normalize( rawTangent ) : dir;

            // Parallel transport: rotate perp frame from prevTangent → tangent (Rodrigues)
            if( r > 0 )
                currentPerp1 = parallelTransport( prevTangent, tangent, currentPerp1 );

            // Re-orthogonalize to prevent error accumulation
            glm::vec3 curP2 = glm::normalize( glm::cross( tangent, currentPerp1 ) );
            currentPerp1    = glm::normalize( glm::cross( curP2, tangent ) );

            // Accumulated axial twist offset for this ring
            float twistOffset = static_cast<float>( r ) * twistPerRing;

            for( uint32_t j = 0; j < ringSize; ++j )
            {
                float     angle = static_cast<float>( j ) * dAngle + twistOffset;
                float     c = cosf( angle ), s = sinf( angle );
                glm::vec3 pos    = ringCenter + job.tubeRadius * ( c * currentPerp1 + s * curP2 );
                glm::vec3 normal = glm::normalize( c * currentPerp1 + s * curP2 );
                result.positions.push_back( glm::vec4( pos, 1.0f ) );
                result.normals.push_back( glm::vec4( normal, 0.0f ) );
            }

            prevTangent = tangent;
            endTangent  = tangent;
        }

        // One segment count entry per branch
        result.segmentCounts.push_back( numRings * ringSize );

        // ---- Junction edges: parent last ring → this branch's first ring ----
        // When ring sizes differ, connect each child cell to the angularly nearest parent cell.
        if( !job.parentLastRing.empty() )
        {
            const uint32_t parentRS     = job.parentRingSize;
            const float    parentDAngle = 2.0f * glm::pi<float>() / static_cast<float>( parentRS );
            const float    childDAngle  = dAngle;
            for( uint32_t cj = 0; cj < ringSize; ++cj )
            {
                float    childAngle = static_cast<float>( cj ) * childDAngle;
                uint32_t bestParent = 0;
                float    bestDist   = 1e9f;
                for( uint32_t pj = 0; pj < parentRS; ++pj )
                {
                    float diff = std::abs( childAngle - static_cast<float>( pj ) * parentDAngle );
                    diff       = std::min( diff, 2.0f * glm::pi<float>() - diff ); // wrap-around
                    if( diff < bestDist )
                    {
                        bestDist   = diff;
                        bestParent = pj;
                    }
                }
                result.edges.push_back( { job.parentLastRing[ bestParent ], cellBase + cj } );
                result.edgeFlags.push_back( 0x4u ); // EDGE_JUNCTION
            }
        }

        // ---- Circumferential edges (closed loop per ring) ----
        for( uint32_t r = 0; r < numRings; ++r )
        {
            uint32_t base = cellBase + r * ringSize;
            for( uint32_t j = 0; j < ringSize; ++j )
            {
                result.edges.push_back( { base + j, base + ( j + 1 ) % ringSize } );
                result.edgeFlags.push_back( 0x1u ); // EDGE_RING
            }
        }

        // ---- Axial edges (between adjacent rings) ----
        for( uint32_t r = 0; r + 1 < numRings; ++r )
        {
            uint32_t b0 = cellBase + r * ringSize;
            uint32_t b1 = cellBase + ( r + 1 ) * ringSize;
            for( uint32_t j = 0; j < ringSize; ++j )
            {
                result.edges.push_back( { b0 + j, b1 + j } );
                result.edgeFlags.push_back( 0x2u ); // EDGE_AXIAL
            }
        }

        // ---- Recurse into children ----
        if( job.depth == 0 )
            return;

        std::uniform_real_distribution<float> jitterDist( -m_angleJitter, m_angleJitter );
        std::uniform_real_distribution<float> planeDist( 0.0f, glm::pi<float>() * 2.0f );
        std::uniform_real_distribution<float> lenVarDist( 0.9f, 1.1f );

        // Collect last ring indices
        std::vector<uint32_t> lastRing( ringSize );
        for( uint32_t j = 0; j < ringSize; ++j )
            lastRing[ j ] = cellBase + ( numRings - 1 ) * ringSize + j;

        // Random split plane using the transported end-of-branch perp frame
        glm::vec3 endPerp2   = glm::normalize( glm::cross( endTangent, currentPerp1 ) );
        float     planeAngle = planeDist( m_rng );
        glm::vec3 p2rand     = glm::normalize( cosf( planeAngle ) * currentPerp1 + sinf( planeAngle ) * endPerp2 );
        glm::vec3 splitAxis  = glm::normalize( glm::cross( endTangent, p2rand ) );

        // Two child directions (symmetric split + independent jitter)
        auto rotDir = [ & ]( float angleDeg ) -> glm::vec3 {
            float rad = glm::radians( angleDeg );
            return glm::normalize( glm::vec3( glm::rotate( glm::mat4( 1.0f ), rad, splitAxis ) * glm::vec4( endTangent, 0.0f ) ) );
        };

        float angle1    = m_branchingAngle + jitterDist( m_rng );
        float angle2    = m_branchingAngle + jitterDist( m_rng );
        float childLen1 = job.length * m_lengthFalloff * lenVarDist( m_rng );
        float childLen2 = job.length * m_lengthFalloff * lenVarDist( m_rng );

        glm::vec3 dir1 = rotDir( +angle1 );
        glm::vec3 dir2 = rotDir( -angle2 );

        // Child origin: exact Bezier endpoint (B(1) = P2)
        const glm::vec3 childOrigin = P2;

        BranchJob   child1{};
        const float childRadius = job.tubeRadius * m_tubeRadiusFalloff;

        child1.origin         = childOrigin + m_cellSpacing * dir1; // first ring offset one step from junction
        child1.direction      = dir1;
        child1.perp1          = perp1From( dir1, currentPerp1 ); // use transported perp as hint
        child1.length         = childLen1;
        child1.tubeRadius     = childRadius;
        child1.depth          = job.depth - 1;
        child1.parentRingSize = ringSize;
        child1.parentLastRing = lastRing;

        BranchJob child2{};
        child2.origin         = childOrigin + m_cellSpacing * dir2;
        child2.direction      = dir2;
        child2.perp1          = perp1From( dir2, currentPerp1 );
        child2.length         = childLen2;
        child2.tubeRadius     = childRadius;
        child2.depth          = job.depth - 1;
        child2.parentRingSize = ringSize;
        child2.parentLastRing = lastRing;

        buildBranch( child1, result );
        buildBranch( child2, result );
    }

    // -------------------------------------------------------------------------
    // Helpers
    // -------------------------------------------------------------------------

    glm::vec3 VesselTreeGenerator::parallelTransport( glm::vec3 t0, glm::vec3 t1, glm::vec3 perp )
    {
        glm::vec3 axis = glm::cross( t0, t1 );
        float     sinA = glm::length( axis );
        float     cosA = glm::dot( t0, t1 );
        if( sinA < 1e-6f )
            return perp; // tangents nearly parallel — no rotation needed
        axis /= sinA;
        // Rodrigues: p' = p*cosA + cross(axis, p)*sinA + axis*dot(axis, p)*(1-cosA)
        return perp * cosA + glm::cross( axis, perp ) * sinA + axis * glm::dot( axis, perp ) * ( 1.0f - cosA );
    }

    glm::vec3 VesselTreeGenerator::perp1From( glm::vec3 dir, glm::vec3 hint )
    {
        // If a valid hint is given, project it onto the plane perpendicular to dir
        if( glm::dot( hint, hint ) > 0.001f )
        {
            glm::vec3 projected = hint - glm::dot( hint, dir ) * dir;
            if( glm::dot( projected, projected ) > 1e-6f )
                return glm::normalize( projected );
        }
        // Fallback: pick an axis not collinear with dir
        glm::vec3 up = ( fabsf( dir.y ) < 0.9f ) ? glm::vec3( 0.0f, 1.0f, 0.0f ) : glm::vec3( 1.0f, 0.0f, 0.0f );
        return glm::normalize( glm::cross( dir, up ) );
    }

} // namespace DigitalTwin
