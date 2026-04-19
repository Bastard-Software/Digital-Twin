#include "simulation/VesselTreeGenerator.h"

#include <algorithm>
#include <cmath>
#include <glm/gtc/constants.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

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

    VesselTreeGenerator& VesselTreeGenerator::SetOrigin( glm::vec3 v )             { m_origin = v; return *this; }
    VesselTreeGenerator& VesselTreeGenerator::SetDirection( glm::vec3 v )          { m_direction = glm::normalize( v ); return *this; }
    VesselTreeGenerator& VesselTreeGenerator::SetLength( float v )                 { m_length = v; return *this; }
    VesselTreeGenerator& VesselTreeGenerator::SetTubeRadius( float v )             { m_tubeRadius = v; return *this; }
    VesselTreeGenerator& VesselTreeGenerator::SetBranchingAngle( float v )         { m_branchingAngle = v; return *this; }
    VesselTreeGenerator& VesselTreeGenerator::SetBranchingDepth( uint32_t v )      { m_branchingDepth = v; return *this; }
    VesselTreeGenerator& VesselTreeGenerator::SetLengthFalloff( float v )          { m_lengthFalloff = v; return *this; }
    VesselTreeGenerator& VesselTreeGenerator::SetAngleJitter( float v )            { m_angleJitter = v; return *this; }
    VesselTreeGenerator& VesselTreeGenerator::SetBranchProbability( float v )      { m_branchProb = v; return *this; }
    VesselTreeGenerator& VesselTreeGenerator::SetSeed( uint32_t v )                { m_seed = v; return *this; }
    VesselTreeGenerator& VesselTreeGenerator::SetTubeRadiusFalloff( float v )      { m_tubeRadiusFalloff = v; return *this; }
    VesselTreeGenerator& VesselTreeGenerator::SetCurvature( float v )              { m_curvature = v; return *this; }
    VesselTreeGenerator& VesselTreeGenerator::SetBranchTwist( float v )            { m_branchTwist = v; return *this; }
    VesselTreeGenerator& VesselTreeGenerator::SetECCircumferentialWidth( float v ) { m_ecCircWidth = std::max( 1e-4f, v ); return *this; }
    VesselTreeGenerator& VesselTreeGenerator::SetCellAspectRatio( float v )        { m_cellAspect = std::max( 1e-4f, v ); return *this; }

    // -------------------------------------------------------------------------
    // Build
    // -------------------------------------------------------------------------

    VesselTreeResult VesselTreeGenerator::Build()
    {
        m_rng.seed( m_seed );

        VesselTreeResult result;

        BranchJob trunk{};
        trunk.origin          = m_origin;
        trunk.direction       = glm::normalize( m_direction );
        trunk.perp1           = perp1From( trunk.direction );
        trunk.length          = m_length;
        trunk.tubeRadius      = m_tubeRadius;
        trunk.depth           = m_branchingDepth;
        trunk.ringAnglePhase  = 0.0f;

        buildBranch( trunk, result );

        result.totalCells = static_cast<uint32_t>( result.cells.size() );
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

        // Adaptive ring count: cells per ring = max(2, round(2π·r / ECWidth)).
        // Dual-seam minimum (Bär 1984); 1-cell autocellular capillary not modelled at point-agent scale.
        const uint32_t ringSize    = std::max( 2u,
            static_cast<uint32_t>( std::round( 2.0f * glm::pi<float>() * job.tubeRadius / m_ecCircWidth ) ) );
        const float    axialStep   = m_ecCircWidth * m_cellAspect; // rhomboid length along flow
        const uint32_t numRings    = std::max( 1u, static_cast<uint32_t>( job.length / axialStep ) + 1u );
        const float    dAngle      = 2.0f * glm::pi<float>() / static_cast<float>( ringSize );
        // Staggered brick pattern: alternate rings circumferentially offset by half a cell-width
        // (Davies 2009 — brick-pattern interlocks avoid longitudinal-seam mechanical instability).
        const float    staggerStep = dAngle * 0.5f;

        // ---- Quadratic Bezier centreline: P0 → P1 → P2 ----
        const glm::vec3 P0 = job.origin;
        const glm::vec3 P2 = job.origin + job.length * dir;
        glm::vec3       P1 = ( P0 + P2 ) * 0.5f;
        if( m_curvature > 0.0f )
        {
            std::uniform_real_distribution<float> curvDist( -1.0f, 1.0f );
            float                                 offsetMag = m_curvature * job.length;
            P1 += ( curvDist( m_rng ) * p1 + curvDist( m_rng ) * p2 ) * offsetMag;
        }

        glm::vec3   rawTangent0  = 2.0f * ( P1 - P0 );
        glm::vec3   prevTangent  = ( glm::length( rawTangent0 ) > 1e-4f ) ? glm::normalize( rawTangent0 ) : dir;
        glm::vec3   currentPerp1 = p1;
        glm::vec3   endTangent   = dir;
        const float twistPerRing = glm::radians( m_branchTwist );

        const uint32_t branchFirstIdx = static_cast<uint32_t>( result.cells.size() );

        // Tiny positional jitter (~1% of cell spacing) breaks the perfect mathematical
        // ring symmetry that would otherwise trigger a coherent radial-breathing mode
        // under the JKR stiffness — all cells equally off-equilibrium in the same
        // direction cannot settle independently and instead oscillate in phase.
        // Biologically defensible: real endothelium is not geometrically exact.
        std::uniform_real_distribution<float> placeJitter( -0.01f, 0.01f );

        // ---- Place cells ring by ring ----
        for( uint32_t r = 0; r < numRings; ++r )
        {
            float t  = ( numRings > 1 ) ? static_cast<float>( r ) / static_cast<float>( numRings - 1 ) : 0.0f;
            float mt = 1.0f - t;

            glm::vec3 ringCenter = mt * mt * P0 + 2.0f * mt * t * P1 + t * t * P2;
            glm::vec3 rawTangent = 2.0f * mt * ( P1 - P0 ) + 2.0f * t * ( P2 - P1 );
            glm::vec3 tangent    = ( glm::length( rawTangent ) > 1e-4f ) ? glm::normalize( rawTangent ) : dir;

            if( r > 0 )
                currentPerp1 = parallelTransport( prevTangent, tangent, currentPerp1 );

            glm::vec3 curP2 = glm::normalize( glm::cross( tangent, currentPerp1 ) );
            currentPerp1    = glm::normalize( glm::cross( curP2, tangent ) );

            const float twistOffset   = static_cast<float>( r ) * twistPerRing;
            const float staggerOffset = job.ringAnglePhase + ( ( r & 1u ) ? staggerStep : 0.0f );

            for( uint32_t j = 0; j < ringSize; ++j )
            {
                float     angle = static_cast<float>( j ) * dAngle + twistOffset + staggerOffset;
                float     c = cosf( angle ), s = sinf( angle );
                glm::vec3 radial = glm::normalize( c * currentPerp1 + s * curP2 );
                glm::vec3 circum = glm::normalize( glm::cross( tangent, radial ) );

                // Per-cell symmetry-breaking jitter on the (radial, circumferential) basis.
                // Axial jitter omitted — the coherent breathing mode is radial/circumferential
                // and axial jitter would disperse the rings, breaking tests that group cells
                // by exact axial position.
                float jitterR = placeJitter( m_rng ) * job.tubeRadius;
                float jitterC = placeJitter( m_rng ) * m_ecCircWidth;
                glm::vec3 pos = ringCenter
                              + ( job.tubeRadius + jitterR ) * radial
                              + jitterC * circum;

                // Orientation quaternion: maps local (X,Y,Z) → world (axial, radial, circum).
                // Local +Y is the outward face normal for both CurvedTile and ElongatedQuad;
                // local +X is the axial flow direction of the elongated rhomboid.
                glm::mat3 basis( tangent, radial, circum );
                glm::quat q = glm::quat_cast( basis );

                VesselCellSpec cell{};
                cell.position        = glm::vec4( pos, 1.0f );
                cell.orientation     = glm::vec4( q.x, q.y, q.z, q.w );
                cell.polaritySeed    = glm::vec4( radial, 1.0f );       // basal-outward, full magnitude
                cell.morphologyIndex = 0u;                              // Phase 2.3: all elongated rhomboid
                result.cells.push_back( cell );
            }

            prevTangent = tangent;
            endTangent  = tangent;
        }

        // ---- Recurse into children (branching retained; defect insertion comes in 2.4/2.5) ----
        if( job.depth == 0 )
            return;

        std::uniform_real_distribution<float> jitterDist( -m_angleJitter, m_angleJitter );
        std::uniform_real_distribution<float> planeDist( 0.0f, glm::pi<float>() * 2.0f );
        std::uniform_real_distribution<float> lenVarDist( 0.9f, 1.1f );
        std::uniform_real_distribution<float> probDist( 0.0f, 1.0f );

        if( probDist( m_rng ) > m_branchProb )
            return;

        glm::vec3 endPerp2   = glm::normalize( glm::cross( endTangent, currentPerp1 ) );
        float     planeAngle = planeDist( m_rng );
        glm::vec3 p2rand     = glm::normalize( cosf( planeAngle ) * currentPerp1 + sinf( planeAngle ) * endPerp2 );
        glm::vec3 splitAxis  = glm::normalize( glm::cross( endTangent, p2rand ) );

        auto rotDir = [ & ]( float angleDeg ) -> glm::vec3 {
            float rad = glm::radians( angleDeg );
            return glm::normalize( glm::vec3( glm::rotate( glm::mat4( 1.0f ), rad, splitAxis ) * glm::vec4( endTangent, 0.0f ) ) );
        };

        float     angle1      = m_branchingAngle + jitterDist( m_rng );
        float     angle2      = m_branchingAngle + jitterDist( m_rng );
        float     childLen1   = job.length * m_lengthFalloff * lenVarDist( m_rng );
        float     childLen2   = job.length * m_lengthFalloff * lenVarDist( m_rng );
        glm::vec3 dir1        = rotDir( +angle1 );
        glm::vec3 dir2        = rotDir( -angle2 );
        const float childRadius = job.tubeRadius * m_tubeRadiusFalloff;
        const glm::vec3 childOrigin = P2;

        // Carry staggered-ring phase across the junction so children keep the brick pattern
        // on the first ring (Phase 2.5 will retro-fit carina heptagons here).
        const float childPhase = job.ringAnglePhase + ( ( numRings & 1u ) ? staggerStep : 0.0f );

        BranchJob child1{};
        child1.origin          = childOrigin + axialStep * dir1;
        child1.direction       = dir1;
        child1.perp1           = perp1From( dir1, currentPerp1 );
        child1.length          = childLen1;
        child1.tubeRadius      = childRadius;
        child1.depth           = job.depth - 1;
        child1.ringAnglePhase  = childPhase;

        BranchJob child2{};
        child2.origin          = childOrigin + axialStep * dir2;
        child2.direction       = dir2;
        child2.perp1           = perp1From( dir2, currentPerp1 );
        child2.length          = childLen2;
        child2.tubeRadius      = childRadius;
        child2.depth           = job.depth - 1;
        child2.ringAnglePhase  = childPhase;

        buildBranch( child1, result );
        buildBranch( child2, result );

        (void)branchFirstIdx; // parent-cell tracking deferred to Phase 2.5 (carina heptagons)
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
            return perp;
        axis /= sinA;
        return perp * cosA + glm::cross( axis, perp ) * sinA + axis * glm::dot( axis, perp ) * ( 1.0f - cosA );
    }

    glm::vec3 VesselTreeGenerator::perp1From( glm::vec3 dir, glm::vec3 hint )
    {
        if( glm::dot( hint, hint ) > 0.001f )
        {
            glm::vec3 projected = hint - glm::dot( hint, dir ) * dir;
            if( glm::dot( projected, projected ) > 1e-6f )
                return glm::normalize( projected );
        }
        glm::vec3 up = ( fabsf( dir.y ) < 0.9f ) ? glm::vec3( 0.0f, 1.0f, 0.0f ) : glm::vec3( 1.0f, 0.0f, 0.0f );
        return glm::normalize( glm::cross( dir, up ) );
    }

} // namespace DigitalTwin
