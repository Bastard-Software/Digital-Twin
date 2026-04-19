#include "simulation/VesselTreeGenerator.h"

#include <algorithm>
#include <cmath>
#include <glm/gtc/constants.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

namespace DigitalTwin
{
    namespace
    {
        // Phase 2.5: flag the K cells on a ring whose radial direction most closely aligns
        // with `worldInwardDir`. For a child branch, `worldInwardDir` points toward the
        // sibling branch — these are the inside-facing cells of the Y-junction apex
        // (Chiu & Chien 2011 cobblestone biology).
        void FlagCarinaCellsFacingDirection(
            VesselTreeResult& result,
            uint32_t          ringStart,
            uint32_t          ringSize,
            glm::vec3         ringCenter,
            glm::vec3         worldInwardDir,
            uint32_t          k = 2u )
        {
            if( ringSize == 0 || glm::dot( worldInwardDir, worldInwardDir ) < 1e-8f ) return;

            const glm::vec3 inward = glm::normalize( worldInwardDir );
            std::vector<std::pair<float, uint32_t>> scores;
            scores.reserve( ringSize );
            for( uint32_t j = 0; j < ringSize; ++j )
            {
                glm::vec3 radial = glm::vec3( result.cells[ ringStart + j ].position ) - ringCenter;
                float     len    = glm::length( radial );
                if( len < 1e-6f ) continue;
                radial /= len;
                scores.emplace_back( glm::dot( radial, inward ), ringStart + j );
            }
            std::sort( scores.begin(), scores.end(),
                       []( const auto& a, const auto& b ) { return a.first > b.first; } );
            const uint32_t count = std::min( k, static_cast<uint32_t>( scores.size() ) );
            for( uint32_t i = 0; i < count; ++i )
                result.cells[ scores[ i ].second ].isCarina = 1u;
        }

        // Phase 2.5: flag the K cells on a ring closest to a plane (minimum |dot(radial,
        // planeNormal)|). For the parent-last-ring at a Y-junction, the bisection plane
        // contains the parent's tangent and the split axis; cells with minimum projection
        // onto its normal sit ON the flow-divider line biologically.
        void FlagCarinaCellsOnPlane(
            VesselTreeResult& result,
            uint32_t          ringStart,
            uint32_t          ringSize,
            glm::vec3         ringCenter,
            glm::vec3         planeNormal,
            uint32_t          k = 2u )
        {
            if( ringSize == 0 || glm::dot( planeNormal, planeNormal ) < 1e-8f ) return;

            const glm::vec3 normal = glm::normalize( planeNormal );
            std::vector<std::pair<float, uint32_t>> scores;
            scores.reserve( ringSize );
            for( uint32_t j = 0; j < ringSize; ++j )
            {
                glm::vec3 radial = glm::vec3( result.cells[ ringStart + j ].position ) - ringCenter;
                float     len    = glm::length( radial );
                if( len < 1e-6f ) continue;
                radial /= len;
                scores.emplace_back( std::fabs( glm::dot( radial, normal ) ), ringStart + j );
            }
            std::sort( scores.begin(), scores.end(),
                       []( const auto& a, const auto& b ) { return a.first < b.first; } );
            const uint32_t count = std::min( k, static_cast<uint32_t>( scores.size() ) );
            for( uint32_t i = 0; i < count; ++i )
                result.cells[ scores[ i ].second ].isCarina = 1u;
        }
    } // namespace

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
    VesselTreeGenerator& VesselTreeGenerator::SetTubeRadiusEnd( float v )          { m_tubeRadiusEnd = v; return *this; }
    VesselTreeGenerator& VesselTreeGenerator::SetStoneWalesAtTaperTransitions( bool v ) { m_stoneWalesAtTaperTransitions = v; return *this; }

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
        trunk.endTubeRadius   = m_tubeRadiusEnd; // Phase 2.4: trunk may taper
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

        // Phase 2.4: radius interpolates linearly along the branch when endTubeRadius > 0.
        // Per-ring radius drives per-ring cell count: `ring_r = max(2, round(2π·r_r / ECWidth))`.
        // Dual-seam minimum (Bär 1984).
        const float    startRadius = job.tubeRadius;
        const float    endRadius   = ( job.endTubeRadius > 0.0f ) ? job.endTubeRadius : job.tubeRadius;
        const float    axialStep   = m_ecCircWidth * m_cellAspect;
        const uint32_t numRings    = std::max( 1u, static_cast<uint32_t>( job.length / axialStep ) + 1u );

        auto ringRadiusAt = [&]( uint32_t r ) {
            if( numRings <= 1 ) return startRadius;
            float t = static_cast<float>( r ) / static_cast<float>( numRings - 1 );
            return startRadius * ( 1.0f - t ) + endRadius * t;
        };
        auto ringSizeFor = [&]( float radius ) {
            return std::max( 2u,
                static_cast<uint32_t>( std::round( 2.0f * glm::pi<float>() * radius / m_ecCircWidth ) ) );
        };

        // Precompute per-ring cell counts so transitions are visible before placing anything.
        std::vector<uint32_t> ringSizes( numRings );
        for( uint32_t r = 0; r < numRings; ++r )
            ringSizes[ r ] = ringSizeFor( ringRadiusAt( r ) );

        // Assign morphology indices per (ring, cellInRing) position.  Default 0 = elongated
        // rhomboid (Phase 2.3); at each ring-count transition, Stone-Wales 5/7 defect pairs
        // replace a symmetric subset of cells (Stone & Wales 1986; Iijima 1991).
        //   Tapering  (ringSizes[r] > ringSizes[r+1]):  ΔN heptagons on ring r (wider / parent
        //                                               side), ΔN pentagons on ring r+1
        //                                               (narrower / child side).
        //   Widening  (reversed):                       pentagons on parent, heptagons on child.
        // Morphology indices: 0 = rhomboid, 1 = pentagon, 2 = heptagon — matches the order
        // PuzzlePiecePaletteDemo registers and the morphologyIndex convention in Phenotype.h.
        std::vector<std::vector<uint32_t>> cellMorph( numRings );
        for( uint32_t r = 0; r < numRings; ++r )
            cellMorph[ r ].assign( ringSizes[ r ], 0u );

        // Symmetric distribution of `count` defects around a ring of `ringSize` cells,
        // skipping positions already occupied by prior transitions. A middle ring may be
        // both the child of one transition (pentagons already placed) and the parent of
        // the next (heptagons to place); skipping preserves the pentagon-count ==
        // heptagon-count invariant across a tapered chain.
        auto distributeDefectIndices = [&]( uint32_t ringSize, uint32_t count,
                                            const std::vector<uint32_t>& existing,
                                            std::vector<uint32_t>& out ) {
            out.clear();
            if( count == 0 || ringSize == 0 ) return;
            count = std::min( count, ringSize );
            for( uint32_t k = 0; k < count; ++k )
            {
                uint32_t idx = static_cast<uint32_t>( std::round(
                    static_cast<float>( k ) * static_cast<float>( ringSize ) / static_cast<float>( count ) ) );
                if( idx >= ringSize ) idx = ringSize - 1u;
                uint32_t tries = 0;
                while( tries < ringSize &&
                       ( existing[ idx ] != 0u ||
                         std::find( out.begin(), out.end(), idx ) != out.end() ) )
                {
                    idx = ( idx + 1u ) % ringSize;
                    ++tries;
                }
                out.push_back( idx );
            }
        };
        std::vector<uint32_t> defectIdxParent, defectIdxChild;

        // Phase 2.4.5: continuous-taper defect insertion is opt-in. Default off — the rhombus
        // tiles produced by the demos look more biological when no oversized 5/7 polygons
        // intrude on smooth tapering. Phase 2.5 will flip this path to fire at bifurcation
        // carinas instead, where defects are genuinely topological.
        for( uint32_t r = 0; r + 1 < numRings && m_stoneWalesAtTaperTransitions; ++r )
        {
            if( ringSizes[ r ] == ringSizes[ r + 1 ] ) continue;

            const uint32_t nP = ringSizes[ r ];
            const uint32_t nC = ringSizes[ r + 1 ];
            const uint32_t dNRaw   = ( nP > nC ) ? ( nP - nC ) : ( nC - nP );
            const bool     tapering = nP > nC;
            // Cap defect count at the narrower ring's size so pentagon count == heptagon
            // count is preserved (each 5/7 pair is topologically neutral; the invariant
            // requires symmetric counts on both sides of the transition). When ΔN exceeds
            // `min(nP, nC)` the excess ring-count delta cannot be resolved as 5/7 pairs at
            // a single transition — finer-grained tapers or higher-order defects would be
            // required. Phase 2.4 accepts the cap as a documented topological limit.
            const uint32_t dN = std::min( { dNRaw, nP, nC } );

            distributeDefectIndices( nP, dN, cellMorph[ r ],     defectIdxParent );
            distributeDefectIndices( nC, dN, cellMorph[ r + 1 ], defectIdxChild );

            for( uint32_t i : defectIdxParent )
                cellMorph[ r ][ i ] = tapering ? 2u : 1u;     // wider side: heptagon (tapering) / pentagon (widening)
            for( uint32_t i : defectIdxChild )
                cellMorph[ r + 1 ][ i ] = tapering ? 1u : 2u; // narrower side opposite
        }

        // Staggered brick pattern: alternate rings circumferentially offset by half a cell-width
        // (Davies 2009 — brick-pattern interlocks avoid longitudinal-seam mechanical instability).
        // dAngle varies per ring with taper; stagger uses the LOCAL ring's dAngle.
        auto ringDAngle = [&]( uint32_t r ) {
            return 2.0f * glm::pi<float>() / static_cast<float>( ringSizes[ r ] );
        };

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

        // Phase 2.5: remember per-ring start index + ring centre so we can flag carina
        // cells after the RNG-consuming split computation runs (preserving the Phase 2.3
        // RNG sequence — Reproducible_SameSeed).
        std::vector<uint32_t>  ringStartIdx( numRings );
        std::vector<glm::vec3> ringCenters( numRings );

        // Tiny positional jitter (~1% of cell spacing) breaks the perfect mathematical
        // ring symmetry that would otherwise trigger a coherent radial-breathing mode
        // under the JKR stiffness — all cells equally off-equilibrium in the same
        // direction cannot settle independently and instead oscillate in phase.
        // Biologically defensible: real endothelium is not geometrically exact.
        std::uniform_real_distribution<float> placeJitter( -0.01f, 0.01f );

        // ---- Place cells ring by ring ----
        for( uint32_t r = 0; r < numRings; ++r )
        {
            ringStartIdx[ r ] = static_cast<uint32_t>( result.cells.size() );

            float t  = ( numRings > 1 ) ? static_cast<float>( r ) / static_cast<float>( numRings - 1 ) : 0.0f;
            float mt = 1.0f - t;

            glm::vec3 ringCenter = mt * mt * P0 + 2.0f * mt * t * P1 + t * t * P2;
            ringCenters[ r ]     = ringCenter;
            glm::vec3 rawTangent = 2.0f * mt * ( P1 - P0 ) + 2.0f * t * ( P2 - P1 );
            glm::vec3 tangent    = ( glm::length( rawTangent ) > 1e-4f ) ? glm::normalize( rawTangent ) : dir;

            if( r > 0 )
                currentPerp1 = parallelTransport( prevTangent, tangent, currentPerp1 );

            glm::vec3 curP2 = glm::normalize( glm::cross( tangent, currentPerp1 ) );
            currentPerp1    = glm::normalize( glm::cross( curP2, tangent ) );

            const uint32_t ringSize      = ringSizes[ r ];
            const float    localRadius   = ringRadiusAt( r );
            const float    dAngle        = ringDAngle( r );
            const float    twistOffset   = static_cast<float>( r ) * twistPerRing;
            const float    staggerOffset = job.ringAnglePhase + ( ( r & 1u ) ? dAngle * 0.5f : 0.0f );

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
                float jitterR = placeJitter( m_rng ) * localRadius;
                float jitterC = placeJitter( m_rng ) * m_ecCircWidth;
                glm::vec3 pos = ringCenter
                              + ( localRadius + jitterR ) * radial
                              + jitterC * circum;

                // Orientation quaternion: maps local (X,Y,Z) → world (axial, radial, circum).
                // Local +Y is the outward face normal for rhomboid / pentagon / heptagon tiles;
                // local +X is the axial flow direction. Shared across all morphology variants
                // so the orientation pipeline stays morphology-agnostic (Phase 2.1).
                glm::mat3 basis( tangent, radial, circum );
                glm::quat q = glm::quat_cast( basis );

                VesselCellSpec cell{};
                cell.position        = glm::vec4( pos, 1.0f );
                cell.orientation     = glm::vec4( q.x, q.y, q.z, q.w );
                cell.polaritySeed    = glm::vec4( radial, 1.0f );       // basal-outward, full magnitude
                cell.morphologyIndex = cellMorph[ r ][ j ];             // Phase 2.4: 0/1/2 = rhomboid/pentagon/heptagon
                result.cells.push_back( cell );
            }

            prevTangent = tangent;
            endTangent  = tangent;
        }

        // Phase 2.5: if this branch was spawned by a bifurcation, its first-ring cells
        // nearest to the sibling branch are the child-side carinas. Trunk branches have
        // `carinaInwardDir == 0` and are skipped.
        FlagCarinaCellsFacingDirection(
            result, ringStartIdx[ 0 ], ringSizes[ 0 ], ringCenters[ 0 ], job.carinaInwardDir );

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
        const glm::vec3 childOrigin = P2;

        // Phase 2.5: flag parent-last-ring cells on the bisection plane. The plane
        // contains the parent's end tangent and the split axis; its normal is `p2rand`
        // (the in-plane direction along which dir1/dir2 separate). Cells with the
        // smallest |dot(radial, p2rand)| sit ON the flow-divider line — the parent-side
        // carina cells (Chiu & Chien 2011).
        FlagCarinaCellsOnPlane(
            result,
            ringStartIdx[ numRings - 1u ],
            ringSizes[ numRings - 1u ],
            ringCenters[ numRings - 1u ],
            p2rand );

        // Phase 2.5: per-child tapering propagation. Murray's law (m_tubeRadiusFalloff,
        // default 0.79 per Murray 1926 DOI 10.1073/pnas.12.3.207) applies AT the bifurcation
        // as a discrete radius drop; within each child the parent's proportional taper is
        // preserved so a 3-level tree can go artery → arteriole → capillary continuously
        // across bifurcations. Trunk-only behaviour (no taper) is preserved when
        // `job.endTubeRadius < 0` — children also stay constant-radius.
        const float internalTaperRatio = ( startRadius > 1e-6f ) ? ( endRadius / startRadius ) : 1.0f;
        const float childStartRadius   = endRadius * m_tubeRadiusFalloff;
        const float childEndRadius     = ( job.endTubeRadius > 0.0f ) ? childStartRadius * internalTaperRatio : -1.0f;

        // Carry staggered-ring phase across the junction so children keep the brick pattern
        // on the first ring. Propagate stagger phase using the LAST ring's dAngle (may differ
        // from first under taper).
        const float lastRingDAngle = ringDAngle( numRings - 1u );
        const float childPhase     = job.ringAnglePhase + ( ( numRings & 1u ) ? lastRingDAngle * 0.5f : 0.0f );

        BranchJob child1{};
        child1.origin          = childOrigin + axialStep * dir1;
        child1.direction       = dir1;
        child1.perp1           = perp1From( dir1, currentPerp1 );
        child1.length          = childLen1;
        child1.tubeRadius      = childStartRadius;
        child1.endTubeRadius   = childEndRadius;
        child1.depth           = job.depth - 1;
        child1.ringAnglePhase  = childPhase;
        // Child1 is on the +p2rand side of the split (dir1 = rotate(endTangent, +angle, splitAxis));
        // its sibling child2 sits on the -p2rand side. So child1's carina-inward world direction
        // is -p2rand (pointing across the Y-junction apex toward child2).
        child1.carinaInwardDir = -p2rand;

        BranchJob child2{};
        child2.origin          = childOrigin + axialStep * dir2;
        child2.direction       = dir2;
        child2.perp1           = perp1From( dir2, currentPerp1 );
        child2.length          = childLen2;
        child2.tubeRadius      = childStartRadius;
        child2.endTubeRadius   = childEndRadius;
        child2.depth           = job.depth - 1;
        child2.ringAnglePhase  = childPhase;
        child2.carinaInwardDir = +p2rand;

        buildBranch( child1, result );
        buildBranch( child2, result );

        (void)branchFirstIdx; // reserved for future downstream indexing
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
