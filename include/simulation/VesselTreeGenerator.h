#pragma once
#include "core/Core.h"

#include <glm/glm.hpp>
#include <cstdint>
#include <random>
#include <utility>
#include <vector>

namespace DigitalTwin
{
    struct VesselTreeResult
    {
        std::vector<glm::vec4>                    positions;     // xyz=pos, w=1.0
        std::vector<glm::vec4>                    normals;       // per-cell outward radial normal (w=0)
        std::vector<std::pair<uint32_t,uint32_t>> edges;         // all edges: ring + axial + junction
        std::vector<uint32_t>                     segmentCounts; // one per branch = ringSize * numRings
        uint32_t                                  totalCells = 0;
    };

    /**
     * @brief CPU-side generator for branching vessel trees with 2D ring topology.
     *
     * Each branch is a tube of rings (ringSize cells per ring). Edges include:
     *   - Circumferential: consecutive cells within each ring (closed loop)
     *   - Axial:           corresponding cells between adjacent rings
     *   - Junction:        last ring of parent to first ring of each child branch
     *
     * Usage:
     *   auto tree = VesselTreeGenerator::BranchingTree()
     *       .SetOrigin({-15, 15, 0}).SetDirection({1,0,0}).SetLength(30.0f)
     *       .SetRingSize(6).SetBranchingDepth(2).SetSeed(42)
     *       .Build();
     */
    class DT_API VesselTreeGenerator
    {
    public:
        static VesselTreeGenerator BranchingTree();

        VesselTreeGenerator& SetOrigin( glm::vec3 origin );
        VesselTreeGenerator& SetDirection( glm::vec3 direction );
        VesselTreeGenerator& SetLength( float length );
        VesselTreeGenerator& SetCellSpacing( float spacing );
        VesselTreeGenerator& SetRingSize( uint32_t ringSize );
        VesselTreeGenerator& SetTubeRadius( float radius );
        VesselTreeGenerator& SetBranchingAngle( float angleDeg );
        VesselTreeGenerator& SetBranchingDepth( uint32_t depth );
        VesselTreeGenerator& SetLengthFalloff( float falloff );
        VesselTreeGenerator& SetAngleJitter( float jitterDeg );
        VesselTreeGenerator& SetBranchProbability( float probability );
        VesselTreeGenerator& SetSeed( uint32_t seed );
        // Radius multiplier applied to each child branch (Murray's law ≈ 0.79 for symmetric bifurcation).
        VesselTreeGenerator& SetTubeRadiusFalloff( float falloff );
        // Fixed arc-length per cell (circumferential direction). When 0, auto-derived from
        // SetTubeRadius() and SetRingSize() so the trunk ring is unchanged.
        VesselTreeGenerator& SetCellWidth( float width );
        // Maximum lateral deviation of a branch midpoint, as a fraction of its length.
        // 0 = straight (default). 0.1-0.2 gives natural-looking gentle curves.
        VesselTreeGenerator& SetCurvature( float curvature );
        // Axial twist: degrees of rotation added to each successive ring along a branch.
        // 0 = no twist (default). 5-15 degrees gives a natural helical arrangement.
        VesselTreeGenerator& SetBranchTwist( float degrees );

        VesselTreeResult Build();

    private:
        VesselTreeGenerator() = default;

        struct BranchJob
        {
            glm::vec3 origin;
            glm::vec3 direction;
            glm::vec3 perp1; // maintained from parent for orientation continuity at junctions
            float     length;
            float     tubeRadius;
            uint32_t  depth;
            uint32_t  parentRingSize = 0; // ring size of parent branch (for junction edge mapping)
            // Indices of the parent's last ring in result.positions (empty for the trunk)
            std::vector<uint32_t> parentLastRing;
        };

        void buildBranch( BranchJob job, VesselTreeResult& result );

        // Compute a vector perpendicular to dir, maintaining continuity from a hint if provided.
        static glm::vec3 perp1From( glm::vec3 dir, glm::vec3 hint = glm::vec3( 0.0f ) );
        // Rodrigues rotation: rotate `perp` from being perpendicular to t0 to being perpendicular to t1.
        static glm::vec3 parallelTransport( glm::vec3 t0, glm::vec3 t1, glm::vec3 perp );

        // Parameters
        glm::vec3 m_origin         = { 0.0f, 0.0f, 0.0f };
        glm::vec3 m_direction      = { 1.0f, 0.0f, 0.0f };
        float     m_length         = 20.0f;
        float     m_cellSpacing    = 2.0f;
        uint32_t  m_ringSize       = 6;
        float     m_tubeRadius     = 1.5f;
        float     m_branchingAngle = 35.0f;
        uint32_t  m_branchingDepth = 2;
        float     m_lengthFalloff  = 0.6f;
        float     m_angleJitter    = 10.0f;
        float     m_branchProb          = 0.8f;
        float     m_tubeRadiusFalloff   = 0.79f;
        float     m_cellWidth           = 0.0f; // 0 = auto-derive at Build() time
        float     m_curvature           = 0.0f; // 0 = straight
        float     m_branchTwist        = 0.0f; // degrees per ring step
        uint32_t  m_seed                = 42;

        std::mt19937 m_rng;
    };

} // namespace DigitalTwin
