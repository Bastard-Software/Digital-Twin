#pragma once
#include "core/Core.h"

#include <glm/glm.hpp>
#include <cstdint>
#include <random>
#include <vector>

namespace DigitalTwin
{
    /**
     * @brief Per-cell placement spec emitted by `VesselTreeGenerator`.
     *
     * Item 2 Phase 2.3: Item 1's cell-based physics (JKR + VE-cadherin catch-bond +
     * lateral adhesion + BM-gated polarity with junctional propagation) holds the
     * tree together once cells are placed — no persistent edge or segment data.
     */
    struct VesselCellSpec
    {
        glm::vec4 position;        // xyz=world position, w=1.0 (alive flag)
        glm::vec4 orientation;     // quaternion: local +Y → radial outward, local +X → axial flow
        glm::vec4 polaritySeed;    // xyz=radial outward (basal direction), w=1.0 magnitude
        uint32_t  morphologyIndex; // 0=elongated rhomboid (Phase 2.3); 1/2 reserved for Phase 2.4
        uint32_t  _pad0 = 0;
        uint32_t  _pad1 = 0;
        uint32_t  _pad2 = 0;
    };

    struct VesselTreeResult
    {
        std::vector<VesselCellSpec> cells;
        uint32_t                    totalCells = 0;
    };

    /**
     * @brief CPU-side generator that places endothelial cells along a branching
     *        vessel tree. Phase 2.3: straight-tube + branching backbone with
     *        adaptive per-ring cell count (Aird 2007 DOI 10.1161/01.RES.0000255691.76142.4a)
     *        and pre-seeded radial-outward polarity (Mellman & Nelson 2008
     *        DOI 10.1038/nrm2523; polarity self-sustains via propagation —
     *        Bryant 2010, St Johnston & Ahringer 2010 — once every cell is polar).
     *
     *        Phases 2.4 + 2.5 will retro-fit Stone-Wales 5/7 defects at diameter
     *        transitions and carina heptagons at bifurcations; Phase 2.3 ships
     *        morphology index 0 (elongated rhomboid) for all cells.
     *
     * Usage:
     *   auto tree = VesselTreeGenerator::BranchingTree()
     *       .SetOrigin({0,0,0}).SetDirection({1,0,0}).SetLength(20.0f)
     *       .SetTubeRadius(1.5f).SetBranchingDepth(0)
     *       .SetECCircumferentialWidth(1.0f).SetCellAspectRatio(5.0f)
     *       .Build();
     */
    class DT_API VesselTreeGenerator
    {
    public:
        static VesselTreeGenerator BranchingTree();

        VesselTreeGenerator& SetOrigin( glm::vec3 origin );
        VesselTreeGenerator& SetDirection( glm::vec3 direction );
        VesselTreeGenerator& SetLength( float length );
        VesselTreeGenerator& SetTubeRadius( float radius );
        VesselTreeGenerator& SetBranchingAngle( float angleDeg );
        VesselTreeGenerator& SetBranchingDepth( uint32_t depth );
        VesselTreeGenerator& SetLengthFalloff( float falloff );
        VesselTreeGenerator& SetAngleJitter( float jitterDeg );
        VesselTreeGenerator& SetBranchProbability( float probability );
        VesselTreeGenerator& SetSeed( uint32_t seed );
        // Murray's law child radius factor (≈ 0.79 for symmetric bifurcation; Murray 1926).
        VesselTreeGenerator& SetTubeRadiusFalloff( float falloff );
        // Quadratic-Bezier lateral deviation as a fraction of branch length (0 = straight).
        VesselTreeGenerator& SetCurvature( float curvature );
        // Axial twist per ring step (degrees); natural helical arrangement.
        VesselTreeGenerator& SetBranchTwist( float degrees );

        // Phase 2.3: EC circumferential footprint. Ring count per branch =
        // max(2, round(2π · tubeRadius / ecCircumferentialWidth)). Min 2 reflects the
        // dual-seam capillary lower bound (Bär 1984); 1-cell autocellular capillaries
        // are explicitly not modelled at this point-agent scale.
        VesselTreeGenerator& SetECCircumferentialWidth( float width );

        // Phase 2.3: elongated-rhomboid length / width aspect (arterial ECs 5–20:1 in
        // laminar flow per Davies 2009). Drives axial ring spacing = ecCircWidth × aspect.
        VesselTreeGenerator& SetCellAspectRatio( float aspect );

        // Phase 2.4: end-of-trunk tube radius for linear taper. When > 0 the trunk radius
        // is linearly interpolated from `tubeRadius` at the origin to `tubeRadiusEnd` at
        // the far end, producing axial ring-count transitions (e.g. arteriole → capillary).
        // Default -1 = no taper (Phase 2.3 behaviour preserved). Only applies to the trunk;
        // child branches remain constant-radius internally.
        VesselTreeGenerator& SetTubeRadiusEnd( float radius );

        // Phase 2.4.5: opt-in Stone-Wales 5/7 defect insertion at ring-count transitions.
        // Default FALSE — continuous tapering renders as pure rhombus tiles with no
        // defect polygons, because 5/7 defects at continuous transitions place
        // visually-misplaced oversized meshes at lattice positions that don't reflect
        // genuine topological features (user visual verification 2026-04-19). The
        // defect-placement infrastructure is preserved here for Phase 2.5 carina
        // re-use, where bifurcations ARE genuinely topological (Y-junction apex =
        // pair-of-pants, Gauss-Bonnet mandates heptagons). Unit tests that exercise
        // the infrastructure flip this flag true explicitly.
        VesselTreeGenerator& SetStoneWalesAtTaperTransitions( bool enabled );

        VesselTreeResult Build();

    private:
        VesselTreeGenerator() = default;

        struct BranchJob
        {
            glm::vec3 origin;
            glm::vec3 direction;
            glm::vec3 perp1;        // parallel-transported frame for junction continuity
            float     length;
            float     tubeRadius;
            float     endTubeRadius = -1.0f; // Phase 2.4: < 0 → constant radius (no taper)
            uint32_t  depth;
            float     ringAnglePhase = 0.0f; // staggered brick offset carried into child
        };

        void buildBranch( BranchJob job, VesselTreeResult& result );

        static glm::vec3 perp1From( glm::vec3 dir, glm::vec3 hint = glm::vec3( 0.0f ) );
        static glm::vec3 parallelTransport( glm::vec3 t0, glm::vec3 t1, glm::vec3 perp );

        // --- Parameters ---
        glm::vec3 m_origin           = { 0.0f, 0.0f, 0.0f };
        glm::vec3 m_direction        = { 1.0f, 0.0f, 0.0f };
        float     m_length           = 20.0f;
        float     m_tubeRadius       = 1.5f;
        float     m_branchingAngle   = 35.0f;
        uint32_t  m_branchingDepth   = 0;   // default straight tube for Phase 2.3 visual verification
        float     m_lengthFalloff    = 0.6f;
        float     m_angleJitter      = 10.0f;
        float     m_branchProb       = 1.0f;
        float     m_tubeRadiusFalloff = 0.79f;
        float     m_curvature        = 0.0f;
        float     m_branchTwist      = 0.0f;
        uint32_t  m_seed             = 42;
        float     m_ecCircWidth      = 1.0f; // simulation-unit scale; arteriole-analogous default
        float     m_cellAspect       = 5.0f; // Davies 2009 arterial-EC flow alignment
        float     m_tubeRadiusEnd    = -1.0f; // Phase 2.4: trunk taper end radius; < 0 = no taper
        bool      m_stoneWalesAtTaperTransitions = false; // Phase 2.4.5: opt-in defect insertion for continuous taper

        std::mt19937 m_rng;
    };

} // namespace DigitalTwin
