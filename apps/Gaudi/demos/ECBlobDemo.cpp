#include "Demos.h"

#include <simulation/BiomechanicsGenerator.h>
#include <simulation/MorphologyGenerator.h>
#include <simulation/SpatialDistribution.h>

#include <glm/glm.hpp>
#include <cmath>

namespace Gaudi::Demos
{
    namespace
    {
        // ── Shared EC cloud geometry ──────────────────────────────────────────
        // These are the initial-state parameters shared by both ECBlobDemo (no plate)
        // and EC2DMatrigelDemo (with plate). Keeping them in one place guarantees the two
        // demos start from an identical cloud — the only allowed divergence between
        // them is the BasementMembrane behaviour (added in Phase 2).
        //
        // Geometry is an elongated cylinder along +X, lifted so its lowest cells sit
        // at y ≈ 0.5 — inside future plate anchorage distance for EC2DMatrigelDemo.
        // The renderer is Y-up, so the basement-membrane plate lives in the XZ
        // plane (normal = +Y) and the cluster sits on top of it, elongated along
        // the X axis. In the camera that reads as a horizontal log on the floor.
        //
        // Regular cubic-lattice placement — no random pair-overlap. Random
        // UniformInCylinder placement creates occasional pairs at very small
        // distance; even mild overlap generates Hertz repulsion (pow(ov,1.5))
        // that flings cells past the interaction cutoff before adhesion can
        // respond, and once outside the cutoff there is NO force at all so they
        // never return. Regular lattice at ~equilibrium spacing avoids this
        // entirely and matches the in-vitro centrifuged-aggregate starting
        // condition (cells in regular close-packing contact).
        constexpr float    k_cloudRadius      = 2.0f;   // cylinder radius
        constexpr float    k_cloudHalfLength  = 6.0f;   // ~3:1 elongation
        constexpr float    k_cloudCenterY     = 2.5f;   // lowest y ≈ 0.5 (above plate at y=0)
        constexpr float    k_latticeSpacing   = 1.2f;   // ≈ 10% above JKR eq dist 1.06 → net attractive
        constexpr float    k_cellMaxRadius    = 0.75f;

        // CurvedTile parameters — tile arc-angle and axial spacing only affect
        // how the flat mesh is cut; orientations are set per cell below.
        constexpr float    k_tileArcAngle    = 20.0f;  // 360/18 ≈ packable ring
        constexpr float    k_tileAxialSpacing= 1.05f;
        constexpr float    k_tileThickness   = 0.25f;
    }

    // Defined here; declared in Demos.h. Populates the given AgentGroup with an
    // elongated cloud of random endothelial cells (CurvedTile morphology, oriented
    // outward from the cluster centroid). Does NOT attach behaviours — the caller
    // composes the behaviour stack (and adds BasementMembrane for EC2DMatrigelDemo).
    void SeedECCloud( DigitalTwin::AgentGroup& group, uint32_t /*seed*/ )
    {
        // Regular-lattice placement in an elongated cylinder along +X, sitting
        // above y=0 so EC2DMatrigelDemo's basement-membrane plate (XZ plane, +Y normal)
        // catches the bottom cells.
        auto positions = DigitalTwin::SpatialDistribution::LatticeInCylinder(
            k_latticeSpacing,
            k_cloudRadius,
            k_cloudHalfLength,
            glm::vec3( 0.0f, k_cloudCenterY, 0.0f ),
            glm::vec3( 1.0f, 0.0f, 0.0f ) );

        // Initial orientation: shortest-arc quaternion from model +Y to a unit
        // vector pointing from the cluster centroid to the cell. This gives surface
        // cells a sensible basal-out orientation from frame 0 while keeping the
        // blob visually coherent. Interior cells with near-zero displacement get
        // identity — CellPolarity will settle them toward symmetric (w → 0).
        glm::vec3 centroid( 0.0f );
        for( const auto& p : positions ) centroid += glm::vec3( p );
        if( !positions.empty() ) centroid /= static_cast<float>( positions.size() );

        std::vector<glm::vec4> orientations;
        orientations.reserve( positions.size() );
        for( const auto& p : positions )
        {
            glm::vec3 d = glm::vec3( p ) - centroid;
            float     l = glm::length( d );
            if( l > 0.01f )
            {
                d /= l;
                // Shortest-arc rotation from +Y to d as a unit quaternion.
                // Axis = cross(+Y, d) / sin(theta); for theta near 0 we fall back to identity.
                glm::vec3 axis = glm::cross( glm::vec3( 0.0f, 1.0f, 0.0f ), d );
                float     s   = glm::length( axis );
                if( s > 0.01f )
                {
                    axis /= s;
                    // Half-angle identities: cos(θ/2)=inv, sin(θ/2)=inv when θ=90°.
                    // Use a stable form for arbitrary angles.
                    float cosT = glm::dot( glm::vec3( 0.0f, 1.0f, 0.0f ), d );
                    float w    = std::sqrt( 0.5f * ( 1.0f + cosT ) );
                    float sHalf= std::sqrt( std::max( 0.0f, 0.5f * ( 1.0f - cosT ) ) );
                    orientations.push_back( glm::vec4( axis * sHalf, w ) );
                }
                else
                {
                    // d aligned with ±Y: identity for +Y, 180° around X for -Y.
                    if( d.y > 0.0f ) orientations.push_back( glm::vec4( 0.0f, 0.0f, 0.0f, 1.0f ) );
                    else             orientations.push_back( glm::vec4( 1.0f, 0.0f, 0.0f, 0.0f ) );
                }
            }
            else
            {
                orientations.push_back( glm::vec4( 0.0f, 0.0f, 0.0f, 1.0f ) );
            }
        }

        group
            .SetCount( static_cast<uint32_t>( positions.size() ) )
            .SetMorphology( DigitalTwin::MorphologyGenerator::CreateCurvedTile(
                k_tileArcAngle, k_tileAxialSpacing, k_tileThickness, k_cloudRadius ) )
            .SetDistribution( positions )
            .SetOrientations( orientations )
            // Endothelium colour — salmon/rose, matching vessel-wall H&E staining
            // and textbook cartoon colouring. Distinct from tumour (red) and
            // immune (blue) cells in other demos.
            .SetColor( glm::vec4( 0.88f, 0.42f, 0.46f, 1.0f ) );
    }

    void SetupECBlobDemo( DigitalTwin::SimulationBlueprint& blueprint )
    {
        // ── EC blob, no ECM cue ───────────────────────────────────────────────
        // Phase 1 baseline. An unstructured elongated cloud of endothelial cells
        // with JKR + VE-cadherin adhesion + apical-basal polarity — the same
        // physics stack as EC2DMatrigelDemo, but WITHOUT a basement-membrane plate.
        //
        // Biological analog: endothelial cells in suspension / ultra-low-attachment
        // culture, where absent ECM contact prevents polarity establishment and
        // the classic in-vitro outcome is solid spheroids (NOT tubes). This demo
        // is intentionally the negative control of EC2DMatrigelDemo.
        //
        // Phase 1 expectation: the elongated cloud compresses into a solid sausage
        // within a few seconds. No lumen. This is the "blobbing collapse" baseline
        // that later phases (net-negative apical, cortical tension, catch-bond)
        // progressively fix — but only in EC2DMatrigelDemo, which has the plate cue.
        // ECBlobDemo stays blob-like throughout the plan.

        blueprint.SetName( "EC Blob" );
        blueprint.SetDomainSize( glm::vec3( 40.0f ), 1.5f );

        blueprint.ConfigureSpatialPartitioning()
            .SetMethod( DigitalTwin::SpatialPartitioningMethod::HashGrid )
            .SetCellSize( 3.0f )
            .SetMaxDensity( 64 )
            .SetComputeHz( 60.0f );

        auto& ecs = blueprint.AddAgentGroup( "Endothelial Cells" );
        SeedECCloud( ecs, 42 );

        // ── Biomechanics ───────────────────────────────────────────────────────
        // Biology: in vitro suspension culture. ECs aggregate via VE-cadherin
        // homotypic adhesion into solid spheroids. They do NOT establish
        // apical-basal polarity — polarity requires a symmetry-breaking cue
        // (BM integrin engagement) which this demo explicitly lacks. Without
        // integrin, ECs would eventually undergo anoikis in real biology; we
        // don't model cell death and so approximate "freshly suspended ECs."
        //
        // Revision note (Phase 4.5 review). Earlier revisions added apical
        // repulsion (Phase 3), cortical tension (Phase 4), and junctional
        // propagation (Phase 4.5) to this demo. Those mechanisms belong to the
        // 3D collagen gel tube-formation biology (Strilic 2009 cord hollowing),
        // NOT to the 2D / suspension setups modelled here. Layered on the
        // suspension baseline they produced cluster disruption without
        // corresponding phenotype benefit. Removed the CellPolarity behaviour
        // outright — in a no-BM setting it was spuriously polarising surface
        // cells via the neighbour-centroid cue. Cortical tension kept at 0.5
        // because it IS a cell-intrinsic property (actomyosin contractility)
        // that doesn't require any ECM context to be valid.
        //
        // AdhesionEnergy 5.0 widens the JKR attractive basin to ~0.44 (from
        // ~0.07 at the Phase-1 default 2.0). Biological anchor: VE-cadherin
        // homotypic adhesion energy in the 0.5-2 mN/m range; 5.0 is a
        // dimensionless proxy scaled to sim units.
        // Phase 4.5-B: lateralAdhesionScale = 0.15 adds translational pull to
        // hull sub-sphere pairs, modelling VE-cadherin belt junctions. At
        // face-to-face contact many hull pairs overlap and their translations
        // sum; at corner-only contact the pull is small. This drives edge-to-
        // edge cell commitment rather than the random 3D packing that pure
        // omnidirectional point-particle adhesion produced.
        auto jkr = DigitalTwin::BiomechanicsGenerator::JKR()
                       .SetYoungsModulus( 20.0f )
                       .SetPoissonRatio( 0.4f )
                       .SetAdhesionEnergy( 5.0f )
                       .SetMaxInteractionRadius( 0.75f )
                       .SetDampingCoefficient( 150.0f )
                       .SetCorticalTension( 0.5f )
                       .SetLateralAdhesionScale( 0.15f )
                       .Build();
        ecs.AddBehaviour( jkr ).SetHz( 60.0f );

        // VE-cadherin (channel z) — homophilic EC–EC adhesion. Drives spheroid
        // formation through lateral junctional assembly (VE-cad belts).
        //
        // Phase 5 — catch-bond parameters unified across all three demos. Same
        // molecular machinery (VE-cadherin X-dimer) → same catch-bond kinetics
        // regardless of environment. Rakshit 2012: peak ~2× stiffening at ~30 pN.
        // In ECBlobDemo (spheroid, minimal tensile load) the multiplier has
        // little effect — the aggregate rest state hovers near the compressed
        // regime where loadNorm is below the peak. The parameter is present so
        // the three demos share byte-identical CadherinAdhesion configuration.
        ecs.AddBehaviour( DigitalTwin::Behaviours::CadherinAdhesion{
                              glm::vec4( 0.0f, 0.0f, 1.0f, 0.0f ),
                              0.05f,  // expressionRate
                              0.001f, // degradationRate
                              2.0f,   // couplingStrength
                              2.0f,   // catchBondStrength (Rakshit 2012 VE-cad X-dimer)
                              0.3f    // catchBondPeakLoad
                          } )
            .SetHz( 60.0f );

        // ── CellPolarity (Step A — unified cell behaviours across all demos) ──
        //
        // Same CellPolarity parameters as EC2DMatrigelDemo and the future
        // ECTubeDemo. Biological principle: ECs have intrinsic PAR/Crumbs +
        // VE-cadherin polarity machinery regardless of substrate. What
        // differs between demos is the ENVIRONMENT, not the cell's
        // molecular kit.
        //
        // In ECBlobDemo specifically: no BM plate → `plateWeight = 0` for
        // every cell → no propagation seed → polarity magnitude stays at 0
        // (post-Step-A polarity model gates magnitude by plate+propagation
        // only, not by geometric centroid). The cluster remains effectively
        // unpolarised and behaves as a pure cadherin/lateral-adhesion
        // aggregate. This models anoikis biology: suspended ECs without
        // integrin engagement cannot maintain apical-basal polarity.
        //
        // The apical/basal values are kept conservative (apical = 0.3, i.e.
        // weakly ATTRACTIVE rather than Strilic's -1.0 repulsive) because the
        // 2D demos don't need cord hollowing. Phase 6 will sweep toward more
        // aggressive values once the 3D ECM demo exists to validate them.
        DigitalTwin::Behaviours::CellPolarity polarity;
        polarity.regulationRate      = 0.2f;
        polarity.apicalRepulsion     = 0.3f;
        polarity.basalAdhesion       = 1.5f;
        polarity.propagationStrength = 1.0f;
        ecs.AddBehaviour( polarity ).SetHz( 60.0f );

        // Thermal noise — minimal jitter to help cells escape local energy
        // minima during rearrangement.
        ecs.AddBehaviour( DigitalTwin::Behaviours::BrownianMotion{ 0.1f } ).SetHz( 60.0f );
    }
} // namespace Gaudi::Demos
