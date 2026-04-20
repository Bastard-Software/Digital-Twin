#include "Demos.h"

#include <simulation/BiomechanicsGenerator.h>
#include <simulation/MorphologyGenerator.h>
#include <simulation/Phenotype.h>
#include <simulation/SpatialDistribution.h>
#include <simulation/VesselTreeGenerator.h>

namespace Gaudi::Demos
{
    // Phase 2.6 — full artery → arteriole → capillary hierarchy demo. 4-level tree
    // (trunk + L1 + L2 + L3 = 15 branches, 7 bifurcations, 8 capillary leaves).
    // Demonstrates the complete Item 2 pipeline composed end-to-end:
    //   - Phase 2.3 adaptive ring count per radius (Aird 2007
    //     DOI 10.1161/01.RES.0000255691.76142.4a)
    //   - Phase 2.3 pre-seeded polarity with Phase-4.5 junctional propagation
    //     self-sustaining the magnitude (Bryant 2010; St Johnston & Ahringer 2010)
    //   - Phase 2.4.5 rhombus / diamond tile tessellation (Davies 2009
    //     DOI 10.1038/nrcardio.2009.14)
    //   - Phase 2.5 per-child tapering propagation and carina-cell flagging at
    //     every bifurcation (cobblestone EC biology per Chiu & Chien 2011
    //     DOI 10.1152/physrev.00047.2009)
    //
    // No new generator work vs Phase 2.5 — pure composition on the cell-based
    // substrate Item 1 physics proved sufficient.
    //
    // Anatomical simplification for Murray's law: the biologically canonical
    // symmetric 2-way split is r_child = 2^(-1/3) × r_parent ≈ 0.794 × r_parent
    // (Murray 1926 DOI 10.1073/pnas.12.3.207). To go from a ~20 µm arteriole to a
    // ~2 µm capillary in vivo takes 5–8 bifurcations. This demo telescopes that
    // transition into 3 bifurcations by using Murray 0.5 — equivalent to ~3 real
    // Murray-0.79 levels aggregated (0.794^3 ≈ 0.5). Each demo-level therefore
    // represents roughly "one step down the Pries & Secomb 2005 arteriole →
    // metarteriole → precapillary → capillary hierarchy."
    //
    // Ring-size cascade (ecWidth 1.0 → ring = max(2, round(2π·r))):
    //     Trunk r = 3.00 → ring 19 (arteriole scale)
    //     L1    r = 1.50 → ring  9 (post-capillary venule / arteriole crossover)
    //     L2    r = 0.75 → ring  5 (terminal arteriole)
    //     L3    r ≈ 0.38 → ring  2 (dual-seam capillary, floor-clamped per Bär 1984
    //                               DOI 10.1002/cne.902320402; autocellular 1-cell
    //                               capillaries are explicitly not modelled at this
    //                               point-agent scale)
    //
    // Per-branch ring counts at length 32 / lengthFalloff 0.65: 22 / 14 / 10 / 7.
    // All ≥ the Phase 2.5 stability budget (~7 rings per branch at 30° branching
    // angle) so the junction orientation mismatch can't destabilise short branches.
    //
    // Known visual artefact (carried from Phase 2.5): a gap / bumpiness is visible
    // at each of the 7 Y-junction carina apices. Rhombus tiles cannot bridge the
    // three-ring local topology where parent + two daughters meet. Phase 2.6.5
    // dynamic topology (Voronoi polygons from JKR spatial-hash neighbours) closes
    // these automatically — Phase 2.6 ships geometry correctness only.
    void SetupDesignedVesselDemo( DigitalTwin::SimulationBlueprint& blueprint )
    {
        blueprint.SetName( "Designed Vessel" );
        // Trunk 32 + L1/L2/L3 length falloffs at ±30° branching can reach ~55 units
        // off-origin in the worst branch direction (L1 × 0.65 + L2 × 0.65² + L3 ×
        // 0.65³ cascaded with cos(angle) projections). Domain 140 gives a ~15-unit
        // margin on each side after the bounds validator adds maxInteractionRadius.
        blueprint.SetDomainSize( glm::vec3( 140.0f ), 1.5f );

        blueprint.ConfigureSpatialPartitioning()
            .SetMethod( DigitalTwin::SpatialPartitioningMethod::HashGrid )
            .SetCellSize( 3.0f )
            .SetMaxDensity( 64 )
            .SetComputeHz( 60.0f );

        const float ecWidth     = 1.0f;
        const float aspect      = 1.5f;
        const float trunkRadius = 3.0f;
        const float length      = 32.0f;

        auto tree = DigitalTwin::VesselTreeGenerator::BranchingTree()
                        .SetOrigin( glm::vec3( -length * 0.5f, 0.0f, 0.0f ) )
                        .SetDirection( glm::vec3( 1.0f, 0.0f, 0.0f ) )
                        .SetLength( length )
                        .SetTubeRadius( trunkRadius )
                        .SetECCircumferentialWidth( ecWidth )
                        .SetCellAspectRatio( aspect )
                        .SetBranchingDepth( 3 )            // trunk + L1 + L2 + L3 = 4 levels
                        .SetBranchingAngle( 30.0f )        // slightly less than BranchingTreeDemo's 35°
                                                           //   so the deeper 3-level tree fits comfortably
                        .SetAngleJitter( 5.0f )
                        .SetBranchProbability( 1.0f )
                        .SetLengthFalloff( 0.65f )
                        .SetTubeRadiusFalloff( 0.5f )      // anatomical simplification — see header comment
                        .SetCurvature( 0.15f )             // per-branch Bezier midpoint deflection; each
                                                           //   branch bends organically, matching the in-vivo
                                                           //   appearance of vasculature vs the mathematically
                                                           //   straight trunks that procedural trees usually
                                                           //   produce. Proven at 0.2 by CurvedTubeDemo; 0.15
                                                           //   here so the deeper L3 branches don't curl too
                                                           //   hard within their short 7-ring length.
                        .SetSeed( 42 )
                        .Build();

        auto& ecs = blueprint.AddAgentGroup( "Designed Vessel Endothelium" );
        ecs.SetVesselTree( tree )
           .SetColor( glm::vec4( 0.88f, 0.42f, 0.46f, 1.0f ) );

        // Single rhombus variant throughout. Carina cells flagged by Phase 2.5 but
        // not rendered differently — Phase 2.6.5 dynamic topology will pick up the
        // `isCarina` flag to produce 6-to-8-sided Voronoi polygons at flow dividers
        // (Chiu & Chien 2011 cobblestone biology).
        ecs.SetMorphologyVariants(
            DigitalTwin::CellType::Default,
            {
                DigitalTwin::MorphologyGenerator::CreateRhombus(
                    /*longDiagonal=*/ecWidth * aspect * 2.0f,
                    /*shortDiagonal=*/ecWidth,
                    /*thickness=*/0.2f ),
            } );

        // Physics stack byte-identical to the Phase 2.5 BranchingTreeDemo, which
        // is in turn byte-identical to the Item 1 trilogy (ECBlob / EC2DMatrigel /
        // ECTube) — the scientific claim of Item 1 is that the same molecular kit
        // produces the right phenotype in every ECM context, and Phase 2.6 extends
        // that to the full vascular hierarchy.
        auto jkr = DigitalTwin::BiomechanicsGenerator::JKR()
                       .SetYoungsModulus( 20.0f )
                       .SetPoissonRatio( 0.4f )
                       .SetAdhesionEnergy( 5.0f )
                       .SetMaxInteractionRadius( 0.75f )
                       .SetDampingCoefficient( 500.0f )
                       .SetCorticalTension( 0.5f )
                       .SetLateralAdhesionScale( 0.15f )
                       .Build();
        ecs.AddBehaviour( jkr ).SetHz( 60.0f );

        ecs.AddBehaviour( DigitalTwin::Behaviours::CadherinAdhesion{
                              glm::vec4( 0.0f, 0.0f, 1.0f, 0.0f ),
                              0.05f,  // expressionRate
                              0.001f, // degradationRate
                              2.0f,   // couplingStrength
                              2.0f,   // catchBondStrength (Rakshit 2012 VE-cad X-dimer)
                              0.3f    // catchBondPeakLoad
                          } )
            .SetHz( 60.0f );

        DigitalTwin::Behaviours::CellPolarity polarity;
        polarity.regulationRate      = 0.2f;
        polarity.apicalRepulsion     = 0.3f;
        polarity.basalAdhesion       = 1.5f;
        polarity.propagationStrength = 1.0f;
        ecs.AddBehaviour( polarity ).SetHz( 60.0f );

        ecs.AddBehaviour( DigitalTwin::Behaviours::BrownianMotion{ 0.02f } ).SetHz( 60.0f );
    }
} // namespace Gaudi::Demos
