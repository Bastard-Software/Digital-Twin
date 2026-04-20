#include "Demos.h"

#include <simulation/BiomechanicsGenerator.h>
#include <simulation/MorphologyGenerator.h>
#include <simulation/Phenotype.h>
#include <simulation/SpatialDistribution.h>
#include <simulation/VesselTreeGenerator.h>

namespace Gaudi::Demos
{
    // Phase 2.5 — 3-level symmetric Y-branching tree. Demonstrates the three Phase 2.5
    // additions to VesselTreeGenerator::buildBranch:
    //
    //   1. Per-child tapering propagation. Trunk tapers r=3.0 → 2.5 (~17% internal
    //      drop). Murray's law (0.79 per Murray 1926 DOI 10.1073/pnas.12.3.207) applies
    //      AT the bifurcation as a discrete radius drop; within each child the parent's
    //      proportional taper shape is preserved. Three levels take the vessel from
    //      ~3 down to ~1.1 — an artery → arteriole → capillary-scale progression.
    //
    //   2. Parent-last-ring carina flagging. At each Y-junction, the 2 cells sitting
    //      on the bisection plane get `isCarina = true`. Phase 2.5 does not render
    //      them differently (all cells are rhombus tiles). Phase 2.6.5 dynamic
    //      topology will render them as 6-to-8-sided Voronoi polygons, matching the
    //      cobblestone EC morphology observed at real flow dividers (Chiu & Chien 2011
    //      DOI 10.1152/physrev.00047.2009; van der Heiden 2013).
    //
    //   3. Child-first-ring carina flagging. The 2 cells on each child's first ring
    //      nearest to the sibling branch get `isCarina = true` for the same reason.
    //
    // Known visual artefact (carried over from Phase 2.4.5): rhombus tiles cannot
    // bridge the three-ring local topology at each Y-junction carina, so a gap /
    // bumpiness is visible at each bifurcation apex. Phase 2.6.5 dynamic topology
    // closes these automatically — Phase 2.5 ships geometry correctness only.
    //
    // Physics stack is Item 1's quiescent regime: JKR + VE-cad catch-bond
    // (Rakshit 2012 DOI 10.1073/pnas.1208349109) + lateral adhesion + pre-seeded
    // polarity self-sustained via Phase-4.5 junctional propagation (Bryant 2010;
    // St Johnston & Ahringer 2010). Damping 500, Brownian 0.02 — identical to the
    // tube demos.
    void SetupBranchingTreeDemo( DigitalTwin::SimulationBlueprint& blueprint )
    {
        blueprint.SetName( "Branching Tree" );
        blueprint.SetDomainSize( glm::vec3( 80.0f ), 1.5f );

        blueprint.ConfigureSpatialPartitioning()
            .SetMethod( DigitalTwin::SpatialPartitioningMethod::HashGrid )
            .SetCellSize( 3.0f )
            .SetMaxDensity( 64 )
            .SetComputeHz( 60.0f );

        const float ecWidth        = 1.0f;
        const float aspect         = 1.5f;
        const float trunkRadius    = 3.0f;
        const float trunkRadiusEnd = 2.5f;  // gentle internal trunk taper
        // Extended to 22.5 — longer branches let each level build more in-branch cadherin
        // contacts before the junction orientation mismatch destabilises. Per-branch ring
        // counts: 15 (trunk) / 10 (L1) / 7 (L2). The remaining inter-ring gaps at taper
        // transitions and carina apices are fundamental to static-primitive rendering of
        // a surface whose underlying lattice topology changes discretely (Phase 2.4.5 +
        // 2.5 known artefacts). Phase 2.6.5 dynamic topology (Voronoi polygons from each
        // cell's JKR spatial-hash neighbours) closes them by computing polygon shape from
        // actual cell positions rather than a fixed rhombus template — matching how real
        // endothelium defines its boundaries via VE-cadherin contacts (Halbleib & Nelson
        // 2006 DOI 10.1038/nrm1975).
        const float length         = 22.5f;

        auto tree = DigitalTwin::VesselTreeGenerator::BranchingTree()
                        .SetOrigin( glm::vec3( -length * 0.5f, 0.0f, 0.0f ) )
                        .SetDirection( glm::vec3( 1.0f, 0.0f, 0.0f ) )
                        .SetLength( length )
                        .SetTubeRadius( trunkRadius )
                        .SetTubeRadiusEnd( trunkRadiusEnd )
                        .SetECCircumferentialWidth( ecWidth )
                        .SetCellAspectRatio( aspect )
                        .SetBranchingDepth( 2 )           // 3 levels total: trunk + L1 + L2
                        .SetBranchingAngle( 35.0f )
                        .SetAngleJitter( 5.0f )
                        .SetBranchProbability( 1.0f )
                        .SetLengthFalloff( 0.65f )
                        .SetTubeRadiusFalloff( 0.79f )    // Murray 1926 symmetric split
                        .SetSeed( 42 )
                        .Build();

        auto& ecs = blueprint.AddAgentGroup( "Branching Tree Endothelium" );
        ecs.SetVesselTree( tree )
           .SetColor( glm::vec4( 0.88f, 0.42f, 0.46f, 1.0f ) )
           .SetDynamicTopology( true ); // Phase 2.6.5.b opt-in

        // Single rhombus variant throughout the tree — carina cells render identically
        // to non-carina cells in Phase 2.5. Diamond-tiling geometry: `longDiagonal =
        // 2 × axialStep` so diamonds corner-align with neighbours across staggered rings.
        ecs.SetMorphologyVariants(
            DigitalTwin::CellType::Default,
            {
                DigitalTwin::MorphologyGenerator::CreateRhombus(
                    /*longDiagonal=*/ecWidth * aspect * 2.0f,
                    /*shortDiagonal=*/ecWidth,
                    /*thickness=*/0.2f ),
            } );

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
                              2.0f,   // catchBondStrength (Rakshit 2012)
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
