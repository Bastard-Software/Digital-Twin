#include "Demos.h"

#include <simulation/BiomechanicsGenerator.h>
#include <simulation/MorphologyGenerator.h>
#include <simulation/Phenotype.h>
#include <simulation/SpatialDistribution.h>
#include <simulation/VesselTreeGenerator.h>

namespace Gaudi::Demos
{
    // Phase 2.3 — single straight tube built by the refactored VesselTreeGenerator.
    //
    // The generator emits per-cell position + quaternion orientation + radial-outward
    // polarity seed; no edge graph, no segment metadata. Item 1's cell-based physics
    // (JKR + VE-cad catch-bond + lateral adhesion + cortical tension + CellPolarity
    // with Phase-4.5 junctional propagation) holds the tube together once the cells
    // are placed. Pre-seeded polarity self-sustains through propagation — no BM plate
    // needed, matching the biology of mature in-vivo vessels which inherit polarity
    // from development (Mellman & Nelson 2008 DOI 10.1038/nrm2523; Iruela-Arispe &
    // Davis 2009 DOI 10.1016/j.devcel.2009.08.011).
    //
    // Adaptive ring count: 2π·radius / ECCircumferentialWidth (Aird 2007
    // DOI 10.1161/01.RES.0000255691.76142.4a). Staggered brick pattern: alternate
    // rings are offset by half a cell-width circumferentially (Davies 2009
    // DOI 10.1038/nrcardio.2009.14 — brick interlocks avoid longitudinal-seam
    // instability under JKR + catch-bond loads).
    //
    // Phase 2.4 will add Stone-Wales 5/7 defects at diameter transitions; Phase 2.5
    // adds carina heptagons at bifurcations. Phase 2.3 ships morphology index 0
    // (elongated rhomboid) for every cell.
    void SetupStraightTubeDemo( DigitalTwin::SimulationBlueprint& blueprint )
    {
        blueprint.SetName( "Straight Tube" );
        blueprint.SetDomainSize( glm::vec3( 40.0f ), 1.5f );

        blueprint.ConfigureSpatialPartitioning()
            .SetMethod( DigitalTwin::SpatialPartitioningMethod::HashGrid )
            .SetCellSize( 3.0f )
            .SetMaxDensity( 64 )
            .SetComputeHz( 60.0f );

        // Straight tube parameters. Radius 2.5 + ECWidth 1.2 → ~13 cells per ring
        // (arteriole-like per Aird 2007). Aspect ratio 1.5 gives modest rhomboid
        // elongation; Phase 2.5's `DesignedVesselDemo` will use the full 5–8:1
        // arterial elongation once diameter transitions + carina heptagons are in.
        const float tubeRadius = 2.5f;
        const float ecWidth    = 1.2f;
        const float aspect     = 1.5f;
        const float length     = 15.0f;

        auto tree = DigitalTwin::VesselTreeGenerator::BranchingTree()
                        .SetOrigin( glm::vec3( -length * 0.5f, 0.0f, 0.0f ) )
                        .SetDirection( glm::vec3( 1.0f, 0.0f, 0.0f ) )
                        .SetLength( length )
                        .SetTubeRadius( tubeRadius )
                        .SetECCircumferentialWidth( ecWidth )
                        .SetCellAspectRatio( aspect )
                        .SetBranchingDepth( 0 )
                        .SetSeed( 42 )
                        .Build();

        auto& ecs = blueprint.AddAgentGroup( "Vessel Endothelium" );
        ecs.SetVesselTree( tree )
           .SetColor( glm::vec4( 0.88f, 0.42f, 0.46f, 1.0f ) ); // endothelium rose (Item 1 palette)

        // Register the elongated rhomboid as the variant-0 mesh for CellType::Default.
        // Every cell carries morphologyIndex=0 for Phase 2.3, so variant 0 is the
        // only draw that fires. Thickness matched to ECBlobDemo (0.2) for visual
        // consistency; length = ecWidth * aspect so tile footprint matches axial spacing.
        ecs.SetMorphologyVariants(
            DigitalTwin::CellType::Default,
            {
                DigitalTwin::MorphologyGenerator::CreateElongatedQuad(
                    /*length=*/ecWidth * aspect, /*width=*/ecWidth, /*thickness=*/0.2f ),
            } );

        // ── Item 1 behaviour stack (byte-identical to ECBlob / EC2DMatrigel / ECTube) ──
        // Same molecular kit regardless of environment; only the ECM context differs.
        // Phase 2.3 tube has no BM plate — pre-seeded polarity + propagation sustains
        // the polarity field without integrin cue.
        auto jkr = DigitalTwin::BiomechanicsGenerator::JKR()
                       .SetYoungsModulus( 20.0f )
                       .SetPoissonRatio( 0.4f )
                       .SetAdhesionEnergy( 5.0f )
                       .SetMaxInteractionRadius( 0.75f )
                       .SetDampingCoefficient( 500.0f ) // higher than Item 1 — quiescent tube regime; kills per-cell bobbing from asymmetric ring overlap

                       .SetCorticalTension( 0.5f )
                       .SetLateralAdhesionScale( 0.15f )
                       .Build();
        ecs.AddBehaviour( jkr ).SetHz( 60.0f );

        ecs.AddBehaviour( DigitalTwin::Behaviours::CadherinAdhesion{
                              glm::vec4( 0.0f, 0.0f, 1.0f, 0.0f ), // VE-cadherin channel
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

        // Brownian strength lowered vs Item 1 trilogy (was 0.1): a tube born at
        // equilibrium from the generator does not need thermal-escape dynamics,
        // and 0.1 is visible as per-frame edge jitter when cells aren't rearranging.
        // 0.02 keeps minimal noise for any residual rebalancing.
        ecs.AddBehaviour( DigitalTwin::Behaviours::BrownianMotion{ 0.02f } ).SetHz( 60.0f );
    }
} // namespace Gaudi::Demos
