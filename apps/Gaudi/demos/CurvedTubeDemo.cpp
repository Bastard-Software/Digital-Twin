#include "Demos.h"

#include <simulation/BiomechanicsGenerator.h>
#include <simulation/MorphologyGenerator.h>
#include <simulation/Phenotype.h>
#include <simulation/SpatialDistribution.h>
#include <simulation/VesselTreeGenerator.h>

namespace Gaudi::Demos
{
    // Phase 2.3 addendum — curved straight tube (non-zero `curvature`).
    //
    // Exercises the quadratic-Bezier centreline path + parallel-transported orientation
    // frames under the full Item-1 behaviour stack. Verifies that polarity propagation,
    // catch-bond stabilisation, and lateral adhesion all stay happy when the tube axis
    // isn't colinear with a world axis — the geometry Phase 2.6 DesignedVesselDemo
    // will encounter on every branch.
    //
    // Provisional demo: will either be removed once DesignedVesselDemo ships in Phase 2.6,
    // or evolved into a branching-tree demo during Phase 2.5.
    void SetupCurvedTubeDemo( DigitalTwin::SimulationBlueprint& blueprint )
    {
        blueprint.SetName( "Curved Tube" );
        blueprint.SetDomainSize( glm::vec3( 50.0f ), 1.5f );

        blueprint.ConfigureSpatialPartitioning()
            .SetMethod( DigitalTwin::SpatialPartitioningMethod::HashGrid )
            .SetCellSize( 3.0f )
            .SetMaxDensity( 64 )
            .SetComputeHz( 60.0f );

        const float tubeRadius = 2.5f;
        const float ecWidth    = 1.2f;
        const float aspect     = 1.5f;
        const float length     = 25.0f;  // longer than StraightTube so curvature is visible

        auto tree = DigitalTwin::VesselTreeGenerator::BranchingTree()
                        .SetOrigin( glm::vec3( -length * 0.5f, 0.0f, 0.0f ) )
                        .SetDirection( glm::vec3( 1.0f, 0.0f, 0.0f ) )
                        .SetLength( length )
                        .SetTubeRadius( tubeRadius )
                        .SetECCircumferentialWidth( ecWidth )
                        .SetCellAspectRatio( aspect )
                        .SetBranchingDepth( 0 )
                        .SetCurvature( 0.2f )  // 20% lateral deflection of the centreline
                        .SetSeed( 7 )
                        .Build();

        auto& ecs = blueprint.AddAgentGroup( "Vessel Endothelium" );
        ecs.SetVesselTree( tree )
           .SetColor( glm::vec4( 0.88f, 0.42f, 0.46f, 1.0f ) );

        ecs.SetMorphologyVariants(
            DigitalTwin::CellType::Default,
            {
                DigitalTwin::MorphologyGenerator::CreateElongatedQuad(
                    /*length=*/ecWidth * aspect, /*width=*/ecWidth, /*thickness=*/0.2f ),
            } );

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
