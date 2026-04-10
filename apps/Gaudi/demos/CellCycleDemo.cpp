#include "Demos.h"

#include <simulation/BiologyGenerator.h>
#include <simulation/BiomechanicsGenerator.h>
#include <simulation/MorphologyGenerator.h>
#include <simulation/SpatialDistribution.h>

namespace Gaudi::Demos
{
    void SetupCellCycleDemo( DigitalTwin::SimulationBlueprint& blueprint )
    {
        // Small domain — easy to see everything
        blueprint.SetName( "CellCycle Demo" );
        blueprint.SetDomainSize( glm::vec3( 20.0f ), 2.0f );

        blueprint.ConfigureSpatialPartitioning()
            .SetMethod( DigitalTwin::SpatialPartitioningMethod::HashGrid )
            .SetCellSize( 3.0f )
            .SetMaxDensity( 64 )
            .SetComputeHz( 60.0f );

        // Oxygen: moderate initial level — will be consumed by tumor
        blueprint.AddGridField( "Oxygen" )
            .SetInitializer( DigitalTwin::GridInitializer::Constant( 40.0f ) )
            .SetDiffusionCoefficient( 2.0f )
            .SetDecayRate( 0.0f )
            .SetComputeHz( 60.0f );

        // 5 tumor cells in a tight cluster at origin
        auto tumorPositions = DigitalTwin::SpatialDistribution::UniformInSphere( 5, 2.0f );

        auto& tumor = blueprint.AddAgentGroup( "TumorCells" )
                           .SetCount( 5 )
                           .SetMorphology( DigitalTwin::MorphologyGenerator::CreateSphere( 1.5f ) )
                           .SetDistribution( tumorPositions )
                           .SetColor( glm::vec4( 0.2f, 0.85f, 0.3f, 1.0f ) ); // Green

        // Brownian jitter — small, just for visual life
        tumor.AddBehaviour( DigitalTwin::Behaviours::BrownianMotion{ 0.2f } ).SetHz( 60.0f );

        // O2 consumption — aggressive, drives hypoxia quickly in small domain
        tumor.AddBehaviour( DigitalTwin::Behaviours::ConsumeField{ "Oxygen", 30.0f } ).SetHz( 60.0f );

        // CellCycle: division → hypoxia → necrosis
        // Low diffusion + small domain means O2 depletes fast → visible lifecycle transitions
        tumor
            .AddBehaviour( DigitalTwin::BiologyGenerator::StandardCellCycle()
                               .SetBaseDoublingTime( 4.0f / 3600.0f )  // Fast division to see it happen
                               .SetProliferationOxygenTarget( 40.0f )
                               .SetArrestPressureThreshold( 15.0f )
                               .SetHypoxiaOxygenThreshold( 20.0f )     // Wide hypoxia window
                               .SetNecrosisOxygenThreshold( 8.0f )     // Well below hypoxia
                               .SetApoptosisRate( 0.0f )
                               .Build() )
            .SetHz( 60.0f );

        // JKR biomechanics — cells push each other apart
        tumor
            .AddBehaviour( DigitalTwin::BiomechanicsGenerator::JKR()
                               .SetYoungsModulus( 20.0f )
                               .SetPoissonRatio( 0.4f )
                               .SetAdhesionEnergy( 1.5f )
                               .SetMaxInteractionRadius( 1.5f )
                               .Build() )
            .SetHz( 60.0f );
    }
} // namespace Gaudi::Demos
