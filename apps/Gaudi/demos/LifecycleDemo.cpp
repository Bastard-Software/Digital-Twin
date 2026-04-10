#include "Demos.h"

#include <simulation/BiologyGenerator.h>
#include <simulation/MorphologyGenerator.h>

namespace Gaudi::Demos
{
    void SetupLifecycleDemo( DigitalTwin::SimulationBlueprint& blueprint )
    {
        // Single cell consumes its local O2 supply → turns Hypoxic (purple) → Necrotic (dark).
        // Demonstrates the Live→Hypoxic→Necrotic colour transitions driven by CellCycle.
        blueprint.SetName( "Lifecycle" );
        blueprint.SetDomainSize( glm::vec3( 15.0f ), 1.0f );

        blueprint.ConfigureSpatialPartitioning()
            .SetMethod( DigitalTwin::SpatialPartitioningMethod::HashGrid )
            .SetCellSize( 3.0f )
            .SetMaxDensity( 64 )
            .SetComputeHz( 60.0f );

        // Small O2 reservoir: tight Gaussian so the cell drains it in ~10s.
        blueprint.AddGridField( "Oxygen" )
            .SetInitializer( DigitalTwin::GridInitializer::Gaussian( glm::vec3( 0.0f ), 3.0f, 50.0f ) )
            .SetDiffusionCoefficient( 0.5f )
            .SetDecayRate( 0.0f )
            .SetComputeHz( 60.0f );

        auto& cell = blueprint.AddAgentGroup( "Cell" )
                         .SetCount( 1 )
                         .SetMorphology( DigitalTwin::MorphologyGenerator::CreateSphere( 1.0f ) )
                         .SetDistribution( { glm::vec4( 0.0f, 0.0f, 0.0f, 1.0f ) } )
                         .SetColor( glm::vec4( 0.9f, 0.3f, 0.3f, 1.0f ) ); // red

        // Consumes O2 aggressively — triggers Hypoxic in ~3s, Necrotic in ~6s.
        cell.AddBehaviour( DigitalTwin::Behaviours::ConsumeField{ "Oxygen", 30.0f } ).SetHz( 60.0f );

        cell.AddBehaviour( DigitalTwin::BiologyGenerator::StandardCellCycle()
                               .SetBaseDoublingTime( 999.0f )
                               .SetProliferationOxygenTarget( 60.0f )
                               .SetArrestPressureThreshold( 999.0f )
                               .SetHypoxiaOxygenThreshold( 30.0f )
                               .SetNecrosisOxygenThreshold( 5.0f )
                               .SetApoptosisRate( 0.0f )
                               .Build() )
            .SetHz( 10.0f );
    }
} // namespace Gaudi::Demos
