#include "Demos.h"

#include <simulation/MorphologyGenerator.h>

namespace Gaudi::Demos
{
    void SetupConsumeDemo( DigitalTwin::SimulationBlueprint& blueprint )
    {
        // One cell drains a Gaussian pre-seeded field.
        // Goal: clearly see the local depletion spreading outward from the cell.
        blueprint.SetName( "Consume Demo" );
        blueprint.SetDomainSize( glm::vec3( 20.0f ), 1.0f );

        blueprint.ConfigureSpatialPartitioning()
            .SetMethod( DigitalTwin::SpatialPartitioningMethod::HashGrid )
            .SetCellSize( 3.0f )
            .SetMaxDensity( 64 )
            .SetComputeHz( 60.0f );

        // Field starts full; diffusion replenishes from edges as cell depletes the local peak.
        blueprint.AddGridField( "Substance" )
            .SetInitializer( DigitalTwin::GridInitializer::Gaussian( glm::vec3( 0.0f ), 6.0f, 100.0f ) )
            .SetDiffusionCoefficient( 2.5f )
            .SetDecayRate( 0.0f )
            .SetComputeHz( 60.0f );

        auto& cells = blueprint.AddAgentGroup( "ConsumingCell" )
                          .SetCount( 1 )
                          .SetMorphology( DigitalTwin::MorphologyGenerator::CreateSphere( 1.0f ) )
                          .SetDistribution( { glm::vec4( 0.0f, 0.0f, 0.0f, 1.0f ) } )
                          .SetColor( glm::vec4( 1.0f, 0.35f, 0.1f, 1.0f ) ); // orange

        // Rate=100 creates a fast, clearly visible depletion hole at the cell position.
        cells.AddBehaviour( DigitalTwin::Behaviours::ConsumeField{ "Substance", 1000.0f } )
            .SetHz( 60.0f );
    }
} // namespace Gaudi::Demos
