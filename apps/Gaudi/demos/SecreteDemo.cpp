#include "Demos.h"

#include <simulation/MorphologyGenerator.h>

namespace Gaudi::Demos
{
    void SetupSecreteDemo( DigitalTwin::SimulationBlueprint& blueprint )
    {
        // One cell pumps substance into an empty field.
        // Goal: clearly see field growing outward from the cell.
        blueprint.SetName( "Secrete Demo" );
        blueprint.SetDomainSize( glm::vec3( 20.0f ), 1.0f );

        blueprint.ConfigureSpatialPartitioning()
            .SetMethod( DigitalTwin::SpatialPartitioningMethod::HashGrid )
            .SetCellSize( 3.0f )
            .SetMaxDensity( 64 )
            .SetComputeHz( 60.0f );

        // Field starts empty; no decay so substance accumulates and spreads visibly.
        blueprint.AddGridField( "Substance" )
            .SetInitializer( DigitalTwin::GridInitializer::Constant( 0.0f ) )
            .SetDiffusionCoefficient( 2.5f )
            .SetDecayRate( 0.0f )
            .SetComputeHz( 60.0f );

        auto& cells = blueprint.AddAgentGroup( "SecretingCell" )
                          .SetCount( 1 )
                          .SetMorphology( DigitalTwin::MorphologyGenerator::CreateSphere( 1.0f ) )
                          .SetDistribution( { glm::vec4( 0.0f, 0.0f, 0.0f, 1.0f ) } )
                          .SetColor( glm::vec4( 0.2f, 0.6f, 1.0f, 1.0f ) ); // blue

        // Rate=500 fills the peak voxel quickly — clearly visible against normalization.
        cells.AddBehaviour( DigitalTwin::Behaviours::SecreteField{ "Substance", 1000.0f } )
            .SetHz( 60.0f );
    }
} // namespace Gaudi::Demos
