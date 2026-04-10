#include "Demos.h"

#include <simulation/MorphologyGenerator.h>

namespace Gaudi::Demos
{
    void SetupEmptyBlueprint( DigitalTwin::SimulationBlueprint& blueprint )
    {
        blueprint.SetName( "Untitled" );
        blueprint.SetDomainSize( glm::vec3( 30.0f ), 2.0f );
        blueprint.ConfigureSpatialPartitioning()
            .SetMethod( DigitalTwin::SpatialPartitioningMethod::HashGrid )
            .SetCellSize( 3.0f )
            .SetMaxDensity( 64 )
            .SetComputeHz( 60.0f );
    }
} // namespace Gaudi::Demos
