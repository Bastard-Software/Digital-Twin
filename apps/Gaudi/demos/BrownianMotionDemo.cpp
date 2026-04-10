#include "Demos.h"

#include <simulation/MorphologyGenerator.h>

#include <vector>

namespace Gaudi::Demos
{
    void SetupBrownianMotionDemo( DigitalTwin::SimulationBlueprint& blueprint )
    {
        // 27 cells arranged in a 3x3x3 grid perform pure thermal random-walk.
        // No fields, no forces — each cell drifts independently.
        blueprint.SetName( "Brownian Motion" );
        blueprint.SetDomainSize( glm::vec3( 20.0f ), 1.0f );

        blueprint.ConfigureSpatialPartitioning()
            .SetMethod( DigitalTwin::SpatialPartitioningMethod::HashGrid )
            .SetCellSize( 3.0f )
            .SetMaxDensity( 64 )
            .SetComputeHz( 60.0f );

        // 3x3x3 grid of cells, 5-unit spacing → fits within 20-unit domain
        std::vector<glm::vec4> positions;
        for( int x = -1; x <= 1; ++x )
            for( int y = -1; y <= 1; ++y )
                for( int z = -1; z <= 1; ++z )
                    positions.push_back( glm::vec4( x * 5.0f, y * 5.0f, z * 5.0f, 1.0f ) );

        auto& cells = blueprint.AddAgentGroup( "Particles" )
                          .SetCount( static_cast<uint32_t>( positions.size() ) )
                          .SetMorphology( DigitalTwin::MorphologyGenerator::CreateSphere( 0.5f ) )
                          .SetDistribution( positions )
                          .SetColor( glm::vec4( 0.9f, 0.6f, 0.1f, 1.0f ) ); // amber

        cells.AddBehaviour( DigitalTwin::Behaviours::BrownianMotion{ 2.0f } ).SetHz( 60.0f );
    }
} // namespace Gaudi::Demos
