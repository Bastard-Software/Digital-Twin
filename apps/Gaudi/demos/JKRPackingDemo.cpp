#include "Demos.h"

#include <simulation/BiomechanicsGenerator.h>
#include <simulation/MorphologyGenerator.h>
#include <simulation/SpatialDistribution.h>

namespace Gaudi::Demos
{
    void SetupJKRPackingDemo( DigitalTwin::SimulationBlueprint& blueprint )
    {
        // 40 cells packed into a tight sphere at origin — pure Hertz repulsion pushes them
        // outward into a stable close-packing arrangement. High damping slows the expansion
        // so you can watch the cluster bloom over ~10 seconds.
        blueprint.SetName( "JKR Packing" );
        blueprint.SetDomainSize( glm::vec3( 20.0f ), 1.0f );

        blueprint.ConfigureSpatialPartitioning()
            .SetMethod( DigitalTwin::SpatialPartitioningMethod::HashGrid )
            .SetCellSize( 3.0f )
            .SetMaxDensity( 64 )
            .SetComputeHz( 60.0f );

        auto positions = DigitalTwin::SpatialDistribution::UniformInSphere( 40, 0.5f );

        auto& cells = blueprint.AddAgentGroup( "Cells" )
                          .SetCount( 40 )
                          .SetMorphology( DigitalTwin::MorphologyGenerator::CreateSphere( 0.8f ) )
                          .SetDistribution( positions )
                          .SetColor( glm::vec4( 0.3f, 0.7f, 0.95f, 1.0f ) ); // cyan-blue

        // Damping formula in shader: displacement /= (1 + damping * dt). With dt=1/60:
        //   damping=10000 → divide by ~168 → expansion at ~1 unit/s, settling over ~10s.
        cells.AddBehaviour( DigitalTwin::BiomechanicsGenerator::JKR()
                                .SetYoungsModulus( 5.0f )
                                .SetPoissonRatio( 0.4f )
                                .SetAdhesionEnergy( 0.0f )
                                .SetMaxInteractionRadius( 1.5f )
                                .SetDampingCoefficient( 250.0f )
                                .Build() )
            .SetHz( 60.0f );
    }
} // namespace Gaudi::Demos
