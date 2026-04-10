#include "Demos.h"

namespace Gaudi::Demos
{
    void SetupDiffusionDecayDemo( DigitalTwin::SimulationBlueprint& blueprint )
    {
        // Field-only demo: a sphere of high concentration diffuses and decays to equilibrium.
        // No cells — watch the substance spread outward and fade.
        blueprint.SetName( "Diffusion & Decay" );
        blueprint.SetDomainSize( glm::vec3( 20.0f ), 1.0f );

        blueprint.ConfigureSpatialPartitioning()
            .SetMethod( DigitalTwin::SpatialPartitioningMethod::HashGrid )
            .SetCellSize( 3.0f )
            .SetMaxDensity( 64 )
            .SetComputeHz( 60.0f );

        // Gaussian peak at origin: smooth gradient from 100 at center to 0 at edges.
        // High diffusion spreads it visibly; strong decay pulls it back to zero.
        blueprint.AddGridField( "Substance" )
            .SetInitializer( DigitalTwin::GridInitializer::Gaussian( glm::vec3( 0.0f ), 3.0f, 100.0f ) )
            .SetDiffusionCoefficient( 3.0f )
            .SetDecayRate( 0.5f )
            .SetComputeHz( 60.0f );
    }
} // namespace Gaudi::Demos
