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

        // Multiple Gaussian sources scattered asymmetrically through the domain: distinct
        // peaks diffuse outward, overlap and brighten where they meet, then decay to zero.
        const std::vector<glm::vec3> sources = {
            { -6.0f, -4.0f,  2.0f },
            {  5.0f,  5.0f, -3.0f },
            { -3.0f,  6.0f,  4.0f },
            {  4.0f, -5.0f, -4.0f },
            {  0.0f,  0.0f,  0.0f },
            { -7.0f,  1.0f, -6.0f },
            {  6.0f, -2.0f,  5.0f },
        };
        blueprint.AddGridField( "Substance" )
            .SetInitializer( DigitalTwin::GridInitializer::MultiGaussian( sources, 1.5f, 100.0f ) )
            .SetDiffusionCoefficient( 8.0f )
            .SetDecayRate( 1.5f )
            .SetComputeHz( 60.0f );
    }
} // namespace Gaudi::Demos
