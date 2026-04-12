#include "Demos.h"

#include <simulation/BiomechanicsGenerator.h>
#include <simulation/MorphologyGenerator.h>

#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>
#include <cmath>

namespace Gaudi::Demos
{
    void SetupECContactDemo( DigitalTwin::SimulationBlueprint& blueprint )
    {
        // 5 flat endothelial tiles in a 2D cross.
        // Outer tiles start at different Y rotations to stress the self-alignment.
        // A cadherin-driven edge alignment torque (cross(myEdge, nbrEdge)) fires once
        // VE-cadherin is expressed and is always restoring regardless of initial rotation
        // direction or magnitude.  All 4 outer tiles converge to edge-to-edge contact
        // within ~10 s regardless of starting angle.
        blueprint.SetName( "EC Contact" );
        blueprint.SetDomainSize( glm::vec3( 12.0f ), 1.5f );

        blueprint.ConfigureSpatialPartitioning()
            .SetMethod( DigitalTwin::SpatialPartitioningMethod::HashGrid )
            .SetCellSize( 3.0f )
            .SetMaxDensity( 64 )
            .SetComputeHz( 60.0f );

        const float tileSize = 1.4f;
        const float spacing  = 1.47f;

        // Helper: quaternion for rotation of angleDeg around Y.
        auto qY = [&]( float angleDeg ) -> glm::vec4 {
            float h = angleDeg * glm::pi<float>() / 360.0f; // half-angle in radians
            return glm::vec4( 0.0f, std::sin( h ), 0.0f, std::cos( h ) );
        };

        std::vector<glm::vec4> positions = {
            glm::vec4(  0.0f,    0.0f,  0.0f,    1.0f ),  // center
            glm::vec4(  spacing, 0.0f,  0.0f,    1.0f ),  // right
            glm::vec4( -spacing, 0.0f,  0.0f,    1.0f ),  // left
            glm::vec4(  0.0f,    0.0f,  spacing, 1.0f ),  // front
            glm::vec4(  0.0f,    0.0f, -spacing, 1.0f ),  // back
        };

        // Varied initial rotations — each outer tile starts at a different angle.
        // The edge alignment torque drives all of them to 0 deg independently.
        std::vector<glm::vec4> orientations = {
            glm::vec4( 0.0f, 0.0f, 0.0f, 1.0f ),  // center: identity
            qY( +15.0f ),                            // right:  +15 deg Y
            qY( -12.0f ),                            // left:   -12 deg Y
            qY( +20.0f ),                            // front:  +20 deg Y
            qY(  -8.0f ),                            // back:    -8 deg Y
        };

        auto jkr = DigitalTwin::BiomechanicsGenerator::JKR()
                       .SetYoungsModulus( 20.0f )
                       .SetPoissonRatio( 0.4f )
                       .SetAdhesionEnergy( 2.0f )
                       .SetMaxInteractionRadius( 0.75f )
                       .SetDampingCoefficient( 200.0f )
                       .Build();

        auto& ecs = blueprint.AddAgentGroup( "Endothelial Cells" )
                        .SetCount( 5 )
                        .SetMorphology( DigitalTwin::MorphologyGenerator::CreateTile(
                            tileSize, tileSize, 0.2f ) )
                        .SetDistribution( positions )
                        .SetOrientations( orientations )
                        .SetColor( glm::vec4( 0.2f, 0.75f, 0.55f, 1.0f ) );

        ecs.AddBehaviour( jkr ).SetHz( 60.0f );

        ecs.AddBehaviour( DigitalTwin::Behaviours::CadherinAdhesion{
                              glm::vec4( 0.0f, 0.0f, 1.0f, 0.0f ),
                              0.025f,  // expressionRate — ramp (~35s to 50%); sole timescale for both rotation and adhesion
                              0.001f, // degradationRate
                              2.0f    // couplingStrength
                          } )
            .SetHz( 60.0f );
    }
} // namespace Gaudi::Demos
