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
        // 5 endothelial cells on a cylinder surface (r=3) in a cross pattern.
        //
        // Tile dimensions match the full-cadherin JKR equilibrium (~1.39),
        // but cells start at the no-cadherin equilibrium (~1.47).
        //
        // Expected sequence:
        //   t=0:  cadherin expression = 0 → adhForce *= 0 → pure repulsion
        //         cells sit at ~1.47 spacing, gap ~0.1 between tile edges
        //   t>0:  VE-cadherin ramps up (expressionRate 0.2) → adhesion grows
        //         cells pulled inward until gap closes at ~1.39 spacing
        //
        // JKR equilibrium (no cadherin): repulsion=11.9, adhesion=2.0
        //   overlap = (2/11.9)² ≈ 0.028  →  distance ≈ 1.47
        //
        // JKR equilibrium (full cadherin, coupling=2): effective adhesion=4.0
        //   overlap = (4/11.9)² ≈ 0.113  →  distance ≈ 1.39
        //
        // Tile arc width at r=3: 26° → 2π×3×(26/360) ≈ 1.36 ≈ equil − 0.03
        blueprint.SetName( "EC Contact" );
        blueprint.SetDomainSize( glm::vec3( 20.0f ), 1.5f );

        blueprint.ConfigureSpatialPartitioning()
            .SetMethod( DigitalTwin::SpatialPartitioningMethod::HashGrid )
            .SetCellSize( 3.0f )
            .SetMaxDensity( 64 )
            .SetComputeHz( 60.0f );

        const float tubeRadius   = 3.0f;
        const float arcAngle     = 26.0f;                  // tile width ≈ 1.36
        const float tileHeight   = 1.36f;                  // match circumferential width
        const float initAngle    = 28.0f * glm::pi<float>() / 180.0f; // initial spacing ≈ 1.47
        const float initAxial    = 1.47f;                  // initial axial spacing

        // ── Hand-placed cross: center + 2 circumferential + 2 axial ──────────
        struct Cell { float angle; float axialY; };
        Cell cells[] = {
            {  0.0f,        0.0f },         // center
            {  initAngle,   0.0f },         // right
            { -initAngle,   0.0f },         // left
            {  0.0f,        initAxial },    // top
            {  0.0f,       -initAxial },    // bottom
        };

        std::vector<glm::vec4> positions;
        std::vector<glm::vec4> normals;

        for( const auto& c : cells )
        {
            float cx = tubeRadius * std::cos( c.angle );
            float cz = tubeRadius * std::sin( c.angle );
            positions.push_back( glm::vec4( cx, c.axialY, cz, 1.0f ) );

            glm::vec3 n = glm::normalize( glm::vec3( cx, 0.0f, cz ) );
            normals.push_back( glm::vec4( n, 0.0f ) );
        }

        // ── Biomechanics ───────────────────────────────────────────────────────
        auto jkr = DigitalTwin::BiomechanicsGenerator::JKR()
                       .SetYoungsModulus( 20.0f )
                       .SetPoissonRatio( 0.4f )
                       .SetAdhesionEnergy( 2.0f )
                       .SetMaxInteractionRadius( 0.75f )
                       .SetDampingCoefficient( 200.0f )
                       .Build();

        // ── Agent group ────────────────────────────────────────────────────────
        auto& ecs = blueprint.AddAgentGroup( "Endothelial Cells" )
                        .SetCount( 5 )
                        .SetMorphology( DigitalTwin::MorphologyGenerator::CreateCurvedTile(
                            arcAngle, tileHeight, 0.25f, tubeRadius ) )
                        .SetDistribution( positions )
                        .SetOrientations( normals )
                        .SetColor( glm::vec4( 0.2f, 0.75f, 0.55f, 1.0f ) );

        ecs.AddBehaviour( jkr ).SetHz( 60.0f );

        // VE-cadherin (channel z) — fast ramp for visible gap-close
        ecs.AddBehaviour( DigitalTwin::Behaviours::CadherinAdhesion{
                              glm::vec4( 0.0f, 0.0f, 1.0f, 0.0f ),
                              0.2f,   // expressionRate — fast ramp for demo visibility
                              0.001f, // degradationRate
                              2.0f    // couplingStrength
                          } )
            .SetHz( 60.0f );
    }
} // namespace Gaudi::Demos
