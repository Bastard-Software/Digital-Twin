#include "Demos.h"

#include <simulation/BiomechanicsGenerator.h>
#include <simulation/MorphologyGenerator.h>
#include <simulation/SpatialDistribution.h>

#include <glm/glm.hpp>

namespace Gaudi::Demos
{
    void SetupEndothelialTubeDemo( DigitalTwin::SimulationBlueprint& blueprint )
    {
        // ~200 VE-cadherin endothelial cells randomly distributed inside a solid cylinder.
        //
        // CurvedTile morphology oriented radially outward from each cell's initial position.
        // Orientations are static (set once), but cells self-organise to the tube surface
        // anyway, so tiles end up pointing outward where they matter.
        //
        // Self-organisation mechanism:
        //   1. CellPolarity shader derives outward direction from neighbor centroid each frame.
        //      Surface cells polarise outward (w → 1); interior cells stay symmetric (w → 0).
        //   2. Polarity-modulated JKR weakens apical-apical (inner) contacts and strengthens
        //      basal-basal (outer) contacts.
        //   3. Net effect: interior contacts become net-repulsive → cavity opens → lumen forms.
        //
        // Parameter choices for self-organisation (vs. stable-shell) dynamics:
        //   Damping 150 (not 500) — cells must be free to rearrange.
        //   Brownian 0.4 — thermal noise helps escape local energy minima.
        //   AdhesionEnergy 2.0 — stronger cohesion prevents the cluster dispersing.
        //   RegulationRate 0.2 — faster polarity adaptation during the initial reorganisation.
        blueprint.SetName( "Endothelial Tube" );
        blueprint.SetDomainSize( glm::vec3( 40.0f ), 1.5f );

        blueprint.ConfigureSpatialPartitioning()
            .SetMethod( DigitalTwin::SpatialPartitioningMethod::HashGrid )
            .SetCellSize( 3.0f )
            .SetMaxDensity( 64 )
            .SetComputeHz( 60.0f );

        // ── Initial placement — random inside solid cylinder ──────────────────
        const float    tubeRadius   = 3.0f;
        const float    halfLength   = 6.0f;
        const uint32_t count        = 200;
        // arcAngle sized for ~18 cells per ring at tubeRadius (circumference/~1.05 per cell)
        const float    arcAngle     = 360.0f / 18.0f;
        const float    axialSpacing = 1.05f;

        auto positions = DigitalTwin::SpatialDistribution::UniformInCylinder(
            count, tubeRadius, halfLength,
            glm::vec3( 0.0f ), glm::vec3( 0.0f, 1.0f, 0.0f ),
            0.0f, 42 );

        // ── Per-cell outward normals from initial position ────────────────────
        // Each tile is oriented so its outer face points radially outward from the Y axis.
        // Cells near the axis (r < 1e-3) fall back to +X to avoid degenerate normals.
        std::vector<glm::vec4> normals;
        normals.reserve( count );
        for( const auto& p : positions )
        {
            glm::vec3 radial( p.x, 0.0f, p.z );
            float     len = glm::length( radial );
            glm::vec3 n   = ( len > 1e-3f ) ? radial / len : glm::vec3( 1.0f, 0.0f, 0.0f );
            normals.push_back( glm::vec4( n, 0.0f ) );
        }

        // ── Biomechanics ───────────────────────────────────────────────────────
        auto jkr = DigitalTwin::BiomechanicsGenerator::JKR()
                       .SetYoungsModulus( 20.0f )
                       .SetPoissonRatio( 0.4f )
                       .SetAdhesionEnergy( 2.0f )        // stronger cohesion during self-org
                       .SetMaxInteractionRadius( 0.75f )
                       .SetDampingCoefficient( 150.0f )  // free to rearrange
                       .Build();

        // ── Agent group ────────────────────────────────────────────────────────
        auto& ecs = blueprint.AddAgentGroup( "Endothelial Cells" )
                        .SetCount( count )
                        .SetMorphology( DigitalTwin::MorphologyGenerator::CreateCurvedTile(
                            arcAngle, axialSpacing, 0.25f, tubeRadius ) )
                        .SetDistribution( positions )
                        .SetOrientations( normals )
                        .SetColor( glm::vec4( 0.2f, 0.75f, 0.55f, 1.0f ) );

        ecs.AddBehaviour( jkr ).SetHz( 60.0f );

        // VE-cadherin (channel z) — homophilic EC–EC adhesion
        ecs.AddBehaviour( DigitalTwin::Behaviours::CadherinAdhesion{
                              glm::vec4( 0.0f, 0.0f, 1.0f, 0.0f ),
                              0.05f,  // expressionRate
                              0.001f, // degradationRate
                              2.0f    // couplingStrength
                          } )
            .SetHz( 60.0f );

        // Apical-basal polarity — drives lumen formation
        DigitalTwin::Behaviours::CellPolarity polarity;
        polarity.regulationRate  = 0.2f;
        polarity.apicalRepulsion = 0.3f;
        polarity.basalAdhesion   = 1.5f;
        ecs.AddBehaviour( polarity ).SetHz( 60.0f );

        // Brownian motion — thermal noise helps cells escape local energy minima
        ecs.AddBehaviour( DigitalTwin::Behaviours::BrownianMotion{ 0.4f } ).SetHz( 60.0f );
    }
} // namespace Gaudi::Demos
