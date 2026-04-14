#include "Demos.h"

#include <simulation/BiomechanicsGenerator.h>
#include <simulation/MorphologyGenerator.h>
#include <simulation/SpatialDistribution.h>

#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>
#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

namespace Gaudi::Demos
{
    void SetupCellMechanicsZooDemo( DigitalTwin::SimulationBlueprint& blueprint )
    {
        // Immune transmigration through an endothelial wall into a mixed tumour.
        //
        // Layout (Z axis):
        //   Z=-10  Immune cells (Sphere, yellow) -- chemotaxis toward tumour
        //   Z=  0  EC wall (Tile, green) -- vertical VE-cadherin barrier
        //   Z=+10  Tumour mass (SpikySphere + Cube + Ellipsoid + Sphere, mixed)
        //
        // Sequence:
        //   t=0      Pre-seeded chemokine gradient; immune cells start migrating.
        //   t=5-15s  Immune cells reach EC wall, push tiles apart (diapedesis).
        //   t=15+    VE-cadherin pulls tiles back into an aligned wall.
        //   t=15-30s Immune cells infiltrate tumour, colliding with 4 different shapes.
        //
        // Cross-group adhesion: all tumour groups use E-cadherin (cohesive mass).
        // EC wall uses VE-cadherin (self-healing barrier). Immune cells have no
        // cadherin -> adhScale=0 for all cross-interactions -> pure repulsion -> bounce.

        blueprint.SetName( "Cell Mechanics Zoo" );
        blueprint.SetDomainSize( glm::vec3( 60.0f ), 1.5f );

        blueprint.ConfigureSpatialPartitioning()
            .SetMethod( DigitalTwin::SpatialPartitioningMethod::HashGrid )
            .SetCellSize( 3.0f )
            .SetMaxDensity( 64 )
            .SetComputeHz( 60.0f );

        // Chemokine field -- pre-seeded Gaussian so immune cells migrate immediately.
        // Tumour secretion sustains and sharpens the gradient over time.
        blueprint.AddGridField( "Chemokine" )
            .SetInitializer( DigitalTwin::GridInitializer::Gaussian(
                glm::vec3( 0.0f, 0.0f, 10.0f ), 10.0f, 200.0f ) )
            .SetDiffusionCoefficient( 3.0f )
            .SetDecayRate( 0.02f )
            .SetComputeHz( 60.0f );

        // ── Quaternion helpers ────────────────────────────────────────────────────────
        auto qY = []( float angleDeg ) -> glm::vec4 {
            float h = angleDeg * glm::pi<float>() / 360.0f;
            return glm::vec4( 0.0f, std::sin( h ), 0.0f, std::cos( h ) );
        };
        auto qX = []( float angleDeg ) -> glm::vec4 {
            float h = angleDeg * glm::pi<float>() / 360.0f;
            return glm::vec4( std::sin( h ), 0.0f, 0.0f, std::cos( h ) );
        };
        auto qmul = []( glm::vec4 a, glm::vec4 b ) -> glm::vec4 {
            return glm::vec4(
                a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
                a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
                a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
                a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z );
        };

        // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        // TUMOUR MASS (4 groups, centered at Z=+10)
        // Lattice positions shuffled and split into 4 morphology groups.
        // All share E-cadherin (channel x) so the whole tumour is cohesive.
        // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        {
            auto allPos = DigitalTwin::SpatialDistribution::LatticeInSphere(
                1.0f, 4.5f, glm::vec3( 0.0f, 0.0f, 10.0f ) );
            std::mt19937 rng( 42 );
            std::shuffle( allPos.begin(), allPos.end(), rng );

            uint32_t total   = static_cast<uint32_t>( allPos.size() );
            uint32_t quarter = total / 4;

            auto slice = [&]( uint32_t from, uint32_t to ) {
                return std::vector<glm::vec4>( allPos.begin() + from,
                                               allPos.begin() + std::min( to, total ) );
            };

            // E-cadherin shared by all 4 tumour groups.
            // expressionRate=100 => profile reaches max in 1 frame, so adhesion
            // is active before the single frame of initial repulsion can scatter cells.
            auto cadherinEcad = DigitalTwin::Behaviours::CadherinAdhesion{
                glm::vec4( 1.0f, 0.0f, 0.0f, 0.0f ), 100.0f, 0.001f, 3.0f };

            // ── Tumour Spiky (SpikySphere, dark red) ──────────────────────────────────
            {
                auto pos = slice( 0, quarter );
                auto& grp = blueprint.AddAgentGroup( "Tumour Spiky" )
                                .SetCount( quarter )
                                .SetMorphology( DigitalTwin::MorphologyGenerator::CreateSpikySphere( 0.357f, 1.4f ) )
                                .SetDistribution( pos )
                                .SetColor( glm::vec4( 0.85f, 0.15f, 0.10f, 1.0f ) );

                grp.AddBehaviour( DigitalTwin::BiomechanicsGenerator::JKR()
                                      .SetYoungsModulus( 10.0f ).SetPoissonRatio( 0.4f )
                                      .SetAdhesionEnergy( 1.25f ).SetMaxInteractionRadius( 0.70f )
                                      .SetDampingCoefficient( 200.0f ).Build() )
                    .SetHz( 60.0f );
                grp.AddBehaviour( cadherinEcad ).SetHz( 60.0f );
                grp.AddBehaviour( DigitalTwin::Behaviours::SecreteField{ "Chemokine", 100.0f } )
                    .SetHz( 60.0f );
                grp.AddBehaviour( DigitalTwin::Behaviours::BrownianMotion{ 0.1f } )
                    .SetHz( 60.0f );
            }

            // ── Tumour Cubes (Cube, orange) ───────────────────────────────────────────
            {
                auto pos = slice( quarter, 2 * quarter );
                auto& grp = blueprint.AddAgentGroup( "Tumour Cubes" )
                                .SetCount( quarter )
                                .SetMorphology( DigitalTwin::MorphologyGenerator::CreateCube( 0.6f ) )
                                .SetDistribution( pos )
                                .SetColor( glm::vec4( 0.95f, 0.55f, 0.15f, 1.0f ) );

                grp.AddBehaviour( DigitalTwin::BiomechanicsGenerator::JKR()
                                      .SetYoungsModulus( 10.0f ).SetPoissonRatio( 0.4f )
                                      .SetAdhesionEnergy( 1.25f ).SetMaxInteractionRadius( 0.70f )
                                      .SetDampingCoefficient( 200.0f ).Build() )
                    .SetHz( 60.0f );
                grp.AddBehaviour( cadherinEcad ).SetHz( 60.0f );
                grp.AddBehaviour( DigitalTwin::Behaviours::SecreteField{ "Chemokine", 100.0f } )
                    .SetHz( 60.0f );
                grp.AddBehaviour( DigitalTwin::Behaviours::BrownianMotion{ 0.1f } )
                    .SetHz( 60.0f );
            }

            // ── Tumour Stromal (Ellipsoid, blue) ──────────────────────────────────────
            {
                auto pos = slice( 2 * quarter, 3 * quarter );
                auto& grp = blueprint.AddAgentGroup( "Tumour Stromal" )
                                .SetCount( quarter )
                                .SetMorphology( DigitalTwin::MorphologyGenerator::CreateEllipsoid( 0.35f, 0.5f ) )
                                .SetDistribution( pos )
                                .SetColor( glm::vec4( 0.30f, 0.50f, 0.90f, 1.0f ) );

                grp.AddBehaviour( DigitalTwin::BiomechanicsGenerator::JKR()
                                      .SetYoungsModulus( 10.0f ).SetPoissonRatio( 0.4f )
                                      .SetAdhesionEnergy( 1.25f ).SetMaxInteractionRadius( 0.70f )
                                      .SetDampingCoefficient( 200.0f ).Build() )
                    .SetHz( 60.0f );
                grp.AddBehaviour( cadherinEcad ).SetHz( 60.0f );
                grp.AddBehaviour( DigitalTwin::Behaviours::SecreteField{ "Chemokine", 100.0f } )
                    .SetHz( 60.0f );
                grp.AddBehaviour( DigitalTwin::Behaviours::BrownianMotion{ 0.1f } )
                    .SetHz( 60.0f );
            }

            // ── Tumour Round (Sphere, purple) — cohesive part of tumour mass ─────────
            {
                uint32_t remainder = total - 3 * quarter;
                auto     pos       = slice( 3 * quarter, total );
                auto& grp = blueprint.AddAgentGroup( "Tumour Round" )
                                .SetCount( remainder )
                                .SetMorphology( DigitalTwin::MorphologyGenerator::CreateSphere( 0.5f ) )
                                .SetDistribution( pos )
                                .SetColor( glm::vec4( 0.65f, 0.20f, 0.55f, 1.0f ) );

                grp.AddBehaviour( DigitalTwin::BiomechanicsGenerator::JKR()
                                      .SetYoungsModulus( 10.0f ).SetPoissonRatio( 0.4f )
                                      .SetAdhesionEnergy( 1.25f ).SetMaxInteractionRadius( 0.70f )
                                      .SetDampingCoefficient( 200.0f ).Build() )
                    .SetHz( 60.0f );
                grp.AddBehaviour( cadherinEcad ).SetHz( 60.0f );
                grp.AddBehaviour( DigitalTwin::Behaviours::SecreteField{ "Chemokine", 100.0f } )
                    .SetHz( 60.0f );
                grp.AddBehaviour( DigitalTwin::Behaviours::BrownianMotion{ 0.1f } )
                    .SetHz( 60.0f );
            }
        }

        // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        // EC WALL (Tile, green, Z=0)
        // 4x3 grid of vertical tiles forming a vessel-wall barrier.
        // Tiles rotated 90 deg around X so flat faces point +/-Z (toward immune).
        // Small Y perturbations let edge-alignment torque demonstrate convergence.
        // VE-cadherin reforms the wall after immune cells push through.
        // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        {
            const float   tileSpacing = 1.47f;
            const glm::vec4 baseQ     = qX( 90.0f ); // stand tiles vertical (normal -> +Z)

            std::vector<glm::vec4> positions;
            std::vector<glm::vec4> orientations;

            const float perturb[] = { 0.0f, +3.0f, -2.0f, +1.5f,
                                      -3.0f, +2.0f, -1.0f, +2.5f,
                                      +1.0f, -2.5f, +3.0f, -1.5f };
            int idx = 0;
            for( int iy = -1; iy <= 1; ++iy )     // 3 rows vertical
            {
                for( int ix = -1; ix <= 2; ++ix )  // 4 columns horizontal
                {
                    positions.push_back( glm::vec4(
                        ( ix - 0.5f ) * tileSpacing, // center the 4 columns
                        iy * tileSpacing,
                        0.0f, 1.0f ) );
                    orientations.push_back( qmul( qY( perturb[ idx ] ), baseQ ) );
                    ++idx;
                }
            }

            auto& ecs = blueprint.AddAgentGroup( "EC Wall" )
                            .SetCount( 12 )
                            .SetMorphology( DigitalTwin::MorphologyGenerator::CreateTile(
                                1.4f, 1.4f, 0.2f ) )
                            .SetDistribution( positions )
                            .SetOrientations( orientations )
                            .SetColor( glm::vec4( 0.2f, 0.75f, 0.55f, 1.0f ) );

            ecs.AddBehaviour( DigitalTwin::BiomechanicsGenerator::JKR()
                                  .SetYoungsModulus( 20.0f ).SetPoissonRatio( 0.4f )
                                  .SetAdhesionEnergy( 2.0f ).SetMaxInteractionRadius( 0.75f )
                                  .SetDampingCoefficient( 200.0f ).Build() )
                .SetHz( 60.0f );

            ecs.AddBehaviour( DigitalTwin::Behaviours::CadherinAdhesion{
                                  glm::vec4( 0.0f, 0.0f, 1.0f, 0.0f ), // VE-cadherin
                                  0.025f, 0.001f, 2.0f } )
                .SetHz( 60.0f );
        }

        // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        // IMMUNE CELLS (Sphere, yellow, Z=-10)
        // Chemotaxis toward tumour chemokine; zero adhesion -> bounces off EC wall
        // and all tumour cell types. Brownian jitter models active random search.
        // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        {
            auto positions = DigitalTwin::SpatialDistribution::UniformInSphere(
                10, 1.5f, glm::vec3( 0.0f, 0.0f, -10.0f ) );

            auto& immune = blueprint.AddAgentGroup( "Immune Cells" )
                               .SetCount( 10 )
                               .SetMorphology( DigitalTwin::MorphologyGenerator::CreateSphere( 0.5f ) )
                               .SetDistribution( positions )
                               .SetColor( glm::vec4( 0.95f, 0.85f, 0.15f, 1.0f ) );

            immune.AddBehaviour( DigitalTwin::BiomechanicsGenerator::JKR()
                                     .SetYoungsModulus( 8.0f ).SetPoissonRatio( 0.4f )
                                     .SetAdhesionEnergy( 0.0f ).SetMaxInteractionRadius( 0.5f )
                                     .SetDampingCoefficient( 120.0f ).Build() )
                .SetHz( 60.0f );

            immune.AddBehaviour( DigitalTwin::Behaviours::Chemotaxis{
                                     "Chemokine", 5.0f, 0.01f, 2.0f } )
                .SetHz( 60.0f );

            immune.AddBehaviour( DigitalTwin::Behaviours::BrownianMotion{ 0.3f } )
                .SetHz( 60.0f );
        }
    }
} // namespace Gaudi::Demos
