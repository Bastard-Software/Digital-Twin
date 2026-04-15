#include "Demos.h"

#include <simulation/BiomechanicsGenerator.h>
#include <simulation/MorphologyGenerator.h>
#include <simulation/SpatialDistribution.h>

namespace Gaudi::Demos
{
    void SetupStressTestDemo( DigitalTwin::SimulationBlueprint& blueprint )
    {
        // GPU stress test: ~100,000 agents across 3 groups in a large domain.
        // Goal: drive compute timings into ms range to produce meaningful profiler data.
        // Target: ~30 FPS on a mid-range discrete GPU.
        //
        // Layout:
        //   Centre (r=60):  Tumour core   — 50,000 cells, JKR + Cadherin + Brownian + Secrete + Consume
        //   Shell  (r=90):  Stromal cells — 45,000 cells, JKR + Brownian + Consume
        //   Outer  (r=110): Immune cells  —  5,000 cells, JKR + Chemotaxis (slow) + Brownian

        blueprint.SetName( "Stress Test" );
        blueprint.SetDomainSize( glm::vec3( 250.0f ), 5.0f );

        blueprint.ConfigureSpatialPartitioning()
            .SetMethod( DigitalTwin::SpatialPartitioningMethod::HashGrid )
            .SetCellSize( 3.0f )
            .SetMaxDensity( 64 )
            .SetComputeHz( 60.0f );

        // VEGF: secreted by tumour, drives immune chemotaxis
        blueprint.AddGridField( "VEGF" )
            .SetInitializer( DigitalTwin::GridInitializer::Constant( 0.0f ) )
            .SetDiffusionCoefficient( 3.0f )
            .SetDecayRate( 0.05f )
            .SetComputeHz( 60.0f );

        // Oxygen: pre-filled, consumed by all groups
        blueprint.AddGridField( "Oxygen" )
            .SetInitializer( DigitalTwin::GridInitializer::Constant( 100.0f ) )
            .SetDiffusionCoefficient( 5.0f )
            .SetDecayRate( 0.0f )
            .SetComputeHz( 60.0f );

        // ── Tumour Core (50,000 agents, red) ─────────────────────────────────────
        {
            auto positions = DigitalTwin::SpatialDistribution::UniformInSphere(
                50000, 60.0f, glm::vec3( 0.0f ) );

            auto& tumour = blueprint.AddAgentGroup( "Tumour" )
                               .SetCount( 50000 )
                               .SetMorphology( DigitalTwin::MorphologyGenerator::CreateSphere( 0.5f ) )
                               .SetDistribution( positions )
                               .SetColor( glm::vec4( 0.85f, 0.15f, 0.10f, 1.0f ) );

            tumour.AddBehaviour( DigitalTwin::BiomechanicsGenerator::JKR()
                                     .SetYoungsModulus( 10.0f ).SetPoissonRatio( 0.4f )
                                     .SetAdhesionEnergy( 1.0f ).SetMaxInteractionRadius( 0.6f )
                                     .SetDampingCoefficient( 200.0f ).Build() )
                .SetHz( 60.0f );

            tumour.AddBehaviour( DigitalTwin::Behaviours::CadherinAdhesion{
                                     glm::vec4( 1.0f, 0.0f, 0.0f, 0.0f ), 2.0f, 0.001f, 3.0f } )
                .SetHz( 60.0f );

            tumour.AddBehaviour( DigitalTwin::Behaviours::BrownianMotion{ 0.05f } )
                .SetHz( 60.0f );

            tumour.AddBehaviour( DigitalTwin::Behaviours::SecreteField{ "VEGF", 50.0f } )
                .SetHz( 60.0f );

            tumour.AddBehaviour( DigitalTwin::Behaviours::ConsumeField{ "Oxygen", 10.0f } )
                .SetHz( 60.0f );
        }

        // ── Stromal Cells (45,000 agents, blue) ──────────────────────────────────
        {
            auto positions = DigitalTwin::SpatialDistribution::UniformInSphere(
                45000, 90.0f, glm::vec3( 0.0f ) );

            auto& stromal = blueprint.AddAgentGroup( "Stromal" )
                                .SetCount( 45000 )
                                .SetMorphology( DigitalTwin::MorphologyGenerator::CreateSphere( 0.5f ) )
                                .SetDistribution( positions )
                                .SetColor( glm::vec4( 0.30f, 0.55f, 0.90f, 1.0f ) );

            stromal.AddBehaviour( DigitalTwin::BiomechanicsGenerator::JKR()
                                      .SetYoungsModulus( 8.0f ).SetPoissonRatio( 0.4f )
                                      .SetAdhesionEnergy( 0.5f ).SetMaxInteractionRadius( 0.6f )
                                      .SetDampingCoefficient( 150.0f ).Build() )
                .SetHz( 60.0f );

            stromal.AddBehaviour( DigitalTwin::Behaviours::BrownianMotion{ 0.1f } )
                .SetHz( 60.0f );

            stromal.AddBehaviour( DigitalTwin::Behaviours::ConsumeField{ "Oxygen", 5.0f } )
                .SetHz( 60.0f );
        }

        // ── Immune Cells (5,000 agents, yellow) — slow drift toward VEGF ─────────
        {
            auto positions = DigitalTwin::SpatialDistribution::UniformInSphere(
                5000, 110.0f, glm::vec3( 0.0f ) );

            auto& immune = blueprint.AddAgentGroup( "Immune" )
                               .SetCount( 5000 )
                               .SetMorphology( DigitalTwin::MorphologyGenerator::CreateSphere( 0.5f ) )
                               .SetDistribution( positions )
                               .SetColor( glm::vec4( 0.95f, 0.85f, 0.15f, 1.0f ) );

            immune.AddBehaviour( DigitalTwin::BiomechanicsGenerator::JKR()
                                     .SetYoungsModulus( 8.0f ).SetPoissonRatio( 0.4f )
                                     .SetAdhesionEnergy( 0.0f ).SetMaxInteractionRadius( 0.5f )
                                     .SetDampingCoefficient( 120.0f ).Build() )
                .SetHz( 60.0f );

            immune.AddBehaviour( DigitalTwin::Behaviours::Chemotaxis{
                                     "VEGF", 0.1f, 0.01f, 2.0f } )  // strength 0.1 — very slow inward drift
                .SetHz( 60.0f );

            immune.AddBehaviour( DigitalTwin::Behaviours::BrownianMotion{ 0.05f } )  // reduced noise
                .SetHz( 60.0f );

            immune.AddBehaviour( DigitalTwin::Behaviours::ConsumeField{ "Oxygen", 3.0f } )
                .SetHz( 60.0f );
        }
    }
} // namespace Gaudi::Demos
