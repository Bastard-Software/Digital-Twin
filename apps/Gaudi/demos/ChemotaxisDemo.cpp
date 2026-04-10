#include "Demos.h"

#include <simulation/BiologyGenerator.h>
#include <simulation/MorphologyGenerator.h>
#include <simulation/Phenotype.h>

namespace Gaudi::Demos
{
    void SetupChemotaxisDemo( DigitalTwin::SimulationBlueprint& blueprint )
    {
        // Two-cell demo:
        //   Source (green) : consumes O2 → goes Hypoxic after ~5s → starts secreting VEGF.
        //   Responder (blue): slowly chemotaxes toward the growing VEGF cloud.
        //
        // Sequence: green stays green → turns purple (Hypoxic) → VEGF cloud grows from it
        //           → blue cell drifts toward green over 20-30 seconds.
        blueprint.SetName( "Chemotaxis Demo" );
        blueprint.SetDomainSize( glm::vec3( 30.0f ), 1.5f );

        blueprint.ConfigureSpatialPartitioning()
            .SetMethod( DigitalTwin::SpatialPartitioningMethod::HashGrid )
            .SetCellSize( 3.0f )
            .SetMaxDensity( 64 )
            .SetComputeHz( 60.0f );

        // O2: tight Gaussian so source depletes its local supply in a few seconds.
        // No decay → limited reservoir; diffusion replenishes slowly from surroundings.
        blueprint.AddGridField( "Oxygen" )
            .SetInitializer( DigitalTwin::GridInitializer::Gaussian( glm::vec3( 0.0f ), 5.0f, 60.0f ) )
            .SetDiffusionCoefficient( 0.5f )
            .SetDecayRate( 0.0f )
            .SetComputeHz( 60.0f );

        // VEGF: starts empty; source secretes once Hypoxic; no decay so cloud grows steadily.
        blueprint.AddGridField( "VEGF" )
            .SetInitializer( DigitalTwin::GridInitializer::Constant( 0.0f ) )
            .SetDiffusionCoefficient( 0.5f )
            .SetDecayRate( 0.0f )
            .SetComputeHz( 60.0f );

        // ── Source cell (green → purple when Hypoxic) ────────────────────────
        auto& source = blueprint.AddAgentGroup( "Source" )
                           .SetCount( 1 )
                           .SetMorphology( DigitalTwin::MorphologyGenerator::CreateSphere( 1.0f ) )
                           .SetDistribution( { glm::vec4( 0.0f, 0.0f, 0.0f, 1.0f ) } )
                           .SetColor( glm::vec4( 0.15f, 0.85f, 0.25f, 1.0f ) ); // green

        // Consume O2 until local concentration drops below hypoxia threshold.
        source.AddBehaviour( DigitalTwin::Behaviours::ConsumeField{ "Oxygen", 15.0f } ).SetHz( 60.0f );

        // CellCycle tracks lifecycle only (no division — doubling time = 999 hours).
        source.AddBehaviour( DigitalTwin::BiologyGenerator::StandardCellCycle()
                                 .SetBaseDoublingTime( 999.0f )
                                 .SetProliferationOxygenTarget( 60.0f )
                                 .SetArrestPressureThreshold( 999.0f )
                                 .SetHypoxiaOxygenThreshold( 30.0f )
                                 .SetNecrosisOxygenThreshold( 1.0f )
                                 .SetApoptosisRate( 0.0f )
                                 .Build() )
            .SetHz( 10.0f );

        // Secrete VEGF only while Hypoxic — starts automatically once lifecycle flips.
        source.AddBehaviour( DigitalTwin::Behaviours::SecreteField{
                                 "VEGF", 200.0f,
                                 DigitalTwin::LifecycleState::Hypoxic } )
            .SetHz( 60.0f );

        // ── Responder cell (blue) ─────────────────────────────────────────────
        auto& responder = blueprint.AddAgentGroup( "Responder" )
                              .SetCount( 1 )
                              .SetMorphology( DigitalTwin::MorphologyGenerator::CreateSphere( 1.0f ) )
                              .SetDistribution( { glm::vec4( 10.0f, 0.0f, 0.0f, 1.0f ) } )
                              .SetColor( glm::vec4( 0.2f, 0.45f, 1.0f, 1.0f ) ); // blue

        // Low sensitivity and max-velocity so the drift is clearly visible but slow.
        responder.AddBehaviour( DigitalTwin::Behaviours::Chemotaxis{ "VEGF", 2.0f, 0.001f, 1.5f } )
            .SetHz( 60.0f );
    }
} // namespace Gaudi::Demos
