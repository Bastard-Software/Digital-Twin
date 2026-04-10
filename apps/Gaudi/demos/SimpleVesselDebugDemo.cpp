#include "Demos.h"

#include <simulation/BiologyGenerator.h>
#include <simulation/MorphologyGenerator.h>
#include <simulation/Phenotype.h>
#include <simulation/SpatialDistribution.h>

namespace Gaudi::Demos
{
    void SetupSimpleVesselDebugDemo( DigitalTwin::SimulationBlueprint& blueprint )
    {
        // Simple vessel sprouting demo (Guo et al. 2024 biology).
        //
        // 1 source cell at (0,0,0): consumes O2 → Hypoxic → secretes VEGF.
        // 1 vessel of 15 cells at y=+10 (10 units from source).
        //
        // Correct two-step differentiation pathway:
        //   1. PhalanxActivation: VEGF > threshold → PhalanxCell activates to StalkCell
        //      (models VEGF-induced EC activation; only ~3-5 cells nearest source activate)
        //   2. NotchDll4: activated StalkCells compete; winner (highest VEGF) → TipCell
        //   3. Chemotaxis: TipCell migrates toward VEGF source
        //   4. CellCycle (directedMitosis): StalkCell behind TipCell divides → stalk elongates
        //   5. VesselSpring: all non-Phalanx cells tethered — chain spacing maintained
        //
        // Expected sequence: ~t=5s Hypoxic → ~t=10s TipCell selected → migrates → stalk grows.

        blueprint.SetName( "Simple Vessel Debug" );
        blueprint.SetDomainSize( glm::vec3( 40.0f ), 2.0f );

        blueprint.ConfigureSpatialPartitioning()
            .SetMethod( DigitalTwin::SpatialPartitioningMethod::HashGrid )
            .SetCellSize( 3.0f )
            .SetMaxDensity( 64 )
            .SetComputeHz( 60.0f );

        // Source is at (0,0,0), 10 units below the vessel at y=+10.
        // Effective VEGF characteristic length: λ = h*√(D/k) = 2*√(0.5/0.1) = 4.47 units
        // (factor of h²=4 from discrete diffusion — shader does not divide by dx²).
        // At 10 units: ~10% of peak VEGF reaches vessel center, dropping sharply to the sides.

        // Oxygen: sigma=12 ensures vessel at y=+10 gets O2 ≈ 42 (above ProliferationTarget=40).
        // Source at (0,0,0) starts at peak=60, consumes at 10/s → below 30 → Hypoxic.
        blueprint.AddGridField( "Oxygen" )
            .SetInitializer( DigitalTwin::GridInitializer::Gaussian( glm::vec3( 0.0f, 0.0f, 0.0f ), 12.0f, 60.0f ) )
            .SetDiffusionCoefficient( 2.0f )
            .SetDecayRate( 0.0f )
            .SetComputeHz( 60.0f );

        // VEGF: decay=0.1 (λ=4.47 effective units), rate=300 reaches vessel from 10 units.
        blueprint.AddGridField( "VEGF" )
            .SetInitializer( DigitalTwin::GridInitializer::Constant( 0.0f ) )
            .SetDiffusionCoefficient( 0.5f )
            .SetDecayRate( 0.1f )
            .SetComputeHz( 60.0f );

        // --- Source cell 10 units below vessel at (0,0,0) ---
        auto& source = blueprint.AddAgentGroup( "SourceCell" )
                           .SetCount( 1 )
                           .SetMorphology( DigitalTwin::MorphologyGenerator::CreateSphere( 1.2f ) )
                           .SetDistribution( { glm::vec4( 0.0f, 0.0f, 0.0f, 1.0f ) } )
                           .SetColor( glm::vec4( 1.0f, 0.5f, 0.1f, 1.0f ) ); // orange → turns purple when hypoxic

        source.AddBehaviour( DigitalTwin::Behaviours::ConsumeField{ "Oxygen", 10.0f } ).SetHz( 60.0f );
        source.AddBehaviour( DigitalTwin::BiologyGenerator::StandardCellCycle()
                                 .SetBaseDoublingTime( 99.0f )          // no division
                                 .SetProliferationOxygenTarget( 60.0f )
                                 .SetArrestPressureThreshold( 999.0f )
                                 .SetHypoxiaOxygenThreshold( 30.0f )    // goes hypoxic quickly
                                 .SetNecrosisOxygenThreshold( 1.0f )
                                 .SetApoptosisRate( 0.0f )
                                 .Build() )
            .SetHz( 10.0f );
        source.AddBehaviour( DigitalTwin::Behaviours::SecreteField{
                                 "VEGF", 300.0f,
                                 DigitalTwin::LifecycleState::Hypoxic } )
            .SetHz( 60.0f );

        // --- Vessel: 15 cells at y=+10 ---
        auto vesselPos = DigitalTwin::SpatialDistribution::VesselLine( 15, glm::vec3( -14, 10, 0 ), glm::vec3( 14, 10, 0 ) );

        auto& vessel = blueprint.AddAgentGroup( "VesselCells" )
                           .SetCount( 15 )
                           .SetMorphology( DigitalTwin::MorphologyGenerator::CreateCylinder( 0.5f, 1.8f ) )
                           .SetDistribution( vesselPos )
                           .SetColor( glm::vec4( 0.5f, 0.15f, 0.15f, 1.0f ) )               // dark red base
                           .SetInitialCellType( static_cast<int>( DigitalTwin::CellType::PhalanxCell ) )
                           .AddCellTypeMorphology( static_cast<int>( DigitalTwin::CellType::PhalanxCell ),
                               DigitalTwin::MorphologyGenerator::CreateCylinder( 0.5f, 1.8f ),
                               glm::vec4( 0.5f, 0.15f, 0.15f, 1.0f ) )   // dark red — quiescent
                           .AddCellTypeMorphology( static_cast<int>( DigitalTwin::CellType::StalkCell ),
                               DigitalTwin::MorphologyGenerator::CreateCylinder( 0.6f, 2.2f ),
                               glm::vec4( 1.0f, 0.8f, 0.1f, 1.0f ) )     // yellow — activated
                           .AddCellTypeMorphology( static_cast<int>( DigitalTwin::CellType::TipCell ),
                               DigitalTwin::MorphologyGenerator::CreateSpikySphere( 1.0f, 1.35f ),
                               glm::vec4( 0.0f, 1.0f, 0.3f, 1.0f ) );    // bright green — TipCell

        // 1. VesselSeed: wire up 15 cells into a linear chain
        vessel.AddBehaviour( DigitalTwin::Behaviours::VesselSeed{ std::vector<uint32_t>{ 15u } } );

        // 2. PhalanxActivation: spatial gate — only cells with VEGF > threshold activate.
        //    Threshold=20 targets center 3-5 cells. ConsumeField (below) limits lateral spread.
        //    Tune threshold by checking VEGF inspector after source goes Hypoxic.
        vessel.AddBehaviour( DigitalTwin::Behaviours::PhalanxActivation{
                /* vegfFieldName         */ "VEGF",
                /* activationThreshold   */ 20.0f,
                /* deactivationThreshold */ 5.0f } )
            .SetHz( 60.0f );

        // 3. ConsumeField: VEGFR1 (stalk-enriched decoy receptor) — VEGF sink shapes the gradient.
        //    Rate=2.0 (strong sink) prevents lateral VEGF creep → stops multi-TipCell activation.
        vessel.AddBehaviour( DigitalTwin::Behaviours::ConsumeField{ "VEGF", 2.0f } )
            .SetRequiredCellType( DigitalTwin::CellType::StalkCell )
            .SetHz( 60.0f );

        // 4. NotchDll4: lateral inhibition among activated StalkCells → 1 TipCell.
        vessel.AddBehaviour( DigitalTwin::Behaviours::NotchDll4{
                /* dll4ProductionRate   */ 1.0f,
                /* dll4DecayRate        */ 0.1f,
                /* notchInhibitionGain  */ 20.0f,
                /* vegfr2BaseExpression */ 1.0f,
                /* tipThreshold         */ 0.90f,
                /* stalkThreshold       */ 0.20f,
                /* vegfFieldName        */ "VEGF",
                /* subSteps             */ 20u } )
            .SetHz( 60.0f );

        // 5. CellCycle: StalkCell in proliferation zone (adjacent to TipCell) divides → stalk elongates.
        //    TipCell guard in mitosis_vessel_append.comp prevents TipCells from dividing.
        auto stalkCycle = DigitalTwin::BiologyGenerator::StandardCellCycle()
                              .SetBaseDoublingTime( 6.0f / 3600.0f )  // 6 seconds per doubling
                              .SetProliferationOxygenTarget( 40.0f )
                              .SetArrestPressureThreshold( 5.0f )
                              .SetHypoxiaOxygenThreshold( 5.0f )
                              .SetNecrosisOxygenThreshold( 1.0f )
                              .SetApoptosisRate( 0.0f )
                              .Build();
        stalkCycle.directedMitosis = true;
        vessel.AddBehaviour( stalkCycle ).SetHz( 10.0f );

        // 6. Chemotaxis: TipCell follows VEGF gradient toward source.
        vessel.AddBehaviour( DigitalTwin::Behaviours::Chemotaxis{ "VEGF", 6.0f, 0.002f, 4.0f, 3.0f } )
            .SetHz( 60.0f )
            .SetRequiredCellType( DigitalTwin::CellType::TipCell );

        // 7. VesselSpring: all non-Phalanx cells get spring forces — chain spacing maintained.
        //    StalkCell daughters from directed mitosis need springs to prevent blobbing.
        //    PhalanxCells are anchored by vessel_mechanics.comp hardcode (line 48), not by this filter.
        vessel.AddBehaviour( DigitalTwin::Behaviours::VesselSpring{ 15.0f, 1.5f, 10.0f } )
            .SetHz( 60.0f );
    }
} // namespace Gaudi::Demos
