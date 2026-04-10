#include "Demos.h"

#include <simulation/BiologyGenerator.h>
#include <simulation/MorphologyGenerator.h>
#include <simulation/Phenotype.h>
#include <simulation/VesselTreeGenerator.h>

namespace Gaudi::Demos
{
    void SetupVesselSproutingDemo( DigitalTwin::SimulationBlueprint& blueprint )
    {
        // Single vessel sprout demo — Phase 3.
        //
        // A straight 2D tube vessel (6-cell rings, no branches) sits 10 units above
        // a hypoxic source cell. The biology pipeline drives sprouting from the tube:
        //
        //   1. Source consumes O2 → Hypoxic → secretes VEGF (~t=5s)
        //   2. PhalanxActivation: VEGF > 20 wakes cells closest to source (~t=10s)
        //   3. NotchDll4: lateral inhibition selects exactly 1 TipCell from the ring (~t=12s)
        //   4. Chemotaxis: TipCell migrates toward source down the VEGF gradient
        //   5. Directed mitosis: StalkCell behind TipCell divides → 1D sprout chain extends
        //   6. Ring cells with only ring edges to TipCell mature back to PhalanxCell
        //      (edge-flag filtering in mitosis_vessel_append.comp)
        //
        // Tube geometry: ringSize=6, length=24, spacing=1.5 → ~17 rings × 6 = ~102 cells.
        // Origin at (-12, 10, 0), direction +x → tube center at (0, 10, 0), 10 units above source.

        blueprint.SetName( "Vessel Sprouting" );
        blueprint.SetDomainSize( glm::vec3( 60.0f ), 2.0f );

        blueprint.ConfigureSpatialPartitioning()
            .SetMethod( DigitalTwin::SpatialPartitioningMethod::HashGrid )
            .SetCellSize( 3.0f )
            .SetMaxDensity( 64 )
            .SetComputeHz( 60.0f );

        // Oxygen: Gaussian centred at source; sigma=15 ensures vessel at y=+10 gets O2≈52.
        // Source consumes at 40/s → drops below hypoxiaO2=30 in ~0.5s.
        blueprint.AddGridField( "Oxygen" )
            .SetInitializer( DigitalTwin::GridInitializer::Gaussian( glm::vec3( 0.0f ), 15.0f, 60.0f ) )
            .SetDiffusionCoefficient( 2.0f )
            .SetDecayRate( 0.0f )
            .SetComputeHz( 60.0f );

        // VEGF: D=0.5, decay=0.12 → λ≈2 voxels. VEGF reaches the vessel (5 voxels away)
        // with a meaningful gradient so chemotaxis can drive TipCell migration.
        // High StalkCell/TipCell consumption (60/40 per cell) depletes VEGF below the PhalanxActivation
        // threshold after the ring matures, preventing PhalanxActivation from fighting maturation.
        blueprint.AddGridField( "VEGF" )
            .SetInitializer( DigitalTwin::GridInitializer::Constant( 0.0f ) )
            .SetDiffusionCoefficient( 0.5f )
            .SetDecayRate( 0.12f )
            .SetComputeHz( 60.0f );

        // --- Source cell at (0,0,0) ---
        auto& source = blueprint.AddAgentGroup( "SourceCell" )
            .SetCount( 1 )
            .SetMorphology( DigitalTwin::MorphologyGenerator::CreateSphere( 1.2f ) )
            .SetDistribution( { glm::vec4( 0.0f, 0.0f, 0.0f, 1.0f ) } )
            .SetColor( glm::vec4( 1.0f, 0.5f, 0.1f, 1.0f ) ); // orange → purple when hypoxic

        source.AddBehaviour( DigitalTwin::Behaviours::ConsumeField{ "Oxygen", 40.0f } ).SetHz( 60.0f );
        source.AddBehaviour( DigitalTwin::BiologyGenerator::StandardCellCycle()
                                 .SetBaseDoublingTime( 99.0f )
                                 .SetProliferationOxygenTarget( 60.0f )
                                 .SetArrestPressureThreshold( 999.0f )
                                 .SetHypoxiaOxygenThreshold( 30.0f )
                                 .SetNecrosisOxygenThreshold( 1.0f )
                                 .SetApoptosisRate( 0.0f )
                                 .Build() )
            .SetHz( 10.0f );
        source.AddBehaviour( DigitalTwin::Behaviours::SecreteField{
                                 "VEGF", 800.0f,
                                 static_cast<int>( DigitalTwin::LifecycleState::Hypoxic ) } )
            .SetHz( 60.0f );

        // --- Vessel tube: straight trunk, no branches ---
        auto tree = DigitalTwin::VesselTreeGenerator::BranchingTree()
            .SetOrigin( glm::vec3( -12.0f, 10.0f, 0.0f ) )
            .SetDirection( glm::vec3( 1.0f, 0.0f, 0.0f ) )
            .SetLength( 24.0f )
            .SetCellSpacing( 1.5f )
            .SetRingSize( 6 )
            .SetTubeRadius( 1.5f )
            .SetBranchingDepth( 0 )   // straight tube — no branches
            .SetCurvature( 0.0f )     // perfectly straight for predictable sprouting
            .SetSeed( 42 )
            .Build();

        DigitalTwin::Behaviours::VesselSeed seed;
        seed.segmentCounts = tree.segmentCounts;
        seed.explicitEdges = tree.edges;
        seed.edgeFlags     = tree.edgeFlags; // RING=0x1, AXIAL=0x2 — used by mitosis shader

        auto& vessel = blueprint.AddAgentGroup( "VesselTube" )
            .SetCount( tree.totalCells )
            .SetMorphology( DigitalTwin::MorphologyGenerator::CreateCurvedTile( 60.0f, 1.35f, 0.25f, 1.5f ) )
            .SetDistribution( tree.positions )
            .SetOrientations( tree.normals )
            .SetColor( glm::vec4( 0.5f, 0.15f, 0.15f, 1.0f ) )
            .SetInitialCellType( static_cast<int>( DigitalTwin::CellType::PhalanxCell ) )
            .AddCellTypeMorphology( static_cast<int>( DigitalTwin::CellType::PhalanxCell ),
                DigitalTwin::MorphologyGenerator::CreateCurvedTile( 60.0f, 1.35f, 0.25f, 1.5f ),
                glm::vec4( 0.5f, 0.15f, 0.15f, 1.0f ) )   // dark red — quiescent
            .AddCellTypeMorphology( static_cast<int>( DigitalTwin::CellType::StalkCell ),
                DigitalTwin::MorphologyGenerator::CreateCurvedTile( 60.0f, 1.35f, 0.25f, 1.5f ),
                glm::vec4( 1.0f, 0.8f, 0.1f, 1.0f ) )     // yellow — activated
            .AddCellTypeMorphology( static_cast<int>( DigitalTwin::CellType::TipCell ),
                DigitalTwin::MorphologyGenerator::CreateSpikySphere( 1.0f, 1.35f ),
                glm::vec4( 0.0f, 1.0f, 0.3f, 1.0f ) );    // bright green — migrating tip

        // 1. VesselSeed: upload ring + axial edges with flags for biology shader filtering
        vessel.AddBehaviour( seed );

        // 2. PhalanxActivation: threshold=35 targets only the 1-2 rings directly facing source.
        //    Deactivation=7 gives hysteresis without cycling.
        vessel.AddBehaviour( DigitalTwin::Behaviours::PhalanxActivation{
                "VEGF", 35.0f, 7.0f } )
            .SetHz( 60.0f );

        // 3a. VEGFR1 sink: StalkCells consume VEGF heavily so the activation zone stays local.
        //     High rate means the few rings that activate eat up VEGF before it can spread axially.
        //     Also depletes VEGF below PhalanxActivation threshold so ring cells can mature properly.
        vessel.AddBehaviour( DigitalTwin::Behaviours::ConsumeField{ "VEGF", 120.0f } )
            .SetRequiredCellType( static_cast<int>( DigitalTwin::CellType::StalkCell ) )
            .SetHz( 60.0f );

        // 3b. VEGFR1 passive sink: PhalanxCells express VEGFR1 (decoy receptor) at low levels.
        //     This attenuates VEGF along the whole tube length, preventing far rings from activating.
        vessel.AddBehaviour( DigitalTwin::Behaviours::ConsumeField{ "VEGF", 4.0f } )
            .SetRequiredCellType( static_cast<int>( DigitalTwin::CellType::PhalanxCell ) )
            .SetHz( 60.0f );

        // 3c. VEGFR2 sink: TipCell consumes VEGF (high VEGFR2 expression in vivo).
        vessel.AddBehaviour( DigitalTwin::Behaviours::ConsumeField{ "VEGF", 60.0f } )
            .SetRequiredCellType( static_cast<int>( DigitalTwin::CellType::TipCell ) )
            .SetHz( 60.0f );

        // 4. NotchDll4: lateral inhibition selects exactly 1 TipCell from activated StalkCells.
        //    subSteps=40 + inhibitionGain=40 ensures the inhibition wave propagates around the
        //    full ring within the 3s grace period, preventing multi-TipCell selection.
        vessel.AddBehaviour( DigitalTwin::Behaviours::NotchDll4{
                1.0f, 0.1f, 40.0f, 1.0f, 0.90f, 0.20f, "VEGF", 40u } )
            .SetHz( 60.0f );

        // 5. CellCycle: directedMitosis=true → StalkCell axially adjacent to TipCell divides
        //    along the vessel axis, extending the sprout. Non-proliferating cells mature to PhalanxCell.
        //    5s doubling time (2x faster than before) so the sprout chain extends more noticeably.
        auto stalkCycle = DigitalTwin::BiologyGenerator::StandardCellCycle()
                              .SetBaseDoublingTime( 2.0f / 3600.0f )
                              .SetProliferationOxygenTarget( 40.0f )
                              .SetArrestPressureThreshold( 5.0f )
                              .SetHypoxiaOxygenThreshold( 5.0f )
                              .SetNecrosisOxygenThreshold( 1.0f )
                              .SetApoptosisRate( 0.0f )
                              .Build();
        stalkCycle.directedMitosis = true;
        vessel.AddBehaviour( stalkCycle ).SetHz( 10.0f );

        // 6. Chemotaxis: TipCell follows the VEGF gradient toward the source.
        //    sensitivity=2 so StalkCell division can keep pace.
        //    contactInhibitionDensity=0: disabled. Once the spatial hash covers all agent groups
        //    (fixed via globalHashCapacity scaling in SimulationBuilder), a TipCell inside its origin
        //    ring finds ~15-20 vessel neighbors within hashCellSize*2 = 6 units, saturating any
        //    sensible density threshold and freezing the tip. Phase 5 VE-Cadherin adhesion will
        //    provide proper tip-tube separation mechanics.
        vessel.AddBehaviour( DigitalTwin::Behaviours::Chemotaxis{ "VEGF", 2.0f, 0.002f, 0.8f, 0.0f } )
            .SetHz( 60.0f )
            .SetRequiredCellType( static_cast<int>( DigitalTwin::CellType::TipCell ) );

        // 7. VesselSpring: maintains ring shape and sprout chain spacing.
        //    Per-edge rest lengths from geometry handle mixed ring/sprout lengths.
        //    anchorPhalanxCells=true freezes the tube; only TipCell and sprout chain move.
        DigitalTwin::Behaviours::VesselSpring spring{};
        spring.springStiffness    = 15.0f;
        spring.restingLength      = 1.5f;
        spring.dampingCoefficient = 10.0f;
        spring.anchorPhalanxCells = true;
        vessel.AddBehaviour( spring ).SetHz( 60.0f );

        // Biomechanics (JKR) intentionally omitted for the vessel group.
        // VesselSpring + PhalanxCell anchoring maintain tube integrity.
        // JKR Hertzian repulsion on closely-packed ring geometry (spacing=1.5, maxRadius=1.5)
        // produces explosive forces — overlap=1.5 → force≈27 per pair.
        // Phase 5 VE-Cadherin adhesion will provide proper vessel cell-cell contact mechanics.
    }
} // namespace Gaudi::Demos
