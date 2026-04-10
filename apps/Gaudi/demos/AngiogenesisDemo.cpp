#include "Demos.h"

#include <simulation/BiologyGenerator.h>
#include <simulation/MorphologyGenerator.h>
#include <simulation/Phenotype.h>
#include <simulation/VesselTreeGenerator.h>

namespace Gaudi::Demos
{
    void SetupAngiogenesisDemo( DigitalTwin::SimulationBlueprint& blueprint )
    {
        // Full angiogenesis demo — Phase 4.
        //
        // A Y-shaped bifurcated vessel: a trunk from the left that forks into two branches.
        // A hypoxic tumour sits ahead of the fork (~10 units from each branch tip), in the
        // notch of the Y. Both branches independently sprout toward the tumour:
        //
        //   1. Tumour consumes O2 → Hypoxic → secretes VEGF (~t=2s)
        //   2. PhalanxActivation wakes the rings nearest each branch tip (~t=5s)
        //   3. NotchDll4 selects 1 TipCell per branch (~t=7s)
        //   4. Both TipCells migrate toward the tumour via chemotaxis
        //   5. Directed mitosis extends a sprout chain behind each TipCell
        //   6. Sprouts meet near the tumour → Anastomosis fires (~t=18s)
        //   7. Matured vessel loop perfuses O2 back into the domain
        //
        // Vessel geometry:
        //   Trunk: origin (-15, 0, 0), direction +X, length=10 → junction at (-5, 0, 0)
        //   BranchingAngle=40°, LengthFalloff=0.8 → each branch ~8 units long
        //   Branch tips ≈ (1.1, ±5.1, 0) — ~10 units from the tumour at (10, 0, 0)

        blueprint.SetName( "Vessel Angiogenesis" );
        // Domain=82 → 41×41×41 voxels at voxelSize=2.  With an ODD voxel count, (0,0,0) is a
        // voxel centre, so the tumour at (10,0,0) hits EXACTLY the centre of its voxel and
        // VEGF is released symmetrically in ±Y — critical for equal activation of both branches.
        // (Domain=80 gives 40 voxels per axis; voxel centres land at odd positions ±1,±3,…
        //  so the tumour at y=0 mapped to voxel-centre y=1, making upper/lower branches unequal.)
        blueprint.SetDomainSize( glm::vec3( 82.0f ), 2.0f );

        blueprint.ConfigureSpatialPartitioning()
            .SetMethod( DigitalTwin::SpatialPartitioningMethod::HashGrid )
            .SetCellSize( 3.0f )
            .SetMaxDensity( 64 )
            .SetComputeHz( 60.0f );

        // O2: Gaussian centred at the vessel junction; sigma=20 keeps vessel rings well-oxygenated at t=0.
        // Decay=0.05 → diffusion length λ = sqrt(D/decay) = sqrt(2/0.05) ≈ 6.3 units.
        // Cells >6 units from the nearest vessel segment become hypoxic naturally.
        blueprint.AddGridField( "Oxygen" )
            .SetInitializer( DigitalTwin::GridInitializer::Gaussian( glm::vec3( -5.0f, 0.0f, 0.0f ), 20.0f, 80.0f ) )
            .SetDiffusionCoefficient( 2.0f )
            .SetDecayRate( 0.05f )
            .SetComputeHz( 60.0f );

        // VEGF: D=0.5, decay=0.12 → λ≈2 units (same as sprouting demo).
        // Steeper gradient ensures only the tip ring of each branch exceeds the activation threshold.
        blueprint.AddGridField( "VEGF" )
            .SetInitializer( DigitalTwin::GridInitializer::Constant( 0.0f ) )
            .SetDiffusionCoefficient( 0.5f )
            .SetDecayRate( 0.12f )
            .SetComputeHz( 60.0f );

        // --- Tumour source — ahead of the fork, in the notch of the Y ---
        auto& tumour = blueprint.AddAgentGroup( "Tumour" )
            .SetCount( 1 )
            .SetMorphology( DigitalTwin::MorphologyGenerator::CreateSphere( 1.2f ) )
            .SetDistribution( { glm::vec4( 10.0f, 0.0f, 0.0f, 1.0f ) } )
            .SetColor( glm::vec4( 1.0f, 0.5f, 0.1f, 1.0f ) );

        tumour.AddBehaviour( DigitalTwin::Behaviours::ConsumeField{ "Oxygen", 60.0f } ).SetHz( 60.0f );
        tumour.AddBehaviour( DigitalTwin::BiologyGenerator::StandardCellCycle()
                                 .SetBaseDoublingTime( 99.0f )
                                 .SetProliferationOxygenTarget( 60.0f )
                                 .SetArrestPressureThreshold( 999.0f )
                                 .SetHypoxiaOxygenThreshold( 25.0f )
                                 .SetNecrosisOxygenThreshold( 0.0f )  // never necrotic — stays hypoxic and secretes VEGF throughout
                                 .SetApoptosisRate( 0.0f )
                                 .Build() )
            .SetHz( 10.0f );
        tumour.AddBehaviour( DigitalTwin::Behaviours::SecreteField{
                                 "VEGF", 800.0f,
                                 DigitalTwin::LifecycleState::Hypoxic } )
            .SetHz( 60.0f );

        // --- Bifurcated vessel tree ---
        // Trunk from (-20,0,0) along +X; junction at (-5,0,0); two branches at ±40°.
        // Trunk=15 units, LengthFalloff=0.9 → branches 13.5 units each.
        // Branch tips at approximately (5.3, ±8.7, 0) — ~10 units from tumour at (10, 0, 0).
        auto tree = DigitalTwin::VesselTreeGenerator::BranchingTree()
            .SetOrigin( glm::vec3( -20.0f, 0.0f, 0.0f ) )
            .SetDirection( glm::vec3( 1.0f, 0.0f, 0.0f ) )
            .SetLength( 15.0f )
            .SetCellSpacing( 1.5f )
            .SetRingSize( 6 )
            .SetTubeRadius( 1.5f )
            .SetBranchingAngle( 40.0f )
            .SetBranchingDepth( 1 )
            .SetLengthFalloff( 0.9f )
            .SetAngleJitter( 0.0f )    // symmetric branches — no random jitter
            .SetBranchProbability( 1.0f )
            .SetTubeRadiusFalloff( 0.79f )
            .SetCurvature( 0.0f )
            .SetSeed( 42 )
            .Build();

        DigitalTwin::Behaviours::VesselSeed seed;
        seed.segmentCounts = tree.segmentCounts;
        seed.explicitEdges = tree.edges;
        seed.edgeFlags     = tree.edgeFlags;

        auto& vessel = blueprint.AddAgentGroup( "VesselTree" )
            .SetCount( tree.totalCells )
            .SetMorphology( DigitalTwin::MorphologyGenerator::CreateCurvedTile( 60.0f, 1.35f, 0.25f, 1.5f ) )
            .SetDistribution( tree.positions )
            .SetOrientations( tree.normals )
            .SetColor( glm::vec4( 0.5f, 0.15f, 0.15f, 1.0f ) )
            .SetInitialCellType( static_cast<int>( DigitalTwin::CellType::PhalanxCell ) )
            .AddCellTypeMorphology( static_cast<int>( DigitalTwin::CellType::PhalanxCell ),
                DigitalTwin::MorphologyGenerator::CreateCurvedTile( 60.0f, 1.35f, 0.25f, 1.5f ),
                glm::vec4( 0.5f, 0.15f, 0.15f, 1.0f ) )
            .AddCellTypeMorphology( static_cast<int>( DigitalTwin::CellType::StalkCell ),
                DigitalTwin::MorphologyGenerator::CreateCurvedTile( 60.0f, 1.35f, 0.25f, 1.5f ),
                glm::vec4( 1.0f, 0.8f, 0.1f, 1.0f ) )
            .AddCellTypeMorphology( static_cast<int>( DigitalTwin::CellType::TipCell ),
                DigitalTwin::MorphologyGenerator::CreateSpikySphere( 1.0f, 1.35f ),
                glm::vec4( 0.0f, 1.0f, 0.3f, 1.0f ) );

        vessel.AddBehaviour( seed );

        // Anastomosis runs FIRST — before PhalanxActivation — so it can check the TipCell
        // pair while both cells are still cellType=1.  If PhalanxActivation ran first, a
        // transient VEGF dip (StalkCell ring consumption ≈720/s vs tumor secretion 800/s)
        // could revert a TipCell to PhalanxCell in the same frame, permanently missing the
        // connection window.  The spatial hash used here is built at frame start so positions
        // are consistent regardless of where in the frame Anastomosis dispatches.
        vessel.AddBehaviour( DigitalTwin::Behaviours::Anastomosis{ 5.0f, true } ).SetHz( 60.0f );

        // PhalanxActivation: activationThreshold=45 (not the sprouting demo's 35).
        // The Y-branches are angled 40° toward the tumor, so they curve *through* their
        // closest-approach point. Distance to tumor along the final ~4 rings of each branch
        // spans only 9.64–9.85 units — a 0.21-unit window — so the VEGF field is nearly
        // flat there. Using the sprouting threshold (35) activates ~24 cells per branch,
        // ~2.5x the sprouting demo's clean 6–12 cells. Raising to 45 shrinks the band to
        // the 1–2 rings at the true VEGF peak, matching the sprouting demo's look.
        // Deactivation=7: same as sprouting — StalkCells behind the tip quickly mature once
        // local VEGF drops below 7 (120/s consumption depletes the local voxel fast).
        vessel.AddBehaviour( DigitalTwin::Behaviours::PhalanxActivation{ "VEGF", 45.0f, 7.0f } )
            .SetHz( 60.0f );

        vessel.AddBehaviour( DigitalTwin::Behaviours::ConsumeField{ "VEGF", 120.0f } )
            .SetRequiredCellType( DigitalTwin::CellType::StalkCell )
            .SetHz( 60.0f );
        vessel.AddBehaviour( DigitalTwin::Behaviours::ConsumeField{ "VEGF", 4.0f } )
            .SetRequiredCellType( DigitalTwin::CellType::PhalanxCell )
            .SetHz( 60.0f );
        vessel.AddBehaviour( DigitalTwin::Behaviours::ConsumeField{ "VEGF", 60.0f } )
            .SetRequiredCellType( DigitalTwin::CellType::TipCell )
            .SetHz( 60.0f );

        vessel.AddBehaviour( DigitalTwin::Behaviours::NotchDll4{
                1.0f, 0.1f, 40.0f, 1.0f, 0.90f, 0.20f, "VEGF", 40u } )
            .SetHz( 60.0f );

        // Matched to sprouting demo: 2s doubling, sensitivity=2.0, gain=0.8.
        // Faster division + faster tip keeps the sprout chain compact. StalkCells behind the
        // tip mature quickly (deactivation=7) so the chain body is PhalanxCell (dark red),
        // matching the clean single-yellow-StalkCell look from the sprouting demo.
        auto stalkCycle = DigitalTwin::BiologyGenerator::StandardCellCycle()
                              .SetBaseDoublingTime( 2.0f / 3600.0f )
                              .SetProliferationOxygenTarget( 40.0f )
                              .SetArrestPressureThreshold( 5.0f )
                              .SetHypoxiaOxygenThreshold( 5.0f )
                              .SetNecrosisOxygenThreshold( 1.0f )
                              .SetApoptosisRate( 0.0f )
                              .Build();
        stalkCycle.directedMitosis = true;
        // Restrict to StalkCells only — TipCells migrate into the hypoxic tumour zone
        // intentionally; running the lifecycle on them turns them blue (state=Hypoxic) when O2 < 5.
        vessel.AddBehaviour( stalkCycle )
            .SetHz( 10.0f )
            .SetRequiredCellType( DigitalTwin::CellType::StalkCell );

        // Matched to sprouting demo: sensitivity=2.0, gain=0.8.
        // contactInhibitionDensity=0: disabled (same reason as sprouting demo — vessel ring
        // neighbors saturate any threshold once the hash covers all groups).
        vessel.AddBehaviour( DigitalTwin::Behaviours::Chemotaxis{ "VEGF", 2.0f, 0.002f, 0.8f, 0.0f } )
            .SetHz( 60.0f )
            .SetRequiredCellType( DigitalTwin::CellType::TipCell );

        DigitalTwin::Behaviours::VesselSpring spring{};
        spring.springStiffness    = 15.0f;
        spring.restingLength      = 1.5f;
        spring.dampingCoefficient = 10.0f;
        spring.anchorPhalanxCells = true;
        vessel.AddBehaviour( spring ).SetHz( 60.0f );

        // PhalanxCell-only perfusion: unconnected sprout StalkCells have no lumen and must not
        // inject O2 — that would re-oxygenate the tumour prematurely and kill the VEGF signal.
        // After anastomosis, PhalanxActivation matures the new StalkCells to PhalanxCells
        // (VEGF drops once the vessel loop delivers O2), and they then begin perfusing.
        vessel.AddBehaviour( DigitalTwin::Behaviours::Perfusion{ "Oxygen", 4.0f } )
            .SetHz( 60.0f )
            .SetRequiredCellType( DigitalTwin::CellType::PhalanxCell );

        // JKR intentionally omitted for the vessel group — see sprouting demo for rationale.
        // Phase 5 VE-Cadherin adhesion will provide proper vessel cell-cell contact mechanics.
    }
} // namespace Gaudi::Demos
