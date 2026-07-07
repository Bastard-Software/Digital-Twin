#include "Demos.h"

#include <simulation/BiologyGenerator.h>
#include <simulation/BiomechanicsGenerator.h>
#include <simulation/MorphologyGenerator.h>
#include <simulation/SpatialDistribution.h>

namespace Gaudi::Demos
{
    // Heterogeneous tumour microenvironment — cancer immunoediting in one spheroid (~2k cells).
    //
    // A single solid tumour sphere intersperses FOUR fates that differ only in how the immune
    // system sees them (Marusyk & Polyak 2010 — intratumoural heterogeneity):
    //   - Immunogenic (red):    presents neoantigen bit 0 at full MHC-I density (1.0) -> cleared fast.
    //   - MHC-I-low escape (orange): presents the SAME bit 0 but downregulated to 0.2 -> cleared
    //     slowly, so a fraction persists (the graded-recognition escape clone). MHC-I downregulation
    //     is the commonest real immune-escape route; modelled by AntigenExpression.presentationLevel.
    //   - Antigen-loss (blue):  presents a DIFFERENT bit (1) -> shares nothing with the repertoire ->
    //     invisible, survives, and SECRETES the decoy chemokine.
    //   - Normal tissue (grey, NormalCell): no neoantigen -> never targeted, spared amid the clearance
    //     (the visual proof of CTL specificity — healthy cells survive between dying tumour cells).
    // Elimination vs equilibrium vs escape (Schreiber 2011), all in one lesion.
    //
    // Two immune populations infiltrate the SAME sphere (no artificial spatial split):
    //   - Cytotoxic CD8+ T cells (green): antigen-specific killers (AntigenKilling on bit 0),
    //     pre-infiltrated (TILs), activated at the vaccine gate. Each kill releases cytotoxic
    //     cytokine; they home up its gradient. They clear Live AND quiescent targets (a contact-
    //     inhibited cell in the dense core is still granzyme-killed).
    //   - Innate followers (yellow): NO antigen recognition — macrophage/myeloid-like cells that
    //     integrate BOTH chemokine gradients (cytotoxic storm + tumour decoy) via two Chemotaxis
    //     behaviours, so they converge on the inflamed tumour alongside the CTLs rather than being
    //     siphoned off elsewhere.
    //
    // Expected arc: at the vaccine gate the CTLs clear the immunogenic clone fast (red -> pale husks)
    // and the MHC-low clone slowly (orange lingers, a thinning escape population), sparing the
    // interspersed normal cells and never touching the antigen-loss clone (which keeps secreting its
    // decoy). A cytokine storm flares through the sphere; innate followers pour in up the combined
    // gradient. End state: a spheroid of pale immune-killed husks studded with surviving grey normal
    // tissue, blue antigen-loss cells, and a residue of orange escapees — immunoediting made visible.
    void SetupTumorMicroenvironmentDemo( DigitalTwin::SimulationBlueprint& blueprint )
    {
        using namespace DigitalTwin;

        blueprint.SetName( "Tumour Microenvironment" );
        blueprint.SetDomainSize( glm::vec3( 80.0f ), 2.0f );

        blueprint.ConfigureSpatialPartitioning()
            .SetMethod( SpatialPartitioningMethod::HashGrid )
            .SetCellSize( 6.0f )
            .SetMaxDensity( 64 )
            .SetComputeHz( 60.0f );

        constexpr float k_vaccineTime = 8.0f;
        const glm::vec3 centre( 0.0f );
        const float     radius = 28.0f; // one shared tumour sphere — loosely packed so immune cells weave between cells

        // ── Fields ───────────────────────────────────────────────────────────────────
        // Oxygen: large reservoir so a ~2k-cell spheroid does not starve before the immune response
        // plays out (this demo is about recognition, not hypoxia).
        blueprint.AddGridField( "Oxygen" )
            .SetInitializer( GridInitializer::Gaussian( centre, 34.0f, 600.0f ) )
            .SetDiffusionCoefficient( 6.0f )
            .SetDecayRate( 0.0f )
            .SetComputeHz( 60.0f );

        // Cytotoxic cytokine: released per CTL kill (IL-6/TNF-α/IFN-γ-like). Low diffusion/decay so it
        // builds a bright inflammation cloud through the spheroid.
        blueprint.AddGridField( "CytotoxicCytokine" )
            .SetInitializer( GridInitializer::Constant( 0.0f ) )
            .SetDiffusionCoefficient( 0.2f )
            .SetDecayRate( 0.005f )
            .SetComputeHz( 60.0f );

        // Decoy chemokine: secreted by the antigen-loss cells (tumour-derived immunosuppressant).
        blueprint.AddGridField( "DecoyChemokine" )
            .SetInitializer( GridInitializer::Constant( 0.0f ) )
            .SetDiffusionCoefficient( 0.4f )
            .SetDecayRate( 0.01f )
            .SetComputeHz( 60.0f );

        // ── Shared biology ───────────────────────────────────────────────────────────
        auto tumorCellCycle = BiologyGenerator::StandardCellCycle()
                                  .SetBaseDoublingTime( 200.0f / 3600.0f ) // ~100s effective doubling — slow, demo-paced
                                  .SetProliferationOxygenTarget( 60.0f )
                                  .SetArrestPressureThreshold( 20.0f )
                                  .SetHypoxiaOxygenThreshold( 12.0f )
                                  .SetNecrosisOxygenThreshold( 5.0f )
                                  .SetApoptosisRate( 0.0f )
                                  .Build();

        auto tumorMechanics = BiomechanicsGenerator::JKR()
                                  .SetYoungsModulus( 20.0f )
                                  .SetPoissonRatio( 0.4f )
                                  .SetAdhesionEnergy( 1.5f )
                                  .SetMaxInteractionRadius( 1.5f )
                                  .SetDampingCoefficient( 40.0f ) // dashpot — damps JKR repulsion so cells settle instead of "exploding" apart
                                  .Build();

        auto immuneMechanics = BiomechanicsGenerator::JKR()
                                   .SetYoungsModulus( 8.0f )
                                   .SetPoissonRatio( 0.4f )
                                   .SetAdhesionEnergy( 0.0f )
                                   .SetMaxInteractionRadius( 1.0f )
                                   .SetDampingCoefficient( 80.0f )
                                   .Build();

        // ── Tumour sphere: four interspersed fates (same centre + radius so they mix) ──────────
        // Immunogenic (red) — full presentation, cleared fast.
        auto& immunogenic = blueprint.AddAgentGroup( "Immunogenic" )
                                .SetCount( 500 )
                                .SetMorphology( MorphologyGenerator::CreateSphere( 1.5f ) )
                                .SetDistribution( SpatialDistribution::UniformInSphere( 500, radius, centre ) )
                                .SetInitialCellType( 4 ) // TumorCell
                                .SetColor( glm::vec4( 0.9f, 0.2f, 0.15f, 1.0f ) );
        immunogenic.AddBehaviour( Behaviours::AntigenExpression{ 0b001u, 1.0f } ).SetHz( 60.0f );
        immunogenic.AddBehaviour( Behaviours::ConsumeField{ "Oxygen", 2.0f } ).SetHz( 60.0f );
        immunogenic.AddBehaviour( tumorCellCycle ).SetHz( 60.0f );
        immunogenic.AddBehaviour( tumorMechanics ).SetHz( 60.0f );

        // MHC-I-low escape (orange) — same bit 0, downregulated presentation, cleared slowly.
        auto& mhcLow = blueprint.AddAgentGroup( "MHC-low escape" )
                           .SetCount( 400 )
                           .SetMorphology( MorphologyGenerator::CreateSphere( 1.5f ) )
                           .SetDistribution( SpatialDistribution::UniformInSphere( 400, radius, centre ) )
                           .SetInitialCellType( 4 )
                           .SetColor( glm::vec4( 0.95f, 0.55f, 0.15f, 1.0f ) );
        mhcLow.AddBehaviour( Behaviours::AntigenExpression{ 0b001u, 0.2f } ).SetHz( 60.0f );
        mhcLow.AddBehaviour( Behaviours::ConsumeField{ "Oxygen", 2.0f } ).SetHz( 60.0f );
        mhcLow.AddBehaviour( tumorCellCycle ).SetHz( 60.0f );
        mhcLow.AddBehaviour( tumorMechanics ).SetHz( 60.0f );

        // Antigen-loss (blue) — different antigen bit, invisible to the repertoire, secretes decoy.
        auto& antigenLoss = blueprint.AddAgentGroup( "Antigen-loss" )
                                .SetCount( 350 )
                                .SetMorphology( MorphologyGenerator::CreateSphere( 1.5f ) )
                                .SetDistribution( SpatialDistribution::UniformInSphere( 350, radius, centre ) )
                                .SetInitialCellType( 4 )
                                .SetColor( glm::vec4( 0.2f, 0.5f, 0.85f, 1.0f ) );
        antigenLoss.AddBehaviour( Behaviours::AntigenExpression{ 0b010u, 1.0f } ).SetHz( 60.0f ); // bit 1 — no overlap with repertoire
        antigenLoss.AddBehaviour( Behaviours::ConsumeField{ "Oxygen", 2.0f } ).SetHz( 60.0f );
        antigenLoss.AddBehaviour( Behaviours::SecreteField{ "DecoyChemokine", 100.0f } ).SetHz( 60.0f );
        antigenLoss.AddBehaviour( tumorCellCycle ).SetHz( 60.0f );
        antigenLoss.AddBehaviour( tumorMechanics ).SetHz( 60.0f );

        // Normal tissue (grey, NormalCell) — no neoantigen -> never targeted. Inert (no CellCycle).
        auto& normal = blueprint.AddAgentGroup( "Normal tissue" )
                           .SetCount( 350 )
                           .SetMorphology( MorphologyGenerator::CreateSphere( 1.4f ) )
                           .SetDistribution( SpatialDistribution::UniformInSphere( 350, radius, centre ) )
                           .SetInitialCellType( 6 ) // NormalCell
                           .SetColor( glm::vec4( 0.62f, 0.62f, 0.64f, 1.0f ) );
        normal.AddBehaviour( Behaviours::ConsumeField{ "Oxygen", 1.0f } ).SetHz( 60.0f );
        normal.AddBehaviour( tumorMechanics ).SetHz( 60.0f );

        // ── Immune populations (both infiltrate the same sphere) ───────────────────────────────
        // Cytotoxic CD8+ T cells (green): pre-infiltrated killers homing up the cytotoxic gradient.
        auto& ctl = blueprint.AddAgentGroup( "Cytotoxic T cells" )
                        .SetCount( 250 )
                        .SetMorphology( MorphologyGenerator::CreateSphere( 1.0f ) )
                        .SetDistribution( SpatialDistribution::UniformInSphere( 250, radius, centre ) )
                        .SetInitialCellType( 5 ) // ImmuneCell
                        .SetColor( glm::vec4( 0.15f, 0.9f, 0.3f, 1.0f ) );
        ctl.AddBehaviour( Behaviours::BrownianMotion{ 1.1f } ).SetHz( 60.0f ); // livelier patrol jitter than the tumour cells
        ctl.AddBehaviour( Behaviours::Chemotaxis{ "CytotoxicCytokine", 8.0f, 0.05f, 4.0f } ).SetHz( 60.0f );
        ctl.AddBehaviour( immuneMechanics ).SetHz( 60.0f );
        ctl.AddBehaviour( Behaviours::AntigenKilling{
                              0b001u,        // recognizes bit 0
                              6.0f,          // contactDistance
                              0.05f,         // killProbPerSec — gradual but a touch brisker than the first pass
                              k_vaccineTime, // vaccine gate
                              "CytotoxicCytokine",
                              40.0f,         // cytokineReleasePerKill
                              0.008f } )     // cytokineFeedbackGain — gentle storm (no runaway burst)
            .SetHz( 60.0f );

        // Innate followers (yellow): no killing. Two Chemotaxis behaviours — they integrate BOTH the
        // cytotoxic storm and the decoy chemokine, so they converge on the tumour with the CTLs.
        auto& follower = blueprint.AddAgentGroup( "Innate followers" )
                             .SetCount( 150 )
                             .SetMorphology( MorphologyGenerator::CreateSphere( 1.0f ) )
                             .SetDistribution( SpatialDistribution::UniformInSphere( 150, 34.0f, centre ) )
                             .SetInitialCellType( 5 )
                             .SetColor( glm::vec4( 0.95f, 0.85f, 0.2f, 1.0f ) );
        follower.AddBehaviour( Behaviours::BrownianMotion{ 1.1f } ).SetHz( 60.0f ); // livelier patrol jitter than the tumour cells
        follower.AddBehaviour( Behaviours::Chemotaxis{ "CytotoxicCytokine", 8.0f, 0.05f, 4.0f } ).SetHz( 60.0f );
        follower.AddBehaviour( Behaviours::Chemotaxis{ "DecoyChemokine", 8.0f, 0.05f, 4.0f } ).SetHz( 60.0f );
        follower.AddBehaviour( immuneMechanics ).SetHz( 60.0f );
    }
} // namespace Gaudi::Demos
