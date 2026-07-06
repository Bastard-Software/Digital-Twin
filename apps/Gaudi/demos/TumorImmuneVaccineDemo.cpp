#include "Demos.h"

#include <simulation/BiologyGenerator.h>
#include <simulation/BiomechanicsGenerator.h>
#include <simulation/MorphologyGenerator.h>
#include <simulation/SpatialDistribution.h>

namespace Gaudi::Demos
{
    // Antigen-specific cytotoxic killing + cytokine storm feedback (Steps 1.1-1.4).
    //
    // Two tumour subclones grow side by side, sharing identical biology except for the
    // neoantigen they present:
    //   TumorMatched  (red)  presents antigen bit 0 -- recognised by the vaccine repertoire.
    //   TumorEscaping (blue) presents antigen bit 1 -- an antigen-loss variant invisible to it.
    //
    // Sequence:
    //   t=0-15s   Both subclones grow unchecked (O2-gated CellCycle). Immune cells patrol
    //             the surrounding volume (BrownianMotion); no cytokine yet so Chemotaxis is
    //             a no-op gradient of zero.
    //   t=15s     Vaccine "administered" -- AntigenKilling.startTime gates the repertoire on.
    //   t=15s+    Immune cells contacting TumorMatched cells kill them (perforin/granzyme,
    //             Poisson draw on killProbPerSec) and release cytokine at the kill site.
    //             Chemotaxis recruits more immune cells toward the inflamed zone; local
    //             cytokine feeds back into killProbPerSec (cytokineFeedbackGain), producing
    //             an emergent, self-reinforcing clearance front -- a cytokine storm.
    //   throughout TumorEscaping is never recognised (no shared antigen bit) and keeps
    //             growing through the whole sequence -- immune escape.
    //
    // Biological validation: same cell biology (CellCycle, Biomechanics) in every group,
    // divergent fate purely from antigen genotype + antigen-specific recognition -- the
    // positive/negative control design already used by the EC demo trilogy.
    void SetupTumorImmuneVaccineDemo( DigitalTwin::SimulationBlueprint& blueprint )
    {
        blueprint.SetName( "Tumor-Immune Vaccine Response" );
        blueprint.SetDomainSize( glm::vec3( 60.0f ), 2.0f );

        blueprint.ConfigureSpatialPartitioning()
            .SetMethod( DigitalTwin::SpatialPartitioningMethod::HashGrid )
            .SetCellSize( 6.0f )
            .SetMaxDensity( 64 )
            .SetComputeHz( 60.0f );

        constexpr float k_vaccineTime = 15.0f;

        // Oxygen: generous reservoir spanning both subclone clusters -- growth is limited by
        // the vaccine response in this demo, not premature starvation. (Tuned empirically:
        // the original peak=80/radius=16 depleted under compounding CellCycle growth well
        // before t=15s, so the whole matched cluster went hypoxic/necrotic before the
        // vaccine ever fired -- diffusionCoefficient=4 also lets the reservoir re-equalise
        // faster than local consumption can outrun it.)
        blueprint.AddGridField( "Oxygen" )
            .SetInitializer( DigitalTwin::GridInitializer::Gaussian( glm::vec3( 0.0f ), 20.0f, 250.0f ) )
            .SetDiffusionCoefficient( 4.0f )
            .SetDecayRate( 0.0f )
            .SetComputeHz( 60.0f );

        // Cytokine: starts empty; each kill injects a burst at the kill site. Tuned for
        // VISIBILITY on the default field-overlay scale (maxValue=100, transparent below ~1.0):
        // a small release on a fast-diffusing/fast-decaying field peaks near zero and never
        // renders. Low diffusion keeps the burst concentrated into a bright local cloud and
        // low decay lets it linger as a visible inflammation zone through the assault, then
        // slowly resolve once the matched subclone is cleared. cytokineReleasePerKill (below)
        // sets the magnitude; feedbackGain is scaled down to match so the storm loop stays bounded.
        blueprint.AddGridField( "Cytokine" )
            .SetInitializer( DigitalTwin::GridInitializer::Constant( 0.0f ) )
            .SetDiffusionCoefficient( 0.2f )
            .SetDecayRate( 0.005f )
            .SetComputeHz( 60.0f );

        // Shared tumour biology -- identical CellCycle + Biomechanics for both subclones.
        // Doubling time is demo-paced (not biological), tuned so growth is visible over the
        // pre-vaccine window without exploding consumption faster than Oxygen can supply it.
        auto tumorCellCycle = DigitalTwin::BiologyGenerator::StandardCellCycle()
                                  .SetBaseDoublingTime( 80.0f / 3600.0f ) // ~40s effective doubling
                                  .SetProliferationOxygenTarget( 60.0f )
                                  .SetArrestPressureThreshold( 20.0f )
                                  .SetHypoxiaOxygenThreshold( 20.0f )
                                  .SetNecrosisOxygenThreshold( 8.0f )
                                  .SetApoptosisRate( 0.0f )
                                  .Build();

        auto tumorMechanics = DigitalTwin::BiomechanicsGenerator::JKR()
                                  .SetYoungsModulus( 20.0f )
                                  .SetPoissonRatio( 0.4f )
                                  .SetAdhesionEnergy( 1.5f )
                                  .SetMaxInteractionRadius( 1.5f )
                                  .Build();

        // ── TumorMatched (red) — antigen bit 0, recognised by the vaccine ────────────
        auto& tumorMatched = blueprint.AddAgentGroup( "TumorMatched" )
                                 .SetCount( 24 )
                                 .SetMorphology( DigitalTwin::MorphologyGenerator::CreateSphere( 1.5f ) )
                                 .SetDistribution( DigitalTwin::SpatialDistribution::UniformInSphere( 24, 7.0f, glm::vec3( -7.0f, 0.0f, 0.0f ) ) )
                                 .SetInitialCellType( 4 ) // TumorCell
                                 .SetColor( glm::vec4( 0.9f, 0.25f, 0.15f, 1.0f ) );

        tumorMatched.AddBehaviour( DigitalTwin::Behaviours::AntigenExpression{ 0b001u } ).SetHz( 60.0f );
        tumorMatched.AddBehaviour( DigitalTwin::Behaviours::ConsumeField{ "Oxygen", 6.0f } ).SetHz( 60.0f );
        tumorMatched.AddBehaviour( tumorCellCycle ).SetHz( 60.0f );
        tumorMatched.AddBehaviour( tumorMechanics ).SetHz( 60.0f );

        // ── TumorEscaping (blue) — antigen bit 1, an antigen-loss variant ────────────
        auto& tumorEscaping = blueprint.AddAgentGroup( "TumorEscaping" )
                                  .SetCount( 10 )
                                  .SetMorphology( DigitalTwin::MorphologyGenerator::CreateSphere( 1.5f ) )
                                  .SetDistribution( DigitalTwin::SpatialDistribution::UniformInSphere( 10, 5.0f, glm::vec3( 7.0f, 0.0f, 0.0f ) ) )
                                  .SetInitialCellType( 4 ) // TumorCell
                                  .SetColor( glm::vec4( 0.2f, 0.5f, 0.85f, 1.0f ) );

        tumorEscaping.AddBehaviour( DigitalTwin::Behaviours::AntigenExpression{ 0b010u } ).SetHz( 60.0f );
        tumorEscaping.AddBehaviour( DigitalTwin::Behaviours::ConsumeField{ "Oxygen", 6.0f } ).SetHz( 60.0f );
        tumorEscaping.AddBehaviour( tumorCellCycle ).SetHz( 60.0f );
        tumorEscaping.AddBehaviour( tumorMechanics ).SetHz( 60.0f );

        // ── ImmuneCells (green) — patrol, recruit toward cytokine, kill matched antigen ──
        // Spawn radius tuned so BrownianMotion patrol actually brings cells within
        // contactDistance in a reasonable time (empirically: at radius=22 cells drifted
        // for 90s without ever getting close -- pure random walk over that distance is
        // too slow to be watchable).
        auto& immune = blueprint.AddAgentGroup( "ImmuneCells" )
                            .SetCount( 16 )
                            .SetMorphology( DigitalTwin::MorphologyGenerator::CreateSphere( 1.0f ) )
                            .SetDistribution( DigitalTwin::SpatialDistribution::UniformInSphere( 16, 14.0f, glm::vec3( 0.0f ) ) )
                            .SetInitialCellType( 5 ) // ImmuneCell
                            .SetColor( glm::vec4( 0.15f, 0.9f, 0.3f, 1.0f ) );

        immune.AddBehaviour( DigitalTwin::Behaviours::BrownianMotion{ 0.6f } ).SetHz( 60.0f );
        immune.AddBehaviour( DigitalTwin::Behaviours::Chemotaxis{ "Cytokine", 8.0f, 0.05f, 4.0f } ).SetHz( 60.0f );
        immune
            .AddBehaviour( DigitalTwin::BiomechanicsGenerator::JKR()
                               .SetYoungsModulus( 8.0f )
                               .SetPoissonRatio( 0.4f )
                               .SetAdhesionEnergy( 0.0f )
                               .SetMaxInteractionRadius( 1.0f )
                               .SetDampingCoefficient( 80.0f )
                               .Build() )
            .SetHz( 60.0f );
        immune.AddBehaviour( DigitalTwin::Behaviours::AntigenKilling{
                                 0b001u,        // recognizedAntigens -- matches TumorMatched only
                                 6.0f,          // contactDistance -- generous immunological sensing radius
                                                // for this coarse model; also gives a gradual clearance
                                                // front (not every immune cell in range from frame 1)
                                 0.4f,          // killProbPerSec -- decisive: clears matched before it regrows
                                 k_vaccineTime, // startTime -- vaccine gate
                                 "Cytokine",    // cytokineField
                                 40.0f,         // cytokineReleasePerKill -- large so the field is visibly bright
                                 0.03f } )      // cytokineFeedbackGain -- scaled to the larger field; ~2x storm boost
            .SetHz( 60.0f );
    }
} // namespace Gaudi::Demos
