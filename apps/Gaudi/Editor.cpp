#include "Editor.h"

#include "IconsFontAwesome5.h"
#include <core/FileSystem.h>
#include <core/Log.h>
#include <imgui.h>
#include <random>
#include <simulation/BiologyGenerator.h>
#include <simulation/BiomechanicsGenerator.h>
#include <simulation/MorphologyGenerator.h>
#include <simulation/Phenotype.h>
#include <simulation/SpatialDistribution.h>

namespace Gaudi
{
    void Editor::Init()
    {
        DigitalTwin::DigitalTwinConfig config;
        config.headless        = false;
        config.windowDesc.mode = DigitalTwin::WindowMode::FULLSCREEN_WINDOWED;
        m_engine.Initialize( config );

        ImGui::SetCurrentContext( static_cast<ImGuiContext*>( m_engine.GetImGuiContext() ) );
        ImGuiIO&    io      = ImGui::GetIO();
        auto*       fs      = m_engine.GetFileSystem();
        std::string roboto  = fs->ResolvePath( "fonts/Roboto-Medium.ttf" ).string();
        std::string faSolid = fs->ResolvePath( "fonts/fa-solid-900.ttf" ).string();
        io.Fonts->AddFontFromFileTTF( roboto.c_str(), 16.0f );
        ImFontConfig iconCfg;
        iconCfg.MergeMode                 = true;
        iconCfg.GlyphMinAdvanceX          = 16.0f;
        iconCfg.PixelSnapH                = true;
        static const ImWchar iconRanges[] = { ICON_MIN_FA, ICON_MAX_FA, 0 };
        io.Fonts->AddFontFromFileTTF( faSolid.c_str(), 16.0f, &iconCfg, iconRanges );

        SetupInitialBlueprint();
        // SetupChemotaxisDemo();
    }

    void Editor::SetupInitialBlueprint()
    {
        // ==========================================================================================
        // 1. Domain & Spatial Partitioning Setup
        // ==========================================================================================
        m_blueprint.SetName( "Tumor Growth" );
        m_blueprint.SetDomainSize( glm::vec3( 50.0f ), 2.0f );

        m_blueprint.ConfigureSpatialPartitioning()
            .SetMethod( DigitalTwin::SpatialPartitioningMethod::HashGrid )
            .SetCellSize( 3.0f )
            .SetMaxDensity( 64 )
            .SetComputeHz( 60.0f );

        // ==========================================================================================
        // 2. Environmental Fields (PDEs)
        // ==========================================================================================

        // A. OXYGEN: Starts concentrated in the center, consumed by cells.
        m_blueprint.AddGridField( "Oxygen" )
            .SetInitializer( DigitalTwin::GridInitializer::Gaussian( glm::vec3( 0.0f, 0.0f, 0.0f ), 15.0f, 80.0f ) )
            .SetDiffusionCoefficient( 5.0f )
            .SetDecayRate( 0.0f ) // No natural decay, only consumed by cells
            .SetComputeHz( 60.0f );

        // B. VEGF (Vascular Endothelial Growth Factor): The SOS signal!
        // Starts completely empty (0.0). Diffuses slowly and decays over time.
        m_blueprint.AddGridField( "VEGF" )
            .SetInitializer( DigitalTwin::GridInitializer::Constant( 0.0f ) )
            .SetDiffusionCoefficient( 2.0f )
            .SetDecayRate( 0.05f )
            .SetComputeHz( 60.0f );

        // ==========================================================================================
        // 3. Agent Groups (Patient Zero)
        // ==========================================================================================
        std::vector<glm::vec4> patientZero = { glm::vec4( 0.0f, 0.0f, 0.0f, 1.0f ) };

        auto& tumorCells = m_blueprint.AddAgentGroup( "TumorCells" )
                               .SetCount( 1 )
                               .SetMorphology( DigitalTwin::MorphologyGenerator::CreateSphere( 1.5f ) )
                               .SetDistribution( patientZero )
                               .SetColor( glm::vec4( 0.2f, 0.8f, 0.3f, 1.0f ) ); // Bright green

        // ==========================================================================================
        // 4. Behaviours (Physics & Biology)
        // ==========================================================================================

        // A. Brownian motion for organic, fluid-like movement
        tumorCells.AddBehaviour( DigitalTwin::Behaviours::BrownianMotion{ 0.5f } ).SetHz( 60.0f );

        // B. Oxygen Consumption: Moderately high to ensure the core starves as the tumor grows
        tumorCells.AddBehaviour( DigitalTwin::Behaviours::ConsumeField{ "Oxygen", 20.0f } ).SetHz( 60.0f );

        // C. VEGF Secretion: Conditioned on State Hypoxic.
        // Cells will only emit this field when they are starving for oxygen!
        tumorCells.AddBehaviour( DigitalTwin::Behaviours::SecreteField{ "VEGF", 100.0f, static_cast<int>( DigitalTwin::LifecycleState::Hypoxic ) } )
            .SetHz( 60.0f );

        // D. Biology (Cell Cycle): Oxygen-driven proliferation & Hypoxia trigger
        tumorCells
            .AddBehaviour( DigitalTwin::BiologyGenerator::StandardCellCycle()
                               .SetBaseDoublingTime( 5.0f / 3600.0f ) // Rapid division for demonstration
                               .SetProliferationOxygenTarget( 50.0f )
                               .SetArrestPressureThreshold( 15.0f )
                               .SetHypoxiaOxygenThreshold( 25.0f )  // If O2 < 25.0, cell becomes Hypoxic (State 2)
                               .SetNecrosisOxygenThreshold( 22.0f ) // If O2 < 22.0, cell dies
                               .SetApoptosisRate( 0.0f )
                               .Build() )
            .SetHz( 60.0f );

        // E. Biomechanics (JKR Model): Softer cells with less adhesion
        tumorCells
            .AddBehaviour( DigitalTwin::BiomechanicsGenerator::JKR()
                               .SetYoungsModulus( 20.0f )
                               .SetPoissonRatio( 0.4f )
                               .SetAdhesionEnergy( 1.5f )
                               .SetMaxInteractionRadius( 1.5f )
                               .Build() )
            .SetHz( 60.0f );

        // ==========================================================================================
        // 5. Endothelial Tip Cells — migrate toward VEGF secreted by Hypoxic tumour cells
        // ==========================================================================================

        // Pre-formed vessel line — horizontal segment offset above tumour center
        auto& endo = m_blueprint.AddAgentGroup( "EndothelialCells" )
                         .SetCount( 20 )
                         .SetMorphology( DigitalTwin::MorphologyGenerator::CreateSphere( 1.0f ) )
                         .SetDistribution( DigitalTwin::SpatialDistribution::VesselLine( 20, glm::vec3( -15, 5, 0 ), glm::vec3( 15, 5, 0 ) ) )
                         .SetColor( glm::vec4( 1.0f, 0.3f, 0.3f, 1.0f ) ); // Red

        // Notch-Dll4 lateral inhibition — differentiates Tip vs Stalk cells each frame
        endo.AddBehaviour( DigitalTwin::Behaviours::NotchDll4{
                /* dll4ProductionRate   */ 1.0f,
                /* dll4DecayRate        */ 0.1f,
                /* notchInhibitionGain  */ 1.0f,
                /* vegfr2BaseExpression */ 1.0f,
                /* tipThreshold         */ 0.8f,
                /* stalkThreshold       */ 0.3f } )
            .SetHz( 60.0f );

        // Chemotaxis toward VEGF — MUST be before Biomechanics for correct ComputeGraph order
        endo.AddBehaviour( DigitalTwin::Behaviours::Chemotaxis{ "VEGF", 2.0f, 0.005f, 8.0f } ).SetHz( 60.0f );
        endo.AddBehaviour( DigitalTwin::Behaviours::BrownianMotion{ 0.1f } ).SetHz( 60.0f );
        endo.AddBehaviour( DigitalTwin::BiomechanicsGenerator::JKR()
                               .SetYoungsModulus( 15.0f )
                               .SetPoissonRatio( 0.4f )
                               .SetAdhesionEnergy( 1.0f )
                               .SetMaxInteractionRadius( 1.5f )
                               .Build() )
            .SetHz( 60.0f );

        m_engine.SetBlueprint( m_blueprint );
    }

    void Editor::AddPanel( std::shared_ptr<EditorPanel> panel )
    {
        m_panels.push_back( panel );
        panel->OnAttach();
    }

    void Editor::Run()
    {
        DT_INFO( "Starting Gaudi Editor Loop..." );
        ImGui::SetCurrentContext( static_cast<ImGuiContext*>( m_engine.GetImGuiContext() ) );

        while( !m_engine.IsWindowClosed() )
        {
            m_engine.BeginFrame();

            // Pass a lambda that iterates over our panel list
            m_engine.RenderUI( [ this ]() {
                for( auto& panel: m_panels )
                {
                    panel->OnUIRender();
                }
            } );

            m_engine.EndFrame();
        }
    }

    void Editor::Shutdown()
    {
        for( auto& panel: m_panels )
        {
            panel->OnDetach();
        }
        m_panels.clear();

        DT_INFO( "Gaudi Editor closing..." );
        m_engine.Shutdown();
    }
    void Editor::SetupChemotaxisDemo()
    {
        // Fresh blueprint — wipe any previous state
        m_blueprint = DigitalTwin::SimulationBlueprint{};

        // ==========================================================================================
        // 1. Domain
        // ==========================================================================================
        m_blueprint.SetName( "Chemotaxis Demo" );
        m_blueprint.SetDomainSize( glm::vec3( 50.0f ), 2.0f );

        m_blueprint.ConfigureSpatialPartitioning()
            .SetMethod( DigitalTwin::SpatialPartitioningMethod::HashGrid )
            .SetCellSize( 3.0f )
            .SetMaxDensity( 64 )
            .SetComputeHz( 60.0f );

        // ==========================================================================================
        // 2. VEGF field — pre-seeded Gaussian at origin so gradient is visible from frame 1.
        //    Source cells top it up continuously; decay is minimal so the cloud stays stable.
        // ==========================================================================================
        m_blueprint.AddGridField( "VEGF" )
            .SetInitializer( DigitalTwin::GridInitializer::Gaussian( glm::vec3( 0.0f ), 8.0f, 200.0f ) )
            .SetDiffusionCoefficient( 0.8f )
            .SetDecayRate( 0.01f )
            .SetComputeHz( 60.0f );

        // ==========================================================================================
        // 3. HypoxicCores — 3 cells at the centre, always secreting VEGF (no CellCycle needed).
        //    Orange so they stand out as the attractant source.
        // ==========================================================================================
        std::vector<glm::vec4> corePositions = {
            glm::vec4( -1.5f,  0.5f, 0.0f, 1.0f ),
            glm::vec4(  1.5f,  0.5f, 0.0f, 1.0f ),
            glm::vec4(  0.0f, -1.0f, 0.0f, 1.0f ),
        };

        auto& cores = m_blueprint.AddAgentGroup( "HypoxicCores" )
                          .SetCount( 3 )
                          .SetMorphology( DigitalTwin::MorphologyGenerator::CreateSphere( 1.5f ) )
                          .SetDistribution( corePositions )
                          .SetColor( glm::vec4( 1.0f, 0.6f, 0.0f, 1.0f ) ); // Orange

        // Unconditional secretion — no requiredState so VEGF flows from simulation start
        cores.AddBehaviour( DigitalTwin::Behaviours::SecreteField{ "VEGF", 500.0f } ).SetHz( 60.0f );
        cores.AddBehaviour( DigitalTwin::BiomechanicsGenerator::JKR()
                                .SetYoungsModulus( 20.0f )
                                .SetPoissonRatio( 0.4f )
                                .SetAdhesionEnergy( 1.5f )
                                .SetMaxInteractionRadius( 1.5f )
                                .Build() )
            .SetHz( 60.0f );

        // ==========================================================================================
        // 4. EndothelialCells — ring of 8 cells at r ≈ 18 in the XZ plane.
        //    No BrownianMotion so directed migration is unambiguous.
        //    Red so the inward collapse toward the orange source is visually obvious.
        // ==========================================================================================
        const float r = 18.0f;
        std::vector<glm::vec4> endoPositions = {
            glm::vec4(  r,     0.0f,  0.0f,  1.0f ),
            glm::vec4(  r * 0.707f, 0.0f,  r * 0.707f, 1.0f ),
            glm::vec4(  0.0f,  0.0f,  r,     1.0f ),
            glm::vec4( -r * 0.707f, 0.0f,  r * 0.707f, 1.0f ),
            glm::vec4( -r,     0.0f,  0.0f,  1.0f ),
            glm::vec4( -r * 0.707f, 0.0f, -r * 0.707f, 1.0f ),
            glm::vec4(  0.0f,  0.0f, -r,     1.0f ),
            glm::vec4(  r * 0.707f, 0.0f, -r * 0.707f, 1.0f ),
        };

        auto& endo = m_blueprint.AddAgentGroup( "EndothelialCells" )
                         .SetCount( 8 )
                         .SetMorphology( DigitalTwin::MorphologyGenerator::CreateSphere( 1.0f ) )
                         .SetDistribution( endoPositions )
                         .SetColor( glm::vec4( 1.0f, 0.2f, 0.2f, 1.0f ) ); // Red

        // Chemotaxis BEFORE Biomechanics — gradient movement resolved first, then collision
        endo.AddBehaviour( DigitalTwin::Behaviours::Chemotaxis{ "VEGF", 6.0f, 0.001f, 15.0f } ).SetHz( 60.0f );
        endo.AddBehaviour( DigitalTwin::BiomechanicsGenerator::JKR()
                               .SetYoungsModulus( 15.0f )
                               .SetPoissonRatio( 0.4f )
                               .SetAdhesionEnergy( 1.0f )
                               .SetMaxInteractionRadius( 1.5f )
                               .Build() )
            .SetHz( 60.0f );

        m_engine.SetBlueprint( m_blueprint );
    }

} // namespace Gaudi