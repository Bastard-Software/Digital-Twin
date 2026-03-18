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
    }

    void Editor::SetupInitialBlueprint()
    {
        // ==========================================================================================
        // 1. Domain & Spatial Partitioning Setup
        // ==========================================================================================
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
        tumorCells.AddBehaviour( DigitalTwin::Behaviours::SecreteField{ "VEGF", 100.0f, static_cast<int>( DigitalTwin::PhenotypeState::Hypoxic ) } )
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
} // namespace Gaudi