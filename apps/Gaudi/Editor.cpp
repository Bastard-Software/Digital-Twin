#include "Editor.h"

#include <core/Log.h>
#include <imgui.h>
#include <random>
#include <simulation/MorphologyGenerator.h>
#include <simulation/SpatialDistribution.h>

namespace Gaudi
{
    void Editor::Init()
    {
        DigitalTwin::DigitalTwinConfig config;
        config.headless        = false;
        config.windowDesc.mode = DigitalTwin::WindowMode::FULLSCREEN_WINDOWED;
        m_engine.Initialize( config );

        SetupInitialBlueprint();
    }

    void Editor::SetupInitialBlueprint()
    {
        // 1. Define physical simulation domain (200x200x200 micrometers)
        m_blueprint.SetDomainSize( glm::vec3( 200.0f ), 2.0f );

        // 2A. Add Oxygen Field (O2) - Blood vessels network
        std::vector<glm::vec3>                bloodVessels;
        std::mt19937                          rng( 42 );
        std::uniform_real_distribution<float> dist( -60.0f, 60.0f );
        for( int i = 0; i < 8; ++i )
        {
            bloodVessels.push_back( glm::vec3( dist( rng ), dist( rng ), dist( rng ) ) );
        }

        m_blueprint.AddGridField( "Oxygen" )
            .SetInitializer( DigitalTwin::GridInitializer::MultiGaussian( bloodVessels, 25.0f, 100.0f ) )
            .SetDiffusionCoefficient( 0.4f )
            .SetDecayRate( 0.005f )
            .SetComputeHz( 120.0f );

        // 2B. Add Lactate Field (Acid byproduct)
        // It starts completely empty. Cells will produce it.
        m_blueprint.AddGridField( "Lactate" )
            .SetInitializer( DigitalTwin::GridInitializer::Constant( 0.0f ) )
            .SetDiffusionCoefficient( 0.8f ) // Diffuses relatively quickly
            .SetDecayRate( 0.01f )           // Slowly washes away into the bloodstream
            .SetComputeHz( 120.0f );

        // 3. Add Tumor Cells
        auto& tumorCells = m_blueprint.AddAgentGroup( "TumorCells" )
                               .SetCount( 8000 )
                               .SetMorphology( DigitalTwin::MorphologyGenerator::CreateSphere( 1.5f ) )
                               .SetDistribution( DigitalTwin::SpatialDistribution::UniformInSphere( 8000, 25.0f ) )
                               .SetColor( glm::vec4( 0.1f, 0.8f, 0.2f, 1.0f ) );

        // 4. Attach Multiple Biological Behaviours
        tumorCells.AddBehaviour( DigitalTwin::Behaviours::BrownianMotion{ 0.3f } ).SetHz( 60.0f );

        // Consume Oxygen from the "Oxygen" grid
        tumorCells.AddBehaviour( DigitalTwin::Behaviours::ConsumeField{ "Oxygen", 8.0f } ).SetHz( 60.0f );

        // Secrete Acid into the "Lactate" grid
        tumorCells.AddBehaviour( DigitalTwin::Behaviours::SecreteField{ "Lactate", 15.0f } ).SetHz( 60.0f );

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