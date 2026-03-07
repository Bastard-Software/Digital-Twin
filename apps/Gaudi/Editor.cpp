#include "Editor.h"

#include <core/Log.h>
#include <imgui.h>
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
        // Example simulation setup
        m_blueprint.AddAgentGroup( "CancerCells" )
            .SetCount( 50 )
            .SetMorphology( DigitalTwin::MorphologyGenerator::CreateCube( 3.0f ) )
            .SetDistribution( DigitalTwin::SpatialDistribution::UniformInBox( 50, glm::vec3( 20.0f ) ) )
            .SetColor( glm::vec4( 0.9f, 0.1f, 0.1f, 1.0f ) )
            .AddBehaviour( DigitalTwin::Behaviours::BrownianMotion{ 0.5f } )
            .SetHz( 30.0f );

        m_blueprint.AddAgentGroup( "T-Cells" )
            .SetCount( 500 )
            .SetMorphology( DigitalTwin::MorphologyGenerator::CreateCube( 1.0f ) )
            .SetDistribution( DigitalTwin::SpatialDistribution::UniformInSphere( 500, 75.0f ) )
            .SetColor( glm::vec4( 0.2f, 0.8f, 0.3f, 1.0f ) )
            .AddBehaviour( DigitalTwin::Behaviours::BrownianMotion{ 5.0f } )
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