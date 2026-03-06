#include <DigitalTwin.h>
#include <core/Log.h>
#include <imgui.h>
#include <iostream>
#include <platform/Input.h>
#include <simulation/MorphologyGenerator.h>
#include <simulation/SpatialDistribution.h>

int main()
{
    DigitalTwin::DigitalTwin       engine;
    DigitalTwin::DigitalTwinConfig config;
    config.headless        = false;
    config.windowDesc.mode = DigitalTwin::WindowMode::FULLSCREEN_WINDOWED;
    engine.Initialize( config );

    // Setting up simulation
    DigitalTwin::SimulationBlueprint blueprint;
    blueprint.AddAgentGroup( "CancerCells" )
        .SetCount( 50 )
        .SetMorphology( DigitalTwin::MorphologyGenerator::CreateCube( 3.0f ) )
        .SetDistribution( DigitalTwin::SpatialDistribution::UniformInBox( 50, glm::vec3( 20.0f ) ) )
        .SetColor( glm::vec4( 0.9f, 0.1f, 0.1f, 1.0f ) ) // RED
        .AddBehaviour( DigitalTwin::Behaviours::BrownianMotion{ 0.5f } )
        .SetHz( 30.0f );
    blueprint.AddAgentGroup( "T-Cells" )
        .SetCount( 500 )
        .SetMorphology( DigitalTwin::MorphologyGenerator::CreateCube( 1.0f ) )
        .SetDistribution( DigitalTwin::SpatialDistribution::UniformInSphere( 500, 75.0f ) )
        .SetColor( glm::vec4( 0.2f, 0.8f, 0.3f, 1.0f ) ) // GREEN
        .AddBehaviour( DigitalTwin::Behaviours::BrownianMotion{ 5.0f } )
        .SetHz( 60.0f );

    // Inform engine about the setup
    engine.SetBlueprint( blueprint );

    // Main Engine Loop
    DT_INFO( "Starting Editor..." );
    ImGui::SetCurrentContext( ( ImGuiContext* )engine.GetImGuiContext() );

    while( !engine.IsWindowClosed() )
    {
        const auto& ctx = engine.BeginFrame();

        engine.RenderUI( [ & ]() {
            // SIMULATION CONTROLS PANEL
            ImGui::Begin( "Simulation Controls" );

            DigitalTwin::EngineState state = engine.GetState();

            if( state == DigitalTwin::EngineState::STOPPED )
            {
                ImGui::TextColored( ImVec4( 0.6f, 0.6f, 0.6f, 1.0f ), "Mode: EDIT (Ready to allocate)" );
                ImGui::Separator();

                ImGui::PushStyleColor( ImGuiCol_Button, ImVec4( 0.15f, 0.6f, 0.15f, 1.0f ) );
                ImGui::PushStyleColor( ImGuiCol_ButtonHovered, ImVec4( 0.2f, 0.8f, 0.2f, 1.0f ) );
                if( ImGui::Button( "  >  PLAY  ", ImVec2( 120, 35 ) ) )
                    engine.Play();
                ImGui::PopStyleColor( 2 );
            }
            else if( state == DigitalTwin::EngineState::PLAYING )
            {
                ImGui::TextColored( ImVec4( 0.2f, 0.8f, 0.2f, 1.0f ), "Mode: RUNNING" );
                ImGui::Separator();

                ImGui::PushStyleColor( ImGuiCol_Button, ImVec4( 0.8f, 0.6f, 0.1f, 1.0f ) );
                ImGui::PushStyleColor( ImGuiCol_ButtonHovered, ImVec4( 0.9f, 0.7f, 0.2f, 1.0f ) );
                if( ImGui::Button( "  ||  PAUSE  ", ImVec2( 100, 35 ) ) )
                    engine.Pause();
                ImGui::PopStyleColor( 2 );

                ImGui::SameLine();
                ImGui::PushStyleColor( ImGuiCol_Button, ImVec4( 0.7f, 0.2f, 0.2f, 1.0f ) );
                ImGui::PushStyleColor( ImGuiCol_ButtonHovered, ImVec4( 0.9f, 0.3f, 0.3f, 1.0f ) );
                if( ImGui::Button( "  []  STOP  ", ImVec2( 100, 35 ) ) )
                    engine.Stop();
                ImGui::PopStyleColor( 2 );
            }
            else if( state == DigitalTwin::EngineState::PAUSED )
            {
                ImGui::TextColored( ImVec4( 0.8f, 0.8f, 0.2f, 1.0f ), "Mode: PAUSED" );
                ImGui::Separator();

                ImGui::PushStyleColor( ImGuiCol_Button, ImVec4( 0.15f, 0.6f, 0.15f, 1.0f ) );
                ImGui::PushStyleColor( ImGuiCol_ButtonHovered, ImVec4( 0.2f, 0.8f, 0.2f, 1.0f ) );
                if( ImGui::Button( "  >  RESUME  ", ImVec2( 100, 35 ) ) )
                    engine.Play();
                ImGui::PopStyleColor( 2 );

                ImGui::SameLine();
                ImGui::PushStyleColor( ImGuiCol_Button, ImVec4( 0.7f, 0.2f, 0.2f, 1.0f ) );
                ImGui::PushStyleColor( ImGuiCol_ButtonHovered, ImVec4( 0.9f, 0.3f, 0.3f, 1.0f ) );
                if( ImGui::Button( "  []  STOP  ", ImVec2( 100, 35 ) ) )
                    engine.Stop();
                ImGui::PopStyleColor( 2 );
            }

            ImGui::Spacing();
            ImGui::Text( "Total Agent Groups: %zu", blueprint.GetGroups().size() );
            ImGui::End();
        } );

        engine.EndFrame();
    }

    DT_INFO( "Editor closing..." );
    engine.Shutdown();

    return 0;
}