#include "SimulationControlsPanel.h"

#include <DigitalTwin.h>
#include <imgui.h>

namespace Gaudi
{
    SimulationControlsPanel::SimulationControlsPanel( DigitalTwin::DigitalTwin& engine, DigitalTwin::SimulationBlueprint& blueprint )
        : EditorPanel( "Simulation Controls" )
        , m_engine( engine )
        , m_blueprint( blueprint )
    {
    }

    void SimulationControlsPanel::OnUIRender()
    {
        ImGui::Begin( m_name.c_str() );

        DigitalTwin::EngineState state = m_engine.GetState();

        if( state == DigitalTwin::EngineState::STOPPED )
        {
            ImGui::TextColored( ImVec4( 0.6f, 0.6f, 0.6f, 1.0f ), "Mode: EDIT (Ready to allocate)" );
            ImGui::Separator();

            ImGui::PushStyleColor( ImGuiCol_Button, ImVec4( 0.15f, 0.6f, 0.15f, 1.0f ) );
            ImGui::PushStyleColor( ImGuiCol_ButtonHovered, ImVec4( 0.2f, 0.8f, 0.2f, 1.0f ) );
            if( ImGui::Button( "  >  PLAY  ", ImVec2( 120, 35 ) ) )
            {
                m_engine.Play();
            }
            ImGui::PopStyleColor( 2 );
        }
        else if( state == DigitalTwin::EngineState::PLAYING )
        {
            ImGui::TextColored( ImVec4( 0.2f, 0.8f, 0.2f, 1.0f ), "Mode: RUNNING" );
            ImGui::Separator();

            ImGui::PushStyleColor( ImGuiCol_Button, ImVec4( 0.8f, 0.6f, 0.1f, 1.0f ) );
            ImGui::PushStyleColor( ImGuiCol_ButtonHovered, ImVec4( 0.9f, 0.7f, 0.2f, 1.0f ) );
            if( ImGui::Button( "  ||  PAUSE  ", ImVec2( 100, 35 ) ) )
            {
                m_engine.Pause();
            }
            ImGui::PopStyleColor( 2 );

            ImGui::SameLine();
            ImGui::PushStyleColor( ImGuiCol_Button, ImVec4( 0.7f, 0.2f, 0.2f, 1.0f ) );
            ImGui::PushStyleColor( ImGuiCol_ButtonHovered, ImVec4( 0.9f, 0.3f, 0.3f, 1.0f ) );
            if( ImGui::Button( "  []  STOP  ", ImVec2( 100, 35 ) ) )
            {
                m_engine.Stop();
            }
            ImGui::PopStyleColor( 2 );
        }
        else if( state == DigitalTwin::EngineState::PAUSED )
        {
            ImGui::TextColored( ImVec4( 0.8f, 0.8f, 0.2f, 1.0f ), "Mode: PAUSED" );
            ImGui::Separator();

            ImGui::PushStyleColor( ImGuiCol_Button, ImVec4( 0.15f, 0.6f, 0.15f, 1.0f ) );
            ImGui::PushStyleColor( ImGuiCol_ButtonHovered, ImVec4( 0.2f, 0.8f, 0.2f, 1.0f ) );
            if( ImGui::Button( "  >  RESUME  ", ImVec2( 100, 35 ) ) )
            {
                m_engine.Play();
            }
            ImGui::PopStyleColor( 2 );

            ImGui::SameLine();
            ImGui::PushStyleColor( ImGuiCol_Button, ImVec4( 0.7f, 0.2f, 0.2f, 1.0f ) );
            ImGui::PushStyleColor( ImGuiCol_ButtonHovered, ImVec4( 0.9f, 0.3f, 0.3f, 1.0f ) );
            if( ImGui::Button( "  []  STOP  ", ImVec2( 100, 35 ) ) )
            {
                m_engine.Stop();
            }
            ImGui::PopStyleColor( 2 );
        }

        ImGui::Spacing();
        ImGui::Text( "Total Agent Groups: %zu", m_blueprint.GetGroups().size() );

        ImGui::End();
    }
} // namespace Gaudi