#include "SimulationControlsPanel.h"
#include "../IconsFontAwesome5.h"

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

        // Status dot + label
        ImVec4 dotColor;
        const char* stateLabel;
        if( state == DigitalTwin::EngineState::PLAYING )
        {
            dotColor   = ImVec4( 0.2f, 0.8f, 0.2f, 1.0f );
            stateLabel = "RUNNING";
        }
        else if( state == DigitalTwin::EngineState::PAUSED )
        {
            dotColor   = ImVec4( 0.9f, 0.7f, 0.1f, 1.0f );
            stateLabel = "PAUSED";
        }
        else
        {
            dotColor   = ImVec4( 0.5f, 0.5f, 0.5f, 1.0f );
            stateLabel = "RESET";
        }

        ImGui::TextColored( dotColor, "%s", stateLabel );

        ImGui::Separator();

        // Centered button group
        const float buttonSize  = 38.0f;
        const float spacing     = ImGui::GetStyle().ItemSpacing.x;
        const float groupWidth  = buttonSize * 3.0f + spacing * 2.0f;
        const float windowWidth = ImGui::GetContentRegionAvail().x;
        ImGui::SetCursorPosX( ImGui::GetCursorPosX() + ( windowWidth - groupWidth ) * 0.5f );

        ImGui::PushStyleVar( ImGuiStyleVar_FrameRounding, 8.0f );

        // Play / Resume button
        bool playDisabled = ( state == DigitalTwin::EngineState::PLAYING );
        if( playDisabled )
            ImGui::BeginDisabled();
        ImGui::PushStyleColor( ImGuiCol_Button, ImVec4( 0.15f, 0.6f, 0.15f, 1.0f ) );
        ImGui::PushStyleColor( ImGuiCol_ButtonHovered, ImVec4( 0.2f, 0.8f, 0.2f, 1.0f ) );
        if( ImGui::Button( ICON_FA_PLAY "##play", ImVec2( buttonSize, buttonSize ) ) )
        {
            m_engine.Play();
        }
        ImGui::PopStyleColor( 2 );
        if( playDisabled )
            ImGui::EndDisabled();
        if( ImGui::IsItemHovered( ImGuiHoveredFlags_AllowWhenDisabled ) )
            ImGui::SetTooltip( state == DigitalTwin::EngineState::PAUSED ? "Resume" : "Play" );

        ImGui::SameLine();

        // Pause button
        bool pauseDisabled = ( state != DigitalTwin::EngineState::PLAYING );
        if( pauseDisabled )
            ImGui::BeginDisabled();
        ImGui::PushStyleColor( ImGuiCol_Button, ImVec4( 0.7f, 0.5f, 0.05f, 1.0f ) );
        ImGui::PushStyleColor( ImGuiCol_ButtonHovered, ImVec4( 0.9f, 0.7f, 0.1f, 1.0f ) );
        if( ImGui::Button( ICON_FA_PAUSE "##pause", ImVec2( buttonSize, buttonSize ) ) )
        {
            m_engine.Pause();
        }
        ImGui::PopStyleColor( 2 );
        if( pauseDisabled )
            ImGui::EndDisabled();
        if( ImGui::IsItemHovered( ImGuiHoveredFlags_AllowWhenDisabled ) )
            ImGui::SetTooltip( "Pause" );

        ImGui::SameLine();

        // Reset button
        bool resetDisabled = ( state == DigitalTwin::EngineState::RESET );
        if( resetDisabled )
            ImGui::BeginDisabled();
        ImGui::PushStyleColor( ImGuiCol_Button, ImVec4( 0.6f, 0.15f, 0.15f, 1.0f ) );
        ImGui::PushStyleColor( ImGuiCol_ButtonHovered, ImVec4( 0.9f, 0.3f, 0.3f, 1.0f ) );
        if( ImGui::Button( ICON_FA_SYNC "##reset", ImVec2( buttonSize, buttonSize ) ) )
        {
            m_engine.Stop();
        }
        ImGui::PopStyleColor( 2 );
        if( resetDisabled )
            ImGui::EndDisabled();
        if( ImGui::IsItemHovered( ImGuiHoveredFlags_AllowWhenDisabled ) )
            ImGui::SetTooltip( "Reset Simulation" );

        ImGui::PopStyleVar();

        ImGui::End();
    }
} // namespace Gaudi
