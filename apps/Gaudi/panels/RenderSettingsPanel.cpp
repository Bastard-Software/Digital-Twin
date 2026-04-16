#include "RenderSettingsPanel.h"

#include <DigitalTwin.h>
#include <imgui.h>

namespace Gaudi
{
    RenderSettingsPanel::RenderSettingsPanel( DigitalTwin::DigitalTwin& engine )
        : EditorPanel( "Render Settings" )
        , m_engine( engine )
    {
    }

    void RenderSettingsPanel::DrawSection( const char* label, std::function<void()> body )
    {
        if( ImGui::CollapsingHeader( label, ImGuiTreeNodeFlags_DefaultOpen ) )
        {
            ImGui::Indent( 8.0f );
            body();
            ImGui::Unindent( 8.0f );
            ImGui::Spacing();
        }
    }

    void RenderSettingsPanel::OnUIRender()
    {
        if( !m_visible )
            return;

        ImGui::SetNextWindowSize( ImVec2( 300, 0 ), ImGuiCond_Appearing );
        if( !ImGui::Begin( "Render Settings", &m_visible ) )
        {
            ImGui::End();
            return;
        }

        // ── Anti-Aliasing ─────────────────────────────────────────────────
        DrawSection( "Anti-Aliasing", [&]() {
            uint32_t current = m_engine.GetMSAA();
            uint32_t maxMSAA = m_engine.GetMaxMSAA();

            if( ImGui::RadioButton( "Off", current == 1 ) )
                m_engine.SetMSAA( 1 );

            ImGui::SameLine();

            const bool msaa4Ok = ( maxMSAA >= 4 );
            if( !msaa4Ok )
                ImGui::BeginDisabled();

            if( ImGui::RadioButton( "4\xc3\x97 MSAA", current == 4 ) )
                m_engine.SetMSAA( 4 );

            if( !msaa4Ok )
            {
                ImGui::EndDisabled();
                if( ImGui::IsItemHovered( ImGuiHoveredFlags_AllowWhenDisabled ) )
                    ImGui::SetTooltip( "4x MSAA not supported by this GPU" );
            }
        } );

        // ── Present ───────────────────────────────────────────────────────
        DrawSection( "Present", [&]() {
            bool vsync = m_engine.GetVSync();
            if( ImGui::Checkbox( "V-Sync", &vsync ) )
                m_engine.SetVSync( vsync );
        } );

        // ── Reserved: future sections (uncomment to add) ──────────────────
        // DrawSection( "Debug Overlays", [&]() {
        //     // Show Normals / Forces / Velocity / Cell Types toggles
        //     // wired to per-pass debug push constants
        // } );
        // DrawSection( "Resolution Scale", [&]() {
        //     // 0.5x / 0.75x / 1x / 1.5x / 2x radio
        // } );
        // DrawSection( "Post Process", [&]() {
        //     // Gamma, tone map, clear colour
        // } );

        ImGui::End();
    }

} // namespace Gaudi
