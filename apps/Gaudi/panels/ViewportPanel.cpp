#include "ViewportPanel.h"

#include <DigitalTwin.h>
#include <algorithm>
#include <imgui.h>

namespace Gaudi
{
    ViewportPanel::ViewportPanel( DigitalTwin::DigitalTwin& engine )
        : EditorPanel( "Scene Viewport" )
        , m_engine( engine )
    {
    }

    void ViewportPanel::OnUIRender()
    {
        ImGui::PushStyleVar( ImGuiStyleVar_WindowPadding, ImVec2( 0.0f, 0.0f ) );
        if( ImGui::Begin( m_name.c_str() ) )
        {
            ImVec2 size = ImGui::GetContentRegionAvail();

            uint32_t newWidth  = std::max( 1u, static_cast<uint32_t>( size.x ) );
            uint32_t newHeight = std::max( 1u, static_cast<uint32_t>( size.y ) );

            uint32_t currentWidth = 0, currentHeight = 0;
            m_engine.GetViewportSize( currentWidth, currentHeight );

            if( newWidth != currentWidth || newHeight != currentHeight )
            {
                m_engine.SetViewportSize( newWidth, newHeight );
            }

            void* texID = m_engine.GetSceneTextureID();
            if( texID )
            {
                ImGui::Image( texID, size );

                // Only forward mouse input to the camera when the viewport image
                // is the hovered widget. This is how the editor routes scroll/MMB
                // to the camera without conflicting with other panels.
                const bool hovered = ImGui::IsItemHovered();
                if( hovered )
                    DriveCamera();
            }
        }
        ImGui::End();
        ImGui::PopStyleVar();
    }

    void ViewportPanel::DriveCamera()
    {
        ImGuiIO& io = ImGui::GetIO();

        // Scroll → zoom.
        if( io.MouseWheel != 0.0f )
            m_engine.ZoomCamera( io.MouseWheel );

        // Middle-mouse drag → orbit (or pan with Shift).
        if( ImGui::IsMouseDragging( ImGuiMouseButton_Middle, 0.0f ) )
        {
            const ImVec2 d = io.MouseDelta;
            if( d.x != 0.0f || d.y != 0.0f )
            {
                if( io.KeyShift )
                    m_engine.PanCamera( d.x, d.y );
                else
                    m_engine.OrbitCamera( d.x, d.y );
            }
        }
    }
} // namespace Gaudi
