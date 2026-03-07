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
            }
        }
        ImGui::End();
        ImGui::PopStyleVar();
    }
} // namespace Gaudi