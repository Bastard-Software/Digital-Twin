#include "EnvironmentPanel.h"

#include <DigitalTwin.h>
#include <imgui.h>

namespace Gaudi
{
    EnvironmentPanel::EnvironmentPanel( DigitalTwin::DigitalTwin& engine, DigitalTwin::SimulationBlueprint& blueprint )
        : EditorPanel( "Environment Settings" )
        , m_engine( engine )
        , m_blueprint( blueprint )
    {
        // Load default settings from the engine
        m_settings = engine.GetGridVisualization();
    }

    void EnvironmentPanel::OnUIRender()
    {
        ImGui::Begin( m_name.c_str() );

        const auto& fields = m_blueprint.GetGridFields();
        if( fields.empty() )
        {
            ImGui::TextColored( ImVec4( 0.6f, 0.6f, 0.6f, 1.0f ), "No Grid Fields present in the blueprint." );
            ImGui::End();
            return;
        }

        // Main toggle for visualization overlay
        ImGui::Checkbox( "Enable Grid Visualization", &m_settings.active );

        if( m_settings.active )
        {
            ImGui::Separator();

            // Field selection combo box
            if( ImGui::BeginCombo( "Target Field", fields[ m_settings.fieldIndex ].GetName().c_str() ) )
            {
                for( int i = 0; i < fields.size(); ++i )
                {
                    bool isSelected = ( m_settings.fieldIndex == i );
                    if( ImGui::Selectable( fields[ i ].GetName().c_str(), isSelected ) )
                        m_settings.fieldIndex = i;

                    if( isSelected )
                        ImGui::SetItemDefaultFocus();
                }
                ImGui::EndCombo();
            }

            ImGui::Spacing();

            // Visualization Mode Radio Buttons
            ImGui::Text( "Visualization Mode:" );
            int modeInt = static_cast<int>( m_settings.mode );
            ImGui::RadioButton( "2D Slice (Heatmap)", &modeInt, 1 );
            ImGui::SameLine();
            ImGui::RadioButton( "Volumetric Cloud", &modeInt, 0 );
            m_settings.mode = static_cast<DigitalTwin::GridVisualizationMode>( modeInt );

            ImGui::Spacing();

            // Slicing control (only relevant if 2D Slice is active)
            if( m_settings.mode == DigitalTwin::GridVisualizationMode::SLICE_2D )
            {
                ImGui::SliderFloat( "Z-Depth Slice", &m_settings.sliceZ, 0.0f, 1.0f, "%.2f" );
                ImGui::SliderFloat( "Slice Opacity", &m_settings.opacitySlice, 0.0f, 1.0f, "%.3f" );
            }
            else // VOLUMETRIC_CLOUD
            {
                ImGui::SliderFloat( "Cloud Opacity", &m_settings.opacityCloud, 0.0f, 0.2f, "%.4f" );
            }

            // General appearance controls
            ImGui::ColorEdit4( "Base Color", &m_settings.colorMap.x );

            // Apply settings to the simulation engine
            m_engine.SetGridVisualization( m_settings );
        }

        ImGui::End();
    }
} // namespace Gaudi