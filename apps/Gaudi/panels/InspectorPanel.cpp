#include "InspectorPanel.h"

#include <DigitalTwin.h>
#include <imgui.h>
#include <simulation/Behaviours.h>
#include <simulation/GridField.h>

namespace Gaudi
{
    InspectorPanel::InspectorPanel( DigitalTwin::DigitalTwin& engine, DigitalTwin::SimulationBlueprint& blueprint, EditorSelection& selection )
        : EditorPanel( "Inspector" )
        , m_engine( engine )
        , m_blueprint( blueprint )
        , m_selection( selection )
    {
    }

    void InspectorPanel::RenderGroupInspector( DigitalTwin::AgentGroup& group )
    {
        ImGui::TextDisabled( "Agent Group" );
        ImGui::Separator();

        ImGui::LabelText( "Name", "%s", group.GetName().c_str() );

        int count = static_cast<int>( group.GetCount() );
        if( ImGui::InputInt( "Count", &count ) )
        {
            if( count < 1 )
                count = 1;
            group.SetCount( static_cast<uint32_t>( count ) );
        }

        glm::vec4 color = group.GetColor();
        if( ImGui::ColorEdit4( "Color", &color.x ) )
        {
            group.SetColor( color );
        }
    }

    void InspectorPanel::RenderBehaviourInspector( DigitalTwin::BehaviourRecord& record )
    {
        ImGui::TextDisabled( "Behaviour" );
        ImGui::Separator();

        std::visit(
            [ & ]( auto& b ) {
                using T = std::decay_t<decltype( b )>;

                if constexpr( std::is_same_v<T, DigitalTwin::Behaviours::BrownianMotion> )
                {
                    ImGui::SliderFloat( "Speed", &b.speed, 0.0f, 10.0f );
                }
                else if constexpr( std::is_same_v<T, DigitalTwin::Behaviours::ConsumeField> )
                {
                    ImGui::LabelText( "Field", "%s", b.fieldName.c_str() );
                    ImGui::SliderFloat( "Rate", &b.rate, 0.0f, 200.0f );
                    ImGui::InputInt( "Required State", &b.requiredState );
                    if( b.requiredState < -1 )
                        b.requiredState = -1;
                }
                else if constexpr( std::is_same_v<T, DigitalTwin::Behaviours::SecreteField> )
                {
                    ImGui::LabelText( "Field", "%s", b.fieldName.c_str() );
                    ImGui::SliderFloat( "Rate", &b.rate, 0.0f, 200.0f );
                    ImGui::InputInt( "Required State", &b.requiredState );
                    if( b.requiredState < -1 )
                        b.requiredState = -1;
                }
                else if constexpr( std::is_same_v<T, DigitalTwin::Behaviours::Biomechanics> )
                {
                    ImGui::SliderFloat( "Repulsion Stiffness", &b.repulsionStiffness, 0.0f, 50.0f );
                    ImGui::SliderFloat( "Adhesion Strength", &b.adhesionStrength, 0.0f, 20.0f );
                    ImGui::SliderFloat( "Max Radius", &b.maxRadius, 0.5f, 5.0f );
                }
                else if constexpr( std::is_same_v<T, DigitalTwin::Behaviours::CellCycle> )
                {
                    ImGui::SliderFloat( "Growth Rate /sec", &b.growthRatePerSec, 0.0f, 0.01f, "%.6f" );
                    ImGui::SliderFloat( "Target O2", &b.targetO2, 0.0f, 100.0f );
                    ImGui::SliderFloat( "Arrest Pressure", &b.arrestPressure, 0.0f, 50.0f );
                    ImGui::SliderFloat( "Necrosis O2", &b.necrosisO2, 0.0f, 50.0f );
                    ImGui::SliderFloat( "Hypoxia O2", &b.hypoxiaO2, 0.0f, 50.0f );
                    ImGui::SliderFloat( "Apoptosis /sec", &b.apoptosisProbPerSec, 0.0f, 0.01f, "%.6f" );
                }
            },
            record.behaviour );

        ImGui::Spacing();
        ImGui::SliderFloat( "Target Hz", &record.targetHz, 1.0f, 120.0f );
    }

    void InspectorPanel::RenderGridFieldInspector( DigitalTwin::GridField& field, int fieldIndex )
    {
        ImGui::TextDisabled( "Grid Field" );
        ImGui::Separator();

        ImGui::LabelText( "Name", "%s", field.GetName().c_str() );

        // ── Blueprint parameters (edit only in RESET state) ───────────
        const bool blueprintEditable = ( m_engine.GetState() == DigitalTwin::EngineState::RESET );
        if( !blueprintEditable )
            ImGui::BeginDisabled();

        float diffusion = field.GetDiffusionCoefficient();
        if( ImGui::SliderFloat( "Diffusion", &diffusion, 0.0f, 20.0f ) )
            field.SetDiffusionCoefficient( diffusion );

        float decay = field.GetDecayRate();
        if( ImGui::SliderFloat( "Decay Rate", &decay, 0.0f, 1.0f, "%.4f" ) )
            field.SetDecayRate( decay );

        float hz = field.GetComputeHz();
        if( ImGui::SliderFloat( "Compute Hz", &hz, 1.0f, 120.0f ) )
            field.SetComputeHz( hz );

        if( !blueprintEditable )
            ImGui::EndDisabled();

        // ── Visualization settings (always editable) ──────────────────
        ImGui::Spacing();
        ImGui::SeparatorText( "Visualization" );

        DigitalTwin::GridVisualizationSettings vis = m_engine.GetGridVisualization();
        bool isActive = vis.active && ( vis.fieldIndex == fieldIndex );

        if( ImGui::Checkbox( "Show this field", &isActive ) )
        {
            vis.active     = isActive;
            vis.fieldIndex = fieldIndex;
            m_engine.SetGridVisualization( vis );
        }

        if( isActive )
        {
            // Sync fieldIndex in case we just turned it on
            vis.fieldIndex = fieldIndex;

            int mode = static_cast<int>( vis.mode );
            ImGui::RadioButton( "2D Slice", &mode, static_cast<int>( DigitalTwin::GridVisualizationMode::SLICE_2D ) );
            ImGui::SameLine();
            ImGui::RadioButton( "Volumetric", &mode, static_cast<int>( DigitalTwin::GridVisualizationMode::VOLUMETRIC_CLOUD ) );
            vis.mode = static_cast<DigitalTwin::GridVisualizationMode>( mode );

            if( vis.mode == DigitalTwin::GridVisualizationMode::SLICE_2D )
            {
                ImGui::SliderFloat( "Slice Z", &vis.sliceZ, 0.0f, 1.0f );
                ImGui::SliderFloat( "Opacity", &vis.opacitySlice, 0.0f, 1.0f );
            }
            else
            {
                ImGui::SliderFloat( "Opacity", &vis.opacityCloud, 0.0f, 0.2f, "%.4f" );
            }

            ImGui::ColorEdit4( "Color", &vis.colorMap.x );

            m_engine.SetGridVisualization( vis );
        }
    }

    void InspectorPanel::OnUIRender()
    {
        ImGui::Begin( m_name.c_str() );

        if( m_selection.gridFieldIndex >= 0 )
        {
            // Grid field selected — blueprint params + visualization (mixed editability, handled inside)
            auto& fields = m_blueprint.GetGridFieldsMutable();
            if( m_selection.gridFieldIndex < static_cast<int>( fields.size() ) )
                RenderGridFieldInspector( fields[ m_selection.gridFieldIndex ], m_selection.gridFieldIndex );
        }
        else if( m_selection.groupIndex >= 0 )
        {
            // Agent group or behaviour selected — disabled when not RESET
            const bool editable = ( m_engine.GetState() == DigitalTwin::EngineState::RESET );
            if( !editable )
                ImGui::BeginDisabled();

            auto& groups = m_blueprint.GetGroupsMutable();
            if( m_selection.groupIndex < static_cast<int>( groups.size() ) )
            {
                auto& group = groups[ m_selection.groupIndex ];
                if( m_selection.behaviourIndex < 0 )
                {
                    RenderGroupInspector( group );
                }
                else
                {
                    auto& behaviours = group.GetBehavioursMutable();
                    if( m_selection.behaviourIndex < static_cast<int>( behaviours.size() ) )
                        RenderBehaviourInspector( behaviours[ m_selection.behaviourIndex ] );
                }
            }

            if( !editable )
                ImGui::EndDisabled();
        }
        else
        {
            ImGui::TextDisabled( "Select an item in the Hierarchy" );
        }

        ImGui::End();
    }
} // namespace Gaudi
