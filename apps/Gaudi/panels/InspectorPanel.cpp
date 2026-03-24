#include "InspectorPanel.h"

#include "../IconsFontAwesome5.h"
#include <DigitalTwin.h>
#include <algorithm>
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

        const DigitalTwin::EngineState state = m_engine.GetState();
        const bool                     live  = ( state == DigitalTwin::EngineState::PLAYING || state == DigitalTwin::EngineState::PAUSED );

        bool changed = false;

        std::visit(
            [ & ]( auto& b ) {
                using T = std::decay_t<decltype( b )>;

                if constexpr( std::is_same_v<T, DigitalTwin::Behaviours::BrownianMotion> )
                {
                    changed |= ImGui::SliderFloat( "Speed", &b.speed, 0.0f, 10.0f );
                }
                else if constexpr( std::is_same_v<T, DigitalTwin::Behaviours::ConsumeField> )
                {
                    ImGui::LabelText( "Field", "%s", b.fieldName.c_str() );
                    changed |= ImGui::SliderFloat( "Rate", &b.rate, 0.0f, 200.0f );
                    changed |= ImGui::InputInt( "Required Lifecycle State", &b.requiredLifecycleState );
                    if( b.requiredLifecycleState < -1 )
                        b.requiredLifecycleState = -1;
                }
                else if constexpr( std::is_same_v<T, DigitalTwin::Behaviours::SecreteField> )
                {
                    ImGui::LabelText( "Field", "%s", b.fieldName.c_str() );
                    changed |= ImGui::SliderFloat( "Rate", &b.rate, 0.0f, 200.0f );
                    changed |= ImGui::InputInt( "Required Lifecycle State", &b.requiredLifecycleState );
                    if( b.requiredLifecycleState < -1 )
                        b.requiredLifecycleState = -1;
                }
                else if constexpr( std::is_same_v<T, DigitalTwin::Behaviours::Biomechanics> )
                {
                    changed |= ImGui::SliderFloat( "Repulsion Stiffness", &b.repulsionStiffness, 0.0f, 50.0f );
                    changed |= ImGui::SliderFloat( "Adhesion Strength", &b.adhesionStrength, 0.0f, 20.0f );
                    changed |= ImGui::SliderFloat( "Max Radius", &b.maxRadius, 0.5f, 5.0f );
                }
                else if constexpr( std::is_same_v<T, DigitalTwin::Behaviours::CellCycle> )
                {
                    changed |= ImGui::SliderFloat( "Growth Rate /sec", &b.growthRatePerSec, 0.0f, 0.01f, "%.6f" );
                    changed |= ImGui::SliderFloat( "Target O2", &b.targetO2, 0.0f, 100.0f );
                    changed |= ImGui::SliderFloat( "Arrest Pressure", &b.arrestPressure, 0.0f, 50.0f );
                    changed |= ImGui::SliderFloat( "Hypoxia O2", &b.hypoxiaO2, 0.0f, 50.0f );
                    changed |= ImGui::SliderFloat( "Necrosis O2", &b.necrosisO2, 0.0f, 50.0f );
                    changed |= ImGui::SliderFloat( "Apoptosis /sec", &b.apoptosisProbPerSec, 0.0f, 0.01f, "%.6f" );

                    // Enforce ordering: necrosisO2 < hypoxiaO2 < targetO2
                    b.necrosisO2 = std::min( b.necrosisO2, b.hypoxiaO2 - 0.1f );
                    b.hypoxiaO2  = std::min( b.hypoxiaO2, b.targetO2 - 0.1f );
                    b.necrosisO2 = std::max( b.necrosisO2, 0.0f );

                    ImGui::TextDisabled( "necrosis < hypoxia < target O2" );
                    changed |= ImGui::Checkbox( "Directed Mitosis", &b.directedMitosis );
                    if( live && ImGui::IsItemHovered() )
                        ImGui::SetTooltip( "Requires Stop + Play to take effect" );
                }
                else if constexpr( std::is_same_v<T, DigitalTwin::Behaviours::Chemotaxis> )
                {
                    ImGui::LabelText( "Target Field", "%s", b.fieldName.c_str() );
                    changed |= ImGui::SliderFloat( "Sensitivity",  &b.chemotacticSensitivity,   0.01f, 20.0f );
                    changed |= ImGui::SliderFloat( "Saturation",   &b.receptorSaturation,       0.0f,  1.0f, "%.4f" );
                    changed |= ImGui::SliderFloat( "Max Velocity", &b.maxVelocity,              0.1f,  50.0f );
                    changed |= ImGui::SliderFloat( "Contact Inhibition Density", &b.contactInhibitionDensity, 0.0f, 30.0f );
                    ImGui::TextDisabled( "0 = disabled" );
                }
                else if constexpr( std::is_same_v<T, DigitalTwin::Behaviours::NotchDll4> )
                {
                    changed |= ImGui::SliderFloat( "Dll4 Production Rate",   &b.dll4ProductionRate,   0.0f, 10.0f );
                    changed |= ImGui::SliderFloat( "Dll4 Decay Rate",        &b.dll4DecayRate,        0.0f, 1.0f, "%.3f" );
                    changed |= ImGui::SliderFloat( "Notch Inhibition Gain",  &b.notchInhibitionGain,  0.0f, 200.0f );
                    changed |= ImGui::SliderFloat( "VEGFR2 Base Expression", &b.vegfr2BaseExpression, 0.0f, 5.0f );
                    changed |= ImGui::SliderFloat( "Tip Threshold",          &b.tipThreshold,         0.0f, 1.0f );
                    changed |= ImGui::SliderFloat( "Stalk Threshold",        &b.stalkThreshold,       0.0f, 1.0f );
                    ImGui::TextDisabled( "stalkThreshold < tipThreshold required" );
                    ImGui::BeginDisabled( live );
                    int subSteps = static_cast<int>( b.subSteps );
                    if( ImGui::SliderInt( "Sub-Steps", &subSteps, 1, 50 ) )
                    {
                        b.subSteps = static_cast<uint32_t>( subSteps );
                        changed    = true;
                    }
                    ImGui::EndDisabled();
                    if( live )
                        ImGui::TextDisabled( "Sub-Steps requires Stop + Play to take effect" );
                }
                else if constexpr( std::is_same_v<T, DigitalTwin::Behaviours::PhalanxActivation> )
                {
                    ImGui::LabelText( "VEGF Field", "%s", b.vegfFieldName.c_str() );
                    changed |= ImGui::SliderFloat( "Activation Threshold",   &b.activationThreshold,   0.1f, 200.0f );
                    changed |= ImGui::SliderFloat( "Deactivation Threshold", &b.deactivationThreshold, 0.0f, 100.0f );
                    ImGui::TextDisabled( "deactivation < activation required" );
                }
                else if constexpr( std::is_same_v<T, DigitalTwin::Behaviours::VesselSpring> )
                {
                    changed |= ImGui::SliderFloat( "Spring Stiffness", &b.springStiffness, 0.1f, 50.0f );
                    changed |= ImGui::SliderFloat( "Resting Length",   &b.restingLength,   0.5f, 10.0f );
                }
                else if constexpr( std::is_same_v<T, DigitalTwin::Behaviours::Anastomosis> )
                {
                    changed |= ImGui::SliderFloat( "Contact Distance", &b.contactDistance, 0.1f, 10.0f );
                }
                else if constexpr( std::is_same_v<T, DigitalTwin::Behaviours::Perfusion> )
                {
                    ImGui::LabelText( "Field", "%s", b.fieldName.c_str() );
                    changed |= ImGui::SliderFloat( "Rate", &b.rate, 0.01f, 50.0f );
                }
                else if constexpr( std::is_same_v<T, DigitalTwin::Behaviours::Drain> )
                {
                    ImGui::LabelText( "Field", "%s", b.fieldName.c_str() );
                    changed |= ImGui::SliderFloat( "Rate", &b.rate, 0.01f, 50.0f );
                }
            },
            record.behaviour );

        ImGui::Spacing();
        ImGui::SliderFloat( "Target Hz", &record.targetHz, 1.0f, 120.0f );

        ImGui::Spacing();
        ImGui::SeparatorText( "State Filtering" );
        changed |= ImGui::InputInt( "Required Lifecycle State", &record.requiredLifecycleState );
        if( record.requiredLifecycleState < -1 ) record.requiredLifecycleState = -1;
        changed |= ImGui::InputInt( "Required Cell Type", &record.requiredCellType );
        if( record.requiredCellType < -1 ) record.requiredCellType = -1;

        if( live )
        {
            ImGui::Spacing();
            ImGui::TextColored( ImVec4( 0.4f, 0.9f, 0.4f, 1.0f ), ICON_FA_BOLT " Live" );
            if( changed )
                m_engine.HotReload( m_blueprint );
        }
    }

    void InspectorPanel::RenderGridFieldInspector( DigitalTwin::GridField& field, int fieldIndex )
    {
        ImGui::TextDisabled( "Grid Field" );
        ImGui::Separator();

        ImGui::LabelText( "Name", "%s", field.GetName().c_str() );

        // ── Blueprint parameters ───────────────────────────────────────
        const DigitalTwin::EngineState state = m_engine.GetState();
        const bool                     live  = ( state == DigitalTwin::EngineState::PLAYING || state == DigitalTwin::EngineState::PAUSED );
        bool                           changed = false;

        float diffusion = field.GetDiffusionCoefficient();
        if( ImGui::SliderFloat( "Diffusion", &diffusion, 0.0f, 20.0f ) )
        {
            field.SetDiffusionCoefficient( diffusion );
            changed = true;
        }

        float decay = field.GetDecayRate();
        if( ImGui::SliderFloat( "Decay Rate", &decay, 0.0f, 1.0f, "%.4f" ) )
        {
            field.SetDecayRate( decay );
            changed = true;
        }

        float hz = field.GetComputeHz();
        if( ImGui::SliderFloat( "Compute Hz", &hz, 1.0f, 120.0f ) )
            field.SetComputeHz( hz );

        if( live )
        {
            ImGui::Spacing();
            ImGui::TextColored( ImVec4( 0.4f, 0.9f, 0.4f, 1.0f ), ICON_FA_BOLT " Live" );
            if( changed )
                m_engine.HotReload( m_blueprint );
        }

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
            auto& groups = m_blueprint.GetGroupsMutable();
            if( m_selection.groupIndex < static_cast<int>( groups.size() ) )
            {
                auto& group = groups[ m_selection.groupIndex ];
                if( m_selection.behaviourIndex < 0 )
                {
                    // Group-level properties are structural — only editable in RESET
                    const bool editable = ( m_engine.GetState() == DigitalTwin::EngineState::RESET );
                    if( !editable )
                        ImGui::BeginDisabled();
                    RenderGroupInspector( group );
                    if( !editable )
                        ImGui::EndDisabled();
                }
                else
                {
                    // Behaviour parameters are hot-reloadable — RenderBehaviourInspector manages its own state
                    auto& behaviours = group.GetBehavioursMutable();
                    if( m_selection.behaviourIndex < static_cast<int>( behaviours.size() ) )
                        RenderBehaviourInspector( behaviours[ m_selection.behaviourIndex ] );
                }
            }
        }
        else
        {
            ImGui::TextDisabled( "Select an item in the Hierarchy" );
        }

        // Vessel visualization — always visible (allows pre-configuring before Play)
        {
            ImGui::Spacing();
            ImGui::SeparatorText( "Vessel Network" );

            DigitalTwin::VesselVisualizationSettings vesselVis = m_engine.GetVesselVisualization();
            bool vesselChanged = false;

            vesselChanged |= ImGui::Checkbox( "Show Vessel Lines", &vesselVis.active );

            if( vesselVis.active )
            {
                vesselChanged |= ImGui::ColorEdit4( "Line Color", &vesselVis.lineColor.x );
            }

            if( vesselChanged )
                m_engine.SetVesselVisualization( vesselVis );
        }

        ImGui::End();
    }
} // namespace Gaudi
