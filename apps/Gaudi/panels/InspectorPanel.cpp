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
                    {
                        int ls = ( b.requiredLifecycleState == DigitalTwin::LifecycleState::Any )
                                     ? -1
                                     : static_cast<int>( static_cast<uint32_t>( b.requiredLifecycleState ) );
                        if( ImGui::InputInt( "Required Lifecycle State", &ls ) )
                        {
                            b.requiredLifecycleState = ( ls < 0 )
                                                           ? DigitalTwin::LifecycleState::Any
                                                           : static_cast<DigitalTwin::LifecycleState>( static_cast<uint32_t>( ls ) );
                            changed = true;
                        }
                    }
                }
                else if constexpr( std::is_same_v<T, DigitalTwin::Behaviours::SecreteField> )
                {
                    ImGui::LabelText( "Field", "%s", b.fieldName.c_str() );
                    changed |= ImGui::SliderFloat( "Rate", &b.rate, 0.0f, 200.0f );
                    {
                        int ls = ( b.requiredLifecycleState == DigitalTwin::LifecycleState::Any )
                                     ? -1
                                     : static_cast<int>( static_cast<uint32_t>( b.requiredLifecycleState ) );
                        if( ImGui::InputInt( "Required Lifecycle State", &ls ) )
                        {
                            b.requiredLifecycleState = ( ls < 0 )
                                                           ? DigitalTwin::LifecycleState::Any
                                                           : static_cast<DigitalTwin::LifecycleState>( static_cast<uint32_t>( ls ) );
                            changed = true;
                        }
                    }
                }
                else if constexpr( std::is_same_v<T, DigitalTwin::Behaviours::CellPolarity> )
                {
                    changed |= ImGui::SliderFloat( "Regulation Rate",   &b.regulationRate,  0.0f, 1.0f, "%.3f" );
                    changed |= ImGui::SliderFloat( "Apical Repulsion",  &b.apicalRepulsion, 0.0f, 1.0f );
                    changed |= ImGui::SliderFloat( "Basal Adhesion",    &b.basalAdhesion,   1.0f, 3.0f );
                }
                else if constexpr( std::is_same_v<T, DigitalTwin::Behaviours::CadherinAdhesion> )
                {
                    ImGui::SeparatorText( "Target Expression" );
                    changed |= ImGui::SliderFloat( "E-Cadherin",  &b.targetExpression.x, 0.0f, 1.0f );
                    changed |= ImGui::SliderFloat( "N-Cadherin",  &b.targetExpression.y, 0.0f, 1.0f );
                    changed |= ImGui::SliderFloat( "VE-Cadherin", &b.targetExpression.z, 0.0f, 1.0f );
                    changed |= ImGui::SliderFloat( "Cadherin-11", &b.targetExpression.w, 0.0f, 1.0f );
                    ImGui::Spacing();
                    changed |= ImGui::SliderFloat( "Expression Rate",   &b.expressionRate,   0.0f, 0.1f,  "%.4f" );
                    changed |= ImGui::SliderFloat( "Degradation Rate",  &b.degradationRate,  0.0f, 0.01f, "%.5f" );
                    changed |= ImGui::SliderFloat( "Coupling Strength", &b.couplingStrength, 0.0f, 5.0f );
                    ImGui::TextDisabled( "Affinity matrix set via blueprint API" );
                }
                else if constexpr( std::is_same_v<T, DigitalTwin::Behaviours::Biomechanics> )
                {
                    changed |= ImGui::SliderFloat( "Repulsion Stiffness", &b.repulsionStiffness, 0.0f, 100.0f );
                    changed |= ImGui::SliderFloat( "Adhesion Strength", &b.adhesionStrength, 0.0f, 20.0f );
                    changed |= ImGui::SliderFloat( "Max Radius", &b.maxRadius, 0.5f, 5.0f );
                    changed |= ImGui::SliderFloat( "Damping", &b.dampingCoefficient, 0.0f, 100.0f );
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
                    changed |= ImGui::SliderFloat( "Spring Stiffness", &b.springStiffness,    0.1f, 50.0f );
                    changed |= ImGui::SliderFloat( "Resting Length",   &b.restingLength,      0.5f, 10.0f );
                    changed |= ImGui::SliderFloat( "Damping",          &b.dampingCoefficient, 0.0f, 50.0f );
                }
                else if constexpr( std::is_same_v<T, DigitalTwin::Behaviours::Anastomosis> )
                {
                    changed |= ImGui::SliderFloat( "Contact Distance", &b.contactDistance, 0.1f, 10.0f );
                    changed |= ImGui::Checkbox( "Allow Tip-to-Stalk", &b.allowTipToStalk );
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
        {
            int ls = ( record.requiredLifecycleState == DigitalTwin::LifecycleState::Any )
                         ? -1
                         : static_cast<int>( static_cast<uint32_t>( record.requiredLifecycleState ) );
            if( ImGui::InputInt( "Required Lifecycle State", &ls ) )
            {
                record.requiredLifecycleState = ( ls < 0 )
                                                    ? DigitalTwin::LifecycleState::Any
                                                    : static_cast<DigitalTwin::LifecycleState>( static_cast<uint32_t>( ls ) );
                changed = true;
            }
            int ct = ( record.requiredCellType == DigitalTwin::CellType::Any )
                         ? -1
                         : static_cast<int>( static_cast<uint32_t>( record.requiredCellType ) );
            if( ImGui::InputInt( "Required Cell Type", &ct ) )
            {
                record.requiredCellType = ( ct < 0 )
                                              ? DigitalTwin::CellType::Any
                                              : static_cast<DigitalTwin::CellType>( static_cast<uint32_t>( ct ) );
                changed = true;
            }
        }

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

        // Grow the cache to cover the current number of fields.
        const int numFields = static_cast<int>( m_blueprint.GetGridFields().size() );
        if( static_cast<int>( m_fieldVisCache.size() ) < numFields )
            m_fieldVisCache.resize( numFields );

        DigitalTwin::GridFieldVisualization&   fieldVis = m_fieldVisCache[ fieldIndex ];
        DigitalTwin::GridVisualizationSettings vis      = m_engine.GetGridVisualization();
        bool isActive = vis.active && ( vis.fieldIndex == fieldIndex );

        if( ImGui::Checkbox( "Show this field", &isActive ) )
        {
            vis.active     = isActive;
            vis.fieldIndex = fieldIndex;
            vis.fieldVis   = fieldVis;
            m_engine.SetGridVisualization( vis );
        }

        if( isActive )
        {
            vis.fieldIndex = fieldIndex;
            vis.fieldVis   = fieldVis;

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

            ImGui::Spacing();
            ImGui::SeparatorText( "Colormap" );

            ImGui::DragFloat( "Min", &fieldVis.minValue, 1.0f );
            ImGui::DragFloat( "Max", &fieldVis.maxValue, 1.0f );
            if( fieldVis.maxValue <= fieldVis.minValue )
                fieldVis.maxValue = fieldVis.minValue + 1.0f;

            ImGui::SliderFloat( "Alpha Cutoff", &fieldVis.alphaCutoff, 0.0f, 0.5f, "%.3f" );
            ImGui::SetItemTooltip( "Normalized values below this are fully transparent." );

            ImGui::SliderFloat( "Gamma", &fieldVis.gamma, 0.1f, 3.0f, "%.2f" );
            ImGui::SetItemTooltip( "< 1: lifts weak gradients into view\n> 1: focuses colour on concentration peaks" );

            static const char* colormapNames[] = { "JET", "OXYGEN", "HOT", "PLASMA", "VEGF", "CUSTOM" };
            int cmIdx = static_cast<int>( fieldVis.colormap );
            if( ImGui::Combo( "Colormap", &cmIdx, colormapNames, 6 ) )
                fieldVis.colormap = static_cast<DigitalTwin::Colormap>( cmIdx );

            if( fieldVis.colormap == DigitalTwin::Colormap::CUSTOM )
            {
                ImGui::ColorEdit3( "Low  (t=0.0)", &fieldVis.customLow.x );
                ImGui::ColorEdit3( "Mid  (t=0.5)", &fieldVis.customMid.x );
                ImGui::ColorEdit3( "High (t=1.0)", &fieldVis.customHigh.x );
            }

            // Gradient preview strip — samples the same math as the shader
            {
                auto sampleColormap = [ & ]( float t ) -> glm::vec3 {
                    // Apply gamma before sampling (matches shader behaviour)
                    float tg = glm::pow( t, glm::max( fieldVis.gamma, 0.01f ) );
                    auto threeStop = [ ]( glm::vec3 a, glm::vec3 b, glm::vec3 c, float s ) -> glm::vec3 {
                        return ( s < 0.5f ) ? glm::mix( a, b, s * 2.0f ) : glm::mix( b, c, ( s - 0.5f ) * 2.0f );
                    };
                    auto fourStop = [ ]( glm::vec3 c0, glm::vec3 c1, glm::vec3 c2, glm::vec3 c3, float s ) -> glm::vec3 {
                        if( s < 0.333f )      return glm::mix( c0, c1, s / 0.333f );
                        else if( s < 0.667f ) return glm::mix( c1, c2, ( s - 0.333f ) / 0.334f );
                        else                  return glm::mix( c2, c3, ( s - 0.667f ) / 0.333f );
                    };
                    switch( fieldVis.colormap )
                    {
                        case DigitalTwin::Colormap::OXYGEN:
                            return fourStop( { 0.0f, 0.0f, 0.4f }, { 0.0f, 0.8f, 0.8f }, { 0.9f, 0.9f, 0.0f }, { 0.8f, 0.0f, 0.0f }, tg );
                        case DigitalTwin::Colormap::HOT:
                            return fourStop( { 0.0f, 0.0f, 0.0f }, { 1.0f, 0.0f, 0.0f }, { 1.0f, 1.0f, 0.0f }, { 1.0f, 1.0f, 1.0f }, tg );
                        case DigitalTwin::Colormap::PLASMA:
                            return fourStop( { 0.05f, 0.03f, 0.53f }, { 0.49f, 0.01f, 0.66f }, { 0.88f, 0.31f, 0.30f }, { 0.99f, 0.91f, 0.14f }, tg );
                        case DigitalTwin::Colormap::VEGF:
                            return fourStop( { 0.05f, 0.0f, 0.1f }, { 0.8f, 0.8f, 0.0f }, { 0.9f, 0.4f, 0.0f }, { 0.9f, 0.0f, 0.7f }, tg );
                        case DigitalTwin::Colormap::CUSTOM:
                            return threeStop( fieldVis.customLow, fieldVis.customMid, fieldVis.customHigh, tg );
                        default: // JET
                        {
                            float r = glm::clamp( 1.5f - glm::abs( 4.0f * tg - 3.0f ), 0.0f, 1.0f );
                            float g = glm::clamp( 1.5f - glm::abs( 4.0f * tg - 2.0f ), 0.0f, 1.0f );
                            float b = glm::clamp( 1.5f - glm::abs( 4.0f * tg - 1.0f ), 0.0f, 1.0f );
                            return { r, g, b };
                        }
                    }
                };

                ImDrawList* dl    = ImGui::GetWindowDrawList();
                ImVec2      pos   = ImGui::GetCursorScreenPos();
                float       width = ImGui::GetContentRegionAvail().x;
                const float h     = 12.0f;
                const int   N     = 64;
                for( int k = 0; k < N; ++k )
                {
                    float     t   = static_cast<float>( k ) / static_cast<float>( N - 1 );
                    glm::vec3 c   = sampleColormap( t );
                    ImU32     col = IM_COL32( static_cast<int>( c.r * 255 ), static_cast<int>( c.g * 255 ), static_cast<int>( c.b * 255 ), 255 );
                    float     x0  = pos.x + ( static_cast<float>( k ) / N ) * width;
                    float     x1  = pos.x + ( static_cast<float>( k + 1 ) / N ) * width;
                    dl->AddRectFilled( ImVec2( x0, pos.y ), ImVec2( x1, pos.y + h ), col );
                }
                ImGui::Dummy( ImVec2( width, h ) );
            }

            vis.fieldVis = fieldVis;
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
