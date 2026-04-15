#include "HierarchyPanel.h"

#include <DigitalTwin.h>
#include <imgui.h>
#include <simulation/Behaviours.h>
#include "../IconsFontAwesome5.h"

namespace Gaudi
{
    // ── Pending modal state ───────────────────────────────────────────────────
    // OpenPopup must be called at the same ID-stack level as BeginPopupModal.
    // Context menus live inside a popup stack, so we defer by setting a flag here
    // and calling OpenPopup just before BeginPopupModal at window level.

    enum class PendingOp
    {
        None,
        RenameSimulation,
        AddGroup,
        RenameGroup,
        AddField,
        RenameField,
    };

    static PendingOp s_pendingOp    = PendingOp::None;
    static int       s_pendingIdx   = -1;
    static char      s_nameBuf[ 128 ] = {};

    // ─────────────────────────────────────────────────────────────────────────

    HierarchyPanel::HierarchyPanel( DigitalTwin::DigitalTwin& engine, DigitalTwin::SimulationBlueprint& blueprint, EditorSelection& selection )
        : EditorPanel( "Hierarchy" )
        , m_engine( engine )
        , m_blueprint( blueprint )
        , m_selection( selection )
    {
    }

    static std::string GetBehaviourLabel( const DigitalTwin::BehaviourVariant& v )
    {
        return std::visit(
            []( const auto& b ) -> std::string {
                using T = std::decay_t<decltype( b )>;
                if constexpr( std::is_same_v<T, DigitalTwin::Behaviours::BrownianMotion> )
                    return "Brownian Motion";
                else if constexpr( std::is_same_v<T, DigitalTwin::Behaviours::ConsumeField> )
                    return "Consume: " + b.fieldName;
                else if constexpr( std::is_same_v<T, DigitalTwin::Behaviours::SecreteField> )
                    return "Secrete: " + b.fieldName;
                else if constexpr( std::is_same_v<T, DigitalTwin::Behaviours::Biomechanics> )
                    return "Biomechanics";
                else if constexpr( std::is_same_v<T, DigitalTwin::Behaviours::CellCycle> )
                    return "Cell Cycle";
                else if constexpr( std::is_same_v<T, DigitalTwin::Behaviours::Chemotaxis> )
                    return "Chemotaxis: " + b.fieldName;
                else if constexpr( std::is_same_v<T, DigitalTwin::Behaviours::NotchDll4> )
                    return "Notch-Dll4";
                else if constexpr( std::is_same_v<T, DigitalTwin::Behaviours::Anastomosis> )
                    return "Anastomosis";
                else if constexpr( std::is_same_v<T, DigitalTwin::Behaviours::Perfusion> )
                    return "Perfusion: " + b.fieldName;
                else if constexpr( std::is_same_v<T, DigitalTwin::Behaviours::Drain> )
                    return "Drain: " + b.fieldName;
                else if constexpr( std::is_same_v<T, DigitalTwin::Behaviours::VesselSeed> )
                    return "Vessel Seed";
                else if constexpr( std::is_same_v<T, DigitalTwin::Behaviours::VesselSpring> )
                    return "Vessel Spring";
                else if constexpr( std::is_same_v<T, DigitalTwin::Behaviours::PhalanxActivation> )
                    return "Phalanx Activation";
                else if constexpr( std::is_same_v<T, DigitalTwin::Behaviours::CadherinAdhesion> )
                    return "Cadherin Adhesion";
                else if constexpr( std::is_same_v<T, DigitalTwin::Behaviours::CellPolarity> )
                    return "Cell Polarity";
                else
                    return "Unknown";
            },
            v );
    }

    void HierarchyPanel::OnUIRender()
    {
        ImGui::Begin( m_name.c_str() );

        // No blueprint loaded yet — show a placeholder.
        if( m_blueprint.GetName().empty() )
        {
            ImGui::Spacing();
            ImGui::TextDisabled( "Use  File > New  or  Demos  to get started." );
            ImGui::End();
            return;
        }

        const bool isReset = ( m_engine.GetState() == DigitalTwin::EngineState::RESET );

        // ── Top-level simulation node ─────────────────────────────────────────
        ImGuiTreeNodeFlags simFlags = ImGuiTreeNodeFlags_DefaultOpen | ImGuiTreeNodeFlags_SpanAvailWidth | ImGuiTreeNodeFlags_OpenOnArrow;
        bool simOpen = ImGui::TreeNodeEx( "##sim", simFlags, "%s", m_blueprint.GetName().c_str() );

        if( ImGui::BeginPopupContextItem( "##simCtx" ) )
        {
            if( ImGui::MenuItem( "Rename Simulation" ) )
            {
                strncpy_s( s_nameBuf, m_blueprint.GetName().c_str(), sizeof( s_nameBuf ) - 1 );
                s_pendingOp  = PendingOp::RenameSimulation;
                s_pendingIdx = -1;
            }
            ImGui::EndPopup();
        }

        if( simOpen )
        {
            ImGuiTreeNodeFlags sectionFlags = ImGuiTreeNodeFlags_DefaultOpen | ImGuiTreeNodeFlags_SpanAvailWidth | ImGuiTreeNodeFlags_OpenOnArrow;

            // ── Agent Groups ──────────────────────────────────────────────────
            bool agentSectionOpen = ImGui::TreeNodeEx( "##agentGroups", sectionFlags, "Agent Groups" );

            if( ImGui::BeginPopupContextItem( "##agentGroupsCtx" ) )
            {
                ImGui::BeginDisabled( !isReset );
                if( ImGui::MenuItem( "Add Agent Group" ) )
                {
                    snprintf( s_nameBuf, sizeof( s_nameBuf ), "AgentGroup_%d",
                              static_cast<int>( m_blueprint.GetGroups().size() ) );
                    s_pendingOp  = PendingOp::AddGroup;
                    s_pendingIdx = -1;
                }
                ImGui::EndDisabled();
                ImGui::EndPopup();
            }

            if( agentSectionOpen )
            {
                auto& groups = m_blueprint.GetGroupsMutable();
                for( int i = 0; i < static_cast<int>( groups.size() ); ++i )
                {
                    auto& group = groups[ i ];

                    ImGuiTreeNodeFlags groupFlags = ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_DefaultOpen;
                    if( m_selection.groupIndex == i && m_selection.behaviourIndex == -1 )
                        groupFlags |= ImGuiTreeNodeFlags_Selected;

                    ImGui::PushID( i );
                    bool open = ImGui::TreeNodeEx( "##group", groupFlags, "%s", group.GetName().c_str() );

                    // Capture tree node click BEFORE the eye button changes the last item.
                    bool groupClicked = ImGui::IsItemClicked() && !ImGui::IsItemToggledOpen();

                    // Eye icon — right-aligned on the same row, after the group name.
                    {
                        bool        visible   = group.IsVisible();
                        const char* icon      = visible ? ICON_FA_EYE : ICON_FA_EYE_SLASH;
                        float       iconWidth = ImGui::CalcTextSize( icon ).x + ImGui::GetStyle().FramePadding.x * 2.0f;

                        ImGui::SameLine( ImGui::GetContentRegionMax().x - iconWidth );

                        ImGui::PushStyleColor( ImGuiCol_Button, ImVec4( 0, 0, 0, 0 ) );
                        ImGui::PushStyleColor( ImGuiCol_ButtonHovered, ImVec4( 1, 1, 1, 0.1f ) );
                        ImGui::PushStyleColor( ImGuiCol_ButtonActive, ImVec4( 1, 1, 1, 0.2f ) );
                        if( !visible )
                            ImGui::PushStyleColor( ImGuiCol_Text, ImGui::GetStyleColorVec4( ImGuiCol_TextDisabled ) );

                        if( ImGui::SmallButton( icon ) )
                        {
                            group.SetVisible( !visible );
                            m_engine.SetGroupVisible( i, !visible );
                            groupClicked = false; // eye click must not also select the group
                        }

                        if( !visible )
                            ImGui::PopStyleColor();
                        ImGui::PopStyleColor( 3 );
                    }

                    if( groupClicked )
                    {
                        m_selection.groupIndex     = i;
                        m_selection.behaviourIndex = -1;
                        m_selection.gridFieldIndex = -1;
                    }

                    if( ImGui::BeginPopupContextItem( "##groupCtx" ) )
                    {
                        ImGui::BeginDisabled( !isReset );

                        if( ImGui::BeginMenu( "Add Behaviour" ) )
                        {
                            if( ImGui::MenuItem( "Brownian Motion" ) )
                            {
                                group.AddBehaviour( DigitalTwin::Behaviours::BrownianMotion{} ).SetHz( 60 );
                                m_engine.SetBlueprint( m_blueprint );
                            }
                            if( ImGui::BeginMenu( "Consume Field" ) )
                            {
                                const auto& fields = m_blueprint.GetGridFields();
                                if( fields.empty() )
                                    ImGui::TextDisabled( "(no grid fields defined)" );
                                for( const auto& f : fields )
                                {
                                    if( ImGui::MenuItem( f.GetName().c_str() ) )
                                    {
                                        group.AddBehaviour( DigitalTwin::Behaviours::ConsumeField{ f.GetName(), 1.0f } ).SetHz( 60 );
                                        m_engine.SetBlueprint( m_blueprint );
                                    }
                                }
                                ImGui::EndMenu();
                            }
                            if( ImGui::BeginMenu( "Secrete Field" ) )
                            {
                                const auto& fields = m_blueprint.GetGridFields();
                                if( fields.empty() )
                                    ImGui::TextDisabled( "(no grid fields defined)" );
                                for( const auto& f : fields )
                                {
                                    if( ImGui::MenuItem( f.GetName().c_str() ) )
                                    {
                                        group.AddBehaviour( DigitalTwin::Behaviours::SecreteField{ f.GetName(), 1.0f } ).SetHz( 60 );
                                        m_engine.SetBlueprint( m_blueprint );
                                    }
                                }
                                ImGui::EndMenu();
                            }
                            if( ImGui::MenuItem( "Biomechanics" ) )
                            {
                                group.AddBehaviour( DigitalTwin::Behaviours::Biomechanics{} ).SetHz( 60 );
                                m_engine.SetBlueprint( m_blueprint );
                            }
                            if( ImGui::MenuItem( "Cell Cycle" ) )
                            {
                                group.AddBehaviour( DigitalTwin::Behaviours::CellCycle{ 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f } ).SetHz( 60 );
                                m_engine.SetBlueprint( m_blueprint );
                            }
                            if( ImGui::BeginMenu( "Chemotaxis" ) )
                            {
                                const auto& fields = m_blueprint.GetGridFields();
                                if( fields.empty() )
                                    ImGui::TextDisabled( "(no grid fields defined)" );
                                for( const auto& f : fields )
                                {
                                    if( ImGui::MenuItem( f.GetName().c_str() ) )
                                    {
                                        group.AddBehaviour( DigitalTwin::Behaviours::Chemotaxis{ f.GetName() } ).SetHz( 60 );
                                        m_engine.SetBlueprint( m_blueprint );
                                    }
                                }
                                ImGui::EndMenu();
                            }
                            if( ImGui::MenuItem( "Notch-Dll4" ) )
                            {
                                group.AddBehaviour( DigitalTwin::Behaviours::NotchDll4{} ).SetHz( 60 );
                                m_engine.SetBlueprint( m_blueprint );
                            }
                            if( ImGui::MenuItem( "Anastomosis" ) )
                            {
                                group.AddBehaviour( DigitalTwin::Behaviours::Anastomosis{} ).SetHz( 60 );
                                m_engine.SetBlueprint( m_blueprint );
                            }
                            if( ImGui::MenuItem( "Cadherin Adhesion" ) )
                            {
                                group.AddBehaviour( DigitalTwin::Behaviours::CadherinAdhesion{} ).SetHz( 60 );
                                m_engine.SetBlueprint( m_blueprint );
                            }
                            if( ImGui::MenuItem( "Cell Polarity" ) )
                            {
                                group.AddBehaviour( DigitalTwin::Behaviours::CellPolarity{} ).SetHz( 60 );
                                m_engine.SetBlueprint( m_blueprint );
                            }
                            if( ImGui::BeginMenu( "Perfusion" ) )
                            {
                                const auto& fields = m_blueprint.GetGridFields();
                                if( fields.empty() )
                                    ImGui::TextDisabled( "(no grid fields defined)" );
                                for( const auto& f : fields )
                                {
                                    if( ImGui::MenuItem( f.GetName().c_str() ) )
                                    {
                                        group.AddBehaviour( DigitalTwin::Behaviours::Perfusion{ f.GetName() } ).SetHz( 60 );
                                        m_engine.SetBlueprint( m_blueprint );
                                    }
                                }
                                ImGui::EndMenu();
                            }
                            if( ImGui::BeginMenu( "Drain" ) )
                            {
                                const auto& fields = m_blueprint.GetGridFields();
                                if( fields.empty() )
                                    ImGui::TextDisabled( "(no grid fields defined)" );
                                for( const auto& f : fields )
                                {
                                    if( ImGui::MenuItem( f.GetName().c_str() ) )
                                    {
                                        group.AddBehaviour( DigitalTwin::Behaviours::Drain{ f.GetName() } ).SetHz( 60 );
                                        m_engine.SetBlueprint( m_blueprint );
                                    }
                                }
                                ImGui::EndMenu();
                            }
                            ImGui::EndMenu();
                        }

                        ImGui::Separator();

                        if( ImGui::MenuItem( "Duplicate" ) )
                        {
                            DigitalTwin::AgentGroup copy = group;
                            copy.SetName( group.GetName() + "_Copy" );
                            groups.push_back( std::move( copy ) );
                            m_engine.SetBlueprint( m_blueprint );
                        }

                        if( ImGui::MenuItem( "Rename" ) )
                        {
                            strncpy_s( s_nameBuf, group.GetName().c_str(), sizeof( s_nameBuf ) - 1 );
                            s_pendingOp  = PendingOp::RenameGroup;
                            s_pendingIdx = i;
                        }

                        ImGui::Separator();

                        if( ImGui::MenuItem( "Remove Agent Group" ) )
                        {
                            if( m_selection.groupIndex == i )
                            {
                                m_selection.groupIndex     = -1;
                                m_selection.behaviourIndex = -1;
                            }
                            else if( m_selection.groupIndex > i )
                            {
                                --m_selection.groupIndex;
                            }
                            groups.erase( groups.begin() + i );
                            m_engine.SetBlueprint( m_blueprint );
                            ImGui::EndDisabled();
                            ImGui::EndPopup();
                            if( open )
                                ImGui::TreePop();
                            ImGui::PopID();
                            continue;
                        }

                        ImGui::EndDisabled();
                        ImGui::EndPopup();
                    }

                    if( open )
                    {
                        auto& behaviours = group.GetBehavioursMutable();
                        for( int j = 0; j < static_cast<int>( behaviours.size() ); ++j )
                        {
                            bool        selected = ( m_selection.groupIndex == i && m_selection.behaviourIndex == j );
                            std::string label    = GetBehaviourLabel( behaviours[ j ].behaviour );

                            ImGui::PushID( j );
                            ImGui::Selectable( label.c_str(), selected );

                            if( ImGui::IsItemClicked() )
                            {
                                m_selection.groupIndex     = i;
                                m_selection.behaviourIndex = j;
                                m_selection.gridFieldIndex = -1;
                            }

                            if( ImGui::BeginPopupContextItem( "##behaviourCtx" ) )
                            {
                                ImGui::BeginDisabled( !isReset );
                                if( ImGui::MenuItem( "Remove Behaviour" ) )
                                {
                                    if( m_selection.groupIndex == i )
                                    {
                                        if( m_selection.behaviourIndex == j )
                                            m_selection.behaviourIndex = -1;
                                        else if( m_selection.behaviourIndex > j )
                                            --m_selection.behaviourIndex;
                                    }
                                    behaviours.erase( behaviours.begin() + j );
                                    m_engine.SetBlueprint( m_blueprint );
                                    ImGui::EndDisabled();
                                    ImGui::EndPopup();
                                    ImGui::PopID();
                                    continue;
                                }
                                ImGui::EndDisabled();
                                ImGui::EndPopup();
                            }

                            ImGui::PopID();
                        }
                        ImGui::TreePop();
                    }

                    ImGui::PopID();
                }
                ImGui::TreePop();
            }

            // ── Grid Fields ───────────────────────────────────────────────────
            bool fieldSectionOpen = ImGui::TreeNodeEx( "##gridFields", sectionFlags, "Grid Fields" );

            if( ImGui::BeginPopupContextItem( "##gridFieldsCtx" ) )
            {
                ImGui::BeginDisabled( !isReset );
                if( ImGui::MenuItem( "Add Grid Field" ) )
                {
                    snprintf( s_nameBuf, sizeof( s_nameBuf ), "GridField_%d",
                              static_cast<int>( m_blueprint.GetGridFields().size() ) );
                    s_pendingOp  = PendingOp::AddField;
                    s_pendingIdx = -1;
                }
                ImGui::EndDisabled();
                ImGui::EndPopup();
            }

            if( fieldSectionOpen )
            {
                auto& fields = m_blueprint.GetGridFieldsMutable();
                for( int i = 0; i < static_cast<int>( fields.size() ); ++i )
                {
                    auto& field   = fields[ i ];
                    bool  selected = ( m_selection.gridFieldIndex == i );

                    ImGui::PushID( 1000 + i );
                    ImGui::Selectable( field.GetName().c_str(), selected );

                    if( ImGui::IsItemClicked() )
                    {
                        m_selection.gridFieldIndex = i;
                        m_selection.groupIndex     = -1;
                        m_selection.behaviourIndex = -1;
                    }

                    if( ImGui::BeginPopupContextItem( "##fieldCtx" ) )
                    {
                        ImGui::BeginDisabled( !isReset );

                        if( ImGui::MenuItem( "Duplicate" ) )
                        {
                            DigitalTwin::GridField copy( field.GetName() + "_Copy" );
                            copy.SetDiffusionCoefficient( field.GetDiffusionCoefficient() )
                                .SetDecayRate( field.GetDecayRate() )
                                .SetComputeHz( field.GetComputeHz() )
                                .SetInitializer( field.GetInitializer() );
                            fields.push_back( std::move( copy ) );
                            m_engine.SetBlueprint( m_blueprint );
                        }

                        if( ImGui::MenuItem( "Rename" ) )
                        {
                            strncpy_s( s_nameBuf, field.GetName().c_str(), sizeof( s_nameBuf ) - 1 );
                            s_pendingOp  = PendingOp::RenameField;
                            s_pendingIdx = i;
                        }

                        ImGui::Separator();

                        if( ImGui::MenuItem( "Remove Grid Field" ) )
                        {
                            if( m_selection.gridFieldIndex == i )
                                m_selection.gridFieldIndex = -1;
                            else if( m_selection.gridFieldIndex > i )
                                --m_selection.gridFieldIndex;

                            fields.erase( fields.begin() + i );
                            m_engine.SetBlueprint( m_blueprint );
                            ImGui::EndDisabled();
                            ImGui::EndPopup();
                            ImGui::PopID();
                            continue;
                        }

                        ImGui::EndDisabled();
                        ImGui::EndPopup();
                    }

                    ImGui::PopID();
                }
                ImGui::TreePop();
            }

            ImGui::TreePop();
        }

        // ── Deferred name modal ───────────────────────────────────────────────
        // Must be opened and rendered at window level, outside any popup stack.

        if( s_pendingOp != PendingOp::None )
            ImGui::OpenPopup( "##nameModal" );

        if( ImGui::BeginPopupModal( "##nameModal", nullptr, ImGuiWindowFlags_AlwaysAutoResize ) )
        {
            switch( s_pendingOp )
            {
                case PendingOp::RenameSimulation: ImGui::Text( "Rename Simulation:" );  break;
                case PendingOp::AddGroup:         ImGui::Text( "New Agent Group name:" ); break;
                case PendingOp::RenameGroup:      ImGui::Text( "Rename Agent Group:" );   break;
                case PendingOp::AddField:         ImGui::Text( "New Grid Field name:" );  break;
                case PendingOp::RenameField:      ImGui::Text( "Rename Grid Field:" );    break;
                default: break;
            }

            ImGui::SetNextItemWidth( 260.0f );
            bool confirm = ImGui::InputText( "##nameInput", s_nameBuf, sizeof( s_nameBuf ),
                                             ImGuiInputTextFlags_EnterReturnsTrue | ImGuiInputTextFlags_AutoSelectAll );
            if( ImGui::IsWindowAppearing() )
                ImGui::SetKeyboardFocusHere( -1 );

            ImGui::Spacing();
            if( confirm || ImGui::Button( "OK", ImVec2( 120, 0 ) ) )
            {
                const std::string name( s_nameBuf );
                switch( s_pendingOp )
                {
                    case PendingOp::RenameSimulation:
                        m_blueprint.SetName( name );
                        m_engine.SetBlueprint( m_blueprint );
                        break;

                    case PendingOp::AddGroup:
                        m_blueprint.AddAgentGroup( name );
                        m_engine.SetBlueprint( m_blueprint );
                        break;

                    case PendingOp::RenameGroup:
                        if( s_pendingIdx >= 0 && s_pendingIdx < static_cast<int>( m_blueprint.GetGroupsMutable().size() ) )
                        {
                            m_blueprint.GetGroupsMutable()[ s_pendingIdx ].SetName( name );
                            m_engine.SetBlueprint( m_blueprint );
                        }
                        break;

                    case PendingOp::AddField:
                        m_blueprint.AddGridField( name );
                        m_engine.SetBlueprint( m_blueprint );
                        break;

                    case PendingOp::RenameField:
                        if( s_pendingIdx >= 0 && s_pendingIdx < static_cast<int>( m_blueprint.GetGridFieldsMutable().size() ) )
                        {
                            auto& f = m_blueprint.GetGridFieldsMutable()[ s_pendingIdx ];
                            DigitalTwin::GridField renamed( name );
                            renamed.SetDiffusionCoefficient( f.GetDiffusionCoefficient() )
                                   .SetDecayRate( f.GetDecayRate() )
                                   .SetComputeHz( f.GetComputeHz() )
                                   .SetInitializer( f.GetInitializer() );
                            f = std::move( renamed );
                            m_engine.SetBlueprint( m_blueprint );
                        }
                        break;

                    default: break;
                }
                s_pendingOp = PendingOp::None;
                ImGui::CloseCurrentPopup();
            }

            ImGui::SameLine();
            if( ImGui::Button( "Cancel", ImVec2( 120, 0 ) ) )
            {
                s_pendingOp = PendingOp::None;
                ImGui::CloseCurrentPopup();
            }

            ImGui::EndPopup();
        }

        ImGui::End();
    }
} // namespace Gaudi
