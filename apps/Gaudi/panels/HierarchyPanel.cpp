#include "HierarchyPanel.h"

#include <DigitalTwin.h>
#include <imgui.h>
#include <simulation/Behaviours.h>

namespace Gaudi
{
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
                else
                    return "Unknown";
            },
            v );
    }

    void HierarchyPanel::OnUIRender()
    {
        ImGui::Begin( m_name.c_str() );

        // ── Top-level simulation node ─────────────────────────────────
        ImGuiTreeNodeFlags simFlags = ImGuiTreeNodeFlags_DefaultOpen | ImGuiTreeNodeFlags_SpanAvailWidth | ImGuiTreeNodeFlags_OpenOnArrow;
        bool simOpen = ImGui::TreeNodeEx( "##sim", simFlags, "%s", m_blueprint.GetName().c_str() );

        if( simOpen )
        {
            // ── Agent Groups ──────────────────────────────────────────
            ImGuiTreeNodeFlags sectionFlags = ImGuiTreeNodeFlags_DefaultOpen | ImGuiTreeNodeFlags_SpanAvailWidth | ImGuiTreeNodeFlags_OpenOnArrow;
            bool agentSectionOpen = ImGui::TreeNodeEx( "##agentGroups", sectionFlags, "Agent Groups" );

            if( agentSectionOpen )
            {
                const auto& groups = m_blueprint.GetGroups();
                for( int i = 0; i < static_cast<int>( groups.size() ); ++i )
                {
                    const auto& group = groups[ i ];

                    ImGuiTreeNodeFlags groupFlags = ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_SpanAvailWidth | ImGuiTreeNodeFlags_DefaultOpen;
                    if( m_selection.groupIndex == i && m_selection.behaviourIndex == -1 )
                        groupFlags |= ImGuiTreeNodeFlags_Selected;

                    bool open = ImGui::TreeNodeEx( reinterpret_cast<void*>( static_cast<intptr_t>( i ) ), groupFlags, "%s", group.GetName().c_str() );

                    if( ImGui::IsItemClicked() && !ImGui::IsItemToggledOpen() )
                    {
                        m_selection.groupIndex     = i;
                        m_selection.behaviourIndex = -1;
                        m_selection.gridFieldIndex = -1;
                    }

                    if( open )
                    {
                        const auto& behaviours = group.GetBehaviours();
                        for( int j = 0; j < static_cast<int>( behaviours.size() ); ++j )
                        {
                            bool        selected = ( m_selection.groupIndex == i && m_selection.behaviourIndex == j );
                            std::string label    = GetBehaviourLabel( behaviours[ j ].behaviour );

                            ImGui::PushID( j );
                            if( ImGui::Selectable( label.c_str(), selected ) )
                            {
                                m_selection.groupIndex     = i;
                                m_selection.behaviourIndex = j;
                                m_selection.gridFieldIndex = -1;
                            }
                            ImGui::PopID();
                        }
                        ImGui::TreePop();
                    }
                }
                ImGui::TreePop();
            }

            // ── Grid Fields ───────────────────────────────────────────
            bool fieldSectionOpen = ImGui::TreeNodeEx( "##gridFields", sectionFlags, "Grid Fields" );

            if( fieldSectionOpen )
            {
                const auto& fields = m_blueprint.GetGridFields();
                for( int i = 0; i < static_cast<int>( fields.size() ); ++i )
                {
                    bool selected = ( m_selection.gridFieldIndex == i );

                    ImGui::PushID( 1000 + i );
                    if( ImGui::Selectable( fields[ i ].GetName().c_str(), selected ) )
                    {
                        m_selection.gridFieldIndex = i;
                        m_selection.groupIndex     = -1;
                        m_selection.behaviourIndex = -1;
                    }
                    ImGui::PopID();
                }
                ImGui::TreePop();
            }

            ImGui::TreePop();
        }

        ImGui::End();
    }
} // namespace Gaudi
