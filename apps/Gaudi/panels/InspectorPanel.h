#pragma once
#include "EditorPanel.h"
#include "../EditorSelection.h"
#include <DigitalTwinTypes.h>
#include <simulation/SimulationBlueprint.h>
#include <vector>

namespace DigitalTwin
{
    class DigitalTwin;
}

namespace Gaudi
{
    class InspectorPanel : public EditorPanel
    {
    public:
        InspectorPanel( DigitalTwin::DigitalTwin& engine, DigitalTwin::SimulationBlueprint& blueprint, EditorSelection& selection );
        ~InspectorPanel() override = default;

        void OnUIRender() override;

    private:
        void RenderSimulationInspector();
        void RenderGroupInspector( DigitalTwin::AgentGroup& group );
        void RenderBehaviourInspector( DigitalTwin::BehaviourRecord& record );
        void RenderGridFieldInspector( DigitalTwin::GridField& field, int fieldIndex );

        DigitalTwin::DigitalTwin&         m_engine;
        DigitalTwin::SimulationBlueprint& m_blueprint;
        EditorSelection&                  m_selection;

        // Per-field visualization settings — persists while fields stay in the blueprint.
        std::vector<DigitalTwin::GridFieldVisualization> m_fieldVisCache;
    };
} // namespace Gaudi
