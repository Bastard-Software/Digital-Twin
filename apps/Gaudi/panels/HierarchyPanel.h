#pragma once
#include "EditorPanel.h"
#include "../EditorSelection.h"
#include <simulation/SimulationBlueprint.h>

namespace DigitalTwin
{
    class DigitalTwin;
}

namespace Gaudi
{
    class HierarchyPanel : public EditorPanel
    {
    public:
        HierarchyPanel( DigitalTwin::DigitalTwin& engine, DigitalTwin::SimulationBlueprint& blueprint, EditorSelection& selection );
        ~HierarchyPanel() override = default;

        void OnUIRender() override;

    private:
        DigitalTwin::DigitalTwin&         m_engine;
        DigitalTwin::SimulationBlueprint& m_blueprint;
        EditorSelection&                  m_selection;
    };
} // namespace Gaudi
