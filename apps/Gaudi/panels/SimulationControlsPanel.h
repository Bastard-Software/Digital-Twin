#pragma once
#include "EditorPanel.h"
#include <simulation/SimulationBlueprint.h>

namespace DigitalTwin
{
    class DigitalTwin;
}

namespace Gaudi
{
    class SimulationControlsPanel : public EditorPanel
    {
    public:
        SimulationControlsPanel( DigitalTwin::DigitalTwin& engine, DigitalTwin::SimulationBlueprint& blueprint );
        ~SimulationControlsPanel() override = default;

        void OnUIRender() override;

    private:
        DigitalTwin::DigitalTwin&         m_engine;
        DigitalTwin::SimulationBlueprint& m_blueprint;
    };
} // namespace Gaudi