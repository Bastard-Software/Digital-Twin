#pragma once
#include "EditorPanel.h"
#include <simulation/SimulationBlueprint.h>
#include <DigitalTwinTypes.h>

namespace DigitalTwin
{
    class DigitalTwin;
}

namespace Gaudi
{
    class EnvironmentPanel : public EditorPanel
    {
    public:
        EnvironmentPanel( DigitalTwin::DigitalTwin& engine, DigitalTwin::SimulationBlueprint& blueprint );
        ~EnvironmentPanel() override = default;

        void OnUIRender() override;

    private:
        DigitalTwin::DigitalTwin&              m_engine;
        DigitalTwin::SimulationBlueprint&      m_blueprint;
        DigitalTwin::GridVisualizationSettings m_settings;
    };
} // namespace Gaudi