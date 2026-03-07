#pragma once
#include "panels/EditorPanel.h"
#include <DigitalTwin.h>
#include <memory>
#include <vector>

namespace Gaudi
{
    class Editor
    {
    public:
        Editor()  = default;
        ~Editor() = default;

        void Init();
        void Run();
        void Shutdown();

        void AddPanel( std::shared_ptr<EditorPanel> panel );

        DigitalTwin::DigitalTwin&         GetEngine() { return m_engine; }
        DigitalTwin::SimulationBlueprint& GetBlueprint() { return m_blueprint; }

    private:
        void SetupInitialBlueprint();

    private:
        DigitalTwin::DigitalTwin                  m_engine;
        DigitalTwin::SimulationBlueprint          m_blueprint;
        std::vector<std::shared_ptr<EditorPanel>> m_panels;
    };
} // namespace Gaudi