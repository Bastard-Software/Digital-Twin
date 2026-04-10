#pragma once
#include "EditorSelection.h"
#include "panels/EditorPanel.h"
#include <DigitalTwin.h>
#include <memory>
#include <vector>

namespace Gaudi
{
    using DemoSetupFn = void ( * )( DigitalTwin::SimulationBlueprint& );

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
        EditorSelection&                  GetSelection() { return m_selection; }

        void RenderMainMenuBar();
        void RenderDemoBrowser();

    private:
        void LoadDemo( DemoSetupFn fn );

    private:
        DigitalTwin::DigitalTwin                  m_engine;
        DigitalTwin::SimulationBlueprint          m_blueprint;
        EditorSelection                           m_selection;
        std::vector<std::shared_ptr<EditorPanel>> m_panels;
        bool                                      m_shouldQuit      = false;
        bool                                      m_showDemoBrowser = false;
    };
} // namespace Gaudi