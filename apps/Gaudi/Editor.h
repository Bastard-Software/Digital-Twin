#pragma once
#include "EditorSelection.h"
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
        EditorSelection&                  GetSelection() { return m_selection; }

        void RenderMainMenuBar();
        void RenderDemoBrowser();

    private:
        void LoadDemo( void ( Editor::*setupFn )() );

        void SetupEmptyBlueprint();
        void SetupDiffusionDecayDemo();
        void SetupBrownianMotionDemo();
        void SetupJKRPackingDemo();
        void SetupLifecycleDemo();
        void SetupSecreteDemo();
        void SetupConsumeDemo();
        void SetupChemotaxisDemo();
        void SetupCellCycleDemo();
        void SetupSimpleVesselDebugDemo();
        void SetupInitialBlueprint();   // legacy full angiogenesis, kept for internal use

    private:
        DigitalTwin::DigitalTwin                  m_engine;
        DigitalTwin::SimulationBlueprint          m_blueprint;
        EditorSelection                           m_selection;
        std::vector<std::shared_ptr<EditorPanel>> m_panels;
        bool                                      m_shouldQuit      = false;
        bool                                      m_showDemoBrowser = false;
    };
} // namespace Gaudi