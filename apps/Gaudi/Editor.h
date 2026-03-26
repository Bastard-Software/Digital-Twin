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

    private:
        void SetupInitialBlueprint();
        void SetupChemotaxisDemo();
        void SetupCellCycleDemo();
        void SetupAngiogenesisDemo();
        void SetupSimpleVesselDebugDemo();

    private:
        DigitalTwin::DigitalTwin                  m_engine;
        DigitalTwin::SimulationBlueprint          m_blueprint;
        EditorSelection                           m_selection;
        std::vector<std::shared_ptr<EditorPanel>> m_panels;
    };
} // namespace Gaudi