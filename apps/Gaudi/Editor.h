#pragma once
#include "EditorSelection.h"
#include "panels/ConsolePanel.h"
#include "panels/EditorPanel.h"
#include "panels/RenderSettingsPanel.h"
#include <DigitalTwin.h>
#include <memory>
#include <optional>
#include <spdlog/sinks/ringbuffer_sink.h>
#include <string>
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

        std::shared_ptr<spdlog::sinks::ringbuffer_sink_mt> GetLogSink() const { return m_logSink; }
        void SetConsolePanel( std::shared_ptr<ConsolePanel> panel ) { m_consolePanel = panel; }
        void SetRenderSettingsPanel( std::shared_ptr<RenderSettingsPanel> panel ) { m_renderSettingsPanel = panel; }

        void RenderMainMenuBar();
        void RenderDemoBrowser();

    private:
        void LoadDemo( DemoSetupFn fn,
                       const std::optional<DigitalTwin::GridVisualizationSettings>& vizPreset = std::nullopt );

    private:
        DigitalTwin::DigitalTwin                  m_engine;
        DigitalTwin::SimulationBlueprint          m_blueprint;
        EditorSelection                           m_selection;
        std::vector<std::shared_ptr<EditorPanel>> m_panels;
        std::shared_ptr<spdlog::sinks::ringbuffer_sink_mt> m_logSink;
        std::shared_ptr<ConsolePanel>             m_consolePanel;
        std::shared_ptr<RenderSettingsPanel>      m_renderSettingsPanel;
        std::string                               m_userIniPath;
        bool                                      m_shouldQuit      = false;
        bool                                      m_showDemoBrowser = false;
    };
} // namespace Gaudi