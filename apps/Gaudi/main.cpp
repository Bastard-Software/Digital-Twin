#include "Editor.h"

#include "panels/ConsolePanel.h"
#include "panels/HierarchyPanel.h"
#include "panels/InspectorPanel.h"
#include "panels/RenderSettingsPanel.h"
#include "panels/SimulationControlsPanel.h"
#include "panels/ViewportPanel.h"

int main()
{
    Gaudi::Editor app;
    app.Init();

    app.AddPanel( std::make_shared<Gaudi::SimulationControlsPanel>( app.GetEngine(), app.GetBlueprint() ) );
    app.AddPanel( std::make_shared<Gaudi::ViewportPanel>( app.GetEngine() ) );
    app.AddPanel( std::make_shared<Gaudi::HierarchyPanel>( app.GetEngine(), app.GetBlueprint(), app.GetSelection() ) );
    app.AddPanel( std::make_shared<Gaudi::InspectorPanel>( app.GetEngine(), app.GetBlueprint(), app.GetSelection() ) );

    auto consolePanel = std::make_shared<Gaudi::ConsolePanel>( app.GetLogSink() );
    app.SetConsolePanel( consolePanel );
    app.AddPanel( consolePanel );

    auto renderSettingsPanel = std::make_shared<Gaudi::RenderSettingsPanel>( app.GetEngine() );
    app.SetRenderSettingsPanel( renderSettingsPanel );
    app.AddPanel( renderSettingsPanel );

    app.Run();

    app.Shutdown();
    return 0;
}
