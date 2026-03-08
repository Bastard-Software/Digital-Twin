#include "Editor.h"
#include "panels/SimulationControlsPanel.h"
#include "panels/ViewportPanel.h"
#include "panels/EnvironmentPanel.h"

int main()
{
    Gaudi::Editor app;
    app.Init();

    app.AddPanel( std::make_shared<Gaudi::SimulationControlsPanel>( app.GetEngine(), app.GetBlueprint() ) );
    app.AddPanel( std::make_shared<Gaudi::ViewportPanel>( app.GetEngine() ) );
    app.AddPanel( std::make_shared<Gaudi::EnvironmentPanel>( app.GetEngine(), app.GetBlueprint() ) );

    app.Run();

    app.Shutdown();
    return 0;
}