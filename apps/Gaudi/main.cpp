#include <DigitalTwin.h>
#include <core/Log.h>
#include <iostream>
#include <platform/Input.h>

int main()
{

    DigitalTwin::DigitalTwin       engine;
    DigitalTwin::DigitalTwinConfig config;
    config.headless = false;
    engine.Initialize( config );
    DT_INFO( "Starting Editor..." );

    // Create the window via Engine API
    auto window = engine.CreateWindow( "Gaudi Editor", 1280, 720 );

    if( !window.IsValid() )
    {
        DT_ERROR( "Failed to create main window!" );
        engine.Shutdown();
        return -1;
    }

    // Main Engine Loop
    while( !engine.IsWindowColsed(window) )
    {
        // 1. Poll events (input, window resize, etc.)
        engine.OnUpdate();
    }

    DT_INFO( "Editor closing..." );
    engine.Shutdown();

    return 0;
}