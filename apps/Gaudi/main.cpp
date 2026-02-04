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

    // Main Engine Loop
    while( !engine.IsWindowClosed() )
    {
        // 1. Poll events (input, window resize, etc.)
        engine.BeginFrame();

        // 2. Editor Logic Here

        // 3. End Frame
        engine.EndFrame();
    }

    DT_INFO( "Editor closing..." );
    engine.Shutdown();

    return 0;
}