#include <DigitalTwin.h>
#include <core/Log.h>
#include <iostream>

int main()
{

    DigitalTwin::DigitalTwin       engine;
    DigitalTwin::DigitalTwinConfig config;
    engine.Initialize( config );
    DT_INFO( "Starting Editor..." );

    engine.Print();

    DT_INFO( "Editor closing..." );
    engine.Shutdown();

    return 0;
}