#pragma once
#include "core/Log.hpp"
#include "runtime/Application.hpp"

// The user must implement this function to return their Simulation instance.
extern DigitalTwin::Simulation* DigitalTwin::CreateSimulation();

int main( int argc, char** argv )
{
    DigitalTwin::Log::Init();

    // 1. Create the User's Experiment
    auto* experiment = DigitalTwin::CreateSimulation();

    // 2. Wrap it in the Application Host
    DigitalTwin::AppConfig config;
    config.width  = 1280;
    config.height = 720;
    DigitalTwin::Application app( experiment, config );

    // 3. Run
    try
    {
        app.Run();
    }
    catch( const std::exception& e )
    {
        DT_CORE_CRITICAL( "Application crash: {}", e.what() );
        return -1;
    }

    // 4. Cleanup
    delete experiment;
    return 0;
}