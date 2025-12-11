#include "core/Log.hpp"
#include "runtime/Engine.hpp"
#include "simulation/Simulation.hpp"

using namespace DigitalTwin;

int main()
{
    // 1. Configure Engine
    EngineConfig config;
    config.headless = false; // We want a window!
    config.width    = 1280;
    config.height   = 720;

    // 2. Initialize Runtime
    Engine engine;
    if( engine.Init( config ) != Result::SUCCESS )
    {
        DT_CORE_CRITICAL( "Engine failed to initialize!" );
        return -1;
    }

    // 3. Configure Simulation (The "Biological" part)
    Simulation sim( engine );

    sim.SetMicroenvironment( 0.5f, 9.81f );

    DT_CORE_INFO( "Spawning initial population..." );
    for( int i = 0; i < 1000; ++i )
    {
        sim.SpawnCell( { ( float )i * 0.1f, 0.0f, 0.0f }, // Position
                       { 0.1f, 0.0f, 0.0f },              // Velocity
                       { 1.0f, 0.0f, 0.0f, 1.0f }         // Color (Red)
        );
    }

    // 4. Upload Config to GPU
    sim.InitializeGPU();

    // 5. Run Main Loop
    engine.Run( sim );

    return 0;
}