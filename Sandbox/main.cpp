#include <app/Simulation.hpp>
#include <core/Base.hpp>
#include <runtime/Engine.hpp>

int main()
{
    DigitalTwin::EngineConfig engineConfig;
    engineConfig.headless = false;
    DigitalTwin::Engine::Init( engineConfig );

    DT_INFO( "=== CellSim Minimal Demo ===\n" );

    DigitalTwin::Simulation sim;
    sim.Init();

    while( !sim.IsComplete() )
    {
        sim.Step();
    }

    DT_TRACE( "=== Demo finished after %u steps ===\n", sim.GetCurrentStep() );
    return 0;
}