#include <app/Simulation.hpp>
#include <core/Base.hpp>

int main()
{
    DigitalTwin::Log::Init();

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