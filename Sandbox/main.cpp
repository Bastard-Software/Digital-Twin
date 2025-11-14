#include <Core/Simulation.hpp>

int main()
{
    printf( "=== CellSim Minimal Demo ===\n" );

    DigitalTwin::Simulation sim;
    sim.Initialize();

    while( !sim.IsComplete() )
    {
        sim.Step();
    }

    printf( "=== Demo finished after %u steps ===\n", sim.GetCurrentStep() );
    return 0;
}