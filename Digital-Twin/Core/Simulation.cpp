#include "Simulation.hpp"

namespace DigitalTwin
{
    Simulation::Simulation()
        : m_currentStep( 0 )
    { 
        printf( "Simulation created\n" );
    }

    Simulation::~Simulation()
    {
        printf( "Simulation destroyed\n" );
    }

    void Simulation::Initialize()
    {
        printf( "Simulation initialized\n" );
        m_currentStep = 0;
    }

    void Simulation::Step()
    {
        ++m_currentStep;
        printf( "Simulation step %u executed\n", m_currentStep );
    }

    bool_t Simulation::IsComplete() const
    {
        return m_currentStep >= m_config.maxSteps;
    }
} // namespace DigitalTwin