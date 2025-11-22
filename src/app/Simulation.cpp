#include "app/Simulation.hpp"

#include <glm/glm.hpp>
#include <shaderc/shaderc.hpp>
#include <volk.h>

namespace DigitalTwin
{
    Simulation::Simulation()
        : m_currentStep( 0 )
    {
        printf( "Simulation created\n" );
        volkInitialize();
        glm::vec4 temp = glm::vec4( 1.0f );
        temp;
        auto entity = m_registry.create();
        entity;

    }

    Simulation::~Simulation()
    {
        volkFinalize();
        printf( "Simulation destroyed\n" );
    }

    void Simulation::Initialize()
    {
        printf( "Simulation initialized\n" );
        printf( "Volk version: %u\n", VOLK_HEADER_VERSION );
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