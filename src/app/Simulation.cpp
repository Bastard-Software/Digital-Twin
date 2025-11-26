#include "app/Simulation.hpp"

#include "core/Base.hpp"
#include "runtime/Engine.hpp"
#include <glm/glm.hpp>
#include <shaderc/shaderc.hpp>
#include <volk.h>

namespace DigitalTwin
{
    Simulation::Simulation()
        : m_currentStep( 0 )
    {
        if( !Engine::IsInitialized() )
        {
            DT_CORE_WARN( "Simulation started without explicit Engine initialization. Using defaults." );

            EngineConfig defaultConfig;
            defaultConfig.headless = false;
            Engine::Init( defaultConfig );
        }

        DT_CORE_INFO( "Simulation created" );
        volkInitialize();
        glm::vec4 temp = glm::vec4( 1.0f );
        temp;
        auto entity = m_registry.create();
        entity;
    }

    Simulation::~Simulation()
    {
        volkFinalize();
        DT_CORE_INFO( "Simulation destroyed" );
    }

    void Simulation::Init()
    {
        DT_CORE_TRACE( "Simulation initialized\n" );
        DT_CORE_TRACE( "Volk version: %u\n", VOLK_HEADER_VERSION );
        m_currentStep = 0;
    }

    void Simulation::Step()
    {
        ++m_currentStep;
        DT_CORE_TRACE( "Simulation step %u executed\n", m_currentStep );
    }

    bool_t Simulation::IsComplete() const
    {
        return m_currentStep >= m_config.maxSteps;
    }
} // namespace DigitalTwin