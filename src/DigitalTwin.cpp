#include "DigitalTwin.h"

#include "core/Log.h"
#include "core/memory/MemorySystem.h"
#include <iostream>

namespace DigitalTwin
{

    // Impl
    struct DigitalTwin::Impl
    {
        DigitalTwinConfig             m_config;
        bool                          m_initialized;
        std::unique_ptr<MemorySystem> m_memorySystem;

        Impl()
            : m_initialized( false )
        {
        }

        Result Initialize( const DigitalTwinConfig& config )
        {
            if( m_initialized )
                return Result::SUCCESS;

            // 1. Config
            m_config = config;

            // 2. Logger
            Log::Init();
            DT_INFO( "Initializing DigitalTwin Engine..." );

            // 3. Memory
            m_memorySystem = std::make_unique<MemorySystem>();
            m_memorySystem->Initialize();

            m_initialized = true;

            return Result::SUCCESS;
        }

        void Shutdown()
        {
            if( !m_initialized )
                return;

            DT_INFO( "Shutting down..." );

            if( m_memorySystem )
            {
                m_memorySystem->Shutdown();
                m_memorySystem.reset();
            }
            m_initialized = false;
        }
    };

    DigitalTwin::DigitalTwin()
        : m_impl( std::make_unique<Impl>() )
    {
        std::cout << "[DLL] DigitalTwin Initialized." << std::endl;
    }

    DigitalTwin::~DigitalTwin()
    {
        std::cout << "[DLL] DigitalTwin Destroyed." << std::endl;
    }

    Result DigitalTwin::Initialize( const DigitalTwinConfig& config )
    {
        return m_impl->Initialize( config );
    }

    void DigitalTwin::Shutdown()
    {
        m_impl->Shutdown();
    }

    void DigitalTwin::Print()
    {
        std::cout << "------------------------------------------------" << std::endl;
        std::cout << "Hello from DLL!" << std::endl;
        std::cout << "Linker works properly if you see this message." << std::endl;
        std::cout << "------------------------------------------------" << std::endl;
    }

} // namespace DigitalTwin