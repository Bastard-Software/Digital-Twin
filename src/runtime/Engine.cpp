#include "runtime/Engine.hpp"

#include "core/FileSystem.hpp"
#include "core/Log.hpp"
#include "rhi/RHI.hpp"
#include "simulation/Simulation.hpp"
#include <chrono>
#include <thread>

namespace DigitalTwin
{
    Engine::Engine() = default;

    Engine::~Engine()
    {
        Shutdown();
    }

    Result Engine::Init( const EngineConfig& config )
    {
        // 1. Initialize Logging and
        Log::Init();
        FileSystem::Init();

        if( m_initialized )
        {
            DT_CORE_WARN( "[Engine] Already initialized!" );
            return Result::SUCCESS;
        }

        m_config = config;
        DT_CORE_INFO( "[Engine] Initializing... Mode: {} ({}x{})", m_config.headless ? "HEADLESS" : "GRAPHICS", m_config.width, m_config.height );

        // 2. Create Window (if not headless)
        if( !m_config.headless )
        {
            WindowConfig winConfig;
            winConfig.width  = m_config.width;
            winConfig.height = m_config.height;
            winConfig.title  = "Digital Twin Simulation";
            m_window         = CreateScope<Window>( winConfig );
        }

        // 3. Initialize RHI Core (Volk, Instance)
        RHIConfig rhiConfig;
        rhiConfig.enableValidation = true;
        rhiConfig.headless         = m_config.headless;

        if( RHI::Init( rhiConfig ) != Result::SUCCESS )
        {
            DT_CORE_CRITICAL( "[Engine] Failed to initialize RHI!" );
            return Result::FAIL;
        }

        // 4. Create GPU Device (Adapter 0 - usually discrete GPU)
        // Note: In a robust engine, we would select the best adapter.
        m_device = RHI::CreateDevice( 0 );
        if( !m_device )
        {
            DT_CORE_CRITICAL( "[Engine] Failed to create Logical Device!" );
            return Result::FAIL;
        }

        // 5. Initialize Streaming adn Resources Manager
        m_streamingManager = CreateRef<StreamingManager>( m_device );
        if( m_streamingManager->Init() != Result::SUCCESS )
        {
            DT_CORE_CRITICAL( "[Engine] Failed to initialize StreamingManager!" );
            return Result::FAIL;
        }
        m_resourceManager = CreateRef<ResourceManager>( m_device, m_streamingManager );

        m_initialized = true;
        return Result::SUCCESS;
    }

    void Engine::Shutdown()
    {
        if( m_initialized )
        {
            DT_CORE_INFO( "[Engine] Shutting down..." );

            WaitIdle();

            // Release managers first (they hold buffers)
            m_resourceManager.reset();
            m_streamingManager.reset();

            // Destroy Device using RHI helper
            if( m_device )
            {
                RHI::DestroyDevice( m_device );
                m_device.reset();
            }

            // Destroy Window
            m_window.reset();

            // Shutdown RHI Core
            RHI::Shutdown();

            m_initialized = false;
        }
    }

    void Engine::BeginFrame()
    {
        // 1. Increment global frame counter
        m_frameCounter++;

        // 2. Reset transient descriptor pools.
        // This is crucial! Every frame (even compute-only), we might allocate
        // new descriptors for binding groups. We need to clear the old ones.
        // if( m_device )
        // {
        //     m_device->ResetDescriptorPools();
        // }
    }

    void Engine::WaitIdle()
    {
        if( m_device )
        {
            m_device->WaitIdle();
        }
    }
} // namespace DigitalTwin