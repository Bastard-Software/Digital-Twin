#include "runtime/Engine.hpp"

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
        // 1. Initialize Logging
        Log::Init();

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

        // 5. Initialize Streaming Manager (Data Transport)
        m_streamingManager = CreateRef<StreamingManager>( m_device );
        if( m_streamingManager->Init() != Result::SUCCESS )
        {
            DT_CORE_CRITICAL( "[Engine] Failed to initialize StreamingManager!" );
            return Result::FAIL;
        }

        m_initialized = true;
        return Result::SUCCESS;
    }

    void Engine::Shutdown()
    {
        if( m_initialized )
        {
            DT_CORE_INFO( "[Engine] Shutting down..." );

            // Wait for GPU to finish all work before destroying resources
            if( m_device )
            {
                m_device->GetAPI().vkDeviceWaitIdle( m_device->GetHandle() );
            }

            // Release managers first (they hold buffers)
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

    void Engine::Run( Simulation& simulation )
    {
        DT_CORE_INFO( "[Engine] Starting Main Loop..." );

        // Fixed timestep for physics/biology (e.g. 10ms = 100Hz)
        const double dt      = 0.01;
        bool         running = true;

        while( running )
        {
            // 1. Platform Events
            if( m_window )
            {
                m_window->OnUpdate();
                if( m_window->IsClosed() )
                    running = false;
            }

            // 2. Simulation Step (Compute)
            simulation.Step( static_cast<float>( dt ) );

            // 3. Rendering (Placeholder)
            // if (!m_config.headless) renderer.Draw(simulation.GetContext());

            // 4. Frame Limiter / VSync emulation
            std::this_thread::sleep_for( std::chrono::milliseconds( 16 ) ); // ~60 FPS

            m_frameCounter++;

            // For PoC safety: Stop after some frames if headless
            if( m_config.headless && m_frameCounter > 1000 )
            {
                running = false;
            }
        }

        DT_CORE_INFO( "[Engine] Loop finished after {} frames.", m_frameCounter );
    }
} // namespace DigitalTwin