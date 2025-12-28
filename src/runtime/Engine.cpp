#include "runtime/Engine.hpp"

#include "core/FileSystem.hpp"
#include "core/Log.hpp"
#include "platform/Input.hpp"
#include "renderer/Renderer.hpp"
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
        Log::Init();
        FileSystem::Init();

        if( m_initialized )
        {
            DT_CORE_WARN( "[Engine] Already initialized!" );
            return Result::SUCCESS;
        }

        m_config = config;
        DT_CORE_INFO( "[Engine] Init: {} ({}x{})", m_config.headless ? "HEADLESS" : "GUI", m_config.width, m_config.height );

        // 1. Create Window
        if( !m_config.headless )
        {
            WindowConfig winConfig;
            winConfig.width  = m_config.width;
            winConfig.height = m_config.height;
            winConfig.title  = "Digital Twin Simulation";
            m_window         = CreateRef<Window>( winConfig );
        }

        // 2. Init RHI Core
        RHIConfig rhiConfig;
        rhiConfig.enableValidation = true; // Dev mode
        rhiConfig.headless         = m_config.headless;

        if( RHI::Init( rhiConfig ) != Result::SUCCESS )
        {
            return Result::FAIL;
        }

        // 3. Create Device
        // Adapter 0
        m_device = RHI::CreateDevice( 0 );
        if( !m_device )
            return Result::FAIL;

        // 4. Initialize Device
        DeviceDesc deviceDesc;
        deviceDesc.headless = m_config.headless;
        if( m_device->Init( deviceDesc ) != Result::SUCCESS )
        {
            DT_CORE_CRITICAL( "[Engine] Device Init failed!" );
            return Result::FAIL;
        }

        // 5. Managers
        m_streamingManager = CreateRef<StreamingManager>( m_device );
        if( m_streamingManager->Init() != Result::SUCCESS )
            return Result::FAIL;

        m_resourceManager = CreateRef<ResourceManager>( m_device, m_streamingManager );

        m_initialized = true;
        return Result::SUCCESS;
    }

    void Engine::Shutdown()
    {
        if( m_initialized )
        {
            WaitIdle();

            m_resourceManager.reset();
            m_streamingManager.reset();

            if( m_device )
            {
                RHI::DestroyDevice( m_device );
                m_device.reset();
            }

            m_window.reset();
            RHI::Shutdown();

            m_initialized = false;
        }
    }

    void Engine::WaitIdle()
    {
        if( m_device )
            m_device->WaitIdle();
    }

    void Engine::PollEvents()
    {
        if( m_window )
            m_window->OnUpdate();
    }

    void Engine::BeginFrame()
    {
        PollEvents();
        m_frameCounter++;
        m_resourceManager->BeginFrame( m_frameCounter );
    }

    void Engine::EndFrame( Simulation& simulation, Renderer& renderer )
    {
        auto resSync = m_resourceManager->EndFrame();

        std::vector<VkSemaphore> waitSems;
        std::vector<uint64_t>    waitVals;

        if( resSync.semaphore )
        {
            waitSems.push_back( resSync.semaphore );
            waitVals.push_back( resSync.value );
        }

        // Renderer handles swapchain presentation internally
        renderer.RenderUI( waitSems, waitVals );

        Input::ResetScroll();
    }
} // namespace DigitalTwin