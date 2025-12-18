#include "runtime/Application.hpp"

#include "compute/ComputeKernel.hpp"
#include "core/FileSystem.hpp"
#include "core/Log.hpp"
#include "platform/Input.hpp"
#include <chrono>
#include <thread>

namespace DigitalTwin
{
    Application::Application( const AppConfig& config )
        : m_config( config )
    {
    }

    Application::~Application()
    {
        m_engine->WaitIdle();
        m_graph.Clear();

        // Order matters: Renderer/Sim depend on Engine/Device
        m_renderer.reset();
        m_simulation.reset();
        m_computeEngine.reset();
        m_engine.reset();
    }

    void Application::InitCore()
    {
        // 1. Initialize low-level Engine
        EngineConfig eConfig;
        eConfig.width    = m_config.width;
        eConfig.height   = m_config.height;
        eConfig.headless = m_config.headless;

        m_engine = CreateScope<Engine>();
        if( m_engine->Init( eConfig ) != Result::SUCCESS )
        {
            throw std::runtime_error( "Failed to initialize Engine" );
        }

        // 2. Initialize Subsystems
        m_simulation = CreateScope<Simulation>( *m_engine );
        m_renderer   = CreateScope<Renderer>( *m_engine );

        m_computeEngine = CreateRef<ComputeEngine>( m_engine->GetDevice() );
        m_computeEngine->Init();
    }

    void Application::Run()
    {
        InitCore();

        DT_CORE_INFO( "[Application] Phase 1: Configuring World..." );
        // 1. User sets up the agents (CPU side)
        OnConfigureWorld();

        DT_CORE_INFO( "[Application] Uploading Initial State to GPU..." );
        // 2. Engine creates buffers based on user config
        // This ensures GetCellBuffer() is valid in the next step.
        m_simulation->InitializeGPU();

        DT_CORE_INFO( "[Application] Phase 2: Configuring Physics..." );
        // 3. User sets up shaders (GPU side) - Buffers actully exist now!
        OnConfigurePhysics();

        if( !m_config.headless )
        {
            m_engine->GetWindow()->Show();
            m_renderer->GetCamera().SetDistance( 20.0f );
        }

        DT_CORE_INFO( "[Application] Starting Main Loop" );

        // Main Loop
        while( m_running )
        {
            // A. Platform Events
            if( m_engine->GetWindow() )
            {
                m_engine->GetWindow()->OnUpdate();
                if( m_engine->GetWindow()->IsClosed() )
                    Close();
                Input::ResetScroll();
            }

            // B. Resource Sync
            m_engine->GetResourceManager()->BeginFrame( m_engine->GetFrameCount() );

            // C. Logic Update
            m_renderer->OnUpdate( 0.016f );

            // D. Compute Step (Execute the graph built in OnSetup)
            uint64_t fenceValue = 0;

            // Only execute if graph has tasks
            if( m_simulation->GetContext()->GetMaxCellCount() > 0 )
            {
                fenceValue = m_computeEngine->ExecuteGraph( m_graph, m_simulation->GetContext()->GetMaxCellCount() );
            }

            // E. Render Step
            if( !m_config.headless )
            {
                Scene scene;
                scene.camera         = &m_renderer->GetCamera();
                scene.instanceBuffer = m_simulation->GetContext()->GetCellBuffer();
                scene.instanceCount  = m_simulation->GetContext()->GetMaxCellCount();
                scene.activeMeshIDs  = m_simulation->GetActiveMeshes();

                auto computeQueue = m_engine->GetDevice()->GetComputeQueue();
                auto resSync      = m_engine->GetResourceManager()->EndFrame();

                std::vector<VkSemaphore> waitSems = { computeQueue->GetTimelineSemaphore() };
                std::vector<uint64_t>    waitVals = { fenceValue };

                if( resSync.semaphore )
                {
                    waitSems.push_back( resSync.semaphore );
                    waitVals.push_back( resSync.value );
                }

                m_renderer->Render( scene, waitSems, waitVals );
            }
            else
            {
                m_engine->GetResourceManager()->EndFrame();
            }

            // F. Frame Cap
            std::this_thread::sleep_for( std::chrono::milliseconds( 16 ) );

            if( m_config.headless && m_engine->GetFrameCount() > 100 )
                Close();
        }
    }

    void Application::Close()
    {
        m_running = false;
    }
} // namespace DigitalTwin