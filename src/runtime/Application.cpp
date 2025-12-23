#include "runtime/Application.hpp"

#include "core/FileSystem.hpp"
#include "core/Log.hpp"
#include "core/Timer.hpp"
#include "platform/Input.hpp"
#include "simulation/SimulationContext.hpp"
#include <thread>

namespace DigitalTwin
{
    Application::Application( Simulation* userSimulation, const AppConfig& config )
        : m_simulation( userSimulation )
        , m_config( config )
    {
        DT_CORE_ASSERT( m_simulation, "User Simulation cannot be null!" );
    }

    Application::~Application()
    {
        if( m_engine )
        {
            m_engine->WaitIdle();
        }

        // Reset order matters
        m_renderer.reset();
        m_computeEngine.reset();
        m_engine.reset();
    }

    void Application::InitCore()
    {
        // 1. Initialize Engine
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
        m_computeEngine = CreateRef<ComputeEngine>( m_engine->GetDevice() );
        m_renderer      = CreateScope<Renderer>( *m_engine );

        // 3. Initialize User Simulation
        // This injects the dependencies into the user's class and triggers OnConfigureWorld/Systems
        if( m_simulation )
        {
            m_simulation->InitializeRuntime( *m_engine, m_computeEngine );
        }

        // 4. Show Window
        if( !m_config.headless && m_engine->GetWindow() )
        {
            m_engine->GetWindow()->Show();
            m_renderer->GetCamera().SetDistance( 20.0f );
        }
    }

    void Application::Run()
    {
        InitCore();

        DT_CORE_INFO( "[Application] Starting Main Loop..." );

        Timer timer;

        while( m_running )
        {
            float dt = timer.Elapsed();
            timer.Reset();

            // 1. Engine Housekeeping
            m_engine->BeginFrame();

            // 2. Process Window Events
            if( m_engine->GetWindow() )
            {
                m_engine->GetWindow()->OnUpdate();
                if( m_engine->GetWindow()->IsClosed() )
                    Close();
            }

            // 3. Simulation Tick (Includes Time Scaling, Scheduler, GPU Dispatch)
            if( m_simulation )
            {
                m_simulation->Tick( dt );
            }

            // 4. Render
            if( !m_config.headless )
            {
                Render();
            }

            // 5. Reset Input State
            if( m_engine->GetWindow() )
            {
                Input::ResetScroll();
            }
        }
    }

    void Application::Render()
    {
        // 1. Get Resource Manager
        auto resMgr = m_engine->GetResourceManager();

        // 2. Begin Transfer Frame
        // Prepares staging buffers and acquires the next swapchain image index if needed (depending on implementation)
        resMgr->BeginFrame( m_engine->GetFrameCount() );

        // 3. Update Renderer Logic
        // Updates camera matrices, UI state, and other per-frame logic
        m_renderer->OnUpdate( 0.016f );

        // 4. Prepare Scene Data
        // Collects all necessary buffers and mesh IDs from the simulation context
        Scene scene;
        scene.camera         = &m_renderer->GetCamera();
        scene.instanceBuffer = m_simulation->GetContext()->GetCellBuffer();
        scene.instanceCount  = m_simulation->GetContext()->GetMaxCellCount();
        scene.activeMeshIDs  = m_simulation->GetActiveMeshes();

        // 5. Submit Transfer Commands (CRITICAL FIX)
        // We must call EndFrame() BEFORE rendering. This submits the transfer command buffer to the GPU.
        // It returns a semaphore signaling when the data upload is complete.
        auto resSync = resMgr->EndFrame();

        // 6. Prepare Synchronization Primitives
        // The graphics queue needs to wait for two things:
        // A. Compute Physics to finish (so positions are updated)
        // B. Data Transfer to finish (so buffers/matrices are valid)
        std::vector<VkSemaphore> waitSems;
        std::vector<uint64_t>    waitVals;

        // A. Wait for Transfer
        if( resSync.semaphore != VK_NULL_HANDLE )
        {
            waitSems.push_back( resSync.semaphore );
            waitVals.push_back( resSync.value );
        }

        // B. Wait for Compute
        // Get the timeline value signaled by the compute engine this frame
        uint64_t waitValue = m_simulation->GetComputeSignal();

        // Retrieve the compute queue's timeline semaphore
        auto computeQueue = m_engine->GetDevice()->GetComputeQueue();
        if( computeQueue && waitValue > 0 )
        {
            VkSemaphore computeSem = computeQueue->GetTimelineSemaphore();
            if( computeSem != VK_NULL_HANDLE )
            {
                waitSems.push_back( computeSem );
                waitVals.push_back( waitValue );
            }
        }

        // 7. Execute Rendering
        // Submits graphics commands, waiting on the specified semaphores.
        // This also handles the final Swapchain Present.
        m_renderer->Render( scene, waitSems, waitVals );
    }

    void Application::Close()
    {
        m_running = false;
    }

} // namespace DigitalTwin