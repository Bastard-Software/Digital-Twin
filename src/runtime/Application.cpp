#include "runtime/Application.hpp"

#include "compute/ComputeKernel.hpp"
#include "core/FileSystem.hpp"
#include "core/Log.hpp"
#include "core/Timer.hpp"
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

        DT_CORE_INFO( "[Application] Starting HPC Loop" );

        // Timers
        Timer timer;
        float accumulator = 0.0f;

        // Settings
        const float physicsStep         = 1.0f / 60.0f; // Fizyka liczona dok³adnie 60 razy na sek
        const float renderStep          = 1.0f / 30.0f; // Render co ~33ms
        float       timeSinceLastRender = 0.0f;

        // Synchronization logic
        uint64_t lastFenceValue = 0;

        timer.Reset();

        while( m_running )
        {
            // 1. Measure delta time
            // We clamp huge delta times (e.g. debugging breakpoints) to avoid spiral of death
            float dt = timer.Elapsed();
            timer.Reset();
            if( dt > 0.1f )
                dt = 0.1f;

            accumulator += dt;
            timeSinceLastRender += dt;

            // 2. Poll Events (Always active)
            if( m_engine->GetWindow() )
            {
                m_engine->GetWindow()->OnUpdate();
                if( m_engine->GetWindow()->IsClosed() )
                    Close();
                Input::ResetScroll();
            }

            // 3. PHYSICS LOOP (Fixed Update)
            // If the game runs slow, this loop might run multiple times per frame to catch up.
            // If the game runs fast, this might not run at all in some iterations.
            bool physicsUpdated = false;

            while( accumulator >= physicsStep )
            {
                // A. Sync Protection
                // We are about to submit work. Ensure previous GPU work is done.
                if( lastFenceValue > 0 )
                {
                    auto device = m_engine->GetDevice();
                    auto queue  = device->GetComputeQueue();
                    // Wait for GPU to finish the PREVIOUS step before submitting a NEW one.
                    device->WaitForQueue( queue, lastFenceValue );
                }

                // B. Engine Prep (Reset pools logic managed by Engine)
                // Note: ideally we call BeginFrame once per logic tick
                m_engine->BeginFrame();

                // C. Execute Physics
                if( m_simulation->GetContext()->GetMaxCellCount() > 0 && !m_graph.IsEmpty() )
                {
                    lastFenceValue = m_computeEngine->ExecuteGraph( m_graph, m_simulation->GetContext()->GetMaxCellCount() );
                }

                accumulator -= physicsStep;
                physicsUpdated = true;
            }

            // 4. RENDER LOOP (Capped)
            // Render only if enough time passed AND we actually updated physics (to avoid jitter)
            if( !m_config.headless && timeSinceLastRender >= renderStep )
            {
                // Reset render timer, keep remainder to maintain smooth pacing
                timeSinceLastRender = 0.0f;

                m_engine->GetResourceManager()->BeginFrame( m_engine->GetFrameCount() );
                m_renderer->OnUpdate( renderStep );

                Scene scene;
                scene.camera         = &m_renderer->GetCamera();
                scene.instanceBuffer = m_simulation->GetContext()->GetCellBuffer();
                scene.instanceCount  = m_simulation->GetContext()->GetMaxCellCount();
                scene.activeMeshIDs  = m_simulation->GetActiveMeshes();

                auto computeQueue = m_engine->GetDevice()->GetComputeQueue();
                auto resSync      = m_engine->GetResourceManager()->EndFrame();

                std::vector<VkSemaphore> waitSems = { computeQueue->GetTimelineSemaphore() };
                std::vector<uint64_t>    waitVals = { lastFenceValue };

                if( resSync.semaphore )
                {
                    waitSems.push_back( resSync.semaphore );
                    waitVals.push_back( resSync.value );
                }

                m_renderer->Render( scene, waitSems, waitVals );
            }
            else
            {
                // Sleep to prevent burning CPU if we are way ahead of schedule
                // (Optional, but good for laptops)
                if( accumulator < physicsStep )
                {
                    std::this_thread::sleep_for( std::chrono::milliseconds( 1 ) );
                }
            }
        }
    }

    void Application::Close()
    {
        m_running = false;
    }
} // namespace DigitalTwin