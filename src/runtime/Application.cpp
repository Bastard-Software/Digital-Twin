#include "runtime/Application.hpp"

#include "core/FileSystem.hpp"
#include "core/Log.hpp"
#include "core/Timer.hpp"
#include "platform/Input.hpp"
#include "simulation/SimulationContext.hpp"
#include "renderer/ImGuiLayer.hpp"
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
        // 1. Begin Frame (Acquire Swapchain Image)
        auto resMgr = m_engine->GetResourceManager();
        resMgr->BeginFrame( m_engine->GetFrameCount() );

        // BeginFrame resets the command buffer and prepares it for recording.
        // Returns nullptr if swapchain is out of date (window resize pending).
        auto cmd = m_renderer->GetContext()->BeginFrame();
        if( !cmd )
            return;

        // 2. Prepare Scene Data
        Scene scene;
        scene.instanceBuffer = m_simulation->GetContext()->GetCellBuffer();
        scene.instanceCount = m_simulation->GetContext()->GetMaxCellCount();
        scene.activeMeshIDs = m_simulation->GetActiveMeshes();
        scene.camera        = &m_renderer->GetCamera();

        // 3. UI Logic & Resize Handling (CRITICAL: MUST BE BEFORE RENDER SIMULATION)
        // We handle UI first because resizing the ImGui Viewport window triggers 'ResizeViewport'.
        // ResizeViewport destroys old textures and creates new ones.
        // If we rendered simulation first, we would be recording commands to textures that get destroyed immediately after.
        if( auto gui = m_renderer->GetGui() )
        {
            gui->Begin();
            gui->BeginDockspace();

            // -- Control Panel --
            ImGui::Begin( "Control Panel" );
            if( ImGui::Button( m_running ? "Pause ||" : "Play >", ImVec2( -1, 0 ) ) )
            {
                m_running = !m_running;
                if( m_running )
                    m_simulation->Resume();
                else
                    m_simulation->Pause();
            }
            ImGui::Separator();

            // Stats
            ImGui::Text( "Active Cells: %d", scene.instanceCount );
            ImGui::Text( "Frame Time: %.3f ms", 1000.0f / ImGui::GetIO().Framerate );

            if( m_simulation )
                m_simulation->OnRenderGui();
            ImGui::End();

            // -- Viewport Window --
            ImGui::PushStyleVar( ImGuiStyleVar_WindowPadding, ImVec2( 0, 0 ) );
            ImGui::Begin( "Viewport" );

            ImVec2 viewportSize = ImGui::GetContentRegionAvail();

            // Check if viewport size changed.
            // If yes, this will WaitIdle, destroy old textures, and create new ones.
            m_renderer->ResizeViewport( ( uint32_t )viewportSize.x, ( uint32_t )viewportSize.y );

            // Display the texture (uses the ID matching the current frame resources)
            ImGui::Image( m_renderer->GetViewportTextureID(), viewportSize );

            // Optional: Handle input block when mouse is not hovering viewport
            // m_renderer->GetCamera().SetInputEnabled( ImGui::IsItemHovered() );

            ImGui::End();
            ImGui::PopStyleVar();

            gui->EndDockspace();
        }

        // 4. Render Simulation (Offscreen Pass)
        // Now it is safe to record render commands because textures are guaranteed to be valid and sized correctly.
        m_renderer->RenderSimulation( scene );

        // 5. Submit Resource Transfers
        auto resSync = resMgr->EndFrame();

        // Prepare semaphores for the main graphics submission
        std::vector<VkSemaphore> waitSems;
        std::vector<uint64_t>    waitVals;

        // Wait for Resource Uploads (Transfer Queue)
        if( resSync.semaphore )
        {
            waitSems.push_back( resSync.semaphore );
            waitVals.push_back( resSync.value );
        }

        // Wait for Physics/Compute Engine
        uint64_t waitValue    = m_simulation->GetComputeSignal();
        auto     computeQueue = m_engine->GetDevice()->GetComputeQueue();
        if( computeQueue && waitValue > 0 )
        {
            VkSemaphore computeSem = computeQueue->GetTimelineSemaphore();
            if( computeSem )
            {
                waitSems.push_back( computeSem );
                waitVals.push_back( waitValue );
            }
        }

        // 6. Render UI to Swapchain & Present
        // This submits the command buffer (containing Sim + UI passes) and presents the image.
        m_renderer->RenderUI( waitSems, waitVals );
    }

    void Application::Close()
    {
        m_running = false;
    }

} // namespace DigitalTwin