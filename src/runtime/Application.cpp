#include "runtime/Application.hpp"

#include "core/Log.hpp"
#include "core/Timer.hpp"
#include "platform/Input.hpp"
#include "platform/KeyCodes.hpp"
#include "renderer/ImGuiLayer.hpp"
#include "simulation/SimulationContext.hpp"
#include <imgui.h>

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
            m_engine->WaitIdle();
        m_renderer.reset();
        m_computeEngine.reset();
        m_engine.reset();
    }

    void Application::InitCore()
    {
        EngineConfig eConfig;
        eConfig.width    = m_config.width;
        eConfig.height   = m_config.height;
        eConfig.headless = m_config.headless;

        m_engine = CreateScope<Engine>();
        if( m_engine->Init( eConfig ) != Result::SUCCESS )
            throw std::runtime_error( "Failed to initialize Engine" );

        m_computeEngine = CreateRef<ComputeEngine>( m_engine->GetDevice() );
        m_computeEngine->Init();

        // FIX: Renderer constructor takes Engine&.
        // Swapchain/Context created internally in Renderer.
        m_renderer = CreateScope<Renderer>( *m_engine );

        if( m_simulation )
            m_simulation->InitializeRuntime( *m_engine, m_computeEngine );

        if( !m_config.headless && m_engine->GetWindow() )
        {
            m_engine->GetWindow()->Show();
            m_renderer->GetCamera().SetDistance( 20.0f );
        }
    }

    void Application::Run()
    {
        InitCore();
        DT_CORE_INFO( "[Application] Main Loop Started." );

        Timer timer;
        m_running     = true;
        bool isPaused = false;

        while( m_running )
        {
            float dt = timer.Elapsed();
            timer.Reset();
            if( dt > 0.1f )
                dt = 0.1f;

            // 1. Engine Start (Input, Transfer Buffers)
            m_engine->BeginFrame();
            m_renderer->OnUpdate( dt );

            if( m_engine->GetWindow()->IsClosed() )
                m_running = false;

            // 2. Simulation (Tick -> Uploads)
            if( m_simulation )
                m_simulation->Tick( dt );

            // 3. Render (Graphics)
            if( !m_config.headless )
            {
                // A. Acquire Image
                if( !m_renderer->BeginFrame() )
                {
                    // Swapchain invalid/minimized, skip render but finish engine frame
                    m_engine->EndFrame( *m_simulation, *m_renderer ); // Wait, this might crash if render didn't submit.
                    // Just continue loop
                    continue;
                }

                // B. Setup Scene
                Scene scene;
                scene.camera          = &m_renderer->GetCamera();
                scene.instanceBuffer  = m_simulation->GetContext()->GetRenderBuffer();
                scene.activeInstances = m_simulation->GetContext()->GetCounterBuffer();
                scene.instanceCount   = m_simulation->GetContext()->GetMaxCellCount();
                scene.activeMeshIDs   = m_simulation->GetActiveMeshes();

                // C. ImGui (Update & Resize)
                if( auto gui = m_renderer->GetGui() )
                {
                    gui->Begin();
                    gui->BeginDockspace();

                    ImGui::Begin( "Control" );
                    if( ImGui::Button( isPaused ? "Play" : "Pause" ) )
                    {
                        isPaused = !isPaused;
                        if( isPaused )
                            m_simulation->Pause();
                        else
                            m_simulation->Resume();
                    }

                    ImGui::Text( "Render: %.1f FPS (%.3f ms)", ImGui::GetIO().Framerate, 1000.0f / ImGui::GetIO().Framerate );
                    ImGui::Separator();

                    if( m_simulation )
                        m_simulation->OnRenderGui();
                    ImGui::End();

                    ImGui::PushStyleVar( ImGuiStyleVar_WindowPadding, ImVec2( 0, 0 ) );
                    ImGui::Begin( "Viewport" );
                    ImVec2 size = ImGui::GetContentRegionAvail();
                    if( size.x > 0 && size.y > 0 )
                        m_renderer->ResizeViewport( ( uint32_t )size.x, ( uint32_t )size.y );
                    ImGui::Image( m_renderer->GetViewportTextureID(), size );
                    ImGui::End();
                    ImGui::PopStyleVar();

                    gui->EndDockspace();
                }

                // D. Draw Scene (Offscreen)
                m_renderer->RenderSimulation( scene );

                // 4. Submit & Present
                m_engine->EndFrame( *m_simulation, *m_renderer );
            }
            else
            {
                // Headless path: just end engine frame to cycle resources
                // m_engine->EndFrame(...) - refactor if needed for headless
            }
        }
    }
} // namespace DigitalTwin