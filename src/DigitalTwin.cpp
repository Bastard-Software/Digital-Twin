#include "DigitalTwin.h"

#include "SetupHelpers.h"
#include "compute/GraphDispatcher.h"
#include "core/FileSystem.h"
#include "core/Log.h"
#include "core/Timer.h"
#include "core/jobs/JobSystem.h"
#include "core/memory/MemorySystem.h"
#include "platform/PlatformSystem.h"
#include "renderer/Camera.h"
#include "renderer/Renderer.h"
#include "renderer/Scene.h"
#include "resources/ResourceManager.h"
#include "resources/StreamingManager.h"
#include "rhi/Buffer.h"
#include "rhi/Device.h"
#include "rhi/GPUProfiler.h"
#include "rhi/RHI.h"
#include "rhi/Swapchain.h"
#include "rhi/ThreadContext.h"
#include "simulation/MorphologyGenerator.h"
#include "simulation/SimulationBuilder.h"
#include "simulation/SimulationState.h"
#include "simulation/SpatialDistribution.h"
#include <imgui.h>
#include <iostream>

#if defined( CreateWindow )
#    undef CreateWindow
#endif

namespace DigitalTwin
{
    struct FrameData
    {
        ThreadContextHandle graphicsContext;
        ThreadContextHandle computeContext;

        BufferHandle  agentBuffer;
        TextureHandle sceneTexture;

        uint64_t computeFinishedValue  = 0;
        uint64_t graphicsFinishedValue = 0;
    };

    // Impl
    struct DigitalTwin::Impl
    {
        DigitalTwinConfig       m_config;
        bool                    m_initialized;
        Scope<MemorySystem>     m_memorySystem;
        Scope<Timer>            m_timer;
        Scope<JobSystem>        m_jobSystem;
        Scope<FileSystem>       m_fileSystem;
        Scope<PlatformSystem>   m_platformSystem;
        Scope<RHI>              m_rhi;
        Scope<Device>           m_device;
        Scope<ResourceManager>  m_resourceManager;
        Scope<StreamingManager> m_streamingManager;
        Scope<Window>           m_window;
        Scope<Swapchain>        m_swapchain;
        Scope<Renderer>         m_renderer;
        Scope<Camera>           m_camera;
        SimulationBlueprint     m_blueprint;
        SimulationState         m_simulationState;
        EngineState             m_state = EngineState::RESET;
        ValidationResult        m_lastValidationResult;

        // Frame Data
        static const uint32_t FRAMES_IN_FLIGHT = 2;
        FrameData             m_frames[ FRAMES_IN_FLIGHT ];
        uint32_t              m_flightIndex       = 0; // Flight Index: Controls CPU resources and Sync Semaphores (0 -> 1 -> 0)
        uint32_t              m_currentImageIndex = 0; // Image Index: Controls which Swapchain Image we render to (obtained from Acquire)
        FrameContext          m_currentContext;

        // FPS tracking
        float    m_fpsTimer   = 0.0f;
        uint32_t m_fpsFrames  = 0;
        float    m_currentFps = 0.0f;
        float    m_totalTime  = 0.0f;

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
            m_memorySystem = CreateScope<MemorySystem>();
            m_memorySystem->Initialize();

            // 4. Timer
            m_timer = CreateScope<Timer>();

            // 5. Job System
            m_jobSystem = CreateScope<JobSystem>();
            JobSystem::Config jobConfig;
            jobConfig.forceSingleThreaded = true; // Start in single-threaded mode for debugging
            if( m_jobSystem->Initialize( jobConfig ) != Result::SUCCESS )
            {
                m_jobSystem.reset();
                m_timer.reset();
                m_memorySystem->Shutdown();
                m_memorySystem.reset();
            }

            // 6. FileSystem
            m_fileSystem                      = CreateScope<FileSystem>( m_memorySystem.get() );
            std::filesystem::path projectRoot = std::filesystem::current_path();
            std::filesystem::path engineRoot  = Helpers::FindEngineRoot();
            std::filesystem::path internalAssets;

            if( !engineRoot.empty() )
            {
                internalAssets = engineRoot / "assets";
            }
            else
            {
                // Last ditch effort: check if assets are next to the executable
                if( std::filesystem::exists( std::filesystem::current_path() / "assets" ) )
                {
                    internalAssets = std::filesystem::current_path() / "assets";
                }
            }

            // Initialize VFS
            Result fsRes = m_fileSystem->Initialize( projectRoot, internalAssets );
            if( fsRes != Result::SUCCESS )
            {
                // If checking strict project root failed, try CWD as fallback
                DT_WARN( "Project Root detection might have failed. Falling back to CWD." );
                fsRes = m_fileSystem->Initialize( std::filesystem::current_path(), internalAssets );

                if( fsRes != Result::SUCCESS )
                {
                    m_fileSystem.reset();
                    m_jobSystem->Shutdown();
                    m_jobSystem.reset();
                    m_timer.reset();
                    m_memorySystem->Shutdown();
                    m_memorySystem.reset();

                    DT_ERROR( "Critical: FileSystem could not be initialized." );
                    return fsRes;
                }
            }

            // 7. Platform System
            if( !m_config.headless )
            {
                m_platformSystem = CreateScope<PlatformSystem>();

                Result psRes = m_platformSystem->Initialize();
                if( psRes != Result::SUCCESS )
                {
                    m_platformSystem.reset();
                    m_fileSystem->Shutdown();
                    m_fileSystem.reset();
                    m_jobSystem->Shutdown();
                    m_jobSystem.reset();
                    m_timer.reset();
                    m_memorySystem->Shutdown();
                    m_memorySystem.reset();

                    DT_ERROR( "Failed to initialize PlatformSystem." );
                    return Result::FAIL;
                }
            }
            else
            {
                DT_INFO( "Running in Headless mode. Platform System skipped." );
            }

            // 8. RHI
            m_rhi = CreateScope<RHI>();

            RHIConfig rhiConfig;
            rhiConfig.headless         = m_config.headless;
            rhiConfig.enableValidation = true;

            std::vector<const char*> platformExtensions;
            if( m_platformSystem )
                platformExtensions = m_platformSystem->GetRequiredVulkanExtensions();

            if( m_rhi->Initialize( rhiConfig, platformExtensions ) != Result::SUCCESS )
            {
                m_rhi.reset();
                m_platformSystem->Shutdown();
                m_platformSystem.reset();
                m_fileSystem->Shutdown();
                m_fileSystem.reset();
                m_jobSystem->Shutdown();
                m_jobSystem.reset();
                m_timer.reset();
                m_memorySystem->Shutdown();
                m_memorySystem.reset();

                DT_ERROR( "Failed to initialize RHI." );
                return Result::FAIL;
            }

            // 9. Device
            const auto& adapters = m_rhi->GetAdapters();
            if( adapters.empty() )
            {
                m_rhi->Shutdown();
                m_rhi.reset();
                m_platformSystem->Shutdown();
                m_platformSystem.reset();
                m_fileSystem->Shutdown();
                m_fileSystem.reset();
                m_jobSystem->Shutdown();
                m_jobSystem.reset();
                m_timer.reset();
                m_memorySystem->Shutdown();
                m_memorySystem.reset();
                DT_ERROR( "No GPU Adapters found!" );
                return Result::FAIL;
            }
            uint32_t selectedAdapter = Helpers::SelectGPU( m_config.gpuType, adapters );

            if( m_rhi->CreateDevice( selectedAdapter, m_device ) != Result::SUCCESS )
            {
                m_device.reset();
                m_rhi->Shutdown();
                m_rhi.reset();
                m_platformSystem->Shutdown();
                m_platformSystem.reset();
                m_fileSystem->Shutdown();
                m_fileSystem.reset();
                m_jobSystem->Shutdown();
                m_jobSystem.reset();
                m_timer.reset();
                m_memorySystem->Shutdown();
                m_memorySystem.reset();
                DT_ERROR( "Failed to create Logical Device." );
                return Result::FAIL;
            }

            // 10. Resource Manager & Streaming Manager
            m_resourceManager = CreateScope<ResourceManager>( m_device.get(), m_memorySystem.get(), m_fileSystem.get() );
            if( m_resourceManager->Initialize() != Result::SUCCESS )
            {
                m_resourceManager.reset();
                m_device->Shutdown();
                m_device.reset();
                m_rhi->Shutdown();
                m_rhi.reset();
                m_platformSystem->Shutdown();
                m_platformSystem.reset();
                m_fileSystem->Shutdown();
                m_fileSystem.reset();
                m_jobSystem->Shutdown();
                m_jobSystem.reset();
                m_timer.reset();
                m_memorySystem->Shutdown();
                m_memorySystem.reset();
                DT_ERROR( "Failed to initialize Resource Manager." );
                return Result::FAIL;
            }

            m_streamingManager = CreateScope<StreamingManager>( m_device.get(), m_resourceManager.get() );
            if( m_streamingManager->Initialize() != Result::SUCCESS )
            {
                m_streamingManager.reset();
                m_resourceManager->Shutdown();
                m_resourceManager.reset();
                m_device->Shutdown();
                m_device.reset();
                m_rhi->Shutdown();
                m_rhi.reset();
                m_platformSystem->Shutdown();
                m_platformSystem.reset();
                m_fileSystem->Shutdown();
                m_fileSystem.reset();
                m_jobSystem->Shutdown();
                m_jobSystem.reset();
                m_timer.reset();
                m_memorySystem->Shutdown();
                m_memorySystem.reset();
                DT_ERROR( "Failed to initialize Streaming Manager." );
                return Result::FAIL;
            }

            // 11. Window, Swapchain, Renderer & Camera (if not headless)
            if( !m_config.headless && m_platformSystem )
            {
                m_window = m_platformSystem->CreateWindow( m_config.windowDesc );
                if( !m_window )
                {
                    m_window.reset();
                    m_streamingManager->Shutdown();
                    m_streamingManager.reset();
                    m_resourceManager->Shutdown();
                    m_resourceManager.reset();
                    m_device->Shutdown();
                    m_device.reset();
                    m_rhi->Shutdown();
                    m_rhi.reset();
                    m_platformSystem->Shutdown();
                    m_platformSystem.reset();
                    m_fileSystem->Shutdown();
                    m_fileSystem.reset();
                    m_jobSystem->Shutdown();
                    m_jobSystem.reset();
                    m_timer.reset();
                    m_memorySystem->Shutdown();
                    m_memorySystem.reset();
                    DT_ERROR( "Failed to create Main Window." );
                    return Result::FAIL;
                }

                m_swapchain = CreateScope<Swapchain>( m_device.get() );
                if( m_swapchain->Create( m_window.get(), false ) != Result::SUCCESS )
                {
                    m_swapchain.reset();
                    m_window.reset();
                    m_streamingManager->Shutdown();
                    m_streamingManager.reset();
                    m_resourceManager->Shutdown();
                    m_resourceManager.reset();
                    m_device->Shutdown();
                    m_device.reset();
                    m_rhi->Shutdown();
                    m_rhi.reset();
                    m_platformSystem->Shutdown();
                    m_platformSystem.reset();
                    m_fileSystem->Shutdown();
                    m_fileSystem.reset();
                    m_jobSystem->Shutdown();
                    m_jobSystem.reset();
                    m_timer.reset();
                    m_memorySystem->Shutdown();
                    m_memorySystem.reset();
                    DT_ERROR( "Failed to create Main Swapchain." );
                    return Result::FAIL;
                }

                m_renderer = CreateScope<Renderer>( m_device.get(), m_swapchain.get(), m_resourceManager.get(), m_streamingManager.get() );
                if( m_renderer->Create() != Result::SUCCESS )
                {
                    m_renderer.reset();
                    m_swapchain->Destroy();
                    m_swapchain.reset();
                    m_window.reset();
                    m_streamingManager->Shutdown();
                    m_streamingManager.reset();
                    m_resourceManager->Shutdown();
                    m_resourceManager.reset();
                    m_device->Shutdown();
                    m_device.reset();
                    m_rhi->Shutdown();
                    m_rhi.reset();
                    m_platformSystem->Shutdown();
                    m_platformSystem.reset();
                    m_fileSystem->Shutdown();
                    m_fileSystem.reset();
                    m_jobSystem->Shutdown();
                    m_jobSystem.reset();
                    m_timer.reset();
                    m_memorySystem->Shutdown();
                    m_memorySystem.reset();
                    DT_ERROR( "Failed to initialize Renderer." );
                    return Result::FAIL;
                }

                // Temp code - set up a basic camera for testing. This will be replaced with a more robust system later.
                m_camera = CreateScope<Camera>( 45.0f, 800.0f / 600.0f, 0.1f, 1000.0f );
                m_camera->SetFocalPoint( { 0.0f, 0.0f, 0.0f } );
                m_camera->SetDistance( 250.0f );

                // Create render resources
                for( uint32_t ndx = 0; ndx < FRAMES_IN_FLIGHT; ++ndx )
                {
                    FrameData& currFrame      = m_frames[ ndx ];
                    currFrame.graphicsContext = m_device->CreateThreadContext( QueueType::GRAPHICS );
                    currFrame.computeContext  = m_device->CreateThreadContext( QueueType::COMPUTE );

                    TextureDesc rtDesc;
                    rtDesc.width           = m_swapchain->GetExtent().width;
                    rtDesc.height          = m_swapchain->GetExtent().height;
                    rtDesc.format          = m_swapchain->GetFormat();
                    rtDesc.usage           = TextureUsage::RENDER_TARGET | TextureUsage::SAMPLED;
                    currFrame.sceneTexture = m_resourceManager->CreateTexture( rtDesc );
                }
            }

            m_initialized = true;

            return Result::SUCCESS;
        }

        void Shutdown()
        {
            if( !m_initialized )
                return;

            DT_INFO( "Shutting down..." );

            if( m_renderer )
            {
                m_renderer->Destroy();
                m_renderer.reset();
            }

            if( m_swapchain )
            {
                m_swapchain->Destroy();
                m_swapchain.reset();
            }

            if( m_window )
            {
                m_platformSystem->RemoveWindow( m_window.get() );
                m_window.reset();
            }

            if( m_streamingManager )
            {
                m_streamingManager->Shutdown();
                m_streamingManager.reset();
            }

            if( m_resourceManager )
            {
                m_resourceManager->Shutdown();
                m_resourceManager.reset();
            }

            if( m_device )
            {
                m_device->Shutdown();
                m_device.reset();
            }

            if( m_rhi )
            {
                m_rhi->Shutdown();
                m_rhi.reset();
            }

            if( m_platformSystem )
            {
                m_platformSystem->Shutdown();
                m_platformSystem.reset();
            }

            if( m_fileSystem )
            {
                m_fileSystem->Shutdown();
                m_fileSystem.reset();
            }

            if( m_jobSystem )
            {
                m_jobSystem->Shutdown();
                m_jobSystem.reset();
            }

            if( m_timer )
            {
                m_timer.reset();
            }

            if( m_memorySystem )
            {
                m_memorySystem->Shutdown();
                m_memorySystem.reset();
            }
            m_initialized = false;
        }

        void OnUpdate()
        {
            if( m_jobSystem )
            {
                DT_CORE_ASSERT( m_jobSystem->IsMainThread(), "OnUpdate must be called on the Main Thread!" );
            }

            // 1. Poll Events (GLFW updates window size inside this call)
            if( m_platformSystem )
            {
                m_platformSystem->OnUpdate();
            }

            m_timer->Tick();
            float dt = m_timer->GetDeltaTime();

            m_fpsFrames++;
            m_fpsTimer += dt;
            if( m_fpsTimer >= 1.0f )
            {
                m_currentFps = m_fpsFrames / m_fpsTimer;
                m_fpsFrames  = 0;
                m_fpsTimer -= 1.0f;
            }

            if( m_camera )
            {
                m_camera->OnUpdate( dt, m_platformSystem->GetInput() );
            }

            // 2. Check for Resize
            if( m_window && m_window->WasResized() )
            {
                // Check for minimization
                if( m_window->IsMinimized() )
                {
                    // Optional: Pause logic logic here
                    return;
                }

                DT_INFO( "Window resized to {0}x{1}. Recreating Swapchain...", m_window->GetWidth(), m_window->GetHeight() );

                // Recreate Swapchain
                if( m_swapchain )
                {
                    m_swapchain->Recreate();
                }

                // Acknowledge the flag so we don't recreate every frame
                m_window->ResetResizeFlag();
            }

            // 3. Process Main Thread Jobs
            if( m_jobSystem )
            {
                m_jobSystem->ProcessMainThread();
            }
        }

        const FrameContext& BeginFrame()
        {
            m_resourceManager->BeginFrame();
            m_streamingManager->BeginFrame();

            // 1. Advance Flight Index
            m_flightIndex        = ( m_flightIndex + 1 ) % FRAMES_IN_FLIGHT;
            FrameData& currFrame = m_frames[ m_flightIndex ];

            // 2. CPU-GPU Synchronization
            // We must wait for BOTH queues to finish with this slot's resources.
            // Graphics might be done, but Compute (running in parallel) might still be busy.
            std::vector<VkSemaphore> waitSemas;
            std::vector<uint64_t>    waitValues;

            // Wait for Graphics (Windowed mode)
            if( !m_config.headless && currFrame.graphicsFinishedValue > 0 )
            {
                waitSemas.push_back( m_device->GetGraphicsQueue()->GetTimelineSemaphore() );
                waitValues.push_back( currFrame.graphicsFinishedValue );
            }

            // Wait for Compute (Always)
            // Even in windowed mode, Compute N runs independently of Graphics N (which renders N-1).
            // So checking Graphics completion is NOT enough to ensure Compute N is done.
            if( currFrame.computeFinishedValue > 0 )
            {
                waitSemas.push_back( m_device->GetComputeQueue()->GetTimelineSemaphore() );
                waitValues.push_back( currFrame.computeFinishedValue );
            }

            // Execute Wait
            if( !waitSemas.empty() )
            {
                m_device->WaitForSemaphores( waitSemas, waitValues );
            }

            // 3. Acquire Image (Windowed Only)
            if( !m_config.headless )
            {
                VkSemaphore sem = m_swapchain->GetImageAvailableSemaphore( m_flightIndex );
                m_swapchain->AcquireNextImage( &m_currentImageIndex, sem );
            }
            else
            {
                m_currentImageIndex = m_flightIndex;
            }

            // 4. Reset Command Pools
            // Now safe because we waited for both queues!
            ThreadContext* gfxCtx = m_device->GetThreadContext( currFrame.graphicsContext );
            ThreadContext* cmpCtx = m_device->GetThreadContext( currFrame.computeContext );
            gfxCtx->Reset();
            cmpCtx->Reset();

            // 5. Allocate Command Buffers
            m_currentContext.graphicsCmd = gfxCtx->CreateCommandBuffer();
            m_currentContext.computeCmd  = cmpCtx->CreateCommandBuffer();

            // 6. Setup Context
            m_currentContext.frameIndex   = m_flightIndex;
            m_currentContext.sceneTexture = currFrame.sceneTexture;

            // 7. Start UI
            if( !m_config.headless && m_renderer )
                m_renderer->BeginUI();

            return m_currentContext;
        }

        void EndFrame()
        {
            if( m_renderer && !m_config.headless )
                m_renderer->EndUI();

            FrameData& currFrame = m_frames[ m_flightIndex ];

            ThreadContext* gfxCtx = m_device->GetThreadContext( currFrame.graphicsContext );
            ThreadContext* cmpCtx = m_device->GetThreadContext( currFrame.computeContext );

            // Allocate a secondary, independent graphics command buffer specifically for the Simulation Phase
            CommandBufferHandle gfxSimCmdHandle = gfxCtx->CreateCommandBuffer();

            // Fetch the primary command buffers for this frame
            CommandBuffer* gfxRenderCmd = gfxCtx->GetCommandBuffer( m_currentContext.graphicsCmd );
            CommandBuffer* compCmd      = cmpCtx->GetCommandBuffer( m_currentContext.computeCmd );
            CommandBuffer* gfxSimCmd    = gfxCtx->GetCommandBuffer( gfxSimCmdHandle );

            uint64_t transferSyncValue = m_streamingManager->EndFrame();

            // Track synchronization values for hardware-level barriers
            uint64_t computeSimSyncValue  = 0;
            uint64_t graphicsSimSyncValue = 0;

            Queue* computeQueue  = m_device->GetComputeQueue();
            Queue* graphicsQueue = m_device->GetGraphicsQueue();

            // =========================================================
            // PHASE 1: SIMULATION PASS (Async Dispatch)
            // =========================================================
            if( m_state == EngineState::PLAYING && m_simulationState.IsValid() )
            {
                float dt = m_timer->GetDeltaTime();
                m_totalTime += dt;

                // Record computational work onto BOTH queues simultaneously
                compCmd->Begin();
                gfxSimCmd->Begin();

                uint32_t finalActive = GraphDispatcher::Dispatch( &m_simulationState.computeGraph, compCmd, gfxSimCmd, dt, m_totalTime, m_simulationState.currentReadIndex );

                gfxSimCmd->End();
                compCmd->End();

                // latestAgentBuffer: the side that received the last position write this frame,
                // determined by the chain-flip mechanism. Used by the renderer so it always
                // reads the most recent agent positions without blinking on pause.
                m_simulationState.latestAgentBuffer = finalActive;

                // currentReadIndex drives the active-index for ALL tasks (field textures + agents)
                // next frame. It must alternate every frame so the field ping-pong evolves correctly
                // (diffusion always needs to read from the side it previously wrote to).
                m_simulationState.currentReadIndex = ( m_simulationState.currentReadIndex + 1 ) % 2;

                // 1A. Submit to COMPUTE queue
                {
                    std::vector<CommandBuffer*> cmdBuffers = { compCmd };
                    std::vector<VkSemaphore>    waitSemas;
                    std::vector<uint64_t>       waitVals;

                    // Simulation must wait for any pending memory transfers to finish
                    if( transferSyncValue > 0 )
                    {
                        waitSemas.push_back( m_device->GetTransferQueue()->GetTimelineSemaphore() );
                        waitVals.push_back( transferSyncValue );
                    }

                    computeQueue->Submit( cmdBuffers, waitSemas, waitVals );
                    computeSimSyncValue = computeQueue->GetLastSubmittedValue();
                }

                // 1B. Submit to GRAPHICS queue (Simulation workload only)
                {
                    std::vector<CommandBuffer*> cmdBuffers = { gfxSimCmd };
                    std::vector<VkSemaphore>    waitSemas;
                    std::vector<uint64_t>       waitVals;

                    // Graphics simulation must also wait for transfers
                    if( transferSyncValue > 0 )
                    {
                        waitSemas.push_back( m_device->GetTransferQueue()->GetTimelineSemaphore() );
                        waitVals.push_back( transferSyncValue );
                    }

                    graphicsQueue->Submit( cmdBuffers, waitSemas, waitVals );
                    graphicsSimSyncValue = graphicsQueue->GetLastSubmittedValue();
                }
            }

            // =========================================================
            // PHASE 2: RENDER PASS (Drawing to screen)
            // =========================================================
            gfxRenderCmd->Begin();

            if( !m_config.headless && m_renderer )
            {
                SimulationState* stateToRender = nullptr;
                if( m_state != EngineState::RESET && m_simulationState.IsValid() )
                {
                    stateToRender = &m_simulationState;
                }

                // Draw simulation geometry
                m_renderer->RenderSimulation( gfxRenderCmd, stateToRender, m_camera.get(), m_flightIndex );

                // Draw Editor UI
                Texture* backbuffer = m_swapchain->GetTexture( m_currentImageIndex );
                m_renderer->RecordUIPass( gfxRenderCmd, backbuffer );
            }

            gfxRenderCmd->End();

            // =========================================================
            // PHASE 3: GRAPHICS SUBMIT (Strictly synchronized with Simulation)
            // =========================================================
            if( !m_config.headless )
            {
                std::vector<CommandBuffer*> cmdBuffers = { gfxRenderCmd };
                std::vector<VkSemaphore>    waitSemas;
                std::vector<uint64_t>       waitValues;
                std::vector<VkSemaphore>    signalSemas;
                std::vector<uint64_t>       signalValues;

                // Wait A: Swapchain image must be acquired and ready
                waitSemas.push_back( m_swapchain->GetImageAvailableSemaphore( m_flightIndex ) );
                waitValues.push_back( 0 );

                // Wait B: Wait for the Compute Simulation to finish mutating agent buffers
                if( computeSimSyncValue > 0 )
                {
                    waitSemas.push_back( computeQueue->GetTimelineSemaphore() );
                    waitValues.push_back( computeSimSyncValue );
                }

                // Wait C: Wait for the Graphics Simulation to finish
                // (Even though it runs on the same hardware queue as rendering, the timeline semaphore
                // ensures a full memory barrier protects the geometry buffers)
                if( graphicsSimSyncValue > 0 )
                {
                    waitSemas.push_back( graphicsQueue->GetTimelineSemaphore() );
                    waitValues.push_back( graphicsSimSyncValue );
                }

                // Wait D: If simulation is RESET but a transfer occurred, render must wait for it
                if( m_state != EngineState::PLAYING && transferSyncValue > 0 )
                {
                    waitSemas.push_back( m_device->GetTransferQueue()->GetTimelineSemaphore() );
                    waitValues.push_back( transferSyncValue );
                }

                // Signal: Render is completely finished, ready for presentation
                VkSemaphore renderFinishedSem = m_swapchain->GetRenderFinishedSemaphore( m_flightIndex );
                signalSemas.push_back( renderFinishedSem );
                signalValues.push_back( 0 );

                graphicsQueue->Submit( cmdBuffers, waitSemas, waitValues, signalSemas, signalValues );

                // Save the checkpoint for the next frame's Present()
                currFrame.graphicsFinishedValue = graphicsQueue->GetLastSubmittedValue();
            }

            // =========================================================
            // PHASE 4: PRESENT
            // =========================================================
            if( !m_config.headless )
            {
                VkSemaphore renderFinishedSem = m_swapchain->GetRenderFinishedSemaphore( m_flightIndex );
                m_swapchain->Present( m_currentImageIndex, renderFinishedSem );
            }
        }
    };

    DigitalTwin::DigitalTwin()
        : m_impl( CreateScope<Impl>() )
    {
    }

    DigitalTwin::~DigitalTwin()
    {
    }

    Result DigitalTwin::Initialize( const DigitalTwinConfig& config )
    {
        return m_impl->Initialize( config );
    }

    void DigitalTwin::Shutdown()
    {
        m_impl->Shutdown();
    }

    void DigitalTwin::SetBlueprint( const SimulationBlueprint& blueprint )
    {
        m_impl->m_blueprint = blueprint;
    }

    const FrameContext& DigitalTwin::BeginFrame()
    {
        m_impl->OnUpdate();

        return m_impl->BeginFrame();
    }

    void DigitalTwin::EndFrame()
    {
        m_impl->EndFrame();
    }

    void DigitalTwin::Step()
    {
        DT_ASSERT( false, "Not implemented yet!" );
    }

    void DebugDumpAgentBuffers( SimulationState& state, ResourceManager* rm, StreamingManager* stream )
    {
        if( !state.IsValid() )
            return;

        // Ensure all compute queues have finished before we readback
        // (Assuming you call this when simulation is paused/stopped, so GPU should be idle)

        DT_INFO( "==========================================================" );
        DT_INFO( "[DEBUG] DUMPING AGENT BUFFERS STATE" );
        DT_INFO( "==========================================================" );

        uint32_t totalActive = 0;
        // --- 1. Read Atomic Counters (Actual group sizes) ---
        if( state.agentCountBuffer.IsValid() )
        {
            // Assuming we allocated enough space for groupCount uint32_t counters
            size_t countBytes = state.groupCount * sizeof( uint32_t );
            if( countBytes > 0 )
            {
                std::vector<uint32_t> counters( state.groupCount );
                stream->ReadbackBufferImmediate( state.agentCountBuffer, counters.data(), countBytes );

                for( size_t i = 0; i < state.groupCount; ++i )
                {
                    DT_INFO( "  Group [{}]: {} active agents (Atomic Counter)", i, counters[ i ] );
                    totalActive += counters[ i ];
                }
            }
        }

        // --- Helper lambda to analyze a single buffer ---
        auto AnalyzeBuffer = [ & ]( BufferHandle handle, const std::string& name ) {
            if( !handle.IsValid() )
                return;

            size_t   bufSize   = rm->GetBuffer( handle )->GetSize();
            uint32_t maxAgents = bufSize / sizeof( glm::vec4 );

            std::vector<glm::vec4> agents( maxAgents );
            stream->ReadbackBufferImmediate( handle, agents.data(), bufSize );

            uint32_t validCount   = 0;
            uint32_t zeroPosCount = 0;

            for( uint32_t i = 0; i < maxAgents; ++i )
            {
                if( agents[ i ].w > 0.0f )
                    validCount++;
                if( agents[ i ].x == 0.0f && agents[ i ].y == 0.0f && agents[ i ].z == 0.0f )
                    zeroPosCount++;
            }

            DT_INFO( "  {}: MaxCapacity: {}, Valid (w>0): {}, At (0,0,0): {}", name, maxAgents, validCount, zeroPosCount );

            for( uint32_t i = 0; i < totalActive; ++i )
            {
                DT_INFO( "    Agent {}: pos({:0.2f}, {:0.2f}, {:0.2f}), w={:0.2f}", i, agents[ i ].x, agents[ i ].y, agents[ i ].z, agents[ i ].w );
            }
        };

        // --- 2. Analyze Ping-Pong Buffers ---
        AnalyzeBuffer( state.agentBuffers[ 0 ], "Buffer[0]" );
        AnalyzeBuffer( state.agentBuffers[ 1 ], "Buffer[1]" );

        DT_INFO( "==========================================================" );
    }

    void DigitalTwin::Play()
    {
        if( m_impl->m_state == EngineState::RESET )
        {
            m_impl->m_lastValidationResult = SimulationValidator::Validate( m_impl->m_blueprint );
            for( const auto& issue : m_impl->m_lastValidationResult.issues )
            {
                if( issue.severity == ValidationIssue::Severity::Error )
                    DT_ERROR( "[SimulationValidator] {}", issue.message );
                else
                    DT_WARN( "[SimulationValidator] {}", issue.message );
            }
            if( !m_impl->m_lastValidationResult.IsValid() )
            {
                DT_ERROR( "[SimulationBuilder] Blueprint validation failed. Aborting build." );
                return;
            }

            if( m_impl->m_device )
                m_impl->m_device->WaitIdle();

            SimulationBuilder builder( m_impl->m_resourceManager.get(), m_impl->m_streamingManager.get() );
            m_impl->m_simulationState = builder.Build( m_impl->m_blueprint );

            m_impl->m_state = EngineState::PLAYING;
            DT_INFO( "Simulation State: PLAYING (Allocated)" );
        }
        else if( m_impl->m_state == EngineState::PAUSED )
        {
            m_impl->m_state = EngineState::PLAYING;
            DT_INFO( "Simulation State: PLAYING (Resumed)" );
        }
    }

    void DigitalTwin::Pause()
    {
        if( m_impl->m_state == EngineState::PLAYING )
        {
            m_impl->m_state = EngineState::PAUSED;
            DT_INFO( "Simulation State: PAUSED" );
        }
    }

    void DigitalTwin::Stop()
    {
        if( m_impl->m_state != EngineState::RESET )
        {
            if( m_impl->m_device )
                m_impl->m_device->WaitIdle();
            // Uncomment for reedback logging
            // DebugDumpAgentBuffers( m_impl->m_simulationState, m_impl->m_resourceManager.get(), m_impl->m_streamingManager.get() );

            // Reset visualization flags
            auto gridSettings   = m_impl->m_renderer->GetGridVisualization();
            gridSettings.active = false;
            m_impl->m_renderer->SetGridVisualization( gridSettings );

            // Destroy GPU buffers
            if( m_impl->m_simulationState.IsValid() )
            {
                m_impl->m_simulationState.Destroy( m_impl->m_resourceManager.get() );
            }

            m_impl->m_state = EngineState::RESET;
            DT_INFO( "Simulation State: RESET (Memory freed)" );
        }
    }

    void DigitalTwin::HotReload( const SimulationBlueprint& blueprint )
    {
        if( m_impl->m_state == EngineState::RESET )
            return;

        SimulationBuilder builder( m_impl->m_resourceManager.get(), m_impl->m_streamingManager.get() );
        builder.UpdateParameters( blueprint, m_impl->m_simulationState );
    }

    EngineState DigitalTwin::GetState() const
    {
        return m_impl->m_state;
    }

    const ValidationResult& DigitalTwin::GetLastValidationResult() const
    {
        return m_impl->m_lastValidationResult;
    }

    bool DigitalTwin::IsWindowClosed()
    {
        if( m_impl->m_window )
        {
            return m_impl->m_window->IsClosed();
        }

        return false;
    }

    void DigitalTwin::SetGridVisualization( const GridVisualizationSettings& settings )
    {
        if( m_impl->m_renderer )
            m_impl->m_renderer->SetGridVisualization( settings );
    }

    const GridVisualizationSettings& DigitalTwin::GetGridVisualization() const
    {
        if( m_impl->m_renderer )
            return m_impl->m_renderer->GetGridVisualization();
        static GridVisualizationSettings def;
        return def;
    }

    void DigitalTwin::RenderUI( std::function<void()> uiCallback )
    {
        if( m_impl->m_renderer )
        {
            m_impl->m_renderer->RenderUI( [ & ]() {
                // Engine stats rendering
                ImGuiIO&    io               = ImGui::GetIO();
                const float PAD              = 30.0f;
                ImVec2      window_pos       = ImVec2( io.DisplaySize.x - PAD, PAD );
                ImVec2      window_pos_pivot = ImVec2( 1.0f, 0.0f );

                ImGui::SetNextWindowPos( window_pos, ImGuiCond_Always, window_pos_pivot );
                ImGui::SetNextWindowBgAlpha( 0.3f );
                if( ImGui::Begin( "Engine Stats", nullptr,
                                  ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoSavedSettings |
                                      ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoNav | ImGuiWindowFlags_NoMove ) )
                {
                    ImGui::Text( "FPS: %.1f", m_impl->m_currentFps );
                    ImGui::Text( "Frame Time: %.2f ms", 1000.0f / ( m_impl->m_currentFps > 0.01f ? m_impl->m_currentFps : 1.0f ) );

                    if( GPUProfiler* profiler = m_impl->m_device->GetProfiler() )
                    {
                        {
                            // Memory usage
                            ImGui::Separator();
                            ImGui::TextColored( ImVec4( 1.0f, 0.8f, 0.2f, 1.0f ), "GPU Memory" );
                            const auto memStats = profiler->GetMemoryStats();
                            double     usedMB   = static_cast<double>( memStats.currentUsage ) / ( 1024.0 * 1024.0 );
                            double     totalMB  = static_cast<double>( memStats.totalBudget ) / ( 1024.0 * 1024.0 );
                            ImGui::TextColored( ImVec4( 0.4f, 0.8f, 1.0f, 1.0f ), "VRAM: %.1f MB / %.1f MB", usedMB, totalMB );
                            float fraction = ( memStats.totalBudget > 0 )
                                                 ? static_cast<float>( memStats.currentUsage ) / static_cast<float>( memStats.totalBudget )
                                                 : 0.0f;

                            if( fraction > 0.85f )
                            {
                                ImGui::PushStyleColor( ImGuiCol_PlotHistogram, ImVec4( 0.8f, 0.2f, 0.2f, 1.0f ) );
                            }
                            else if( fraction > 0.65f )
                            {
                                ImGui::PushStyleColor( ImGuiCol_PlotHistogram, ImVec4( 0.8f, 0.8f, 0.2f, 1.0f ) );
                            }
                            else
                            {
                                ImGui::PushStyleColor( ImGuiCol_PlotHistogram, ImVec4( 0.2f, 0.8f, 0.2f, 1.0f ) );
                            }

                            ImGui::ProgressBar( fraction, ImVec2( -FLT_MIN, 0.0f ) );
                            ImGui::PopStyleColor();
                        }

                        {
                            // Pipeline stats
                            ImGui::Separator();
                            ImGui::TextColored( ImVec4( 1.0f, 0.8f, 0.2f, 1.0f ), "GPU Stats" );
                            const auto& results = profiler->GetResults();
                            for( const auto& [ zoneName, data ]: results )
                            {
                                ImGui::Text( "- [%s] -", zoneName.c_str() );
                                ImGui::Text( "  Time: %.3f ms", data.timeMs );

                                // Output in Millions (M) for readability
                                ImGui::TextColored( ImVec4( 0.5f, 1.0f, 0.5f, 1.0f ), "  Verts: %.3f M",
                                                    ( float )data.vertexShaderInvocations / 1000000.0f );
                                ImGui::TextColored( ImVec4( 1.0f, 0.5f, 0.5f, 1.0f ), "  Frags: %.3f M",
                                                    ( float )data.fragmentShaderInvocations / 1000000.0f );
                                ImGui::TextColored( ImVec4( 0.8f, 0.8f, 0.8f, 1.0f ), "  Clip : %.3f M",
                                                    ( float )data.clippingInvocations / 1000000.0f );
                            }
                        }
                    }
                }
                ImGui::End();

                if( uiCallback )
                    uiCallback();
            } );
        }
    }

    void DigitalTwin::SetViewportSize( uint32_t width, uint32_t height )
    {
        if( m_impl->m_renderer )
        {
            m_impl->m_renderer->SetViewportSize( width, height );
            if( m_impl->m_camera )
            {
                m_impl->m_camera->OnResize( width, height );
            }
        }
    }

    void DigitalTwin::GetViewportSize( uint32_t& outWidth, uint32_t& outHeight ) const
    {
        if( m_impl->m_renderer )
        {
            m_impl->m_renderer->GetViewportSize( outWidth, outHeight );
        }
        else
        {
            outWidth  = 0;
            outHeight = 0;
        }
    }

    void* DigitalTwin::GetSceneTextureID() const
    {
        if( m_impl->m_renderer )
        {
            return m_impl->m_renderer->GetSceneTextureID( m_impl->m_flightIndex );
        }
        return nullptr;
    }

    void* DigitalTwin::GetImGuiTextureID( TextureHandle handle )
    {
        ( void )handle;
        DT_ASSERT( false, "Not implemented" );
        return nullptr;
    }

    void* DigitalTwin::GetImGuiContext()
    {
        if( m_impl->m_renderer )
        {
            return m_impl->m_renderer->GetImGuiContext();
        }
        return nullptr;
    }

    FileSystem* DigitalTwin::GetFileSystem() const
    {
        return m_impl->m_fileSystem.get();
    }

} // namespace DigitalTwin