#include "DigitalTwin.h"

#include "core/FileSystem.h"
#include "core/Log.h"
#include "core/jobs/JobSystem.h"
#include "core/memory/MemorySystem.h"
#include "platform/PlatformSystem.h"
#include "renderer/Renderer.h"
#include "resources/ResourceManager.h"
#include "resources/StreamingManager.h"
#include "rhi/Device.h"
#include "rhi/RHI.h"
#include "rhi/Swapchain.h"
#include "rhi/ThreadContext.h"
#include <iostream>

#if defined( CreateWindow )
#    undef CreateWindow
#endif

namespace DigitalTwin
{

    std::filesystem::path FindProjectRoot()
    {
        std::filesystem::path p = std::filesystem::current_path();

        // Safety limit to avoid scanning entire disk
        for( int i = 0; i < 10; ++i )
        {
            // Check if this looks like a source folder:
            // 1. Has CMakeLists.txt
            // 2. Does NOT have 'CMakeFiles' folder (which implies it's a build dir)
            // 3. Does NOT have 'CMakeCache.txt'
            bool hasCMake = std::filesystem::exists( p / "CMakeLists.txt" );
            bool isBuild  = std::filesystem::exists( p / "CMakeFiles" ) || std::filesystem::exists( p / "CMakeCache.txt" );

            if( hasCMake && !isBuild )
            {
                return p;
            }

            // Also check if we simply need to strip "build" from path (Common in VS)
            std::string pathStr  = p.generic_string();
            size_t      buildPos = pathStr.find( "/build/" );
            if( buildPos != std::string::npos )
            {
                std::string newPathStr = pathStr;
                newPathStr.replace( buildPos, 7, "/" ); // Remove "build/"
                std::filesystem::path trySource = newPathStr;
                if( std::filesystem::exists( trySource ) )
                {
                    return trySource;
                }
            }

            if( p.has_parent_path() )
            {
                p = p.parent_path();
            }
            else
            {
                break;
            }
        }

        // Fallback: Just return CWD if heuristic fails
        return std::filesystem::current_path();
    }

    std::filesystem::path FindEngineRoot( const std::filesystem::path& projectRoot )
    {
        std::filesystem::path p = projectRoot;

        for( int i = 0; i < 10; ++i )
        {
            // We recognize Engine Root by existence of "src", "include" and "assets"
            if( std::filesystem::exists( p / "src" ) && std::filesystem::exists( p / "include" ) && std::filesystem::exists( p / "assets" ) )
            {
                return p;
            }
            if( p.has_parent_path() )
            {
                p = p.parent_path();
            }
            else
            {
                break;
            }
        }
        return {}; // Not found
    }

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

        // Frame Data
        static const uint32_t FRAMES_IN_FLIGHT = 2;
        FrameData             m_frames[ FRAMES_IN_FLIGHT ];
        uint32_t              m_flightIndex       = 0; // Flight Index: Controls CPU resources and Sync Semaphores (0 -> 1 -> 0)
        uint32_t              m_currentImageIndex = 0; // Image Index: Controls which Swapchain Image we render to (obtained from Acquire)
        FrameContext          m_currentContext;

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

            // 4. Job System
            m_jobSystem = CreateScope<JobSystem>();
            JobSystem::Config jobConfig;
            jobConfig.forceSingleThreaded = true; // Start in single-threaded mode for debugging
            if( m_jobSystem->Initialize( jobConfig ) != Result::SUCCESS )
            {
                m_jobSystem.reset();
            }

            // 5. FileSystem
            m_fileSystem = CreateScope<FileSystem>( m_memorySystem.get() );
            // Resolve Project Root (Priority 1 - User/App files)
            std::filesystem::path projectRoot;
            if( m_config.rootDirectory && m_config.rootDirectory[ 0 ] != '\0' )
            {
                projectRoot = m_config.rootDirectory;
            }
            else
            {
                projectRoot = FindProjectRoot();
            }

            // Resolve Engine Assets (Priority 2 - Fallback/Default files)
            std::filesystem::path engineRoot = FindEngineRoot( projectRoot );
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
                    m_memorySystem->Shutdown();
                    m_memorySystem.reset();

                    DT_ERROR( "Critical: FileSystem could not be initialized." );
                    return fsRes;
                }
            }

            // 6. Platform System
            if( !m_config.headless )
            {
                m_platformSystem = CreateScope<PlatformSystem>();

                Result psRes = m_platformSystem->Initialize();
                if( fsRes != Result::SUCCESS )
                {
                    m_platformSystem.reset();
                    m_fileSystem->Shutdown();
                    m_fileSystem.reset();
                    m_jobSystem->Shutdown();
                    m_jobSystem.reset();
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

            // 7. RHI
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
                m_memorySystem->Shutdown();
                m_memorySystem.reset();

                DT_ERROR( "Failed to initialize RHI." );
                return Result::FAIL;
            }

            // 8. Device
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
                m_memorySystem->Shutdown();
                m_memorySystem.reset();
                DT_ERROR( "No GPU Adapters found!" );
                return Result::FAIL;
            }
            uint32_t selectedAdapter = SelectGPU( m_config.gpuType, adapters );

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
                m_memorySystem->Shutdown();
                m_memorySystem.reset();
                DT_ERROR( "Failed to create Logical Device." );
                return Result::FAIL;
            }

            // 9. Resource Manager & Streaming Manager
            m_resourceManager = CreateScope<ResourceManager>( m_device.get(), m_memorySystem.get() );
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
                m_memorySystem->Shutdown();
                m_memorySystem.reset();
                DT_ERROR( "Failed to initialize Streaming Manager." );
                return Result::FAIL;
            }

            // 10. Window, Swapchain & Renderer (if not headless)
            if( !m_config.headless && m_platformSystem )
            {
                m_window = m_platformSystem->CreateWindow( { m_config.windowTitle, m_config.windowWidth, m_config.windowHeight } );
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
                    m_memorySystem->Shutdown();
                    m_memorySystem.reset();
                    DT_ERROR( "Failed to create Main Window." );
                    return Result::FAIL;
                }

                m_swapchain = CreateScope<Swapchain>( m_device.get() );
                if( m_swapchain->Create( m_window.get(), true ) != Result::SUCCESS )
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
                    m_memorySystem->Shutdown();
                    m_memorySystem.reset();
                    DT_ERROR( "Failed to initialize Renderer." );
                    return Result::FAIL;
                }

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

            if( m_memorySystem )
            {
                m_memorySystem->Shutdown();
                m_memorySystem.reset();
            }
            m_initialized = false;
        }

        /**
         * @brief Selects the best GPU adapter index based on the user preference.
         * @param preference The GPUType preference (Default, Discrete, Integrated).
         * @param adapters The list of available adapters.
         * @return The index of the selected adapter.
         */
        uint32_t SelectGPU( GPUType preference, const std::vector<AdapterInfo>& adapters )
        {
            if( adapters.empty() )
                return 0;

            // 1. Handle Explicit Preference: DISCRETE
            // Return the first discrete GPU found.
            if( preference == GPUType::DISCRETE )
            {
                for( uint32_t i = 0; i < adapters.size(); ++i )
                {
                    if( adapters[ i ].IsDiscrete() )
                    {
                        DT_INFO( "GPU Selection: Forced DISCRETE. Found: {0}", adapters[ i ].name );
                        return i;
                    }
                }
                DT_WARN( "GPU Selection: Preferred DISCRETE GPU not found. Falling back to DEFAULT logic." );
            }
            // 2. Handle Explicit Preference: INTEGRATED
            // Return the first integrated GPU found.
            else if( preference == GPUType::INTEGRATED )
            {
                for( uint32_t i = 0; i < adapters.size(); ++i )
                {
                    // Check explicitly for integrated type
                    if( adapters[ i ].type == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU )
                    {
                        DT_INFO( "GPU Selection: Forced INTEGRATED. Found: {0}", adapters[ i ].name );
                        return i;
                    }
                }
                DT_WARN( "GPU Selection: Preferred INTEGRATED GPU not found. Falling back to DEFAULT logic." );
            }

            // 3. Handle DEFAULT (or Fallback)
            // Algorithm:
            // - Prioritize Discrete GPUs over Integrated.
            // - Within the same category, pick the one with the largest VRAM.

            int      bestDiscreteIndex = -1;
            uint64_t maxDiscreteVRAM   = 0;

            int      bestIntegratedIndex = -1;
            uint64_t maxIntegratedVRAM   = 0;

            for( uint32_t i = 0; i < adapters.size(); ++i )
            {
                const auto& info = adapters[ i ];
                if( info.IsDiscrete() )
                {
                    if( info.deviceMemorySize >= maxDiscreteVRAM )
                    {
                        maxDiscreteVRAM   = info.deviceMemorySize;
                        bestDiscreteIndex = ( int )i;
                    }
                }
                else
                {
                    // Treat everything else (Integrated, Virtual, CPU) as secondary
                    if( info.deviceMemorySize >= maxIntegratedVRAM )
                    {
                        maxIntegratedVRAM   = info.deviceMemorySize;
                        bestIntegratedIndex = ( int )i;
                    }
                }
            }

            // Prioritize Discrete
            if( bestDiscreteIndex != -1 )
            {
                DT_INFO( "GPU Selection: DEFAULT -> Selected Best Discrete: {0}", adapters[ bestDiscreteIndex ].name );
                return ( uint32_t )bestDiscreteIndex;
            }

            // Then Integrated
            if( bestIntegratedIndex != -1 )
            {
                DT_INFO( "GPU Selection: DEFAULT -> Selected Best Integrated: {0}", adapters[ bestIntegratedIndex ].name );
                return ( uint32_t )bestIntegratedIndex;
            }

            // Absolute fallback
            DT_WARN( "GPU Selection: Logic failed to find optimal GPU. Returning index 0." );
            return 0;
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

            // 6. Begin Recording
            CommandBuffer* gfx = gfxCtx->GetCommandBuffer( m_currentContext.graphicsCmd );
            gfx->Begin();
            CommandBuffer* comp = cmpCtx->GetCommandBuffer( m_currentContext.computeCmd );
            comp->Begin();

            // 7. Setup Context
            m_currentContext.frameIndex   = m_flightIndex;
            m_currentContext.sceneTexture = currFrame.sceneTexture;

            // TODO: ping pong logic here

            // 8. Start UI
            if( !m_config.headless && m_renderer )
                m_renderer->BeginUI();

            return m_currentContext;
        }

        void EndFrame()
        {
            if( m_renderer && !m_config.headless )
                m_renderer->EndUI();

            FrameData& currFrame = m_frames[ m_flightIndex ];

            ThreadContext* gfxCtx  = m_device->GetThreadContext( currFrame.graphicsContext );
            ThreadContext* cmpCtx  = m_device->GetThreadContext( currFrame.computeContext );
            CommandBuffer* gfxCmd  = gfxCtx->GetCommandBuffer( m_currentContext.graphicsCmd );
            CommandBuffer* compCmd = cmpCtx->GetCommandBuffer( m_currentContext.computeCmd );

            // 1. Record UI Pass
            if( !m_config.headless && m_renderer )
            {
                Texture* backbuffer = m_swapchain->GetTexture( m_currentImageIndex );
                m_renderer->RecordUIPass( gfxCmd, backbuffer );
            }

            gfxCmd->End();
            compCmd->End();

            Queue* computeQueue  = m_device->GetComputeQueue();
            Queue* graphicsQueue = m_device->GetGraphicsQueue();

            // 2. SUBMIT TRANSFER
            uint64_t transferSyncValue = m_streamingManager->EndFrame();

            // 3. SUBMIT COMPUTE
            {
                std::vector<CommandBuffer*> cmdBuffers = { compCmd };
                std::vector<VkSemaphore>    waitSemas;
                std::vector<uint64_t>       waitVals;

                // Wait for transfers if any occured
                if( transferSyncValue > 0 )
                {
                    waitSemas.push_back( m_device->GetTransferQueue()->GetTimelineSemaphore() );
                    waitVals.push_back( transferSyncValue );
                }

                computeQueue->Submit( cmdBuffers, waitSemas, waitVals );

                // Track compute completion for this slot
                currFrame.computeFinishedValue = computeQueue->GetLastSubmittedValue();
            }

            // 4. SUBMIT GRAPHICS
            if( !m_config.headless )
            {
                std::vector<CommandBuffer*> cmdBuffers = { gfxCmd };

                std::vector<VkSemaphore> waitSemas;
                std::vector<uint64_t>    waitValues;
                std::vector<VkSemaphore> signalSemas;
                std::vector<uint64_t>    signalValues;

                // Wait 1: Transfer if resources were uploaded
                if( transferSyncValue > 0 )
                {
                    waitSemas.push_back( m_device->GetTransferQueue()->GetTimelineSemaphore() );
                    waitValues.push_back( transferSyncValue );
                }

                // Wait 2: Swapchain Image
                waitSemas.push_back( m_swapchain->GetImageAvailableSemaphore( m_flightIndex ) );
                waitValues.push_back( 0 );

                // Wait 2: Previous Compute Frame (Pipelining)
                uint32_t   prevFlightIndex = m_flightIndex ^ 1;
                FrameData& prevFrame       = m_frames[ prevFlightIndex ];
                if( prevFrame.computeFinishedValue > 0 )
                {
                    waitSemas.push_back( computeQueue->GetTimelineSemaphore() );
                    waitValues.push_back( prevFrame.computeFinishedValue );
                }

                // Signal: Render Finished
                VkSemaphore renderFinishedSem = m_swapchain->GetRenderFinishedSemaphore( m_flightIndex );
                signalSemas.push_back( renderFinishedSem );
                signalValues.push_back( 0 );

                graphicsQueue->Submit( cmdBuffers, waitSemas, waitValues, signalSemas, signalValues );

                // Track graphics completion for this slot
                currFrame.graphicsFinishedValue = graphicsQueue->GetLastSubmittedValue();
            }
            else
            {
                // TODO: Not true, fix that
                // In headless, graphics isn't used, so reset this tracker to avoid waiting on old values
                currFrame.graphicsFinishedValue = 0;
            }

            // 5. PRESENT
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

    bool DigitalTwin::IsWindowClosed()
    {
        if( m_impl->m_window )
        {
            return m_impl->m_window->IsClosed();
        }

        return false;
    }

    void DigitalTwin::RenderUI( std::function<void()> uiCallback )
    {
        if( m_impl->m_renderer )
        {
            m_impl->m_renderer->RenderUI( uiCallback );
        }
    }

    void* DigitalTwin::GetImGuiTextureID( TextureHandle handle )
    {
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