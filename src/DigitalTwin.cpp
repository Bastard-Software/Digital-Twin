#include "DigitalTwin.h"

#include "core/FileSystem.h"
#include "core/Log.h"
#include "core/jobs/JobSystem.h"
#include "core/memory/MemorySystem.h"
#include "platform/PlatformSystem.h"
#include "resources/ResourceMAnager.h"
#include "rhi/Device.h"
#include "rhi/RHI.h"
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

    // Impl
    struct DigitalTwin::Impl
    {
        DigitalTwinConfig      m_config;
        bool                   m_initialized;
        Scope<MemorySystem>    m_memorySystem;
        Scope<JobSystem>       m_jobSystem;
        Scope<FileSystem>      m_fileSystem;
        Scope<PlatformSystem>  m_platformSystem;
        Scope<RHI>             m_rhi;
        Scope<Device>          m_device;
        Scope<ResourceManager> m_resourceManager;

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

            // 9. Resource Manager
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

            m_initialized = true;

            return Result::SUCCESS;
        }

        void Shutdown()
        {
            if( !m_initialized )
                return;

            DT_INFO( "Shutting down..." );

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

    Scope<Window> DigitalTwin::CreateWindow( const std::string& title, uint32_t width, uint32_t height )
    {
        WindowDesc desc = { title, width, height };

        return m_impl->m_platformSystem->CreateWindow( desc );
    }

    void DigitalTwin::OnUpdate()
    {
        m_impl->m_platformSystem->OnUpdate();
    }

    FileSystem* DigitalTwin::GetFileSystem() const
    {
        return m_impl->m_fileSystem.get();
    }

} // namespace DigitalTwin