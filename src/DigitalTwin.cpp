#include "DigitalTwin.h"

#include "core/FileSystem.h"
#include "core/Log.h"
#include "core/memory/MemorySystem.h"
#include <iostream>

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
        DigitalTwinConfig             m_config;
        bool                          m_initialized;
        std::unique_ptr<MemorySystem> m_memorySystem;
        std::unique_ptr<FileSystem>   m_fileSystem;

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
            m_memorySystem = std::make_unique<MemorySystem>();
            m_memorySystem->Initialize();

            // 4. FileSystem
            m_fileSystem = std::make_unique<FileSystem>( m_memorySystem.get() );
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
                    m_memorySystem->Shutdown();
                    m_memorySystem.reset();

                    DT_ERROR( "Critical: FileSystem could not be initialized." );
                    return fsRes;
                }
            }

            // 5. ....

            m_initialized = true;

            return Result::SUCCESS;
        }

        void Shutdown()
        {
            if( !m_initialized )
                return;

            DT_INFO( "Shutting down..." );

            if( m_fileSystem )
            {
                m_fileSystem->Shutdown();
                m_fileSystem.reset();
            }

            if( m_memorySystem )
            {
                m_memorySystem->Shutdown();
                m_memorySystem.reset();
            }
            m_initialized = false;
        }
    };

    DigitalTwin::DigitalTwin()
        : m_impl( std::make_unique<Impl>() )
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

    void DigitalTwin::Print()
    {
        std::cout << "------------------------------------------------" << std::endl;
        std::cout << "Hello from DLL!" << std::endl;
        std::cout << "Linker works properly if you see this message." << std::endl;
        std::cout << "------------------------------------------------" << std::endl;
    }

    FileSystem* DigitalTwin::GetFileSystem() const
    {
        return m_impl->m_fileSystem.get();
    }

} // namespace DigitalTwin