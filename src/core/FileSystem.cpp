#include "core/FileSystem.h"

#include "core/Log.h"
#include "core/memory/MemorySystem.h"
#include <fstream>

namespace DigitalTwin
{

    FileSystem::FileSystem( MemorySystem* memorySystem )
        : m_memorySystem( memorySystem )
    {
    }

    FileSystem::~FileSystem()
    {
        Shutdown();
    }

    Result FileSystem::Initialize( const std::filesystem::path& projectRoot, const std::filesystem::path& engineAssetsPath )
    {
        if( m_initialized )
            return Result::SUCCESS;

        // 1. Verify Project Root (Must exist for writing)
        if( !std::filesystem::exists( projectRoot ) )
        {
            DT_ERROR( "FileSystem: Project Root '{0}' does not exist.", projectRoot.string() );
            return Result::INVALID_ARGS;
        }
        m_projectRoot = std::filesystem::absolute( projectRoot );

        // 2. Verify Engine Assets (Optional, but usually required for engine defaults)
        if( std::filesystem::exists( engineAssetsPath ) )
        {
            m_engineAssetsPath = std::filesystem::absolute( engineAssetsPath );
        }
        else
        {
            DT_WARN( "FileSystem: Engine Assets path '{0}' not found. Default assets won't load.", engineAssetsPath.string() );
            // We don't fail here, maybe the user wants to run without defaults.
        }

        m_initialized = true;
        DT_INFO( "FileSystem Initialized." );
        DT_INFO( "  Project Root:  {0}", m_projectRoot.string() );
        DT_INFO( "  Engine Assets: {0}", m_engineAssetsPath.string() );

        return Result::SUCCESS;
    }

    void FileSystem::Shutdown()
    {
        if( m_initialized )
        {
            DT_INFO( "FileSystem Shutdown." );
            m_initialized = false;
        }
    }

    Result FileSystem::ReadFile( const std::string& relativePath, void** ppOutData, size_t* outSize )
    {
        if( !m_initialized )
            return Result::FAIL;
        if( !ppOutData || !outSize )
            return Result::INVALID_ARGS;

        std::filesystem::path fullPath = ResolvePath( relativePath );

        // 1. Open File in binary mode, start at end to get size
        std::ifstream file( fullPath, std::ios::ate | std::ios::binary );
        if( !file.is_open() )
        {
            DT_ERROR( "FileSystem: Failed to open file '{0}'", fullPath.string() );
            return Result::FAIL;
        }

        // 2. Get File Size
        size_t fileSize = ( size_t )file.tellg();
        *outSize        = fileSize;

        if( fileSize == 0 )
        {
            *ppOutData = nullptr;
            return Result::SUCCESS; // Empty file is a valid read
        }

        // 3. Allocate Memory using Engine Allocator (Tracked!)
        // Accessing the internal allocator via MemorySystem
        IAllocator* allocator = m_memorySystem->GetSystemAllocator();
        void*       buffer    = allocator->Allocate( fileSize, __FILE__, __LINE__ );

        if( !buffer )
        {
            DT_ERROR( "FileSystem: Failed to allocate {0} bytes for file '{1}'", fileSize, relativePath );
            return Result::OUT_OF_MEMORY;
        }

        // 4. Read Data
        file.seekg( 0 );
        file.read( static_cast<char*>( buffer ), fileSize );
        file.close();

        *ppOutData = buffer;

        return Result::SUCCESS;
    }

    Result FileSystem::WriteFile( const std::string& relativePath, const void* pData, size_t size )
    {
        if( !m_initialized )
            return Result::FAIL;

        std::filesystem::path fullPath = m_projectRoot / relativePath;

        // Ensure directory structure exists
        std::filesystem::path dir = fullPath.parent_path();
        if( !dir.empty() && !std::filesystem::exists( dir ) )
        {
            std::error_code ec;
            std::filesystem::create_directories( dir, ec );
            if( ec )
            {
                DT_ERROR( "FileSystem: Failed to create directories for '{0}'. Error: {1}", fullPath.string(), ec.message() );
                return Result::FAIL;
            }
        }

        std::ofstream file( fullPath, std::ios::binary );
        if( !file.is_open() )
        {
            DT_ERROR( "FileSystem: Failed to open file for writing: '{0}'", fullPath.string() );
            return Result::FAIL;
        }

        file.write( static_cast<const char*>( pData ), size );
        file.close();

        return Result::SUCCESS;
    }

    void FileSystem::FreeFileBuffer( void* pData )
    {
        if( pData && m_memorySystem )
        {
            // Route deallocation through the same system that allocated it
            m_memorySystem->GetSystemAllocator()->Free( pData );
        }
    }

    bool FileSystem::FileExists( const std::string& relativePath ) const
    {
        return std::filesystem::exists( ResolvePath( relativePath ) );
    }

    std::filesystem::path FileSystem::ResolvePath( const std::string& relativePath ) const
    {
        // 1. Check Project Root (Override)
        std::filesystem::path userPath = m_projectRoot / relativePath;
        if( std::filesystem::exists( userPath ) )
        {
            return userPath;
        }

        // 2. Check Engine Assets (Fallback)
        if( !m_engineAssetsPath.empty() )
        {
            std::filesystem::path enginePath = m_engineAssetsPath / relativePath;
            if( std::filesystem::exists( enginePath ) )
            {
                return enginePath;
            }
        }

        // 3. Default: Return Project Root path (e.g., for creating new files)
        return userPath;
    }

} // namespace DigitalTwin