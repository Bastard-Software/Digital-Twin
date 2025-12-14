#include "core/FileSystem.hpp"

#include "core/Log.hpp"

namespace DigitalTwin
{
    // Define the static member
    std::filesystem::path FileSystem::s_RootDirectory;

    void FileSystem::Init()
    {
        if( !s_RootDirectory.empty() )
        {
            return;
        }

        // 1. Determine the Root Directory
#if defined( DT_ASSET_DIR )
        // Development Mode: CMake passed the source directory path
        s_RootDirectory = std::filesystem::path( DT_ASSET_DIR );
        DT_CORE_INFO( "FileSystem: Running in DEV mode. Root: '{}'", s_RootDirectory.string() );
#else
        // Release Mode: Assets are expected to be next to the executable
        s_RootDirectory = std::filesystem::current_path() / "assets";
        DT_CORE_INFO( "FileSystem: Running in RELEASE mode. Root: '{}'", s_RootDirectory.string() );
#endif

        // 2. Validate existence
        if( !std::filesystem::exists( s_RootDirectory ) )
        {
            DT_CORE_CRITICAL( "FileSystem: Assets directory does not exist at: {}", s_RootDirectory.string() );
            // In a real engine, we might want to throw or crash here
        }
    }

    std::filesystem::path FileSystem::GetPath( const std::string& path )
    {
        // Operator / in std::filesystem automatically handles separator slashes
        return s_RootDirectory / path;
    }

    const std::filesystem::path& FileSystem::GetRoot()
    {
        return s_RootDirectory;
    }

} // namespace DigitalTwin