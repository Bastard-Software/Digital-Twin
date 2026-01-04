#pragma once
#include "Core.h"
#include <filesystem>
#include <string>

namespace DigitalTwin
{

    class MemorySystem;

    class DT_API FileSystem
    {
    public:
        // Constructor requires internal MemorySystem*
        FileSystem( MemorySystem* memorySystem );
        ~FileSystem();

        /**
         * @brief Initializes the VFS.
         * @param projectRoot The main working directory (read/write).
         * @param engineAssetsPath The fallback directory for engine internal assets (read-only).
         */
        Result Initialize( const std::filesystem::path& projectRoot, const std::filesystem::path& engineAssetsPath );
        void   Shutdown();

        // --- PUBLIC API ---

        Result ReadFile( const std::string& relativePath, void** ppOutData, size_t* outSize );
        Result WriteFile( const std::string& relativePath, const void* pData, size_t size );
        void   FreeFileBuffer( void* pData );

        /**
         * @brief Checks if a file exists in Project Root OR Engine Assets.
         */
        bool FileExists( const std::string& relativePath ) const;

        /**
         * @brief Resolves relative path to absolute path using priority:
         * 1. Project Root (User override)
         * 2. Engine Assets (Default fallback)
         * Returns Project Root path if file doesn't exist anywhere (for writing).
         */
        std::filesystem::path ResolvePath( const std::string& relativePath ) const;

    private:
        std::filesystem::path m_projectRoot;      // Read/Write (Priority 1)
        std::filesystem::path m_engineAssetsPath; // Read-Only (Priority 2)
        MemorySystem*         m_memorySystem = nullptr;
        bool                  m_initialized  = false;
    };

} // namespace DigitalTwin::Core