#pragma once
#include "core/Base.hpp"
#include <filesystem>
#include <string>

namespace DigitalTwin
{
    /**
     * @brief Centralized system for resolving file paths.
     * Handles the difference between "Development Mode" (absolute paths to source)
     * and "Release Mode" (relative paths to executable).
     */
    class FileSystem
    {
    public:
        /**
         * @brief Initializes the file system.
         * Determines the root directory based on CMake definitions.
         */
        static void Init();

        /**
         * @brief Resolves a relative path to a full, absolute path.
         * @param path The relative path (e.g., "shaders/graphics/cell.vert")
         * @return The absolute path on disk.
         */
        static std::filesystem::path GetPath( const std::string& path );

        /**
         * @brief Returns the root assets directory.
         */
        static const std::filesystem::path& GetRoot();

    private:
        static std::filesystem::path s_RootDirectory;
    };
} // namespace DigitalTwin