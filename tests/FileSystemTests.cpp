#include "core/FileSystem.h"
#include "core/Log.h"
#include "core/memory/MemorySystem.h"
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>

using namespace DigitalTwin;

class FileSystemTest : public ::testing::Test
{
protected:
    // This runs before every TEST_F
    void SetUp() override
    {
        // 1. Initialize Memory System manually
        m_memorySystem = std::make_unique<MemorySystem>();
        m_memorySystem->Initialize();

        // 2. Create FileSystem injecting the memory system
        m_fileSystem = std::make_unique<FileSystem>( m_memorySystem.get() );

        // 3. Prepare temporary directory structure for testing
        // Structure:
        // temp_test_env/
        //   ├── project_root/    (Writable, Priority 1)
        //   └── engine_assets/   (Read-Only Fallback, Priority 2)

        std::filesystem::path tempBase = std::filesystem::current_path() / "temp_test_env";
        m_projectRoot                  = tempBase / "project_root";
        m_engineAssets                 = tempBase / "engine_assets";

        // Clean up previous runs if any
        if( std::filesystem::exists( tempBase ) )
        {
            std::filesystem::remove_all( tempBase );
        }

        std::filesystem::create_directories( m_projectRoot );
        std::filesystem::create_directories( m_engineAssets );

        // 4. Initialize FileSystem with both paths
        Result res = m_fileSystem->Initialize( m_projectRoot, m_engineAssets );
        ASSERT_EQ( res, Result::SUCCESS );
    }

    // This runs after every TEST_F
    void TearDown() override
    {
        m_fileSystem->Shutdown();

        // MemorySystem shutdown will verify leaks from FileSystem operations
        m_memorySystem->Shutdown();

        // Clean up disk
        std::filesystem::path tempBase = m_projectRoot.parent_path();
        if( std::filesystem::exists( tempBase ) )
        {
            std::filesystem::remove_all( tempBase );
        }
    }

    std::unique_ptr<MemorySystem> m_memorySystem;
    std::unique_ptr<FileSystem>   m_fileSystem;
    std::filesystem::path         m_projectRoot;
    std::filesystem::path         m_engineAssets;
};

// 1. Basic Write and Read Cycle (Project Root)
TEST_F( FileSystemTest, WriteReadCycle )
{
    const std::string filename = "config.txt";
    const std::string content  = "DigitalTwin Config Data";

    // Write: Should write to m_ProjectRoot
    Result res = m_fileSystem->WriteFile( filename, content.data(), content.size() );
    EXPECT_EQ( res, Result::SUCCESS );
    EXPECT_TRUE( std::filesystem::exists( m_projectRoot / filename ) );

    // Read: Should read back from m_ProjectRoot
    void*  buffer = nullptr;
    size_t size   = 0;
    res           = m_fileSystem->ReadFile( filename, &buffer, &size );

    EXPECT_EQ( res, Result::SUCCESS );
    EXPECT_EQ( size, content.size() );
    ASSERT_NE( buffer, nullptr );

    // Verify content
    std::string readContent( static_cast<char*>( buffer ), size );
    EXPECT_EQ( readContent, content );

    // Free memory (Crucial for MemorySystem tracking)
    m_fileSystem->FreeFileBuffer( buffer );
}

// 2. Fallback Mechanism (Engine Internal Assets)
TEST_F( FileSystemTest, FallbackToInternalAssets )
{
    // Scenario: User requests "shaders/default.vert".
    // It does NOT exist in Project Root.
    // It EXISTS in Engine Assets.
    // Result: Should load from Engine Assets.

    // Manually create a file in Engine Assets using std::ofstream
    // (FileSystem::WriteFile always writes to ProjectRoot, so we cheat here for setup)
    std::filesystem::create_directories( m_engineAssets / "shaders" );
    std::ofstream file( m_engineAssets / "shaders" / "default.vert" );
    file << "#version 450 core\nvoid main(){}";
    file.close();

    // Verify FileExists logic
    EXPECT_TRUE( m_fileSystem->FileExists( "shaders/default.vert" ) );

    // Attempt to read
    void*  buffer = nullptr;
    size_t size   = 0;
    Result res    = m_fileSystem->ReadFile( "shaders/default.vert", &buffer, &size );

    EXPECT_EQ( res, Result::SUCCESS );
    EXPECT_GT( size, 0 );

    m_fileSystem->FreeFileBuffer( buffer );
}

// 3. Override Mechanism (Priority Check)
TEST_F( FileSystemTest, ProjectRootOverridesEngineAssets )
{
    // Scenario: File exists in BOTH locations.
    // Result: Should load the one from Project Root.

    const std::string filename       = "settings.ini";
    const std::string defaultContent = "Resolution=720p";
    const std::string userContent    = "Resolution=4K";

    // 1. Create Default file in Engine Assets
    {
        std::ofstream file( m_engineAssets / filename );
        file << defaultContent;
    }

    // 2. Create User Override in Project Root
    // We can use the Engine API for this one
    m_fileSystem->WriteFile( filename, userContent.data(), userContent.size() );

    // 3. Read back
    void*  buffer = nullptr;
    size_t size   = 0;
    m_fileSystem->ReadFile( filename, &buffer, &size );

    // 4. Verify we got the User Content
    std::string readContent( static_cast<char*>( buffer ), size );
    EXPECT_EQ( readContent, userContent );
    EXPECT_NE( readContent, defaultContent );

    m_fileSystem->FreeFileBuffer( buffer );
}

// 4. Verify Memory Tracking Integration
TEST_F( FileSystemTest, AllocationIsTracked )
{
    const std::string filename = "data.bin";
    const size_t      fileSize = 1024;
    std::vector<char> dummy( fileSize, 0xFF );

    m_fileSystem->WriteFile( filename, dummy.data(), fileSize );

#ifdef DT_DEBUG
    // Capture current allocation state
    size_t initialCount = m_memorySystem->GetAllocationCount();
#endif

    void*  buffer = nullptr;
    size_t size   = 0;
    m_fileSystem->ReadFile( filename, &buffer, &size );

#ifdef DT_DEBUG
    // Should have exactly one more allocation (the file buffer)
    size_t afterCount = m_memorySystem->GetAllocationCount();
    EXPECT_EQ( afterCount, initialCount + 1 );
#endif

    m_fileSystem->FreeFileBuffer( buffer );

#ifdef DT_DEBUG
    // Should be back to initial state
    EXPECT_EQ( m_memorySystem->GetAllocationCount(), initialCount );
#endif
}

// 5. Handling Non-Existent Files
TEST_F( FileSystemTest, ReadMissingFile )
{
    void*  buffer = nullptr;
    size_t size   = 0;
    Result res    = m_fileSystem->ReadFile( "ghost_file.txt", &buffer, &size );

    EXPECT_NE( res, Result::SUCCESS );
    EXPECT_EQ( buffer, nullptr );
}

// 6. Subdirectory Creation on Write
TEST_F( FileSystemTest, AutoCreateDirectories )
{
    const std::string path    = "levels/level1/data/map.dat";
    const std::string content = "DATA";

    // WriteFile should recursively create directories
    Result res = m_fileSystem->WriteFile( path, content.data(), content.size() );
    EXPECT_EQ( res, Result::SUCCESS );

    // Verify physical existence
    EXPECT_TRUE( std::filesystem::exists( m_projectRoot / "levels" / "level1" / "data" / "map.dat" ) );
}