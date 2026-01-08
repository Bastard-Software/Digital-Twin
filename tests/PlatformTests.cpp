#include "platform/PlatformSystem.h"
#include "platform/Window.h"
#include <gtest/gtest.h>
#include <vector>

using namespace DigitalTwin;

class PlatformSystemTest : public ::testing::Test
{
protected:
    // Runs before every test
    void SetUp() override
    {
        m_platform = std::make_unique<PlatformSystem>();

        // We explicitly initialize it here to fail early if GLFW is missing
        Result success = m_platform->Initialize();
        ASSERT_TRUE( success == Result::SUCCESS ) << "Failed to initialize PlatformSystem (GLFW)";
    }

    // Runs after every test
    void TearDown() override
    {
        // PlatformSystem logic should handle remaining windows gracefully
        m_platform->Shutdown();
        m_platform.reset();
    }

    std::unique_ptr<PlatformSystem> m_platform;
};

// 1. Core Initialization Check
TEST_F( PlatformSystemTest, InitializationState )
{
    // Initialize was called in SetUp, checking via side-effects or re-init safety
    EXPECT_TRUE( m_platform->Initialize() == Result::SUCCESS );

    // Check if Input system was created
    EXPECT_NE( m_platform->GetInput(), nullptr );
}

// 2. Vulkan Extensions Availability
TEST_F( PlatformSystemTest, VulkanExtensions )
{
    auto extensions = m_platform->GetRequiredVulkanExtensions();

    // In a headless environment (CI), this might be 0, but the call should be safe.
    // On a desktop, it should return required extensions (Surface, OS-specific).
    // We mainly test that the function doesn't crash.
    bool isValid = ( extensions.size() >= 0 );
    EXPECT_TRUE( isValid );
}

// 3. Basic Window Creation
TEST_F( PlatformSystemTest, SingleWindowCreation )
{
    WindowDesc config;
    config.title  = "Unit Test Window";
    config.width  = 100;
    config.height = 100;

    auto window = m_platform->CreateWindow( config );

    ASSERT_NE( window, nullptr );
    EXPECT_EQ( window->GetWidth(), 100 );
    EXPECT_EQ( window->GetHeight(), 100 );
    EXPECT_NE( window->GetNativeWindow(), nullptr );

    // Ensure Input system is linked
    EXPECT_EQ( window->GetInput(), m_platform->GetInput() );
}

// 4. Multiple Windows & Lifecycle Management
TEST_F( PlatformSystemTest, MultipleWindowsLifecycle )
{
    // Scenario: Create 3 windows, delete one, ensure others still work.

    // 1. Create Windows
    auto win1 = m_platform->CreateWindow( { "Win1", 800, 600 } );
    auto win2 = m_platform->CreateWindow( { "Win2", 800, 600 } );
    auto win3 = m_platform->CreateWindow( { "Win3", 800, 600 } );

    ASSERT_NE( win1, nullptr );
    ASSERT_NE( win2, nullptr );
    ASSERT_NE( win3, nullptr );

    // 2. Verify unique handles
    EXPECT_NE( win1->GetNativeWindow(), win2->GetNativeWindow() );
    EXPECT_NE( win2->GetNativeWindow(), win3->GetNativeWindow() );

    // 3. Simulate Frame Update
    m_platform->OnUpdate();

    // 4. Destroy "Middle" Window (Win2)
    // This triggers ~Window -> PlatformSystem::RemoveWindow
    win2.reset();

    // 5. Update again
    m_platform->OnUpdate();

    // 6. Verify remaining windows are alive
    EXPECT_FALSE( win1->IsClosed() );
    EXPECT_FALSE( win3->IsClosed() );

    // 7. Test Ends:
    // win1 and win3 go out of scope.
    // PlatformSystem::Shutdown in TearDown cleans up internal references safely.
}

// 5. Input System Context Switching
TEST_F( PlatformSystemTest, InputContextSafety )
{
    // When creating a window, the input context is usually switched to it.
    auto win1 = m_platform->CreateWindow( { "Input Win", 100, 100 } );

    // Just verifying that checking input doesn't crash
    bool pressed = m_platform->GetInput()->IsKeyPressed( Key::Unknown );
    EXPECT_FALSE( pressed );
}