#include "platform/Input.hpp"
#include "platform/Window.hpp"
#include <GLFW/glfw3.h> // For key codes
#include <gtest/gtest.h>

using namespace DigitalTwin;

class PlatformTest : public ::testing::Test
{
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F( PlatformTest, CreateWindow )
{
    // Skip if running in a headless CI environment without a display
    // GLFW will fail to init if no display is attached.
    // Ideally, we would detect this, but for now we assume local dev env.

    WindowConfig config;
    config.title  = "Test Window";
    config.width  = 800;
    config.height = 600;
    config.vsync  = false;

    // Window constructor initializes GLFW and creates window
    // It should assert/throw on failure, but GTest handles crashes as failures too.
    try
    {
        Window window( config );

        EXPECT_EQ( window.GetWidth(), 800 );
        EXPECT_EQ( window.GetHeight(), 600 );
        EXPECT_NE( window.GetNativeWindow(), nullptr );
        EXPECT_FALSE( window.IsClosed() );

        // Pump events once
        window.OnUpdate();
    }
    catch( ... )
    {
        // If GLFW fails (e.g. CI), we might want to skip or fail gracefully
        FAIL() << "Failed to create window. Is a display connected?";
    }
}

TEST_F( PlatformTest, InputContext )
{
    // Creating a window sets the Input context
    WindowConfig config;
    config.width  = 100;
    config.height = 100;

    Window window( config );

    // We can't physically press keys in a unit test without OS-level injection,
    // but we can check if the API call doesn't crash.
    bool pressed = Input::IsKeyPressed( GLFW_KEY_SPACE );
    EXPECT_FALSE( pressed ); // Should be false unless a ghost is pressing space

    auto [ x, y ] = Input::GetMousePosition();
    // Coordinates might be anything depending on where the cursor was
    EXPECT_GE( x, -10000.0f );
}