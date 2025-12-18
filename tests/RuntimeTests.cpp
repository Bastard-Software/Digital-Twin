#include "rhi/RHI.hpp"
#include "runtime/Engine.hpp"
#include "runtime/Application.hpp"
#include <gtest/gtest.h>

using namespace DigitalTwin;

class RuntimeTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Ensure clean state before each test
        // Engine::Init calls RHI::Init, so we must ensure RHI is down.
        if( RHI::IsInitialized() )
        {
            RHI::Shutdown();
        }
    }

    void TearDown() override
    {
        // Cleanup global RHI if the test left it initialized
        if( RHI::IsInitialized() )
        {
            RHI::Shutdown();
        }
    }
};

TEST_F( RuntimeTest, ShouldInitializeSuccessfully )
{
    // Arrange
    EngineConfig config;
    config.headless = true;
    Engine engine;

    // Act
    Result res = engine.Init( config );

    // Assert
    EXPECT_EQ( res, Result::SUCCESS );
    EXPECT_TRUE( engine.IsInitialized() ) << "Engine should be initialized after Init call";
    EXPECT_TRUE( engine.IsHeadless() ) << "Engine should be in headless mode based on config";
}

TEST_F( RuntimeTest, ShouldShutdownCorrectly )
{
    // Arrange
    Engine engine;
    engine.Init(); // Default config is headless=true
    EXPECT_TRUE( engine.IsInitialized() );

    // Act
    engine.Shutdown();

    // Assert
    EXPECT_FALSE( engine.IsInitialized() ) << "Engine should not be initialized after Shutdown";
}

TEST_F( RuntimeTest, DefaultConfigIsHeadless )
{
    EngineConfig defaultConfig;
    EXPECT_TRUE( defaultConfig.headless ) << "Default engine config should be headless=true";
}