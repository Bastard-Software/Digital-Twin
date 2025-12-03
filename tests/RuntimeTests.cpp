#include "runtime/Engine.hpp"
#include <gtest/gtest.h>

using namespace DigitalTwin;

class RuntimeTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        if( Engine::IsInitialized() )
        {
            Engine::Shutdown();
        }
    }

    void TearDown() override
    {
        if( Engine::IsInitialized() )
        {
            Engine::Shutdown();
        }
    }
};

TEST_F( RuntimeTest, ShouldInitializeSuccessfully )
{
    // Arrange
    EngineConfig config;
    config.headless = true;

    // Act
    Engine::Init( config );

    // Assert
    EXPECT_TRUE( Engine::IsInitialized() ) << "Engine should be initialized after Init call";
    EXPECT_TRUE( Engine::IsHeadless() ) << "Engine should be in headless mode based on config";
}

TEST_F( RuntimeTest, ShouldShutdownCorrectly )
{
    // Arrange
    Engine::Init();
    EXPECT_TRUE( Engine::IsInitialized() );

    // Act
    Engine::Shutdown();

    // Assert
    EXPECT_FALSE( Engine::IsInitialized() ) << "Engine should not be initialized after Shutdown";
}

TEST_F( RuntimeTest, DefaultConfigIsHeadless )
{
    // Testowanie domyœlnych wartoœci
    EngineConfig defaultConfig;
    EXPECT_TRUE( defaultConfig.headless ) << "Default engine config should be headless=true";
}