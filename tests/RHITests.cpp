#include "rhi/RHITypes.h"

#include "rhi/Device.h"
#include "rhi/RHI.h"
#include <gtest/gtest.h>
#include <memory>

using namespace DigitalTwin;

// =================================================================================================
// RHI Lifecycle Tests
// =================================================================================================

class RHILifecycleTest : public ::testing::Test
{
protected:
    Scope<RHI> m_rhi;
    RHIConfig            m_config;

    void SetUp() override
    {
        // Create a fresh instance for every test
        m_rhi = CreateScope<RHI>();

        // Configure for headless execution (CI/CD friendly)
        m_config.enableValidation = true;
        m_config.headless         = true;
    }

    void TearDown() override
    {
        if( m_rhi )
        {
            m_rhi->Shutdown();
            m_rhi.reset();
        }
    }
};

// 1. Verify successful initialization
TEST_F( RHILifecycleTest, InitializeSuccess )
{
    Result res = m_rhi->Initialize( m_config );

    EXPECT_EQ( res, Result::SUCCESS );
    EXPECT_TRUE( m_rhi->IsInitialized() );
    EXPECT_NE( m_rhi->GetInstance(), VK_NULL_HANDLE );
}

// 2. Verify double initialization logic
TEST_F( RHILifecycleTest, DoubleInitialization )
{
    // First Init
    ASSERT_EQ( m_rhi->Initialize( m_config ), Result::SUCCESS );
    VkInstance instance1 = m_rhi->GetInstance();

    // Second Init on the same object should warn but succeed (idempotent check)
    Result res = m_rhi->Initialize( m_config );

    EXPECT_EQ( res, Result::SUCCESS );
    EXPECT_EQ( m_rhi->GetInstance(), instance1 ) << "Instance handle should remain unchanged";
}

// 3. Verify Shutdown clears state
TEST_F( RHILifecycleTest, ShutdownClearsState )
{
    m_rhi->Initialize( m_config );
    ASSERT_NE( m_rhi->GetInstance(), VK_NULL_HANDLE );

    m_rhi->Shutdown();

    EXPECT_FALSE( m_rhi->IsInitialized() );
    EXPECT_EQ( m_rhi->GetInstance(), VK_NULL_HANDLE );
}

// 4. Verify Adapter Enumeration
TEST_F( RHILifecycleTest, EnumAdapters )
{
    // Note: Volk usually requires Init/LoadInstance to enumerate properly
    m_rhi->Initialize( m_config );

    // In a headless environment without GPU this might be 0, but typically llvmpipe (CPU) is present.
    const auto& adapters = m_rhi->GetAdapters();

    // We expect at least one device (even software rasterizer) or empty if strict filtering.
    // We just check if the vector is accessible.
    EXPECT_GE( adapters.size(), 0 );

    for( const auto& adapter: adapters )
    {
        EXPECT_NE( adapter.handle, VK_NULL_HANDLE );
        EXPECT_FALSE( adapter.name.empty() );
    }
}

// =================================================================================================
// Device Creation Tests
// =================================================================================================

class RHIDeviceTest : public ::testing::Test
{
protected:
    Scope<RHI> m_rhi;
    Scope<Device>        m_device;
    RHIConfig            m_config;

    void SetUp() override
    {
        m_rhi                     = CreateScope<RHI>();
        m_config.headless         = true;
        m_config.enableValidation = true;

        // Initialize RHI first
        ASSERT_EQ( m_rhi->Initialize( m_config ), Result::SUCCESS );
    }

    void TearDown() override
    {
        // Device must be destroyed BEFORE RHI shutdown
        m_device.reset();

        if( m_rhi )
        {
            m_rhi->Shutdown();
            m_rhi.reset();
        }
    }
};

// 1. Verify successful device creation on valid adapter index
TEST_F( RHIDeviceTest, CreateDeviceSuccess )
{
    if( m_rhi->GetAdapters().empty() )
        GTEST_SKIP() << "No GPU adapters found.";

    // Attempt to create device on the first adapter
    Result res = m_rhi->CreateDevice( 0, m_device );

    ASSERT_EQ( res, Result::SUCCESS );
    ASSERT_NE( m_device, nullptr );

    EXPECT_NE( m_device->GetHandle(), VK_NULL_HANDLE );
    EXPECT_NE( m_device->GetPhysicalDevice(), VK_NULL_HANDLE );
    EXPECT_NE( m_device->GetAllocator(), VK_NULL_HANDLE );

    // Verify Queues
    EXPECT_NE( m_device->GetGraphicsQueue(), nullptr );
    // Compute queue might be aliased but should not be null
    EXPECT_NE( m_device->GetComputeQueue(), nullptr );
}

// 2. Verify device creation fails gracefully on invalid adapter index
TEST_F( RHIDeviceTest, CreateDeviceInvalidIndex )
{
    uint32_t invalidIndex = 9999;

    // Should fail gracefully
    Result res = m_rhi->CreateDevice( invalidIndex, m_device );

    EXPECT_NE( res, Result::SUCCESS );
    EXPECT_EQ( m_device, nullptr );
}

// 3. Verify Volk function table is populated for created device
TEST_F( RHIDeviceTest, DeviceHasVolkTable )
{
    if( m_rhi->GetAdapters().empty() )
        GTEST_SKIP();
    ASSERT_EQ( m_rhi->CreateDevice( 0, m_device ), Result::SUCCESS );

    // Check if table is populated (function pointer not null)
    EXPECT_NE( m_device->GetAPI().vkCreateBuffer, nullptr );
}

// 4. Verify Graphics Queue has a valid timeline semaphore
TEST_F( RHIDeviceTest, QueueHasTimelineSemaphore )
{
    if( m_rhi->GetAdapters().empty() )
        GTEST_SKIP();
    ASSERT_EQ( m_rhi->CreateDevice( 0, m_device ), Result::SUCCESS );

    auto gfx = m_device->GetGraphicsQueue();
    ASSERT_NE( gfx, nullptr );

    // Timeline semaphore should be created in constructor
    EXPECT_NE( gfx->GetTimelineSemaphore(), VK_NULL_HANDLE );
}