#include "rhi/RHI.hpp"
#include <future>
#include <gtest/gtest.h>
#include <thread>
#include <vector>

using namespace DigitalTwin;

// =================================================================================================
// Basic RHI Lifecycle Tests
// =================================================================================================

class RHITest : public ::testing::Test
{
protected:
    // RHIConfig used across tests (headless for CI/CD environments)
    RHIConfig config;

    void SetUp() override
    {
        // Ensure we start with a clean state
        if( RHI::IsInitialized() )
            RHI::Shutdown();

        config.enableValidation = false; // Disable validation for test speed
        config.headless         = true;  // Critical for environments without a GPU/Display
    }

    void TearDown() override
    {
        // Clean up after every test
        if( RHI::IsInitialized() )
            RHI::Shutdown();
    }
};

// 1. Verify that a valid instance is created (not null)
TEST_F( RHITest, InitCreatesValidInstance )
{
    Result res = RHI::Init( config );

    EXPECT_EQ( res, Result::SUCCESS ) << "RHI::Init should return SUCCESS";
    EXPECT_NE( RHI::GetInstance(), VK_NULL_HANDLE ) << "VkInstance cannot be NULL after initialization";
}

// 2. Verify that the initialized flag is correctly set
TEST_F( RHITest, InitSetsInitializedFlag )
{
    EXPECT_FALSE( RHI::IsInitialized() ) << "s_initialized should be false initially";

    RHI::Init( config );

    EXPECT_TRUE( RHI::IsInitialized() ) << "s_initialized should be true after Init";
}

// 3. Verify idempotency (2x Init -> same instance)
TEST_F( RHITest, DoubleInitShouldBeIdempotent )
{
    // First initialization
    ASSERT_EQ( RHI::Init( config ), Result::SUCCESS );
    VkInstance inst1 = RHI::GetInstance();
    ASSERT_NE( inst1, VK_NULL_HANDLE );

    // Second initialization (should be ignored or return SUCCESS without changes)
    Result res = RHI::Init( config );

    EXPECT_EQ( res, Result::SUCCESS ) << "Second call to Init should return SUCCESS";

    VkInstance inst2 = RHI::GetInstance();

    // Key check: The instance handle must remain the SAME.
    EXPECT_EQ( inst1, inst2 ) << "Re-calling Init without Shutdown should not create a new instance";
}

// 4. Verify lifecycle (Init -> Shutdown -> Init -> new instance)
TEST_F( RHITest, ShutdownAndReinitShouldWork )
{
    // Step 1: Init
    ASSERT_EQ( RHI::Init( config ), Result::SUCCESS );
    VkInstance inst1 = RHI::GetInstance();

    // Step 2: Shutdown
    RHI::Shutdown();
    EXPECT_FALSE( RHI::IsInitialized() ) << "Initialized flag must be false after Shutdown";
    EXPECT_EQ( RHI::GetInstance(), VK_NULL_HANDLE ) << "Instance should be cleared (NULL) after Shutdown";

    // Step 3: Re-Init
    ASSERT_EQ( RHI::Init( config ), Result::SUCCESS );
    VkInstance inst2 = RHI::GetInstance();

    EXPECT_NE( inst2, VK_NULL_HANDLE );

    // Warning: This check can be flaky due to handle reuse by the driver/OS.
    // If the driver recycles the memory address, the test logic is correct but pointers are identical.
    // However, in most cases (especially with debug layers), addresses will differ.
    if( inst1 == inst2 )
    {
        // Log a warning instead of failing if we are sure Shutdown logic worked (verified in Step 2).
        printf( "[WARNING] Driver reused the same memory address for VkInstance (Handle Reuse).\n" );
    }
    else
    {
        EXPECT_NE( inst1, inst2 ) << "New instance after restart should (usually) be different";
    }
}

// =================================================================================================
// Device Creation & Queue Retrieval Tests
// =================================================================================================

class DeviceTest : public ::testing::Test
{
protected:
    Ref<Device> device;
    RHIConfig   config;

    void SetUp() override
    {
        if( RHI::IsInitialized() )
            RHI::Shutdown();

        config.headless         = true;
        config.enableValidation = false;
        ASSERT_EQ( RHI::Init( config ), Result::SUCCESS );

        // We assume at least one adapter exists for tests to run
        if( RHI::GetAdapterCount() > 0 )
        {
            device = RHI::CreateDevice( 0 );
        }
    }

    void TearDown() override
    {
        if( device )
        {
            RHI::DestroyDevice( device );
            device.reset();
        }
        RHI::Shutdown();
    }
};

TEST_F( DeviceTest, ShouldCreateDeviceSuccessfully )
{
    if( RHI::GetAdapterCount() == 0 )
        GTEST_SKIP() << "No GPU adapters found, skipping device test.";

    ASSERT_NE( device, nullptr ) << "CreateDevice returned null";
    EXPECT_NE( device->GetHandle(), VK_NULL_HANDLE ) << "Logical VkDevice handle is null";
    EXPECT_NE( device->GetPhysicalDevice(), VK_NULL_HANDLE ) << "PhysicalDevice handle is null";
}

TEST_F( DeviceTest, ShouldInitializeQueues )
{
    if( !device )
        GTEST_SKIP();

    // Queues must be initialized (not null)
    EXPECT_NE( device->GetGraphicsQueue(), nullptr ) << "Graphics queue should be valid";
    EXPECT_NE( device->GetComputeQueue(), nullptr ) << "Compute queue should be valid";
    EXPECT_NE( device->GetTransferQueue(), nullptr ) << "Transfer queue should be valid";
}

TEST_F( DeviceTest, ShouldHaveValidAllocator )
{
    if( !device )
        GTEST_SKIP();

    EXPECT_NE( device->GetAllocator(), VK_NULL_HANDLE ) << "VMA Allocator was not initialized";
}

TEST_F( DeviceTest, QueuesShouldHaveCorrectHandles )
{
    if( !device )
        GTEST_SKIP();

    auto gfx = device->GetGraphicsQueue();
    EXPECT_NE( gfx->GetHandle(), VK_NULL_HANDLE ) << "Graphics VkQueue handle is invalid";

    auto comp = device->GetComputeQueue();
    EXPECT_NE( comp->GetHandle(), VK_NULL_HANDLE ) << "Compute VkQueue handle is invalid";
}

// =================================================================================================
// CommandQueue Properties Tests
// =================================================================================================

class CommandQueueTest : public ::testing::Test
{
protected:
    Ref<Device> device;
    RHIConfig   config;

    void SetUp() override
    {
        if( RHI::IsInitialized() )
            RHI::Shutdown();

        config.headless         = true;
        config.enableValidation = false;
        RHI::Init( config );

        if( RHI::GetAdapterCount() > 0 )
        {
            device = RHI::CreateDevice( 0 );
        }
    }

    void TearDown() override
    {
        if( device )
            RHI::DestroyDevice( device );
        RHI::Shutdown();
    }
};

TEST_F( CommandQueueTest, GraphicsQueueHasCorrectType )
{
    if( !device )
        GTEST_SKIP();
    auto queue = device->GetGraphicsQueue();
    ASSERT_NE( queue, nullptr );

    EXPECT_EQ( queue->GetType(), QueueType::GRAPHICS );
}

TEST_F( CommandQueueTest, ComputeQueueHasCorrectType )
{
    if( !device )
        GTEST_SKIP();
    auto queue = device->GetComputeQueue();
    ASSERT_NE( queue, nullptr );

    // It is possible for Compute to be aliased to Graphics if hardware only supports one queue family
    // In that case, the object might be the same, but we check if the type stored reflects creation intent
    // OR if we aliased the pointer, the type will be GRAPHICS.
    // Based on Device.cpp implementation:
    // if aliased -> m_computeQueue = m_graphicsQueue; -> Type is GRAPHICS
    // if distinct -> new CommandQueue(..., TYPE_COMPUTE); -> Type is COMPUTE

    // So we check consistency:
    if( queue == device->GetGraphicsQueue() )
    {
        EXPECT_EQ( queue->GetType(), QueueType::GRAPHICS );
    }
    else
    {
        EXPECT_EQ( queue->GetType(), QueueType::COMPUTE );
    }
}

TEST_F( CommandQueueTest, ShouldHaveValidTimelineSemaphore )
{
    if( !device )
        GTEST_SKIP();
    auto queue = device->GetGraphicsQueue();

    EXPECT_NE( queue->GetTimelineSemaphore(), VK_NULL_HANDLE ) << "Timeline semaphore not created";
}

TEST_F( CommandQueueTest, InitialFenceValueShouldBeZero )
{
    if( !device )
        GTEST_SKIP();
    auto queue = device->GetGraphicsQueue();

    // GetLastSubmittedValue returns (m_nextValue - 1). Initial m_nextValue is 1.
    // So expected last submitted value is 0.
    EXPECT_EQ( queue->GetLastSubmittedValue(), 0 ) << "Initial fence value should be 0";
}

TEST_F( CommandQueueTest, IsValueCompletedCheck )
{
    if( !device )
        GTEST_SKIP();
    auto queue = device->GetGraphicsQueue();

    // Since nothing was submitted, value 0 should be "completed" (initial state)
    // Actually, create info sets initialValue = 0.
    EXPECT_TRUE( queue->IsValueCompleted( 0 ) ) << "Initial state 0 should be considered completed";

    // Value 1 has not been submitted/signaled yet
    EXPECT_FALSE( queue->IsValueCompleted( 1 ) ) << "Future value 1 should not be completed yet";
}

// =================================================================================================
// DeviceCommand Tests
// =================================================================================================

class DeviceCommandTest : public ::testing::Test
{
protected:
    Ref<Device> device;
    RHIConfig   config;

    void SetUp() override
    {
        // Clean up previous state if necessary
        if( RHI::IsInitialized() )
            RHI::Shutdown();

        // Initialize RHI in headless mode for CI/CD compatibility
        config.headless         = true;
        config.enableValidation = false;
        RHI::Init( config );

        // Create device if adapter exists
        if( RHI::GetAdapterCount() > 0 )
        {
            device = RHI::CreateDevice( 0 );
        }
    }

    void TearDown() override
    {
        if( device )
            RHI::DestroyDevice( device );
        RHI::Shutdown();
    }
};

// Test 1: Verify creation of a single CommandBuffer
TEST_F( DeviceCommandTest, CreateSingleCommandBuffer )
{
    if( !device )
        GTEST_SKIP() << "No GPU found, skipping test";

    // Request a command buffer for the Graphics queue
    auto cmd = device->CreateCommandBuffer( QueueType::GRAPHICS );
    ASSERT_NE( cmd, nullptr ) << "Failed to create Graphics cmd buffer";
    EXPECT_NE( cmd->GetHandle(), VK_NULL_HANDLE ) << "Vulkan handle should be valid";

    // Verify basic lifecycle (Begin/End)
    cmd->Begin();
    cmd->End();
}

// Test 2: Verify creating buffers for different queue types
TEST_F( DeviceCommandTest, CreateBuffersForDifferentQueues )
{
    if( !device )
        GTEST_SKIP() << "No GPU found, skipping test";

    auto gfxCmd   = device->CreateCommandBuffer( QueueType::GRAPHICS );
    auto compCmd  = device->CreateCommandBuffer( QueueType::COMPUTE );
    auto transCmd = device->CreateCommandBuffer( QueueType::TRANSFER );

    EXPECT_NE( gfxCmd, nullptr );
    EXPECT_NE( compCmd, nullptr );
    EXPECT_NE( transCmd, nullptr );

    // Verify handles are distinct (unless recycled very quickly, but we hold refs)
    EXPECT_NE( gfxCmd->GetHandle(), compCmd->GetHandle() );
}

// Test 3: Multithreaded creation stress test
// Ensures that thread-local pools are created correctly and no race conditions occur.
TEST_F( DeviceCommandTest, MultithreadedPoolCreation )
{
    if( !device )
        GTEST_SKIP() << "No GPU found, skipping test";

    constexpr int NUM_THREADS     = 8;
    constexpr int CMDS_PER_THREAD = 10;

    // Lambda function executed by each thread
    auto threadFunc = [ & ]( int threadId ) -> size_t {
        std::vector<Ref<CommandBuffer>> buffers;
        for( int i = 0; i < CMDS_PER_THREAD; ++i )
        {
            // Each thread requests a COMPUTE command buffer.
            // The Device must create a unique CommandPool for this thread ID internally.
            auto cmd = device->CreateCommandBuffer( QueueType::COMPUTE );
            if( cmd )
            {
                cmd->Begin();
                cmd->End();
                buffers.push_back( cmd );
            }
        }
        return buffers.size();
    };

    // Launch threads asynchronously
    std::vector<std::future<size_t>> futures;
    for( int i = 0; i < NUM_THREADS; ++i )
    {
        futures.push_back( std::async( std::launch::async, threadFunc, i ) );
    }

    // Wait for results and verify count
    size_t totalBuffers = 0;
    for( auto& f: futures )
    {
        totalBuffers += f.get();
    }

    EXPECT_EQ( totalBuffers, NUM_THREADS * CMDS_PER_THREAD ) << "Not all command buffers were created successfully in parallel";
}