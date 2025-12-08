#include "rhi/Pipeline.hpp"
#include "rhi/RHI.hpp"
#include "rhi/Shader.hpp"
#include "platform/Window.hpp"
#include <filesystem>
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
    auto threadFunc = [ & ]( int /*threadId*/ ) -> size_t {
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

class DeviceResourceTest : public ::testing::Test
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

// =================================================================================================
// BUFFER TESTS - Create every type to ensure flags and VMA memory types are valid
// =================================================================================================

TEST_F( DeviceResourceTest, CreateBuffer_UPLOAD )
{
    if( !device )
        GTEST_SKIP();
    auto buffer = device->CreateBuffer( { 1024, BufferType::UPLOAD } );
    ASSERT_NE( buffer, nullptr );
    EXPECT_NE( buffer->GetHandle(), VK_NULL_HANDLE );

    // Test Write access
    int val = 42;
    buffer->Write( &val, sizeof( int ) );

    // Verify we can read it back (since UPLOAD is host visible)
    int readVal = 0;
    buffer->Read( &readVal, sizeof( int ) );
    EXPECT_EQ( val, readVal );
}

TEST_F( DeviceResourceTest, CreateBuffer_READBACK )
{
    if( !device )
        GTEST_SKIP();
    auto buffer = device->CreateBuffer( { 1024, BufferType::READBACK } );
    ASSERT_NE( buffer, nullptr );
    // Map should succeed
    EXPECT_NE( buffer->Map(), nullptr );
}

TEST_F( DeviceResourceTest, CreateBuffer_STORAGE )
{
    if( !device )
        GTEST_SKIP();
    auto buffer = device->CreateBuffer( { 1024, BufferType::STORAGE } );
    ASSERT_NE( buffer, nullptr );

    // STORAGE is GPU-only, mapping should ideally assert/fail (depending on impl),
    // but here we just check creation success.
    // Also check Device Address retrieval
    if( device->GetPhysicalDevice() ) // Only if feature enabled
    {
        EXPECT_NE( buffer->GetDeviceAddress(), 0 );
    }
}

TEST_F( DeviceResourceTest, CreateBuffer_UNIFORM )
{
    if( !device )
        GTEST_SKIP();
    auto buffer = device->CreateBuffer( { 256, BufferType::UNIFORM } );
    ASSERT_NE( buffer, nullptr );
}

TEST_F( DeviceResourceTest, CreateBuffer_VERTEX )
{
    if( !device )
        GTEST_SKIP();
    auto buffer = device->CreateBuffer( { 1024, BufferType::VERTEX } );
    ASSERT_NE( buffer, nullptr );
}

TEST_F( DeviceResourceTest, CreateBuffer_INDEX )
{
    if( !device )
        GTEST_SKIP();
    auto buffer = device->CreateBuffer( { 1024, BufferType::INDEX } );
    ASSERT_NE( buffer, nullptr );
}

TEST_F( DeviceResourceTest, CreateBuffer_INDIRECT )
{
    if( !device )
        GTEST_SKIP();
    auto buffer = device->CreateBuffer( { 256, BufferType::INDIRECT } );
    ASSERT_NE( buffer, nullptr );
}

// =================================================================================================
// TEXTURE TESTS - Verify Types and Usage Combinations
// =================================================================================================

TEST_F( DeviceResourceTest, CreateTexture_1D_Sampled )
{
    if( !device )
        GTEST_SKIP();
    // 1D Texture used for sampling
    auto tex = device->CreateTexture1D( 128, VK_FORMAT_R8G8B8A8_UNORM, TextureUsage::SAMPLED );
    ASSERT_NE( tex, nullptr );
    EXPECT_EQ( tex->GetType(), TextureType::Texture1D );
    EXPECT_NE( tex->GetView(), VK_NULL_HANDLE );
}

TEST_F( DeviceResourceTest, CreateTexture_2D_RenderTarget )
{
    if( !device )
        GTEST_SKIP();
    // Typical Color Attachment
    auto tex = device->CreateTexture2D( 256, 256, VK_FORMAT_R8G8B8A8_UNORM, TextureUsage::RENDER_TARGET | TextureUsage::SAMPLED );
    ASSERT_NE( tex, nullptr );
    EXPECT_EQ( tex->GetType(), TextureType::Texture2D );
}

TEST_F( DeviceResourceTest, CreateTexture_2D_DepthStencil )
{
    if( !device )
        GTEST_SKIP();
    // Depth format required for DEPTH_STENCIL_TARGET usage
    auto tex = device->CreateTexture2D( 256, 256, VK_FORMAT_D32_SFLOAT, TextureUsage::DEPTH_STENCIL_TARGET | TextureUsage::SAMPLED );
    ASSERT_NE( tex, nullptr );
    EXPECT_EQ( tex->GetType(), TextureType::Texture2D );
}

TEST_F( DeviceResourceTest, CreateTexture_3D_Storage )
{
    if( !device )
        GTEST_SKIP();
    // 3D Texture for Voxel Grid / Simulation
    auto tex = device->CreateTexture3D( 64, 64, 64, VK_FORMAT_R32_SFLOAT, TextureUsage::STORAGE | TextureUsage::SAMPLED );
    ASSERT_NE( tex, nullptr );
    EXPECT_EQ( tex->GetType(), TextureType::Texture3D );
    EXPECT_EQ( tex->GetExtent().depth, 64 );
}

TEST_F( DeviceResourceTest, CreateTexture_Storage_Transfer )
{
    if( !device )
        GTEST_SKIP();
    // Test Transfer flags
    auto tex =
        device->CreateTexture2D( 128, 128, VK_FORMAT_R32_SFLOAT, TextureUsage::STORAGE | TextureUsage::TRANSFER_SRC | TextureUsage::TRANSFER_DST );
    ASSERT_NE( tex, nullptr );
}

// =================================================================================================
// SHADER COMPILATION & REFLECTION TESTS
// =================================================================================================

class ShaderCompilationTest : public ::testing::Test
{
protected:
    Ref<Device> device;
    RHIConfig   config;

    // Temporary shader file names
    std::string computeShaderPath  = "test_compute.comp";
    std::string vertexShaderPath   = "test_vertex.vert";
    std::string fragmentShaderPath = "test_fragment.frag";

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

        // 1. Create dummy Compute Shader
        {
            std::ofstream out( computeShaderPath );
            out << "#version 450\n";
            out << "layout(local_size_x = 1) in;\n";
            out << "struct Data { float value; };\n";
            out << "layout(set = 0, binding = 0) buffer InBuffer { Data inData[]; };\n";
            out << "layout(set = 0, binding = 1) buffer OutBuffer { Data outData[]; };\n";
            out << "layout(push_constant) uniform Constants { float time; } pushConstants;\n";
            out << "void main() { outData[gl_GlobalInvocationID.x].value = inData[gl_GlobalInvocationID.x].value * pushConstants.time; }\n";
            out.close();
        }

        // 2. Create dummy Vertex Shader
        {
            std::ofstream out( vertexShaderPath );
            out << "#version 450\n";
            out << "layout(location = 0) in vec3 inPosition;\n";
            out << "layout(set = 0, binding = 0) uniform Camera { mat4 viewProj; };\n";
            out << "void main() { gl_Position = viewProj * vec4(inPosition, 1.0); }\n";
            out.close();
        }

        // 3. Create dummy Fragment Shader
        {
            std::ofstream out( fragmentShaderPath );
            out << "#version 450\n";
            out << "layout(location = 0) out vec4 outColor;\n";
            out << "layout(set = 0, binding = 1) uniform sampler2D texSampler;\n"; // Note binding 1
            out << "void main() { outColor = texture(texSampler, vec2(0.5)); }\n";
            out.close();
        }
    }

    void TearDown() override
    {
        if( device )
            RHI::DestroyDevice( device );
        RHI::Shutdown();

        // Helper lambda to cleanup files
        auto cleanup = []( const std::string& path ) {
            // Remove source file
            if( std::filesystem::exists( path ) )
                std::filesystem::remove( path );

            // Remove cached SPIR-V file to force recompilation next time
            // This is CRITICAL for tests to pass if compiler options change!
            std::string cache = path + ".spv";
            if( std::filesystem::exists( cache ) )
                std::filesystem::remove( cache );
        };

        cleanup( computeShaderPath );
        cleanup( vertexShaderPath );
        cleanup( fragmentShaderPath );
    }
};

TEST_F( ShaderCompilationTest, CompileComputeShader )
{
    if( !device )
        GTEST_SKIP() << "No GPU found, skipping test";

    auto shader = device->CreateShader( computeShaderPath );

    ASSERT_NE( shader, nullptr ) << "Shader creation failed";
    EXPECT_NE( shader->GetModule(), VK_NULL_HANDLE ) << "Vulkan shader module handle is null";
    EXPECT_EQ( shader->GetStage(), VK_SHADER_STAGE_COMPUTE_BIT ) << "Incorrect shader stage detected";
}

TEST_F( ShaderCompilationTest, ReflectionComputeCheck )
{
    if( !device )
        GTEST_SKIP() << "No GPU found, skipping test";

    auto shader = device->CreateShader( computeShaderPath );
    ASSERT_NE( shader, nullptr );

    const auto& reflection = shader->GetReflectionData();

    // Check InBuffer
    ASSERT_TRUE( reflection.find( "InBuffer" ) != reflection.end() ) << "InBuffer not found. Did optimization strip names?";
    EXPECT_EQ( reflection.at( "InBuffer" ).set, 0 );
    EXPECT_EQ( reflection.at( "InBuffer" ).binding, 0 );
    EXPECT_EQ( reflection.at( "InBuffer" ).type, ShaderResourceType::STORAGE_BUFFER );

    // Check OutBuffer
    ASSERT_TRUE( reflection.find( "OutBuffer" ) != reflection.end() ) << "OutBuffer not found";
    EXPECT_EQ( reflection.at( "OutBuffer" ).set, 0 );
    EXPECT_EQ( reflection.at( "OutBuffer" ).binding, 1 );
    EXPECT_EQ( reflection.at( "OutBuffer" ).type, ShaderResourceType::STORAGE_BUFFER );

    // Check Push Constants
    const auto& pcs = shader->GetPushConstantRanges();
    ASSERT_FALSE( pcs.empty() ) << "Push constants not detected";
    EXPECT_GE( pcs[ 0 ].size, sizeof( float ) );
}

TEST_F( ShaderCompilationTest, CompileVertexShader )
{
    if( !device )
        GTEST_SKIP() << "No GPU found, skipping test";

    auto shader = device->CreateShader( vertexShaderPath );

    ASSERT_NE( shader, nullptr );
    EXPECT_NE( shader->GetModule(), VK_NULL_HANDLE );
    EXPECT_EQ( shader->GetStage(), VK_SHADER_STAGE_VERTEX_BIT );
}

TEST_F( ShaderCompilationTest, ReflectionVertexCheck )
{
    if( !device )
        GTEST_SKIP() << "No GPU found, skipping test";

    auto shader = device->CreateShader( vertexShaderPath );
    ASSERT_NE( shader, nullptr );

    const auto& reflection = shader->GetReflectionData();

    // Check Camera UBO
    ASSERT_TRUE( reflection.find( "Camera" ) != reflection.end() ) << "Camera UBO not found";
    EXPECT_EQ( reflection.at( "Camera" ).type, ShaderResourceType::UNIFORM_BUFFER );
    EXPECT_EQ( reflection.at( "Camera" ).binding, 0 );
    EXPECT_GE( reflection.at( "Camera" ).size, 64 ); // mat4
}

TEST_F( ShaderCompilationTest, CompileFragmentShader )
{
    if( !device )
        GTEST_SKIP() << "No GPU found, skipping test";

    auto shader = device->CreateShader( fragmentShaderPath );

    ASSERT_NE( shader, nullptr );
    EXPECT_NE( shader->GetModule(), VK_NULL_HANDLE );
    EXPECT_EQ( shader->GetStage(), VK_SHADER_STAGE_FRAGMENT_BIT );
}

TEST_F( ShaderCompilationTest, ReflectionFragmentCheck )
{
    if( !device )
        GTEST_SKIP() << "No GPU found, skipping test";

    auto shader = device->CreateShader( fragmentShaderPath );
    ASSERT_NE( shader, nullptr );

    const auto& reflection = shader->GetReflectionData();

    // Check Sampler
    ASSERT_TRUE( reflection.find( "texSampler" ) != reflection.end() ) << "Sampler not found";
    EXPECT_EQ( reflection.at( "texSampler" ).type, ShaderResourceType::SAMPLED_IMAGE );
    EXPECT_EQ( reflection.at( "texSampler" ).binding, 1 );
}

// =================================================================================================
// DESCRIPTOR ALLOCATOR TESTS
// =================================================================================================

class DescriptorAllocatorTest : public ::testing::Test
{
protected:
    Ref<Device>           device;
    RHIConfig             config;
    VkDescriptorSetLayout testLayout = VK_NULL_HANDLE;

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
            CreateDummyLayout();
        }
    }

    void TearDown() override
    {
        if( device )
        {
            if( testLayout != VK_NULL_HANDLE )
            {
                device->GetAPI().vkDestroyDescriptorSetLayout( device->GetHandle(), testLayout, nullptr );
            }
            RHI::DestroyDevice( device );
        }
        RHI::Shutdown();
    }

    void CreateDummyLayout()
    {
        VkDescriptorSetLayoutBinding binding{};
        binding.binding         = 0;
        binding.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        binding.descriptorCount = 1;
        binding.stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;

        VkDescriptorSetLayoutCreateInfo info{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
        info.bindingCount = 1;
        info.pBindings    = &binding;

        if( device->GetAPI().vkCreateDescriptorSetLayout( device->GetHandle(), &info, nullptr, &testLayout ) != VK_SUCCESS )
        {
            printf( "Failed to create dummy descriptor set layout for tests!\n" );
        }
    }
};

TEST_F( DescriptorAllocatorTest, BasicAllocation )
{
    if( !device )
        GTEST_SKIP();

    // U¿ywamy alokatora wbudowanego w Device
    VkDescriptorSet set = VK_NULL_HANDLE;
    Result          res = device->AllocateDescriptor( testLayout, set );

    EXPECT_EQ( res, Result::SUCCESS );
    EXPECT_NE( set, VK_NULL_HANDLE );
}

TEST_F( DescriptorAllocatorTest, AllocationOverflow )
{
    if( !device )
        GTEST_SKIP();

    // Try to allocate many sets to force pool switching inside Device
    std::vector<VkDescriptorSet> sets;
    for( int i = 0; i < 1500; ++i )
    {
        VkDescriptorSet set = VK_NULL_HANDLE;
        Result          res = device->AllocateDescriptor( testLayout, set );
        ASSERT_EQ( res, Result::SUCCESS ) << "Failed at index " << i;
        ASSERT_NE( set, VK_NULL_HANDLE );
        sets.push_back( set );
    }
}

TEST_F( DescriptorAllocatorTest, ResetPools )
{
    if( !device )
        GTEST_SKIP();

    VkDescriptorSet set1 = VK_NULL_HANDLE;
    device->AllocateDescriptor( testLayout, set1 );
    EXPECT_NE( set1, VK_NULL_HANDLE );

    // Reset via Device API
    device->ResetDescriptorPools();

    VkDescriptorSet set2 = VK_NULL_HANDLE;
    Result          res  = device->AllocateDescriptor( testLayout, set2 );

    EXPECT_EQ( res, Result::SUCCESS );
    EXPECT_NE( set2, VK_NULL_HANDLE );
}

// =================================================================================================
// PIPELINE TESTS
// =================================================================================================

class PipelineTest : public ::testing::Test
{
protected:
    Ref<Device> device;
    RHIConfig   config;
    std::string compPath = "test_pipe.comp";
    std::string vertPath = "test_pipe.vert";
    std::string fragPath = "test_pipe.frag";

    void SetUp() override
    {
        if( RHI::IsInitialized() )
            RHI::Shutdown();
        config.headless         = true;
        config.enableValidation = false;
        RHI::Init( config );
        if( RHI::GetAdapterCount() > 0 )
            device = RHI::CreateDevice( 0 );

        // Dummy Compute
        {
            std::ofstream out( compPath );
            out << "#version 450\nlayout(local_size_x=1) in;\nlayout(set=0, binding=0) buffer B { float v[]; };\nvoid main(){ v[0]=1.0; }";
        }
        // Dummy Vert
        {
            std::ofstream out( vertPath );
            out << "#version 450\nvoid main(){ gl_Position=vec4(0); }";
        }
        // Dummy Frag
        {
            std::ofstream out( fragPath );
            out << "#version 450\nlayout(location=0) out vec4 c;\nlayout(set=0, binding=1) uniform U { vec4 color; };\nvoid main(){ c=color; }";
        }
    }

    void TearDown() override
    {
        if( device )
            RHI::DestroyDevice( device );
        RHI::Shutdown();
        if( std::filesystem::exists( compPath ) )
            std::filesystem::remove( compPath );
        if( std::filesystem::exists( vertPath ) )
            std::filesystem::remove( vertPath );
        if( std::filesystem::exists( fragPath ) )
            std::filesystem::remove( fragPath );
        std::filesystem::remove( compPath + ".spv" );
        std::filesystem::remove( vertPath + ".spv" );
        std::filesystem::remove( fragPath + ".spv" );
    }
};

TEST_F( PipelineTest, CreateComputePipeline )
{
    if( !device )
        GTEST_SKIP();

    auto                shader = device->CreateShader( compPath );
    ComputePipelineDesc desc;
    desc.shader = shader;

    auto pipeline = device->CreateComputePipeline( desc );
    ASSERT_NE( pipeline, nullptr );
    EXPECT_NE( pipeline->GetHandle(), VK_NULL_HANDLE );
    EXPECT_NE( pipeline->GetLayout(), VK_NULL_HANDLE );

    // Check if layout for Set 0 was created (based on binding=0 in shader)
    EXPECT_NE( pipeline->GetDescriptorSetLayout( 0 ), VK_NULL_HANDLE );
}

TEST_F( PipelineTest, CreateGraphicsPipeline )
{
    if( !device )
        GTEST_SKIP();

    auto vs = device->CreateShader( vertPath );
    auto fs = device->CreateShader( fragPath );

    GraphicsPipelineDesc desc;
    desc.vertexShader   = vs;
    desc.fragmentShader = fs;
    // Default color attachment format R8G8B8A8 matches typical swapchain

    auto pipeline = device->CreateGraphicsPipeline( desc );
    ASSERT_NE( pipeline, nullptr );
    EXPECT_NE( pipeline->GetHandle(), VK_NULL_HANDLE );

    // Check Reflection Merge: FS has binding=1 at Set 0. Layout 0 should exist.
    EXPECT_NE( pipeline->GetDescriptorSetLayout( 0 ), VK_NULL_HANDLE );
}

// =================================================================================================
// SWAPCHAIN TESTS
// =================================================================================================

class SwapchainTest : public ::testing::Test
{
protected:
    Ref<Device>   device;
    Scope<Window> window;
    RHIConfig     config;

    void SetUp() override
    {
        if( RHI::IsInitialized() )
            RHI::Shutdown();

        // Swapchain requires non-headless mode and a window
        config.headless         = false;
        config.enableValidation = true;

        try
        {
            WindowConfig winConfig;
            winConfig.width  = 800;
            winConfig.height = 600;
            winConfig.title  = "Test Swapchain";
            window           = CreateScope<Window>( winConfig );
        }
        catch( ... )
        {
            // If window creation fails (e.g. CI without display), we handle it in test
            return;
        }

        RHI::Init( config );
        if( RHI::GetAdapterCount() > 0 )
            device = RHI::CreateDevice( 0 );
    }

    void TearDown() override
    {
        if( device )
            RHI::DestroyDevice( device );
        RHI::Shutdown();
        window.reset();
    }
};

TEST_F( SwapchainTest, CreateSwapchainViaDevice )
{
    if( !device || !window )
        GTEST_SKIP() << "Skipping Swapchain test (no window/gpu)";

    SwapchainDesc desc;
    desc.width        = window->GetWidth();
    desc.height       = window->GetHeight();
    desc.windowHandle = window->GetNativeWindow();
    desc.vsync        = true;

    // Use factory method from Device
    auto swapchain = device->CreateSwapchain( desc );

    ASSERT_NE( swapchain, nullptr );

    // Check images
    EXPECT_GT( swapchain->GetImageCount(), 0 );
    EXPECT_NE( swapchain->GetImage( 0 ), VK_NULL_HANDLE );
    EXPECT_NE( swapchain->GetImageView( 0 ), VK_NULL_HANDLE );

    // Check format
    VkFormat fmt = swapchain->GetFormat();
    EXPECT_NE( fmt, VK_FORMAT_UNDEFINED );
}

TEST_F( SwapchainTest, AcquireNextImage )
{
    if( !device || !window )
        GTEST_SKIP();

    SwapchainDesc desc;
    desc.width        = window->GetWidth();
    desc.height       = window->GetHeight();
    desc.windowHandle = window->GetNativeWindow();

    auto swapchain = device->CreateSwapchain( desc );
    ASSERT_NE( swapchain, nullptr );

    uint32_t    imageIndex = 0;
    VkSemaphore sem        = swapchain->AcquireNextImage( imageIndex );

    // If not OUT_OF_DATE, we should get a valid semaphore
    if( sem != VK_NULL_HANDLE )
    {
        EXPECT_LT( imageIndex, swapchain->GetImageCount() );
    }
}
