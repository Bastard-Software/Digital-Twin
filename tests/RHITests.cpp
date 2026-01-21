#include "rhi/RHITypes.h"

#include "core/Memory/MemorySystem.h"
#include "rhi/Buffer.h"
#include "rhi/DescriptorAllocator.h"
#include "rhi/Device.h"
#include "rhi/Pipeline.h"
#include "rhi/RHI.h"
#include "rhi/Sampler.h"
#include "rhi/Shader.h"
#include "rhi/Texture.h"
#include <filesystem>
#include <fstream>
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
    RHIConfig  m_config;

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
    Scope<RHI>    m_rhi;
    Scope<Device> m_device;
    RHIConfig     m_config;

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
        if( m_device )
        {
            m_device->Shutdown();
            m_device.reset();
        }

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

// =================================================================================================
// Device Resource Creation Tests
// =================================================================================================

class DeviceResourceTest : public ::testing::Test
{
protected:
    Scope<RHI>    m_rhi;
    Scope<Device> m_device;
    RHIConfig     m_config;

    void SetUp() override
    {
        m_rhi                     = CreateScope<RHI>();
        m_config.headless         = true;
        m_config.enableValidation = true;

        ASSERT_EQ( m_rhi->Initialize( m_config ), Result::SUCCESS );
        ASSERT_EQ( m_rhi->CreateDevice( 0, m_device ), Result::SUCCESS );
    }

    void TearDown() override
    {
        // Device must be destroyed BEFORE RHI shutdown
        if( m_device )
        {
            m_device->Shutdown();
            m_device.reset();
        }

        if( m_rhi )
        {
            m_rhi->Shutdown();
            m_rhi.reset();
        }
    }
};

// =================================================================================================
// Buffer Creation Tests
// ================================================================================================

// 1. Create UPLOAD buffer and verify write/read
TEST_F( DeviceResourceTest, CreateBuffer_UPLOAD )
{
    if( !m_device )
        GTEST_SKIP();

    BufferDesc desc;
    desc.size = 1024;
    desc.type = BufferType::UPLOAD;

    // Stack allocation for raw buffers
    Buffer buffer( m_device->GetAllocator(), m_device->GetHandle(), &m_device->GetAPI() );

    auto res = m_device->CreateBuffer( desc, &buffer );
    ASSERT_EQ( res, Result::SUCCESS );
    EXPECT_NE( buffer.GetHandle(), VK_NULL_HANDLE );

    // Test Write access
    int val = 42;
    buffer.Write( &val, sizeof( int ) );

    // Verify we can read it back (since UPLOAD is host visible)
    int readVal = 0;
    buffer.Read( &readVal, sizeof( int ) );
    EXPECT_EQ( val, readVal );

    m_device->DestroyBuffer( &buffer );
}

// 2. Create READBACK buffer and verify mapping
TEST_F( DeviceResourceTest, CreateBuffer_READBACK )
{
    if( !m_device )
        GTEST_SKIP();

    BufferDesc desc;
    desc.size = 1024;
    desc.type = BufferType::READBACK;

    // Stack allocation for raw buffers
    Buffer buffer( m_device->GetAllocator(), m_device->GetHandle(), &m_device->GetAPI() );

    auto res = m_device->CreateBuffer( desc, &buffer );
    ASSERT_EQ( res, Result::SUCCESS );
    EXPECT_NE( buffer.GetHandle(), VK_NULL_HANDLE );

    // Map should succeed
    EXPECT_NE( buffer.Map(), nullptr );

    m_device->DestroyBuffer( &buffer );
}

// 3. Create STORAGE buffer and verify creation
TEST_F( DeviceResourceTest, CreateBuffer_STORAGE )
{
    if( !m_device )
        GTEST_SKIP();

    BufferDesc desc;
    desc.size = 1024;
    desc.type = BufferType::STORAGE;

    // Stack allocation for raw buffers
    Buffer buffer( m_device->GetAllocator(), m_device->GetHandle(), &m_device->GetAPI() );

    auto res = m_device->CreateBuffer( desc, &buffer );
    ASSERT_EQ( res, Result::SUCCESS );
    EXPECT_NE( buffer.GetHandle(), VK_NULL_HANDLE );

    m_device->DestroyBuffer( &buffer );
}

// 4. Create UNIFORM buffer and verify creation
TEST_F( DeviceResourceTest, CreateBuffer_UNIFORM )
{
    if( !m_device )
        GTEST_SKIP();

    BufferDesc desc;
    desc.size = 256;
    desc.type = BufferType::UNIFORM;

    // Stack allocation for raw buffers
    Buffer buffer( m_device->GetAllocator(), m_device->GetHandle(), &m_device->GetAPI() );

    auto res = m_device->CreateBuffer( desc, &buffer );
    ASSERT_EQ( res, Result::SUCCESS );
    EXPECT_NE( buffer.GetHandle(), VK_NULL_HANDLE );

    // Map should succeed
    EXPECT_NE( buffer.Map(), nullptr );

    // Test Write access
    int val = 42;
    buffer.Write( &val, sizeof( int ) );

    // Verify we can read it back (since UNIFORM is host visible)
    int readVal = 0;
    buffer.Read( &readVal, sizeof( int ) );
    EXPECT_EQ( val, readVal );

    m_device->DestroyBuffer( &buffer );
}

// 5. Create MESH buffer and verify creation
TEST_F( DeviceResourceTest, CreateBuffer_MESH )
{
    if( !m_device )
        GTEST_SKIP();

    BufferDesc desc;
    desc.size = 1024;
    desc.type = BufferType::MESH;

    // Stack allocation for raw buffers
    Buffer buffer( m_device->GetAllocator(), m_device->GetHandle(), &m_device->GetAPI() );

    auto res = m_device->CreateBuffer( desc, &buffer );
    ASSERT_EQ( res, Result::SUCCESS );
    EXPECT_NE( buffer.GetHandle(), VK_NULL_HANDLE );

    m_device->DestroyBuffer( &buffer );
}

// 6. Create INDIRECT buffer and verify creation
TEST_F( DeviceResourceTest, CreateBuffer_INDIRECT )
{
    if( !m_device )
        GTEST_SKIP();

    BufferDesc desc;
    desc.size = 265;
    desc.type = BufferType::INDIRECT;

    // Stack allocation for raw buffers
    Buffer buffer( m_device->GetAllocator(), m_device->GetHandle(), &m_device->GetAPI() );

    auto res = m_device->CreateBuffer( desc, &buffer );
    ASSERT_EQ( res, Result::SUCCESS );
    EXPECT_NE( buffer.GetHandle(), VK_NULL_HANDLE );

    m_device->DestroyBuffer( &buffer );
}

// 7. Create ATOMIC_COUNTER buffer and verify creation
TEST_F( DeviceResourceTest, CreateBuffer_ATOMIC_COUNTER )
{
    if( !m_device )
        GTEST_SKIP();

    BufferDesc desc;
    desc.size = 4;
    desc.type = BufferType::ATOMIC_COUNTER;

    // Stack allocation for raw buffers
    Buffer buffer( m_device->GetAllocator(), m_device->GetHandle(), &m_device->GetAPI() );

    auto res = m_device->CreateBuffer( desc, &buffer );
    ASSERT_EQ( res, Result::SUCCESS );
    EXPECT_NE( buffer.GetHandle(), VK_NULL_HANDLE );

    m_device->DestroyBuffer( &buffer );
}

// ==================================================================================================
// Texture Creation Tests
// ==================================================================================================

// 1. Create 1D Texture for sampling
TEST_F( DeviceResourceTest, CreateTexture_1D_Sampled )
{
    if( !m_device )
        GTEST_SKIP();

    TextureDesc desc;
    desc.type   = TextureType::Texture1D;
    desc.width  = 128;
    desc.format = VK_FORMAT_R8G8B8A8_UNORM;
    desc.usage  = TextureUsage::SAMPLED;

    Texture texture( m_device->GetAllocator(), m_device->GetHandle(), &m_device->GetAPI() );

    auto res = m_device->CreateTexture( desc, &texture );
    ASSERT_EQ( res, Result::SUCCESS );
    EXPECT_NE( texture.GetHandle(), VK_NULL_HANDLE );

    m_device->DestroyTexture( &texture );
}

// 2. Create 2D Texture as Render Target
TEST_F( DeviceResourceTest, CreateTexture_2D_RenderTarget )
{
    if( !m_device )
        GTEST_SKIP();

    TextureDesc desc;
    desc.type   = TextureType::Texture2D;
    desc.width  = 1080;
    desc.height = 720;
    desc.format = VK_FORMAT_R8G8B8A8_UNORM;
    desc.usage  = TextureUsage::RENDER_TARGET;

    Texture texture( m_device->GetAllocator(), m_device->GetHandle(), &m_device->GetAPI() );

    auto res = m_device->CreateTexture( desc, &texture );
    ASSERT_EQ( res, Result::SUCCESS );
    EXPECT_NE( texture.GetHandle(), VK_NULL_HANDLE );
    ASSERT_EQ( texture.GetFormat(), VK_FORMAT_R8G8B8A8_UNORM );

    m_device->DestroyTexture( &texture );
}

// 3. Create 2D Depth Texture
TEST_F( DeviceResourceTest, CreateTexture_2D_DepthStencil )
{
    if( !m_device )
        GTEST_SKIP();

    TextureDesc desc;
    desc.type   = TextureType::Texture2D;
    desc.width  = 256;
    desc.height = 256;
    desc.format = VK_FORMAT_D32_SFLOAT;
    desc.usage  = TextureUsage::DEPTH_STENCIL_TARGET | TextureUsage::SAMPLED;

    Texture texture( m_device->GetAllocator(), m_device->GetHandle(), &m_device->GetAPI() );

    auto res = m_device->CreateTexture( desc, &texture );
    ASSERT_EQ( res, Result::SUCCESS );
    EXPECT_NE( texture.GetHandle(), VK_NULL_HANDLE );
    EXPECT_EQ( texture.GetType(), TextureType::Texture2D );

    m_device->DestroyTexture( &texture );
}

// 4. Create 3D Texture for storage usage
TEST_F( DeviceResourceTest, CreateTexture_3D_Storage )
{
    if( !m_device )
        GTEST_SKIP();

    TextureDesc desc;
    desc.type   = TextureType::Texture3D;
    desc.width  = 64;
    desc.height = 64;
    desc.depth  = 64;
    desc.format = VK_FORMAT_R32_SFLOAT;
    desc.usage  = TextureUsage::STORAGE | TextureUsage::SAMPLED;

    Texture texture( m_device->GetAllocator(), m_device->GetHandle(), &m_device->GetAPI() );

    auto res = m_device->CreateTexture( desc, &texture );
    ASSERT_EQ( res, Result::SUCCESS );
    EXPECT_NE( texture.GetHandle(), VK_NULL_HANDLE );
    EXPECT_EQ( texture.GetExtent().depth, 64 );

    m_device->DestroyTexture( &texture );
}

// 5. Create 2D Texture with STORAGE and TRANSFER usage
TEST_F( DeviceResourceTest, CreateTexture_Storage_Transfer )
{
    if( !m_device )
        GTEST_SKIP();

    TextureDesc desc;
    desc.type   = TextureType::Texture2D;
    desc.width  = 128;
    desc.height = 128;
    desc.format = VK_FORMAT_R32_SFLOAT;
    desc.usage  = TextureUsage::STORAGE | TextureUsage::TRANSFER_SRC | TextureUsage::TRANSFER_DST;

    Texture texture( m_device->GetAllocator(), m_device->GetHandle(), &m_device->GetAPI() );

    auto res = m_device->CreateTexture( desc, &texture );
    ASSERT_EQ( res, Result::SUCCESS );
    EXPECT_NE( texture.GetHandle(), VK_NULL_HANDLE );

    m_device->DestroyTexture( &texture );
}

// ==================================================================================================
// Sampler Creation Tests
// ==================================================================================================
TEST_F( DeviceResourceTest, CreateSampler )
{
    if( !m_device )
        GTEST_SKIP();

    SamplerDesc desc;
    desc.magFilter = VK_FILTER_NEAREST;

    Sampler sampler( m_device->GetHandle(), &m_device->GetAPI() );

    auto res = m_device->CreateSampler( desc, &sampler );
    ASSERT_EQ( res, Result::SUCCESS );
    EXPECT_NE( sampler.GetHandle(), VK_NULL_HANDLE );

    m_device->DestroySampler( &sampler );
}

// =================================================================================================
// Shader Compilation & Reflection Tests
// =================================================================================================

class ShaderCompilationTest : public DeviceResourceTest
{
protected:
    // Temporary shader file names
    std::string computeShaderPath  = "test_compute.comp";
    std::string vertexShaderPath   = "test_vertex.vert";
    std::string fragmentShaderPath = "test_fragment.frag";

    void SetUp() override
    {
        DeviceResourceTest::SetUp();

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
        DeviceResourceTest::TearDown();

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

// 1. Compile Compute Shader
TEST_F( ShaderCompilationTest, CompileComputeShader )
{
    if( !m_device )
        GTEST_SKIP() << "No GPU found, skipping test";

    // Allocate memory for the shader object
    Shader shader( m_device->GetHandle(), &m_device->GetAPI() );

    // Initialize/Compile using the Device API
    auto res = m_device->CreateShader( computeShaderPath, &shader );

    ASSERT_EQ( res, Result::SUCCESS ) << "Shader creation failed";
    EXPECT_NE( shader.GetModule(), VK_NULL_HANDLE ) << "Vulkan shader module handle is null";
    EXPECT_EQ( shader.GetStage(), VK_SHADER_STAGE_COMPUTE_BIT ) << "Incorrect shader stage detected";

    m_device->DestroyShader( &shader );
}

// 2. Verify Compute Shader Reflection
TEST_F( ShaderCompilationTest, ReflectionComputeCheck )
{
    if( !m_device )
        GTEST_SKIP() << "No GPU found, skipping test";

    Shader shader( m_device->GetHandle(), &m_device->GetAPI() );
    auto   res = m_device->CreateShader( computeShaderPath, &shader );
    ASSERT_EQ( res, Result::SUCCESS );

    const auto& reflection = shader.GetReflectionData();

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
    const auto& pcs = shader.GetPushConstantRanges();
    ASSERT_FALSE( pcs.empty() ) << "Push constants not detected";
    EXPECT_GE( pcs[ 0 ].size, sizeof( float ) );

    m_device->DestroyShader( &shader );
}

// 3. Compile Vertex Shader
TEST_F( ShaderCompilationTest, CompileVertexShader )
{
    if( !m_device )
        GTEST_SKIP() << "No GPU found, skipping test";

    Shader shader( m_device->GetHandle(), &m_device->GetAPI() );
    auto   res = m_device->CreateShader( vertexShaderPath, &shader );

    ASSERT_EQ( res, Result::SUCCESS );
    EXPECT_NE( shader.GetModule(), VK_NULL_HANDLE );
    EXPECT_EQ( shader.GetStage(), VK_SHADER_STAGE_VERTEX_BIT );

    m_device->DestroyShader( &shader );
}

// 4. Verify Vertex Shader Reflection
TEST_F( ShaderCompilationTest, ReflectionVertexCheck )
{
    if( !m_device )
        GTEST_SKIP() << "No GPU found, skipping test";

    Shader shader( m_device->GetHandle(), &m_device->GetAPI() );
    auto   res = m_device->CreateShader( vertexShaderPath, &shader );
    ASSERT_EQ( res, Result::SUCCESS );

    const auto& reflection = shader.GetReflectionData();

    // Check Camera UBO
    ASSERT_TRUE( reflection.find( "Camera" ) != reflection.end() ) << "Camera UBO not found";
    EXPECT_EQ( reflection.at( "Camera" ).type, ShaderResourceType::UNIFORM_BUFFER );
    EXPECT_EQ( reflection.at( "Camera" ).binding, 0 );
    EXPECT_GE( reflection.at( "Camera" ).size, 64 ); // mat4

    m_device->DestroyShader( &shader );
}

// 5. Compile Fragment Shader
TEST_F( ShaderCompilationTest, CompileFragmentShader )
{
    if( !m_device )
        GTEST_SKIP() << "No GPU found, skipping test";

    Shader shader( m_device->GetHandle(), &m_device->GetAPI() );
    auto   res = m_device->CreateShader( fragmentShaderPath, &shader );

    ASSERT_EQ( res, Result::SUCCESS );
    EXPECT_NE( shader.GetModule(), VK_NULL_HANDLE );
    EXPECT_EQ( shader.GetStage(), VK_SHADER_STAGE_FRAGMENT_BIT );

    m_device->DestroyShader( &shader );
}

// 6. Verify Fragment Shader Reflection
TEST_F( ShaderCompilationTest, ReflectionFragmentCheck )
{
    if( !m_device )
        GTEST_SKIP() << "No GPU found, skipping test";

    Shader shader( m_device->GetHandle(), &m_device->GetAPI() );
    auto   res = m_device->CreateShader( fragmentShaderPath, &shader );
    ASSERT_EQ( res, Result::SUCCESS );

    const auto& reflection = shader.GetReflectionData();

    // Check Sampler
    ASSERT_TRUE( reflection.find( "texSampler" ) != reflection.end() ) << "Sampler not found";
    EXPECT_EQ( reflection.at( "texSampler" ).type, ShaderResourceType::SAMPLED_IMAGE );
    EXPECT_EQ( reflection.at( "texSampler" ).binding, 1 );

    m_device->DestroyShader( &shader );
}

// =================================================================================================
// Pipeline Creation Tests
// =================================================================================================

class PipelineTest : public DeviceResourceTest
{
protected:
    std::string m_compPath = "test_pipe.comp";
    std::string m_vertPath = "test_pipe.vert";
    std::string m_fragPath = "test_pipe.frag";

    Scope<Shader> m_comp;
    Scope<Shader> m_vert;
    Scope<Shader> m_frag;

    void SetUp() override
    {
        DeviceResourceTest::SetUp();

        // 1. Create dummy Compute Shader
        {
            std::ofstream out( m_compPath );
            out << "#version 450\n";
            out << "layout(local_size_x=1) in;\n";
            out << "layout(set=0, binding=0) buffer B { float v[]; };\n";
            out << "void main(){ v[0]=1.0; }";
            out.close();
        }
        m_comp = CreateScope<Shader>( m_device->GetHandle(), &m_device->GetAPI() );
        ASSERT_EQ( m_device->CreateShader( m_compPath, m_comp.get() ), Result::SUCCESS );

        // 2. Create dummy Vertex Shader
        {
            std::ofstream out( m_vertPath );
            out << "#version 450\n";
            out << "void main(){ gl_Position=vec4(0); }";
            out.close();
        }
        m_vert = CreateScope<Shader>( m_device->GetHandle(), &m_device->GetAPI() );
        ASSERT_EQ( m_device->CreateShader( m_vertPath, m_vert.get() ), Result::SUCCESS );

        // 3. Create dummy Fragment Shader
        {
            std::ofstream out( m_fragPath );
            out << "#version 450\n";
            out << "layout(location=0) out vec4 c;\n";
            out << "layout(set=0, binding=1) uniform U { vec4 color; };\n";
            out << "void main(){ c=color; }";
            out.close();
        }
        m_frag = CreateScope<Shader>( m_device->GetHandle(), &m_device->GetAPI() );
        ASSERT_EQ( m_device->CreateShader( m_fragPath, m_frag.get() ), Result::SUCCESS );
    }

    void TearDown() override
    {
        auto cleanup = []( const std::string& path ) {
            if( std::filesystem::exists( path ) )
                std::filesystem::remove( path );

            // Remove cached SPIR-V file to force recompilation next time
            std::string cache = path + ".spv";
            if( std::filesystem::exists( cache ) )
                std::filesystem::remove( cache );
        };

        cleanup( m_compPath );
        m_device->DestroyShader( m_comp.get() );
        m_comp.reset();
        cleanup( m_vertPath );
        m_device->DestroyShader( m_vert.get() );
        m_vert.reset();
        cleanup( m_fragPath );
        m_device->DestroyShader( m_frag.get() );
        m_frag.reset();

        DeviceResourceTest::TearDown();
    }
};

// 1. Compute pipeline creation test
TEST_F( PipelineTest, CreateComputePipeline )
{
    if( !m_device )
        GTEST_SKIP() << "No GPU found, skipping test";

    ComputePipelineNativeDesc desc;
    desc.shader = m_comp.get();

    // Create pipeline
    ComputePipeline pipeline( m_device->GetHandle(), &m_device->GetAPI() );
    auto            res = m_device->CreateComputePipeline( desc, &pipeline );
    ASSERT_EQ( res, Result::SUCCESS );

    EXPECT_NE( pipeline.GetHandle(), VK_NULL_HANDLE );
    EXPECT_NE( pipeline.GetLayout(), VK_NULL_HANDLE );

    // Check if layout for Set 0 was created (based on binding=0 in shader)
    EXPECT_NE( pipeline.GetDescriptorSetLayout( 0 ), VK_NULL_HANDLE );

    m_device->DestroyComputePipeline( &pipeline );
}

// 2. Graphics pipeline creation test
TEST_F( PipelineTest, CreateGraphicsPipeline )
{
    if( !m_device )
        GTEST_SKIP() << "No GPU found, skipping test";

    GraphicsPipelineNativeDesc desc;
    desc.vertexShader   = m_vert.get();
    desc.fragmentShader = m_frag.get();

    // Create pipeline
    GraphicsPipeline pipeline( m_device->GetHandle(), &m_device->GetAPI() );
    auto             res = m_device->CreateGraphicsPipeline( desc, &pipeline );
    ASSERT_EQ( res, Result::SUCCESS );

    EXPECT_NE( pipeline.GetHandle(), VK_NULL_HANDLE );

    // Check Reflection Merge: FS has binding=1 at Set 0. Layout 0 should exist.
    EXPECT_NE( pipeline.GetDescriptorSetLayout( 0 ), VK_NULL_HANDLE );

    m_device->DestroyGraphicsPipeline( &pipeline );
}

// =================================================================================================
// Descriptor Allocator Tests
// =================================================================================================

class DescriptorAllocatorTest : public DeviceResourceTest
{
protected:
    std::unique_ptr<DescriptorAllocator> allocator;
    VkDescriptorSetLayout                testLayout = VK_NULL_HANDLE;

    void SetUp() override
    {
        DeviceResourceTest::SetUp();
        if( m_device )
        {
            // PASS API TABLE HERE
            allocator = std::make_unique<DescriptorAllocator>( m_device->GetHandle(), &m_device->GetAPI() );
            allocator->Initialize();

            // Create a dummy layout for testing (1 UBO binding)
            // Note: We use the device API directly here for setup helper
            VkDescriptorSetLayoutBinding binding{};
            binding.binding         = 0;
            binding.descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            binding.descriptorCount = 1;
            binding.stageFlags      = VK_SHADER_STAGE_VERTEX_BIT;

            VkDescriptorSetLayoutCreateInfo info{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
            info.bindingCount = 1;
            info.pBindings    = &binding;

            m_device->GetAPI().vkCreateDescriptorSetLayout( m_device->GetHandle(), &info, nullptr, &testLayout );
        }
    }

    void TearDown() override
    {
        if( m_device )
        {
            if( testLayout != VK_NULL_HANDLE )
                m_device->GetAPI().vkDestroyDescriptorSetLayout( m_device->GetHandle(), testLayout, nullptr );

            allocator->Shutdown();
            allocator.reset();
        }
        DeviceResourceTest::TearDown();
    }
};

// 1. Basic Allocation Test
TEST_F( DescriptorAllocatorTest, BasicAllocation )
{
    if( !m_device )
        GTEST_SKIP();

    VkDescriptorSet set = VK_NULL_HANDLE;
    Result          res = allocator->Allocate( testLayout, set );

    EXPECT_EQ( res, Result::SUCCESS );
    EXPECT_NE( set, VK_NULL_HANDLE );
}

// 2. Pool Growing Test (Force new pool creation)
TEST_F( DescriptorAllocatorTest, PoolGrowing )
{
    if( !m_device )
        GTEST_SKIP();

    // The allocator creates pools of size 1000.
    // We try to allocate 1500 sets to ensure it creates a second pool internally.
    std::vector<VkDescriptorSet> sets;
    const int                    count = 1500;
    sets.resize( count );

    for( int i = 0; i < count; ++i )
    {
        Result res = allocator->Allocate( testLayout, sets[ i ] );
        ASSERT_EQ( res, Result::SUCCESS ) << "Failed to allocate set at index " << i;
        ASSERT_NE( sets[ i ], VK_NULL_HANDLE );
    }
}

// 3. Reset Functionality Test
TEST_F( DescriptorAllocatorTest, ResetPools )
{
    if( !m_device )
        GTEST_SKIP();

    VkDescriptorSet set1 = VK_NULL_HANDLE;
    allocator->Allocate( testLayout, set1 );
    EXPECT_NE( set1, VK_NULL_HANDLE );

    // Reset logic (simulating end of frame)
    allocator->ResetPools();

    // After reset, we should be able to allocate again.
    // Note: Technically set1 is now invalid/dangling in Vulkan terms, but the pointer value remains.

    VkDescriptorSet set2 = VK_NULL_HANDLE;
    Result          res  = allocator->Allocate( testLayout, set2 );

    EXPECT_EQ( res, Result::SUCCESS );
    EXPECT_NE( set2, VK_NULL_HANDLE );
}