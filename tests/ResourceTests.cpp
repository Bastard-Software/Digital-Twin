#include "core/memory/MemorySystem.h"
#include "resources/ResourceManager.h"
#include "rhi/Buffer.h"
#include "rhi/Device.h"
#include "rhi/RHI.h"
#include <filesystem>
#include <gtest/gtest.h>

using namespace DigitalTwin;

// =================================================================================================
// Resource Manager Lifecycle Tests
// =================================================================================================

class ResourceManagerTest : public ::testing::Test
{
protected:
    // Systems required for the test
    Scope<RHI>             m_rhi;
    Scope<Device>          m_device;
    RHIConfig              m_config;
    MemorySystem           m_mem;
    Scope<ResourceManager> m_rm;

    void SetUp() override
    {
        m_rhi                     = CreateScope<RHI>();
        m_config.headless         = true;
        m_config.enableValidation = true;

        m_mem.Initialize();
        ASSERT_EQ( m_rhi->Initialize( m_config ), Result::SUCCESS );
        ASSERT_EQ( m_rhi->CreateDevice( 0, m_device ), Result::SUCCESS );
        m_rm = CreateScope<ResourceManager>( m_device.get(), &m_mem );
        ASSERT_EQ( m_rm->Initialize(), Result::SUCCESS );
    }

    void TearDown() override
    {
        if( m_rm )
        {
            m_rm->Shutdown();
            m_rm.reset();
        }

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

        m_mem.Shutdown();
    }
};

// 1. Handle copy test
TEST_F( ResourceManagerTest, HandleCopySemantics )
{
    if( !m_device )
        GTEST_SKIP();

    // Create a resource to get a valid handle
    BufferDesc   desc{ 256, BufferType::UNIFORM };
    BufferHandle originalHandle = m_rm->CreateBuffer( desc );
    ASSERT_TRUE( originalHandle.IsValid() );

    // 1. Copy Constructor
    BufferHandle copiedHandle = originalHandle;
    EXPECT_EQ( originalHandle, copiedHandle ) << "Copied handle should be equal to original";
    EXPECT_EQ( originalHandle.GetIndex(), copiedHandle.GetIndex() );
    EXPECT_EQ( originalHandle.GetGeneration(), copiedHandle.GetGeneration() );

    // 2. Assignment Operator
    BufferHandle assignedHandle;
    assignedHandle = originalHandle;
    EXPECT_EQ( originalHandle, assignedHandle ) << "Assigned handle should be equal to original";

    // 3. Pointer Retrieval
    // All handles should resolve to the exact same memory address
    Buffer* ptr1 = m_rm->GetBuffer( originalHandle );
    Buffer* ptr2 = m_rm->GetBuffer( copiedHandle );
    Buffer* ptr3 = m_rm->GetBuffer( assignedHandle );

    EXPECT_EQ( ptr1, ptr2 );
    EXPECT_EQ( ptr1, ptr3 );
    EXPECT_NE( ptr1, nullptr );

    m_rm->DestroyBuffer( originalHandle );
}

// 2. Resource lifecycle test
TEST_F( ResourceManagerTest, BufferCreationAndDestructionFlow )
{
    if( !m_device )
        GTEST_SKIP();

    // Setup: Define a generic buffer description
    BufferDesc desc{};
    desc.size = 1024;
    desc.type = BufferType::UNIFORM;

    // 1. Create Buffer 1 & Get Handle
    BufferHandle handle1 = m_rm->CreateBuffer( desc );

    // Verify handle is valid
    EXPECT_TRUE( handle1.IsValid() ) << "Buffer 1 handle should be valid after creation";
    EXPECT_NE( m_rm->GetBuffer( handle1 ), nullptr ) << "Should be able to retrieve Buffer 1 pointer";

    // 2. Create Buffer 2 & Get Handle
    BufferHandle handle2 = m_rm->CreateBuffer( desc );

    // Verify handle is valid and distinct from Buffer 1
    EXPECT_TRUE( handle2.IsValid() ) << "Buffer 2 handle should be valid";
    EXPECT_NE( handle1, handle2 ) << "Handles for different resources must be different";
    EXPECT_NE( m_rm->GetBuffer( handle2 ), nullptr ) << "Should be able to retrieve Buffer 2 pointer";

    // 3. Destroy Buffer 2
    m_rm->DestroyBuffer( handle2 );

    // 4. Try to Get Buffer 2 (Expect Failure)
    // The handle logic (generation counter) should prevent access to a destroyed resource
    Buffer* retrievedBuffer2 = m_rm->GetBuffer( handle2 );

    EXPECT_EQ( retrievedBuffer2, nullptr ) << "Accessing destroyed Buffer 2 should return nullptr";

    // Verify Buffer 1 is still alive and accessible (independence check)
    EXPECT_NE( m_rm->GetBuffer( handle1 ), nullptr ) << "Buffer 1 should remain valid after Buffer 2 destruction";

    // Call BeginFrame to simulate garbage collection of zombie resources
    // (This ensures memory is actually freed via the MemorySystem deleter)
    m_rm->BeginFrame();
}

// 3. All resources creation test
TEST_F( ResourceManagerTest, CreateAllResourceTypes )
{
    if( !m_device )
        GTEST_SKIP();

    // --- 1. Buffer ---
    BufferDesc   bufDesc{ 1024, BufferType::STORAGE };
    BufferHandle hBuf = m_rm->CreateBuffer( bufDesc );
    EXPECT_TRUE( hBuf.IsValid() );
    EXPECT_NE( m_rm->GetBuffer( hBuf ), nullptr ) << "Failed to retrieve Buffer";

    // --- 2. Texture ---
    TextureDesc texDesc;
    texDesc.width      = 64;
    texDesc.height     = 64;
    texDesc.format     = VK_FORMAT_R8G8B8A8_UNORM;
    TextureHandle hTex = m_rm->CreateTexture( texDesc );
    EXPECT_TRUE( hTex.IsValid() );
    EXPECT_NE( m_rm->GetTexture( hTex ), nullptr ) << "Failed to retrieve Texture";

    // --- 3. Sampler ---
    SamplerDesc   sampDesc;
    SamplerHandle hSamp = m_rm->CreateSampler( sampDesc );
    EXPECT_TRUE( hSamp.IsValid() );
    EXPECT_NE( m_rm->GetSampler( hSamp ), nullptr ) << "Failed to retrieve Sampler";

    // --- 4. Shaders ---
    // We need temporary files for shader compilation
    std::string compPath = "res_test_comp.comp";
    std::string vertPath = "res_test_vert.vert";
    std::string fragPath = "res_test_frag.frag";

    auto createShaderFile = []( const std::string& path, const std::string& source ) {
        std::ofstream out( path );
        out << source;
        out.close();
    };

    createShaderFile( compPath, "#version 450\nlayout(local_size_x=1) in; void main(){}" );
    createShaderFile( vertPath, "#version 450\nvoid main(){ gl_Position=vec4(0); }" );
    createShaderFile( fragPath, "#version 450\nlayout(location=0) out vec4 c; void main(){ c=vec4(1); }" );

    ShaderHandle hCompShader = m_rm->CreateShader( compPath );
    EXPECT_TRUE( hCompShader.IsValid() );
    EXPECT_NE( m_rm->GetShader( hCompShader ), nullptr ) << "Failed to retrieve Compute Shader";

    ShaderHandle hVertShader = m_rm->CreateShader( vertPath );
    EXPECT_TRUE( hVertShader.IsValid() );

    ShaderHandle hFragShader = m_rm->CreateShader( fragPath );
    EXPECT_TRUE( hFragShader.IsValid() );

    // --- 5. Compute Pipeline ---
    ComputePipelineDesc compPipeDesc;
    compPipeDesc.shader = hCompShader;

    // Using CreatePipeline overload for Compute
    ComputePipelineHandle hCompPipe = m_rm->CreatePipeline( compPipeDesc );
    EXPECT_TRUE( hCompPipe.IsValid() );
    EXPECT_NE( m_rm->GetPipeline( hCompPipe ), nullptr ) << "Failed to retrieve Compute Pipeline";

    // --- 6. Graphics Pipeline ---
    GraphicsPipelineDesc gfxPipeDesc;
    gfxPipeDesc.vertexShader   = hVertShader;
    gfxPipeDesc.fragmentShader = hFragShader;

    // Using CreatePipeline overload for Graphics
    GraphicsPipelineHandle hGfxPipe = m_rm->CreatePipeline( gfxPipeDesc );
    EXPECT_TRUE( hGfxPipe.IsValid() );
    EXPECT_NE( m_rm->GetPipeline( hGfxPipe ), nullptr ) << "Failed to retrieve Graphics Pipeline";

    // Cleanup temporary files
    auto removeFile = []( const std::string& path ) {
        if( std::filesystem::exists( path ) )
            std::filesystem::remove( path );
        if( std::filesystem::exists( path + ".spv" ) )
            std::filesystem::remove( path + ".spv" );
    };
    removeFile( compPath );
    removeFile( vertPath );
    removeFile( fragPath );
}