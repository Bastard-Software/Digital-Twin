#include "SetupHelpers.h"
#include "resources/ResourceManager.h"
#include "resources/StreamingManager.h"
#include "rhi/BindingGroup.h"
#include "rhi/Buffer.h"
#include "rhi/Device.h"
#include "rhi/Pipeline.h"
#include "rhi/Queue.h"
#include "rhi/RHI.h"
#include "rhi/Texture.h"
#include "rhi/ThreadContext.h"
#include "simulation/SimulationBlueprint.h"
#include "simulation/SimulationBuilder.h"
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>

// Private headers directly included for testing internal mechanics
#include "compute/ComputeGraph.h"
#include "compute/ComputeTask.h"
#include "compute/GraphDispatcher.h"

using namespace DigitalTwin;

class ComputeTest : public ::testing::Test
{
protected:
    Scope<MemorySystem>     m_memory;
    Scope<FileSystem>       m_fileSystem;
    Scope<RHI>              m_rhi;
    Scope<Device>           m_device;
    Scope<ResourceManager>  m_rm;
    Scope<StreamingManager> m_stream;

    void SetUp() override
    {
        m_memory = CreateScope<MemorySystem>();
        m_memory->Initialize();

        std::filesystem::path projectRoot = std::filesystem::current_path();
        std::filesystem::path engineRoot  = Helpers::FindEngineRoot();
        std::filesystem::path internalAssets;

        if( !engineRoot.empty() )
        {
            internalAssets = engineRoot / "assets";
        }
        else
        {
            // Last ditch effort: check if assets are next to the executable
            if( std::filesystem::exists( std::filesystem::current_path() / "assets" ) )
            {
                internalAssets = std::filesystem::current_path() / "assets";
            }
        }
        m_fileSystem = CreateScope<FileSystem>( m_memory.get() );
        m_fileSystem->Initialize( projectRoot, internalAssets );

        m_rhi = CreateScope<RHI>();
        RHIConfig config{ true, true };
        m_rhi->Initialize( config );

        if( m_rhi->GetAdapters().empty() )
            GTEST_SKIP();
        m_rhi->CreateDevice( 0, m_device );
        m_rm = CreateScope<ResourceManager>( m_device.get(), m_memory.get(), m_fileSystem.get() );
        m_rm->Initialize();

        m_stream = CreateScope<StreamingManager>( m_device.get(), m_rm.get() );
        m_stream->Initialize();
    }

    void TearDown() override
    {
        if( m_stream )
            m_stream->Shutdown();
        if( m_rm )
            m_rm->Shutdown();
        if( m_device )
            m_device->Shutdown();
        if( m_rhi )
            m_rhi->Shutdown();
        if( m_fileSystem )
            m_fileSystem->Shutdown();
        if( m_memory )
            m_memory->Shutdown();
    }
};

// =================================================================================================
// Compute engine lifecycle tests
// =================================================================================================

// 1. Verify that the task frequency limiter works correctly
TEST_F( ComputeTest, TaskFrequencyExecution )
{
    ComputePushConstants pc{};
    ComputeTask          task( nullptr, nullptr, nullptr, 10.0f /* 10 Hz */, pc, glm::uvec3( 1 ) );

    EXPECT_FALSE( task.ShouldExecute( 0.05f ) );
    EXPECT_TRUE( task.ShouldExecute( 0.06f ) );
    EXPECT_FALSE( task.ShouldExecute( 0.05f ) );
}

// 2. Full graph execution test via Dispatcher
TEST_F( ComputeTest, GraphExecution )
{
    if( !m_device )
        GTEST_SKIP();

    std::string   testShader = "test_compute_graph.comp";
    std::ofstream out( testShader );
    out << "#version 450\n"
        << "layout(local_size_x=256) in;\n"
        << "layout(set=0, binding=0) buffer P { float pos[]; };\n"
        << "layout(push_constant) uniform PC { float dt; float totalTime; float fParam0; float fParam1; float fParam2; float fParam3; float fParam4; "
           "float fParam5; float fParam6; float fParam7; uint offset; uint maxCapacity; uint "
           "uParam0; uint uParam1; vec4 domainSize; uvec4 gridSize; } pc;\n"
        << "void main() { if(gl_GlobalInvocationID.x < pc.maxCapacity) { pos[pc.offset + gl_GlobalInvocationID.x] += pc.fParam0; } }";
    out.close();
    ShaderHandle shaderHandle = m_rm->CreateShader( testShader );

    ComputePipelineDesc compDesc;
    compDesc.shader                  = shaderHandle;
    ComputePipelineHandle pipeHandle = m_rm->CreatePipeline( compDesc );
    ComputePipeline*      pipe       = m_rm->GetPipeline( pipeHandle );

    BindingGroupHandle bgHandle0 = m_rm->CreateBindingGroup( pipeHandle, 0 );
    BindingGroupHandle bgHandle1 = m_rm->CreateBindingGroup( pipeHandle, 0 );
    BindingGroup*      bg0       = m_rm->GetBindingGroup( bgHandle0 );
    BindingGroup*      bg1       = m_rm->GetBindingGroup( bgHandle1 );

    BufferHandle buf = m_rm->CreateBuffer( { 1024, BufferType::STORAGE } );
    bg0->Bind( 0, m_rm->GetBuffer( buf ) );
    bg1->Bind( 0, m_rm->GetBuffer( buf ) );
    bg0->Build();
    bg1->Build();

    ComputePushConstants pc{ 0.16f, 1.0f, 1.0f, 0, 0, 100 };
    ComputeTask          task( pipe, bg0, bg1, 0.0f, pc, glm::uvec3( 1 ) );

    ComputeGraph graph;
    graph.AddTask( task );

    auto compCtxHandle = m_device->CreateThreadContext( QueueType::COMPUTE );
    auto compCtx       = m_device->GetThreadContext( compCtxHandle );
    auto compCmd       = compCtx->GetCommandBuffer( compCtx->CreateCommandBuffer() );

    auto gfxCtxHandle = m_device->CreateThreadContext( QueueType::GRAPHICS );
    auto gfxCtx       = m_device->GetThreadContext( gfxCtxHandle );
    auto gfxCmd       = gfxCtx->GetCommandBuffer( gfxCtx->CreateCommandBuffer() );

    // Act like EndFrame
    compCmd->Begin();
    gfxCmd->Begin();
    GraphDispatcher::Dispatch( &graph, compCmd, gfxCmd, 0.016f, 1.0f, 0 );
    gfxCmd->End();
    compCmd->End();

    m_device->GetComputeQueue()->Submit( { compCmd } );
    m_device->GetComputeQueue()->WaitIdle();

    m_rm->DestroyBuffer( buf );

    std::filesystem::remove( testShader );
    std::filesystem::remove( testShader + ".spv" );
}

// =================================================================================================
// Compute shader tests
// =================================================================================================

// 1. Compute field interaction test
TEST_F( ComputeTest, Shader_FieldInteraction_AtomicAdd )
{
    if( !m_device )
        GTEST_SKIP();

    // 1. Prepare raw data for 1 Agent
    std::vector<glm::vec4> agents        = { glm::vec4( 0.0f, 0.0f, 0.0f, 1.0f ) };
    uint32_t               currentAgents = static_cast<uint32_t>( agents.size() );
    size_t                 agentsSize    = agents.size() * sizeof( glm::vec4 );
    size_t                 countSize     = currentAgents * sizeof( uint32_t );

    // Grid: 10x10x10 = 1000 voxels (Atomic Target)
    uint32_t         voxelCount = 1000;
    std::vector<int> deltas( voxelCount, 0 );
    size_t           deltasSize = deltas.size() * sizeof( int );

    // 2. Allocate & Upload Buffers directly
    BufferHandle agentBuffer = m_rm->CreateBuffer( { agentsSize, BufferType::STORAGE, "TestAgentBuffer" } );
    BufferHandle deltaBuffer = m_rm->CreateBuffer( { deltasSize, BufferType::STORAGE, "TestDeltaBuffer" } );
    BufferHandle countBuf    = m_rm->CreateBuffer( { countSize, BufferType::STORAGE, "TestCountBuffer" } );

    std::vector<BufferUploadRequest> uploads = { { agentBuffer, agents.data(), agentsSize, 0 },
                                                 { deltaBuffer, deltas.data(), deltasSize, 0 },
                                                 { countBuf, &currentAgents, countSize, 0 } };
    m_stream->UploadBufferImmediate( uploads );

    // 3. Setup Pipeline & BindingGroup directly via RHI
    ComputePipelineDesc pipeDesc{};
    pipeDesc.shader                  = m_rm->CreateShader( "shaders/compute/field_interaction.comp" );
    pipeDesc.debugName               = "TestFieldInteraction";
    ComputePipelineHandle pipeHandle = m_rm->CreatePipeline( pipeDesc );
    ComputePipeline*      pipeline   = m_rm->GetPipeline( pipeHandle );

    BindingGroupHandle bgHandle = m_rm->CreateBindingGroup( pipeHandle, 0 );
    BindingGroup*      bg       = m_rm->GetBindingGroup( bgHandle );
    bg->Bind( 0, m_rm->GetBuffer( agentBuffer ) );
    bg->Bind( 1, m_rm->GetBuffer( deltaBuffer ) );
    bg->Bind( 2, m_rm->GetBuffer( countBuf ) );
    bg->Bind( 3, m_rm->GetBuffer( agentBuffer ) ); // Temp for silencing validation
    bg->Build();

    // 4. Setup Push Constants
    ComputePushConstants pc{};
    pc.dt          = 1.0f;
    pc.fParam0     = 15.0f; // Simulate 15.0 units of interaction
    pc.maxCapacity = 1;
    pc.offset      = 0;
    pc.uParam1     = 0;
    pc.domainSize  = glm::vec4( 10.0f );
    pc.gridSize    = glm::uvec4( 10, 10, 10, 0 );

    // 5. Dispatch Manually
    auto compCtxHandle = m_device->CreateThreadContext( QueueType::COMPUTE );
    auto compCtx       = m_device->GetThreadContext( compCtxHandle );
    auto compCmd       = compCtx->GetCommandBuffer( compCtx->CreateCommandBuffer() );

    compCmd->Begin();
    compCmd->SetPipeline( pipeline );
    compCmd->SetBindingGroup( bg, pipeline->GetLayout(), VK_PIPELINE_BIND_POINT_COMPUTE );
    compCmd->PushConstants( pipeline->GetLayout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof( pc ), &pc );
    compCmd->Dispatch( 1, 1, 1 );
    compCmd->End();

    m_device->GetComputeQueue()->Submit( { compCmd } );
    m_device->GetComputeQueue()->WaitIdle();

    // 6. Readback and Assert
    std::vector<int> resultDeltas( voxelCount );
    m_stream->ReadbackBufferImmediate( deltaBuffer, resultDeltas.data(), deltasSize );

    // Center voxel index in 10x10x10: 5 + 5*10 + 5*100 = 555
    uint32_t centerIdx = 555;

    // Expected logic in shader: intDelta = int(param1 * dt * 100000.0)
    int expectedAtomicValue = static_cast<int>( 15.0f * 1.0f * 100000.0f );

    EXPECT_EQ( resultDeltas[ centerIdx ], expectedAtomicValue ) << "Shader atomicAdd logic failed or agent mapped incorrectly!";
    EXPECT_EQ( resultDeltas[ 0 ], 0 ) << "Data corruption in unrelated grid voxel!";

    m_rm->DestroyBuffer( agentBuffer );
    m_rm->DestroyBuffer( deltaBuffer );
    m_rm->DestroyBuffer( countBuf );
}

// 2. Compute diffusion and decay test
TEST_F( ComputeTest, Shader_Diffusion_Integration )
{
    if( !m_device )
        GTEST_SKIP();

    // 1. Prepare raw data for a 10x10x10 grid
    uint32_t width = 10, height = 10, depth = 10;
    uint32_t voxelCount   = width * height * depth;
    size_t   gridByteSize = voxelCount * sizeof( float );

    // Center voxel index in 10x10x10: 5 + 5*10 + 5*100 = 555
    uint32_t centerIdx = 555;

    // Initial grid state (Ping)
    std::vector<float> initialGrid( voxelCount, 0.0f );
    initialGrid[ centerIdx ] = 100.0f; // Center has 100 units

    // Empty target grid (Pong)
    std::vector<float> emptyGrid( voxelCount, 0.0f );

    // Delta buffer (simulate an agent deposited 50.0 units into the center voxel)
    std::vector<int> initialDeltas( voxelCount, 0 );
    int              simulatedAtomicInteraction = static_cast<int>( 50.0f * 100000.0f );
    initialDeltas[ centerIdx ]                  = simulatedAtomicInteraction;

    size_t deltasSize = initialDeltas.size() * sizeof( int );

    // 2. Allocate & Upload Resources
    TextureDesc texDesc{};
    texDesc.type   = TextureType::Texture3D;
    texDesc.width  = width;
    texDesc.height = height;
    texDesc.depth  = depth;
    texDesc.format = VK_FORMAT_R32_SFLOAT;
    texDesc.usage  = TextureUsage::STORAGE | TextureUsage::TRANSFER_DST | TextureUsage::TRANSFER_SRC;

    // We need 2 textures for Ping-Pong
    texDesc.debugName           = "TestReadTexture";
    TextureHandle readTexHandle = m_rm->CreateTexture( texDesc );

    texDesc.debugName            = "TestWriteTexture";
    TextureHandle writeTexHandle = m_rm->CreateTexture( texDesc );

    BufferHandle deltaBuffer = m_rm->CreateBuffer( { deltasSize, BufferType::STORAGE, "TestDeltaBuffer" } );

    // Upload initial data
    std::vector<TextureUploadRequest> texUploads = { { readTexHandle, initialGrid.data(), gridByteSize },
                                                     { writeTexHandle, emptyGrid.data(), gridByteSize } };
    std::vector<BufferUploadRequest>  bufUploads = { { deltaBuffer, initialDeltas.data(), deltasSize, 0 } };

    m_stream->UploadTextureImmediate( texUploads );
    m_stream->UploadBufferImmediate( bufUploads );

    // 3. Setup Pipeline & BindingGroup directly via RHI
    ComputePipelineDesc pipeDesc{};
    pipeDesc.shader                  = m_rm->CreateShader( "shaders/compute/diffusion.comp" );
    pipeDesc.debugName               = "TestDiffusion";
    ComputePipelineHandle pipeHandle = m_rm->CreatePipeline( pipeDesc );
    ComputePipeline*      pipeline   = m_rm->GetPipeline( pipeHandle );

    BindingGroupHandle bgHandle = m_rm->CreateBindingGroup( pipeHandle, 0 );
    BindingGroup*      bg       = m_rm->GetBindingGroup( bgHandle );

    // Bind Texture Read, Texture Write, and Delta Buffer
    bg->Bind( 0, m_rm->GetTexture( readTexHandle ), VK_IMAGE_LAYOUT_GENERAL );
    bg->Bind( 1, m_rm->GetTexture( writeTexHandle ), VK_IMAGE_LAYOUT_GENERAL );
    bg->Bind( 2, m_rm->GetBuffer( deltaBuffer ) );
    bg->Build();

    // 4. Setup Push Constants
    ComputePushConstants pc{};
    pc.dt          = 1.0f;
    pc.fParam0     = 0.0f; // No diffusion
    pc.fParam1     = 0.0f; // No decay
    pc.maxCapacity = 0;
    pc.offset      = 0;
    pc.domainSize  = glm::vec4( 10.0f );
    pc.gridSize    = glm::uvec4( width, height, depth, 0 );

    // 5. Dispatch Manually
    auto compCtxHandle = m_device->CreateThreadContext( QueueType::COMPUTE );
    auto compCtx       = m_device->GetThreadContext( compCtxHandle );
    auto compCmd       = compCtx->GetCommandBuffer( compCtx->CreateCommandBuffer() );

    compCmd->Begin();
    compCmd->SetPipeline( pipeline );
    compCmd->SetBindingGroup( bg, pipeline->GetLayout(), VK_PIPELINE_BIND_POINT_COMPUTE );
    compCmd->PushConstants( pipeline->GetLayout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof( pc ), &pc );

    // Dispatch is based on 8x8x8 local_size defined in diffusion.comp
    compCmd->Dispatch( ( width + 7 ) / 8, ( height + 7 ) / 8, ( depth + 7 ) / 8 );
    compCmd->End();

    m_device->GetComputeQueue()->Submit( { compCmd } );
    m_device->GetComputeQueue()->WaitIdle();

    // 6. Readback and Assert
    std::vector<float> resultGrid( voxelCount );
    m_stream->ReadbackTextureImmediate( writeTexHandle, resultGrid.data(), gridByteSize );

    std::vector<int> resultDeltas( voxelCount );
    m_stream->ReadbackBufferImmediate( deltaBuffer, resultDeltas.data(), deltasSize );

    // The center voxel started at 100.0. The agent added 50.0.
    // Since diffusion and decay are 0.0, the new value should be exactly 150.0.
    EXPECT_FLOAT_EQ( resultGrid[ centerIdx ], 150.0f ) << "Diffusion shader did not integrate atomic interaction properly!";

    // The delta buffer MUST be reset to 0 by the atomicExchange in the shader
    EXPECT_EQ( resultDeltas[ centerIdx ], 0 ) << "Atomic buffer was not cleared by atomicExchange!";

    // Cleanup
    m_rm->DestroyTexture( readTexHandle );
    m_rm->DestroyTexture( writeTexHandle );
    m_rm->DestroyBuffer( deltaBuffer );
}

// 3. Compute spatial hashing test for agents
TEST_F( ComputeTest, Shader_HashAgents_Logic )
{
    if( !m_device )
        GTEST_SKIP();

    // 1. Prepare raw data for Agents in different regions
    // We test positive, shifted, and negative coordinates to ensure floor() logic works
    std::vector<glm::vec4> agents = {
        glm::vec4( 0.5f, 0.5f, 0.5f, 1.0f ),  // Should be in cell (0, 0, 0) if cellSize is 2.0
        glm::vec4( 2.5f, 0.5f, 0.5f, 1.0f ),  // Should be in cell (1, 0, 0)
        glm::vec4( -1.5f, 2.5f, -2.5f, 1.0f ) // Should be in cell (-1, 1, -2)
    };

    uint32_t agentCount = static_cast<uint32_t>( agents.size() );
    size_t   agentsSize = agentCount * sizeof( glm::vec4 );

    // Output struct mimicking the shader's AgentHash
    struct AgentHash
    {
        uint32_t hash;
        uint32_t agentIndex;
    };

    std::vector<AgentHash> initialHashes( agentCount, { 0, 0 } );
    size_t                 hashesSize = agentCount * sizeof( AgentHash );

    // 2. Allocate & Upload Buffers
    BufferHandle agentBuffer = m_rm->CreateBuffer( { agentsSize, BufferType::STORAGE, "TestAgentBuffer" } );
    BufferHandle hashBuffer  = m_rm->CreateBuffer( { hashesSize, BufferType::STORAGE, "TestHashBuffer" } );

    std::vector<BufferUploadRequest> uploads = { { agentBuffer, agents.data(), agentsSize, 0 }, { hashBuffer, initialHashes.data(), hashesSize, 0 } };
    m_stream->UploadBufferImmediate( uploads );

    // 3. Setup Pipeline & BindingGroup directly via RHI
    ComputePipelineDesc pipeDesc{};
    pipeDesc.shader                  = m_rm->CreateShader( "shaders/compute/hash_agents.comp" );
    pipeDesc.debugName               = "TestHashAgents";
    ComputePipelineHandle pipeHandle = m_rm->CreatePipeline( pipeDesc );
    ComputePipeline*      pipeline   = m_rm->GetPipeline( pipeHandle );

    BindingGroupHandle bgHandle = m_rm->CreateBindingGroup( pipeHandle, 0 );
    BindingGroup*      bg       = m_rm->GetBindingGroup( bgHandle );
    bg->Bind( 0, m_rm->GetBuffer( agentBuffer ) );
    bg->Bind( 1, m_rm->GetBuffer( hashBuffer ) );
    bg->Build();

    // 4. Setup Push Constants
    float                cellSize = 2.0f; // Each virtual bounding box is 2.0 x 2.0 x 2.0 units
    ComputePushConstants pc{};
    pc.dt          = 1.0f;
    pc.fParam0     = cellSize;
    pc.maxCapacity = agentCount;
    pc.offset      = 0;

    // 5. Dispatch Manually
    auto compCtxHandle = m_device->CreateThreadContext( QueueType::COMPUTE );
    auto compCtx       = m_device->GetThreadContext( compCtxHandle );
    auto compCmd       = compCtx->GetCommandBuffer( compCtx->CreateCommandBuffer() );

    compCmd->Begin();
    compCmd->SetPipeline( pipeline );
    compCmd->SetBindingGroup( bg, pipeline->GetLayout(), VK_PIPELINE_BIND_POINT_COMPUTE );
    compCmd->PushConstants( pipeline->GetLayout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof( pc ), &pc );

    // We use a 1D local_size of 256 for linear agent buffers
    compCmd->Dispatch( ( agentCount + 255 ) / 256, 1, 1 );
    compCmd->End();

    m_device->GetComputeQueue()->Submit( { compCmd } );
    m_device->GetComputeQueue()->WaitIdle();

    // 6. Readback and Assert
    std::vector<AgentHash> resultHashes( agentCount );
    m_stream->ReadbackBufferImmediate( hashBuffer, resultHashes.data(), hashesSize );

    // C++ implementation of the exact same hash function used in GLSL
    auto expectedHashLogic = []( glm::vec3 pos, float cSize ) -> uint32_t {
        int cx = static_cast<int>( std::floor( pos.x / cSize ) );
        int cy = static_cast<int>( std::floor( pos.y / cSize ) );
        int cz = static_cast<int>( std::floor( pos.z / cSize ) );

        const uint32_t p1 = 73856093;
        const uint32_t p2 = 19349663;
        const uint32_t p3 = 83492791;

        // Using static_cast to uint32_t preserves the bit pattern of negative numbers for bitwise XOR
        return ( static_cast<uint32_t>( cx ) * p1 ) ^ ( static_cast<uint32_t>( cy ) * p2 ) ^ ( static_cast<uint32_t>( cz ) * p3 );
    };

    // Verify GPU results against CPU expectations
    for( uint32_t i = 0; i < agentCount; ++i )
    {
        EXPECT_EQ( resultHashes[ i ].agentIndex, i ) << "Shader failed to write the correct agent index!";

        uint32_t expectedHash = expectedHashLogic( agents[ i ], cellSize );
        EXPECT_EQ( resultHashes[ i ].hash, expectedHash ) << "GPU calculated Hash does not match CPU for Agent " << i;
    }

    // Cleanup
    m_rm->DestroyBuffer( agentBuffer );
    m_rm->DestroyBuffer( hashBuffer );
}

// 4. Compute spatial hashing offsets generation test
TEST_F( ComputeTest, Shader_BuildOffsets_Logic )
{
    if( !m_device )
        GTEST_SKIP();

    // 1. Prepare raw data simulating an already SORTED hash array
    // We intentionally group agents to see if the shader correctly identifies the STARTING index of each group
    struct AgentHash
    {
        uint32_t hash;
        uint32_t agentIndex;
    };

    std::vector<AgentHash> sortedHashes = {
        { 15, 2 }, // Index 0: Hash 15 starts here
        { 15, 0 }, // Index 1: Hash 15 continues
        { 15, 5 }, // Index 2: Hash 15 continues
        { 42, 1 }, // Index 3: Hash 42 starts here
        { 42, 4 }, // Index 4: Hash 42 continues
        { 105, 3 } // Index 5: Hash 105 starts here
    };

    uint32_t agentCount = static_cast<uint32_t>( sortedHashes.size() );
    size_t   hashesSize = agentCount * sizeof( AgentHash );

    // Offset array (Dictionary) initialized to 0xFFFFFFFF (Empty marker)
    // We use a small array size for the test, but the shader uses modulo (%) to fit any hash
    uint32_t              offsetArraySize = 256;
    std::vector<uint32_t> initialOffsets( offsetArraySize, 0xFFFFFFFF );
    size_t                offsetsSize = offsetArraySize * sizeof( uint32_t );

    // 2. Allocate & Upload Buffers
    BufferHandle hashBuffer   = m_rm->CreateBuffer( { hashesSize, BufferType::STORAGE, "TestSortedHashes" } );
    BufferHandle offsetBuffer = m_rm->CreateBuffer( { offsetsSize, BufferType::STORAGE, "TestCellOffsets" } );

    std::vector<BufferUploadRequest> uploads = { { hashBuffer, sortedHashes.data(), hashesSize, 0 },
                                                 { offsetBuffer, initialOffsets.data(), offsetsSize, 0 } };
    m_stream->UploadBufferImmediate( uploads );

    // 3. Setup Pipeline & BindingGroup directly via RHI
    ComputePipelineDesc pipeDesc{};
    pipeDesc.shader                  = m_rm->CreateShader( "shaders/compute/build_offsets.comp" );
    pipeDesc.debugName               = "TestBuildOffsets";
    ComputePipelineHandle pipeHandle = m_rm->CreatePipeline( pipeDesc );
    ComputePipeline*      pipeline   = m_rm->GetPipeline( pipeHandle );

    BindingGroupHandle bgHandle = m_rm->CreateBindingGroup( pipeHandle, 0 );
    BindingGroup*      bg       = m_rm->GetBindingGroup( bgHandle );
    bg->Bind( 0, m_rm->GetBuffer( hashBuffer ) );
    bg->Bind( 1, m_rm->GetBuffer( offsetBuffer ) );
    bg->Build();

    // 4. Setup Push Constants
    ComputePushConstants pc{};
    pc.dt          = 1.0f;
    pc.maxCapacity = agentCount;
    pc.offset      = 0;

    // 5. Dispatch Manually
    auto compCtxHandle = m_device->CreateThreadContext( QueueType::COMPUTE );
    auto compCtx       = m_device->GetThreadContext( compCtxHandle );
    auto compCmd       = compCtx->GetCommandBuffer( compCtx->CreateCommandBuffer() );

    compCmd->Begin();
    compCmd->SetPipeline( pipeline );
    compCmd->SetBindingGroup( bg, pipeline->GetLayout(), VK_PIPELINE_BIND_POINT_COMPUTE );
    compCmd->PushConstants( pipeline->GetLayout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof( pc ), &pc );

    // We use a 1D local_size of 256 for linear agent buffers
    compCmd->Dispatch( ( agentCount + 255 ) / 256, 1, 1 );
    compCmd->End();

    m_device->GetComputeQueue()->Submit( { compCmd } );
    m_device->GetComputeQueue()->WaitIdle();

    // 6. Readback and Assert
    std::vector<uint32_t> resultOffsets( offsetArraySize );
    m_stream->ReadbackBufferImmediate( offsetBuffer, resultOffsets.data(), offsetsSize );

    // Verify GPU results against expected logic
    // Hash 15 started at index 0
    EXPECT_EQ( resultOffsets[ 15 % offsetArraySize ], 0 ) << "Offsets Builder failed to map Hash 15 to index 0!";

    // Hash 42 started at index 3
    EXPECT_EQ( resultOffsets[ 42 % offsetArraySize ], 3 ) << "Offsets Builder failed to map Hash 42 to index 3!";

    // Hash 105 started at index 5
    EXPECT_EQ( resultOffsets[ 105 % offsetArraySize ], 5 ) << "Offsets Builder failed to map Hash 105 to index 5!";

    // Check a random untouched cell to ensure it remains 0xFFFFFFFF
    EXPECT_EQ( resultOffsets[ 10 % offsetArraySize ], 0xFFFFFFFF ) << "Offsets Builder overwrote an empty cell!";
    EXPECT_EQ( resultOffsets[ 50 % offsetArraySize ], 0xFFFFFFFF ) << "Offsets Builder overwrote an empty cell!";

    // Cleanup
    m_rm->DestroyBuffer( hashBuffer );
    m_rm->DestroyBuffer( offsetBuffer );
}

// 5. Compute bitonic sort algorithm test
TEST_F( ComputeTest, Shader_BitonicSort_Logic )
{
    if( !m_device )
        GTEST_SKIP();

    // 1. Prepare intentionally unsorted struct data (Hash, Index)
    // IMPORTANT: GPU Bitonic Sort requires the array size to be a power of two (e.g., 8 elements)
    struct AgentHash
    {
        uint32_t hash;
        uint32_t agentIndex;
    };

    std::vector<AgentHash> rawData = { { 55, 0 }, { 12, 1 }, { 89, 2 }, { 1, 3 }, { 42, 4 }, { 33, 5 }, { 99, 6 }, { 7, 7 } };

    uint32_t count      = static_cast<uint32_t>( rawData.size() );
    size_t   hashesSize = count * sizeof( AgentHash );

    // 2. Allocate & Upload Buffers
    BufferHandle hashBuffer = m_rm->CreateBuffer( { hashesSize, BufferType::STORAGE, "TestSortBuffer" } );

    std::vector<BufferUploadRequest> uploads = { { hashBuffer, rawData.data(), hashesSize, 0 } };
    m_stream->UploadBufferImmediate( uploads );

    // 3. Setup Pipeline & BindingGroup directly via RHI
    ComputePipelineDesc pipeDesc{};
    pipeDesc.shader                  = m_rm->CreateShader( "shaders/compute/bitonic_sort.comp" );
    pipeDesc.debugName               = "TestBitonicSort";
    ComputePipelineHandle pipeHandle = m_rm->CreatePipeline( pipeDesc );
    ComputePipeline*      pipeline   = m_rm->GetPipeline( pipeHandle );

    BindingGroupHandle bgHandle = m_rm->CreateBindingGroup( pipeHandle, 0 );
    BindingGroup*      bg       = m_rm->GetBindingGroup( bgHandle );
    bg->Bind( 0, m_rm->GetBuffer( hashBuffer ) );
    bg->Build();

    // 4. Setup Push Constants (Base)
    ComputePushConstants pc{};
    pc.dt          = 1.0f;
    pc.maxCapacity = count;
    pc.offset      = 0;

    // 5. Dispatch Manually (The Bitonic Sort Execution Loop)
    auto compCtxHandle = m_device->CreateThreadContext( QueueType::COMPUTE );
    auto compCtx       = m_device->GetThreadContext( compCtxHandle );
    auto compCmd       = compCtx->GetCommandBuffer( compCtx->CreateCommandBuffer() );

    compCmd->Begin();
    compCmd->SetPipeline( pipeline );
    compCmd->SetBindingGroup( bg, pipeline->GetLayout(), VK_PIPELINE_BIND_POINT_COMPUTE );

    // Memory barrier struct used to synchronize compute shader passes
    VkMemoryBarrier2 barrier = { VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
    barrier.srcStageMask     = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    barrier.srcAccessMask    = VK_ACCESS_2_SHADER_WRITE_BIT;
    barrier.dstStageMask     = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    barrier.dstAccessMask    = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT;

    VkDependencyInfo depInfo   = { VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
    depInfo.memoryBarrierCount = 1;
    depInfo.pMemoryBarriers    = &barrier;

    // The Magic GPU Loop: Outer loop determines the sorted sequence length, inner loop merges
    for( uint32_t k = 2; k <= count; k <<= 1 )
    {
        for( uint32_t j = k >> 1; j > 0; j >>= 1 )
        {
            pc.uParam0 = j;
            pc.uParam1 = k;

            compCmd->PushConstants( pipeline->GetLayout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof( pc ), &pc );

            // Dispatch 1 thread group (since count is 8, it easily fits in a single 256 local_size block)
            compCmd->Dispatch( ( count + 255 ) / 256, 1, 1 );

            // WE MUST WAIT for the current sort step to finish writing before the next step reads!
            compCmd->PipelineBarrier( &depInfo );
        }
    }

    compCmd->End();

    m_device->GetComputeQueue()->Submit( { compCmd } );
    m_device->GetComputeQueue()->WaitIdle();

    // 6. Readback and Assert
    std::vector<AgentHash> resultData( count );
    m_stream->ReadbackBufferImmediate( hashBuffer, resultData.data(), hashesSize );

    // Assert that the 'hash' values are in perfectly ascending order!
    // Expected order of hashes: 1, 7, 12, 33, 42, 55, 89, 99
    for( size_t i = 1; i < count; ++i )
    {
        EXPECT_LE( resultData[ i - 1 ].hash, resultData[ i ].hash ) << "GPU Sort failed at index " << i << "!";
    }

    // Cleanup
    m_rm->DestroyBuffer( hashBuffer );
}

// 6. Compute JKR forces (Biomechanics) logic test
TEST_F( ComputeTest, Shader_JKRForces_Logic )
{
    if( !m_device )
        GTEST_SKIP();

    // 1. Prepare raw data for 2 Agents overlapping in space
    std::vector<glm::vec4> inAgents = {
        glm::vec4( 0.0f, 0.0f, 0.0f, 1.0f ), // Agent 0 at origin
        glm::vec4( 0.1f, 0.0f, 0.0f, 1.0f )  // Agent 1 slightly to the right
    };

    uint32_t agentCount = static_cast<uint32_t>( inAgents.size() );
    size_t   agentsSize = agentCount * sizeof( glm::vec4 );
    size_t   countSize  = agentCount * sizeof( uint32_t );

    // Output buffers for positions and pressures
    std::vector<glm::vec4> outAgents( agentCount, glm::vec4( 0.0f ) );
    std::vector<float>     outPressures( agentCount, 0.0f );
    size_t                 pressuresSize = agentCount * sizeof( float );

    // Prepare Sorted Hashes
    // Both agents are at ~0.0, so with a cellSize of 3.0 (maxRadius * 2), they are in cell (0,0,0)
    // Hash for (0,0,0) is 0.
    struct AgentHash
    {
        uint32_t hash;
        uint32_t agentIndex;
    };
    std::vector<AgentHash> sortedHashes = {
        { 0, 0 }, // Agent 0 in Hash 0
        { 0, 1 }  // Agent 1 in Hash 0
    };
    size_t hashesSize = agentCount * sizeof( AgentHash );

    // Prepare Offsets Dictionary
    uint32_t              offsetArraySize = 256;
    std::vector<uint32_t> cellOffsets( offsetArraySize, 0xFFFFFFFF );
    cellOffsets[ 0 ]   = 0; // Hash 0 starts at index 0 in the sortedHashes array
    size_t offsetsSize = offsetArraySize * sizeof( uint32_t );

    // 2. Allocate & Upload Buffers
    BufferHandle inAgentsBuf  = m_rm->CreateBuffer( { agentsSize, BufferType::STORAGE, "TestInAgents" } );
    BufferHandle outAgentsBuf = m_rm->CreateBuffer( { agentsSize, BufferType::STORAGE, "TestOutAgents" } );
    BufferHandle pressuresBuf = m_rm->CreateBuffer( { pressuresSize, BufferType::STORAGE, "TestPressures" } );
    BufferHandle hashesBuf    = m_rm->CreateBuffer( { hashesSize, BufferType::STORAGE, "TestSortedHashes" } );
    BufferHandle offsetsBuf   = m_rm->CreateBuffer( { offsetsSize, BufferType::STORAGE, "TestOffsets" } );
    BufferHandle countBuf     = m_rm->CreateBuffer( { countSize, BufferType::STORAGE, "TestCountBuffer" } );

    std::vector<BufferUploadRequest> uploads = { { inAgentsBuf, inAgents.data(), agentsSize, 0 },
                                                 { hashesBuf, sortedHashes.data(), hashesSize, 0 },
                                                 { offsetsBuf, cellOffsets.data(), offsetsSize, 0 },
                                                 { countBuf, &agentCount, countSize, 0 } };
    m_stream->UploadBufferImmediate( uploads );

    // 3. Setup Pipeline & BindingGroup directly via RHI
    ComputePipelineDesc pipeDesc{};
    pipeDesc.shader                  = m_rm->CreateShader( "shaders/compute/jkr_forces.comp" );
    pipeDesc.debugName               = "TestJKRForces";
    ComputePipelineHandle pipeHandle = m_rm->CreatePipeline( pipeDesc );
    ComputePipeline*      pipeline   = m_rm->GetPipeline( pipeHandle );

    // Dummy phenotype buffer for binding 6 (state filtering disabled via push constants)
    BufferHandle phenotypeDummyBuf = m_rm->CreateBuffer( { sizeof( uint32_t ) * 4, BufferType::STORAGE, "TestPhenotypeDummy" } );

    BindingGroupHandle bgHandle = m_rm->CreateBindingGroup( pipeHandle, 0 );
    BindingGroup*      bg       = m_rm->GetBindingGroup( bgHandle );
    bg->Bind( 0, m_rm->GetBuffer( inAgentsBuf ) );
    bg->Bind( 1, m_rm->GetBuffer( outAgentsBuf ) );
    bg->Bind( 2, m_rm->GetBuffer( pressuresBuf ) );
    bg->Bind( 3, m_rm->GetBuffer( hashesBuf ) );
    bg->Bind( 4, m_rm->GetBuffer( offsetsBuf ) );
    bg->Bind( 5, m_rm->GetBuffer( countBuf ) );
    bg->Bind( 6, m_rm->GetBuffer( phenotypeDummyBuf ) );
    bg->Build();

    // 4. Setup Push Constants (Packed exactly as in SimulationBuilder)
    ComputePushConstants pc{};
    pc.dt          = 0.016f; // Standard 60Hz frame time
    pc.maxCapacity = agentCount;
    pc.offset      = 0;
    pc.fParam0     = 50.0f;  // Repulsion Stiffness
    pc.fParam1     = 0.0f;   // Adhesion Strength (ignore for this test)
    pc.fParam2     = -1.0f;  // reqLC = -1 (no filtering)
    pc.fParam3     = -1.0f;  // reqCT = -1 (no filtering)
    pc.uParam0     = offsetArraySize;
    pc.uParam1     = 0;
    // Pack maxRadius into domainSize.w
    pc.domainSize = glm::vec4( 100.0f, 100.0f, 100.0f, 1.5f );

    // 5. Dispatch Manually
    auto compCtxHandle = m_device->CreateThreadContext( QueueType::COMPUTE );
    auto compCtx       = m_device->GetThreadContext( compCtxHandle );
    auto compCmd       = compCtx->GetCommandBuffer( compCtx->CreateCommandBuffer() );

    compCmd->Begin();
    compCmd->SetPipeline( pipeline );
    compCmd->SetBindingGroup( bg, pipeline->GetLayout(), VK_PIPELINE_BIND_POINT_COMPUTE );
    compCmd->PushConstants( pipeline->GetLayout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof( pc ), &pc );

    compCmd->Dispatch( ( agentCount + 255 ) / 256, 1, 1 );
    compCmd->End();

    m_device->GetComputeQueue()->Submit( { compCmd } );
    m_device->GetComputeQueue()->WaitIdle();

    // 6. Readback and Assert
    std::vector<glm::vec4> resultAgents( agentCount );
    std::vector<float>     resultPressures( agentCount );

    m_stream->ReadbackBufferImmediate( outAgentsBuf, resultAgents.data(), agentsSize );
    m_stream->ReadbackBufferImmediate( pressuresBuf, resultPressures.data(), pressuresSize );

    // Verify Physics Logic:
    // Agent 0 should be pushed to the left (negative X) by Agent 1
    EXPECT_LT( resultAgents[ 0 ].x, 0.0f ) << "Agent 0 was not repelled correctly to the left!";
    // Agent 1 should be pushed to the right (positive X) by Agent 0 (greater than its initial 0.1f)
    EXPECT_GT( resultAgents[ 1 ].x, 0.1f ) << "Agent 1 was not repelled correctly to the right!";

    // Both agents should experience identical mechanical pressure from the collision
    EXPECT_GT( resultPressures[ 0 ], 0.0f ) << "Agent 0 did not register collision pressure!";
    EXPECT_GT( resultPressures[ 1 ], 0.0f ) << "Agent 1 did not register collision pressure!";
    EXPECT_FLOAT_EQ( resultPressures[ 0 ], resultPressures[ 1 ] ) << "Newton's Third Law violated: Pressures unequal!";

    // Cleanup
    m_rm->DestroyBuffer( inAgentsBuf );
    m_rm->DestroyBuffer( outAgentsBuf );
    m_rm->DestroyBuffer( pressuresBuf );
    m_rm->DestroyBuffer( hashesBuf );
    m_rm->DestroyBuffer( offsetsBuf );
    m_rm->DestroyBuffer( countBuf );
}

// 7. Compute fenotype update test
TEST_F( ComputeTest, Shader_Biology_UpdatePhenotype )
{
    if( !m_device )
        GTEST_SKIP();

    // 1. Setup 3 agents
    // Agent 0: High pressure (Should become Quiescent)
    // Agent 1: Low O2 (Should become Necrotic)
    // Agent 2: Perfect conditions (Should Grow)
    uint32_t agentCount = 3;
    size_t   agentsSize = agentCount * sizeof( glm::vec4 );
    size_t   countSize  = sizeof( uint32_t );

    std::vector<glm::vec4> agents = { glm::vec4( 0.0f, 0.0f, 0.0f, 1.0f ), glm::vec4( 1.0f, 0.0f, 0.0f, 1.0f ), glm::vec4( 2.0f, 0.0f, 0.0f, 1.0f ) };

    std::vector<float> pressures     = { 5.0f, 0.0f, 0.0f }; // Agent 0 has 5.0 MPa pressure
    size_t             pressuresSize = agentCount * sizeof( float );

    struct PhenotypeData
    {
        uint32_t lifecycleState;
        float    biomass;
        float    timer;
        uint32_t cellType;
    };
    std::vector<PhenotypeData> phenotypes     = { { 0, 0.5f, 0.0f, 0 }, { 0, 0.5f, 0.0f, 0 }, { 0, 0.5f, 0.0f, 0 } };
    size_t                     phenotypesSize = agentCount * sizeof( PhenotypeData );

    // 2. Allocate & Upload Buffers
    BufferHandle agentsBuf    = m_rm->CreateBuffer( { agentsSize, BufferType::STORAGE, "TestAgents" } );
    BufferHandle pressuresBuf = m_rm->CreateBuffer( { pressuresSize, BufferType::STORAGE, "TestPressures" } );
    BufferHandle phenotypeBuf = m_rm->CreateBuffer( { phenotypesSize, BufferType::STORAGE, "TestPhenotypes" } );
    BufferHandle countBuf     = m_rm->CreateBuffer( { countSize, BufferType::INDIRECT, "TestCounter" } );

    m_stream->UploadBufferImmediate( { { agentsBuf, agents.data(), agentsSize, 0 },
                                       { pressuresBuf, pressures.data(), pressuresSize, 0 },
                                       { phenotypeBuf, phenotypes.data(), phenotypesSize, 0 },
                                       { countBuf, &agentCount, countSize, 0 } } );

    DigitalTwin::SimulationBlueprint dummyBp;
    dummyBp.SetDomainSize( glm::vec3( 10.0f ), 1.0f );
    dummyBp.AddGridField( "Oxygen" );
    DigitalTwin::SimulationBuilder dummyBuilder( m_rm.get(), m_stream.get() );
    DigitalTwin::SimulationState   dummyState = dummyBuilder.Build( dummyBp );

    // 3. Setup Pipeline & BindingGroup
    ComputePipelineDesc   pipeDesc{ m_rm->CreateShader( "shaders/compute/biology/update_phenotype.comp" ), "UpdatePhenotype" };
    ComputePipelineHandle pipelineHandle = m_rm->CreatePipeline( pipeDesc );
    ComputePipeline*      pipeline       = m_rm->GetPipeline( pipelineHandle );

    BindingGroup* bg = m_rm->GetBindingGroup( m_rm->CreateBindingGroup( pipelineHandle, 0 ) );
    bg->Bind( 0, m_rm->GetBuffer( agentsBuf ) );
    bg->Bind( 1, m_rm->GetBuffer( pressuresBuf ) );
    bg->Bind( 2, m_rm->GetTexture( dummyState.gridFields[ 0 ].textures[ 0 ] ), VK_IMAGE_LAYOUT_GENERAL );
    bg->Bind( 3, m_rm->GetBuffer( phenotypeBuf ) );
    bg->Bind( 4, m_rm->GetBuffer( countBuf ) );
    bg->Build();

    // 4. Dispatch Pass 1: Perfect O2 (Tests Pressure and Growth)
    ComputePushConstants pc{};
    pc.dt          = 1.0f;
    pc.fParam0     = 0.2f;  // growthRate: 0.2
    pc.fParam1     = 38.0f; // targetO2: 38 mmHg (Fallback value used)
    pc.fParam2     = 2.5f;  // arrestPressure: 2.5 MPa
    pc.fParam3     = 5.0f;  // necrosisO2: 5.0 mmHg
    pc.maxCapacity = agentCount;
    pc.uParam1     = 0;
    pc.gridSize    = glm::uvec4( 0 ); // Disable texture read

    auto compCtxHandle = m_device->CreateThreadContext( QueueType::COMPUTE );
    auto compCtx       = m_device->GetThreadContext( compCtxHandle );
    auto compCmd       = compCtx->GetCommandBuffer( compCtx->CreateCommandBuffer() );

    // Barrier between passes
    compCmd->Begin();
    compCmd->SetPipeline( pipeline );
    compCmd->SetBindingGroup( bg, pipeline->GetLayout(), VK_PIPELINE_BIND_POINT_COMPUTE );
    compCmd->PushConstants( pipeline->GetLayout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof( pc ), &pc );
    compCmd->Dispatch( 1, 1, 1 );
    compCmd->End();

    m_device->GetComputeQueue()->Submit( { compCmd } );
    m_device->GetComputeQueue()->WaitIdle();

    std::vector<PhenotypeData> pass1Result( agentCount );
    m_stream->ReadbackBufferImmediate( phenotypeBuf, pass1Result.data(), phenotypesSize );

    EXPECT_EQ( pass1Result[ 0 ].lifecycleState, 1 ) << "Agent 0 should be Quiescent (1) due to pressure!";
    EXPECT_EQ( pass1Result[ 1 ].lifecycleState, 0 ) << "Agent 1 should remain Live (0)!";
    EXPECT_FLOAT_EQ( pass1Result[ 2 ].biomass, 0.7f ) << "Agent 2 should have grown by 0.2!";

    // Dispatch Pass 2: Hypoxia (Tests Necrosis on Agent 1)
    pc.fParam1 = 2.0f; // Mock local O2 to deadly levels globally

    compCmd->Begin();
    compCmd->SetPipeline( pipeline );
    compCmd->SetBindingGroup( bg, pipeline->GetLayout(), VK_PIPELINE_BIND_POINT_COMPUTE );
    compCmd->PushConstants( pipeline->GetLayout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof( pc ), &pc );
    compCmd->Dispatch( 1, 1, 1 );
    compCmd->End();

    m_device->GetComputeQueue()->Submit( { compCmd } );
    m_device->GetComputeQueue()->WaitIdle();

    std::vector<PhenotypeData> pass2Result( agentCount );
    m_stream->ReadbackBufferImmediate( phenotypeBuf, pass2Result.data(), phenotypesSize );

    EXPECT_EQ( pass2Result[ 1 ].lifecycleState, 4 ) << "Agent 1 should be Necrotic (4) due to Hypoxia!";

    // 6. Cleanup
    m_rm->DestroyBuffer( agentsBuf );
    m_rm->DestroyBuffer( pressuresBuf );
    m_rm->DestroyBuffer( phenotypeBuf );
    m_rm->DestroyBuffer( countBuf );

    dummyState.Destroy( m_rm.get() );
}

// 8. Mitosis shader test
TEST_F( ComputeTest, Shader_Biology_MitosisAppend )
{
    if( !m_device )
        GTEST_SKIP();

    // 1. Prepare data for Mitosis (Buffer capacity = 4, starting count = 1)
    uint32_t maxCapacity = 4;
    size_t   agentsSize  = maxCapacity * sizeof( glm::vec4 );

    // Fill with empty slots (w = 0.0), set Mother Cell at index 0
    std::vector<glm::vec4> agents( maxCapacity, glm::vec4( 0.0f ) );
    agents[ 0 ] = glm::vec4( 10.0f, 10.0f, 10.0f, 1.0f ); // Mother (w=1.0)

    struct PhenotypeData
    {
        uint32_t lifecycleState;
        float    biomass;
        float    timer;
        uint32_t cellType;
    };
    size_t                     phenotypesSize = maxCapacity * sizeof( PhenotypeData );
    std::vector<PhenotypeData> phenotypes( maxCapacity, { 0, 0.0f, 0.0f, 0 } );
    phenotypes[ 0 ] = { 0, 1.1f, 0.0f, 0 }; // Mother is Live and ready to divide (biomass > 1.0)

    uint32_t agentCount = 1; // Counter starts at 1
    size_t   countSize  = sizeof( uint32_t );

    // 2. Allocate & Upload Buffers
    BufferHandle agentsReadBuf  = m_rm->CreateBuffer( { agentsSize, BufferType::STORAGE, "TestAgentsRead" } );
    BufferHandle agentsWriteBuf = m_rm->CreateBuffer( { agentsSize, BufferType::STORAGE, "TestAgentsWrite" } );
    BufferHandle phenotypeBuf   = m_rm->CreateBuffer( { phenotypesSize, BufferType::STORAGE, "TestPhenotypes" } );
    BufferHandle countBuf       = m_rm->CreateBuffer( { countSize, BufferType::STORAGE, "TestCounter" } );

    m_stream->UploadBufferImmediate( { { agentsReadBuf, agents.data(), agentsSize, 0 },
                                       { agentsWriteBuf, agents.data(), agentsSize, 0 },
                                       { phenotypeBuf, phenotypes.data(), phenotypesSize, 0 },
                                       { countBuf, &agentCount, countSize, 0 } } );

    // 3. Setup Pipeline & BindingGroup
    ComputePipelineDesc   pipeDesc{ m_rm->CreateShader( "shaders/compute/biology/mitosis_append.comp" ), "MitosisAppend" };
    ComputePipelineHandle pipelineHandle = m_rm->CreatePipeline( pipeDesc );
    ComputePipeline*      pipeline       = m_rm->GetPipeline( m_rm->CreatePipeline( pipeDesc ) );

    BindingGroup* bg = m_rm->GetBindingGroup( m_rm->CreateBindingGroup( pipelineHandle, 0 ) );
    bg->Bind( 0, m_rm->GetBuffer( agentsReadBuf ) );
    bg->Bind( 1, m_rm->GetBuffer( agentsWriteBuf ) );
    bg->Bind( 2, m_rm->GetBuffer( phenotypeBuf ) );
    bg->Bind( 3, m_rm->GetBuffer( countBuf ) );
    bg->Build();

    // 4. Dispatch
    ComputePushConstants pc{};
    pc.totalTime   = 1.0f; // Used for RNG
    pc.maxCapacity = maxCapacity;
    pc.uParam1     = 0; // Group index 0

    auto compCtxHandle = m_device->CreateThreadContext( QueueType::COMPUTE );
    auto compCtx       = m_device->GetThreadContext( compCtxHandle );
    auto compCmd       = compCtx->GetCommandBuffer( compCtx->CreateCommandBuffer() );

    compCmd->Begin();
    compCmd->SetPipeline( pipeline );
    compCmd->SetBindingGroup( bg, pipeline->GetLayout(), VK_PIPELINE_BIND_POINT_COMPUTE );
    compCmd->PushConstants( pipeline->GetLayout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof( pc ), &pc );
    compCmd->Dispatch( 1, 1, 1 );
    compCmd->End();

    m_device->GetComputeQueue()->Submit( { compCmd } );
    m_device->GetComputeQueue()->WaitIdle();

    // 5. Assertions
    std::vector<glm::vec4>     resAgents( maxCapacity );
    std::vector<PhenotypeData> resPheno( maxCapacity );
    uint32_t                   resCount = 0;

    m_stream->ReadbackBufferImmediate( agentsReadBuf, resAgents.data(), agentsSize );
    m_stream->ReadbackBufferImmediate( phenotypeBuf, resPheno.data(), phenotypesSize );
    m_stream->ReadbackBufferImmediate( countBuf, &resCount, countSize );

    EXPECT_EQ( resCount, 2 ) << "Atomic counter should have incremented to 2!";

    EXPECT_FLOAT_EQ( resPheno[ 0 ].biomass, 0.5f ) << "Mother biomass should split to 0.5!";
    EXPECT_FLOAT_EQ( resPheno[ 1 ].biomass, 0.5f ) << "Daughter biomass should start at 0.5!";
    EXPECT_FLOAT_EQ( resAgents[ 1 ].w, 1.0f ) << "Daughter w-component must be 1.0 (Alive)!";

    m_rm->DestroyBuffer( agentsReadBuf );
    m_rm->DestroyBuffer( agentsWriteBuf );
    m_rm->DestroyBuffer( phenotypeBuf );
    m_rm->DestroyBuffer( countBuf );
}

// 9. Verify separation of concerns: Phenotype update triggers conditional Field Secretion
TEST_F( ComputeTest, Shader_Biology_ConditionalSecretion )
{
    if( !m_device )
        GTEST_SKIP();

    uint32_t maxCapacity = 1;
    size_t   agentsSize  = maxCapacity * sizeof( glm::vec4 );
    size_t   phenoSize   = maxCapacity * 4 * sizeof( uint32_t );
    size_t   countSize   = sizeof( uint32_t );
    size_t   deltaSize   = sizeof( int ); // Simulating 1 voxel

    std::vector<glm::vec4> agents( maxCapacity, glm::vec4( 0.0f, 0.0f, 0.0f, 1.0f ) );
    std::vector<uint32_t>  phenoInit( maxCapacity * 4, 0 ); // state 0 (Live)
    phenoInit[ 0 ]              = 0;                        // State
    uint32_t         agentCount = 1;
    std::vector<int> deltas( 1, 0 );

    // We mock pressure to 0.0
    std::vector<float> pressures( maxCapacity, 0.0f );

    BufferHandle  agentsBuf    = m_rm->CreateBuffer( { agentsSize, BufferType::STORAGE, "TestAgents" } );
    BufferHandle  phenoBuf     = m_rm->CreateBuffer( { phenoSize, BufferType::STORAGE, "TestPheno" } );
    BufferHandle  countBuf     = m_rm->CreateBuffer( { countSize, BufferType::STORAGE, "TestCount" } );
    BufferHandle  deltaBuf     = m_rm->CreateBuffer( { deltaSize, BufferType::STORAGE, "TestDelta" } );
    BufferHandle  pressuresBuf = m_rm->CreateBuffer( { pressures.size() * sizeof( float ), BufferType::STORAGE, "TestPressures" } );
    TextureHandle dummyTex     = m_rm->CreateTexture( { 1, 1, 1, TextureType::Texture3D, VK_FORMAT_R32_SFLOAT, TextureUsage::STORAGE, "Dummy" } );

    m_stream->UploadBufferImmediate( { { agentsBuf, agents.data(), agentsSize, 0 },
                                       { phenoBuf, phenoInit.data(), phenoSize, 0 },
                                       { countBuf, &agentCount, countSize, 0 },
                                       { deltaBuf, deltas.data(), deltaSize, 0 },
                                       { pressuresBuf, pressures.data(), pressures.size() * sizeof( float ), 0 } } );

    // Pipeline 1: Update Phenotype
    ComputePipelineHandle pipePhenoHandle =
        m_rm->CreatePipeline( { m_rm->CreateShader( "shaders/compute/biology/update_phenotype.comp" ), "Pheno" } );
    ComputePipeline* pipePheno = m_rm->GetPipeline( pipePhenoHandle );
    BindingGroup*    bgPheno   = m_rm->GetBindingGroup( m_rm->CreateBindingGroup( pipePhenoHandle, 0 ) );
    bgPheno->Bind( 0, m_rm->GetBuffer( agentsBuf ) );
    bgPheno->Bind( 1, m_rm->GetBuffer( pressuresBuf ) );
    bgPheno->Bind( 2, m_rm->GetTexture( dummyTex ), VK_IMAGE_LAYOUT_GENERAL );
    bgPheno->Bind( 3, m_rm->GetBuffer( phenoBuf ) );
    bgPheno->Bind( 4, m_rm->GetBuffer( countBuf ) );
    bgPheno->Build();

    // Pipeline 2: Field Interaction (Secretion)
    ComputePipelineHandle pipeInteractHandle = m_rm->CreatePipeline( { m_rm->CreateShader( "shaders/compute/field_interaction.comp" ), "Interact" } );
    ComputePipeline*      pipeInteract       = m_rm->GetPipeline( pipeInteractHandle );
    BindingGroup*         bgInteract         = m_rm->GetBindingGroup( m_rm->CreateBindingGroup( pipeInteractHandle, 0 ) );
    bgInteract->Bind( 0, m_rm->GetBuffer( agentsBuf ) );
    bgInteract->Bind( 1, m_rm->GetBuffer( deltaBuf ) );
    bgInteract->Bind( 2, m_rm->GetBuffer( countBuf ) );
    bgInteract->Bind( 3, m_rm->GetBuffer( phenoBuf ) );
    bgInteract->Build();

    auto compCtxHandle = m_device->CreateThreadContext( QueueType::COMPUTE );
    auto compCtx       = m_device->GetThreadContext( compCtxHandle );
    auto compCmd       = compCtx->GetCommandBuffer( compCtx->CreateCommandBuffer() );

    // Step 1: Execute Phenotype Update to force Hypoxia (Local O2 is mocked via fallback)
    ComputePushConstants pcPheno{};
    pcPheno.maxCapacity = 1;
    pcPheno.fParam1     = 10.0f;           // Target/Local O2
    pcPheno.fParam2     = 20.0f;           // ArrestPressure
    pcPheno.fParam3     = 5.0f;            // Necrosis threshold
    pcPheno.fParam4     = 15.0f;           // Hypoxia threshold (O2 10.0 < 15.0 -> State should become 4)
    pcPheno.fParam5     = 0.0f;            // ApoptosisProb
    pcPheno.gridSize    = glm::uvec4( 0 ); // Bypass texture

    // Step 2: Execute Secretion requiring State 4
    ComputePushConstants pcInteract{};
    pcInteract.dt          = 1.0f;
    pcInteract.maxCapacity = 1;
    pcInteract.fParam0     = 5.0f;            // Secrete 5.0 units/sec
    pcInteract.fParam1     = 2.0f;            // ONLY if State == 2 (Hypoxic)
    pcInteract.gridSize    = glm::uvec4( 0 ); // Force flatIdx = 0
    pcInteract.domainSize  = glm::vec4( 10.0f, 10.0f, 10.0f, 0.0f );

    compCmd->Begin();

    VkImageMemoryBarrier2 imgBarrier = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
    imgBarrier.srcStageMask          = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
    imgBarrier.srcAccessMask         = 0;
    imgBarrier.dstStageMask          = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    imgBarrier.dstAccessMask         = VK_ACCESS_2_SHADER_READ_BIT;
    imgBarrier.oldLayout             = VK_IMAGE_LAYOUT_UNDEFINED;
    imgBarrier.newLayout             = VK_IMAGE_LAYOUT_GENERAL;
    imgBarrier.image                 = m_rm->GetTexture( dummyTex )->GetHandle();
    imgBarrier.subresourceRange      = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

    VkDependencyInfo imgDepInfo        = { VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
    imgDepInfo.imageMemoryBarrierCount = 1;
    imgDepInfo.pImageMemoryBarriers    = &imgBarrier;
    compCmd->PipelineBarrier( &imgDepInfo );

    compCmd->SetPipeline( pipePheno );
    compCmd->SetBindingGroup( bgPheno, pipePheno->GetLayout(), VK_PIPELINE_BIND_POINT_COMPUTE );
    compCmd->PushConstants( pipePheno->GetLayout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof( pcPheno ), &pcPheno );
    compCmd->Dispatch( 1, 1, 1 );

    // Barrier
    VkBufferMemoryBarrier2 barrier = { VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2 };
    barrier.srcStageMask           = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    barrier.srcAccessMask          = VK_ACCESS_2_SHADER_WRITE_BIT;
    barrier.dstStageMask           = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    barrier.dstAccessMask          = VK_ACCESS_2_SHADER_READ_BIT;
    barrier.buffer                 = m_rm->GetBuffer( phenoBuf )->GetHandle();
    barrier.offset                 = 0;
    barrier.size                   = VK_WHOLE_SIZE;

    VkDependencyInfo depInfo         = { VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
    depInfo.bufferMemoryBarrierCount = 1;
    depInfo.pBufferMemoryBarriers    = &barrier;
    compCmd->PipelineBarrier( &depInfo );

    compCmd->SetPipeline( pipeInteract );
    compCmd->SetBindingGroup( bgInteract, pipeInteract->GetLayout(), VK_PIPELINE_BIND_POINT_COMPUTE );
    compCmd->PushConstants( pipeInteract->GetLayout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof( pcInteract ), &pcInteract );
    compCmd->Dispatch( 1, 1, 1 );
    compCmd->End();

    m_device->GetComputeQueue()->Submit( { compCmd } );
    m_device->GetComputeQueue()->WaitIdle();

    std::vector<uint32_t> resPheno( 4 );
    std::vector<int>      resDelta( 1 );
    m_stream->ReadbackBufferImmediate( phenoBuf, resPheno.data(), phenoSize );
    m_stream->ReadbackBufferImmediate( deltaBuf, resDelta.data(), deltaSize );

    EXPECT_EQ( resPheno[ 0 ], 2 ) << "Cell failed to transition to Hypoxic state (2)!";
    EXPECT_EQ( resDelta[ 0 ], 500000 ) << "Conditional secretion failed! (Expected 5.0f * 100000)";

    m_rm->DestroyBuffer( agentsBuf );
    m_rm->DestroyBuffer( phenoBuf );
    m_rm->DestroyBuffer( countBuf );
    m_rm->DestroyBuffer( deltaBuf );
    m_rm->DestroyBuffer( pressuresBuf );
}

// =================================================================================================
// Chemotaxis shader tests
// =================================================================================================

// Verify that an agent moves in the direction of a linear X-axis gradient
TEST_F( ComputeTest, Chemotaxis_PureGradient_MovesInGradientDirection )
{
    if( !m_device )
        GTEST_SKIP();

    // 10x10x10 field with a linear X-axis gradient: field(x,y,z) = x * 1.0f
    uint32_t width = 10, height = 10, depth = 10;
    uint32_t voxelCount = width * height * depth;
    size_t   gridBytes  = voxelCount * sizeof( float );

    std::vector<float> fieldData( voxelCount );
    for( uint32_t z = 0; z < depth; ++z )
        for( uint32_t y = 0; y < height; ++y )
            for( uint32_t x = 0; x < width; ++x )
                fieldData[ x + y * width + z * width * height ] = static_cast<float>( x );

    TextureDesc texDesc{};
    texDesc.type      = TextureType::Texture3D;
    texDesc.width     = width;
    texDesc.height    = height;
    texDesc.depth     = depth;
    texDesc.format    = VK_FORMAT_R32_SFLOAT;
    texDesc.usage     = TextureUsage::STORAGE | TextureUsage::TRANSFER_DST | TextureUsage::TRANSFER_SRC;
    texDesc.debugName = "ChemotaxisFieldTex";
    TextureHandle fieldTex = m_rm->CreateTexture( texDesc );
    m_stream->UploadTextureImmediate( fieldTex, fieldData.data(), gridBytes );

    // Agent buffer: 1 agent at world origin (0,0,0), alive (w=1)
    size_t           agentBytes = sizeof( glm::vec4 );
    glm::vec4        agentIn    = glm::vec4( 0.0f, 0.0f, 0.0f, 1.0f );
    glm::vec4        agentOut   = glm::vec4( 0.0f );
    BufferHandle     inBuf      = m_rm->CreateBuffer( { agentBytes, BufferType::STORAGE, "ChemoInAgents" } );
    BufferHandle     outBuf     = m_rm->CreateBuffer( { agentBytes, BufferType::STORAGE, "ChemoOutAgents" } );
    m_stream->UploadBufferImmediate( { { inBuf, &agentIn, agentBytes } } );
    m_stream->UploadBufferImmediate( { { outBuf, &agentOut, agentBytes } } );

    // Count buffer: counts[0] = 1
    uint32_t     count    = 1;
    BufferHandle countBuf = m_rm->CreateBuffer( { sizeof( uint32_t ), BufferType::INDIRECT, "ChemoCountBuf" } );
    m_stream->UploadBufferImmediate( { { countBuf, &count, sizeof( uint32_t ) } } );

    ComputePipelineDesc   pipeDesc{};
    pipeDesc.shader                  = m_rm->CreateShader( "shaders/compute/chemotaxis.comp" );
    ComputePipelineHandle pipeHandle = m_rm->CreatePipeline( pipeDesc );
    ComputePipeline*      pipeline   = m_rm->GetPipeline( pipeHandle );

    // Dummy phenotype buffer for binding 4 (state filtering disabled via push constants)
    BufferHandle phenotypeDummyBuf = m_rm->CreateBuffer( { sizeof( uint32_t ) * 4, BufferType::STORAGE, "ChemoPhenotypeDummy" } );

    BindingGroup* bg = m_rm->GetBindingGroup( m_rm->CreateBindingGroup( pipeHandle, 0 ) );
    bg->Bind( 0, m_rm->GetBuffer( inBuf ) );
    bg->Bind( 1, m_rm->GetBuffer( outBuf ) );
    bg->Bind( 2, m_rm->GetTexture( fieldTex ), VK_IMAGE_LAYOUT_GENERAL );
    bg->Bind( 3, m_rm->GetBuffer( countBuf ) );
    bg->Bind( 4, m_rm->GetBuffer( phenotypeDummyBuf ) );
    bg->Build();

    // Domain 10x10x10, voxels 10x10x10, dt=1.0, sensitivity=1, saturation=0, maxVelocity=100
    ComputePushConstants pc{};
    pc.dt          = 1.0f;
    pc.fParam0     = 1.0f;   // sensitivity
    pc.fParam1     = 0.0f;   // saturation (linear)
    pc.fParam2     = 100.0f; // maxVelocity (no clamp)
    pc.fParam3     = -1.0f;  // reqCT = -1 (no filtering)
    pc.offset      = 0;
    pc.maxCapacity = 1;
    pc.uParam0     = static_cast<uint32_t>( -1 ); // reqLC = -1 (no filtering)
    pc.uParam1     = 0; // grpNdx
    pc.domainSize  = glm::vec4( 10.0f, 10.0f, 10.0f, 0.0f );
    pc.gridSize    = glm::uvec4( 10, 10, 10, 0 );

    auto compCtxHandle = m_device->CreateThreadContext( QueueType::COMPUTE );
    auto compCtx       = m_device->GetThreadContext( compCtxHandle );
    auto compCmd       = compCtx->GetCommandBuffer( compCtx->CreateCommandBuffer() );

    compCmd->Begin();
    compCmd->SetPipeline( pipeline );
    compCmd->SetBindingGroup( bg, pipeline->GetLayout(), VK_PIPELINE_BIND_POINT_COMPUTE );
    compCmd->PushConstants( pipeline->GetLayout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof( pc ), &pc );
    compCmd->Dispatch( 1, 1, 1 );
    compCmd->End();

    m_device->GetComputeQueue()->Submit( { compCmd } );
    m_device->GetComputeQueue()->WaitIdle();

    glm::vec4 result;
    m_stream->ReadbackBufferImmediate( outBuf, &result, sizeof( glm::vec4 ) );

    // Agent started at x=0; linear X gradient → must have moved in +X direction
    EXPECT_GT( result.x, 0.0f ) << "Chemotaxis failed: agent did not move toward the X gradient";
    // Y and Z should not have moved (pure X gradient)
    EXPECT_NEAR( result.y, 0.0f, 1e-4f ) << "Unexpected Y displacement from pure X gradient";
    EXPECT_NEAR( result.z, 0.0f, 1e-4f ) << "Unexpected Z displacement from pure X gradient";
    EXPECT_FLOAT_EQ( result.w, 1.0f ) << "Agent w flag was corrupted";

    m_rm->DestroyBuffer( inBuf );
    m_rm->DestroyBuffer( outBuf );
    m_rm->DestroyBuffer( countBuf );
    m_rm->DestroyTexture( fieldTex );
}

// Verify that an extremely large gradient is clamped to maxVelocity and produces no NaN
TEST_F( ComputeTest, Chemotaxis_HighGradient_ClampedToMaxVelocity )
{
    if( !m_device )
        GTEST_SKIP();

    uint32_t width = 10, height = 10, depth = 10;
    uint32_t voxelCount = width * height * depth;
    size_t   gridBytes  = voxelCount * sizeof( float );

    // Extreme gradient: field(x,y,z) = x * 10000.0f
    std::vector<float> fieldData( voxelCount );
    for( uint32_t z = 0; z < depth; ++z )
        for( uint32_t y = 0; y < height; ++y )
            for( uint32_t x = 0; x < width; ++x )
                fieldData[ x + y * width + z * width * height ] = static_cast<float>( x ) * 10000.0f;

    TextureDesc texDesc{};
    texDesc.type      = TextureType::Texture3D;
    texDesc.width     = width;
    texDesc.height    = height;
    texDesc.depth     = depth;
    texDesc.format    = VK_FORMAT_R32_SFLOAT;
    texDesc.usage     = TextureUsage::STORAGE | TextureUsage::TRANSFER_DST | TextureUsage::TRANSFER_SRC;
    texDesc.debugName = "ChemotaxisHighFieldTex";
    TextureHandle fieldTex = m_rm->CreateTexture( texDesc );
    m_stream->UploadTextureImmediate( fieldTex, fieldData.data(), gridBytes );

    glm::vec4    agentIn  = glm::vec4( 0.0f, 0.0f, 0.0f, 1.0f );
    glm::vec4    agentOut = glm::vec4( 0.0f );
    BufferHandle inBuf    = m_rm->CreateBuffer( { sizeof( glm::vec4 ), BufferType::STORAGE, "ChemoHighInAgents" } );
    BufferHandle outBuf   = m_rm->CreateBuffer( { sizeof( glm::vec4 ), BufferType::STORAGE, "ChemoHighOutAgents" } );
    m_stream->UploadBufferImmediate( { { inBuf, &agentIn, sizeof( glm::vec4 ) } } );
    m_stream->UploadBufferImmediate( { { outBuf, &agentOut, sizeof( glm::vec4 ) } } );

    uint32_t     count    = 1;
    BufferHandle countBuf = m_rm->CreateBuffer( { sizeof( uint32_t ), BufferType::INDIRECT, "ChemoHighCountBuf" } );
    m_stream->UploadBufferImmediate( { { countBuf, &count, sizeof( uint32_t ) } } );

    ComputePipelineDesc   pipeDesc{};
    pipeDesc.shader                  = m_rm->CreateShader( "shaders/compute/chemotaxis.comp" );
    ComputePipelineHandle pipeHandle = m_rm->CreatePipeline( pipeDesc );
    ComputePipeline*      pipeline   = m_rm->GetPipeline( pipeHandle );

    // Dummy phenotype buffer for binding 4 (state filtering disabled via push constants)
    BufferHandle phenotypeDummyBuf = m_rm->CreateBuffer( { sizeof( uint32_t ) * 4, BufferType::STORAGE, "ChemoHighPhenotypeDummy" } );

    BindingGroup* bg = m_rm->GetBindingGroup( m_rm->CreateBindingGroup( pipeHandle, 0 ) );
    bg->Bind( 0, m_rm->GetBuffer( inBuf ) );
    bg->Bind( 1, m_rm->GetBuffer( outBuf ) );
    bg->Bind( 2, m_rm->GetTexture( fieldTex ), VK_IMAGE_LAYOUT_GENERAL );
    bg->Bind( 3, m_rm->GetBuffer( countBuf ) );
    bg->Bind( 4, m_rm->GetBuffer( phenotypeDummyBuf ) );
    bg->Build();

    const float          maxVelocity = 5.0f;
    const float          dt          = 0.016f;
    ComputePushConstants pc{};
    pc.dt          = dt;
    pc.fParam0     = 1.0f;        // sensitivity
    pc.fParam1     = 0.0f;        // saturation
    pc.fParam2     = maxVelocity; // maxVelocity
    pc.fParam3     = -1.0f;       // reqCT = -1 (no filtering)
    pc.offset      = 0;
    pc.maxCapacity = 1;
    pc.uParam0     = static_cast<uint32_t>( -1 ); // reqLC = -1 (no filtering)
    pc.uParam1     = 0;
    pc.domainSize  = glm::vec4( 10.0f, 10.0f, 10.0f, 0.0f );
    pc.gridSize    = glm::uvec4( 10, 10, 10, 0 );

    auto compCtxHandle = m_device->CreateThreadContext( QueueType::COMPUTE );
    auto compCtx       = m_device->GetThreadContext( compCtxHandle );
    auto compCmd       = compCtx->GetCommandBuffer( compCtx->CreateCommandBuffer() );

    compCmd->Begin();
    compCmd->SetPipeline( pipeline );
    compCmd->SetBindingGroup( bg, pipeline->GetLayout(), VK_PIPELINE_BIND_POINT_COMPUTE );
    compCmd->PushConstants( pipeline->GetLayout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof( pc ), &pc );
    compCmd->Dispatch( 1, 1, 1 );
    compCmd->End();

    m_device->GetComputeQueue()->Submit( { compCmd } );
    m_device->GetComputeQueue()->WaitIdle();

    glm::vec4 result;
    m_stream->ReadbackBufferImmediate( outBuf, &result, sizeof( glm::vec4 ) );

    // Displacement must not exceed maxVelocity * dt
    float displacement = glm::length( glm::vec3( result ) - glm::vec3( agentIn ) );
    EXPECT_LE( displacement, maxVelocity * dt + 1e-4f ) << "Chemotaxis exceeded maxVelocity clamp!";

    // No NaN values
    EXPECT_TRUE( result.x == result.x ) << "NaN detected in x component after extreme gradient!";
    EXPECT_TRUE( result.y == result.y ) << "NaN detected in y component after extreme gradient!";
    EXPECT_TRUE( result.z == result.z ) << "NaN detected in z component after extreme gradient!";
    EXPECT_FLOAT_EQ( result.w, 1.0f );

    m_rm->DestroyBuffer( inBuf );
    m_rm->DestroyBuffer( outBuf );
    m_rm->DestroyBuffer( countBuf );
    m_rm->DestroyTexture( fieldTex );
}
