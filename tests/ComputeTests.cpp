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
    pc.fParam4     = 10.0f; // hypoxiaO2: 10.0 mmHg
    pc.maxCapacity = agentCount;
    pc.uParam1     = 0;
    pc.gridSize    = glm::uvec4( 0 ); // Disable texture read → localO2 = targetO2

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

    std::vector<PhenotypeData> pass1Result( agentCount );
    m_stream->ReadbackBufferImmediate( phenotypeBuf, pass1Result.data(), phenotypesSize );

    EXPECT_EQ( pass1Result[ 0 ].lifecycleState, 1 ) << "Agent 0 should be Quiescent (1) due to pressure!";
    EXPECT_EQ( pass1Result[ 1 ].lifecycleState, 0 ) << "Agent 1 should remain Live (0)!";
    EXPECT_FLOAT_EQ( pass1Result[ 2 ].biomass, 0.7f ) << "Agent 2 should have grown by 0.2!";

    // Dispatch Pass 2: Low O2 above necrosis → Live cells become Hypoxic (sequential transition)
    pc.fParam1 = 7.0f; // targetO2 used as fallback: above necrosisO2(5) but below hypoxiaO2(10)

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

    EXPECT_EQ( pass2Result[ 1 ].lifecycleState, 2 ) << "Agent 1 should be Hypoxic (2) — sequential transition from Live!";

    // Dispatch Pass 3: Very low O2 → Hypoxic cells become Necrotic
    pc.fParam1 = 2.0f; // targetO2 used as fallback: below necrosisO2(5)

    compCmd->Begin();
    compCmd->SetPipeline( pipeline );
    compCmd->SetBindingGroup( bg, pipeline->GetLayout(), VK_PIPELINE_BIND_POINT_COMPUTE );
    compCmd->PushConstants( pipeline->GetLayout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof( pc ), &pc );
    compCmd->Dispatch( 1, 1, 1 );
    compCmd->End();

    m_device->GetComputeQueue()->Submit( { compCmd } );
    m_device->GetComputeQueue()->WaitIdle();

    std::vector<PhenotypeData> pass3Result( agentCount );
    m_stream->ReadbackBufferImmediate( phenotypeBuf, pass3Result.data(), phenotypesSize );

    EXPECT_EQ( pass3Result[ 1 ].lifecycleState, 4 ) << "Agent 1 should be Necrotic (4) — from Hypoxic after O2 dropped below necrosis threshold!";

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

    // Edge buffers (bindings 4+5) — required by shader; cellType=0 so no edges written
    struct VesselEdge { uint32_t agentA, agentB; float dist; uint32_t flags; };
    size_t                   edgesSize  = maxCapacity * sizeof( VesselEdge );
    size_t                   edgeCntSz  = sizeof( uint32_t );
    std::vector<VesselEdge>  edgesInit( maxCapacity, { 0, 0, 0.0f, 0 } );
    uint32_t                 edgeCountInit = 0;

    // 2. Allocate & Upload Buffers
    BufferHandle agentsReadBuf  = m_rm->CreateBuffer( { agentsSize, BufferType::STORAGE, "TestAgentsRead" } );
    BufferHandle agentsWriteBuf = m_rm->CreateBuffer( { agentsSize, BufferType::STORAGE, "TestAgentsWrite" } );
    BufferHandle phenotypeBuf   = m_rm->CreateBuffer( { phenotypesSize, BufferType::STORAGE, "TestPhenotypes" } );
    BufferHandle countBuf       = m_rm->CreateBuffer( { countSize, BufferType::STORAGE, "TestCounter" } );
    BufferHandle edgesBuf       = m_rm->CreateBuffer( { edgesSize, BufferType::STORAGE, "TestEdges" } );
    BufferHandle edgeCountBuf   = m_rm->CreateBuffer( { edgeCntSz, BufferType::STORAGE, "TestEdgeCount" } );

    m_stream->UploadBufferImmediate( { { agentsReadBuf, agents.data(), agentsSize, 0 },
                                       { agentsWriteBuf, agents.data(), agentsSize, 0 },
                                       { phenotypeBuf, phenotypes.data(), phenotypesSize, 0 },
                                       { countBuf, &agentCount, countSize, 0 },
                                       { edgesBuf, edgesInit.data(), edgesSize, 0 },
                                       { edgeCountBuf, &edgeCountInit, edgeCntSz, 0 } } );

    // 3. Setup Pipeline & BindingGroup
    ComputePipelineDesc   pipeDesc{ m_rm->CreateShader( "shaders/compute/biology/mitosis_append.comp" ), "MitosisAppend" };
    ComputePipelineHandle pipelineHandle = m_rm->CreatePipeline( pipeDesc );
    ComputePipeline*      pipeline       = m_rm->GetPipeline( m_rm->CreatePipeline( pipeDesc ) );

    BindingGroup* bg = m_rm->GetBindingGroup( m_rm->CreateBindingGroup( pipelineHandle, 0 ) );
    bg->Bind( 0, m_rm->GetBuffer( agentsReadBuf ) );
    bg->Bind( 1, m_rm->GetBuffer( agentsWriteBuf ) );
    bg->Bind( 2, m_rm->GetBuffer( phenotypeBuf ) );
    bg->Bind( 3, m_rm->GetBuffer( countBuf ) );
    bg->Bind( 4, m_rm->GetBuffer( edgesBuf ) );
    bg->Bind( 5, m_rm->GetBuffer( edgeCountBuf ) );
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
    uint32_t                   resCount     = 0;
    uint32_t                   resEdgeCount = 0;

    m_stream->ReadbackBufferImmediate( agentsReadBuf, resAgents.data(), agentsSize );
    m_stream->ReadbackBufferImmediate( phenotypeBuf, resPheno.data(), phenotypesSize );
    m_stream->ReadbackBufferImmediate( countBuf, &resCount, countSize );
    m_stream->ReadbackBufferImmediate( edgeCountBuf, &resEdgeCount, edgeCntSz );

    EXPECT_EQ( resCount, 2 ) << "Atomic counter should have incremented to 2!";

    EXPECT_FLOAT_EQ( resPheno[ 0 ].biomass, 0.5f ) << "Mother biomass should split to 0.5!";
    EXPECT_FLOAT_EQ( resPheno[ 1 ].biomass, 0.5f ) << "Daughter biomass should start at 0.5!";
    EXPECT_FLOAT_EQ( resAgents[ 1 ].w, 1.0f ) << "Daughter w-component must be 1.0 (Alive)!";
    EXPECT_EQ( resEdgeCount, 0u ) << "Default cellType should not create vessel edges!";

    m_rm->DestroyBuffer( agentsReadBuf );
    m_rm->DestroyBuffer( agentsWriteBuf );
    m_rm->DestroyBuffer( phenotypeBuf );
    m_rm->DestroyBuffer( countBuf );
    m_rm->DestroyBuffer( edgesBuf );
    m_rm->DestroyBuffer( edgeCountBuf );
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
    // Dummy hash/offset buffers for bindings 5/6 — contact inhibition disabled (gridSize.w=0)
    BufferHandle dummyHashBuf   = m_rm->CreateBuffer( { 2 * sizeof( uint32_t ), BufferType::STORAGE, "ChemoDummyHash" } );
    BufferHandle dummyOffsetBuf = m_rm->CreateBuffer( { sizeof( uint32_t ),     BufferType::STORAGE, "ChemoDummyOffset" } );

    BindingGroup* bg = m_rm->GetBindingGroup( m_rm->CreateBindingGroup( pipeHandle, 0 ) );
    bg->Bind( 0, m_rm->GetBuffer( inBuf ) );
    bg->Bind( 1, m_rm->GetBuffer( outBuf ) );
    bg->Bind( 2, m_rm->GetTexture( fieldTex ), VK_IMAGE_LAYOUT_GENERAL );
    bg->Bind( 3, m_rm->GetBuffer( countBuf ) );
    bg->Bind( 4, m_rm->GetBuffer( phenotypeDummyBuf ) );
    bg->Bind( 5, m_rm->GetBuffer( dummyHashBuf ) );
    bg->Bind( 6, m_rm->GetBuffer( dummyOffsetBuf ) );
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
    m_rm->DestroyBuffer( dummyHashBuf );
    m_rm->DestroyBuffer( dummyOffsetBuf );
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
    // Dummy hash/offset buffers for bindings 5/6 — contact inhibition disabled (gridSize.w=0)
    BufferHandle dummyHashBuf   = m_rm->CreateBuffer( { 2 * sizeof( uint32_t ), BufferType::STORAGE, "ChemoHighDummyHash" } );
    BufferHandle dummyOffsetBuf = m_rm->CreateBuffer( { sizeof( uint32_t ),     BufferType::STORAGE, "ChemoHighDummyOffset" } );

    BindingGroup* bg = m_rm->GetBindingGroup( m_rm->CreateBindingGroup( pipeHandle, 0 ) );
    bg->Bind( 0, m_rm->GetBuffer( inBuf ) );
    bg->Bind( 1, m_rm->GetBuffer( outBuf ) );
    bg->Bind( 2, m_rm->GetTexture( fieldTex ), VK_IMAGE_LAYOUT_GENERAL );
    bg->Bind( 3, m_rm->GetBuffer( countBuf ) );
    bg->Bind( 4, m_rm->GetBuffer( phenotypeDummyBuf ) );
    bg->Bind( 5, m_rm->GetBuffer( dummyHashBuf ) );
    bg->Bind( 6, m_rm->GetBuffer( dummyOffsetBuf ) );
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
    m_rm->DestroyBuffer( dummyHashBuf );
    m_rm->DestroyBuffer( dummyOffsetBuf );
}

// =================================================================================================
// NotchDll4 shader tests
// =================================================================================================

// Isolated agent (no hash neighbors) — verify Euler ODE step and TipCell assignment.
// No neighbors: meanNeighborD=0 → newNicd stays 0.0 → vegfr2=1.0 (no inhibition)
// new_dll4 = clamp(0.5 + dt*(production*1.0 - decay*dll4), 0, 1)
//          = clamp(0.5 + 1.0*(1.0*1.0 - 0.1*0.5), 0, 1)
//          = clamp(1.45, 0, 1) = 1.0  →  cellType = TipCell (1)
TEST_F( ComputeTest, Shader_NotchDll4_IsolatedAgent_ODE )
{
    if( !m_device )
        GTEST_SKIP();

    // ── Raw data ──────────────────────────────────────────────────────────────
    // 1 alive agent at origin
    std::vector<glm::vec4> agents     = { glm::vec4( 0.0f, 0.0f, 0.0f, 1.0f ) };
    size_t                 agentsSize = sizeof( glm::vec4 );

    struct SignalingData { float dll4; float nicd; float vegfr2; float pad; };
    std::vector<SignalingData> signaling     = { { 0.5f, 0.0f, 1.0f, 0.0f } }; // initial dll4 = 0.5
    size_t                     signalingSize = sizeof( SignalingData );

    struct PhenotypeData { uint32_t lifecycleState; float biomass; float timer; uint32_t cellType; };
    std::vector<PhenotypeData> phenotypes     = { { 0u, 0.5f, 0.0f, 0u } }; // Live, Default
    size_t                     phenotypesSize = sizeof( PhenotypeData );

    // Minimal spatial hash: 1 entry, offsets all 0xFFFFFFFF → no neighbors found
    struct AgentHash { uint32_t hash; uint32_t agentIndex; };
    std::vector<AgentHash> sortedHashes = { { 0u, 0u } };
    size_t                 hashesSize   = sizeof( AgentHash );

    uint32_t              offsetArraySize = 256;
    std::vector<uint32_t> cellOffsets( offsetArraySize, 0xFFFFFFFF ); // all empty
    size_t                offsetsSize = offsetArraySize * sizeof( uint32_t );

    uint32_t agentCount = 1;
    size_t   countSize  = sizeof( uint32_t );

    // ── Allocate & Upload ─────────────────────────────────────────────────────
    BufferHandle  agentBuf  = m_rm->CreateBuffer( { agentsSize,     BufferType::STORAGE,  "NotchAgents"    } );
    BufferHandle  signalBuf = m_rm->CreateBuffer( { signalingSize,  BufferType::STORAGE,  "NotchSignaling" } );
    BufferHandle  phenoBuf  = m_rm->CreateBuffer( { phenotypesSize, BufferType::STORAGE,  "NotchPheno"     } );
    BufferHandle  hashBuf   = m_rm->CreateBuffer( { hashesSize,     BufferType::STORAGE,  "NotchHash"      } );
    BufferHandle  offsetBuf = m_rm->CreateBuffer( { offsetsSize,    BufferType::STORAGE,  "NotchOffsets"   } );
    BufferHandle  countBuf  = m_rm->CreateBuffer( { countSize,      BufferType::INDIRECT, "NotchCount"     } );
    // Dummy 1×1×1 VEGF texture (binding 6, VEGF disabled: gridSize.w=0 → shader uses localVEGF=1.0)
    TextureHandle vegfDummy = m_rm->CreateTexture( { 1, 1, 1, TextureType::Texture3D, VK_FORMAT_R32_SFLOAT, TextureUsage::STORAGE | TextureUsage::TRANSFER_DST, "NotchVEGFDummy" } );
    float         one       = 1.0f;
    m_stream->UploadTextureImmediate( vegfDummy, &one, sizeof( float ) );

    m_stream->UploadBufferImmediate( {
        { agentBuf,  agents.data(),       agentsSize,     0 },
        { signalBuf, signaling.data(),    signalingSize,  0 },
        { phenoBuf,  phenotypes.data(),   phenotypesSize, 0 },
        { hashBuf,   sortedHashes.data(), hashesSize,     0 },
        { offsetBuf, cellOffsets.data(),  offsetsSize,    0 },
        { countBuf,  &agentCount,         countSize,      0 },
    } );

    // ── Pipeline & BindingGroup ───────────────────────────────────────────────
    ComputePipelineDesc   pipeDesc{ m_rm->CreateShader( "shaders/compute/biology/notch_dll4.comp" ), "TestNotchDll4" };
    ComputePipelineHandle pipeHandle = m_rm->CreatePipeline( pipeDesc );
    ComputePipeline*      pipeline   = m_rm->GetPipeline( pipeHandle );

    BindingGroup* bg = m_rm->GetBindingGroup( m_rm->CreateBindingGroup( pipeHandle, 0 ) );
    bg->Bind( 0, m_rm->GetBuffer( agentBuf   ) );
    bg->Bind( 1, m_rm->GetBuffer( signalBuf  ) );
    bg->Bind( 2, m_rm->GetBuffer( phenoBuf   ) );
    bg->Bind( 3, m_rm->GetBuffer( hashBuf    ) );
    bg->Bind( 4, m_rm->GetBuffer( offsetBuf  ) );
    bg->Bind( 5, m_rm->GetBuffer( countBuf   ) );
    bg->Bind( 6, m_rm->GetTexture( vegfDummy ), VK_IMAGE_LAYOUT_GENERAL );
    bg->Build();

    // ── Push Constants ────────────────────────────────────────────────────────
    ComputePushConstants pc{};
    pc.dt          = 1.0f;
    pc.fParam0     = 1.0f;  // dll4ProductionRate
    pc.fParam1     = 0.1f;  // dll4DecayRate
    pc.fParam2     = 1.0f;  // notchInhibitionGain
    pc.fParam3     = 1.0f;  // vegfr2BaseExpression
    pc.fParam4     = 0.8f;  // tipThreshold
    pc.fParam5     = 0.3f;  // stalkThreshold
    pc.offset      = 0;
    pc.maxCapacity = 1;
    pc.uParam0     = offsetArraySize; // hash slot count
    pc.uParam1     = 0;               // grpNdx
    pc.domainSize  = glm::vec4( 100.0f, 100.0f, 100.0f, 15.0f ); // signalingRadius = 15
    pc.gridSize    = glm::uvec4( 1u, 1u, 1u, 0u ); // w=0 → VEGF disabled (localVEGF=1.0)

    // ── Dispatch ──────────────────────────────────────────────────────────────
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

    // ── Readback & Assert ─────────────────────────────────────────────────────
    SignalingData resultSignal{};
    PhenotypeData resultPheno{};
    m_stream->ReadbackBufferImmediate( signalBuf, &resultSignal, signalingSize );
    m_stream->ReadbackBufferImmediate( phenoBuf,  &resultPheno,  phenotypesSize );

    // Exact ODE: new_dll4 = clamp(0.5 + 1.0*(1.0*1.0 - 0.1*0.5), 0, 1) = 1.0
    EXPECT_NEAR( resultSignal.dll4,   1.0f, 1e-4f ) << "Notch ODE incorrect: isolated agent Dll4 should clamp to 1.0";
    EXPECT_EQ(   resultPheno.cellType, 1u )         << "Isolated agent with dll4=1.0 > tipThreshold should be TipCell (1)";

    // Non-Dll4 fields in signaling buffer must be untouched
    EXPECT_NEAR( resultSignal.nicd,   0.0f, 1e-4f );
    EXPECT_NEAR( resultSignal.vegfr2, 1.0f, 1e-4f );

    m_rm->DestroyBuffer( agentBuf  );
    m_rm->DestroyBuffer( signalBuf );
    m_rm->DestroyBuffer( phenoBuf  );
    m_rm->DestroyBuffer( hashBuf   );
    m_rm->DestroyBuffer( offsetBuf );
    m_rm->DestroyBuffer( countBuf  );
    m_rm->DestroyTexture( vegfDummy );
}

// Dead slot guard: agent with w=0 must be skipped entirely — dll4 and cellType unchanged.
TEST_F( ComputeTest, Shader_NotchDll4_DeadSlot_Skipped )
{
    if( !m_device )
        GTEST_SKIP();

    // Dead agent: w = 0.0
    std::vector<glm::vec4> agents     = { glm::vec4( 0.0f, 0.0f, 0.0f, 0.0f ) };
    size_t                 agentsSize = sizeof( glm::vec4 );

    struct SignalingData { float dll4; float nicd; float vegfr2; float pad; };
    std::vector<SignalingData> signaling     = { { 0.5f, 0.0f, 1.0f, 0.0f } };
    size_t                     signalingSize = sizeof( SignalingData );

    struct PhenotypeData { uint32_t lifecycleState; float biomass; float timer; uint32_t cellType; };
    std::vector<PhenotypeData> phenotypes     = { { 0u, 0.5f, 0.0f, 0u } };
    size_t                     phenotypesSize = sizeof( PhenotypeData );

    struct AgentHash { uint32_t hash; uint32_t agentIndex; };
    std::vector<AgentHash> sortedHashes = { { 0u, 0u } };
    size_t                 hashesSize   = sizeof( AgentHash );

    uint32_t              offsetArraySize = 256;
    std::vector<uint32_t> cellOffsets( offsetArraySize, 0xFFFFFFFF );
    size_t                offsetsSize = offsetArraySize * sizeof( uint32_t );

    uint32_t agentCount = 1;
    size_t   countSize  = sizeof( uint32_t );

    BufferHandle  agentBuf  = m_rm->CreateBuffer( { agentsSize,     BufferType::STORAGE,  "DeadNotchAgents"  } );
    BufferHandle  signalBuf = m_rm->CreateBuffer( { signalingSize,  BufferType::STORAGE,  "DeadNotchSignal"  } );
    BufferHandle  phenoBuf  = m_rm->CreateBuffer( { phenotypesSize, BufferType::STORAGE,  "DeadNotchPheno"   } );
    BufferHandle  hashBuf   = m_rm->CreateBuffer( { hashesSize,     BufferType::STORAGE,  "DeadNotchHash"    } );
    BufferHandle  offsetBuf = m_rm->CreateBuffer( { offsetsSize,    BufferType::STORAGE,  "DeadNotchOffsets" } );
    BufferHandle  countBuf  = m_rm->CreateBuffer( { countSize,      BufferType::INDIRECT, "DeadNotchCount"   } );
    TextureHandle vegfDummy = m_rm->CreateTexture( { 1, 1, 1, TextureType::Texture3D, VK_FORMAT_R32_SFLOAT, TextureUsage::STORAGE | TextureUsage::TRANSFER_DST, "DeadNotchVEGFDummy" } );
    float         one       = 1.0f;
    m_stream->UploadTextureImmediate( vegfDummy, &one, sizeof( float ) );

    m_stream->UploadBufferImmediate( {
        { agentBuf,  agents.data(),       agentsSize,     0 },
        { signalBuf, signaling.data(),    signalingSize,  0 },
        { phenoBuf,  phenotypes.data(),   phenotypesSize, 0 },
        { hashBuf,   sortedHashes.data(), hashesSize,     0 },
        { offsetBuf, cellOffsets.data(),  offsetsSize,    0 },
        { countBuf,  &agentCount,         countSize,      0 },
    } );

    ComputePipelineDesc   pipeDesc{ m_rm->CreateShader( "shaders/compute/biology/notch_dll4.comp" ), "TestNotchDll4Dead" };
    ComputePipelineHandle pipeHandle = m_rm->CreatePipeline( pipeDesc );
    ComputePipeline*      pipeline   = m_rm->GetPipeline( pipeHandle );

    BindingGroup* bg = m_rm->GetBindingGroup( m_rm->CreateBindingGroup( pipeHandle, 0 ) );
    bg->Bind( 0, m_rm->GetBuffer( agentBuf   ) );
    bg->Bind( 1, m_rm->GetBuffer( signalBuf  ) );
    bg->Bind( 2, m_rm->GetBuffer( phenoBuf   ) );
    bg->Bind( 3, m_rm->GetBuffer( hashBuf    ) );
    bg->Bind( 4, m_rm->GetBuffer( offsetBuf  ) );
    bg->Bind( 5, m_rm->GetBuffer( countBuf   ) );
    bg->Bind( 6, m_rm->GetTexture( vegfDummy ), VK_IMAGE_LAYOUT_GENERAL );
    bg->Build();

    ComputePushConstants pc{};
    pc.dt          = 1.0f;
    pc.fParam0     = 1.0f;
    pc.fParam1     = 0.1f;
    pc.fParam2     = 1.0f;
    pc.fParam3     = 1.0f;
    pc.fParam4     = 0.8f;
    pc.fParam5     = 0.3f;
    pc.offset      = 0;
    pc.maxCapacity = 1;
    pc.uParam0     = offsetArraySize;
    pc.uParam1     = 0;
    pc.domainSize  = glm::vec4( 100.0f, 100.0f, 100.0f, 15.0f );
    pc.gridSize    = glm::uvec4( 1u, 1u, 1u, 0u );

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

    SignalingData resultSignal{};
    PhenotypeData resultPheno{};
    m_stream->ReadbackBufferImmediate( signalBuf, &resultSignal, signalingSize );
    m_stream->ReadbackBufferImmediate( phenoBuf,  &resultPheno,  phenotypesSize );

    // Dead slot must be untouched
    EXPECT_NEAR( resultSignal.dll4,    0.5f, 1e-4f ) << "Dead slot: dll4 must not change";
    EXPECT_EQ(   resultPheno.cellType, 0u )          << "Dead slot: cellType must remain Default (0)";

    m_rm->DestroyBuffer( agentBuf  );
    m_rm->DestroyBuffer( signalBuf );
    m_rm->DestroyBuffer( phenoBuf  );
    m_rm->DestroyBuffer( hashBuf   );
    m_rm->DestroyBuffer( offsetBuf );
    m_rm->DestroyBuffer( countBuf  );
    m_rm->DestroyTexture( vegfDummy );
}

// Lateral inhibition with asymmetric initial Dll4 — verifiable in a single dispatch (dt=1.0).
//
// Agent 0: dll4=0.9 (dominant)  Agent 1: dll4=0.1 (suppressed)  gain=5, dt=1.0
//
// Agent 0 sees neighbor dll4=0.1:
//   vegfr2_eff = 1/(1 + 5*0.1) = 0.667
//   new_dll4   = clamp(0.9 + (1.0*0.667 - 0.1*0.9)) = clamp(1.477) = 1.0  → TipCell (1)
//
// Agent 1 sees neighbor dll4=0.9:
//   vegfr2_eff = 1/(1 + 5*0.9) = 0.182
//   new_dll4   = clamp(0.1 + (1.0*0.182 - 0.1*0.1)) = clamp(0.272) → StalkCell (2) since 0.272 < stalkThreshold 0.3
TEST_F( ComputeTest, Shader_NotchDll4_LateralInhibition_Asymmetric )
{
    if( !m_device )
        GTEST_SKIP();

    // ── Raw data ──────────────────────────────────────────────────────────────
    // 2 alive agents at (0,0,0) and (1,0,0) — within signaling radius
    std::vector<glm::vec4> agents     = { glm::vec4( 0.0f, 0.0f, 0.0f, 1.0f ), glm::vec4( 1.0f, 0.0f, 0.0f, 1.0f ) };
    size_t                 agentsSize = 2 * sizeof( glm::vec4 );

    struct SignalingData { float dll4; float nicd; float vegfr2; float pad; };
    // Agent 0 starts dominant (high Dll4), Agent 1 starts suppressed (low Dll4)
    std::vector<SignalingData> signaling     = { { 0.9f, 0.0f, 1.0f, 0.0f }, { 0.1f, 0.0f, 1.0f, 0.0f } };
    size_t                     signalingSize = 2 * sizeof( SignalingData );

    struct PhenotypeData { uint32_t lifecycleState; float biomass; float timer; uint32_t cellType; };
    std::vector<PhenotypeData> phenotypes     = { { 0u, 0.5f, 0.0f, 0u }, { 0u, 0.5f, 0.0f, 0u } };
    size_t                     phenotypesSize = 2 * sizeof( PhenotypeData );

    // Hash: both agents in same cell (0,0,0) → hash = 0
    struct AgentHash { uint32_t hash; uint32_t agentIndex; };
    std::vector<AgentHash> sortedHashes = { { 0u, 0u }, { 0u, 1u } };
    size_t                 hashesSize   = 2 * sizeof( AgentHash );

    // Offset: hash 0 starts at index 0 in sortedHashes
    uint32_t              offsetArraySize = 256;
    std::vector<uint32_t> cellOffsets( offsetArraySize, 0xFFFFFFFF );
    cellOffsets[ 0 ] = 0; // hash 0 → sortedHashes[0]
    size_t offsetsSize = offsetArraySize * sizeof( uint32_t );

    // Count: process 2 agents from group 0
    uint32_t agentCount = 2;
    size_t   countSize  = sizeof( uint32_t );

    // ── Allocate & Upload ─────────────────────────────────────────────────────
    BufferHandle  agentBuf  = m_rm->CreateBuffer( { agentsSize,     BufferType::STORAGE,  "LIAgents"   } );
    BufferHandle  signalBuf = m_rm->CreateBuffer( { signalingSize,  BufferType::STORAGE,  "LISignal"   } );
    BufferHandle  phenoBuf  = m_rm->CreateBuffer( { phenotypesSize, BufferType::STORAGE,  "LIPheno"    } );
    BufferHandle  hashBuf   = m_rm->CreateBuffer( { hashesSize,     BufferType::STORAGE,  "LIHash"     } );
    BufferHandle  offsetBuf = m_rm->CreateBuffer( { offsetsSize,    BufferType::STORAGE,  "LIOffsets"  } );
    BufferHandle  countBuf  = m_rm->CreateBuffer( { countSize,      BufferType::INDIRECT, "LICount"    } );
    TextureHandle vegfDummy = m_rm->CreateTexture( { 1, 1, 1, TextureType::Texture3D, VK_FORMAT_R32_SFLOAT, TextureUsage::STORAGE | TextureUsage::TRANSFER_DST, "LIVEGFDummy" } );
    float         one       = 1.0f;
    m_stream->UploadTextureImmediate( vegfDummy, &one, sizeof( float ) );

    m_stream->UploadBufferImmediate( {
        { agentBuf,  agents.data(),       agentsSize,     0 },
        { signalBuf, signaling.data(),    signalingSize,  0 },
        { phenoBuf,  phenotypes.data(),   phenotypesSize, 0 },
        { hashBuf,   sortedHashes.data(), hashesSize,     0 },
        { offsetBuf, cellOffsets.data(),  offsetsSize,    0 },
        { countBuf,  &agentCount,         countSize,      0 },
    } );

    // ── Pipeline & BindingGroup ───────────────────────────────────────────────
    ComputePipelineDesc   pipeDesc{ m_rm->CreateShader( "shaders/compute/biology/notch_dll4.comp" ), "TestNotchDll4LI" };
    ComputePipelineHandle pipeHandle = m_rm->CreatePipeline( pipeDesc );
    ComputePipeline*      pipeline   = m_rm->GetPipeline( pipeHandle );

    BindingGroup* bg = m_rm->GetBindingGroup( m_rm->CreateBindingGroup( pipeHandle, 0 ) );
    bg->Bind( 0, m_rm->GetBuffer( agentBuf   ) );
    bg->Bind( 1, m_rm->GetBuffer( signalBuf  ) );
    bg->Bind( 2, m_rm->GetBuffer( phenoBuf   ) );
    bg->Bind( 3, m_rm->GetBuffer( hashBuf    ) );
    bg->Bind( 4, m_rm->GetBuffer( offsetBuf  ) );
    bg->Bind( 5, m_rm->GetBuffer( countBuf   ) );
    bg->Bind( 6, m_rm->GetTexture( vegfDummy ), VK_IMAGE_LAYOUT_GENERAL );
    bg->Build();

    // ── Push Constants ────────────────────────────────────────────────────────
    ComputePushConstants pc{};
    pc.dt          = 1.0f;
    pc.fParam0     = 1.0f;  // dll4ProductionRate
    pc.fParam1     = 0.1f;  // dll4DecayRate
    pc.fParam2     = 5.0f;  // notchInhibitionGain
    pc.fParam3     = 1.0f;  // vegfr2BaseExpression
    pc.fParam4     = 0.8f;  // tipThreshold
    pc.fParam5     = 0.3f;  // stalkThreshold
    pc.offset      = 0;
    pc.maxCapacity = 2;
    pc.uParam0     = offsetArraySize;
    pc.uParam1     = 0;
    pc.domainSize  = glm::vec4( 100.0f, 100.0f, 100.0f, 15.0f ); // signalingRadius = 15
    pc.gridSize    = glm::uvec4( 1u, 1u, 1u, 0u );

    // ── Dispatch ──────────────────────────────────────────────────────────────
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

    // ── Readback & Assert ─────────────────────────────────────────────────────
    std::vector<SignalingData> resultSignal( 2 );
    std::vector<PhenotypeData> resultPheno( 2 );
    m_stream->ReadbackBufferImmediate( signalBuf, resultSignal.data(), signalingSize );
    m_stream->ReadbackBufferImmediate( phenoBuf,  resultPheno.data(),  phenotypesSize );

    // Agent 0: dominant → TipCell (dll4 clamped to 1.0)
    EXPECT_NEAR( resultSignal[ 0 ].dll4,   1.0f, 1e-4f ) << "Dominant agent dll4 should clamp to 1.0";
    EXPECT_EQ(   resultPheno[ 0 ].cellType, 1u )         << "Dominant agent should be TipCell (1)";

    // Agent 1: suppressed → StalkCell (dll4 = 0.272 < stalkThreshold 0.3)
    EXPECT_LT( resultSignal[ 1 ].dll4,    0.3f ) << "Suppressed agent dll4 should fall below stalkThreshold";
    EXPECT_EQ( resultPheno[ 1 ].cellType, 2u )   << "Suppressed agent should be StalkCell (2)";

    m_rm->DestroyBuffer( agentBuf  );
    m_rm->DestroyBuffer( signalBuf );
    m_rm->DestroyBuffer( phenoBuf  );
    m_rm->DestroyBuffer( hashBuf   );
    m_rm->DestroyBuffer( offsetBuf );
    m_rm->DestroyBuffer( countBuf  );
    m_rm->DestroyTexture( vegfDummy );
}

// VEGF gating: high local VEGF boosts vegfr2 → Dll4 production → TipCell;
// low VEGF suppresses vegfr2 → Dll4 decays → StalkCell.
//
// Setup: 2×2×1 VEGF texture. Agent 0 at (−5,0,0) falls in low-VEGF voxel (0,0,0)=0.0.
// Agent 1 at (+5,0,0) falls in high-VEGF voxel (1,0,0)=2.0.
// Both agents are isolated (no hash neighbors → sumNeighborDll4 = 0).
// Initial dll4 = 0.5.  Production=1, Decay=0.1, Base=0.5, Tip=0.8, dt=1.
//
// Agent 0 (low VEGF=0.0): vegfr2 = 0.5*0.0 / 1 = 0.0
//   new_dll4 = clamp(0.5 + 1*(0.0 - 0.1*0.5)) = clamp(0.45) = 0.45 → StalkCell (< 0.8)
//
// Agent 1 (high VEGF=2.0): vegfr2 = 0.5*2.0 / 1 = 1.0
//   new_dll4 = clamp(0.5 + 1*(1.0 - 0.1*0.5)) = clamp(1.45) = 1.0 → TipCell (> 0.8)
TEST_F( ComputeTest, Shader_NotchDll4_VEGFGating_HighVEGF_PromotesTipCell )
{
    if( !m_device )
        GTEST_SKIP();

    // ── VEGF texture: 2×2×1 ─────────────────────────────────────────────────────
    // Agents are at y=0 in a 20-unit centred domain → n.y=0.5 → voxel y=1.
    // Linear index = x + y*width:  (0,1,0)→2, (1,1,0)→3.
    // Set index 3 (voxel x=1,y=1) = 2.0; index 2 (voxel x=0,y=1) = 0.0.
    uint32_t         vegfW = 2, vegfH = 2, vegfD = 1;
    std::vector<float> vegfData( vegfW * vegfH * vegfD, 0.0f );
    vegfData[ 3 ] = 2.0f; // voxel (1,1,0) = high VEGF for agent 1

    TextureHandle vegfTex = m_rm->CreateTexture( { vegfW, vegfH, vegfD, TextureType::Texture3D,
                                                   VK_FORMAT_R32_SFLOAT,
                                                   TextureUsage::STORAGE | TextureUsage::TRANSFER_DST,
                                                   "VEGFGatingTex" } );
    m_stream->UploadTextureImmediate( vegfTex, vegfData.data(), vegfData.size() * sizeof( float ) );

    // ── 2 isolated agents — no hash neighbors ─────────────────────────────────
    // Domain = 20×20×20 centred at origin.
    // Agent 0 at (-5,0,0): n=(0.25, 0.5, 0.5) → voxel (0,1,0) → index 2 → VEGF=0.0
    // Agent 1 at ( 5,0,0): n=(0.75, 0.5, 0.5) → voxel (1,1,0) → index 3 → VEGF=2.0
    std::vector<glm::vec4> agents     = { glm::vec4( -5.0f, 0.0f, 0.0f, 1.0f ),
                                          glm::vec4(  5.0f, 0.0f, 0.0f, 1.0f ) };
    size_t agentsSize = 2 * sizeof( glm::vec4 );

    struct SignalingData { float dll4; float nicd; float vegfr2; float pad; };
    std::vector<SignalingData> signaling     = { { 0.5f, 0.0f, 1.0f, 0.0f }, { 0.5f, 0.0f, 1.0f, 0.0f } };
    size_t                     signalingSize = 2 * sizeof( SignalingData );

    struct PhenotypeData { uint32_t lifecycleState; float biomass; float timer; uint32_t cellType; };
    std::vector<PhenotypeData> phenotypes     = { { 0u, 0.5f, 0.0f, 0u }, { 0u, 0.5f, 0.0f, 0u } };
    size_t                     phenotypesSize = 2 * sizeof( PhenotypeData );

    struct AgentHash { uint32_t hash; uint32_t agentIndex; };
    std::vector<AgentHash> sortedHashes = { { 0u, 0u }, { 0u, 1u } };
    size_t                 hashesSize   = 2 * sizeof( AgentHash );

    uint32_t              offsetArraySize = 256;
    std::vector<uint32_t> cellOffsets( offsetArraySize, 0xFFFFFFFF ); // all empty — no neighbors
    size_t                offsetsSize = offsetArraySize * sizeof( uint32_t );

    uint32_t agentCount = 2;
    size_t   countSize  = sizeof( uint32_t );

    // ── Allocate & Upload ─────────────────────────────────────────────────────
    BufferHandle agentBuf  = m_rm->CreateBuffer( { agentsSize,     BufferType::STORAGE,  "VGAgents"   } );
    BufferHandle signalBuf = m_rm->CreateBuffer( { signalingSize,  BufferType::STORAGE,  "VGSignal"   } );
    BufferHandle phenoBuf  = m_rm->CreateBuffer( { phenotypesSize, BufferType::STORAGE,  "VGPheno"    } );
    BufferHandle hashBuf   = m_rm->CreateBuffer( { hashesSize,     BufferType::STORAGE,  "VGHash"     } );
    BufferHandle offsetBuf = m_rm->CreateBuffer( { offsetsSize,    BufferType::STORAGE,  "VGOffsets"  } );
    BufferHandle countBuf  = m_rm->CreateBuffer( { countSize,      BufferType::INDIRECT, "VGCount"    } );

    m_stream->UploadBufferImmediate( {
        { agentBuf,  agents.data(),       agentsSize,     0 },
        { signalBuf, signaling.data(),    signalingSize,  0 },
        { phenoBuf,  phenotypes.data(),   phenotypesSize, 0 },
        { hashBuf,   sortedHashes.data(), hashesSize,     0 },
        { offsetBuf, cellOffsets.data(),  offsetsSize,    0 },
        { countBuf,  &agentCount,         countSize,      0 },
    } );

    // ── Pipeline & BindingGroup ───────────────────────────────────────────────
    ComputePipelineDesc   pipeDesc{ m_rm->CreateShader( "shaders/compute/biology/notch_dll4.comp" ), "TestNotchVEGFGate" };
    ComputePipelineHandle pipeHandle = m_rm->CreatePipeline( pipeDesc );
    ComputePipeline*      pipeline   = m_rm->GetPipeline( pipeHandle );

    BindingGroup* bg = m_rm->GetBindingGroup( m_rm->CreateBindingGroup( pipeHandle, 0 ) );
    bg->Bind( 0, m_rm->GetBuffer( agentBuf   ) );
    bg->Bind( 1, m_rm->GetBuffer( signalBuf  ) );
    bg->Bind( 2, m_rm->GetBuffer( phenoBuf   ) );
    bg->Bind( 3, m_rm->GetBuffer( hashBuf    ) );
    bg->Bind( 4, m_rm->GetBuffer( offsetBuf  ) );
    bg->Bind( 5, m_rm->GetBuffer( countBuf   ) );
    bg->Bind( 6, m_rm->GetTexture( vegfTex   ), VK_IMAGE_LAYOUT_GENERAL );
    bg->Build();

    // ── Push Constants ────────────────────────────────────────────────────────
    // vegfr2BaseExpression=0.5 so we can easily check VEGF scaling:
    //   low-VEGF:  vegfr2 = 0.5 * 0.0 = 0.0
    //   high-VEGF: vegfr2 = 0.5 * 2.0 = 1.0
    ComputePushConstants pc{};
    pc.dt          = 1.0f;
    pc.fParam0     = 1.0f;   // dll4ProductionRate
    pc.fParam1     = 0.1f;   // dll4DecayRate
    pc.fParam2     = 0.0f;   // notchInhibitionGain = 0 (no lateral inhibition; isolated)
    pc.fParam3     = 0.5f;   // vegfr2BaseExpression
    pc.fParam4     = 0.8f;   // tipThreshold
    pc.fParam5     = 0.3f;   // stalkThreshold
    pc.offset      = 0;
    pc.maxCapacity = 2;
    pc.uParam0     = offsetArraySize;
    pc.uParam1     = 0;
    pc.domainSize  = glm::vec4( 20.0f, 20.0f, 20.0f, 15.0f );
    pc.gridSize    = glm::uvec4( vegfW, vegfH, vegfD, 1u ); // w=1 → VEGF sampling enabled

    // ── Dispatch ──────────────────────────────────────────────────────────────
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

    // ── Readback & Assert ─────────────────────────────────────────────────────
    std::vector<SignalingData> resultSignal( 2 );
    std::vector<PhenotypeData> resultPheno( 2 );
    m_stream->ReadbackBufferImmediate( signalBuf, resultSignal.data(), signalingSize );
    m_stream->ReadbackBufferImmediate( phenoBuf,  resultPheno.data(),  phenotypesSize );

    // Agent 0 in zero-VEGF: vegfr2=0 → Dll4 decays → 0.45 → StalkCell
    EXPECT_NEAR( resultSignal[ 0 ].dll4,    0.45f, 1e-4f ) << "Low-VEGF agent: dll4 = 0.5 + (0.0 - 0.05) = 0.45";
    EXPECT_EQ(   resultPheno[ 0 ].cellType, 2u )           << "Low-VEGF agent should be StalkCell (2)";

    // Agent 1 in high-VEGF: vegfr2=1 → Dll4 saturates → TipCell
    EXPECT_NEAR( resultSignal[ 1 ].dll4,    1.0f, 1e-4f )  << "High-VEGF agent: dll4 = clamp(0.5 + (1.0 - 0.05)) = 1.0";
    EXPECT_EQ(   resultPheno[ 1 ].cellType, 1u )           << "High-VEGF agent should be TipCell (1)";

    m_rm->DestroyBuffer( agentBuf  );
    m_rm->DestroyBuffer( signalBuf );
    m_rm->DestroyBuffer( phenoBuf  );
    m_rm->DestroyBuffer( hashBuf   );
    m_rm->DestroyBuffer( offsetBuf );
    m_rm->DestroyBuffer( countBuf  );
    m_rm->DestroyTexture( vegfTex  );
}

// =================================================================================================
// phalanx_activation.comp — raw shader tests
// =================================================================================================

// Helper: allocate & upload the minimal buffers needed for phalanx_activation and dispatch once.
// Returns resulting cellType from phenotype readback.
static uint32_t RunPhalanxShader(
    Device*           device,
    ResourceManager*  rm,
    StreamingManager* stream,
    glm::vec4         agentPos,
    uint32_t          initialCellType,
    float             vegfValue,
    float             activationThreshold,
    float             deactivationThreshold )
{
    struct PhenotypeData { uint32_t lifecycleState; float biomass; float timer; uint32_t cellType; };

    // 1×1×1 VEGF texture — agent at origin in 10-unit domain always hits voxel (0,0,0)
    TextureHandle vegfTex = rm->CreateTexture(
        { 1, 1, 1, TextureType::Texture3D, VK_FORMAT_R32_SFLOAT,
          TextureUsage::STORAGE | TextureUsage::TRANSFER_DST, "PhalanxVEGF" } );
    stream->UploadTextureImmediate( vegfTex, &vegfValue, sizeof( float ) );

    std::vector<glm::vec4>     agents     = { agentPos };
    std::vector<PhenotypeData> phenotypes = { { 0u, 0.5f, 0.0f, initialCellType } };
    uint32_t agentCount = 1;

    size_t agentsSize    = sizeof( glm::vec4 );
    size_t phenotypesSize = sizeof( PhenotypeData );
    size_t countSize      = sizeof( uint32_t );

    BufferHandle agentBuf = rm->CreateBuffer( { agentsSize,     BufferType::STORAGE,  "PhalanxAgents" } );
    BufferHandle phenoBuf = rm->CreateBuffer( { phenotypesSize, BufferType::STORAGE,  "PhalanxPheno"  } );
    BufferHandle countBuf = rm->CreateBuffer( { countSize,      BufferType::INDIRECT, "PhalanxCount"  } );

    stream->UploadBufferImmediate( {
        { agentBuf, agents.data(),     agentsSize,     0 },
        { phenoBuf, phenotypes.data(), phenotypesSize, 0 },
        { countBuf, &agentCount,       countSize,      0 },
    } );

    ComputePipelineDesc   pipeDesc{ rm->CreateShader( "shaders/compute/biology/phalanx_activation.comp" ), "TestPhalanx" };
    ComputePipelineHandle pipeHandle = rm->CreatePipeline( pipeDesc );
    ComputePipeline*      pipeline   = rm->GetPipeline( pipeHandle );

    BindingGroup* bg = rm->GetBindingGroup( rm->CreateBindingGroup( pipeHandle, 0 ) );
    bg->Bind( 0, rm->GetBuffer( agentBuf ) );
    bg->Bind( 1, rm->GetBuffer( phenoBuf ) );
    bg->Bind( 2, rm->GetTexture( vegfTex ), VK_IMAGE_LAYOUT_GENERAL );
    bg->Bind( 3, rm->GetBuffer( countBuf ) );
    bg->Build();

    ComputePushConstants pc{};
    pc.dt          = 1.0f / 60.0f;
    pc.fParam0     = activationThreshold;
    pc.fParam1     = deactivationThreshold;
    pc.offset      = 0;
    pc.maxCapacity = 1;
    pc.uParam0     = 0;
    pc.domainSize  = glm::vec4( 10.0f, 10.0f, 10.0f, 0.0f );
    pc.gridSize    = glm::uvec4( 1u, 1u, 1u, 1u ); // w=1 → VEGF sampling enabled

    auto compCtxHandle = device->CreateThreadContext( QueueType::COMPUTE );
    auto compCtx       = device->GetThreadContext( compCtxHandle );
    auto compCmd       = compCtx->GetCommandBuffer( compCtx->CreateCommandBuffer() );

    compCmd->Begin();
    compCmd->SetPipeline( pipeline );
    compCmd->SetBindingGroup( bg, pipeline->GetLayout(), VK_PIPELINE_BIND_POINT_COMPUTE );
    compCmd->PushConstants( pipeline->GetLayout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof( pc ), &pc );
    compCmd->Dispatch( 1, 1, 1 );
    compCmd->End();

    device->GetComputeQueue()->Submit( { compCmd } );
    device->GetComputeQueue()->WaitIdle();

    PhenotypeData result{};
    stream->ReadbackBufferImmediate( phenoBuf, &result, sizeof( PhenotypeData ) );

    rm->DestroyBuffer( agentBuf );
    rm->DestroyBuffer( phenoBuf );
    rm->DestroyBuffer( countBuf );
    rm->DestroyTexture( vegfTex );

    return result.cellType;
}

// PhalanxCell at VEGF=30 (> activationThreshold=20) → becomes StalkCell (2)
TEST_F( ComputeTest, Shader_PhalanxActivation_PhalanxAtHighVEGF_BecomesStalk )
{
    if( !m_device )
        GTEST_SKIP();

    uint32_t result = RunPhalanxShader( m_device.get(), m_rm.get(), m_stream.get(),
        glm::vec4( 0.0f, 0.0f, 0.0f, 1.0f ), // alive agent at origin
        3u,    // PhalanxCell
        30.0f, // VEGF = 30 > activationThreshold = 20
        20.0f, // activationThreshold
        5.0f   // deactivationThreshold
    );

    EXPECT_EQ( result, 2u ) << "PhalanxCell at VEGF=30 should activate to StalkCell (2)";
}

// PhalanxCell at VEGF=2 (< activationThreshold=20) → stays PhalanxCell (3)
TEST_F( ComputeTest, Shader_PhalanxActivation_PhalanxAtLowVEGF_StaysPhalanx )
{
    if( !m_device )
        GTEST_SKIP();

    uint32_t result = RunPhalanxShader( m_device.get(), m_rm.get(), m_stream.get(),
        glm::vec4( 0.0f, 0.0f, 0.0f, 1.0f ),
        3u,   // PhalanxCell
        2.0f, // VEGF = 2 < activationThreshold = 20
        20.0f, 5.0f
    );

    EXPECT_EQ( result, 3u ) << "PhalanxCell at VEGF=2 should remain PhalanxCell (3)";
}

// StalkCell at VEGF=2 (< deactivationThreshold=5) → re-quiesces to PhalanxCell (3)
TEST_F( ComputeTest, Shader_PhalanxActivation_StalkAtLowVEGF_BecomesQuiescent )
{
    if( !m_device )
        GTEST_SKIP();

    uint32_t result = RunPhalanxShader( m_device.get(), m_rm.get(), m_stream.get(),
        glm::vec4( 0.0f, 0.0f, 0.0f, 1.0f ),
        2u,   // StalkCell
        2.0f, // VEGF = 2 < deactivationThreshold = 5
        20.0f, 5.0f
    );

    EXPECT_EQ( result, 3u ) << "StalkCell at VEGF=2 should re-quiesce to PhalanxCell (3)";
}

// TipCell at any VEGF → always skipped, stays TipCell (1)
TEST_F( ComputeTest, Shader_PhalanxActivation_TipCell_AlwaysSkipped )
{
    if( !m_device )
        GTEST_SKIP();

    uint32_t result = RunPhalanxShader( m_device.get(), m_rm.get(), m_stream.get(),
        glm::vec4( 0.0f, 0.0f, 0.0f, 1.0f ),
        1u,   // TipCell
        2.0f, // VEGF below any threshold — would trigger re-quiescence if not a TipCell
        20.0f, 5.0f
    );

    EXPECT_EQ( result, 1u ) << "TipCell must not be re-quiesced — stays TipCell (1)";
}

// =================================================================================================
// anastomosis.comp — raw shader tests
// =================================================================================================

// Two TipCells within contactDistance → both become StalkCell, one edge written.
//
// Agents: A=(0,0,0), B=(2,0,0).  contactDistance=5, hashCellSize=30.
// Both map to hash cell (0,0,0) → hashPos(0,0,0)=0. offset[0]=0.
// Invocation A (idx=0) sees B (idx=1 > 0) → dist=2 < 5 → anastomose.
// Invocation B (idx=1) sees A (idx=0 ≤ 1) → dedup skip.
// Result: both cellType=2 (StalkCell), edgeCount=1, edge.dist≈2.
TEST_F( ComputeTest, Shader_Anastomosis_TwoTipCells_WithinRange )
{
    if( !m_device )
        GTEST_SKIP();

    // ── Raw data ──────────────────────────────────────────────────────────────
    std::vector<glm::vec4> agents     = { glm::vec4( 0.0f, 0.0f, 0.0f, 1.0f ), glm::vec4( 2.0f, 0.0f, 0.0f, 1.0f ) };
    size_t                 agentsSize = 2 * sizeof( glm::vec4 );

    struct PhenotypeData { uint32_t lifecycleState; float biomass; float timer; uint32_t cellType; };
    std::vector<PhenotypeData> phenotypes     = { { 0u, 0.5f, 0.0f, 1u }, { 0u, 0.5f, 0.0f, 1u } }; // both TipCell
    size_t                     phenotypesSize = 2 * sizeof( PhenotypeData );

    // Both agents in hash cell (0,0,0) → hashPos(0,0,0) = 0
    struct AgentHash { uint32_t hash; uint32_t agentIndex; };
    std::vector<AgentHash> sortedHashes = { { 0u, 0u }, { 0u, 1u } };
    size_t                 hashesSize   = 2 * sizeof( AgentHash );

    uint32_t              offsetArraySize = 256;
    std::vector<uint32_t> cellOffsets( offsetArraySize, 0xFFFFFFFF );
    cellOffsets[ 0 ] = 0; // hash 0 → sortedHashes[0]
    size_t offsetsSize = offsetArraySize * sizeof( uint32_t );

    uint32_t agentCount    = 2;
    uint32_t edgeCountInit = 0;

    struct VesselEdge { uint32_t agentA; uint32_t agentB; float dist; uint32_t flags; };
    size_t edgeBufferSize = 16 * sizeof( VesselEdge );

    // ── Allocate & Upload ─────────────────────────────────────────────────────
    BufferHandle agentBuf     = m_rm->CreateBuffer( { agentsSize,           BufferType::STORAGE,  "AnaAgents"    } );
    BufferHandle phenoBuf     = m_rm->CreateBuffer( { phenotypesSize,       BufferType::STORAGE,  "AnaPheno"     } );
    BufferHandle hashBuf      = m_rm->CreateBuffer( { hashesSize,           BufferType::STORAGE,  "AnaHash"      } );
    BufferHandle offsetBuf    = m_rm->CreateBuffer( { offsetsSize,          BufferType::STORAGE,  "AnaOffsets"   } );
    BufferHandle countBuf     = m_rm->CreateBuffer( { sizeof( uint32_t ),   BufferType::INDIRECT, "AnaCount"     } );
    BufferHandle edgeBuf      = m_rm->CreateBuffer( { edgeBufferSize,       BufferType::STORAGE,  "AnaEdges"     } );
    BufferHandle edgeCountBuf = m_rm->CreateBuffer( { sizeof( uint32_t ),   BufferType::STORAGE,  "AnaEdgeCount" } );

    m_stream->UploadBufferImmediate( {
        { agentBuf,     agents.data(),       agentsSize,         0 },
        { phenoBuf,     phenotypes.data(),   phenotypesSize,     0 },
        { hashBuf,      sortedHashes.data(), hashesSize,         0 },
        { offsetBuf,    cellOffsets.data(),  offsetsSize,        0 },
        { countBuf,     &agentCount,         sizeof( uint32_t ), 0 },
        { edgeCountBuf, &edgeCountInit,      sizeof( uint32_t ), 0 },
    } );

    // ── Pipeline & BindingGroup ───────────────────────────────────────────────
    ComputePipelineDesc   pipeDesc{ m_rm->CreateShader( "shaders/compute/biology/anastomosis.comp" ), "TestAnastomosis" };
    ComputePipelineHandle pipeHandle = m_rm->CreatePipeline( pipeDesc );
    ComputePipeline*      pipeline   = m_rm->GetPipeline( pipeHandle );

    BindingGroup* bg = m_rm->GetBindingGroup( m_rm->CreateBindingGroup( pipeHandle, 0 ) );
    bg->Bind( 0, m_rm->GetBuffer( agentBuf     ) );
    bg->Bind( 1, m_rm->GetBuffer( phenoBuf     ) );
    bg->Bind( 2, m_rm->GetBuffer( hashBuf      ) );
    bg->Bind( 3, m_rm->GetBuffer( offsetBuf    ) );
    bg->Bind( 4, m_rm->GetBuffer( countBuf     ) );
    bg->Bind( 5, m_rm->GetBuffer( edgeBuf      ) );
    bg->Bind( 6, m_rm->GetBuffer( edgeCountBuf ) );
    bg->Build();

    // ── Push Constants ────────────────────────────────────────────────────────
    ComputePushConstants pc{};
    pc.dt          = 1.0f / 60.0f;
    pc.fParam0     = 5.0f;  // contactDistance
    pc.offset      = 0;
    pc.maxCapacity = 2;
    pc.uParam0     = offsetArraySize;
    pc.uParam1     = 0;
    pc.domainSize  = glm::vec4( 100.0f, 100.0f, 100.0f, 30.0f ); // .w = hashCellSize

    // ── Dispatch ──────────────────────────────────────────────────────────────
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

    // ── Readback & Assert ─────────────────────────────────────────────────────
    std::vector<PhenotypeData> resultPheno( 2 );
    uint32_t                   resultEdgeCount = 0;
    VesselEdge                 resultEdge{};
    m_stream->ReadbackBufferImmediate( phenoBuf,     resultPheno.data(), phenotypesSize );
    m_stream->ReadbackBufferImmediate( edgeCountBuf, &resultEdgeCount,   sizeof( uint32_t ) );
    m_stream->ReadbackBufferImmediate( edgeBuf,      &resultEdge,        sizeof( VesselEdge ) );

    EXPECT_EQ(   resultPheno[ 0 ].cellType, 2u )   << "Agent 0 must be StalkCell (2) after anastomosis";
    EXPECT_EQ(   resultPheno[ 1 ].cellType, 2u )   << "Agent 1 must be StalkCell (2) after anastomosis";
    EXPECT_EQ(   resultEdgeCount,           1u )   << "Exactly 1 edge must be recorded";
    EXPECT_EQ(   resultEdge.agentA,         0u )   << "Edge agentA must be 0 (lower-index invocation handles the pair)";
    EXPECT_EQ(   resultEdge.agentB,         1u )   << "Edge agentB must be 1";
    EXPECT_NEAR( resultEdge.dist,           2.0f, 1e-4f ) << "Edge dist must equal agent separation";

    m_rm->DestroyBuffer( agentBuf     );
    m_rm->DestroyBuffer( phenoBuf     );
    m_rm->DestroyBuffer( hashBuf      );
    m_rm->DestroyBuffer( offsetBuf    );
    m_rm->DestroyBuffer( countBuf     );
    m_rm->DestroyBuffer( edgeBuf      );
    m_rm->DestroyBuffer( edgeCountBuf );
}

// Dead slot (w=0) pre-labelled as TipCell must be skipped — cellType and edge count unchanged.
TEST_F( ComputeTest, Shader_Anastomosis_DeadSlot_Skipped )
{
    if( !m_device )
        GTEST_SKIP();

    std::vector<glm::vec4> agents     = { glm::vec4( 0.0f, 0.0f, 0.0f, 0.0f ) }; // w=0 → dead
    size_t                 agentsSize = sizeof( glm::vec4 );

    struct PhenotypeData { uint32_t lifecycleState; float biomass; float timer; uint32_t cellType; };
    std::vector<PhenotypeData> phenotypes     = { { 0u, 0.5f, 0.0f, 1u } }; // TipCell but dead
    size_t                     phenotypesSize = sizeof( PhenotypeData );

    struct AgentHash { uint32_t hash; uint32_t agentIndex; };
    std::vector<AgentHash> sortedHashes = { { 0u, 0u } };
    size_t                 hashesSize   = sizeof( AgentHash );

    uint32_t              offsetArraySize = 256;
    std::vector<uint32_t> cellOffsets( offsetArraySize, 0xFFFFFFFF );
    size_t                offsetsSize = offsetArraySize * sizeof( uint32_t );

    uint32_t agentCount    = 1;
    uint32_t edgeCountInit = 0;

    struct VesselEdge { uint32_t agentA; uint32_t agentB; float dist; uint32_t flags; };

    BufferHandle agentBuf     = m_rm->CreateBuffer( { agentsSize,           BufferType::STORAGE,  "DeadAnaAgents"    } );
    BufferHandle phenoBuf     = m_rm->CreateBuffer( { phenotypesSize,       BufferType::STORAGE,  "DeadAnaPheno"     } );
    BufferHandle hashBuf      = m_rm->CreateBuffer( { hashesSize,           BufferType::STORAGE,  "DeadAnaHash"      } );
    BufferHandle offsetBuf    = m_rm->CreateBuffer( { offsetsSize,          BufferType::STORAGE,  "DeadAnaOffsets"   } );
    BufferHandle countBuf     = m_rm->CreateBuffer( { sizeof( uint32_t ),   BufferType::INDIRECT, "DeadAnaCount"     } );
    BufferHandle edgeBuf      = m_rm->CreateBuffer( { sizeof( VesselEdge ), BufferType::STORAGE,  "DeadAnaEdges"     } );
    BufferHandle edgeCountBuf = m_rm->CreateBuffer( { sizeof( uint32_t ),   BufferType::STORAGE,  "DeadAnaEdgeCount" } );

    m_stream->UploadBufferImmediate( {
        { agentBuf,     agents.data(),       agentsSize,         0 },
        { phenoBuf,     phenotypes.data(),   phenotypesSize,     0 },
        { hashBuf,      sortedHashes.data(), hashesSize,         0 },
        { offsetBuf,    cellOffsets.data(),  offsetsSize,        0 },
        { countBuf,     &agentCount,         sizeof( uint32_t ), 0 },
        { edgeCountBuf, &edgeCountInit,      sizeof( uint32_t ), 0 },
    } );

    ComputePipelineDesc   pipeDesc{ m_rm->CreateShader( "shaders/compute/biology/anastomosis.comp" ), "TestAnastomosisDeadSlot" };
    ComputePipelineHandle pipeHandle = m_rm->CreatePipeline( pipeDesc );
    ComputePipeline*      pipeline   = m_rm->GetPipeline( pipeHandle );

    BindingGroup* bg = m_rm->GetBindingGroup( m_rm->CreateBindingGroup( pipeHandle, 0 ) );
    bg->Bind( 0, m_rm->GetBuffer( agentBuf     ) );
    bg->Bind( 1, m_rm->GetBuffer( phenoBuf     ) );
    bg->Bind( 2, m_rm->GetBuffer( hashBuf      ) );
    bg->Bind( 3, m_rm->GetBuffer( offsetBuf    ) );
    bg->Bind( 4, m_rm->GetBuffer( countBuf     ) );
    bg->Bind( 5, m_rm->GetBuffer( edgeBuf      ) );
    bg->Bind( 6, m_rm->GetBuffer( edgeCountBuf ) );
    bg->Build();

    ComputePushConstants pc{};
    pc.dt          = 1.0f / 60.0f;
    pc.fParam0     = 5.0f;
    pc.offset      = 0;
    pc.maxCapacity = 1;
    pc.uParam0     = offsetArraySize;
    pc.uParam1     = 0;
    pc.domainSize  = glm::vec4( 100.0f, 100.0f, 100.0f, 30.0f );

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

    PhenotypeData resultPheno{};
    uint32_t      resultEdgeCount = 0;
    m_stream->ReadbackBufferImmediate( phenoBuf,     &resultPheno,     phenotypesSize );
    m_stream->ReadbackBufferImmediate( edgeCountBuf, &resultEdgeCount, sizeof( uint32_t ) );

    EXPECT_EQ( resultPheno.cellType, 1u ) << "Dead slot: cellType must remain TipCell (1) — dead guard fired";
    EXPECT_EQ( resultEdgeCount,      0u ) << "Dead slot: no edge must be written";

    m_rm->DestroyBuffer( agentBuf     );
    m_rm->DestroyBuffer( phenoBuf     );
    m_rm->DestroyBuffer( hashBuf      );
    m_rm->DestroyBuffer( offsetBuf    );
    m_rm->DestroyBuffer( countBuf     );
    m_rm->DestroyBuffer( edgeBuf      );
    m_rm->DestroyBuffer( edgeCountBuf );
}

// Non-TipCell agents (Default + StalkCell) within contact range must be skipped — no edge written.
TEST_F( ComputeTest, Shader_Anastomosis_NonTipCells_Skipped )
{
    if( !m_device )
        GTEST_SKIP();

    // Two alive agents at close range — Default (0) and StalkCell (2), neither is TipCell (1)
    std::vector<glm::vec4> agents     = { glm::vec4( 0.0f, 0.0f, 0.0f, 1.0f ), glm::vec4( 1.0f, 0.0f, 0.0f, 1.0f ) };
    size_t                 agentsSize = 2 * sizeof( glm::vec4 );

    struct PhenotypeData { uint32_t lifecycleState; float biomass; float timer; uint32_t cellType; };
    std::vector<PhenotypeData> phenotypes     = { { 0u, 0.5f, 0.0f, 0u }, { 0u, 0.5f, 0.0f, 2u } };
    size_t                     phenotypesSize = 2 * sizeof( PhenotypeData );

    struct AgentHash { uint32_t hash; uint32_t agentIndex; };
    std::vector<AgentHash> sortedHashes = { { 0u, 0u }, { 0u, 1u } };
    size_t                 hashesSize   = 2 * sizeof( AgentHash );

    uint32_t              offsetArraySize = 256;
    std::vector<uint32_t> cellOffsets( offsetArraySize, 0xFFFFFFFF );
    cellOffsets[ 0 ] = 0;
    size_t offsetsSize = offsetArraySize * sizeof( uint32_t );

    uint32_t agentCount    = 2;
    uint32_t edgeCountInit = 0;

    struct VesselEdge { uint32_t agentA; uint32_t agentB; float dist; uint32_t flags; };

    BufferHandle agentBuf     = m_rm->CreateBuffer( { agentsSize,           BufferType::STORAGE,  "NTCAgents"    } );
    BufferHandle phenoBuf     = m_rm->CreateBuffer( { phenotypesSize,       BufferType::STORAGE,  "NTCPheno"     } );
    BufferHandle hashBuf      = m_rm->CreateBuffer( { hashesSize,           BufferType::STORAGE,  "NTCHash"      } );
    BufferHandle offsetBuf    = m_rm->CreateBuffer( { offsetsSize,          BufferType::STORAGE,  "NTCOffsets"   } );
    BufferHandle countBuf     = m_rm->CreateBuffer( { sizeof( uint32_t ),   BufferType::INDIRECT, "NTCCount"     } );
    BufferHandle edgeBuf      = m_rm->CreateBuffer( { sizeof( VesselEdge ), BufferType::STORAGE,  "NTCEdges"     } );
    BufferHandle edgeCountBuf = m_rm->CreateBuffer( { sizeof( uint32_t ),   BufferType::STORAGE,  "NTCEdgeCount" } );

    m_stream->UploadBufferImmediate( {
        { agentBuf,     agents.data(),       agentsSize,         0 },
        { phenoBuf,     phenotypes.data(),   phenotypesSize,     0 },
        { hashBuf,      sortedHashes.data(), hashesSize,         0 },
        { offsetBuf,    cellOffsets.data(),  offsetsSize,        0 },
        { countBuf,     &agentCount,         sizeof( uint32_t ), 0 },
        { edgeCountBuf, &edgeCountInit,      sizeof( uint32_t ), 0 },
    } );

    ComputePipelineDesc   pipeDesc{ m_rm->CreateShader( "shaders/compute/biology/anastomosis.comp" ), "TestAnastomosisNTC" };
    ComputePipelineHandle pipeHandle = m_rm->CreatePipeline( pipeDesc );
    ComputePipeline*      pipeline   = m_rm->GetPipeline( pipeHandle );

    BindingGroup* bg = m_rm->GetBindingGroup( m_rm->CreateBindingGroup( pipeHandle, 0 ) );
    bg->Bind( 0, m_rm->GetBuffer( agentBuf     ) );
    bg->Bind( 1, m_rm->GetBuffer( phenoBuf     ) );
    bg->Bind( 2, m_rm->GetBuffer( hashBuf      ) );
    bg->Bind( 3, m_rm->GetBuffer( offsetBuf    ) );
    bg->Bind( 4, m_rm->GetBuffer( countBuf     ) );
    bg->Bind( 5, m_rm->GetBuffer( edgeBuf      ) );
    bg->Bind( 6, m_rm->GetBuffer( edgeCountBuf ) );
    bg->Build();

    ComputePushConstants pc{};
    pc.dt          = 1.0f / 60.0f;
    pc.fParam0     = 5.0f;
    pc.offset      = 0;
    pc.maxCapacity = 2;
    pc.uParam0     = offsetArraySize;
    pc.uParam1     = 0;
    pc.domainSize  = glm::vec4( 100.0f, 100.0f, 100.0f, 30.0f );

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

    std::vector<PhenotypeData> resultPheno( 2 );
    uint32_t                   resultEdgeCount = 0;
    m_stream->ReadbackBufferImmediate( phenoBuf,     resultPheno.data(), phenotypesSize );
    m_stream->ReadbackBufferImmediate( edgeCountBuf, &resultEdgeCount,   sizeof( uint32_t ) );

    EXPECT_EQ( resultPheno[ 0 ].cellType, 0u ) << "Default agent must remain Default (0)";
    EXPECT_EQ( resultPheno[ 1 ].cellType, 2u ) << "StalkCell agent must remain StalkCell (2)";
    EXPECT_EQ( resultEdgeCount,           0u ) << "No edge must be written for non-TipCell agents";

    m_rm->DestroyBuffer( agentBuf     );
    m_rm->DestroyBuffer( phenoBuf     );
    m_rm->DestroyBuffer( hashBuf      );
    m_rm->DestroyBuffer( offsetBuf    );
    m_rm->DestroyBuffer( countBuf     );
    m_rm->DestroyBuffer( edgeBuf      );
    m_rm->DestroyBuffer( edgeCountBuf );
}

// ===========================================================================================
// vessel_components.comp — raw shader tests (no Builder)
// ===========================================================================================

// One edge between agent 0 and agent 1 → single dispatch → both receive label 0.
TEST_F( ComputeTest, Shader_VesselComponents_IsolatedPair )
{
    if( !m_device )
        GTEST_SKIP();

    struct VesselEdge { uint32_t agentA; uint32_t agentB; float dist; uint32_t flags; };

    // One edge: (0, 1)
    std::vector<VesselEdge> edgeData  = { { 0u, 1u, 2.0f, 0u } };
    uint32_t                edgeCount = 1;

    // Identity labels: labels[i] = i
    std::vector<uint32_t> labelData = { 0u, 1u };
    size_t                labelSize = 2 * sizeof( uint32_t );

    BufferHandle edgeBuf      = m_rm->CreateBuffer( { sizeof( VesselEdge ),   BufferType::STORAGE, "VCIEdges"      } );
    BufferHandle edgeCountBuf = m_rm->CreateBuffer( { sizeof( uint32_t ),     BufferType::STORAGE, "VCIEdgeCount"  } );
    BufferHandle componentBuf = m_rm->CreateBuffer( { labelSize,              BufferType::STORAGE, "VCIComponents" } );

    m_stream->UploadBufferImmediate( {
        { edgeBuf,      edgeData.data(),  sizeof( VesselEdge ), 0 },
        { edgeCountBuf, &edgeCount,       sizeof( uint32_t ),   0 },
        { componentBuf, labelData.data(), labelSize,            0 },
    } );

    ComputePipelineDesc   pipeDesc{ m_rm->CreateShader( "shaders/compute/biology/vessel_components.comp" ), "TestVCIsolated" };
    ComputePipelineHandle pipeHandle = m_rm->CreatePipeline( pipeDesc );
    ComputePipeline*      pipeline   = m_rm->GetPipeline( pipeHandle );

    BindingGroup* bg = m_rm->GetBindingGroup( m_rm->CreateBindingGroup( pipeHandle, 0 ) );
    bg->Bind( 0, m_rm->GetBuffer( edgeBuf      ) );
    bg->Bind( 1, m_rm->GetBuffer( edgeCountBuf ) );
    bg->Bind( 2, m_rm->GetBuffer( componentBuf ) );
    bg->Build();

    ComputePushConstants pc{};
    pc.maxCapacity = 2; // safety cap

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

    std::vector<uint32_t> resultLabels( 2 );
    m_stream->ReadbackBufferImmediate( componentBuf, resultLabels.data(), labelSize );

    EXPECT_EQ( resultLabels[ 0 ], resultLabels[ 1 ] ) << "Both agents must share the same component label after 1 pass";
    EXPECT_EQ( resultLabels[ 0 ], 0u )               << "Minimum label (0) must propagate to agent 1";

    m_rm->DestroyBuffer( edgeBuf      );
    m_rm->DestroyBuffer( edgeCountBuf );
    m_rm->DestroyBuffer( componentBuf );
}

// Three agents in a chain: edges (0,1) and (1,2). Two passes must propagate label 0 to all.
TEST_F( ComputeTest, Shader_VesselComponents_Chain_ThreeAgents )
{
    if( !m_device )
        GTEST_SKIP();

    struct VesselEdge { uint32_t agentA; uint32_t agentB; float dist; uint32_t flags; };

    // Chain: 0─1─2
    std::vector<VesselEdge> edgeData  = { { 0u, 1u, 2.0f, 0u }, { 1u, 2u, 2.0f, 0u } };
    uint32_t                edgeCount = 2;

    // Identity labels
    std::vector<uint32_t> labelData = { 0u, 1u, 2u };
    size_t                labelSize = 3 * sizeof( uint32_t );

    BufferHandle edgeBuf      = m_rm->CreateBuffer( { 2 * sizeof( VesselEdge ), BufferType::STORAGE, "VCCEdges"      } );
    BufferHandle edgeCountBuf = m_rm->CreateBuffer( { sizeof( uint32_t ),       BufferType::STORAGE, "VCCEdgeCount"  } );
    BufferHandle componentBuf = m_rm->CreateBuffer( { labelSize,                BufferType::STORAGE, "VCCComponents" } );

    m_stream->UploadBufferImmediate( {
        { edgeBuf,      edgeData.data(),  2 * sizeof( VesselEdge ), 0 },
        { edgeCountBuf, &edgeCount,       sizeof( uint32_t ),       0 },
        { componentBuf, labelData.data(), labelSize,                0 },
    } );

    ComputePipelineDesc   pipeDesc{ m_rm->CreateShader( "shaders/compute/biology/vessel_components.comp" ), "TestVCChain3" };
    ComputePipelineHandle pipeHandle = m_rm->CreatePipeline( pipeDesc );
    ComputePipeline*      pipeline   = m_rm->GetPipeline( pipeHandle );

    BindingGroup* bg = m_rm->GetBindingGroup( m_rm->CreateBindingGroup( pipeHandle, 0 ) );
    bg->Bind( 0, m_rm->GetBuffer( edgeBuf      ) );
    bg->Bind( 1, m_rm->GetBuffer( edgeCountBuf ) );
    bg->Bind( 2, m_rm->GetBuffer( componentBuf ) );
    bg->Build();

    ComputePushConstants pc{};
    pc.maxCapacity = 2; // 2 edges max

    auto compCtxHandle = m_device->CreateThreadContext( QueueType::COMPUTE );
    auto compCtx       = m_device->GetThreadContext( compCtxHandle );

    // Two sequential submissions — a chain of length 2 needs 2 passes to converge.
    // Each submission is separated by WaitIdle so atomic writes from pass N are visible to pass N+1.
    for( int pass = 0; pass < 2; ++pass )
    {
        auto compCmd = compCtx->GetCommandBuffer( compCtx->CreateCommandBuffer() );
        compCmd->Begin();
        compCmd->SetPipeline( pipeline );
        compCmd->SetBindingGroup( bg, pipeline->GetLayout(), VK_PIPELINE_BIND_POINT_COMPUTE );
        compCmd->PushConstants( pipeline->GetLayout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof( pc ), &pc );
        compCmd->Dispatch( 1, 1, 1 );
        compCmd->End();
        m_device->GetComputeQueue()->Submit( { compCmd } );
        m_device->GetComputeQueue()->WaitIdle();
    }

    std::vector<uint32_t> resultLabels( 3 );
    m_stream->ReadbackBufferImmediate( componentBuf, resultLabels.data(), labelSize );

    EXPECT_EQ( resultLabels[ 0 ], 0u ) << "Agent 0 must have label 0";
    EXPECT_EQ( resultLabels[ 1 ], 0u ) << "Agent 1 must have label 0 after 2 passes";
    EXPECT_EQ( resultLabels[ 2 ], 0u ) << "Agent 2 must have label 0 after 2 passes";

    m_rm->DestroyBuffer( edgeBuf      );
    m_rm->DestroyBuffer( edgeCountBuf );
    m_rm->DestroyBuffer( componentBuf );
}

// ===========================================================================================
// perfusion.comp — raw shader tests (no Builder)
// ===========================================================================================

// Shared setup for perfusion.comp raw tests: one agent, one voxel, returns delta after dispatch.
// cellType=2 → StalkCell (should fire), cellType=0 → Default (should skip).
// rate is passed directly to fParam0 (positive=inject, negative=drain).
#define PERFUSION_RAW_TEST_SETUP( TEST_CELLTYPE, TEST_RATE, LABEL )                                  \
    glm::vec4 agentPos( 0.0f, 0.0f, 0.0f, 1.0f );                                                   \
    struct PhenotypeData { uint32_t lifecycleState; float biomass; float timer; uint32_t cellType; }; \
    PhenotypeData pheno{ 0u, 0.5f, 0.0f, ( TEST_CELLTYPE ) };                                        \
    uint32_t agentCount = 1;                                                                          \
    int      deltaInit  = 0;                                                                          \
    BufferHandle agentBuf = m_rm->CreateBuffer( { sizeof( glm::vec4 ),    BufferType::STORAGE,  LABEL "Agents" } ); \
    BufferHandle phenoBuf = m_rm->CreateBuffer( { sizeof( PhenotypeData ), BufferType::STORAGE,  LABEL "Pheno"  } ); \
    BufferHandle countBuf = m_rm->CreateBuffer( { sizeof( uint32_t ),      BufferType::INDIRECT, LABEL "Count"  } ); \
    BufferHandle deltaBuf = m_rm->CreateBuffer( { sizeof( int ),           BufferType::STORAGE,  LABEL "Delta"  } ); \
    m_stream->UploadBufferImmediate( {                                                                \
        { agentBuf, &agentPos,   sizeof( glm::vec4 ),    0 },                                        \
        { phenoBuf, &pheno,      sizeof( PhenotypeData ), 0 },                                       \
        { countBuf, &agentCount, sizeof( uint32_t ),     0 },                                        \
        { deltaBuf, &deltaInit,  sizeof( int ),          0 },                                        \
    } );                                                                                              \
    ComputePipelineDesc   pipeDesc{ m_rm->CreateShader( "shaders/compute/biology/perfusion.comp" ), "TestPerf" LABEL }; \
    ComputePipelineHandle pipeHandle = m_rm->CreatePipeline( pipeDesc );                             \
    ComputePipeline*      pipeline   = m_rm->GetPipeline( pipeHandle );                              \
    BindingGroup* bg = m_rm->GetBindingGroup( m_rm->CreateBindingGroup( pipeHandle, 0 ) );           \
    bg->Bind( 0, m_rm->GetBuffer( agentBuf ) );                                                      \
    bg->Bind( 1, m_rm->GetBuffer( deltaBuf ) );                                                      \
    bg->Bind( 2, m_rm->GetBuffer( countBuf ) );                                                      \
    bg->Bind( 3, m_rm->GetBuffer( phenoBuf ) );                                                      \
    bg->Build();                                                                                      \
    ComputePushConstants pc{};                                                                        \
    pc.dt          = 1.0f;                                                                            \
    pc.fParam0     = ( TEST_RATE );                                                                   \
    pc.offset      = 0;                                                                               \
    pc.maxCapacity = 1;                                                                               \
    pc.uParam1     = 0;                                                                               \
    pc.domainSize  = glm::vec4( 10.0f, 10.0f, 10.0f, 0.0f );                                        \
    pc.gridSize    = glm::uvec4( 1, 1, 1, 0 );                                                       \
    auto compCtxHandle = m_device->CreateThreadContext( QueueType::COMPUTE );                        \
    auto compCtx       = m_device->GetThreadContext( compCtxHandle );                                \
    auto compCmd       = compCtx->GetCommandBuffer( compCtx->CreateCommandBuffer() );                \
    compCmd->Begin();                                                                                 \
    compCmd->SetPipeline( pipeline );                                                                 \
    compCmd->SetBindingGroup( bg, pipeline->GetLayout(), VK_PIPELINE_BIND_POINT_COMPUTE );           \
    compCmd->PushConstants( pipeline->GetLayout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof( pc ), &pc ); \
    compCmd->Dispatch( 1, 1, 1 );                                                                     \
    compCmd->End();                                                                                   \
    m_device->GetComputeQueue()->Submit( { compCmd } );                                              \
    m_device->GetComputeQueue()->WaitIdle();                                                         \
    int result = 0;                                                                                   \
    m_stream->ReadbackBufferImmediate( deltaBuf, &result, sizeof( int ) );                           \
    m_rm->DestroyBuffer( agentBuf );                                                                  \
    m_rm->DestroyBuffer( phenoBuf );                                                                  \
    m_rm->DestroyBuffer( countBuf );                                                                  \
    m_rm->DestroyBuffer( deltaBuf );

// =================================================================================================
// Chemotaxis cell-type filter tests
// =================================================================================================

// TipCell (cellType=1) in a VEGF gradient with reqCT=1 → agent must move toward gradient.
TEST_F( ComputeTest, Chemotaxis_CellTypeFilter_TipCellMoves )
{
    if( !m_device )
        GTEST_SKIP();

    // 10x10x10 field with a linear X gradient: field(x,y,z) = x * 1.0f
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
    texDesc.debugName = "ChemoCTFilterFieldTex";
    TextureHandle fieldTex = m_rm->CreateTexture( texDesc );
    m_stream->UploadTextureImmediate( fieldTex, fieldData.data(), gridBytes );

    // 1 agent at world origin (0,0,0), alive
    glm::vec4    agentIn  = glm::vec4( 0.0f, 0.0f, 0.0f, 1.0f );
    glm::vec4    agentOut = glm::vec4( 0.0f );
    BufferHandle inBuf    = m_rm->CreateBuffer( { sizeof( glm::vec4 ), BufferType::STORAGE, "ChemoCTIn" } );
    BufferHandle outBuf   = m_rm->CreateBuffer( { sizeof( glm::vec4 ), BufferType::STORAGE, "ChemoCTOut" } );
    m_stream->UploadBufferImmediate( { { inBuf, &agentIn, sizeof( glm::vec4 ) } } );
    m_stream->UploadBufferImmediate( { { outBuf, &agentOut, sizeof( glm::vec4 ) } } );

    uint32_t     count    = 1;
    BufferHandle countBuf = m_rm->CreateBuffer( { sizeof( uint32_t ), BufferType::INDIRECT, "ChemoCTCount" } );
    m_stream->UploadBufferImmediate( { { countBuf, &count, sizeof( uint32_t ) } } );

    // Phenotype buffer: cellType = 1 (TipCell) — matches reqCT
    struct PhenotypeData { uint32_t lifecycleState; float biomass; float timer; uint32_t cellType; };
    PhenotypeData pheno = { 0u, 0.5f, 0.0f, 1u }; // TipCell
    BufferHandle  phenoBuf = m_rm->CreateBuffer( { sizeof( PhenotypeData ), BufferType::STORAGE, "ChemoCTPheno" } );
    m_stream->UploadBufferImmediate( { { phenoBuf, &pheno, sizeof( PhenotypeData ) } } );

    ComputePipelineDesc   pipeDesc{};
    pipeDesc.shader                  = m_rm->CreateShader( "shaders/compute/chemotaxis.comp" );
    ComputePipelineHandle pipeHandle = m_rm->CreatePipeline( pipeDesc );
    ComputePipeline*      pipeline   = m_rm->GetPipeline( pipeHandle );

    BindingGroup* bg = m_rm->GetBindingGroup( m_rm->CreateBindingGroup( pipeHandle, 0 ) );
    bg->Bind( 0, m_rm->GetBuffer( inBuf ) );
    bg->Bind( 1, m_rm->GetBuffer( outBuf ) );
    bg->Bind( 2, m_rm->GetTexture( fieldTex ), VK_IMAGE_LAYOUT_GENERAL );
    bg->Bind( 3, m_rm->GetBuffer( countBuf ) );
    bg->Bind( 4, m_rm->GetBuffer( phenoBuf ) );
    // Dummy hash/offset buffers for bindings 5/6 — contact inhibition disabled (gridSize.w=0)
    BufferHandle dummyHashBuf   = m_rm->CreateBuffer( { 2 * sizeof( uint32_t ), BufferType::STORAGE, "ChemoCTDummyHash" } );
    BufferHandle dummyOffsetBuf = m_rm->CreateBuffer( { sizeof( uint32_t ),     BufferType::STORAGE, "ChemoCTDummyOffset" } );
    bg->Bind( 5, m_rm->GetBuffer( dummyHashBuf ) );
    bg->Bind( 6, m_rm->GetBuffer( dummyOffsetBuf ) );
    bg->Build();

    ComputePushConstants pc{};
    pc.dt          = 1.0f;
    pc.fParam0     = 1.0f;   // sensitivity
    pc.fParam1     = 0.0f;   // saturation (linear)
    pc.fParam2     = 100.0f; // maxVelocity
    pc.fParam3     = 1.0f;   // reqCT = 1 (TipCell)
    pc.offset      = 0;
    pc.maxCapacity = 1;
    pc.uParam0     = static_cast<uint32_t>( -1 ); // reqLC = -1 (any)
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

    EXPECT_GT( result.x, 0.0f ) << "TipCell with reqCT=TipCell must move toward the X gradient";
    EXPECT_FLOAT_EQ( result.w, 1.0f );

    m_rm->DestroyBuffer( inBuf );
    m_rm->DestroyBuffer( outBuf );
    m_rm->DestroyBuffer( countBuf );
    m_rm->DestroyBuffer( phenoBuf );
    m_rm->DestroyTexture( fieldTex );
    m_rm->DestroyBuffer( dummyHashBuf );
    m_rm->DestroyBuffer( dummyOffsetBuf );
}

// Default cell (cellType=0) in a VEGF gradient with reqCT=1 (TipCell) → agent must NOT move.
// This verifies that the cell-type filter correctly skips non-TipCells.
TEST_F( ComputeTest, Chemotaxis_CellTypeFilter_NonTipCellSkipped )
{
    if( !m_device )
        GTEST_SKIP();

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
    texDesc.debugName = "ChemoCTSkipFieldTex";
    TextureHandle fieldTex = m_rm->CreateTexture( texDesc );
    m_stream->UploadTextureImmediate( fieldTex, fieldData.data(), gridBytes );

    // Agent at origin — initial position deliberately non-zero in output to detect no-write
    glm::vec4    agentIn  = glm::vec4( 0.0f, 0.0f, 0.0f, 1.0f );
    glm::vec4    sentinel = glm::vec4( -999.0f, -999.0f, -999.0f, -999.0f );
    BufferHandle inBuf    = m_rm->CreateBuffer( { sizeof( glm::vec4 ), BufferType::STORAGE, "ChemoCTSkipIn" } );
    BufferHandle outBuf   = m_rm->CreateBuffer( { sizeof( glm::vec4 ), BufferType::STORAGE, "ChemoCTSkipOut" } );
    m_stream->UploadBufferImmediate( { { inBuf, &agentIn, sizeof( glm::vec4 ) } } );
    m_stream->UploadBufferImmediate( { { outBuf, &sentinel, sizeof( glm::vec4 ) } } );

    uint32_t     count    = 1;
    BufferHandle countBuf = m_rm->CreateBuffer( { sizeof( uint32_t ), BufferType::INDIRECT, "ChemoCTSkipCount" } );
    m_stream->UploadBufferImmediate( { { countBuf, &count, sizeof( uint32_t ) } } );

    // Phenotype: cellType = 0 (Default) — does NOT match reqCT=1 (TipCell)
    struct PhenotypeData { uint32_t lifecycleState; float biomass; float timer; uint32_t cellType; };
    PhenotypeData pheno = { 0u, 0.5f, 0.0f, 0u }; // Default
    BufferHandle  phenoBuf = m_rm->CreateBuffer( { sizeof( PhenotypeData ), BufferType::STORAGE, "ChemoCTSkipPheno" } );
    m_stream->UploadBufferImmediate( { { phenoBuf, &pheno, sizeof( PhenotypeData ) } } );

    ComputePipelineDesc   pipeDesc{};
    pipeDesc.shader                  = m_rm->CreateShader( "shaders/compute/chemotaxis.comp" );
    ComputePipelineHandle pipeHandle = m_rm->CreatePipeline( pipeDesc );
    ComputePipeline*      pipeline   = m_rm->GetPipeline( pipeHandle );

    BindingGroup* bg = m_rm->GetBindingGroup( m_rm->CreateBindingGroup( pipeHandle, 0 ) );
    bg->Bind( 0, m_rm->GetBuffer( inBuf ) );
    bg->Bind( 1, m_rm->GetBuffer( outBuf ) );
    bg->Bind( 2, m_rm->GetTexture( fieldTex ), VK_IMAGE_LAYOUT_GENERAL );
    bg->Bind( 3, m_rm->GetBuffer( countBuf ) );
    bg->Bind( 4, m_rm->GetBuffer( phenoBuf ) );
    // Dummy hash/offset buffers for bindings 5/6 — contact inhibition disabled (gridSize.w=0)
    BufferHandle dummyHashBuf   = m_rm->CreateBuffer( { 2 * sizeof( uint32_t ), BufferType::STORAGE, "ChemoCTSkipDummyHash" } );
    BufferHandle dummyOffsetBuf = m_rm->CreateBuffer( { sizeof( uint32_t ),     BufferType::STORAGE, "ChemoCTSkipDummyOffset" } );
    bg->Bind( 5, m_rm->GetBuffer( dummyHashBuf ) );
    bg->Bind( 6, m_rm->GetBuffer( dummyOffsetBuf ) );
    bg->Build();

    ComputePushConstants pc{};
    pc.dt          = 1.0f;
    pc.fParam0     = 1.0f;   // sensitivity
    pc.fParam1     = 0.0f;   // saturation
    pc.fParam2     = 100.0f; // maxVelocity
    pc.fParam3     = 1.0f;   // reqCT = 1 (TipCell) — filter out Default cells
    pc.offset      = 0;
    pc.maxCapacity = 1;
    pc.uParam0     = static_cast<uint32_t>( -1 );
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

    // Output buffer should retain its sentinel value — the shader returns early without writing
    EXPECT_FLOAT_EQ( result.x, -999.0f ) << "Default cell with reqCT=TipCell must NOT be written to output";
    EXPECT_FLOAT_EQ( result.w, -999.0f ) << "Output buffer sentinel should be untouched";

    m_rm->DestroyBuffer( inBuf );
    m_rm->DestroyBuffer( outBuf );
    m_rm->DestroyBuffer( countBuf );
    m_rm->DestroyBuffer( phenoBuf );
    m_rm->DestroyTexture( fieldTex );
    m_rm->DestroyBuffer( dummyHashBuf );
    m_rm->DestroyBuffer( dummyOffsetBuf );
}

// StalkCell (cellType=2) with positive rate → delta > 0 (injection).
TEST_F( ComputeTest, Shader_Perfusion_StalkCell_InjectsDelta )
{
    if( !m_device ) GTEST_SKIP();
    PERFUSION_RAW_TEST_SETUP( 2u, +5.0f, "PFInject" )
    EXPECT_GT( result, 0 ) << "StalkCell with positive rate must produce a positive delta (injection)";
}

// StalkCell with negative rate → delta < 0 (drain).
TEST_F( ComputeTest, Shader_Drain_StalkCell_RemovesDelta )
{
    if( !m_device ) GTEST_SKIP();
    PERFUSION_RAW_TEST_SETUP( 2u, -5.0f, "PFDrain" )
    EXPECT_LT( result, 0 ) << "StalkCell with negative rate must produce a negative delta (drain)";
}

// Non-StalkCell (Default, cellType=0) → delta unchanged regardless of rate.
TEST_F( ComputeTest, Shader_Perfusion_NonStalkCell_Skipped )
{
    if( !m_device ) GTEST_SKIP();
    PERFUSION_RAW_TEST_SETUP( 0u, +5.0f, "PFSkip" )
    EXPECT_EQ( result, 0 ) << "Default cell must not modify the delta buffer";
}

// build_indirect.comp: 2 agents with different cellTypes scattered into correct draw commands.
// 3 draw commands: default (group 0), default (group 0 fallback), TipCell (group 0, cellType=1), StalkCell (group 0, cellType=2)
// Actually: 1 group with 3 draw commands: default (0xFFFFFFFF), TipCell (1), StalkCell (2).
// Agent 0 = TipCell, Agent 1 = StalkCell → TipCell cmd gets 1 instance, StalkCell cmd gets 1 instance, default gets 0.
TEST_F( ComputeTest, Shader_BuildIndirect_ClassifyCellTypes )
{
    if( !m_device )
        GTEST_SKIP();

    // Create pipeline
    ShaderHandle          sh   = m_rm->CreateShader( "shaders/graphics/build_indirect.comp" );
    ComputePipelineDesc   desc{};
    desc.shader                = sh;
    ComputePipelineHandle pipe = m_rm->CreatePipeline( desc );
    ComputePipeline*      pipePtr = m_rm->GetPipeline( pipe );

    const uint32_t agentCount    = 2;
    const uint32_t drawCmdCount  = 3; // default + TipCell + StalkCell
    const uint32_t groupCapacity = 64; // padded

    // Draw meta: {groupIndex, targetCellType, groupOffset, groupCapacity}
    struct DrawMeta { uint32_t groupIndex, targetCellType, groupOffset, groupCapacity; };
    std::vector<DrawMeta> metaData = {
        { 0, 0xFFFFFFFF, 0, groupCapacity }, // default: catches any unmatched cellType
        { 0, 1,          0, groupCapacity }, // TipCell
        { 0, 2,          0, groupCapacity }, // StalkCell
    };

    // Indirect commands: instanceCount=0 (will be filled), firstInstance points to reorder regions
    struct DrawCommand { uint32_t indexCount, instanceCount, firstIndex, vertexOffset, firstInstance; };
    std::vector<DrawCommand> cmds = {
        { 36, 0, 0, 0, 0 * groupCapacity },                // default  → reorder[0..63]
        { 36, 0, 0, 0, 1 * groupCapacity },                // TipCell  → reorder[64..127]
        { 36, 0, 0, 0, 2 * groupCapacity },                // StalkCell→ reorder[128..191]
    };

    // Agent count buffer (1 group with 2 live agents)
    std::vector<uint32_t> counts = { agentCount };

    // Agent positions: both alive (w=1)
    std::vector<glm::vec4> positions( groupCapacity, glm::vec4( 0 ) );
    positions[ 0 ] = glm::vec4( 0, 0, 0, 1 );
    positions[ 1 ] = glm::vec4( 1, 0, 0, 1 );

    // Phenotype data: agent 0 = TipCell, agent 1 = StalkCell
    struct PhenotypeData { uint32_t lifecycleState; float biomass; float timer; uint32_t cellType; };
    std::vector<PhenotypeData> phenotypes( groupCapacity, { 0, 0.5f, 0.0f, 0 } );
    phenotypes[ 0 ].cellType = 1; // TipCell
    phenotypes[ 1 ].cellType = 2; // StalkCell

    // Reorder buffer (3 * groupCapacity slots)
    uint32_t reorderSize = 3 * groupCapacity;
    std::vector<uint32_t> reorderInit( reorderSize, 0xDEADBEEF );

    // Create GPU buffers
    auto countBuf     = m_rm->CreateBuffer( { sizeof( uint32_t ), BufferType::STORAGE, "BI_Counts" } );
    auto indirectBuf  = m_rm->CreateBuffer( { cmds.size() * sizeof( DrawCommand ), BufferType::STORAGE, "BI_Indirect" } );
    auto phenoBuf     = m_rm->CreateBuffer( { phenotypes.size() * sizeof( PhenotypeData ), BufferType::STORAGE, "BI_Pheno" } );
    auto reorderBuf   = m_rm->CreateBuffer( { reorderSize * sizeof( uint32_t ), BufferType::STORAGE, "BI_Reorder" } );
    auto metaBuf      = m_rm->CreateBuffer( { metaData.size() * sizeof( DrawMeta ), BufferType::STORAGE, "BI_Meta" } );
    auto agentBuf     = m_rm->CreateBuffer( { positions.size() * sizeof( glm::vec4 ), BufferType::STORAGE, "BI_Agents" } );

    m_stream->UploadBufferImmediate( {
        { countBuf,    counts.data(),      counts.size() * sizeof( uint32_t ) },
        { indirectBuf, cmds.data(),        cmds.size() * sizeof( DrawCommand ) },
        { phenoBuf,    phenotypes.data(),  phenotypes.size() * sizeof( PhenotypeData ) },
        { reorderBuf,  reorderInit.data(), reorderInit.size() * sizeof( uint32_t ) },
        { metaBuf,     metaData.data(),    metaData.size() * sizeof( DrawMeta ) },
        { agentBuf,    positions.data(),   positions.size() * sizeof( glm::vec4 ) },
    } );

    // Bind
    BindingGroup* bg = m_rm->GetBindingGroup( m_rm->CreateBindingGroup( pipe, 0 ) );
    bg->Bind( 0, m_rm->GetBuffer( countBuf ) );
    bg->Bind( 1, m_rm->GetBuffer( indirectBuf ) );
    bg->Bind( 2, m_rm->GetBuffer( phenoBuf ) );
    bg->Bind( 3, m_rm->GetBuffer( reorderBuf ) );
    bg->Bind( 4, m_rm->GetBuffer( metaBuf ) );
    bg->Bind( 5, m_rm->GetBuffer( agentBuf ) );
    bg->Build();

    // Record command buffer: dispatch RESET then CLASSIFY
    auto ctxH   = m_device->CreateThreadContext( QueueType::COMPUTE );
    auto ctx    = m_device->GetThreadContext( ctxH );
    auto cmdBuf = ctx->GetCommandBuffer( ctx->CreateCommandBuffer() );

    ComputePushConstants resetPC{};
    resetPC.uParam0 = 0; // reset mode
    resetPC.uParam1 = drawCmdCount; // grpNdx = draw command count

    ComputePushConstants classifyPC{};
    classifyPC.uParam0     = 1; // classify mode
    classifyPC.uParam1     = drawCmdCount;
    classifyPC.maxCapacity = groupCapacity;

    cmdBuf->Begin();

    // Reset dispatch
    cmdBuf->SetPipeline( pipePtr );
    cmdBuf->SetBindingGroup( bg, pipePtr->GetLayout(), VK_PIPELINE_BIND_POINT_COMPUTE );
    cmdBuf->PushConstants( pipePtr->GetLayout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof( ComputePushConstants ), &resetPC );
    cmdBuf->Dispatch( 1, 1, 1 );

    // Barrier between reset and classify
    VkMemoryBarrier2 barrier = { VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
    barrier.srcStageMask     = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    barrier.srcAccessMask    = VK_ACCESS_2_SHADER_WRITE_BIT;
    barrier.dstStageMask     = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    barrier.dstAccessMask    = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT;
    VkDependencyInfo dep     = { VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
    dep.memoryBarrierCount   = 1;
    dep.pMemoryBarriers      = &barrier;
    cmdBuf->PipelineBarrier( &dep );

    // Classify dispatch
    cmdBuf->PushConstants( pipePtr->GetLayout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof( ComputePushConstants ), &classifyPC );
    cmdBuf->Dispatch( ( groupCapacity + 255 ) / 256, 1, 1 );

    cmdBuf->End();
    m_device->GetComputeQueue()->Submit( { cmdBuf } );
    m_device->GetComputeQueue()->WaitIdle();

    // Readback indirect commands
    std::vector<DrawCommand> resultCmds( drawCmdCount );
    m_stream->ReadbackBufferImmediate( indirectBuf, resultCmds.data(), drawCmdCount * sizeof( DrawCommand ) );

    // Default command: should have 0 instances (both agents matched specific cellType commands)
    EXPECT_EQ( resultCmds[ 0 ].instanceCount, 0u ) << "Default draw: no agents with unmatched cellType";

    // TipCell command: agent 0
    EXPECT_EQ( resultCmds[ 1 ].instanceCount, 1u ) << "TipCell draw: exactly 1 agent";

    // StalkCell command: agent 1
    EXPECT_EQ( resultCmds[ 2 ].instanceCount, 1u ) << "StalkCell draw: exactly 1 agent";

    // Readback reorder buffer and verify the agent indices
    std::vector<uint32_t> reorderResult( reorderSize );
    m_stream->ReadbackBufferImmediate( reorderBuf, reorderResult.data(), reorderSize * sizeof( uint32_t ) );

    // TipCell region starts at 1*groupCapacity, first entry should be agent 0
    EXPECT_EQ( reorderResult[ 1 * groupCapacity ], 0u ) << "TipCell reorder slot 0 should be agent index 0";

    // StalkCell region starts at 2*groupCapacity, first entry should be agent 1
    EXPECT_EQ( reorderResult[ 2 * groupCapacity ], 1u ) << "StalkCell reorder slot 0 should be agent index 1";
}

// =================================================================================================
// VesselMechanics (vessel_mechanics.comp) shader tests
// =================================================================================================

// Two agents 4 units apart connected by 1 edge. Resting length=2, k=10.
// After one step: each agent should move 0.5 * k * (dist - restLen) * dt = 0.5*10*2*1 = 10 units toward the other.
// Expected: agents closer together than initial 4 units.
TEST_F( ComputeTest, Shader_VesselMechanics_SpringReducesStretch )
{
    if( !m_device )
        GTEST_SKIP();

    struct VesselEdge { uint32_t agentA; uint32_t agentB; float dist; uint32_t flags; };

    // Two agents: agent 0 at (-2, 0, 0), agent 1 at (2, 0, 0) → 4 units apart
    std::vector<glm::vec4> agentsIn = {
        glm::vec4( -2.0f, 0.0f, 0.0f, 1.0f ),
        glm::vec4(  2.0f, 0.0f, 0.0f, 1.0f ),
    };
    std::vector<glm::vec4> agentsOut( 2, glm::vec4( 0.0f ) );
    size_t agentBytes = 2 * sizeof( glm::vec4 );

    VesselEdge edge = { 0u, 1u, 2.0f, 0u };
    uint32_t   edgeCount = 1;
    uint32_t   agentCount = 2;

    BufferHandle inBuf        = m_rm->CreateBuffer( { agentBytes,          BufferType::STORAGE, "VMInAgents"    } );
    BufferHandle outBuf       = m_rm->CreateBuffer( { agentBytes,          BufferType::STORAGE, "VMOutAgents"   } );
    BufferHandle edgeBuf      = m_rm->CreateBuffer( { sizeof( VesselEdge ),BufferType::STORAGE, "VMEdges"       } );
    BufferHandle edgeCountBuf = m_rm->CreateBuffer( { sizeof( uint32_t ),  BufferType::STORAGE, "VMEdgeCount"   } );
    BufferHandle countBuf     = m_rm->CreateBuffer( { sizeof( uint32_t ),  BufferType::STORAGE, "VMCount"       } );

    m_stream->UploadBufferImmediate( {
        { inBuf,        agentsIn.data(), agentBytes,          0 },
        { outBuf,       agentsOut.data(), agentBytes,         0 },
        { edgeBuf,      &edge,            sizeof( VesselEdge ), 0 },
        { edgeCountBuf, &edgeCount,       sizeof( uint32_t ),  0 },
        { countBuf,     &agentCount,      sizeof( uint32_t ),  0 },
    } );

    ComputePipelineDesc   pipeDesc{ m_rm->CreateShader( "shaders/compute/biology/vessel_mechanics.comp" ), "TestVesselMechanics" };
    ComputePipelineHandle pipeHandle = m_rm->CreatePipeline( pipeDesc );
    ComputePipeline*      pipeline   = m_rm->GetPipeline( pipeHandle );

    BindingGroup* bg = m_rm->GetBindingGroup( m_rm->CreateBindingGroup( pipeHandle, 0 ) );
    bg->Bind( 0, m_rm->GetBuffer( inBuf ) );
    bg->Bind( 1, m_rm->GetBuffer( outBuf ) );
    bg->Bind( 2, m_rm->GetBuffer( edgeBuf ) );
    bg->Bind( 3, m_rm->GetBuffer( edgeCountBuf ) );
    bg->Bind( 4, m_rm->GetBuffer( countBuf ) );
    bg->Build();

    // k=10, restLen=2, dt=1 → stretch=2 → force=10*2=20 per agent
    ComputePushConstants pc{};
    pc.dt          = 1.0f;
    pc.fParam0     = 10.0f; // springStiffness (k)
    pc.fParam1     = 2.0f;  // restingLength
    pc.offset      = 0;
    pc.maxCapacity = 2;
    pc.uParam0     = 0; // grpNdx

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

    std::vector<glm::vec4> result( 2 );
    m_stream->ReadbackBufferImmediate( outBuf, result.data(), agentBytes );

    // Agent 0 should have moved in +X (toward agent 1)
    EXPECT_GT( result[ 0 ].x, agentsIn[ 0 ].x ) << "Agent 0 should move toward agent 1 (+X)";
    // Agent 1 should have moved in -X (toward agent 0)
    EXPECT_LT( result[ 1 ].x, agentsIn[ 1 ].x ) << "Agent 1 should move toward agent 0 (-X)";
    // Gap between agents should be smaller than the initial 4 units
    float gap = result[ 1 ].x - result[ 0 ].x;
    EXPECT_LT( gap, 4.0f ) << "Spring force must reduce the gap between the two agents";
    // w-components (alive flags) must be preserved
    EXPECT_FLOAT_EQ( result[ 0 ].w, 1.0f );
    EXPECT_FLOAT_EQ( result[ 1 ].w, 1.0f );

    m_rm->DestroyBuffer( inBuf );
    m_rm->DestroyBuffer( outBuf );
    m_rm->DestroyBuffer( edgeBuf );
    m_rm->DestroyBuffer( edgeCountBuf );
    m_rm->DestroyBuffer( countBuf );
}
