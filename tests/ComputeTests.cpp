#include "SetupHelpers.h"
#include "resources/ResourceManager.h"
#include "resources/StreamingManager.h"
#include "rhi/BindingGroup.h"
#include "rhi/Buffer.h"
#include "rhi/Device.h"
#include "rhi/Pipeline.h"
#include "rhi/Queue.h"
#include "rhi/RHI.h"
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

// Helper method to readback 3D Grid from GPU into CPU memory
std::vector<float> ReadbackGrid( DigitalTwin::SimulationState& state, uint32_t fieldIndex, DigitalTwin::ResourceManager* rm,
                                 DigitalTwin::Device* device, DigitalTwin::StreamingManager* stream )
{
    // 1. Get the requested grid field from the simulation state
    auto& field = state.gridFields[ fieldIndex ];

    size_t voxelCount = field.width * field.height * field.depth;
    size_t byteSize   = voxelCount * sizeof( float );

    // Pre-allocate CPU memory for the incoming GPU data
    std::vector<float> resultData( voxelCount, 0.0f );

    // 2. Identify which texture is currently holding the readable data (Ping-Pong state logic)
    // This ensures we always read the most up-to-date integrated values after compute dispatches
    uint32_t      currentReadIndex = field.currentReadIndex;
    TextureHandle texToRead        = field.textures[ currentReadIndex ];

    // 3. Perform immediate synchronous readback using our StreamingManager pipeline
    stream->ReadbackTextureImmediate( texToRead, resultData.data(), byteSize );

    return resultData;
}

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
        << "layout(push_constant) uniform PC { float dt; float totalTime; float fParam1; float fParam2; uint offset; uint count; uint "
           "uParam1; uint uParam2; vec4 domainSize; uvec4 gridSize; } pc;\n"
        << "void main() { if(gl_GlobalInvocationID.x < pc.count) { pos[pc.offset + gl_GlobalInvocationID.x] += pc.fParam1; } }";
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
    std::vector<glm::vec4> agents     = { glm::vec4( 0.0f, 0.0f, 0.0f, 1.0f ) };
    size_t                 agentsSize = agents.size() * sizeof( glm::vec4 );

    // Grid: 10x10x10 = 1000 voxels (Atomic Target)
    uint32_t         voxelCount = 1000;
    std::vector<int> deltas( voxelCount, 0 );
    size_t           deltasSize = deltas.size() * sizeof( int );

    // 2. Allocate & Upload Buffers directly
    BufferHandle agentBuffer = m_rm->CreateBuffer( { agentsSize, BufferType::STORAGE, "TestAgentBuffer" } );
    BufferHandle deltaBuffer = m_rm->CreateBuffer( { deltasSize, BufferType::STORAGE, "TestDeltaBuffer" } );

    std::vector<BufferUploadRequest> uploads = { { agentBuffer, agents.data(), agentsSize, 0 }, { deltaBuffer, deltas.data(), deltasSize, 0 } };
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
    bg->Build();

    // 4. Setup Push Constants
    ComputePushConstants pc{};
    pc.dt         = 1.0f;
    pc.fParam1    = 15.0f; // Simulate 15.0 units of interaction
    pc.count      = 1;
    pc.offset     = 0;
    pc.domainSize = glm::vec4( 10.0f );
    pc.gridSize   = glm::uvec4( 10, 10, 10, 0 );

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
    pc.dt         = 1.0f;
    pc.fParam1    = 0.0f; // No diffusion (isolation test)
    pc.fParam2    = 0.0f; // No decay
    pc.count      = 0;
    pc.offset     = 0;
    pc.domainSize = glm::vec4( 10.0f );
    pc.gridSize   = glm::uvec4( width, height, depth, 0 );

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
    pc.dt      = 1.0f;
    pc.fParam1 = cellSize;
    pc.count   = agentCount;
    pc.offset  = 0;

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
    pc.dt     = 1.0f;
    pc.count  = agentCount;
    pc.offset = 0;

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
    pc.dt     = 1.0f;
    pc.count  = count;
    pc.offset = 0;

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
            pc.uParam1 = j;
            pc.uParam2 = k;

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

    std::vector<BufferUploadRequest> uploads = { { inAgentsBuf, inAgents.data(), agentsSize, 0 },
                                                 { hashesBuf, sortedHashes.data(), hashesSize, 0 },
                                                 { offsetsBuf, cellOffsets.data(), offsetsSize, 0 } };
    m_stream->UploadBufferImmediate( uploads );

    // 3. Setup Pipeline & BindingGroup directly via RHI
    ComputePipelineDesc pipeDesc{};
    pipeDesc.shader                  = m_rm->CreateShader( "shaders/compute/jkr_forces.comp" );
    pipeDesc.debugName               = "TestJKRForces";
    ComputePipelineHandle pipeHandle = m_rm->CreatePipeline( pipeDesc );
    ComputePipeline*      pipeline   = m_rm->GetPipeline( pipeHandle );

    BindingGroupHandle bgHandle = m_rm->CreateBindingGroup( pipeHandle, 0 );
    BindingGroup*      bg       = m_rm->GetBindingGroup( bgHandle );
    bg->Bind( 0, m_rm->GetBuffer( inAgentsBuf ) );
    bg->Bind( 1, m_rm->GetBuffer( outAgentsBuf ) );
    bg->Bind( 2, m_rm->GetBuffer( pressuresBuf ) );
    bg->Bind( 3, m_rm->GetBuffer( hashesBuf ) );
    bg->Bind( 4, m_rm->GetBuffer( offsetsBuf ) );
    bg->Build();

    // 4. Setup Push Constants (Packed exactly as in SimulationBuilder)
    ComputePushConstants pc{};
    pc.dt      = 0.016f; // Standard 60Hz frame time
    pc.count   = agentCount;
    pc.offset  = 0;
    pc.fParam1 = 50.0f; // Repulsion Stiffness
    pc.fParam2 = 0.0f;  // Adhesion Strength (ignore for this test)
    pc.uParam1 = offsetArraySize;
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
}

// =================================================================================================
// Compute task tests
// =================================================================================================

// 1. Test grid building
TEST_F( ComputeTest, Builder_GridAllocationAndUpload )
{
    // 1. Setup Blueprint with Domain and GridFields
    SimulationBlueprint blueprint;

    // Domain: 100x100x100 micrometers. Voxel size: 2 micrometers
    // Expected resolution: 50x50x50 voxels
    blueprint.SetDomainSize( glm::vec3( 100.0f ), 2.0f );

    blueprint.AddGridField( "Oxygen" )
        .SetInitializer( DigitalTwin::GridInitializer::Gaussian( glm::vec3( 0.0f ), 10.0f, 100.0f ) )
        .SetDiffusionCoefficient( 0.5f )
        .SetComputeHz( 120.0f );

    // 2. Build state (m_stream is already initialized by ComputeTestFixture)
    DigitalTwin::SimulationBuilder builder( m_rm.get(), m_stream.get() );
    SimulationState                state = builder.Build( blueprint );

    // 3. Verify allocations
    ASSERT_EQ( state.gridFields.size(), 1 ) << "Grid field was not added to the simulation state!";

    auto& oxygenState = state.gridFields[ 0 ];
    EXPECT_EQ( oxygenState.width, 50 );
    EXPECT_EQ( oxygenState.height, 50 );
    EXPECT_EQ( oxygenState.depth, 50 );

    // Ensure ping-pong 3D textures are valid and properly allocated on the device
    EXPECT_TRUE( oxygenState.textures[ 0 ].IsValid() ) << "Ping texture allocation failed!";
    EXPECT_TRUE( oxygenState.textures[ 1 ].IsValid() ) << "Pong texture allocation failed!";

    // Cleanup GPU resources gracefully
    state.Destroy( m_rm.get() );
}

// 2. Test proper grig field consumption
TEST_F( ComputeTest, Behaviour_ConsumeField )
{
    if( !m_device )
        GTEST_SKIP();

    SimulationBlueprint blueprint;
    blueprint.SetDomainSize( glm::vec3( 10.0f ), 1.0f ); // Small 10x10x10 grid for unit testing

    // Base Oxygen at 100.0, no background decay, slight diffusion
    blueprint.AddGridField( "Oxygen" )
        .SetInitializer( DigitalTwin::GridInitializer::Constant( 100.0f ) )
        .SetDiffusionCoefficient( 0.01f )
        .SetComputeHz( 1.0f ); // 1 execution per second to make dt=1.0 for easy math

    // Put exactly 1 agent dead in the center
    std::vector<glm::vec4> oneCell = { glm::vec4( 0.0f, 0.0f, 0.0f, 1.0f ) };

    blueprint.AddAgentGroup( "TestCell" )
        .SetCount( 1 )
        .SetDistribution( oneCell )
        .AddBehaviour( DigitalTwin::Behaviours::ConsumeField{ "Oxygen", 10.0f } ) // Eat 10 oxygen / sec
        .SetHz( 1.0f );                                                           // 1 execution per second

    DigitalTwin::SimulationBuilder builder( m_rm.get(), m_stream.get() );
    SimulationState                state = builder.Build( blueprint );

    // Grab execution graph
    auto compCtxHandle = m_device->CreateThreadContext( QueueType::COMPUTE );
    auto compCtx       = m_device->GetThreadContext( compCtxHandle );
    auto compCmd       = compCtx->GetCommandBuffer( compCtx->CreateCommandBuffer() );

    GraphDispatcher dispatcher;
    compCmd->Begin();
    // Ping pass (Reads from Texture 0, Writes to Texture 1)
    dispatcher.Dispatch( &state.computeGraph, compCmd, nullptr, 1.0f, 1.0f, 0 );
    // Pong pass (Reads from Texture 1, Writes to Texture 0)
    dispatcher.Dispatch( &state.computeGraph, compCmd, nullptr, 1.0f, 2.0f, 1 );
    compCmd->End();

    m_device->GetComputeQueue()->Submit( { compCmd } );
    m_device->GetComputeQueue()->WaitIdle();

    // Readback the grid using our updated StreamingManager pipeline
    std::vector<float> gridData = ReadbackGrid( state, 0, m_rm.get(), m_device.get(), m_stream.get() );

    // Center is at 5,5,5 in a 10x10x10 grid (using 0-based index)
    uint32_t centerIdx = 5 + 5 * 10 + 5 * 100;

    // Original was 100.0. After 2 effective integrations of consumption (Ping and Pong),
    // it should be heavily depleted. We expect it to be less than 95.0.
    EXPECT_LT( gridData[ centerIdx ], 95.0f ) << "Agent failed to consume the field or grid data is corrupted!";

    // A voxel far away (e.g. 0,0,0) should still be relatively untouched (close to 100.0)
    EXPECT_GT( gridData[ 0 ], 99.0f ) << "Diffusion spread too fast or entire grid is losing values!";

    state.Destroy( m_rm.get() );
}

// 3. Test grid field secretion
TEST_F( ComputeTest, Behaviour_SecreteField )
{
    if( !m_device )
        GTEST_SKIP();

    SimulationBlueprint blueprint;
    blueprint.SetDomainSize( glm::vec3( 10.0f ), 1.0f ); // Small 10x10x10 grid

    // Base Lactate at 0.0 (completely empty), slight diffusion
    blueprint.AddGridField( "Lactate" )
        .SetInitializer( DigitalTwin::GridInitializer::Constant( 0.0f ) )
        .SetDiffusionCoefficient( 0.01f )
        .SetComputeHz( 1.0f );

    // Put exactly 1 agent dead in the center
    std::vector<glm::vec4> oneCell = { glm::vec4( 0.0f, 0.0f, 0.0f, 1.0f ) };

    blueprint.AddAgentGroup( "TestCell" )
        .SetCount( 1 )
        .SetDistribution( oneCell )
        .AddBehaviour( DigitalTwin::Behaviours::SecreteField{ "Lactate", 10.0f } ) // Secrete 10 units / sec
        .SetHz( 1.0f );

    DigitalTwin::SimulationBuilder builder( m_rm.get(), m_stream.get() );
    SimulationState                state = builder.Build( blueprint );

    auto compCtxHandle = m_device->CreateThreadContext( QueueType::COMPUTE );
    auto compCtx       = m_device->GetThreadContext( compCtxHandle );
    auto compCmd       = compCtx->GetCommandBuffer( compCtx->CreateCommandBuffer() );

    GraphDispatcher dispatcher;
    compCmd->Begin();
    // Ping pass
    dispatcher.Dispatch( &state.computeGraph, compCmd, nullptr, 1.0f, 1.0f, 0 );
    // Pong pass
    dispatcher.Dispatch( &state.computeGraph, compCmd, nullptr, 1.0f, 2.0f, 1 );
    compCmd->End();

    m_device->GetComputeQueue()->Submit( { compCmd } );
    m_device->GetComputeQueue()->WaitIdle();

    // Readback the grid using our updated StreamingManager pipeline
    std::vector<float> gridData = ReadbackGrid( state, 0, m_rm.get(), m_device.get(), m_stream.get() );

    uint32_t centerIdx = 5 + 5 * 10 + 5 * 100;

    // Center should be heavily saturated with secreted lactate (>9.0 due to slight diffusion loss to neighbors)
    EXPECT_GT( gridData[ centerIdx ], 9.0f ) << "Agent failed to secrete the field properly!";

    // A voxel far away should still be completely clean (close to 0.0)
    EXPECT_LT( gridData[ 0 ], 0.1f ) << "Secretion affected distant voxels unexpectedly!";

    state.Destroy( m_rm.get() );
}

// 4. Test pure grid field diffusion (No Agents)
TEST_F( ComputeTest, Behaviour_PureDiffusion )
{
    if( !m_device )
        GTEST_SKIP();

    SimulationBlueprint blueprint;
    blueprint.SetDomainSize( glm::vec3( 10.0f ), 1.0f ); // 10x10x10 grid

    // Initialize with a Gaussian blob in the center so we have a natural concentration gradient.
    // A high diffusion coefficient ensures it spreads noticeably in just a few frames.
    blueprint.AddGridField( "Morphogen" )
        .SetInitializer( DigitalTwin::GridInitializer::Gaussian( glm::vec3( 0.0f ), 2.0f, 100.0f ) )
        .SetDiffusionCoefficient( 2.5f )
        .SetComputeHz( 10.0f ); // 10 executions per second

    // NOTICE: We specifically do NOT add any AgentGroups.
    // We are testing pure environmental PDE physics here.

    DigitalTwin::SimulationBuilder builder( m_rm.get(), m_stream.get() );
    SimulationState                state = builder.Build( blueprint );

    // 1. Readback the INITIAL state of the grid before any compute dispatches
    std::vector<float> initialGridData = ReadbackGrid( state, 0, m_rm.get(), m_device.get(), m_stream.get() );

    // Center is at 5,5,5. We will also track a voxel slightly off-center (e.g., 2,2,2)
    // to observe the mass flowing from the peak into the valleys.
    uint32_t centerIdx = 5 + 5 * 10 + 5 * 100;
    uint32_t edgeIdx   = 2 + 2 * 10 + 2 * 100;

    float initialCenterValue = initialGridData[ centerIdx ];
    float initialEdgeValue   = initialGridData[ edgeIdx ];

    // 2. Setup compute dispatch
    auto compCtxHandle = m_device->CreateThreadContext( QueueType::COMPUTE );
    auto compCtx       = m_device->GetThreadContext( compCtxHandle );
    auto compCmd       = compCtx->GetCommandBuffer( compCtx->CreateCommandBuffer() );

    GraphDispatcher dispatcher;
    compCmd->Begin();

    // Simulate 10 frames (1 full second at 10Hz).
    // This perfectly stress-tests the texture Ping-Pong mechanism (i % 2).
    for( int i = 0; i < 10; ++i )
    {
        float totalTime = static_cast<float>( i ) * 0.1f;
        dispatcher.Dispatch( &state.computeGraph, compCmd, nullptr, 0.1f, totalTime, i % 2 );
    }

    compCmd->End();

    m_device->GetComputeQueue()->Submit( { compCmd } );
    m_device->GetComputeQueue()->WaitIdle();

    // 3. Readback the FINAL state of the grid after diffusion
    std::vector<float> finalGridData = ReadbackGrid( state, 0, m_rm.get(), m_device.get(), m_stream.get() );

    float finalCenterValue = finalGridData[ centerIdx ];
    float finalEdgeValue   = finalGridData[ edgeIdx ];

    // Assertions:
    // The central peak must have lost concentration as it diffused outwards.
    EXPECT_LT( finalCenterValue, initialCenterValue ) << "Diffusion failed to reduce the peak concentration!";

    // The outer voxels must have gained concentration as the substance spread to them.
    EXPECT_GT( finalEdgeValue, initialEdgeValue ) << "Diffusion failed to increase concentration in outer regions!";

    state.Destroy( m_rm.get() );
}

// 5. Test proper biomechanics build
TEST_F( ComputeTest, Builder_BiomechanicsAllocation )
{
    // 1. Setup Blueprint
    DigitalTwin::SimulationBlueprint blueprint;
    blueprint.SetDomainSize( glm::vec3( 100.0f ), 2.0f );

    // Add a group of cells and attach the Biomechanics behaviour
    std::vector<glm::vec4> dummyCells = { glm::vec4( 0.0f, 0.0f, 0.0f, 1.0f ) };

    blueprint.AddAgentGroup( "TestTissue" )
        .SetCount( 1 )
        .SetDistribution( dummyCells )
        .AddBehaviour( DigitalTwin::Behaviours::Biomechanics{ 15.0f, 2.0f, 1.5f } )
        .SetHz( 60.0f );

    // 2. Build state
    DigitalTwin::SimulationBuilder builder( m_rm.get(), m_stream.get() );
    DigitalTwin::SimulationState   state = builder.Build( blueprint );

    // 3. Verify allocations
    // Biomechanics requires specific storage buffers for spatial hashing and physics.
    // We check if SimulationBuilder successfully detected the behaviour and allocated them.
    EXPECT_TRUE( state.hashBuffer.IsValid() ) << "Biomechanics hash buffer was not allocated!";
    EXPECT_TRUE( state.offsetBuffer.IsValid() ) << "Biomechanics offset buffer was not allocated!";
    EXPECT_TRUE( state.pressureBuffer.IsValid() ) << "Biomechanics pressure buffer was not allocated!";

    // Cleanup GPU resources gracefully
    state.Destroy( m_rm.get() );
}

// 6. Test biomechanics integration
TEST_F( ComputeTest, Behaviour_Biomechanics_Integration )
{
    if( !m_device )
        GTEST_SKIP();

    DigitalTwin::SimulationBlueprint blueprint;
    blueprint.SetDomainSize( glm::vec3( 100.0f ), 2.0f )
        .ConfigureSpatialPartitioning()
        .SetMethod( DigitalTwin::SpatialPartitioningMethod::HashGrid )
        .SetCellSize( 30.0f )
        .SetMaxDensity( 64 )
        .SetComputeHz( 60.0f );

    // Place two agents severely overlapping in physical space.
    // Distance between them is 0.1, while interaction radius is 1.5 (diameter 3.0).
    std::vector<glm::vec4> collidingCells = {
        glm::vec4( 0.0f, 0.0f, 0.0f, 1.0f ), // Agent 0 at origin
        glm::vec4( 0.1f, 0.0f, 0.0f, 1.0f )  // Agent 1 slightly to the right
    };

    blueprint.AddAgentGroup( "Colliders" )
        .SetCount( 2 )
        .SetDistribution( collidingCells )
        .AddBehaviour( DigitalTwin::BiomechanicsGenerator::JKR()
                           .SetYoungsModulus( 50.0f )
                           .SetPoissonRatio( 0.4f )
                           .SetAdhesionEnergy( 0.0f )
                           .SetMaxInteractionRadius( 1.5f )
                           .Build() )
        .SetHz( 60.0f );

    DigitalTwin::SimulationBuilder builder( m_rm.get(), m_stream.get() );
    DigitalTwin::SimulationState   state = builder.Build( blueprint );

    auto compCtxHandle = m_device->CreateThreadContext( QueueType::COMPUTE );
    auto compCtx       = m_device->GetThreadContext( compCtxHandle );
    auto compCmd       = compCtx->GetCommandBuffer( compCtx->CreateCommandBuffer() );

    DigitalTwin::GraphDispatcher dispatcher;

    // Dispatch 1 frame of physics.
    // activeIndex = 0 means it reads from agentBuffers[0] and writes results to agentBuffers[1]
    compCmd->Begin();
    dispatcher.Dispatch( &state.computeGraph, compCmd, nullptr, 0.02f, 0.02f, 0 );
    compCmd->End();

    m_device->GetComputeQueue()->Submit( { compCmd } );
    m_device->GetComputeQueue()->WaitIdle();

    // Readback Agent Positions from the output buffer (index 1)
    std::vector<glm::vec4> resultPositions( 2 );
    m_stream->ReadbackBufferImmediate( state.agentBuffers[ 1 ], resultPositions.data(), 2 * sizeof( glm::vec4 ) );

    // Readback Pressures from the pressure buffer
    std::vector<float> resultPressures( 2 );
    m_stream->ReadbackBufferImmediate( state.pressureBuffer, resultPressures.data(), 2 * sizeof( float ) );

    // Verify Physics Logic:

    // Cell A (index 0) should have been pushed negatively in X
    EXPECT_LT( resultPositions[ 0 ].x, 0.0f ) << "Agent 0 was not repelled correctly to the left!";

    // Cell B (index 1) should have been pushed positively in X (away from A, beyond its initial 0.1f)
    EXPECT_GT( resultPositions[ 1 ].x, 0.1f ) << "Agent 1 was not repelled correctly to the right!";

    // Verify Newton's Third Law (Every action has an equal and opposite reaction)
    EXPECT_GT( resultPressures[ 0 ], 0.0f ) << "Agent 0 did not register collision pressure!";
    EXPECT_GT( resultPressures[ 1 ], 0.0f ) << "Agent 1 did not register collision pressure!";
    EXPECT_FLOAT_EQ( resultPressures[ 0 ], resultPressures[ 1 ] ) << "Newton's Third Law violated: Pressures unequal!";

    state.Destroy( m_rm.get() );
}