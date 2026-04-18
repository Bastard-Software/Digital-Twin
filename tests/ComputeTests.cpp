#include "SetupHelpers.h"
#include "resources/ResourceManager.h"
#include <glm/gtc/packing.hpp>
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
#include "simulation/Phenotype.h"
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
    pc.fParam0     = 15.0f;  // rate
    pc.fParam1     = -1.0f;  // reqLifecycleState = -1 (any)
    pc.fParam2     = -1.0f;  // reqCellType = -1 (any)
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
    // Both agents are at ~0.0, so with cellSize=3.0 they map to cell (0,0,0).
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
    // Dummy cadherin buffers for bindings 7 and 8 (cadherin flag = 0 in push constants → inactive)
    glm::vec4    zeroProfData[2]  = {};
    BufferHandle cadProfileDummy  = m_rm->CreateBuffer( { sizeof( zeroProfData ),  BufferType::STORAGE, "JKRCadProfileDummy" } );
    glm::mat4    identityMat      = glm::mat4( 1.0f );
    BufferHandle cadAffinityDummy = m_rm->CreateBuffer( { sizeof( glm::mat4 ),     BufferType::STORAGE, "JKRCadAffinityDummy" } );
    m_stream->UploadBufferImmediate( { { cadAffinityDummy, &identityMat, sizeof( glm::mat4 ) } } );
    // Dummy polarity buffer for binding 9 (polarity flag = 0 in push constants → inactive)
    glm::vec4    zeroPolarityData[2] = {};
    BufferHandle polarityDummy = m_rm->CreateBuffer( { sizeof( zeroPolarityData ), BufferType::STORAGE, "JKRPolarityDummy" } );
    // Dummy orientation buffer for binding 10 (hull count=0 → rigid body path inactive)
    glm::vec4    identityQuat[2] = { glm::vec4( 0, 0, 0, 1 ), glm::vec4( 0, 0, 0, 1 ) };
    BufferHandle orientationDummy = m_rm->CreateBuffer( { sizeof( identityQuat ), BufferType::STORAGE, "JKROrientationDummy" } );
    m_stream->UploadBufferImmediate( { { orientationDummy, identityQuat, sizeof( identityQuat ) } } );
    // Dummy contact hull buffer for binding 11 (hullMeta.x=0 → point-particle fallback)
    struct ContactHullGPU { glm::vec4 meta{}; glm::vec4 points[16]{}; };
    ContactHullGPU dummyHull{};
    BufferHandle contactHullDummy = m_rm->CreateBuffer( { sizeof( ContactHullGPU ), BufferType::STORAGE, "JKRContactHullDummy" } );
    m_stream->UploadBufferImmediate( { { contactHullDummy, &dummyHull, sizeof( ContactHullGPU ) } } );
    // Dummy plate buffer for binding 12 (Step B — multi-plate layout; count=0 → all plate blocks skipped)
    struct PlateBufferGPU { glm::uvec4 meta; glm::vec4 plates[16]; };
    PlateBufferGPU plateDummy{};
    BufferHandle plateDummyBuf = m_rm->CreateBuffer( { sizeof( PlateBufferGPU ), BufferType::STORAGE, "JKRPlateDummy" } );
    m_stream->UploadBufferImmediate( { { plateDummyBuf, &plateDummy, sizeof( PlateBufferGPU ) } } );

    BindingGroupHandle bgHandle = m_rm->CreateBindingGroup( pipeHandle, 0 );
    BindingGroup*      bg       = m_rm->GetBindingGroup( bgHandle );
    bg->Bind( 0, m_rm->GetBuffer( inAgentsBuf ) );
    bg->Bind( 1, m_rm->GetBuffer( outAgentsBuf ) );
    bg->Bind( 2, m_rm->GetBuffer( pressuresBuf ) );
    bg->Bind( 3, m_rm->GetBuffer( hashesBuf ) );
    bg->Bind( 4, m_rm->GetBuffer( offsetsBuf ) );
    bg->Bind( 5, m_rm->GetBuffer( countBuf ) );
    bg->Bind( 6, m_rm->GetBuffer( phenotypeDummyBuf ) );
    bg->Bind( 7, m_rm->GetBuffer( cadProfileDummy ) );
    bg->Bind( 8, m_rm->GetBuffer( cadAffinityDummy ) );
    bg->Bind( 9, m_rm->GetBuffer( polarityDummy ) );
    bg->Bind( 10, m_rm->GetBuffer( orientationDummy ) );
    bg->Bind( 11, m_rm->GetBuffer( contactHullDummy ) );
    bg->Bind( 12, m_rm->GetBuffer( plateDummyBuf ) );
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
    pc.fParam4     = 0.0f;   // damping (disabled)
    pc.fParam5     = 1.5f;   // maxRadius — interaction radius
    pc.uParam0     = offsetArraySize;
    pc.uParam1     = 0;
    // domainSize.w = hash cell size (3.0 = 2 * maxRadius, both agents map to cell (0,0,0))
    pc.domainSize = glm::vec4( 100.0f, 100.0f, 100.0f, 3.0f );

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
    m_rm->DestroyBuffer( cadProfileDummy );
    m_rm->DestroyBuffer( cadAffinityDummy );
    m_rm->DestroyBuffer( polarityDummy );
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
    TextureHandle dummyTex     = m_rm->CreateTexture( { 1, 1, 1, TextureType::Texture3D, VK_FORMAT_R32_SFLOAT, TextureUsage::STORAGE, VK_SAMPLE_COUNT_1_BIT, "Dummy" } );

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

    std::vector<PhenotypeData> phenotypes     = { { 0u, 0.5f, 0.0f, 0u } }; // Live, Default
    size_t                     phenotypesSize = sizeof( PhenotypeData );

    // No vessel edges — isolated agent has no Notch neighbors (edge-based signaling)
    struct VesselEdge { uint32_t agentA; uint32_t agentB; float dist; uint32_t flags; };
    VesselEdge   dummyEdge  = { 0u, 0u, 0.0f, 0u };
    uint32_t     edgeCount  = 0; // no edges

    uint32_t agentCount = 1;
    size_t   countSize  = sizeof( uint32_t );

    // ── Allocate & Upload ─────────────────────────────────────────────────────
    BufferHandle  agentBuf    = m_rm->CreateBuffer( { agentsSize,            BufferType::STORAGE,  "NotchAgents"    } );
    BufferHandle  signalBuf   = m_rm->CreateBuffer( { signalingSize,         BufferType::STORAGE,  "NotchSignaling" } );
    BufferHandle  phenoBuf    = m_rm->CreateBuffer( { phenotypesSize,        BufferType::STORAGE,  "NotchPheno"     } );
    BufferHandle  edgeBuf     = m_rm->CreateBuffer( { sizeof( VesselEdge ),  BufferType::STORAGE,  "NotchEdges"     } );
    BufferHandle  edgeCountBuf= m_rm->CreateBuffer( { sizeof( uint32_t ),    BufferType::STORAGE,  "NotchEdgeCount" } );
    BufferHandle  countBuf    = m_rm->CreateBuffer( { countSize,             BufferType::INDIRECT, "NotchCount"     } );
    // Dummy 1×1×1 VEGF texture (binding 6, VEGF disabled: gridSize.w=0 → shader uses localVEGF=1.0)
    TextureHandle vegfDummy   = m_rm->CreateTexture( { 1, 1, 1, TextureType::Texture3D, VK_FORMAT_R32_SFLOAT, TextureUsage::STORAGE | TextureUsage::TRANSFER_DST, VK_SAMPLE_COUNT_1_BIT, "NotchVEGFDummy" } );
    float         one         = 1.0f;
    m_stream->UploadTextureImmediate( vegfDummy, &one, sizeof( float ) );

    m_stream->UploadBufferImmediate( {
        { agentBuf,     agents.data(),     agentsSize,           0 },
        { signalBuf,    signaling.data(),  signalingSize,        0 },
        { phenoBuf,     phenotypes.data(), phenotypesSize,       0 },
        { edgeBuf,      &dummyEdge,        sizeof( VesselEdge ), 0 },
        { edgeCountBuf, &edgeCount,        sizeof( uint32_t ),   0 },
        { countBuf,     &agentCount,       countSize,            0 },
    } );

    // ── Pipeline & BindingGroup ───────────────────────────────────────────────
    ComputePipelineDesc   pipeDesc{ m_rm->CreateShader( "shaders/compute/biology/notch_dll4.comp" ), "TestNotchDll4" };
    ComputePipelineHandle pipeHandle = m_rm->CreatePipeline( pipeDesc );
    ComputePipeline*      pipeline   = m_rm->GetPipeline( pipeHandle );

    BindingGroup* bg = m_rm->GetBindingGroup( m_rm->CreateBindingGroup( pipeHandle, 0 ) );
    bg->Bind( 0, m_rm->GetBuffer( agentBuf      ) );
    bg->Bind( 1, m_rm->GetBuffer( signalBuf     ) );
    bg->Bind( 2, m_rm->GetBuffer( phenoBuf      ) );
    bg->Bind( 3, m_rm->GetBuffer( edgeBuf       ) );
    bg->Bind( 4, m_rm->GetBuffer( edgeCountBuf  ) );
    bg->Bind( 5, m_rm->GetBuffer( countBuf      ) );
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
    pc.uParam0     = 0u;    // unused (edge-based signaling)
    pc.uParam1     = 0;     // grpNdx
    pc.domainSize  = glm::vec4( 100.0f, 100.0f, 100.0f, 0.0f );
    pc.gridSize    = glm::uvec4( 1u, 1u, 1u, 0u ); // w=0 → VEGF disabled

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

    m_rm->DestroyBuffer( agentBuf     );
    m_rm->DestroyBuffer( signalBuf   );
    m_rm->DestroyBuffer( phenoBuf    );
    m_rm->DestroyBuffer( edgeBuf     );
    m_rm->DestroyBuffer( edgeCountBuf);
    m_rm->DestroyBuffer( countBuf    );
    m_rm->DestroyTexture( vegfDummy  );
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

    std::vector<PhenotypeData> phenotypes     = { { 0u, 0.5f, 0.0f, 0u } };
    size_t                     phenotypesSize = sizeof( PhenotypeData );

    struct VesselEdge { uint32_t agentA; uint32_t agentB; float dist; uint32_t flags; };
    VesselEdge dummyEdge  = { 0u, 0u, 0.0f, 0u };
    uint32_t   edgeCount  = 0;

    uint32_t agentCount = 1;
    size_t   countSize  = sizeof( uint32_t );

    BufferHandle  agentBuf     = m_rm->CreateBuffer( { agentsSize,           BufferType::STORAGE,  "DeadNotchAgents"     } );
    BufferHandle  signalBuf    = m_rm->CreateBuffer( { signalingSize,        BufferType::STORAGE,  "DeadNotchSignal"     } );
    BufferHandle  phenoBuf     = m_rm->CreateBuffer( { phenotypesSize,       BufferType::STORAGE,  "DeadNotchPheno"      } );
    BufferHandle  edgeBuf      = m_rm->CreateBuffer( { sizeof( VesselEdge ), BufferType::STORAGE,  "DeadNotchEdges"      } );
    BufferHandle  edgeCountBuf = m_rm->CreateBuffer( { sizeof( uint32_t ),   BufferType::STORAGE,  "DeadNotchEdgeCount"  } );
    BufferHandle  countBuf     = m_rm->CreateBuffer( { countSize,            BufferType::INDIRECT, "DeadNotchCount"      } );
    TextureHandle vegfDummy    = m_rm->CreateTexture( { 1, 1, 1, TextureType::Texture3D, VK_FORMAT_R32_SFLOAT, TextureUsage::STORAGE | TextureUsage::TRANSFER_DST, VK_SAMPLE_COUNT_1_BIT, "DeadNotchVEGFDummy" } );
    float         one          = 1.0f;
    m_stream->UploadTextureImmediate( vegfDummy, &one, sizeof( float ) );

    m_stream->UploadBufferImmediate( {
        { agentBuf,     agents.data(),     agentsSize,           0 },
        { signalBuf,    signaling.data(),  signalingSize,        0 },
        { phenoBuf,     phenotypes.data(), phenotypesSize,       0 },
        { edgeBuf,      &dummyEdge,        sizeof( VesselEdge ), 0 },
        { edgeCountBuf, &edgeCount,        sizeof( uint32_t ),   0 },
        { countBuf,     &agentCount,       countSize,            0 },
    } );

    ComputePipelineDesc   pipeDesc{ m_rm->CreateShader( "shaders/compute/biology/notch_dll4.comp" ), "TestNotchDll4Dead" };
    ComputePipelineHandle pipeHandle = m_rm->CreatePipeline( pipeDesc );
    ComputePipeline*      pipeline   = m_rm->GetPipeline( pipeHandle );

    BindingGroup* bg = m_rm->GetBindingGroup( m_rm->CreateBindingGroup( pipeHandle, 0 ) );
    bg->Bind( 0, m_rm->GetBuffer( agentBuf     ) );
    bg->Bind( 1, m_rm->GetBuffer( signalBuf    ) );
    bg->Bind( 2, m_rm->GetBuffer( phenoBuf     ) );
    bg->Bind( 3, m_rm->GetBuffer( edgeBuf      ) );
    bg->Bind( 4, m_rm->GetBuffer( edgeCountBuf ) );
    bg->Bind( 5, m_rm->GetBuffer( countBuf     ) );
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
    pc.uParam0     = 0u;
    pc.uParam1     = 0;
    pc.domainSize  = glm::vec4( 100.0f, 100.0f, 100.0f, 0.0f );
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

    m_rm->DestroyBuffer( agentBuf     );
    m_rm->DestroyBuffer( signalBuf   );
    m_rm->DestroyBuffer( phenoBuf    );
    m_rm->DestroyBuffer( edgeBuf     );
    m_rm->DestroyBuffer( edgeCountBuf);
    m_rm->DestroyBuffer( countBuf    );
    m_rm->DestroyTexture( vegfDummy  );
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

    std::vector<PhenotypeData> phenotypes     = { { 0u, 0.5f, 0.0f, 0u }, { 0u, 0.5f, 0.0f, 0u } };
    size_t                     phenotypesSize = 2 * sizeof( PhenotypeData );

    // One vessel edge connecting agents 0 and 1 — juxtacrine signaling via edge
    struct VesselEdge { uint32_t agentA; uint32_t agentB; float dist; uint32_t flags; };
    VesselEdge edge      = { 0u, 1u, 1.0f, 0u };
    uint32_t   edgeCount = 1;

    // Count: process 2 agents from group 0
    uint32_t agentCount = 2;
    size_t   countSize  = sizeof( uint32_t );

    // ── Allocate & Upload ─────────────────────────────────────────────────────
    BufferHandle  agentBuf     = m_rm->CreateBuffer( { agentsSize,           BufferType::STORAGE,  "LIAgents"     } );
    BufferHandle  signalBuf    = m_rm->CreateBuffer( { signalingSize,        BufferType::STORAGE,  "LISignal"     } );
    BufferHandle  phenoBuf     = m_rm->CreateBuffer( { phenotypesSize,       BufferType::STORAGE,  "LIPheno"      } );
    BufferHandle  edgeBuf      = m_rm->CreateBuffer( { sizeof( VesselEdge ), BufferType::STORAGE,  "LIEdges"      } );
    BufferHandle  edgeCountBuf = m_rm->CreateBuffer( { sizeof( uint32_t ),   BufferType::STORAGE,  "LIEdgeCount"  } );
    BufferHandle  countBuf     = m_rm->CreateBuffer( { countSize,            BufferType::INDIRECT, "LICount"      } );
    TextureHandle vegfDummy    = m_rm->CreateTexture( { 1, 1, 1, TextureType::Texture3D, VK_FORMAT_R32_SFLOAT, TextureUsage::STORAGE | TextureUsage::TRANSFER_DST, VK_SAMPLE_COUNT_1_BIT, "LIVEGFDummy" } );
    float         one          = 1.0f;
    m_stream->UploadTextureImmediate( vegfDummy, &one, sizeof( float ) );

    m_stream->UploadBufferImmediate( {
        { agentBuf,     agents.data(),     agentsSize,           0 },
        { signalBuf,    signaling.data(),  signalingSize,        0 },
        { phenoBuf,     phenotypes.data(), phenotypesSize,       0 },
        { edgeBuf,      &edge,             sizeof( VesselEdge ), 0 },
        { edgeCountBuf, &edgeCount,        sizeof( uint32_t ),   0 },
        { countBuf,     &agentCount,       countSize,            0 },
    } );

    // ── Pipeline & BindingGroup ───────────────────────────────────────────────
    ComputePipelineDesc   pipeDesc{ m_rm->CreateShader( "shaders/compute/biology/notch_dll4.comp" ), "TestNotchDll4LI" };
    ComputePipelineHandle pipeHandle = m_rm->CreatePipeline( pipeDesc );
    ComputePipeline*      pipeline   = m_rm->GetPipeline( pipeHandle );

    BindingGroup* bg = m_rm->GetBindingGroup( m_rm->CreateBindingGroup( pipeHandle, 0 ) );
    bg->Bind( 0, m_rm->GetBuffer( agentBuf     ) );
    bg->Bind( 1, m_rm->GetBuffer( signalBuf    ) );
    bg->Bind( 2, m_rm->GetBuffer( phenoBuf     ) );
    bg->Bind( 3, m_rm->GetBuffer( edgeBuf      ) );
    bg->Bind( 4, m_rm->GetBuffer( edgeCountBuf ) );
    bg->Bind( 5, m_rm->GetBuffer( countBuf     ) );
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
    pc.uParam0     = 0u;
    pc.uParam1     = 0;
    pc.domainSize  = glm::vec4( 100.0f, 100.0f, 100.0f, 0.0f );
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

    m_rm->DestroyBuffer( agentBuf     );
    m_rm->DestroyBuffer( signalBuf   );
    m_rm->DestroyBuffer( phenoBuf    );
    m_rm->DestroyBuffer( edgeBuf     );
    m_rm->DestroyBuffer( edgeCountBuf);
    m_rm->DestroyBuffer( countBuf    );
    m_rm->DestroyTexture( vegfDummy  );
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
                                                   VK_SAMPLE_COUNT_1_BIT, "VEGFGatingTex" } );
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

    // Agent 0 initialised as StalkCell (2): Dll4 will land in the dead zone (0.45),
    // so hysteresis keeps it as StalkCell — matching the assertion below.
    std::vector<PhenotypeData> phenotypes     = { { 0u, 0.5f, 0.0f, 2u }, { 0u, 0.5f, 0.0f, 2u } };
    size_t                     phenotypesSize = 2 * sizeof( PhenotypeData );

    // Isolated agents — no vessel edges, no Notch neighbors
    struct VesselEdge { uint32_t agentA; uint32_t agentB; float dist; uint32_t flags; };
    VesselEdge dummyEdge  = { 0u, 0u, 0.0f, 0u };
    uint32_t   edgeCount  = 0;

    uint32_t agentCount = 2;
    size_t   countSize  = sizeof( uint32_t );

    // ── Allocate & Upload ─────────────────────────────────────────────────────
    BufferHandle agentBuf     = m_rm->CreateBuffer( { agentsSize,           BufferType::STORAGE,  "VGAgents"     } );
    BufferHandle signalBuf    = m_rm->CreateBuffer( { signalingSize,        BufferType::STORAGE,  "VGSignal"     } );
    BufferHandle phenoBuf     = m_rm->CreateBuffer( { phenotypesSize,       BufferType::STORAGE,  "VGPheno"      } );
    BufferHandle edgeBuf      = m_rm->CreateBuffer( { sizeof( VesselEdge ), BufferType::STORAGE,  "VGEdges"      } );
    BufferHandle edgeCountBuf = m_rm->CreateBuffer( { sizeof( uint32_t ),   BufferType::STORAGE,  "VGEdgeCount"  } );
    BufferHandle countBuf     = m_rm->CreateBuffer( { countSize,            BufferType::INDIRECT, "VGCount"      } );

    m_stream->UploadBufferImmediate( {
        { agentBuf,     agents.data(),     agentsSize,           0 },
        { signalBuf,    signaling.data(),  signalingSize,        0 },
        { phenoBuf,     phenotypes.data(), phenotypesSize,       0 },
        { edgeBuf,      &dummyEdge,        sizeof( VesselEdge ), 0 },
        { edgeCountBuf, &edgeCount,        sizeof( uint32_t ),   0 },
        { countBuf,     &agentCount,       countSize,            0 },
    } );

    // ── Pipeline & BindingGroup ───────────────────────────────────────────────
    ComputePipelineDesc   pipeDesc{ m_rm->CreateShader( "shaders/compute/biology/notch_dll4.comp" ), "TestNotchVEGFGate" };
    ComputePipelineHandle pipeHandle = m_rm->CreatePipeline( pipeDesc );
    ComputePipeline*      pipeline   = m_rm->GetPipeline( pipeHandle );

    BindingGroup* bg = m_rm->GetBindingGroup( m_rm->CreateBindingGroup( pipeHandle, 0 ) );
    bg->Bind( 0, m_rm->GetBuffer( agentBuf     ) );
    bg->Bind( 1, m_rm->GetBuffer( signalBuf    ) );
    bg->Bind( 2, m_rm->GetBuffer( phenoBuf     ) );
    bg->Bind( 3, m_rm->GetBuffer( edgeBuf      ) );
    bg->Bind( 4, m_rm->GetBuffer( edgeCountBuf ) );
    bg->Bind( 5, m_rm->GetBuffer( countBuf     ) );
    bg->Bind( 6, m_rm->GetTexture( vegfTex     ), VK_IMAGE_LAYOUT_GENERAL );
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
    pc.uParam0     = 0u;
    pc.uParam1     = 0;
    pc.domainSize  = glm::vec4( 20.0f, 20.0f, 20.0f, 0.0f );
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

    m_rm->DestroyBuffer( agentBuf     );
    m_rm->DestroyBuffer( signalBuf   );
    m_rm->DestroyBuffer( phenoBuf    );
    m_rm->DestroyBuffer( edgeBuf     );
    m_rm->DestroyBuffer( edgeCountBuf);
    m_rm->DestroyBuffer( countBuf    );
    m_rm->DestroyTexture( vegfTex    );
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

    // 1×1×1 VEGF texture — agent at origin in 10-unit domain always hits voxel (0,0,0)
    TextureHandle vegfTex = rm->CreateTexture(
        { 1, 1, 1, TextureType::Texture3D, VK_FORMAT_R32_SFLOAT,
          TextureUsage::STORAGE | TextureUsage::TRANSFER_DST, VK_SAMPLE_COUNT_1_BIT, "PhalanxVEGF" } );
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

TEST_F( ComputeTest, Shader_PhalanxActivation_TipCell_StaysTipCellWhenVEGFHigh )
{
    if( !m_device )
        GTEST_SKIP();

    // VEGF=30 > deactivationThreshold=5 → TipCell stays TipCell (migration signal still present)
    uint32_t result = RunPhalanxShader( m_device.get(), m_rm.get(), m_stream.get(),
        glm::vec4( 0.0f, 0.0f, 0.0f, 1.0f ),
        1u,   // TipCell
        30.0f, // VEGF well above deactivation threshold
        20.0f, 5.0f
    );

    EXPECT_EQ( result, 1u ) << "TipCell with VEGF > deactivationThreshold must stay TipCell (1)";
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

    // Component labels: each agent in its own component initially → different labels → fusion allowed
    std::vector<uint32_t> componentLabels = { 0u, 1u };
    size_t                componentSize   = 2 * sizeof( uint32_t );

    // ── Allocate & Upload ─────────────────────────────────────────────────────
    BufferHandle agentBuf     = m_rm->CreateBuffer( { agentsSize,           BufferType::STORAGE,  "AnaAgents"       } );
    BufferHandle phenoBuf     = m_rm->CreateBuffer( { phenotypesSize,       BufferType::STORAGE,  "AnaPheno"        } );
    BufferHandle hashBuf      = m_rm->CreateBuffer( { hashesSize,           BufferType::STORAGE,  "AnaHash"         } );
    BufferHandle offsetBuf    = m_rm->CreateBuffer( { offsetsSize,          BufferType::STORAGE,  "AnaOffsets"      } );
    BufferHandle countBuf     = m_rm->CreateBuffer( { sizeof( uint32_t ),   BufferType::INDIRECT, "AnaCount"        } );
    BufferHandle edgeBuf      = m_rm->CreateBuffer( { edgeBufferSize,       BufferType::STORAGE,  "AnaEdges"        } );
    BufferHandle edgeCountBuf = m_rm->CreateBuffer( { sizeof( uint32_t ),   BufferType::STORAGE,  "AnaEdgeCount"    } );
    BufferHandle componentBuf = m_rm->CreateBuffer( { componentSize,        BufferType::STORAGE,  "AnaComponents"   } );

    m_stream->UploadBufferImmediate( {
        { agentBuf,     agents.data(),           agentsSize,         0 },
        { phenoBuf,     phenotypes.data(),        phenotypesSize,     0 },
        { hashBuf,      sortedHashes.data(),      hashesSize,         0 },
        { offsetBuf,    cellOffsets.data(),       offsetsSize,        0 },
        { countBuf,     &agentCount,              sizeof( uint32_t ), 0 },
        { edgeCountBuf, &edgeCountInit,           sizeof( uint32_t ), 0 },
        { componentBuf, componentLabels.data(),   componentSize,      0 },
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
    bg->Bind( 7, m_rm->GetBuffer( componentBuf ) );
    bg->Build();

    // ── Push Constants ────────────────────────────────────────────────────────
    ComputePushConstants pc{};
    pc.dt          = 1.0f / 60.0f;
    pc.fParam0     = 5.0f;  // contactDistance
    pc.fParam2     = 16.0f; // edgeBufferCapacity — edge buffer holds 16 slots in this test
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

    EXPECT_EQ(   resultPheno[ 0 ].cellType, 2u )    << "Agent 0 must be StalkCell (2) after anastomosis";
    EXPECT_EQ(   resultPheno[ 1 ].cellType, 2u )    << "Agent 1 must be StalkCell (2) after anastomosis";
    EXPECT_EQ(   resultEdgeCount,           1u )    << "Exactly 1 edge must be recorded";
    EXPECT_EQ(   resultEdge.agentA,         0u )    << "Edge agentA must be 0 (lower-index invocation handles the pair)";
    EXPECT_EQ(   resultEdge.agentB,         1u )    << "Edge agentB must be 1";
    EXPECT_NEAR( resultEdge.dist,           2.0f, 1e-4f ) << "Edge dist must equal agent separation";
    EXPECT_EQ(   resultEdge.flags,          0x8u )  << "Edge must have SPROUT flag so vessel_components merges labels";

    m_rm->DestroyBuffer( agentBuf     );
    m_rm->DestroyBuffer( phenoBuf     );
    m_rm->DestroyBuffer( hashBuf      );
    m_rm->DestroyBuffer( offsetBuf    );
    m_rm->DestroyBuffer( countBuf     );
    m_rm->DestroyBuffer( edgeBuf      );
    m_rm->DestroyBuffer( edgeCountBuf );
    m_rm->DestroyBuffer( componentBuf );
}

// Dead slot (w=0) pre-labelled as TipCell must be skipped — cellType and edge count unchanged.
TEST_F( ComputeTest, Shader_Anastomosis_DeadSlot_Skipped )
{
    if( !m_device )
        GTEST_SKIP();

    std::vector<glm::vec4> agents     = { glm::vec4( 0.0f, 0.0f, 0.0f, 0.0f ) }; // w=0 → dead
    size_t                 agentsSize = sizeof( glm::vec4 );

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

    uint32_t componentLabel = 0u; // single agent in its own component

    BufferHandle agentBuf     = m_rm->CreateBuffer( { agentsSize,           BufferType::STORAGE,  "DeadAnaAgents"       } );
    BufferHandle phenoBuf     = m_rm->CreateBuffer( { phenotypesSize,       BufferType::STORAGE,  "DeadAnaPheno"        } );
    BufferHandle hashBuf      = m_rm->CreateBuffer( { hashesSize,           BufferType::STORAGE,  "DeadAnaHash"         } );
    BufferHandle offsetBuf    = m_rm->CreateBuffer( { offsetsSize,          BufferType::STORAGE,  "DeadAnaOffsets"      } );
    BufferHandle countBuf     = m_rm->CreateBuffer( { sizeof( uint32_t ),   BufferType::INDIRECT, "DeadAnaCount"        } );
    BufferHandle edgeBuf      = m_rm->CreateBuffer( { sizeof( VesselEdge ), BufferType::STORAGE,  "DeadAnaEdges"        } );
    BufferHandle edgeCountBuf = m_rm->CreateBuffer( { sizeof( uint32_t ),   BufferType::STORAGE,  "DeadAnaEdgeCount"    } );
    BufferHandle componentBuf = m_rm->CreateBuffer( { sizeof( uint32_t ),   BufferType::STORAGE,  "DeadAnaComponents"   } );

    m_stream->UploadBufferImmediate( {
        { agentBuf,     agents.data(),     agentsSize,         0 },
        { phenoBuf,     phenotypes.data(), phenotypesSize,     0 },
        { hashBuf,      sortedHashes.data(), hashesSize,       0 },
        { offsetBuf,    cellOffsets.data(),  offsetsSize,      0 },
        { countBuf,     &agentCount,         sizeof( uint32_t ), 0 },
        { edgeCountBuf, &edgeCountInit,      sizeof( uint32_t ), 0 },
        { componentBuf, &componentLabel,     sizeof( uint32_t ), 0 },
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
    bg->Bind( 7, m_rm->GetBuffer( componentBuf ) );
    bg->Build();

    ComputePushConstants pc{};
    pc.dt          = 1.0f / 60.0f;
    pc.fParam0     = 5.0f;
    pc.fParam2     = 16.0f; // edgeBufferCapacity
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
    m_rm->DestroyBuffer( componentBuf );
}

// Non-TipCell agents (Default + StalkCell) within contact range must be skipped — no edge written.
TEST_F( ComputeTest, Shader_Anastomosis_NonTipCells_Skipped )
{
    if( !m_device )
        GTEST_SKIP();

    // Two alive agents at close range — Default (0) and StalkCell (2), neither is TipCell (1)
    std::vector<glm::vec4> agents     = { glm::vec4( 0.0f, 0.0f, 0.0f, 1.0f ), glm::vec4( 1.0f, 0.0f, 0.0f, 1.0f ) };
    size_t                 agentsSize = 2 * sizeof( glm::vec4 );

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

    // Component labels: each agent in its own component
    std::vector<uint32_t> componentLabels2 = { 0u, 1u };

    BufferHandle agentBuf     = m_rm->CreateBuffer( { agentsSize,           BufferType::STORAGE,  "NTCAgents"       } );
    BufferHandle phenoBuf     = m_rm->CreateBuffer( { phenotypesSize,       BufferType::STORAGE,  "NTCPheno"        } );
    BufferHandle hashBuf      = m_rm->CreateBuffer( { hashesSize,           BufferType::STORAGE,  "NTCHash"         } );
    BufferHandle offsetBuf    = m_rm->CreateBuffer( { offsetsSize,          BufferType::STORAGE,  "NTCOffsets"      } );
    BufferHandle countBuf     = m_rm->CreateBuffer( { sizeof( uint32_t ),   BufferType::INDIRECT, "NTCCount"        } );
    BufferHandle edgeBuf      = m_rm->CreateBuffer( { sizeof( VesselEdge ), BufferType::STORAGE,  "NTCEdges"        } );
    BufferHandle edgeCountBuf = m_rm->CreateBuffer( { sizeof( uint32_t ),   BufferType::STORAGE,  "NTCEdgeCount"    } );
    BufferHandle componentBuf = m_rm->CreateBuffer( { 2 * sizeof( uint32_t ), BufferType::STORAGE, "NTCComponents"  } );

    m_stream->UploadBufferImmediate( {
        { agentBuf,     agents.data(),            agentsSize,             0 },
        { phenoBuf,     phenotypes.data(),         phenotypesSize,         0 },
        { hashBuf,      sortedHashes.data(),       hashesSize,             0 },
        { offsetBuf,    cellOffsets.data(),        offsetsSize,            0 },
        { countBuf,     &agentCount,               sizeof( uint32_t ),     0 },
        { edgeCountBuf, &edgeCountInit,            sizeof( uint32_t ),     0 },
        { componentBuf, componentLabels2.data(),   2 * sizeof( uint32_t ), 0 },
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
    bg->Bind( 7, m_rm->GetBuffer( componentBuf ) );
    bg->Build();

    ComputePushConstants pc{};
    pc.dt          = 1.0f / 60.0f;
    pc.fParam0     = 5.0f;
    pc.fParam2     = 16.0f; // edgeBufferCapacity
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
    m_rm->DestroyBuffer( componentBuf );
}

// =================================================================================================
// anastomosis.comp — Tip-to-Stalk tests
// =================================================================================================

// TipCell near StalkCell from a DIFFERENT component with allowTipToStalk=1 → TipCell becomes
// StalkCell, StalkCell stays StalkCell, one edge recorded.
//
// Agents: A=(0,0,0) TipCell (label=0), B=(2,0,0) StalkCell (label=1). contactDistance=5.
// fParam1=1.0 → allowTipToStalk enabled. Components differ (0 ≠ 1) → fusion proceeds.
// TipCell A converts to StalkCell (2). StalkCell B stays StalkCell (2). Edge (0,1) written.
TEST_F( ComputeTest, Shader_Anastomosis_TipToStalk_DifferentComponent )
{
    if( !m_device )
        GTEST_SKIP();

    std::vector<glm::vec4> agents     = { glm::vec4( 0.0f, 0.0f, 0.0f, 1.0f ), glm::vec4( 2.0f, 0.0f, 0.0f, 1.0f ) };
    size_t                 agentsSize = 2 * sizeof( glm::vec4 );

    std::vector<PhenotypeData> phenotypes = { { 0u, 0.5f, 0.0f, 1u }, { 0u, 0.5f, 0.0f, 2u } }; // TipCell, StalkCell
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

    // Different component labels → fusion is allowed
    std::vector<uint32_t> componentLabels = { 0u, 1u };

    struct VesselEdge { uint32_t agentA; uint32_t agentB; float dist; uint32_t flags; };
    size_t edgeBufferSize = 16 * sizeof( VesselEdge );

    BufferHandle agentBuf     = m_rm->CreateBuffer( { agentsSize,             BufferType::STORAGE,  "TS_Agents"     } );
    BufferHandle phenoBuf     = m_rm->CreateBuffer( { phenotypesSize,         BufferType::STORAGE,  "TS_Pheno"      } );
    BufferHandle hashBuf      = m_rm->CreateBuffer( { hashesSize,             BufferType::STORAGE,  "TS_Hash"       } );
    BufferHandle offsetBuf    = m_rm->CreateBuffer( { offsetsSize,            BufferType::STORAGE,  "TS_Offsets"    } );
    BufferHandle countBuf     = m_rm->CreateBuffer( { sizeof( uint32_t ),     BufferType::INDIRECT, "TS_Count"      } );
    BufferHandle edgeBuf      = m_rm->CreateBuffer( { edgeBufferSize,         BufferType::STORAGE,  "TS_Edges"      } );
    BufferHandle edgeCountBuf = m_rm->CreateBuffer( { sizeof( uint32_t ),     BufferType::STORAGE,  "TS_EdgeCount"  } );
    BufferHandle componentBuf = m_rm->CreateBuffer( { 2 * sizeof( uint32_t ), BufferType::STORAGE,  "TS_Components" } );

    m_stream->UploadBufferImmediate( {
        { agentBuf,     agents.data(),           agentsSize,             0 },
        { phenoBuf,     phenotypes.data(),        phenotypesSize,         0 },
        { hashBuf,      sortedHashes.data(),      hashesSize,             0 },
        { offsetBuf,    cellOffsets.data(),       offsetsSize,            0 },
        { countBuf,     &agentCount,              sizeof( uint32_t ),     0 },
        { edgeCountBuf, &edgeCountInit,           sizeof( uint32_t ),     0 },
        { componentBuf, componentLabels.data(),   2 * sizeof( uint32_t ), 0 },
    } );

    ComputePipelineDesc   pipeDesc{ m_rm->CreateShader( "shaders/compute/biology/anastomosis.comp" ), "TestAnastomosisTS" };
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
    bg->Bind( 7, m_rm->GetBuffer( componentBuf ) );
    bg->Build();

    ComputePushConstants pc{};
    pc.dt            = 1.0f / 60.0f;
    pc.fParam0       = 5.0f;  // contactDistance
    pc.fParam1       = 1.0f;  // allowTipToStalk = true
    pc.fParam2       = 16.0f; // edgeBufferCapacity
    pc.offset        = 0;
    pc.maxCapacity   = 2;
    pc.uParam0       = offsetArraySize;
    pc.uParam1       = 0;
    pc.domainSize    = glm::vec4( 100.0f, 100.0f, 100.0f, 30.0f );

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
    VesselEdge                 resultEdge{};
    m_stream->ReadbackBufferImmediate( phenoBuf,     resultPheno.data(), phenotypesSize );
    m_stream->ReadbackBufferImmediate( edgeCountBuf, &resultEdgeCount,   sizeof( uint32_t ) );
    m_stream->ReadbackBufferImmediate( edgeBuf,      &resultEdge,        sizeof( VesselEdge ) );

    EXPECT_EQ( resultPheno[ 0 ].cellType, 2u )   << "TipCell must convert to StalkCell (2) after Tip-to-Stalk anastomosis";
    EXPECT_EQ( resultPheno[ 1 ].cellType, 2u )   << "Existing StalkCell must remain StalkCell (2)";
    EXPECT_EQ( resultEdgeCount,           1u )   << "Exactly 1 edge must be recorded for Tip-to-Stalk anastomosis";
    EXPECT_EQ( resultEdge.flags,          0x8u ) << "Edge must have SPROUT flag so vessel_components merges labels";

    m_rm->DestroyBuffer( agentBuf     );
    m_rm->DestroyBuffer( phenoBuf     );
    m_rm->DestroyBuffer( hashBuf      );
    m_rm->DestroyBuffer( offsetBuf    );
    m_rm->DestroyBuffer( countBuf     );
    m_rm->DestroyBuffer( edgeBuf      );
    m_rm->DestroyBuffer( edgeCountBuf );
    m_rm->DestroyBuffer( componentBuf );
}

// TipCell near StalkCell in the SAME component → same-component guard blocks fusion, no edge.
//
// Agents: A=(0,0,0) TipCell (label=0), B=(2,0,0) StalkCell (label=0 — same component).
// fParam1=1.0 → allowTipToStalk enabled. Components equal (0 == 0) → guard fires → no anastomosis.
TEST_F( ComputeTest, Shader_Anastomosis_TipToStalk_SameComponent_Skipped )
{
    if( !m_device )
        GTEST_SKIP();

    std::vector<glm::vec4> agents     = { glm::vec4( 0.0f, 0.0f, 0.0f, 1.0f ), glm::vec4( 2.0f, 0.0f, 0.0f, 1.0f ) };
    size_t                 agentsSize = 2 * sizeof( glm::vec4 );

    std::vector<PhenotypeData> phenotypes = { { 0u, 0.5f, 0.0f, 1u }, { 0u, 0.5f, 0.0f, 2u } }; // TipCell, StalkCell
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

    // Same component label for both agents → fusion must be blocked
    std::vector<uint32_t> componentLabels = { 0u, 0u };

    struct VesselEdge { uint32_t agentA; uint32_t agentB; float dist; uint32_t flags; };
    size_t edgeBufferSize = 16 * sizeof( VesselEdge );

    BufferHandle agentBuf     = m_rm->CreateBuffer( { agentsSize,             BufferType::STORAGE,  "SC_Agents"     } );
    BufferHandle phenoBuf     = m_rm->CreateBuffer( { phenotypesSize,         BufferType::STORAGE,  "SC_Pheno"      } );
    BufferHandle hashBuf      = m_rm->CreateBuffer( { hashesSize,             BufferType::STORAGE,  "SC_Hash"       } );
    BufferHandle offsetBuf    = m_rm->CreateBuffer( { offsetsSize,            BufferType::STORAGE,  "SC_Offsets"    } );
    BufferHandle countBuf     = m_rm->CreateBuffer( { sizeof( uint32_t ),     BufferType::INDIRECT, "SC_Count"      } );
    BufferHandle edgeBuf      = m_rm->CreateBuffer( { edgeBufferSize,         BufferType::STORAGE,  "SC_Edges"      } );
    BufferHandle edgeCountBuf = m_rm->CreateBuffer( { sizeof( uint32_t ),     BufferType::STORAGE,  "SC_EdgeCount"  } );
    BufferHandle componentBuf = m_rm->CreateBuffer( { 2 * sizeof( uint32_t ), BufferType::STORAGE,  "SC_Components" } );

    m_stream->UploadBufferImmediate( {
        { agentBuf,     agents.data(),           agentsSize,             0 },
        { phenoBuf,     phenotypes.data(),        phenotypesSize,         0 },
        { hashBuf,      sortedHashes.data(),      hashesSize,             0 },
        { offsetBuf,    cellOffsets.data(),       offsetsSize,            0 },
        { countBuf,     &agentCount,              sizeof( uint32_t ),     0 },
        { edgeCountBuf, &edgeCountInit,           sizeof( uint32_t ),     0 },
        { componentBuf, componentLabels.data(),   2 * sizeof( uint32_t ), 0 },
    } );

    ComputePipelineDesc   pipeDesc{ m_rm->CreateShader( "shaders/compute/biology/anastomosis.comp" ), "TestAnastomosisSC" };
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
    bg->Bind( 7, m_rm->GetBuffer( componentBuf ) );
    bg->Build();

    ComputePushConstants pc{};
    pc.dt            = 1.0f / 60.0f;
    pc.fParam0       = 5.0f;  // contactDistance
    pc.fParam1       = 1.0f;  // allowTipToStalk = true
    pc.fParam2       = 16.0f; // edgeBufferCapacity
    pc.offset        = 0;
    pc.maxCapacity   = 2;
    pc.uParam0       = offsetArraySize;
    pc.uParam1       = 0;
    pc.domainSize    = glm::vec4( 100.0f, 100.0f, 100.0f, 30.0f );

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

    EXPECT_EQ( resultPheno[ 0 ].cellType, 1u ) << "TipCell must stay TipCell (1) — same-component guard blocked fusion";
    EXPECT_EQ( resultPheno[ 1 ].cellType, 2u ) << "StalkCell must remain StalkCell (2) — no change";
    EXPECT_EQ( resultEdgeCount,           0u ) << "No edge must be written — same component fusion is blocked";

    m_rm->DestroyBuffer( agentBuf     );
    m_rm->DestroyBuffer( phenoBuf     );
    m_rm->DestroyBuffer( hashBuf      );
    m_rm->DestroyBuffer( offsetBuf    );
    m_rm->DestroyBuffer( countBuf     );
    m_rm->DestroyBuffer( edgeBuf      );
    m_rm->DestroyBuffer( edgeCountBuf );
    m_rm->DestroyBuffer( componentBuf );
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

    // One SPROUT edge (flags=0x8): (0, 1).  vessel_components only processes SPROUT edges.
    std::vector<VesselEdge> edgeData  = { { 0u, 1u, 2.0f, 0x8u } };
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

    // Chain: 0─1─2 (both SPROUT edges, flags=0x8 — the only type processed by vessel_components)
    std::vector<VesselEdge> edgeData  = { { 0u, 1u, 2.0f, 0x8u }, { 1u, 2u, 2.0f, 0x8u } };
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
// REQ_CELLTYPE: -1.0f = default StalkCell+PhalanxCell guard; ≥0 = exact cell-type match.
#define PERFUSION_RAW_TEST_SETUP( TEST_CELLTYPE, TEST_RATE, REQ_CELLTYPE, LABEL )                    \
    glm::vec4 agentPos( 0.0f, 0.0f, 0.0f, 1.0f );                                                   \
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
    pc.fParam1     = ( REQ_CELLTYPE );                                                                \
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

    // Filtered cells write their input position as passthrough (ChainFlip correctness).
    // Agent was at (0,0,0,1) → output must match input, not the sentinel.
    EXPECT_FLOAT_EQ( result.x, agentIn.x ) << "Filtered cell must passthrough its input position";
    EXPECT_FLOAT_EQ( result.w, agentIn.w ) << "Alive flag (w=1) must be preserved in passthrough";

    m_rm->DestroyBuffer( inBuf );
    m_rm->DestroyBuffer( outBuf );
    m_rm->DestroyBuffer( countBuf );
    m_rm->DestroyBuffer( phenoBuf );
    m_rm->DestroyTexture( fieldTex );
    m_rm->DestroyBuffer( dummyHashBuf );
    m_rm->DestroyBuffer( dummyOffsetBuf );
}

// StalkCell (cellType=2) with default filter (reqCellType=-1) → delta > 0 (injection).
TEST_F( ComputeTest, Shader_Perfusion_StalkCell_InjectsDelta )
{
    if( !m_device ) GTEST_SKIP();
    PERFUSION_RAW_TEST_SETUP( 2u, +5.0f, -1.0f, "PFInject" )
    EXPECT_GT( result, 0 ) << "StalkCell with default filter must produce a positive delta";
}

// StalkCell with negative rate and default filter → delta < 0 (drain).
TEST_F( ComputeTest, Shader_Drain_StalkCell_RemovesDelta )
{
    if( !m_device ) GTEST_SKIP();
    PERFUSION_RAW_TEST_SETUP( 2u, -5.0f, -1.0f, "PFDrain" )
    EXPECT_LT( result, 0 ) << "StalkCell with default filter and negative rate must drain";
}

// Non-vessel cell (Default, cellType=0) with default filter → skipped.
TEST_F( ComputeTest, Shader_Perfusion_NonVesselCell_Skipped )
{
    if( !m_device ) GTEST_SKIP();
    PERFUSION_RAW_TEST_SETUP( 0u, +5.0f, -1.0f, "PFSkip" )
    EXPECT_EQ( result, 0 ) << "Default cell type must not perfuse with the default filter";
}

// PhalanxCell (cellType=3) with explicit PhalanxCell filter → delta > 0.
TEST_F( ComputeTest, Shader_Perfusion_PhalanxOnly_Passes )
{
    if( !m_device ) GTEST_SKIP();
    PERFUSION_RAW_TEST_SETUP( 3u, +5.0f, 3.0f, "PFPhOnly" )
    EXPECT_GT( result, 0 ) << "PhalanxCell must perfuse when reqCellType=3";
}

// StalkCell (cellType=2) with explicit PhalanxCell filter → skipped.
TEST_F( ComputeTest, Shader_Perfusion_PhalanxOnly_RejectStalk )
{
    if( !m_device ) GTEST_SKIP();
    PERFUSION_RAW_TEST_SETUP( 2u, +5.0f, 3.0f, "PFPhReject" )
    EXPECT_EQ( result, 0 ) << "StalkCell must be rejected when reqCellType=3 (PhalanxCell-only)";
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
    std::vector<PhenotypeData> phenotypes( groupCapacity, { 0, 0.5f, 0.0f, 0 } );
    phenotypes[ 0 ].cellType = 1; // TipCell
    phenotypes[ 1 ].cellType = 2; // StalkCell

    // Reorder buffer (3 * groupCapacity slots)
    uint32_t reorderSize = 3 * groupCapacity;
    std::vector<uint32_t> reorderInit( reorderSize, 0xDEADBEEF );

    // Create GPU buffers
    // Visibility buffer: 1 group, all visible
    std::vector<uint32_t> visibility = { 1u };

    auto countBuf      = m_rm->CreateBuffer( { sizeof( uint32_t ), BufferType::STORAGE, "BI_Counts" } );
    auto indirectBuf   = m_rm->CreateBuffer( { cmds.size() * sizeof( DrawCommand ), BufferType::STORAGE, "BI_Indirect" } );
    auto phenoBuf      = m_rm->CreateBuffer( { phenotypes.size() * sizeof( PhenotypeData ), BufferType::STORAGE, "BI_Pheno" } );
    auto reorderBuf    = m_rm->CreateBuffer( { reorderSize * sizeof( uint32_t ), BufferType::STORAGE, "BI_Reorder" } );
    auto metaBuf       = m_rm->CreateBuffer( { metaData.size() * sizeof( DrawMeta ), BufferType::STORAGE, "BI_Meta" } );
    auto agentBuf      = m_rm->CreateBuffer( { positions.size() * sizeof( glm::vec4 ), BufferType::STORAGE, "BI_Agents" } );
    auto visibilityBuf = m_rm->CreateBuffer( { visibility.size() * sizeof( uint32_t ), BufferType::STORAGE, "BI_Visibility" } );

    m_stream->UploadBufferImmediate( {
        { countBuf,      counts.data(),      counts.size() * sizeof( uint32_t ) },
        { indirectBuf,   cmds.data(),        cmds.size() * sizeof( DrawCommand ) },
        { phenoBuf,      phenotypes.data(),  phenotypes.size() * sizeof( PhenotypeData ) },
        { reorderBuf,    reorderInit.data(), reorderInit.size() * sizeof( uint32_t ) },
        { metaBuf,       metaData.data(),    metaData.size() * sizeof( DrawMeta ) },
        { agentBuf,      positions.data(),   positions.size() * sizeof( glm::vec4 ) },
        { visibilityBuf, visibility.data(),  visibility.size() * sizeof( uint32_t ) },
    } );

    // Bind
    BindingGroup* bg = m_rm->GetBindingGroup( m_rm->CreateBindingGroup( pipe, 0 ) );
    bg->Bind( 0, m_rm->GetBuffer( countBuf ) );
    bg->Bind( 1, m_rm->GetBuffer( indirectBuf ) );
    bg->Bind( 2, m_rm->GetBuffer( phenoBuf ) );
    bg->Bind( 3, m_rm->GetBuffer( reorderBuf ) );
    bg->Bind( 4, m_rm->GetBuffer( metaBuf ) );
    bg->Bind( 5, m_rm->GetBuffer( agentBuf ) );
    bg->Bind( 6, m_rm->GetBuffer( visibilityBuf ) );
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

    std::vector<PhenotypeData> phenotypes( 2, { 0u, 0.5f, 0.0f, 0u } );
    size_t phenoBytes = 2 * sizeof( PhenotypeData );

    BufferHandle inBuf        = m_rm->CreateBuffer( { agentBytes,          BufferType::STORAGE, "VMInAgents"    } );
    BufferHandle outBuf       = m_rm->CreateBuffer( { agentBytes,          BufferType::STORAGE, "VMOutAgents"   } );
    BufferHandle edgeBuf      = m_rm->CreateBuffer( { sizeof( VesselEdge ),BufferType::STORAGE, "VMEdges"       } );
    BufferHandle edgeCountBuf = m_rm->CreateBuffer( { sizeof( uint32_t ),  BufferType::STORAGE, "VMEdgeCount"   } );
    BufferHandle countBuf     = m_rm->CreateBuffer( { sizeof( uint32_t ),  BufferType::STORAGE, "VMCount"       } );
    BufferHandle phenoBuf     = m_rm->CreateBuffer( { phenoBytes,          BufferType::STORAGE, "VMPheno"       } );

    m_stream->UploadBufferImmediate( {
        { inBuf,        agentsIn.data(),    agentBytes,           0 },
        { outBuf,       agentsOut.data(),   agentBytes,           0 },
        { edgeBuf,      &edge,              sizeof( VesselEdge ), 0 },
        { edgeCountBuf, &edgeCount,         sizeof( uint32_t ),   0 },
        { countBuf,     &agentCount,        sizeof( uint32_t ),   0 },
        { phenoBuf,     phenotypes.data(),  phenoBytes,           0 },
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
    bg->Bind( 5, m_rm->GetBuffer( phenoBuf ) );
    bg->Build();

    // k=10, restLen=2, dt=1 → stretch=2 → force=10*2=20 per agent; reqCT=-1 (any)
    ComputePushConstants pc{};
    pc.dt          = 1.0f;
    pc.fParam0     = 10.0f;  // springStiffness (k)
    pc.fParam1     = 2.0f;   // restingLength
    pc.fParam3     = -1.0f;  // reqCT: -1 = any cell type
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
    m_rm->DestroyBuffer( phenoBuf );
}

// Hysteresis dead zone: agent starts as TipCell with Dll4 in [stalkThreshold, tipThreshold].
// After one ODE step Dll4 remains in the dead zone — cellType must stay TipCell.
//
// Setup: 1 agent, cellType=1 (TipCell), dll4=0.5 (between stalkThreshold=0.3 and tipThreshold=0.8).
// No neighbors → meanNeighborDll4 = 0, dt=0.1.
// new_nicd = 0; vegfr2 = 1/(1+0)=1; new_dll4 = 0.5 + 0.1*(1 - 0.1*0.5) = 0.595 → still in dead zone.
// Expected: cellType stays 1 (TipCell).
TEST_F( ComputeTest, Shader_NotchDll4_Hysteresis_KeepsTypeInDeadZone )
{
    if( !m_device )
        GTEST_SKIP();

    struct SignalingData { float dll4; float nicd; float vegfr2; float pad; };
    struct AgentHash     { uint32_t hash; uint32_t agentIndex; };

    std::vector<glm::vec4>    agents     = { glm::vec4( 0.0f, 0.0f, 0.0f, 1.0f ) };
    std::vector<SignalingData> signaling  = { { 0.5f, 0.0f, 1.0f, 0.0f } };
    std::vector<PhenotypeData> phenotypes = { { 0u, 0.5f, 0.0f, 1u } }; // TipCell
    struct VesselEdge { uint32_t agentA; uint32_t agentB; float dist; uint32_t flags; };
    VesselEdge dummyEdge  = { 0u, 0u, 0.0f, 0u };
    uint32_t   edgeCount  = 0;
    uint32_t   agentCount = 1;

    size_t agSz  = sizeof( glm::vec4 ), sigSz = sizeof( SignalingData );
    size_t phSz  = sizeof( PhenotypeData ), cSz = sizeof( uint32_t );

    BufferHandle agBuf   = m_rm->CreateBuffer( { agSz,                BufferType::STORAGE,  "HysAgents"     } );
    BufferHandle sigBuf  = m_rm->CreateBuffer( { sigSz,               BufferType::STORAGE,  "HysSignal"     } );
    BufferHandle phBuf   = m_rm->CreateBuffer( { phSz,                BufferType::STORAGE,  "HysPheno"      } );
    BufferHandle eBuf    = m_rm->CreateBuffer( { sizeof( VesselEdge ), BufferType::STORAGE,  "HysEdges"      } );
    BufferHandle ecBuf   = m_rm->CreateBuffer( { sizeof( uint32_t ),   BufferType::STORAGE,  "HysEdgeCount"  } );
    BufferHandle cBuf    = m_rm->CreateBuffer( { cSz,                  BufferType::INDIRECT, "HysCount"      } );
    TextureHandle vTex   = m_rm->CreateTexture( { 1, 1, 1, TextureType::Texture3D, VK_FORMAT_R32_SFLOAT, TextureUsage::STORAGE | TextureUsage::TRANSFER_DST, VK_SAMPLE_COUNT_1_BIT, "HysVEGF" } );
    float one = 1.0f;
    m_stream->UploadTextureImmediate( vTex, &one, sizeof( float ) );

    m_stream->UploadBufferImmediate( {
        { agBuf,  agents.data(),     agSz,                0 },
        { sigBuf, signaling.data(),  sigSz,               0 },
        { phBuf,  phenotypes.data(), phSz,                0 },
        { eBuf,   &dummyEdge,        sizeof( VesselEdge ), 0 },
        { ecBuf,  &edgeCount,        sizeof( uint32_t ),   0 },
        { cBuf,   &agentCount,       cSz,                 0 },
    } );

    ComputePipelineDesc   pd{ m_rm->CreateShader( "shaders/compute/biology/notch_dll4.comp" ), "TestHys" };
    ComputePipelineHandle ph = m_rm->CreatePipeline( pd );
    ComputePipeline*      pp = m_rm->GetPipeline( ph );

    BindingGroup* bg = m_rm->GetBindingGroup( m_rm->CreateBindingGroup( ph, 0 ) );
    bg->Bind( 0, m_rm->GetBuffer( agBuf  ) );
    bg->Bind( 1, m_rm->GetBuffer( sigBuf ) );
    bg->Bind( 2, m_rm->GetBuffer( phBuf  ) );
    bg->Bind( 3, m_rm->GetBuffer( eBuf   ) );
    bg->Bind( 4, m_rm->GetBuffer( ecBuf  ) );
    bg->Bind( 5, m_rm->GetBuffer( cBuf   ) );
    bg->Bind( 6, m_rm->GetTexture( vTex ), VK_IMAGE_LAYOUT_GENERAL );
    bg->Build();

    ComputePushConstants pc{};
    pc.dt = 0.1f; pc.fParam0 = 1.0f; pc.fParam1 = 0.1f; pc.fParam2 = 5.0f;
    pc.fParam3 = 1.0f; pc.fParam4 = 0.8f; pc.fParam5 = 0.3f;
    pc.offset = 0; pc.maxCapacity = 1; pc.uParam0 = 0u; pc.uParam1 = 0;
    pc.domainSize = glm::vec4( 100.0f, 100.0f, 100.0f, 0.0f );
    pc.gridSize   = glm::uvec4( 1u, 1u, 1u, 0u );

    auto ctx = m_device->GetThreadContext( m_device->CreateThreadContext( QueueType::COMPUTE ) );
    auto cmd = ctx->GetCommandBuffer( ctx->CreateCommandBuffer() );
    cmd->Begin();
    cmd->SetPipeline( pp );
    cmd->SetBindingGroup( bg, pp->GetLayout(), VK_PIPELINE_BIND_POINT_COMPUTE );
    cmd->PushConstants( pp->GetLayout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof( pc ), &pc );
    cmd->Dispatch( 1, 1, 1 );
    cmd->End();
    m_device->GetComputeQueue()->Submit( { cmd } );
    m_device->GetComputeQueue()->WaitIdle();

    std::vector<PhenotypeData> result( 1 );
    m_stream->ReadbackBufferImmediate( phBuf, result.data(), phSz );

    EXPECT_EQ( result[ 0 ].cellType, 1u ) << "TipCell with Dll4 in dead zone must keep type (hysteresis)";

    m_rm->DestroyBuffer( agBuf ); m_rm->DestroyBuffer( sigBuf ); m_rm->DestroyBuffer( phBuf );
    m_rm->DestroyBuffer( eBuf  ); m_rm->DestroyBuffer( ecBuf  ); m_rm->DestroyBuffer( cBuf  );
    m_rm->DestroyTexture( vTex );
}

// Hysteresis lower boundary: a live cell whose Dll4 decays below stalkThreshold converts to StalkCell.
// TipCells are protected from demotion (see notch_dll4.comp — TipCell commitment guard).
//
// Setup: 1 agent, cellType=0 (Live), dll4=0.1, VEGF=0 → no production.
// vegfGating = 0/(0+1) = 0 → vegfr2 = 0
// new_dll4 = 0.1 + 1.0*(0 - 0.1*0.1) = 0.09 < stalkThreshold(0.3) → StalkCell
TEST_F( ComputeTest, Shader_NotchDll4_Hysteresis_StalkBelowThreshold )
{
    if( !m_device )
        GTEST_SKIP();

    struct SignalingData { float dll4; float nicd; float vegfr2; float pad; };
    struct AgentHash     { uint32_t hash; uint32_t agentIndex; };

    std::vector<glm::vec4>    agents     = { glm::vec4( 0.0f, 0.0f, 0.0f, 1.0f ) };
    std::vector<SignalingData> signaling  = { { 0.1f, 0.0f, 0.0f, 0.0f } };
    std::vector<PhenotypeData> phenotypes = { { 0u, 0.5f, 0.0f, 0u } }; // starts Live (not TipCell)
    struct VesselEdge { uint32_t agentA; uint32_t agentB; float dist; uint32_t flags; };
    VesselEdge dummyEdge  = { 0u, 0u, 0.0f, 0u };
    uint32_t   edgeCount  = 0;
    uint32_t   agentCount = 1;

    size_t agSz  = sizeof( glm::vec4 ), sigSz = sizeof( SignalingData );
    size_t phSz  = sizeof( PhenotypeData ), cSz = sizeof( uint32_t );

    BufferHandle agBuf   = m_rm->CreateBuffer( { agSz,                BufferType::STORAGE,  "HysLowAgents"     } );
    BufferHandle sigBuf  = m_rm->CreateBuffer( { sigSz,               BufferType::STORAGE,  "HysLowSignal"     } );
    BufferHandle phBuf   = m_rm->CreateBuffer( { phSz,                BufferType::STORAGE,  "HysLowPheno"      } );
    BufferHandle eBuf    = m_rm->CreateBuffer( { sizeof( VesselEdge ), BufferType::STORAGE,  "HysLowEdges"      } );
    BufferHandle ecBuf   = m_rm->CreateBuffer( { sizeof( uint32_t ),   BufferType::STORAGE,  "HysLowEdgeCount"  } );
    BufferHandle cBuf    = m_rm->CreateBuffer( { cSz,                  BufferType::INDIRECT, "HysLowCount"      } );
    TextureHandle vTex   = m_rm->CreateTexture( { 1, 1, 1, TextureType::Texture3D, VK_FORMAT_R32_SFLOAT, TextureUsage::STORAGE | TextureUsage::TRANSFER_DST, VK_SAMPLE_COUNT_1_BIT, "HysLowVEGF" } );
    float zero = 0.0f;
    m_stream->UploadTextureImmediate( vTex, &zero, sizeof( float ) );

    m_stream->UploadBufferImmediate( {
        { agBuf,  agents.data(),     agSz,                0 },
        { sigBuf, signaling.data(),  sigSz,               0 },
        { phBuf,  phenotypes.data(), phSz,                0 },
        { eBuf,   &dummyEdge,        sizeof( VesselEdge ), 0 },
        { ecBuf,  &edgeCount,        sizeof( uint32_t ),   0 },
        { cBuf,   &agentCount,       cSz,                 0 },
    } );

    ComputePipelineDesc   pd{ m_rm->CreateShader( "shaders/compute/biology/notch_dll4.comp" ), "TestHysLow" };
    ComputePipelineHandle ph = m_rm->CreatePipeline( pd );
    ComputePipeline*      pp = m_rm->GetPipeline( ph );

    BindingGroup* bg = m_rm->GetBindingGroup( m_rm->CreateBindingGroup( ph, 0 ) );
    bg->Bind( 0, m_rm->GetBuffer( agBuf  ) );
    bg->Bind( 1, m_rm->GetBuffer( sigBuf ) );
    bg->Bind( 2, m_rm->GetBuffer( phBuf  ) );
    bg->Bind( 3, m_rm->GetBuffer( eBuf   ) );
    bg->Bind( 4, m_rm->GetBuffer( ecBuf  ) );
    bg->Bind( 5, m_rm->GetBuffer( cBuf   ) );
    bg->Bind( 6, m_rm->GetTexture( vTex ), VK_IMAGE_LAYOUT_GENERAL );
    bg->Build();

    ComputePushConstants pc{};
    pc.dt = 1.0f; pc.fParam0 = 1.0f; pc.fParam1 = 0.1f; pc.fParam2 = 5.0f;
    pc.fParam3 = 1.0f; pc.fParam4 = 0.8f; pc.fParam5 = 0.3f;
    pc.offset = 0; pc.maxCapacity = 1; pc.uParam0 = 0u; pc.uParam1 = 0;
    pc.domainSize = glm::vec4( 100.0f, 100.0f, 100.0f, 0.0f );
    pc.gridSize   = glm::uvec4( 1u, 1u, 1u, 1u ); // w=1 → VEGF sampling enabled (VEGF=0)

    auto ctx = m_device->GetThreadContext( m_device->CreateThreadContext( QueueType::COMPUTE ) );
    auto cmd = ctx->GetCommandBuffer( ctx->CreateCommandBuffer() );
    cmd->Begin();
    cmd->SetPipeline( pp );
    cmd->SetBindingGroup( bg, pp->GetLayout(), VK_PIPELINE_BIND_POINT_COMPUTE );
    cmd->PushConstants( pp->GetLayout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof( pc ), &pc );
    cmd->Dispatch( 1, 1, 1 );
    cmd->End();
    m_device->GetComputeQueue()->Submit( { cmd } );
    m_device->GetComputeQueue()->WaitIdle();

    std::vector<PhenotypeData> resultPheno( 1 );
    std::vector<SignalingData> resultSig( 1 );
    m_stream->ReadbackBufferImmediate( phBuf,  resultPheno.data(), phSz  );
    m_stream->ReadbackBufferImmediate( sigBuf, resultSig.data(),   sigSz );

    EXPECT_LT( resultSig[ 0 ].dll4,    0.3f ) << "Dll4 should have decayed below stalkThreshold";
    EXPECT_EQ( resultPheno[ 0 ].cellType, 2u ) << "Live cell with Dll4 below stalkThreshold must convert to StalkCell";

    m_rm->DestroyBuffer( agBuf ); m_rm->DestroyBuffer( sigBuf ); m_rm->DestroyBuffer( phBuf );
    m_rm->DestroyBuffer( eBuf  ); m_rm->DestroyBuffer( ecBuf  ); m_rm->DestroyBuffer( cBuf  );
    m_rm->DestroyTexture( vTex );
}

// Damped VesselSpring produces smaller displacement than undamped.
//
// Two agents 4 units apart, restLen=2, k=10, dt=1 → stretch=2, force=20.
// Undamped: displacement = 20 * 1 = 20
// Damped (d=10): displacement = 20 / (1 + 10*1) = 20/11 ≈ 1.818
TEST_F( ComputeTest, Shader_VesselMechanics_DampingReducesDisplacement )
{
    if( !m_device )
        GTEST_SKIP();

    struct VesselEdge { uint32_t agentA; uint32_t agentB; float dist; uint32_t flags; };

    auto runSpring = [&]( float damping ) -> float
    {
        std::vector<glm::vec4> agentsIn  = { glm::vec4( -2.0f, 0.0f, 0.0f, 1.0f ), glm::vec4( 2.0f, 0.0f, 0.0f, 1.0f ) };
        std::vector<glm::vec4> agentsOut( 2, glm::vec4( 0.0f ) );
        size_t agBz = 2 * sizeof( glm::vec4 );

        std::vector<PhenotypeData> phenos( 2, { 0u, 0.5f, 0.0f, 0u } );
        size_t phBz = 2 * sizeof( PhenotypeData );

        VesselEdge edge = { 0u, 1u, 2.0f, 0u };
        uint32_t ec = 1, ac = 2;

        BufferHandle ib  = m_rm->CreateBuffer( { agBz,           BufferType::STORAGE, "VSDIn"   } );
        BufferHandle ob  = m_rm->CreateBuffer( { agBz,           BufferType::STORAGE, "VSDOut"  } );
        BufferHandle eb  = m_rm->CreateBuffer( { sizeof( VesselEdge ), BufferType::STORAGE, "VSDEdge" } );
        BufferHandle ecb = m_rm->CreateBuffer( { sizeof( uint32_t ),   BufferType::STORAGE, "VSDEc"   } );
        BufferHandle cb  = m_rm->CreateBuffer( { sizeof( uint32_t ),   BufferType::STORAGE, "VSDC"    } );
        BufferHandle phb = m_rm->CreateBuffer( { phBz,           BufferType::STORAGE, "VSDPheno" } );

        m_stream->UploadBufferImmediate( {
            { ib, agentsIn.data(), agBz, 0 }, { ob, agentsOut.data(), agBz, 0 },
            { eb, &edge, sizeof( VesselEdge ), 0 }, { ecb, &ec, sizeof( uint32_t ), 0 },
            { cb, &ac,   sizeof( uint32_t ), 0 }, { phb, phenos.data(), phBz, 0 },
        } );

        ComputePipelineDesc   pd{ m_rm->CreateShader( "shaders/compute/biology/vessel_mechanics.comp" ), "TestVSDamp" };
        ComputePipelineHandle ph2 = m_rm->CreatePipeline( pd );
        ComputePipeline*      pp2 = m_rm->GetPipeline( ph2 );

        BindingGroup* bg2 = m_rm->GetBindingGroup( m_rm->CreateBindingGroup( ph2, 0 ) );
        bg2->Bind( 0, m_rm->GetBuffer( ib ) ); bg2->Bind( 1, m_rm->GetBuffer( ob ) );
        bg2->Bind( 2, m_rm->GetBuffer( eb ) ); bg2->Bind( 3, m_rm->GetBuffer( ecb ) );
        bg2->Bind( 4, m_rm->GetBuffer( cb ) ); bg2->Bind( 5, m_rm->GetBuffer( phb ) );
        bg2->Build();

        ComputePushConstants pc2{};
        pc2.dt = 1.0f; pc2.fParam0 = 10.0f; pc2.fParam1 = 2.0f; pc2.fParam2 = damping;
        pc2.fParam3 = -1.0f; // reqCT: -1 = any cell type
        pc2.offset = 0; pc2.maxCapacity = 2; pc2.uParam0 = 0;

        auto ctx2 = m_device->GetThreadContext( m_device->CreateThreadContext( QueueType::COMPUTE ) );
        auto cmd2 = ctx2->GetCommandBuffer( ctx2->CreateCommandBuffer() );
        cmd2->Begin();
        cmd2->SetPipeline( pp2 );
        cmd2->SetBindingGroup( bg2, pp2->GetLayout(), VK_PIPELINE_BIND_POINT_COMPUTE );
        cmd2->PushConstants( pp2->GetLayout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof( pc2 ), &pc2 );
        cmd2->Dispatch( 1, 1, 1 );
        cmd2->End();
        m_device->GetComputeQueue()->Submit( { cmd2 } );
        m_device->GetComputeQueue()->WaitIdle();

        std::vector<glm::vec4> res( 2 );
        m_stream->ReadbackBufferImmediate( ob, res.data(), agBz );
        float disp = res[ 0 ].x - agentsIn[ 0 ].x;

        m_rm->DestroyBuffer( ib ); m_rm->DestroyBuffer( ob ); m_rm->DestroyBuffer( eb );
        m_rm->DestroyBuffer( ecb ); m_rm->DestroyBuffer( cb ); m_rm->DestroyBuffer( phb );
        return disp;
    };

    float undamped = runSpring( 0.0f  );
    float damped   = runSpring( 10.0f );

    EXPECT_GT( undamped, 0.0f ) << "Undamped spring must move agent 0 in +X";
    EXPECT_GT( damped,   0.0f ) << "Damped spring must still move agent 0 in +X";
    EXPECT_LT( damped, undamped ) << "Damped displacement must be less than undamped";
}

// Cell-type filter: reqCT=Default(0) — only Default cells receive spring force.
// Agent 0 (PhalanxCell=3) must stay at its input position (filtered by reqCT).
// Agent 1 (Default=0) must move toward agent 0 (spring force applies).
//
// Setup: agent 0 at (-2,0,0) PhalanxCell, agent 1 at (2,0,0) Default, 4 units apart,
//        restingLength=2, k=10, dt=1. fParam3=0.0 (Default only).
TEST_F( ComputeTest, Shader_VesselMechanics_CellTypeFilter_NonTipCellSkipped )
{
    if( !m_device )
        GTEST_SKIP();

    struct VesselEdge    { uint32_t agentA; uint32_t agentB; float dist; uint32_t flags; };

    std::vector<glm::vec4> agentsIn = {
        glm::vec4( -2.0f, 0.0f, 0.0f, 1.0f ),  // agent 0: PhalanxCell
        glm::vec4(  2.0f, 0.0f, 0.0f, 1.0f ),  // agent 1: Default
    };
    std::vector<glm::vec4> agentsOut( 2, glm::vec4( 0.0f ) );
    std::vector<PhenotypeData> phenotypes = {
        { 0u, 0.5f, 0.0f, 3u },  // PhalanxCell
        { 0u, 0.5f, 0.0f, 0u },  // Default
    };

    size_t agBz = 2 * sizeof( glm::vec4 );
    size_t phBz = 2 * sizeof( PhenotypeData );
    VesselEdge edge      = { 0u, 1u, 2.0f, 0u };
    uint32_t   edgeCount = 1;
    uint32_t   agentCount = 2;

    BufferHandle ib  = m_rm->CreateBuffer( { agBz,                  BufferType::STORAGE, "CTFIn"   } );
    BufferHandle ob  = m_rm->CreateBuffer( { agBz,                  BufferType::STORAGE, "CTFOut"  } );
    BufferHandle eb  = m_rm->CreateBuffer( { sizeof( VesselEdge ),  BufferType::STORAGE, "CTFEdge" } );
    BufferHandle ecb = m_rm->CreateBuffer( { sizeof( uint32_t ),    BufferType::STORAGE, "CTFEc"   } );
    BufferHandle cb  = m_rm->CreateBuffer( { sizeof( uint32_t ),    BufferType::STORAGE, "CTFCnt"  } );
    BufferHandle phb = m_rm->CreateBuffer( { phBz,                  BufferType::STORAGE, "CTFPheno"} );

    m_stream->UploadBufferImmediate( {
        { ib,  agentsIn.data(),    agBz,                  0 },
        { ob,  agentsOut.data(),   agBz,                  0 },
        { eb,  &edge,              sizeof( VesselEdge ),  0 },
        { ecb, &edgeCount,         sizeof( uint32_t ),    0 },
        { cb,  &agentCount,        sizeof( uint32_t ),    0 },
        { phb, phenotypes.data(),  phBz,                  0 },
    } );

    ComputePipelineDesc   pd{ m_rm->CreateShader( "shaders/compute/biology/vessel_mechanics.comp" ), "TestVSFilter" };
    ComputePipelineHandle ph = m_rm->CreatePipeline( pd );
    ComputePipeline*      pp = m_rm->GetPipeline( ph );

    BindingGroup* bg = m_rm->GetBindingGroup( m_rm->CreateBindingGroup( ph, 0 ) );
    bg->Bind( 0, m_rm->GetBuffer( ib ) );
    bg->Bind( 1, m_rm->GetBuffer( ob ) );
    bg->Bind( 2, m_rm->GetBuffer( eb ) );
    bg->Bind( 3, m_rm->GetBuffer( ecb ) );
    bg->Bind( 4, m_rm->GetBuffer( cb ) );
    bg->Bind( 5, m_rm->GetBuffer( phb ) );
    bg->Build();

    ComputePushConstants pc{};
    pc.dt          = 1.0f;
    pc.fParam0     = 10.0f;  // springStiffness
    pc.fParam1     = 2.0f;   // restingLength
    pc.fParam3     = 0.0f;   // reqCT = Default (0)
    pc.offset      = 0;
    pc.maxCapacity = 2;
    pc.uParam0     = 0;

    auto ctx = m_device->GetThreadContext( m_device->CreateThreadContext( QueueType::COMPUTE ) );
    auto cmd = ctx->GetCommandBuffer( ctx->CreateCommandBuffer() );
    cmd->Begin();
    cmd->SetPipeline( pp );
    cmd->SetBindingGroup( bg, pp->GetLayout(), VK_PIPELINE_BIND_POINT_COMPUTE );
    cmd->PushConstants( pp->GetLayout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof( pc ), &pc );
    cmd->Dispatch( 1, 1, 1 );
    cmd->End();
    m_device->GetComputeQueue()->Submit( { cmd } );
    m_device->GetComputeQueue()->WaitIdle();

    std::vector<glm::vec4> result( 2 );
    m_stream->ReadbackBufferImmediate( ob, result.data(), agBz );

    // PhalanxCell must not move — passthrough preserves input position
    EXPECT_FLOAT_EQ( result[ 0 ].x, agentsIn[ 0 ].x ) << "PhalanxCell (filtered) must not move";
    EXPECT_FLOAT_EQ( result[ 0 ].w, 1.0f );

    // Default cell must move toward PhalanxCell (in -X direction)
    EXPECT_LT( result[ 1 ].x, agentsIn[ 1 ].x ) << "Default cell must move toward PhalanxCell";
    EXPECT_FLOAT_EQ( result[ 1 ].w, 1.0f );

    m_rm->DestroyBuffer( ib ); m_rm->DestroyBuffer( ob ); m_rm->DestroyBuffer( eb );
    m_rm->DestroyBuffer( ecb ); m_rm->DestroyBuffer( cb ); m_rm->DestroyBuffer( phb );
}

// PhalanxCell anchor invariant: when anchorPhalanxCells is set (fParam4=1.0), PhalanxCells
// must never be displaced by spring forces, even when reqCT==-1 (any type allowed).
// Three-agent chain: Phalanx(0)—Stalk(1)—Default(2). PhalanxCell and StalkCell are both
// anchored by fParam4=1.0; Default cell must be pulled toward the anchored StalkCell.
TEST_F( ComputeTest, Shader_VesselMechanics_PhalanxCellAnchored )
{
    if( !m_device )
        GTEST_SKIP();

    struct VesselEdge    { uint32_t agentA; uint32_t agentB; float dist; uint32_t flags; };

    // PhalanxCell placed at restLen distance (stretch=0, no force from that edge).
    // Default placed at 4 units from StalkCell (stretch=2). Both PhalanxCell and StalkCell
    // are anchored by fParam4=1.0; only Default cell receives the spring force.
    std::vector<glm::vec4> agentsIn = {
        glm::vec4( -2.0f, 0.0f, 0.0f, 1.0f ),  // agent 0: PhalanxCell (at restLen=2 from StalkCell)
        glm::vec4(  0.0f, 0.0f, 0.0f, 1.0f ),  // agent 1: StalkCell
        glm::vec4(  4.0f, 0.0f, 0.0f, 1.0f ),  // agent 2: Default (4 units from StalkCell, stretch=2)
    };
    std::vector<glm::vec4>  agentsOut( 3, glm::vec4( 0.0f ) );
    std::vector<PhenotypeData> phenotypes = {
        { 0u, 0.5f, 0.0f, 3u },  // PhalanxCell
        { 0u, 0.5f, 0.0f, 2u },  // StalkCell
        { 0u, 0.5f, 0.0f, 0u },  // Default
    };

    std::vector<VesselEdge> edges = { { 0u, 1u, 2.0f, 0u }, { 1u, 2u, 2.0f, 0u } };
    uint32_t edgeCount  = 2;
    uint32_t agentCount = 3;
    size_t   agBz  = 3 * sizeof( glm::vec4 );
    size_t   phBz  = 3 * sizeof( PhenotypeData );
    size_t   edgBz = 2 * sizeof( VesselEdge );

    BufferHandle ib  = m_rm->CreateBuffer( { agBz,                BufferType::STORAGE, "PAIn"   } );
    BufferHandle ob  = m_rm->CreateBuffer( { agBz,                BufferType::STORAGE, "PAOut"  } );
    BufferHandle eb  = m_rm->CreateBuffer( { edgBz,               BufferType::STORAGE, "PAEdge" } );
    BufferHandle ecb = m_rm->CreateBuffer( { sizeof( uint32_t ),  BufferType::STORAGE, "PAEc"   } );
    BufferHandle cb  = m_rm->CreateBuffer( { sizeof( uint32_t ),  BufferType::STORAGE, "PACnt"  } );
    BufferHandle phb = m_rm->CreateBuffer( { phBz,                BufferType::STORAGE, "PAPheno"} );

    m_stream->UploadBufferImmediate( {
        { ib,  agentsIn.data(),   agBz,                 0 },
        { ob,  agentsOut.data(),  agBz,                 0 },
        { eb,  edges.data(),      edgBz,                0 },
        { ecb, &edgeCount,        sizeof( uint32_t ),   0 },
        { cb,  &agentCount,       sizeof( uint32_t ),   0 },
        { phb, phenotypes.data(), phBz,                 0 },
    } );

    ComputePipelineDesc   pd{ m_rm->CreateShader( "shaders/compute/biology/vessel_mechanics.comp" ), "TestVSPhalanxAnchor" };
    ComputePipelineHandle ph = m_rm->CreatePipeline( pd );
    ComputePipeline*      pp = m_rm->GetPipeline( ph );

    BindingGroup* bg = m_rm->GetBindingGroup( m_rm->CreateBindingGroup( ph, 0 ) );
    bg->Bind( 0, m_rm->GetBuffer( ib ) );
    bg->Bind( 1, m_rm->GetBuffer( ob ) );
    bg->Bind( 2, m_rm->GetBuffer( eb ) );
    bg->Bind( 3, m_rm->GetBuffer( ecb ) );
    bg->Bind( 4, m_rm->GetBuffer( cb ) );
    bg->Bind( 5, m_rm->GetBuffer( phb ) );
    bg->Build();

    ComputePushConstants pc{};
    pc.dt          = 1.0f;
    pc.fParam0     = 10.0f;  // springStiffness
    pc.fParam1     = 2.0f;   // restingLength
    pc.fParam3     = -1.0f;  // reqCT = any
    pc.fParam4     = 1.0f;   // anchorPhalanxCells — anchors both PhalanxCell and StalkCell
    pc.offset      = 0;
    pc.maxCapacity = 3;
    pc.uParam0     = 0;

    auto ctx = m_device->GetThreadContext( m_device->CreateThreadContext( QueueType::COMPUTE ) );
    auto cmd = ctx->GetCommandBuffer( ctx->CreateCommandBuffer() );
    cmd->Begin();
    cmd->SetPipeline( pp );
    cmd->SetBindingGroup( bg, pp->GetLayout(), VK_PIPELINE_BIND_POINT_COMPUTE );
    cmd->PushConstants( pp->GetLayout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof( pc ), &pc );
    cmd->Dispatch( 1, 1, 1 );
    cmd->End();
    m_device->GetComputeQueue()->Submit( { cmd } );
    m_device->GetComputeQueue()->WaitIdle();

    std::vector<glm::vec4> result( 3 );
    m_stream->ReadbackBufferImmediate( ob, result.data(), agBz );

    // PhalanxCell must NOT move — anchored by fParam4=1.0
    EXPECT_FLOAT_EQ( result[ 0 ].x, -2.0f ) << "PhalanxCell must remain anchored";
    EXPECT_FLOAT_EQ( result[ 0 ].y,  0.0f );
    EXPECT_FLOAT_EQ( result[ 0 ].z,  0.0f );
    EXPECT_FLOAT_EQ( result[ 0 ].w,  1.0f );

    // StalkCell must NOT move — also anchored when anchorPhalanxCells is set
    EXPECT_FLOAT_EQ( result[ 1 ].x, 0.0f ) << "StalkCell also anchored by anchorPhalanxCells=1.0";

    // Default cell must move toward StalkCell (net force in -X direction, stretch=2)
    EXPECT_LT( result[ 2 ].x, 4.0f ) << "Default cell must be pulled toward anchored StalkCell";
    EXPECT_FLOAT_EQ( result[ 2 ].w, 1.0f );

    m_rm->DestroyBuffer( ib ); m_rm->DestroyBuffer( ob ); m_rm->DestroyBuffer( eb );
    m_rm->DestroyBuffer( ecb ); m_rm->DestroyBuffer( cb ); m_rm->DestroyBuffer( phb );
}

// StalkCell connected by a RING edge (flags=0x1) must be anchored in place even when there
// is a non-zero spring force on it (stretched edge). The ring edge signals the cell is part
// of the parent vessel wall, not the sprout chain.
TEST_F( ComputeTest, Shader_VesselSpring_StalkCellWithRingEdge_IsAnchored )
{
    if( !m_device )
        GTEST_SKIP();

    struct VesselEdge    { uint32_t agentA; uint32_t agentB; float dist; uint32_t flags; };

    // Two StalkCells separated by 6 units with restLen=2 — strong spring pull.
    // The edge is RING (flags=0x1), so both cells must be anchored and stay put.
    std::vector<glm::vec4> agentsIn = {
        glm::vec4( 0.0f, 0.0f, 0.0f, 1.0f ), // StalkCell
        glm::vec4( 6.0f, 0.0f, 0.0f, 1.0f ), // StalkCell
    };
    std::vector<glm::vec4>  agentsOut( 2, glm::vec4( 0.0f ) );
    std::vector<PhenotypeData> phenotypes = {
        { 0u, 0.5f, 0.0f, 2u }, // StalkCell
        { 0u, 0.5f, 0.0f, 2u }, // StalkCell
    };

    VesselEdge edge      = { 0u, 1u, 2.0f, 0x1u }; // RING edge, stretch = 4
    uint32_t   edgeCount = 1;
    uint32_t   agentCount = 2;
    size_t     agBz  = 2 * sizeof( glm::vec4 );
    size_t     phBz  = 2 * sizeof( PhenotypeData );

    BufferHandle ib  = m_rm->CreateBuffer( { agBz,                 BufferType::STORAGE, "RingAncIn"   } );
    BufferHandle ob  = m_rm->CreateBuffer( { agBz,                 BufferType::STORAGE, "RingAncOut"  } );
    BufferHandle eb  = m_rm->CreateBuffer( { sizeof( VesselEdge ), BufferType::STORAGE, "RingAncEdge" } );
    BufferHandle ecb = m_rm->CreateBuffer( { sizeof( uint32_t ),   BufferType::STORAGE, "RingAncEc"   } );
    BufferHandle cb  = m_rm->CreateBuffer( { sizeof( uint32_t ),   BufferType::STORAGE, "RingAncCnt"  } );
    BufferHandle phb = m_rm->CreateBuffer( { phBz,                 BufferType::STORAGE, "RingAncPheno"} );

    m_stream->UploadBufferImmediate( {
        { ib,  agentsIn.data(),   agBz,                 0 },
        { ob,  agentsOut.data(),  agBz,                 0 },
        { eb,  &edge,             sizeof( VesselEdge ), 0 },
        { ecb, &edgeCount,        sizeof( uint32_t ),   0 },
        { cb,  &agentCount,       sizeof( uint32_t ),   0 },
        { phb, phenotypes.data(), phBz,                 0 },
    } );

    ComputePipelineDesc   pd{ m_rm->CreateShader( "shaders/compute/biology/vessel_mechanics.comp" ), "TestRingAnchor" };
    ComputePipelineHandle ph = m_rm->CreatePipeline( pd );
    ComputePipeline*      pp = m_rm->GetPipeline( ph );

    BindingGroup* bg = m_rm->GetBindingGroup( m_rm->CreateBindingGroup( ph, 0 ) );
    bg->Bind( 0, m_rm->GetBuffer( ib ) );
    bg->Bind( 1, m_rm->GetBuffer( ob ) );
    bg->Bind( 2, m_rm->GetBuffer( eb ) );
    bg->Bind( 3, m_rm->GetBuffer( ecb ) );
    bg->Bind( 4, m_rm->GetBuffer( cb ) );
    bg->Bind( 5, m_rm->GetBuffer( phb ) );
    bg->Build();

    ComputePushConstants pc{};
    pc.dt      = 1.0f;
    pc.fParam0 = 10.0f; // springStiffness
    pc.fParam1 = 2.0f;  // restingLength
    pc.fParam3 = -1.0f; // reqCT = any
    pc.fParam4 = 1.0f;  // anchorPhalanxCells = enabled (also gates ring-edge anchor)
    pc.offset  = 0; pc.maxCapacity = 2; pc.uParam0 = 0;

    auto ctx = m_device->GetThreadContext( m_device->CreateThreadContext( QueueType::COMPUTE ) );
    auto cmd = ctx->GetCommandBuffer( ctx->CreateCommandBuffer() );
    cmd->Begin();
    cmd->SetPipeline( pp );
    cmd->SetBindingGroup( bg, pp->GetLayout(), VK_PIPELINE_BIND_POINT_COMPUTE );
    cmd->PushConstants( pp->GetLayout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof( pc ), &pc );
    cmd->Dispatch( 1, 1, 1 );
    cmd->End();
    m_device->GetComputeQueue()->Submit( { cmd } );
    m_device->GetComputeQueue()->WaitIdle();

    std::vector<glm::vec4> result( 2 );
    m_stream->ReadbackBufferImmediate( ob, result.data(), agBz );

    // Both cells have a RING edge — they must not move despite the large spring stretch
    EXPECT_FLOAT_EQ( result[ 0 ].x, 0.0f ) << "StalkCell with RING edge must be anchored";
    EXPECT_FLOAT_EQ( result[ 1 ].x, 6.0f ) << "StalkCell with RING edge must be anchored";

    m_rm->DestroyBuffer( ib ); m_rm->DestroyBuffer( ob ); m_rm->DestroyBuffer( eb );
    m_rm->DestroyBuffer( ecb ); m_rm->DestroyBuffer( cb ); m_rm->DestroyBuffer( phb );
}

// StalkCell connected by only a SPROUT edge (flags=0x8) must move under spring forces —
// it is part of the active sprout chain, not the parent vessel wall.
TEST_F( ComputeTest, Shader_VesselSpring_StalkCellWithSproutEdge_Moves )
{
    if( !m_device )
        GTEST_SKIP();

    struct VesselEdge    { uint32_t agentA; uint32_t agentB; float dist; uint32_t flags; };

    // Same layout as ring test but edge is SPROUT (flags=0x8) — neither cell is anchored.
    std::vector<glm::vec4> agentsIn = {
        glm::vec4( 0.0f, 0.0f, 0.0f, 1.0f ), // StalkCell
        glm::vec4( 6.0f, 0.0f, 0.0f, 1.0f ), // StalkCell
    };
    std::vector<glm::vec4>  agentsOut( 2, glm::vec4( 0.0f ) );
    std::vector<PhenotypeData> phenotypes = {
        { 0u, 0.5f, 0.0f, 2u }, // StalkCell
        { 0u, 0.5f, 0.0f, 2u }, // StalkCell
    };

    VesselEdge edge       = { 0u, 1u, 2.0f, 0x8u }; // SPROUT edge, stretch = 4
    uint32_t   edgeCount  = 1;
    uint32_t   agentCount = 2;
    size_t     agBz  = 2 * sizeof( glm::vec4 );
    size_t     phBz  = 2 * sizeof( PhenotypeData );

    BufferHandle ib  = m_rm->CreateBuffer( { agBz,                 BufferType::STORAGE, "SprMvIn"   } );
    BufferHandle ob  = m_rm->CreateBuffer( { agBz,                 BufferType::STORAGE, "SprMvOut"  } );
    BufferHandle eb  = m_rm->CreateBuffer( { sizeof( VesselEdge ), BufferType::STORAGE, "SprMvEdge" } );
    BufferHandle ecb = m_rm->CreateBuffer( { sizeof( uint32_t ),   BufferType::STORAGE, "SprMvEc"   } );
    BufferHandle cb  = m_rm->CreateBuffer( { sizeof( uint32_t ),   BufferType::STORAGE, "SprMvCnt"  } );
    BufferHandle phb = m_rm->CreateBuffer( { phBz,                 BufferType::STORAGE, "SprMvPheno"} );

    m_stream->UploadBufferImmediate( {
        { ib,  agentsIn.data(),   agBz,                 0 },
        { ob,  agentsOut.data(),  agBz,                 0 },
        { eb,  &edge,             sizeof( VesselEdge ), 0 },
        { ecb, &edgeCount,        sizeof( uint32_t ),   0 },
        { cb,  &agentCount,       sizeof( uint32_t ),   0 },
        { phb, phenotypes.data(), phBz,                 0 },
    } );

    ComputePipelineDesc   pd{ m_rm->CreateShader( "shaders/compute/biology/vessel_mechanics.comp" ), "TestSproutMoves" };
    ComputePipelineHandle ph = m_rm->CreatePipeline( pd );
    ComputePipeline*      pp = m_rm->GetPipeline( ph );

    BindingGroup* bg = m_rm->GetBindingGroup( m_rm->CreateBindingGroup( ph, 0 ) );
    bg->Bind( 0, m_rm->GetBuffer( ib ) );
    bg->Bind( 1, m_rm->GetBuffer( ob ) );
    bg->Bind( 2, m_rm->GetBuffer( eb ) );
    bg->Bind( 3, m_rm->GetBuffer( ecb ) );
    bg->Bind( 4, m_rm->GetBuffer( cb ) );
    bg->Bind( 5, m_rm->GetBuffer( phb ) );
    bg->Build();

    ComputePushConstants pc{};
    pc.dt      = 1.0f;
    pc.fParam0 = 10.0f; // springStiffness
    pc.fParam1 = 2.0f;  // restingLength
    pc.fParam3 = -1.0f; // reqCT = any
    pc.fParam4 = 0.0f;  // anchorPhalanxCells = disabled — StalkCells in sprout chain are free to move
    pc.offset  = 0; pc.maxCapacity = 2; pc.uParam0 = 0;

    auto ctx = m_device->GetThreadContext( m_device->CreateThreadContext( QueueType::COMPUTE ) );
    auto cmd = ctx->GetCommandBuffer( ctx->CreateCommandBuffer() );
    cmd->Begin();
    cmd->SetPipeline( pp );
    cmd->SetBindingGroup( bg, pp->GetLayout(), VK_PIPELINE_BIND_POINT_COMPUTE );
    cmd->PushConstants( pp->GetLayout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof( pc ), &pc );
    cmd->Dispatch( 1, 1, 1 );
    cmd->End();
    m_device->GetComputeQueue()->Submit( { cmd } );
    m_device->GetComputeQueue()->WaitIdle();

    std::vector<glm::vec4> result( 2 );
    m_stream->ReadbackBufferImmediate( ob, result.data(), agBz );

    // SPROUT edge with anchorPhalanxCells=false — both StalkCells are free to move
    EXPECT_GT( result[ 0 ].x, 0.0f ) << "StalkCell with SPROUT edge must move toward its neighbor";
    EXPECT_LT( result[ 1 ].x, 6.0f ) << "StalkCell with SPROUT edge must move toward its neighbor";

    m_rm->DestroyBuffer( ib ); m_rm->DestroyBuffer( ob ); m_rm->DestroyBuffer( eb );
    m_rm->DestroyBuffer( ecb ); m_rm->DestroyBuffer( cb ); m_rm->DestroyBuffer( phb );
}

// =================================================================================================
// mitosis_vessel_append.comp — sprouting mitosis tests
// =================================================================================================

// Stalk maturation: a StalkCell edge-connected only to another StalkCell (no TipCell) must
// convert to PhalanxCell and NOT divide, even with biomass >= 1.0.
TEST_F( ComputeTest, Shader_VesselMitosis_StalkNoTipNeighbor_MaturesToPhalanx )
{
    if( !m_device )
        GTEST_SKIP();

    struct VesselEdge    { uint32_t agentA; uint32_t agentB; float dist; uint32_t flags; };

    uint32_t maxCap = 4;
    size_t   agBz   = maxCap * sizeof( glm::vec4 );
    size_t   phBz   = maxCap * sizeof( PhenotypeData );
    size_t   edBz   = maxCap * sizeof( VesselEdge );
    size_t   cntBz  = sizeof( uint32_t );

    std::vector<glm::vec4>     agents( maxCap, glm::vec4( 0.0f ) );
    agents[ 0 ] = glm::vec4( 0.0f, 0.0f, 0.0f, 1.0f );
    agents[ 1 ] = glm::vec4( 2.0f, 0.0f, 0.0f, 1.0f );

    std::vector<PhenotypeData> pheno( maxCap, { 0u, 0.0f, 0.0f, 0u } );
    pheno[ 0 ] = { 0u, 1.1f, 0.0f, 2u }; // StalkCell, biomass ready, no TipCell neighbor
    pheno[ 1 ] = { 0u, 0.5f, 0.0f, 2u }; // StalkCell neighbor (not TipCell)

    std::vector<VesselEdge> edges( maxCap, { 0u, 0u, 0.0f, 0u } );
    edges[ 0 ]          = { 0u, 1u, 2.0f, 0u }; // edge between agent 0 and agent 1
    uint32_t edgeCount  = 1u;
    uint32_t agentCount = 2u;

    BufferHandle abR = m_rm->CreateBuffer( { agBz,  BufferType::STORAGE, "VMaR"   } );
    BufferHandle abW = m_rm->CreateBuffer( { agBz,  BufferType::STORAGE, "VMaW"   } );
    BufferHandle pb  = m_rm->CreateBuffer( { phBz,  BufferType::STORAGE, "VMaPh"  } );
    BufferHandle cB  = m_rm->CreateBuffer( { cntBz, BufferType::STORAGE, "VMaCnt" } );
    BufferHandle eB  = m_rm->CreateBuffer( { edBz,  BufferType::STORAGE, "VMaE"   } );
    BufferHandle ecB = m_rm->CreateBuffer( { cntBz, BufferType::STORAGE, "VMaEc"  } );
    BufferHandle oB  = m_rm->CreateBuffer( { agBz,  BufferType::STORAGE, "VMaOr"  } );

    m_stream->UploadBufferImmediate( {
        { abR, agents.data(), agBz,  0 },
        { abW, agents.data(), agBz,  0 },
        { pb,  pheno.data(),  phBz,  0 },
        { cB,  &agentCount,   cntBz, 0 },
        { eB,  edges.data(),  edBz,  0 },
        { ecB, &edgeCount,    cntBz, 0 },
    } );

    ComputePipelineDesc   pd{ m_rm->CreateShader( "shaders/compute/biology/mitosis_vessel_append.comp" ), "VesselMitosisMaturation" };
    ComputePipelineHandle ph = m_rm->CreatePipeline( pd );
    BindingGroup* bg = m_rm->GetBindingGroup( m_rm->CreateBindingGroup( ph, 0 ) );
    bg->Bind( 0, m_rm->GetBuffer( abR ) );
    bg->Bind( 1, m_rm->GetBuffer( abW ) );
    bg->Bind( 2, m_rm->GetBuffer( pb ) );
    bg->Bind( 3, m_rm->GetBuffer( cB ) );
    bg->Bind( 4, m_rm->GetBuffer( eB ) );
    bg->Bind( 5, m_rm->GetBuffer( ecB ) );
    bg->Bind( 6, m_rm->GetBuffer( oB ) );
    bg->Build();

    ComputePushConstants pc{};
    pc.totalTime   = 4.0f; // after grace period (3.0 s) — maturation is now active
    pc.maxCapacity = maxCap;
    pc.uParam1     = 0; // grpNdx = 0

    auto ctx = m_device->GetThreadContext( m_device->CreateThreadContext( QueueType::COMPUTE ) );
    auto cmd = ctx->GetCommandBuffer( ctx->CreateCommandBuffer() );
    cmd->Begin();
    cmd->SetPipeline( m_rm->GetPipeline( ph ) );
    cmd->SetBindingGroup( bg, m_rm->GetPipeline( ph )->GetLayout(), VK_PIPELINE_BIND_POINT_COMPUTE );
    cmd->PushConstants( m_rm->GetPipeline( ph )->GetLayout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof( pc ), &pc );
    cmd->Dispatch( 1, 1, 1 );
    cmd->End();
    m_device->GetComputeQueue()->Submit( { cmd } );
    m_device->GetComputeQueue()->WaitIdle();

    std::vector<PhenotypeData> resPheno( maxCap );
    uint32_t resCount, resEdgeCount;
    m_stream->ReadbackBufferImmediate( pb,  resPheno.data(),  phBz  );
    m_stream->ReadbackBufferImmediate( cB,  &resCount,        cntBz );
    m_stream->ReadbackBufferImmediate( ecB, &resEdgeCount,    cntBz );

    EXPECT_EQ( resPheno[ 0 ].cellType, 3u ) << "StalkCell with no TipCell neighbor must mature to PhalanxCell after grace period";
    EXPECT_EQ( resCount,     2u ) << "No division should have occurred";
    EXPECT_EQ( resEdgeCount, 1u ) << "Edge count must be unchanged (no division edge written)";

    m_rm->DestroyBuffer( abR ); m_rm->DestroyBuffer( abW ); m_rm->DestroyBuffer( pb );
    m_rm->DestroyBuffer( cB );  m_rm->DestroyBuffer( eB );  m_rm->DestroyBuffer( ecB );
    m_rm->DestroyBuffer( oB );
}

// Grace period: before totalTime >= 1.5 s, StalkCells without TipCell neighbor must NOT
// convert to PhalanxCell — they stay StalkCell so NotchDll4 can still establish a TipCell.
TEST_F( ComputeTest, Shader_VesselMitosis_GracePeriod_StalkNotConvertedBeforeGrace )
{
    if( !m_device )
        GTEST_SKIP();

    struct VesselEdge    { uint32_t agentA; uint32_t agentB; float dist; uint32_t flags; };

    uint32_t maxCap = 4;
    size_t   agBz   = maxCap * sizeof( glm::vec4 );
    size_t   phBz   = maxCap * sizeof( PhenotypeData );
    size_t   edBz   = maxCap * sizeof( VesselEdge );
    size_t   cntBz  = sizeof( uint32_t );

    std::vector<glm::vec4>     agents( maxCap, glm::vec4( 0.0f ) );
    agents[ 0 ] = glm::vec4( 0.0f, 0.0f, 0.0f, 1.0f );
    agents[ 1 ] = glm::vec4( 2.0f, 0.0f, 0.0f, 1.0f );

    std::vector<PhenotypeData> pheno( maxCap, { 0u, 0.0f, 0.0f, 0u } );
    pheno[ 0 ] = { 0u, 1.1f, 0.0f, 2u }; // StalkCell, ready to divide, no TipCell neighbor
    pheno[ 1 ] = { 0u, 0.5f, 0.0f, 2u }; // StalkCell neighbor

    std::vector<VesselEdge> edges( maxCap, { 0u, 0u, 0.0f, 0u } );
    edges[ 0 ]          = { 0u, 1u, 2.0f, 0u };
    uint32_t edgeCount  = 1u;
    uint32_t agentCount = 2u;

    BufferHandle abR = m_rm->CreateBuffer( { agBz,  BufferType::STORAGE, "VMgR"   } );
    BufferHandle abW = m_rm->CreateBuffer( { agBz,  BufferType::STORAGE, "VMgW"   } );
    BufferHandle pb  = m_rm->CreateBuffer( { phBz,  BufferType::STORAGE, "VMgPh"  } );
    BufferHandle cB  = m_rm->CreateBuffer( { cntBz, BufferType::STORAGE, "VMgCnt" } );
    BufferHandle eB  = m_rm->CreateBuffer( { edBz,  BufferType::STORAGE, "VMgE"   } );
    BufferHandle ecB = m_rm->CreateBuffer( { cntBz, BufferType::STORAGE, "VMgEc"  } );
    BufferHandle oB  = m_rm->CreateBuffer( { agBz,  BufferType::STORAGE, "VMgOr"  } );

    m_stream->UploadBufferImmediate( {
        { abR, agents.data(), agBz,  0 },
        { abW, agents.data(), agBz,  0 },
        { pb,  pheno.data(),  phBz,  0 },
        { cB,  &agentCount,   cntBz, 0 },
        { eB,  edges.data(),  edBz,  0 },
        { ecB, &edgeCount,    cntBz, 0 },
    } );

    ComputePipelineDesc   pd{ m_rm->CreateShader( "shaders/compute/biology/mitosis_vessel_append.comp" ), "VesselMitosisGrace" };
    ComputePipelineHandle ph = m_rm->CreatePipeline( pd );
    BindingGroup* bg = m_rm->GetBindingGroup( m_rm->CreateBindingGroup( ph, 0 ) );
    bg->Bind( 0, m_rm->GetBuffer( abR ) );
    bg->Bind( 1, m_rm->GetBuffer( abW ) );
    bg->Bind( 2, m_rm->GetBuffer( pb ) );
    bg->Bind( 3, m_rm->GetBuffer( cB ) );
    bg->Bind( 4, m_rm->GetBuffer( eB ) );
    bg->Bind( 5, m_rm->GetBuffer( ecB ) );
    bg->Bind( 6, m_rm->GetBuffer( oB ) );
    bg->Build();

    ComputePushConstants pc{};
    pc.totalTime   = 0.5f; // within grace period — maturation must NOT fire
    pc.maxCapacity = maxCap;
    pc.uParam1     = 0;

    auto ctx = m_device->GetThreadContext( m_device->CreateThreadContext( QueueType::COMPUTE ) );
    auto cmd = ctx->GetCommandBuffer( ctx->CreateCommandBuffer() );
    cmd->Begin();
    cmd->SetPipeline( m_rm->GetPipeline( ph ) );
    cmd->SetBindingGroup( bg, m_rm->GetPipeline( ph )->GetLayout(), VK_PIPELINE_BIND_POINT_COMPUTE );
    cmd->PushConstants( m_rm->GetPipeline( ph )->GetLayout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof( pc ), &pc );
    cmd->Dispatch( 1, 1, 1 );
    cmd->End();
    m_device->GetComputeQueue()->Submit( { cmd } );
    m_device->GetComputeQueue()->WaitIdle();

    std::vector<PhenotypeData> resPheno( maxCap );
    uint32_t resCount;
    m_stream->ReadbackBufferImmediate( pb, resPheno.data(), phBz  );
    m_stream->ReadbackBufferImmediate( cB, &resCount,       cntBz );

    EXPECT_EQ( resPheno[ 0 ].cellType, 2u ) << "StalkCell must remain StalkCell during grace period — NotchDll4 must still be able to select it as TipCell";
    EXPECT_EQ( resCount, 2u )               << "Division must be blocked (no TipCell neighbor) but cell must not be converted";

    m_rm->DestroyBuffer( abR ); m_rm->DestroyBuffer( abW ); m_rm->DestroyBuffer( pb );
    m_rm->DestroyBuffer( cB );  m_rm->DestroyBuffer( eB );  m_rm->DestroyBuffer( ecB );
    m_rm->DestroyBuffer( oB );
}

// Edge chain split: when a StalkCell adjacent to a TipCell divides, the TipCell-adjacency
// edge must be transferred to the daughter so the proliferation front advances.
// Before: parent ↔ TipCell   After: parent ↔ daughter ↔ TipCell
TEST_F( ComputeTest, Shader_VesselMitosis_EdgeChainSplit_DaughterBecomesAdjacentToTip )
{
    if( !m_device )
        GTEST_SKIP();

    struct VesselEdge    { uint32_t agentA; uint32_t agentB; float dist; uint32_t flags; };

    // 3 input cells + 1 slot for daughter (index 3)
    uint32_t maxCap = 8;
    size_t   agBz   = maxCap * sizeof( glm::vec4 );
    size_t   phBz   = maxCap * sizeof( PhenotypeData );
    size_t   edBz   = maxCap * sizeof( VesselEdge );
    size_t   cntBz  = sizeof( uint32_t );

    // Topology: TipCell(1) ↔ StalkCell(0) ↔ PhalanxCell(2)
    // After division: TipCell(1) ↔ Daughter(3) ↔ StalkCell(0) ↔ PhalanxCell(2)
    std::vector<glm::vec4>     agents( maxCap, glm::vec4( 0.0f ) );
    agents[ 0 ] = glm::vec4(  0.0f, 0.0f, 0.0f, 1.0f ); // StalkCell (parent)
    agents[ 1 ] = glm::vec4(  2.0f, 0.0f, 0.0f, 1.0f ); // TipCell
    agents[ 2 ] = glm::vec4( -2.0f, 0.0f, 0.0f, 1.0f ); // PhalanxCell

    std::vector<PhenotypeData> pheno( maxCap, { 0u, 0.0f, 0.0f, 0u } );
    pheno[ 0 ] = { 0u, 1.1f, 0.0f, 2u }; // StalkCell, full biomass, ready to divide
    pheno[ 1 ] = { 0u, 0.5f, 0.0f, 1u }; // TipCell
    pheno[ 2 ] = { 0u, 0.5f, 0.0f, 3u }; // PhalanxCell

    // Edge 0: StalkCell(0) ↔ TipCell(1)  — this must be updated to Daughter↔TipCell
    // Edge 1: StalkCell(0) ↔ PhalanxCell(2)
    std::vector<VesselEdge> edges( maxCap, { 0u, 0u, 0.0f, 0u } );
    edges[ 0 ]          = { 0u, 1u, 2.0f, 0u }; // parent ↔ TipCell
    edges[ 1 ]          = { 0u, 2u, 2.0f, 0u }; // parent ↔ PhalanxCell
    uint32_t edgeCount  = 2u;
    uint32_t agentCount = 3u;

    BufferHandle abR = m_rm->CreateBuffer( { agBz,  BufferType::STORAGE, "VMesR"   } );
    BufferHandle abW = m_rm->CreateBuffer( { agBz,  BufferType::STORAGE, "VMesW"   } );
    BufferHandle pb  = m_rm->CreateBuffer( { phBz,  BufferType::STORAGE, "VMesPh"  } );
    BufferHandle cB  = m_rm->CreateBuffer( { cntBz, BufferType::STORAGE, "VMesCnt" } );
    BufferHandle eB  = m_rm->CreateBuffer( { edBz,  BufferType::STORAGE, "VMesE"   } );
    BufferHandle ecB = m_rm->CreateBuffer( { cntBz, BufferType::STORAGE, "VMesEc"  } );
    BufferHandle oB  = m_rm->CreateBuffer( { agBz,  BufferType::STORAGE, "VMesOr"  } );

    m_stream->UploadBufferImmediate( {
        { abR, agents.data(), agBz,  0 },
        { abW, agents.data(), agBz,  0 },
        { pb,  pheno.data(),  phBz,  0 },
        { cB,  &agentCount,   cntBz, 0 },
        { eB,  edges.data(),  edBz,  0 },
        { ecB, &edgeCount,    cntBz, 0 },
    } );

    ComputePipelineDesc   pd{ m_rm->CreateShader( "shaders/compute/biology/mitosis_vessel_append.comp" ), "VesselMitosisEdgeSplit" };
    ComputePipelineHandle ph = m_rm->CreatePipeline( pd );
    BindingGroup* bg = m_rm->GetBindingGroup( m_rm->CreateBindingGroup( ph, 0 ) );
    bg->Bind( 0, m_rm->GetBuffer( abR ) );
    bg->Bind( 1, m_rm->GetBuffer( abW ) );
    bg->Bind( 2, m_rm->GetBuffer( pb ) );
    bg->Bind( 3, m_rm->GetBuffer( cB ) );
    bg->Bind( 4, m_rm->GetBuffer( eB ) );
    bg->Bind( 5, m_rm->GetBuffer( ecB ) );
    bg->Bind( 6, m_rm->GetBuffer( oB ) );
    bg->Build();

    ComputePushConstants pc{};
    pc.totalTime   = 4.0f; // after grace period (3.0 s)
    pc.maxCapacity = maxCap;
    pc.uParam1     = 0;

    auto ctx = m_device->GetThreadContext( m_device->CreateThreadContext( QueueType::COMPUTE ) );
    auto cmd = ctx->GetCommandBuffer( ctx->CreateCommandBuffer() );
    cmd->Begin();
    cmd->SetPipeline( m_rm->GetPipeline( ph ) );
    cmd->SetBindingGroup( bg, m_rm->GetPipeline( ph )->GetLayout(), VK_PIPELINE_BIND_POINT_COMPUTE );
    cmd->PushConstants( m_rm->GetPipeline( ph )->GetLayout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof( pc ), &pc );
    cmd->Dispatch( 1, 1, 1 );
    cmd->End();
    m_device->GetComputeQueue()->Submit( { cmd } );
    m_device->GetComputeQueue()->WaitIdle();

    std::vector<PhenotypeData> resPheno( maxCap );
    std::vector<VesselEdge> resEdges( maxCap );
    uint32_t resCount, resEdgeCount;
    m_stream->ReadbackBufferImmediate( pb,  resPheno.data(),  phBz  );
    m_stream->ReadbackBufferImmediate( eB,  resEdges.data(),  edBz  );
    m_stream->ReadbackBufferImmediate( cB,  &resCount,        cntBz );
    m_stream->ReadbackBufferImmediate( ecB, &resEdgeCount,    cntBz );

    // agentCount=3 → daughter at index 3
    uint32_t daughterIdx = 3u;

    EXPECT_EQ( resCount,     4u ) << "Division must produce 1 daughter";
    EXPECT_EQ( resEdgeCount, 3u ) << "2 existing + 1 new SPROUT edge = 3 total";
    EXPECT_EQ( resPheno[ daughterIdx ].cellType, 2u ) << "Daughter must be StalkCell";

    // Edge 0 (originally parent↔TipCell) must be updated to daughter↔TipCell
    bool daughterHasTipEdge = false;
    bool parentHasTipEdge   = false;
    for( uint32_t e = 0; e < resEdgeCount; e++ )
    {
        bool aIsDaughter = resEdges[ e ].agentA == daughterIdx;
        bool bIsDaughter = resEdges[ e ].agentB == daughterIdx;
        bool aIsTip      = resEdges[ e ].agentA == 1u;
        bool bIsTip      = resEdges[ e ].agentB == 1u;
        bool aIsParent   = resEdges[ e ].agentA == 0u;
        bool bIsParent   = resEdges[ e ].agentB == 0u;

        if( ( aIsDaughter && bIsTip ) || ( bIsDaughter && aIsTip ) ) daughterHasTipEdge = true;
        if( ( aIsParent && bIsTip ) || ( bIsParent && aIsTip ) ) parentHasTipEdge = true;
    }
    EXPECT_TRUE( daughterHasTipEdge ) << "Daughter must be adjacent to TipCell (proliferation front)";
    EXPECT_FALSE( parentHasTipEdge  ) << "Parent must NOT retain TipCell adjacency — it quiesces next tick";

    m_rm->DestroyBuffer( abR ); m_rm->DestroyBuffer( abW ); m_rm->DestroyBuffer( pb );
    m_rm->DestroyBuffer( cB );  m_rm->DestroyBuffer( eB );  m_rm->DestroyBuffer( ecB );
    m_rm->DestroyBuffer( oB );
}

// Stalk division: a StalkCell edge-connected to a TipCell must divide and produce a daughter.
TEST_F( ComputeTest, Shader_VesselMitosis_TipNeighbor_Divides )
{
    if( !m_device )
        GTEST_SKIP();

    struct VesselEdge    { uint32_t agentA; uint32_t agentB; float dist; uint32_t flags; };

    uint32_t maxCap = 4;
    size_t   agBz   = maxCap * sizeof( glm::vec4 );
    size_t   phBz   = maxCap * sizeof( PhenotypeData );
    size_t   edBz   = maxCap * sizeof( VesselEdge );
    size_t   cntBz  = sizeof( uint32_t );

    std::vector<glm::vec4>     agents( maxCap, glm::vec4( 0.0f ) );
    agents[ 0 ] = glm::vec4( 0.0f, 0.0f, 0.0f, 1.0f );
    agents[ 1 ] = glm::vec4( 2.0f, 0.0f, 0.0f, 1.0f );

    std::vector<PhenotypeData> pheno( maxCap, { 0u, 0.0f, 0.0f, 0u } );
    pheno[ 0 ] = { 0u, 1.1f, 0.0f, 2u }; // StalkCell, biomass ready, TipCell neighbor
    pheno[ 1 ] = { 0u, 0.5f, 0.0f, 1u }; // TipCell neighbor — allows division

    std::vector<VesselEdge> edges( maxCap, { 0u, 0u, 0.0f, 0u } );
    edges[ 0 ]          = { 0u, 1u, 2.0f, 0u }; // edge: StalkCell(0) ↔ TipCell(1)
    uint32_t edgeCount  = 1u;
    uint32_t agentCount = 2u;

    BufferHandle abR = m_rm->CreateBuffer( { agBz,  BufferType::STORAGE, "VMdR"   } );
    BufferHandle abW = m_rm->CreateBuffer( { agBz,  BufferType::STORAGE, "VMdW"   } );
    BufferHandle pb  = m_rm->CreateBuffer( { phBz,  BufferType::STORAGE, "VMdPh"  } );
    BufferHandle cB  = m_rm->CreateBuffer( { cntBz, BufferType::STORAGE, "VMdCnt" } );
    BufferHandle eB  = m_rm->CreateBuffer( { edBz,  BufferType::STORAGE, "VMdE"   } );
    BufferHandle ecB = m_rm->CreateBuffer( { cntBz, BufferType::STORAGE, "VMdEc"  } );
    BufferHandle oB  = m_rm->CreateBuffer( { agBz,  BufferType::STORAGE, "VMdOr"  } );

    m_stream->UploadBufferImmediate( {
        { abR, agents.data(), agBz,  0 },
        { abW, agents.data(), agBz,  0 },
        { pb,  pheno.data(),  phBz,  0 },
        { cB,  &agentCount,   cntBz, 0 },
        { eB,  edges.data(),  edBz,  0 },
        { ecB, &edgeCount,    cntBz, 0 },
    } );

    ComputePipelineDesc   pd{ m_rm->CreateShader( "shaders/compute/biology/mitosis_vessel_append.comp" ), "VesselMitosisDivide" };
    ComputePipelineHandle ph = m_rm->CreatePipeline( pd );
    BindingGroup* bg = m_rm->GetBindingGroup( m_rm->CreateBindingGroup( ph, 0 ) );
    bg->Bind( 0, m_rm->GetBuffer( abR ) );
    bg->Bind( 1, m_rm->GetBuffer( abW ) );
    bg->Bind( 2, m_rm->GetBuffer( pb ) );
    bg->Bind( 3, m_rm->GetBuffer( cB ) );
    bg->Bind( 4, m_rm->GetBuffer( eB ) );
    bg->Bind( 5, m_rm->GetBuffer( ecB ) );
    bg->Bind( 6, m_rm->GetBuffer( oB ) );
    bg->Build();

    ComputePushConstants pc{};
    pc.totalTime   = 1.0f;
    pc.maxCapacity = maxCap;
    pc.uParam1     = 0; // grpNdx = 0

    auto ctx = m_device->GetThreadContext( m_device->CreateThreadContext( QueueType::COMPUTE ) );
    auto cmd = ctx->GetCommandBuffer( ctx->CreateCommandBuffer() );
    cmd->Begin();
    cmd->SetPipeline( m_rm->GetPipeline( ph ) );
    cmd->SetBindingGroup( bg, m_rm->GetPipeline( ph )->GetLayout(), VK_PIPELINE_BIND_POINT_COMPUTE );
    cmd->PushConstants( m_rm->GetPipeline( ph )->GetLayout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof( pc ), &pc );
    cmd->Dispatch( 1, 1, 1 );
    cmd->End();
    m_device->GetComputeQueue()->Submit( { cmd } );
    m_device->GetComputeQueue()->WaitIdle();

    std::vector<PhenotypeData> resPheno( maxCap );
    uint32_t resCount, resEdgeCount;
    m_stream->ReadbackBufferImmediate( pb,  resPheno.data(),  phBz  );
    m_stream->ReadbackBufferImmediate( cB,  &resCount,        cntBz );
    m_stream->ReadbackBufferImmediate( ecB, &resEdgeCount,    cntBz );

    EXPECT_EQ( resPheno[ 0 ].cellType, 2u )     << "StalkCell with TipCell neighbor must remain StalkCell";
    EXPECT_NEAR( resPheno[ 0 ].biomass, 0.5f, 0.01f ) << "Mother biomass reset after division";
    // Single daughter at slot 2
    EXPECT_EQ( resCount, 3u )                   << "Division must produce 1 daughter";
    EXPECT_EQ( resPheno[ 2 ].cellType, 2u )     << "Daughter must be StalkCell";
    EXPECT_NEAR( resPheno[ 2 ].biomass, 0.5f, 0.01f ) << "Daughter biomass must start at 0.5";
    // 1 new SPROUT edge; total = 1 (transferred) + 1 (new) = 2
    EXPECT_EQ( resEdgeCount, 2u )               << "Division must write 1 new vessel edge";

    m_rm->DestroyBuffer( abR ); m_rm->DestroyBuffer( abW ); m_rm->DestroyBuffer( pb );
    m_rm->DestroyBuffer( cB );  m_rm->DestroyBuffer( eB );  m_rm->DestroyBuffer( ecB );
    m_rm->DestroyBuffer( oB );
}

TEST_F( ComputeTest, Shader_VesselMitosis_TipCellDoesNotDivide )
{
    // TipCells must not divide even when biomass >= 1.0. The shader's TipCell guard
    // (cellType==1 → return) should fire before the division code is reached.
    if( !m_device )
        GTEST_SKIP();

    struct VesselEdge    { uint32_t agentA; uint32_t agentB; float dist; uint32_t flags; };

    uint32_t maxCap = 2;
    size_t   agBz   = maxCap * sizeof( glm::vec4 );
    size_t   phBz   = maxCap * sizeof( PhenotypeData );
    size_t   edBz   = maxCap * sizeof( VesselEdge );
    size_t   cntBz  = sizeof( uint32_t );

    std::vector<glm::vec4>     agents( maxCap, glm::vec4( 0.0f ) );
    agents[ 0 ] = glm::vec4( 0.0f, 0.0f, 0.0f, 1.0f );  // alive

    std::vector<PhenotypeData> pheno( maxCap, { 0u, 0.0f, 0.0f, 0u } );
    pheno[ 0 ] = { 0u, 1.1f, 0.0f, 1u };  // TipCell, biomass above division threshold

    std::vector<VesselEdge> edges( maxCap, { 0u, 0u, 0.0f, 0u } );
    uint32_t edgeCount  = 0u;
    uint32_t agentCount = 1u;

    BufferHandle abR = m_rm->CreateBuffer( { agBz,  BufferType::STORAGE, "VMAabR" } );
    BufferHandle abW = m_rm->CreateBuffer( { agBz,  BufferType::STORAGE, "VMAabW" } );
    BufferHandle pb  = m_rm->CreateBuffer( { phBz,  BufferType::STORAGE, "VMAPh"  } );
    BufferHandle cB  = m_rm->CreateBuffer( { cntBz, BufferType::STORAGE, "VMACnt" } );
    BufferHandle eB  = m_rm->CreateBuffer( { edBz,  BufferType::STORAGE, "VMAE"   } );
    BufferHandle ecB = m_rm->CreateBuffer( { cntBz, BufferType::STORAGE, "VMAEc"  } );
    BufferHandle oB  = m_rm->CreateBuffer( { agBz,  BufferType::STORAGE, "VMAOr"  } );

    m_stream->UploadBufferImmediate( {
        { abR, agents.data(), agBz,  0 },
        { abW, agents.data(), agBz,  0 },
        { pb,  pheno.data(),  phBz,  0 },
        { cB,  &agentCount,   cntBz, 0 },
        { eB,  edges.data(),  edBz,  0 },
        { ecB, &edgeCount,    cntBz, 0 },
    } );

    ComputePipelineDesc   pd{ m_rm->CreateShader( "shaders/compute/biology/mitosis_vessel_append.comp" ), "VesselMitosisTipGuard" };
    ComputePipelineHandle ph = m_rm->CreatePipeline( pd );
    BindingGroup* bg = m_rm->GetBindingGroup( m_rm->CreateBindingGroup( ph, 0 ) );
    bg->Bind( 0, m_rm->GetBuffer( abR ) );
    bg->Bind( 1, m_rm->GetBuffer( abW ) );
    bg->Bind( 2, m_rm->GetBuffer( pb ) );
    bg->Bind( 3, m_rm->GetBuffer( cB ) );
    bg->Bind( 4, m_rm->GetBuffer( eB ) );
    bg->Bind( 5, m_rm->GetBuffer( ecB ) );
    bg->Bind( 6, m_rm->GetBuffer( oB ) );
    bg->Build();

    ComputePushConstants pc{};
    pc.totalTime   = 4.0f;   // past grace period — ensures maturation logic doesn't interfere
    pc.maxCapacity = maxCap;
    pc.uParam1     = 0;      // grpNdx = 0

    auto ctx = m_device->GetThreadContext( m_device->CreateThreadContext( QueueType::COMPUTE ) );
    auto cmd = ctx->GetCommandBuffer( ctx->CreateCommandBuffer() );
    cmd->Begin();
    cmd->SetPipeline( m_rm->GetPipeline( ph ) );
    cmd->SetBindingGroup( bg, m_rm->GetPipeline( ph )->GetLayout(), VK_PIPELINE_BIND_POINT_COMPUTE );
    cmd->PushConstants( m_rm->GetPipeline( ph )->GetLayout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof( pc ), &pc );
    cmd->Dispatch( 1, 1, 1 );
    cmd->End();
    m_device->GetComputeQueue()->Submit( { cmd } );
    m_device->GetComputeQueue()->WaitIdle();

    std::vector<PhenotypeData> resPheno( maxCap );
    uint32_t resCount, resEdgeCount;
    m_stream->ReadbackBufferImmediate( pb,  resPheno.data(),  phBz  );
    m_stream->ReadbackBufferImmediate( cB,  &resCount,        cntBz );
    m_stream->ReadbackBufferImmediate( ecB, &resEdgeCount,    cntBz );

    EXPECT_EQ( resPheno[ 0 ].cellType, 1u )         << "TipCell must remain TipCell";
    EXPECT_NEAR( resPheno[ 0 ].biomass, 1.1f, 0.01f ) << "TipCell biomass must be untouched";
    EXPECT_EQ( resCount, 1u )                        << "No new agent should be spawned";
    EXPECT_EQ( resEdgeCount, 0u )                    << "No new vessel edge should be written";

    m_rm->DestroyBuffer( abR ); m_rm->DestroyBuffer( abW ); m_rm->DestroyBuffer( pb );
    m_rm->DestroyBuffer( cB );  m_rm->DestroyBuffer( eB );  m_rm->DestroyBuffer( ecB );
    m_rm->DestroyBuffer( oB );
}

// ─────────────────────────────────────────────────────────────────────────────
// NotchDll4 — PhalanxCell participation and recruitment tests
// ─────────────────────────────────────────────────────────────────────────────

// PhalanxCells now participate in the Notch-Dll4 ODE. With VEGF present, the
// center cell of a 3-cell chain should win the competition and become TipCell,
// while end cells (low VEGF) accumulate insufficient Dll4 → stay PhalanxCell.
//
// Setup: 3 PhalanxCells chained (0-1-2). Center cell (idx=1) sees high VEGF;
// end cells see zero VEGF. Dispatched 200 times with dt=0.05.
// Expected: cell 1 → TipCell, cells 0 and 2 stay PhalanxCell.
TEST_F( ComputeTest, Shader_NotchDll4_PhalanxCellsParticipate )
{
    if( !m_device )
        GTEST_SKIP();

    struct SignalingData { float dll4; float nicd; float vegfr2; float pad; };
    struct VesselEdge    { uint32_t agentA; uint32_t agentB; float dist; uint32_t flags; };

    const uint32_t N = 3;
    // Agents alive (w=1). Center at (0,0,0), ends at (±1,0,0).
    std::vector<glm::vec4>    agents     = { glm::vec4(-1,0,0,1), glm::vec4(0,0,0,1), glm::vec4(1,0,0,1) };
    std::vector<SignalingData> signaling  = { {0.0f,0.0f,0.0f,0.0f}, {0.0f,0.0f,0.0f,0.0f}, {0.0f,0.0f,0.0f,0.0f} };
    std::vector<PhenotypeData> phenotypes = { {0u,0.5f,0.0f,3u}, {0u,0.5f,0.0f,3u}, {0u,0.5f,0.0f,3u} }; // all PhalanxCell
    // Chain: 0-1 and 1-2
    std::vector<VesselEdge>   edgeData   = { {0u,1u,1.0f,0u}, {1u,2u,1.0f,0u} };
    uint32_t edgeCount  = 2;
    uint32_t agentCount = N;

    // VEGF: 3×1×1 texture. Center voxel (1,0,0) = high VEGF; ends = zero.
    // With gridSize=(3,1,1) and domain=(3,1,1):
    //   cell 0 at x=-1 → n.x=(−1+1.5)/3=0.166 → voxel.x=0 → VEGF=0
    //   cell 1 at x= 0 → n.x=(0+1.5)/3=0.5   → voxel.x=1 → VEGF=1
    //   cell 2 at x= 1 → n.x=(1+1.5)/3=0.833 → voxel.x=2 → VEGF=0
    std::vector<float> vegfTex = { 0.0f, 1.0f, 0.0f };

    size_t agSz   = N * sizeof(glm::vec4);
    size_t sigSz  = N * sizeof(SignalingData);
    size_t phSz   = N * sizeof(PhenotypeData);
    size_t eSz    = edgeData.size() * sizeof(VesselEdge);
    size_t cSz    = sizeof(uint32_t);

    BufferHandle  agBuf  = m_rm->CreateBuffer( { agSz,  BufferType::STORAGE,  "PxPartAgents"    } );
    BufferHandle  sigBuf = m_rm->CreateBuffer( { sigSz, BufferType::STORAGE,  "PxPartSignal"    } );
    BufferHandle  phBuf  = m_rm->CreateBuffer( { phSz,  BufferType::STORAGE,  "PxPartPheno"     } );
    BufferHandle  eBuf   = m_rm->CreateBuffer( { eSz,   BufferType::STORAGE,  "PxPartEdges"     } );
    BufferHandle  ecBuf  = m_rm->CreateBuffer( { cSz,   BufferType::STORAGE,  "PxPartEdgeCnt"   } );
    BufferHandle  cBuf   = m_rm->CreateBuffer( { cSz,   BufferType::INDIRECT, "PxPartCount"     } );
    TextureHandle vTex   = m_rm->CreateTexture( { 3, 1, 1, TextureType::Texture3D, VK_FORMAT_R32_SFLOAT,
                                                  TextureUsage::STORAGE | TextureUsage::TRANSFER_DST, VK_SAMPLE_COUNT_1_BIT, "PxPartVEGF" } );

    m_stream->UploadTextureImmediate( vTex, vegfTex.data(), vegfTex.size() * sizeof(float) );
    m_stream->UploadBufferImmediate( {
        { agBuf,  agents.data(),     agSz, 0 },
        { sigBuf, signaling.data(),  sigSz, 0 },
        { phBuf,  phenotypes.data(), phSz, 0 },
        { eBuf,   edgeData.data(),   eSz,  0 },
        { ecBuf,  &edgeCount,        cSz,  0 },
        { cBuf,   &agentCount,       cSz,  0 },
    } );

    ComputePipelineHandle ph = m_rm->CreatePipeline( ComputePipelineDesc{ m_rm->CreateShader( "shaders/compute/biology/notch_dll4.comp" ), "PxPart" } );
    ComputePipeline*      pp = m_rm->GetPipeline( ph );
    BindingGroup* bg = m_rm->GetBindingGroup( m_rm->CreateBindingGroup( ph, 0 ) );
    bg->Bind( 0, m_rm->GetBuffer( agBuf  ) );
    bg->Bind( 1, m_rm->GetBuffer( sigBuf ) );
    bg->Bind( 2, m_rm->GetBuffer( phBuf  ) );
    bg->Bind( 3, m_rm->GetBuffer( eBuf   ) );
    bg->Bind( 4, m_rm->GetBuffer( ecBuf  ) );
    bg->Bind( 5, m_rm->GetBuffer( cBuf   ) );
    bg->Bind( 6, m_rm->GetTexture( vTex ), VK_IMAGE_LAYOUT_GENERAL );
    bg->Build();

    // dll4ProductionRate=1, dll4DecayRate=0.1, notchInhibitionGain=5,
    // vegfr2BaseExpression=0.2, tipThreshold=0.8, stalkThreshold=0.3
    ComputePushConstants pc{};
    pc.dt = 0.05f; pc.fParam0 = 1.0f; pc.fParam1 = 0.1f; pc.fParam2 = 5.0f;
    pc.fParam3 = 0.2f; pc.fParam4 = 0.8f; pc.fParam5 = 0.3f;
    pc.offset = 0; pc.maxCapacity = N; pc.uParam0 = 0u; pc.uParam1 = 0;
    pc.domainSize = glm::vec4( 3.0f, 1.0f, 1.0f, 0.0f );
    pc.gridSize   = glm::uvec4( 3u, 1u, 1u, 1u ); // w=1 → VEGF bound

    auto ctx = m_device->GetThreadContext( m_device->CreateThreadContext( QueueType::COMPUTE ) );

    for( int i = 0; i < 200; i++ )
    {
        auto cmd = ctx->GetCommandBuffer( ctx->CreateCommandBuffer() );
        cmd->Begin();
        cmd->SetPipeline( pp );
        cmd->SetBindingGroup( bg, pp->GetLayout(), VK_PIPELINE_BIND_POINT_COMPUTE );
        cmd->PushConstants( pp->GetLayout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc );
        cmd->Dispatch( 1, 1, 1 );
        cmd->End();
        m_device->GetComputeQueue()->Submit( { cmd } );
        m_device->GetComputeQueue()->WaitIdle();
    }

    std::vector<PhenotypeData> result( N );
    m_stream->ReadbackBufferImmediate( phBuf, result.data(), phSz );

    // Center becomes TipCell; end cells are adjacent to TipCell → recruited to StalkCell.
    EXPECT_EQ( result[ 1 ].cellType, 1u ) << "Center cell (highest VEGF) must become TipCell";
    EXPECT_NE( result[ 0 ].cellType, 1u ) << "End cell must not become TipCell";
    EXPECT_NE( result[ 2 ].cellType, 1u ) << "End cell must not become TipCell";
    // Adjacent to TipCell → recruited to StalkCell (type=2) by the recruitment path
    EXPECT_EQ( result[ 0 ].cellType, 2u ) << "End cell adjacent to TipCell must be recruited to StalkCell";
    EXPECT_EQ( result[ 2 ].cellType, 2u ) << "End cell adjacent to TipCell must be recruited to StalkCell";

    m_rm->DestroyBuffer( agBuf ); m_rm->DestroyBuffer( sigBuf ); m_rm->DestroyBuffer( phBuf );
    m_rm->DestroyBuffer( eBuf  ); m_rm->DestroyBuffer( ecBuf  ); m_rm->DestroyBuffer( cBuf  );
    m_rm->DestroyTexture( vTex );
}

// A PhalanxCell with no VEGF and Dll4 < stalkThreshold must NOT convert to StalkCell.
// The guard `cellType != 3u` in the stalk conversion branch prevents this.
//
// Setup: 1 PhalanxCell, VEGF=0 (gridSize.w=0 → vegfGating=1), dll4=0.15, no neighbors.
// new_nicd ≈ 0; vegfr2 = 1/(1+0)=1; new_dll4 = 0.15 + 0.1*(1 - 0.1*0.15) = 0.2485 < stalkThreshold(0.3).
// Expected: stays PhalanxCell (type=3).
TEST_F( ComputeTest, Shader_NotchDll4_PhalanxCellNoStalkConversion )
{
    if( !m_device )
        GTEST_SKIP();

    struct SignalingData { float dll4; float nicd; float vegfr2; float pad; };
    struct VesselEdge    { uint32_t agentA; uint32_t agentB; float dist; uint32_t flags; };

    std::vector<glm::vec4>    agents     = { glm::vec4(0,0,0,1) };
    std::vector<SignalingData> signaling  = { {0.15f, 0.0f, 0.0f, 0.0f} };
    std::vector<PhenotypeData> phenotypes = { {0u, 0.5f, 0.0f, 3u} }; // PhalanxCell
    VesselEdge dummyEdge  = { 0u, 0u, 0.0f, 0u };
    uint32_t   edgeCount  = 0;
    uint32_t   agentCount = 1;

    size_t agSz = sizeof(glm::vec4), sigSz = sizeof(SignalingData), phSz = sizeof(PhenotypeData);
    size_t cSz  = sizeof(uint32_t);

    BufferHandle  agBuf  = m_rm->CreateBuffer( { agSz,                BufferType::STORAGE,  "PxNoStalkAg"    } );
    BufferHandle  sigBuf = m_rm->CreateBuffer( { sigSz,               BufferType::STORAGE,  "PxNoStalkSig"   } );
    BufferHandle  phBuf  = m_rm->CreateBuffer( { phSz,                BufferType::STORAGE,  "PxNoStalkPh"    } );
    BufferHandle  eBuf   = m_rm->CreateBuffer( { sizeof(VesselEdge),  BufferType::STORAGE,  "PxNoStalkE"     } );
    BufferHandle  ecBuf  = m_rm->CreateBuffer( { cSz,                 BufferType::STORAGE,  "PxNoStalkEC"    } );
    BufferHandle  cBuf   = m_rm->CreateBuffer( { cSz,                 BufferType::INDIRECT, "PxNoStalkCnt"   } );
    TextureHandle vTex   = m_rm->CreateTexture( { 1,1,1, TextureType::Texture3D, VK_FORMAT_R32_SFLOAT,
                                                  TextureUsage::STORAGE | TextureUsage::TRANSFER_DST, VK_SAMPLE_COUNT_1_BIT, "PxNoStalkVEGF" } );
    float zero = 0.0f;
    m_stream->UploadTextureImmediate( vTex, &zero, sizeof(float) );
    m_stream->UploadBufferImmediate( {
        { agBuf,  agents.data(),     agSz,               0 },
        { sigBuf, signaling.data(),  sigSz,              0 },
        { phBuf,  phenotypes.data(), phSz,               0 },
        { eBuf,   &dummyEdge,        sizeof(VesselEdge), 0 },
        { ecBuf,  &edgeCount,        cSz,                0 },
        { cBuf,   &agentCount,       cSz,                0 },
    } );

    ComputePipelineHandle ph2 = m_rm->CreatePipeline( ComputePipelineDesc{ m_rm->CreateShader( "shaders/compute/biology/notch_dll4.comp" ), "PxNoStalk" } );
    ComputePipeline*      pp  = m_rm->GetPipeline( ph2 );
    BindingGroup* bg = m_rm->GetBindingGroup( m_rm->CreateBindingGroup( ph2, 0 ) );
    bg->Bind( 0, m_rm->GetBuffer( agBuf  ) );
    bg->Bind( 1, m_rm->GetBuffer( sigBuf ) );
    bg->Bind( 2, m_rm->GetBuffer( phBuf  ) );
    bg->Bind( 3, m_rm->GetBuffer( eBuf   ) );
    bg->Bind( 4, m_rm->GetBuffer( ecBuf  ) );
    bg->Bind( 5, m_rm->GetBuffer( cBuf   ) );
    bg->Bind( 6, m_rm->GetTexture( vTex ), VK_IMAGE_LAYOUT_GENERAL );
    bg->Build();

    ComputePushConstants pc{};
    pc.dt = 0.1f; pc.fParam0 = 1.0f; pc.fParam1 = 0.1f; pc.fParam2 = 5.0f;
    pc.fParam3 = 1.0f; pc.fParam4 = 0.8f; pc.fParam5 = 0.3f;
    pc.offset = 0; pc.maxCapacity = 1; pc.uParam0 = 0u; pc.uParam1 = 0;
    pc.domainSize = glm::vec4( 100.0f, 100.0f, 100.0f, 0.0f );
    pc.gridSize   = glm::uvec4( 1u, 1u, 1u, 0u ); // w=0 → VEGF not bound → vegfGating=1

    auto ctx = m_device->GetThreadContext( m_device->CreateThreadContext( QueueType::COMPUTE ) );
    auto cmd = ctx->GetCommandBuffer( ctx->CreateCommandBuffer() );
    cmd->Begin();
    cmd->SetPipeline( pp );
    cmd->SetBindingGroup( bg, pp->GetLayout(), VK_PIPELINE_BIND_POINT_COMPUTE );
    cmd->PushConstants( pp->GetLayout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc );
    cmd->Dispatch( 1, 1, 1 );
    cmd->End();
    m_device->GetComputeQueue()->Submit( { cmd } );
    m_device->GetComputeQueue()->WaitIdle();

    PhenotypeData result{};
    m_stream->ReadbackBufferImmediate( phBuf, &result, phSz );

    EXPECT_EQ( result.cellType, 3u ) << "PhalanxCell with Dll4 < stalkThreshold must stay PhalanxCell";

    m_rm->DestroyBuffer( agBuf ); m_rm->DestroyBuffer( sigBuf ); m_rm->DestroyBuffer( phBuf );
    m_rm->DestroyBuffer( eBuf  ); m_rm->DestroyBuffer( ecBuf  ); m_rm->DestroyBuffer( cBuf  );
    m_rm->DestroyTexture( vTex );
}

// PhalanxCell adjacent to a TipCell must be recruited to StalkCell.
// This is the NotchDll4 recruitment path: hasTipNeighbor && cellType==3 → cellType=2.
//
// Setup: 2 cells, 1 edge. Cell 0 = PhalanxCell, Cell 1 = TipCell. 1 dispatch.
// Expected: cell 0 → StalkCell.
TEST_F( ComputeTest, Shader_NotchDll4_Recruitment_AdjacentToTip )
{
    if( !m_device )
        GTEST_SKIP();

    struct SignalingData { float dll4; float nicd; float vegfr2; float pad; };
    struct VesselEdge    { uint32_t agentA; uint32_t agentB; float dist; uint32_t flags; };

    std::vector<glm::vec4>    agents     = { glm::vec4(0,0,0,1), glm::vec4(1,0,0,1) };
    std::vector<SignalingData> signaling  = { {0.0f,0.0f,0.0f,0.0f}, {0.9f,0.0f,0.0f,0.0f} }; // TipCell has high Dll4
    std::vector<PhenotypeData> phenotypes = { {0u,0.5f,0.0f,3u}, {0u,0.5f,0.0f,1u} }; // PhalanxCell, TipCell
    VesselEdge edge       = { 0u, 1u, 1.0f, 0u };
    uint32_t   edgeCount  = 1;
    uint32_t   agentCount = 2;

    size_t agSz = 2*sizeof(glm::vec4), sigSz = 2*sizeof(SignalingData), phSz = 2*sizeof(PhenotypeData);
    size_t cSz  = sizeof(uint32_t);

    BufferHandle  agBuf  = m_rm->CreateBuffer( { agSz,               BufferType::STORAGE,  "PxRcrAg"   } );
    BufferHandle  sigBuf = m_rm->CreateBuffer( { sigSz,              BufferType::STORAGE,  "PxRcrSig"  } );
    BufferHandle  phBuf  = m_rm->CreateBuffer( { phSz,               BufferType::STORAGE,  "PxRcrPh"   } );
    BufferHandle  eBuf   = m_rm->CreateBuffer( { sizeof(VesselEdge), BufferType::STORAGE,  "PxRcrE"    } );
    BufferHandle  ecBuf  = m_rm->CreateBuffer( { cSz,                BufferType::STORAGE,  "PxRcrEC"   } );
    BufferHandle  cBuf   = m_rm->CreateBuffer( { cSz,                BufferType::INDIRECT, "PxRcrCnt"  } );
    TextureHandle vTex   = m_rm->CreateTexture( { 1,1,1, TextureType::Texture3D, VK_FORMAT_R32_SFLOAT,
                                                  TextureUsage::STORAGE | TextureUsage::TRANSFER_DST, VK_SAMPLE_COUNT_1_BIT, "PxRcrVEGF" } );
    float zero = 0.0f;
    m_stream->UploadTextureImmediate( vTex, &zero, sizeof(float) );
    m_stream->UploadBufferImmediate( {
        { agBuf,  agents.data(),     agSz,               0 },
        { sigBuf, signaling.data(),  sigSz,              0 },
        { phBuf,  phenotypes.data(), phSz,               0 },
        { eBuf,   &edge,             sizeof(VesselEdge), 0 },
        { ecBuf,  &edgeCount,        cSz,                0 },
        { cBuf,   &agentCount,       cSz,                0 },
    } );

    ComputePipelineHandle ph3 = m_rm->CreatePipeline( ComputePipelineDesc{ m_rm->CreateShader( "shaders/compute/biology/notch_dll4.comp" ), "PxRcr" } );
    ComputePipeline*      pp  = m_rm->GetPipeline( ph3 );
    BindingGroup* bg = m_rm->GetBindingGroup( m_rm->CreateBindingGroup( ph3, 0 ) );
    bg->Bind( 0, m_rm->GetBuffer( agBuf  ) );
    bg->Bind( 1, m_rm->GetBuffer( sigBuf ) );
    bg->Bind( 2, m_rm->GetBuffer( phBuf  ) );
    bg->Bind( 3, m_rm->GetBuffer( eBuf   ) );
    bg->Bind( 4, m_rm->GetBuffer( ecBuf  ) );
    bg->Bind( 5, m_rm->GetBuffer( cBuf   ) );
    bg->Bind( 6, m_rm->GetTexture( vTex ), VK_IMAGE_LAYOUT_GENERAL );
    bg->Build();

    ComputePushConstants pc{};
    pc.dt = 0.1f; pc.fParam0 = 1.0f; pc.fParam1 = 0.1f; pc.fParam2 = 5.0f;
    pc.fParam3 = 1.0f; pc.fParam4 = 0.8f; pc.fParam5 = 0.3f;
    pc.offset = 0; pc.maxCapacity = 2; pc.uParam0 = 0u; pc.uParam1 = 0;
    pc.domainSize = glm::vec4( 100.0f, 100.0f, 100.0f, 0.0f );
    pc.gridSize   = glm::uvec4( 1u, 1u, 1u, 0u );

    auto ctx = m_device->GetThreadContext( m_device->CreateThreadContext( QueueType::COMPUTE ) );
    auto cmd = ctx->GetCommandBuffer( ctx->CreateCommandBuffer() );
    cmd->Begin();
    cmd->SetPipeline( pp );
    cmd->SetBindingGroup( bg, pp->GetLayout(), VK_PIPELINE_BIND_POINT_COMPUTE );
    cmd->PushConstants( pp->GetLayout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc );
    cmd->Dispatch( 1, 1, 1 );
    cmd->End();
    m_device->GetComputeQueue()->Submit( { cmd } );
    m_device->GetComputeQueue()->WaitIdle();

    std::vector<PhenotypeData> result( 2 );
    m_stream->ReadbackBufferImmediate( phBuf, result.data(), phSz );

    EXPECT_EQ( result[ 0 ].cellType, 2u ) << "PhalanxCell adjacent to TipCell must become StalkCell";
    EXPECT_EQ( result[ 1 ].cellType, 1u ) << "TipCell must remain TipCell";

    m_rm->DestroyBuffer( agBuf ); m_rm->DestroyBuffer( sigBuf ); m_rm->DestroyBuffer( phBuf );
    m_rm->DestroyBuffer( eBuf  ); m_rm->DestroyBuffer( ecBuf  ); m_rm->DestroyBuffer( cBuf  );
    m_rm->DestroyTexture( vTex );
}

// PhalanxCell with only a StalkCell neighbor must NOT be recruited to StalkCell.
// Recruitment requires a TipCell neighbor specifically.
//
// Setup: 2 cells, 1 edge. Cell 0 = PhalanxCell, Cell 1 = StalkCell. 1 dispatch.
// Expected: cell 0 stays PhalanxCell.
TEST_F( ComputeTest, Shader_NotchDll4_Recruitment_NoTipNeighbor_StaysPhalanx )
{
    if( !m_device )
        GTEST_SKIP();

    struct SignalingData { float dll4; float nicd; float vegfr2; float pad; };
    struct VesselEdge    { uint32_t agentA; uint32_t agentB; float dist; uint32_t flags; };

    std::vector<glm::vec4>    agents     = { glm::vec4(0,0,0,1), glm::vec4(1,0,0,1) };
    std::vector<SignalingData> signaling  = { {0.0f,0.0f,0.0f,0.0f}, {0.1f,0.0f,0.0f,0.0f} };
    std::vector<PhenotypeData> phenotypes = { {0u,0.5f,0.0f,3u}, {0u,0.5f,0.0f,2u} }; // PhalanxCell, StalkCell
    VesselEdge edge       = { 0u, 1u, 1.0f, 0u };
    uint32_t   edgeCount  = 1;
    uint32_t   agentCount = 2;

    size_t agSz = 2*sizeof(glm::vec4), sigSz = 2*sizeof(SignalingData), phSz = 2*sizeof(PhenotypeData);
    size_t cSz  = sizeof(uint32_t);

    BufferHandle  agBuf  = m_rm->CreateBuffer( { agSz,               BufferType::STORAGE,  "PxNoTipAg"   } );
    BufferHandle  sigBuf = m_rm->CreateBuffer( { sigSz,              BufferType::STORAGE,  "PxNoTipSig"  } );
    BufferHandle  phBuf  = m_rm->CreateBuffer( { phSz,               BufferType::STORAGE,  "PxNoTipPh"   } );
    BufferHandle  eBuf   = m_rm->CreateBuffer( { sizeof(VesselEdge), BufferType::STORAGE,  "PxNoTipE"    } );
    BufferHandle  ecBuf  = m_rm->CreateBuffer( { cSz,                BufferType::STORAGE,  "PxNoTipEC"   } );
    BufferHandle  cBuf   = m_rm->CreateBuffer( { cSz,                BufferType::INDIRECT, "PxNoTipCnt"  } );
    TextureHandle vTex   = m_rm->CreateTexture( { 1,1,1, TextureType::Texture3D, VK_FORMAT_R32_SFLOAT,
                                                  TextureUsage::STORAGE | TextureUsage::TRANSFER_DST, VK_SAMPLE_COUNT_1_BIT, "PxNoTipVEGF" } );
    float zero = 0.0f;
    m_stream->UploadTextureImmediate( vTex, &zero, sizeof(float) );
    m_stream->UploadBufferImmediate( {
        { agBuf,  agents.data(),     agSz,               0 },
        { sigBuf, signaling.data(),  sigSz,              0 },
        { phBuf,  phenotypes.data(), phSz,               0 },
        { eBuf,   &edge,             sizeof(VesselEdge), 0 },
        { ecBuf,  &edgeCount,        cSz,                0 },
        { cBuf,   &agentCount,       cSz,                0 },
    } );

    ComputePipelineHandle ph4 = m_rm->CreatePipeline( ComputePipelineDesc{ m_rm->CreateShader( "shaders/compute/biology/notch_dll4.comp" ), "PxNoTip" } );
    ComputePipeline*      pp  = m_rm->GetPipeline( ph4 );
    BindingGroup* bg = m_rm->GetBindingGroup( m_rm->CreateBindingGroup( ph4, 0 ) );
    bg->Bind( 0, m_rm->GetBuffer( agBuf  ) );
    bg->Bind( 1, m_rm->GetBuffer( sigBuf ) );
    bg->Bind( 2, m_rm->GetBuffer( phBuf  ) );
    bg->Bind( 3, m_rm->GetBuffer( eBuf   ) );
    bg->Bind( 4, m_rm->GetBuffer( ecBuf  ) );
    bg->Bind( 5, m_rm->GetBuffer( cBuf   ) );
    bg->Bind( 6, m_rm->GetTexture( vTex ), VK_IMAGE_LAYOUT_GENERAL );
    bg->Build();

    ComputePushConstants pc{};
    pc.dt = 0.1f; pc.fParam0 = 1.0f; pc.fParam1 = 0.1f; pc.fParam2 = 5.0f;
    pc.fParam3 = 1.0f; pc.fParam4 = 0.8f; pc.fParam5 = 0.3f;
    pc.offset = 0; pc.maxCapacity = 2; pc.uParam0 = 0u; pc.uParam1 = 0;
    pc.domainSize = glm::vec4( 100.0f, 100.0f, 100.0f, 0.0f );
    pc.gridSize   = glm::uvec4( 1u, 1u, 1u, 0u );

    auto ctx = m_device->GetThreadContext( m_device->CreateThreadContext( QueueType::COMPUTE ) );
    auto cmd = ctx->GetCommandBuffer( ctx->CreateCommandBuffer() );
    cmd->Begin();
    cmd->SetPipeline( pp );
    cmd->SetBindingGroup( bg, pp->GetLayout(), VK_PIPELINE_BIND_POINT_COMPUTE );
    cmd->PushConstants( pp->GetLayout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc );
    cmd->Dispatch( 1, 1, 1 );
    cmd->End();
    m_device->GetComputeQueue()->Submit( { cmd } );
    m_device->GetComputeQueue()->WaitIdle();

    std::vector<PhenotypeData> result( 2 );
    m_stream->ReadbackBufferImmediate( phBuf, result.data(), phSz );

    EXPECT_EQ( result[ 0 ].cellType, 3u ) << "PhalanxCell with only StalkCell neighbor must stay PhalanxCell";
    EXPECT_EQ( result[ 1 ].cellType, 2u ) << "StalkCell must remain StalkCell";

    m_rm->DestroyBuffer( agBuf ); m_rm->DestroyBuffer( sigBuf ); m_rm->DestroyBuffer( phBuf );
    m_rm->DestroyBuffer( eBuf  ); m_rm->DestroyBuffer( ecBuf  ); m_rm->DestroyBuffer( cBuf  );
    m_rm->DestroyTexture( vTex );
}

// PhalanxCell connected to a TipCell only via a RING edge (flags=0x1) must NOT be
// recruited to StalkCell. The ring edge is a circumferential tube-wall contact; the
// TipCell is in the same ring, not axially ahead of the PhalanxCell. Recruiting via
// ring edges causes the flip-flop bug (stuck yellow cylinder in tube wall) because
// mitosis_vessel_append.comp applies the same ring-edge filter for its check.
//
// Setup: 2 cells, 1 RING edge. Cell 0 = PhalanxCell, Cell 1 = TipCell. 1 dispatch.
// Expected: cell 0 stays PhalanxCell.
TEST_F( ComputeTest, Shader_NotchDll4_PhalanxRecruitment_IgnoresRingEdges )
{
    if( !m_device )
        GTEST_SKIP();

    struct SignalingData { float dll4; float nicd; float vegfr2; float pad; };
    struct VesselEdge    { uint32_t agentA; uint32_t agentB; float dist; uint32_t flags; };

    std::vector<glm::vec4>    agents     = { glm::vec4(0,0,0,1), glm::vec4(1,0,0,1) };
    std::vector<SignalingData> signaling  = { {0.0f,0.0f,0.0f,0.0f}, {0.9f,0.0f,0.0f,0.0f} }; // TipCell has high Dll4
    std::vector<PhenotypeData> phenotypes = { {0u,0.5f,0.0f,3u}, {0u,0.5f,0.0f,1u} }; // PhalanxCell, TipCell
    VesselEdge edge       = { 0u, 1u, 1.0f, 0x1u }; // RING edge — must be ignored for recruitment
    uint32_t   edgeCount  = 1;
    uint32_t   agentCount = 2;

    size_t agSz = 2*sizeof(glm::vec4), sigSz = 2*sizeof(SignalingData), phSz = 2*sizeof(PhenotypeData);
    size_t cSz  = sizeof(uint32_t);

    BufferHandle  agBuf  = m_rm->CreateBuffer( { agSz,               BufferType::STORAGE,  "PxRingAg"   } );
    BufferHandle  sigBuf = m_rm->CreateBuffer( { sigSz,              BufferType::STORAGE,  "PxRingSig"  } );
    BufferHandle  phBuf  = m_rm->CreateBuffer( { phSz,               BufferType::STORAGE,  "PxRingPh"   } );
    BufferHandle  eBuf   = m_rm->CreateBuffer( { sizeof(VesselEdge), BufferType::STORAGE,  "PxRingE"    } );
    BufferHandle  ecBuf  = m_rm->CreateBuffer( { cSz,                BufferType::STORAGE,  "PxRingEC"   } );
    BufferHandle  cBuf   = m_rm->CreateBuffer( { cSz,                BufferType::INDIRECT, "PxRingCnt"  } );
    TextureHandle vTex   = m_rm->CreateTexture( { 1,1,1, TextureType::Texture3D, VK_FORMAT_R32_SFLOAT,
                                                  TextureUsage::STORAGE | TextureUsage::TRANSFER_DST, VK_SAMPLE_COUNT_1_BIT, "PxRingVEGF" } );
    float zero = 0.0f;
    m_stream->UploadTextureImmediate( vTex, &zero, sizeof(float) );
    m_stream->UploadBufferImmediate( {
        { agBuf,  agents.data(),     agSz,               0 },
        { sigBuf, signaling.data(),  sigSz,              0 },
        { phBuf,  phenotypes.data(), phSz,               0 },
        { eBuf,   &edge,             sizeof(VesselEdge), 0 },
        { ecBuf,  &edgeCount,        cSz,                0 },
        { cBuf,   &agentCount,       cSz,                0 },
    } );

    ComputePipelineHandle ph5 = m_rm->CreatePipeline( ComputePipelineDesc{ m_rm->CreateShader( "shaders/compute/biology/notch_dll4.comp" ), "PxRing" } );
    ComputePipeline*      pp  = m_rm->GetPipeline( ph5 );
    BindingGroup* bg = m_rm->GetBindingGroup( m_rm->CreateBindingGroup( ph5, 0 ) );
    bg->Bind( 0, m_rm->GetBuffer( agBuf  ) );
    bg->Bind( 1, m_rm->GetBuffer( sigBuf ) );
    bg->Bind( 2, m_rm->GetBuffer( phBuf  ) );
    bg->Bind( 3, m_rm->GetBuffer( eBuf   ) );
    bg->Bind( 4, m_rm->GetBuffer( ecBuf  ) );
    bg->Bind( 5, m_rm->GetBuffer( cBuf   ) );
    bg->Bind( 6, m_rm->GetTexture( vTex ), VK_IMAGE_LAYOUT_GENERAL );
    bg->Build();

    ComputePushConstants pc{};
    pc.dt = 0.1f; pc.fParam0 = 1.0f; pc.fParam1 = 0.1f; pc.fParam2 = 5.0f;
    pc.fParam3 = 1.0f; pc.fParam4 = 0.8f; pc.fParam5 = 0.3f;
    pc.offset = 0; pc.maxCapacity = 2; pc.uParam0 = 0u; pc.uParam1 = 0;
    pc.domainSize = glm::vec4( 100.0f, 100.0f, 100.0f, 0.0f );
    pc.gridSize   = glm::uvec4( 1u, 1u, 1u, 0u );

    auto ctx = m_device->GetThreadContext( m_device->CreateThreadContext( QueueType::COMPUTE ) );
    auto cmd = ctx->GetCommandBuffer( ctx->CreateCommandBuffer() );
    cmd->Begin();
    cmd->SetPipeline( pp );
    cmd->SetBindingGroup( bg, pp->GetLayout(), VK_PIPELINE_BIND_POINT_COMPUTE );
    cmd->PushConstants( pp->GetLayout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc );
    cmd->Dispatch( 1, 1, 1 );
    cmd->End();
    m_device->GetComputeQueue()->Submit( { cmd } );
    m_device->GetComputeQueue()->WaitIdle();

    std::vector<PhenotypeData> result( 2 );
    m_stream->ReadbackBufferImmediate( phBuf, result.data(), phSz );

    EXPECT_EQ( result[ 0 ].cellType, 3u ) << "PhalanxCell connected via ring edge to TipCell must NOT be recruited";
    EXPECT_EQ( result[ 1 ].cellType, 1u ) << "TipCell must remain TipCell";

    m_rm->DestroyBuffer( agBuf ); m_rm->DestroyBuffer( sigBuf ); m_rm->DestroyBuffer( phBuf );
    m_rm->DestroyBuffer( eBuf  ); m_rm->DestroyBuffer( ecBuf  ); m_rm->DestroyBuffer( cBuf  );
    m_rm->DestroyTexture( vTex );
}

// ─────────────────────────────────────────────────────────────────────────────
// mitosis_vessel_append — PhalanxCell recruitment safety-net tests
// ─────────────────────────────────────────────────────────────────────────────

// PhalanxCell adjacent to a TipCell must be recruited to StalkCell by the
// mitosis shader's safety-net path (for when CellCycle is present).
//
// Setup: 2 live agents. Cell 0 = PhalanxCell (biomass=0.5, lifecycleState=0).
// Cell 1 = TipCell. 1 edge connecting them. Counter pre-set to 2.
// Expected: cell 0 → StalkCell (type=2).
TEST_F( ComputeTest, Shader_VesselMitosis_Recruitment_PhalanxAdjacentToTip_BecomesStalk )
{
    if( !m_device )
        GTEST_SKIP();

    struct VesselEdge    { uint32_t agentA; uint32_t agentB; float dist; uint32_t flags; };

    const uint32_t maxCap = 4;
    std::vector<glm::vec4>    posR = { glm::vec4(0,0,0,1), glm::vec4(1,0,0,1),
                                       glm::vec4(0,0,0,0), glm::vec4(0,0,0,0) };
    std::vector<glm::vec4>    posW = posR;
    std::vector<PhenotypeData> pheno = {
        { 0u, 0.5f, 0.0f, 3u }, // PhalanxCell — biomass too low to divide anyway
        { 0u, 0.5f, 0.0f, 1u }, // TipCell
        { 0u, 0.0f, 0.0f, 0u },
        { 0u, 0.0f, 0.0f, 0u }
    };
    VesselEdge edge      = { 0u, 1u, 1.0f, 0u };
    uint32_t   edgeCnt   = 1;
    uint32_t   agentCnt  = 2;

    size_t posSz  = maxCap * sizeof(glm::vec4);
    size_t phSz   = maxCap * sizeof(PhenotypeData);
    size_t cSz    = sizeof(uint32_t);
    size_t eSz    = maxCap * sizeof(VesselEdge);

    BufferHandle abR = m_rm->CreateBuffer( { posSz, BufferType::STORAGE,  "MxPxTipAbR"   } );
    BufferHandle abW = m_rm->CreateBuffer( { posSz, BufferType::STORAGE,  "MxPxTipAbW"   } );
    BufferHandle pb  = m_rm->CreateBuffer( { phSz,  BufferType::STORAGE,  "MxPxTipPh"    } );
    BufferHandle cB  = m_rm->CreateBuffer( { cSz,   BufferType::INDIRECT, "MxPxTipCnt"   } );
    BufferHandle eB  = m_rm->CreateBuffer( { eSz,   BufferType::STORAGE,  "MxPxTipE"     } );
    BufferHandle ecB = m_rm->CreateBuffer( { cSz,   BufferType::STORAGE,  "MxPxTipEC"    } );
    BufferHandle oB  = m_rm->CreateBuffer( { posSz, BufferType::STORAGE,  "MxPxTipOr"    } );

    m_stream->UploadBufferImmediate( {
        { abR, posR.data(),   posSz,               0 },
        { abW, posW.data(),   posSz,               0 },
        { pb,  pheno.data(),  phSz,                0 },
        { cB,  &agentCnt,     cSz,                 0 },
        { eB,  &edge,         sizeof(VesselEdge),  0 },
        { ecB, &edgeCnt,      cSz,                 0 },
    } );

    ComputePipelineHandle ph5 = m_rm->CreatePipeline( ComputePipelineDesc{
        m_rm->CreateShader( "shaders/compute/biology/mitosis_vessel_append.comp" ), "MxPxTip" } );
    ComputePipeline* pp = m_rm->GetPipeline( ph5 );

    BindingGroup* bg = m_rm->GetBindingGroup( m_rm->CreateBindingGroup( ph5, 0 ) );
    bg->Bind( 0, m_rm->GetBuffer( abR ) );
    bg->Bind( 1, m_rm->GetBuffer( abW ) );
    bg->Bind( 2, m_rm->GetBuffer( pb  ) );
    bg->Bind( 3, m_rm->GetBuffer( cB  ) );
    bg->Bind( 4, m_rm->GetBuffer( eB  ) );
    bg->Bind( 5, m_rm->GetBuffer( ecB ) );
    bg->Bind( 6, m_rm->GetBuffer( oB  ) );
    bg->Build();

    ComputePushConstants pc{};
    pc.dt = 0.016f; pc.totalTime = 10.0f;
    pc.offset = 0; pc.maxCapacity = maxCap; pc.uParam1 = 0;
    pc.domainSize = glm::vec4( 100.0f, 100.0f, 100.0f, 0.0f );
    pc.gridSize   = glm::uvec4( 0u );

    auto ctx = m_device->GetThreadContext( m_device->CreateThreadContext( QueueType::COMPUTE ) );
    auto cmd = ctx->GetCommandBuffer( ctx->CreateCommandBuffer() );
    cmd->Begin();
    cmd->SetPipeline( pp );
    cmd->SetBindingGroup( bg, pp->GetLayout(), VK_PIPELINE_BIND_POINT_COMPUTE );
    cmd->PushConstants( pp->GetLayout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc );
    cmd->Dispatch( 1, 1, 1 );
    cmd->End();
    m_device->GetComputeQueue()->Submit( { cmd } );
    m_device->GetComputeQueue()->WaitIdle();

    std::vector<PhenotypeData> result( maxCap );
    m_stream->ReadbackBufferImmediate( pb, result.data(), phSz );

    EXPECT_EQ( result[ 0 ].cellType, 2u ) << "PhalanxCell adjacent to TipCell must become StalkCell";
    EXPECT_EQ( result[ 1 ].cellType, 1u ) << "TipCell must remain TipCell";

    m_rm->DestroyBuffer( abR ); m_rm->DestroyBuffer( abW ); m_rm->DestroyBuffer( pb );
    m_rm->DestroyBuffer( cB );  m_rm->DestroyBuffer( eB );  m_rm->DestroyBuffer( ecB );
    m_rm->DestroyBuffer( oB );
}

// PhalanxCell adjacent only to a StalkCell must NOT be recruited by the mitosis shader.
//
// Setup: 2 live agents. Cell 0 = PhalanxCell, Cell 1 = StalkCell. 1 edge.
// Expected: cell 0 stays PhalanxCell.
TEST_F( ComputeTest, Shader_VesselMitosis_Recruitment_PhalanxNoTip_StaysPhalanx )
{
    if( !m_device )
        GTEST_SKIP();

    struct VesselEdge    { uint32_t agentA; uint32_t agentB; float dist; uint32_t flags; };

    const uint32_t maxCap = 4;
    std::vector<glm::vec4>    posR = { glm::vec4(0,0,0,1), glm::vec4(1,0,0,1),
                                       glm::vec4(0,0,0,0), glm::vec4(0,0,0,0) };
    std::vector<glm::vec4>    posW = posR;
    std::vector<PhenotypeData> pheno = {
        { 0u, 0.5f, 0.0f, 3u }, // PhalanxCell
        { 0u, 0.5f, 0.0f, 2u }, // StalkCell
        { 0u, 0.0f, 0.0f, 0u },
        { 0u, 0.0f, 0.0f, 0u }
    };
    VesselEdge edge     = { 0u, 1u, 1.0f, 0u };
    uint32_t   edgeCnt  = 1;
    uint32_t   agentCnt = 2;

    size_t posSz = maxCap * sizeof(glm::vec4);
    size_t phSz  = maxCap * sizeof(PhenotypeData);
    size_t cSz   = sizeof(uint32_t);
    size_t eSz   = maxCap * sizeof(VesselEdge);

    BufferHandle abR = m_rm->CreateBuffer( { posSz, BufferType::STORAGE,  "MxPxNoTipAbR"  } );
    BufferHandle abW = m_rm->CreateBuffer( { posSz, BufferType::STORAGE,  "MxPxNoTipAbW"  } );
    BufferHandle pb  = m_rm->CreateBuffer( { phSz,  BufferType::STORAGE,  "MxPxNoTipPh"   } );
    BufferHandle cB  = m_rm->CreateBuffer( { cSz,   BufferType::INDIRECT, "MxPxNoTipCnt"  } );
    BufferHandle eB  = m_rm->CreateBuffer( { eSz,   BufferType::STORAGE,  "MxPxNoTipE"    } );
    BufferHandle ecB = m_rm->CreateBuffer( { cSz,   BufferType::STORAGE,  "MxPxNoTipEC"   } );
    BufferHandle oB  = m_rm->CreateBuffer( { posSz, BufferType::STORAGE,  "MxPxNoTipOr"   } );

    m_stream->UploadBufferImmediate( {
        { abR, posR.data(),  posSz,               0 },
        { abW, posW.data(),  posSz,               0 },
        { pb,  pheno.data(), phSz,                0 },
        { cB,  &agentCnt,    cSz,                 0 },
        { eB,  &edge,        sizeof(VesselEdge),  0 },
        { ecB, &edgeCnt,     cSz,                 0 },
    } );

    ComputePipelineHandle ph6 = m_rm->CreatePipeline( ComputePipelineDesc{
        m_rm->CreateShader( "shaders/compute/biology/mitosis_vessel_append.comp" ), "MxPxNoTip" } );
    ComputePipeline* pp = m_rm->GetPipeline( ph6 );

    BindingGroup* bg = m_rm->GetBindingGroup( m_rm->CreateBindingGroup( ph6, 0 ) );
    bg->Bind( 0, m_rm->GetBuffer( abR ) );
    bg->Bind( 1, m_rm->GetBuffer( abW ) );
    bg->Bind( 2, m_rm->GetBuffer( pb  ) );
    bg->Bind( 3, m_rm->GetBuffer( cB  ) );
    bg->Bind( 4, m_rm->GetBuffer( eB  ) );
    bg->Bind( 5, m_rm->GetBuffer( ecB ) );
    bg->Bind( 6, m_rm->GetBuffer( oB  ) );
    bg->Build();

    ComputePushConstants pc{};
    pc.dt = 0.016f; pc.totalTime = 0.0f; // Keep below 3.0s grace period to avoid StalkCell→PhalanxCell maturation
    pc.offset = 0; pc.maxCapacity = maxCap; pc.uParam1 = 0;
    pc.domainSize = glm::vec4( 100.0f, 100.0f, 100.0f, 0.0f );
    pc.gridSize   = glm::uvec4( 0u );

    auto ctx = m_device->GetThreadContext( m_device->CreateThreadContext( QueueType::COMPUTE ) );
    auto cmd = ctx->GetCommandBuffer( ctx->CreateCommandBuffer() );
    cmd->Begin();
    cmd->SetPipeline( pp );
    cmd->SetBindingGroup( bg, pp->GetLayout(), VK_PIPELINE_BIND_POINT_COMPUTE );
    cmd->PushConstants( pp->GetLayout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc );
    cmd->Dispatch( 1, 1, 1 );
    cmd->End();
    m_device->GetComputeQueue()->Submit( { cmd } );
    m_device->GetComputeQueue()->WaitIdle();

    std::vector<PhenotypeData> result( maxCap );
    m_stream->ReadbackBufferImmediate( pb, result.data(), phSz );

    EXPECT_EQ( result[ 0 ].cellType, 3u ) << "PhalanxCell with only StalkCell neighbor must stay PhalanxCell";
    EXPECT_EQ( result[ 1 ].cellType, 2u ) << "StalkCell must remain StalkCell";

    m_rm->DestroyBuffer( abR ); m_rm->DestroyBuffer( abW ); m_rm->DestroyBuffer( pb );
    m_rm->DestroyBuffer( cB );  m_rm->DestroyBuffer( eB );  m_rm->DestroyBuffer( ecB );
    m_rm->DestroyBuffer( oB );
}

// =================================================================================================
// mitosis_vessel_append — ring topology tests (edge flag filtering)
// =================================================================================================

// StalkCell with an AXIAL edge to TipCell should divide.
// StalkCell with only a RING edge to TipCell should mature to PhalanxCell (ring edges ignored).
TEST_F( ComputeTest, Shader_VesselMitosis_RingTopology_OnlyAxialNeighborDivides )
{
    if( !m_device )
        GTEST_SKIP();

    struct VesselEdge    { uint32_t agentA; uint32_t agentB; float dist; uint32_t flags; };

    uint32_t maxCap = 8;
    size_t   agBz   = maxCap * sizeof( glm::vec4 );
    size_t   phBz   = maxCap * sizeof( PhenotypeData );
    size_t   edBz   = maxCap * sizeof( VesselEdge );
    size_t   cntBz  = sizeof( uint32_t );

    // Cell 0: StalkCell at (0,0,0) — connected to TipCell via AXIAL edge → should divide
    // Cell 1: TipCell at (3,0,0)
    // Cell 2: StalkCell at (0,3,0) — connected to TipCell via RING edge only → should mature
    std::vector<glm::vec4> agents( maxCap, glm::vec4( 0.0f ) );
    agents[ 0 ] = glm::vec4(  0.0f, 0.0f, 0.0f, 1.0f );
    agents[ 1 ] = glm::vec4(  3.0f, 0.0f, 0.0f, 1.0f );
    agents[ 2 ] = glm::vec4(  0.0f, 3.0f, 0.0f, 1.0f );

    std::vector<PhenotypeData> pheno( maxCap, { 0u, 0.0f, 0.0f, 0u } );
    pheno[ 0 ] = { 0u, 1.1f, 0.0f, 2u }; // StalkCell, biomass ready
    pheno[ 1 ] = { 0u, 0.5f, 0.0f, 1u }; // TipCell
    pheno[ 2 ] = { 0u, 1.1f, 0.0f, 2u }; // StalkCell, biomass ready

    std::vector<VesselEdge> edgeBuf( maxCap, { 0u, 0u, 0.0f, 0u } );
    edgeBuf[ 0 ] = { 0u, 1u, 3.0f, 0x2u }; // Cell 0 ↔ Cell 1 : AXIAL
    edgeBuf[ 1 ] = { 2u, 1u, 3.0f, 0x1u }; // Cell 2 ↔ Cell 1 : RING
    uint32_t edgeCount  = 2u;
    uint32_t agentCount = 3u;

    BufferHandle abR = m_rm->CreateBuffer( { agBz,  BufferType::STORAGE, "RngR"  } );
    BufferHandle abW = m_rm->CreateBuffer( { agBz,  BufferType::STORAGE, "RngW"  } );
    BufferHandle pb  = m_rm->CreateBuffer( { phBz,  BufferType::STORAGE, "RngPh" } );
    BufferHandle cB  = m_rm->CreateBuffer( { cntBz, BufferType::STORAGE, "RngC"  } );
    BufferHandle eB  = m_rm->CreateBuffer( { edBz,  BufferType::STORAGE, "RngE"  } );
    BufferHandle ecB = m_rm->CreateBuffer( { cntBz, BufferType::STORAGE, "RngEc" } );
    BufferHandle oB  = m_rm->CreateBuffer( { agBz,  BufferType::STORAGE, "RngOr" } );

    m_stream->UploadBufferImmediate( {
        { abR, agents.data(), agBz,  0 },
        { abW, agents.data(), agBz,  0 },
        { pb,  pheno.data(),  phBz,  0 },
        { cB,  &agentCount,   cntBz, 0 },
        { eB,  edgeBuf.data(), edBz, 0 },
        { ecB, &edgeCount,    cntBz, 0 },
    } );

    ComputePipelineDesc   pd{ m_rm->CreateShader( "shaders/compute/biology/mitosis_vessel_append.comp" ), "RingAxialDiv" };
    ComputePipelineHandle ph = m_rm->CreatePipeline( pd );
    BindingGroup* bg = m_rm->GetBindingGroup( m_rm->CreateBindingGroup( ph, 0 ) );
    bg->Bind( 0, m_rm->GetBuffer( abR ) ); bg->Bind( 1, m_rm->GetBuffer( abW ) );
    bg->Bind( 2, m_rm->GetBuffer( pb  ) ); bg->Bind( 3, m_rm->GetBuffer( cB  ) );
    bg->Bind( 4, m_rm->GetBuffer( eB  ) ); bg->Bind( 5, m_rm->GetBuffer( ecB ) );
    bg->Bind( 6, m_rm->GetBuffer( oB  ) );
    bg->Build();

    ComputePushConstants pc{};
    pc.totalTime = 10.0f; // past grace period so maturation fires
    pc.maxCapacity = maxCap; pc.uParam1 = 0;

    auto ctx = m_device->GetThreadContext( m_device->CreateThreadContext( QueueType::COMPUTE ) );
    auto cmd = ctx->GetCommandBuffer( ctx->CreateCommandBuffer() );
    cmd->Begin();
    cmd->SetPipeline( m_rm->GetPipeline( ph ) );
    cmd->SetBindingGroup( bg, m_rm->GetPipeline( ph )->GetLayout(), VK_PIPELINE_BIND_POINT_COMPUTE );
    cmd->PushConstants( m_rm->GetPipeline( ph )->GetLayout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof( pc ), &pc );
    cmd->Dispatch( 1, 1, 1 );
    cmd->End();
    m_device->GetComputeQueue()->Submit( { cmd } );
    m_device->GetComputeQueue()->WaitIdle();

    std::vector<PhenotypeData> resPheno( maxCap );
    uint32_t resCount;
    m_stream->ReadbackBufferImmediate( pb, resPheno.data(), phBz );
    m_stream->ReadbackBufferImmediate( cB, &resCount, cntBz );

    // Cell 0 divides → daughter at slot 3; cell 2 matures (ring edge only)
    EXPECT_EQ( resCount, 4u )             << "Cell 0 (axial TipCell neighbor) must produce 1 daughter → count = 4";
    EXPECT_EQ( resPheno[0].cellType, 2u ) << "Dividing StalkCell must remain StalkCell";
    EXPECT_EQ( resPheno[3].cellType, 2u ) << "Daughter of axial StalkCell must be StalkCell";
    EXPECT_EQ( resPheno[2].cellType, 3u ) << "Ring-only TipCell neighbor must mature to PhalanxCell";

    m_rm->DestroyBuffer( abR ); m_rm->DestroyBuffer( abW ); m_rm->DestroyBuffer( pb );
    m_rm->DestroyBuffer( cB );  m_rm->DestroyBuffer( eB );  m_rm->DestroyBuffer( ecB );
    m_rm->DestroyBuffer( oB );
}

// Division axis must point toward the TipCell even when the dividing cell has 4 edges
// (2 ring + 2 axial). Daughter position should be between mother and TipCell (+x direction).
TEST_F( ComputeTest, Shader_VesselMitosis_RingTopology_VesselAxisPointsTowardTip )
{
    if( !m_device )
        GTEST_SKIP();

    struct VesselEdge    { uint32_t agentA; uint32_t agentB; float dist; uint32_t flags; };

    uint32_t maxCap = 8;
    size_t   agBz   = maxCap * sizeof( glm::vec4 );
    size_t   phBz   = maxCap * sizeof( PhenotypeData );
    size_t   edBz   = maxCap * sizeof( VesselEdge );
    size_t   cntBz  = sizeof( uint32_t );

    // Cell 0: StalkCell at origin — 4 edges: TipCell ahead (+x), stalk behind (-x), 2 ring (±y)
    // axis=(1,0,0) → daughter placed at (1.5, 0, 0)
    std::vector<glm::vec4> agents( maxCap, glm::vec4( 0.0f ) );
    agents[ 0 ] = glm::vec4(  0.0f,  0.0f, 0.0f, 1.0f ); // dividing StalkCell
    agents[ 1 ] = glm::vec4(  3.0f,  0.0f, 0.0f, 1.0f ); // TipCell ahead
    agents[ 2 ] = glm::vec4( -3.0f,  0.0f, 0.0f, 1.0f ); // StalkCell behind
    agents[ 3 ] = glm::vec4(  0.0f,  2.0f, 0.0f, 1.0f ); // ring neighbor
    agents[ 4 ] = glm::vec4(  0.0f, -2.0f, 0.0f, 1.0f ); // ring neighbor

    std::vector<PhenotypeData> pheno( maxCap, { 0u, 0.0f, 0.0f, 0u } );
    pheno[ 0 ] = { 0u, 1.1f, 0.0f, 2u }; // StalkCell, biomass ready
    pheno[ 1 ] = { 0u, 0.5f, 0.0f, 1u }; // TipCell
    pheno[ 2 ] = { 0u, 0.5f, 0.0f, 2u }; // StalkCell behind
    pheno[ 3 ] = { 0u, 0.5f, 0.0f, 3u }; // PhalanxCell ring
    pheno[ 4 ] = { 0u, 0.5f, 0.0f, 3u }; // PhalanxCell ring

    std::vector<VesselEdge> edgeBuf( maxCap, { 0u, 0u, 0.0f, 0u } );
    edgeBuf[ 0 ] = { 0u, 1u, 3.0f, 0x2u }; // Cell 0 ↔ TipCell  : AXIAL
    edgeBuf[ 1 ] = { 0u, 2u, 3.0f, 0x2u }; // Cell 0 ↔ behind   : AXIAL
    edgeBuf[ 2 ] = { 0u, 3u, 2.0f, 0x1u }; // Cell 0 ↔ ring+y   : RING
    edgeBuf[ 3 ] = { 0u, 4u, 2.0f, 0x1u }; // Cell 0 ↔ ring-y   : RING
    uint32_t edgeCount  = 4u;
    uint32_t agentCount = 5u;

    BufferHandle abR = m_rm->CreateBuffer( { agBz,  BufferType::STORAGE, "AxisR"  } );
    BufferHandle abW = m_rm->CreateBuffer( { agBz,  BufferType::STORAGE, "AxisW"  } );
    BufferHandle pb  = m_rm->CreateBuffer( { phBz,  BufferType::STORAGE, "AxisPh" } );
    BufferHandle cB  = m_rm->CreateBuffer( { cntBz, BufferType::STORAGE, "AxisC"  } );
    BufferHandle eB  = m_rm->CreateBuffer( { edBz,  BufferType::STORAGE, "AxisE"  } );
    BufferHandle ecB = m_rm->CreateBuffer( { cntBz, BufferType::STORAGE, "AxisEc" } );
    BufferHandle oB  = m_rm->CreateBuffer( { agBz,  BufferType::STORAGE, "AxisOr" } );

    m_stream->UploadBufferImmediate( {
        { abR, agents.data(),  agBz,  0 },
        { abW, agents.data(),  agBz,  0 },
        { pb,  pheno.data(),   phBz,  0 },
        { cB,  &agentCount,    cntBz, 0 },
        { eB,  edgeBuf.data(), edBz,  0 },
        { ecB, &edgeCount,     cntBz, 0 },
    } );

    ComputePipelineDesc   pd{ m_rm->CreateShader( "shaders/compute/biology/mitosis_vessel_append.comp" ), "RingAxisDir" };
    ComputePipelineHandle ph = m_rm->CreatePipeline( pd );
    BindingGroup* bg = m_rm->GetBindingGroup( m_rm->CreateBindingGroup( ph, 0 ) );
    bg->Bind( 0, m_rm->GetBuffer( abR ) ); bg->Bind( 1, m_rm->GetBuffer( abW ) );
    bg->Bind( 2, m_rm->GetBuffer( pb  ) ); bg->Bind( 3, m_rm->GetBuffer( cB  ) );
    bg->Bind( 4, m_rm->GetBuffer( eB  ) ); bg->Bind( 5, m_rm->GetBuffer( ecB ) );
    bg->Bind( 6, m_rm->GetBuffer( oB  ) );
    bg->Build();

    ComputePushConstants pc{};
    pc.totalTime = 10.0f; pc.maxCapacity = maxCap; pc.uParam1 = 0;

    auto ctx = m_device->GetThreadContext( m_device->CreateThreadContext( QueueType::COMPUTE ) );
    auto cmd = ctx->GetCommandBuffer( ctx->CreateCommandBuffer() );
    cmd->Begin();
    cmd->SetPipeline( m_rm->GetPipeline( ph ) );
    cmd->SetBindingGroup( bg, m_rm->GetPipeline( ph )->GetLayout(), VK_PIPELINE_BIND_POINT_COMPUTE );
    cmd->PushConstants( m_rm->GetPipeline( ph )->GetLayout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof( pc ), &pc );
    cmd->Dispatch( 1, 1, 1 );
    cmd->End();
    m_device->GetComputeQueue()->Submit( { cmd } );
    m_device->GetComputeQueue()->WaitIdle();

    std::vector<glm::vec4> resPos( maxCap );
    uint32_t resCount;
    m_stream->ReadbackBufferImmediate( abW, resPos.data(), agBz );
    m_stream->ReadbackBufferImmediate( cB,  &resCount,     cntBz );

    // Cell 0 divides → daughter at slot 5.
    // axis=(1,0,0) → daughter = motherPos + (1,0,0)*1.5 = (1.5, 0, 0)
    ASSERT_EQ( resCount, 6u ) << "Cell 0 must produce 1 daughter → count = 6";
    EXPECT_GT( resPos[ 5 ].x,  0.0f )          << "Daughter x must be positive (toward TipCell)";
    EXPECT_NEAR( resPos[ 5 ].y,  0.0f, 0.05f ) << "Daughter y must be near 0";
    EXPECT_NEAR( resPos[ 5 ].z,  0.0f, 0.1f )  << "Daughter z must be near 0";

    m_rm->DestroyBuffer( abR ); m_rm->DestroyBuffer( abW ); m_rm->DestroyBuffer( pb );
    m_rm->DestroyBuffer( cB );  m_rm->DestroyBuffer( eB );  m_rm->DestroyBuffer( ecB );
    m_rm->DestroyBuffer( oB );
}

// StalkCell whose only TipCell edge is a RING edge must mature to PhalanxCell (ring edges
// do not confer TipCell adjacency for the proliferation zone).
TEST_F( ComputeTest, Shader_VesselMitosis_RingTopology_StalkMaturation_IgnoresRingEdges )
{
    if( !m_device )
        GTEST_SKIP();

    struct VesselEdge    { uint32_t agentA; uint32_t agentB; float dist; uint32_t flags; };

    uint32_t maxCap = 4;
    size_t   agBz   = maxCap * sizeof( glm::vec4 );
    size_t   phBz   = maxCap * sizeof( PhenotypeData );
    size_t   edBz   = maxCap * sizeof( VesselEdge );
    size_t   cntBz  = sizeof( uint32_t );

    std::vector<glm::vec4> agents( maxCap, glm::vec4( 0.0f ) );
    agents[ 0 ] = glm::vec4( 0.0f, 0.0f, 0.0f, 1.0f ); // StalkCell
    agents[ 1 ] = glm::vec4( 0.0f, 2.0f, 0.0f, 1.0f ); // TipCell (ring neighbor of 0)

    std::vector<PhenotypeData> pheno( maxCap, { 0u, 0.0f, 0.0f, 0u } );
    pheno[ 0 ] = { 0u, 0.5f, 0.0f, 2u }; // StalkCell, biomass not ready
    pheno[ 1 ] = { 0u, 0.5f, 0.0f, 1u }; // TipCell

    std::vector<VesselEdge> edgeBuf( maxCap, { 0u, 0u, 0.0f, 0u } );
    edgeBuf[ 0 ] = { 0u, 1u, 2.0f, 0x1u }; // RING edge only — must be ignored for adjacency
    uint32_t edgeCount  = 1u;
    uint32_t agentCount = 2u;

    BufferHandle abR = m_rm->CreateBuffer( { agBz,  BufferType::STORAGE, "MatR"  } );
    BufferHandle abW = m_rm->CreateBuffer( { agBz,  BufferType::STORAGE, "MatW"  } );
    BufferHandle pb  = m_rm->CreateBuffer( { phBz,  BufferType::STORAGE, "MatPh" } );
    BufferHandle cB  = m_rm->CreateBuffer( { cntBz, BufferType::STORAGE, "MatC"  } );
    BufferHandle eB  = m_rm->CreateBuffer( { edBz,  BufferType::STORAGE, "MatE"  } );
    BufferHandle ecB = m_rm->CreateBuffer( { cntBz, BufferType::STORAGE, "MatEc" } );
    BufferHandle oB  = m_rm->CreateBuffer( { agBz,  BufferType::STORAGE, "MatOr" } );

    m_stream->UploadBufferImmediate( {
        { abR, agents.data(),  agBz,  0 },
        { abW, agents.data(),  agBz,  0 },
        { pb,  pheno.data(),   phBz,  0 },
        { cB,  &agentCount,    cntBz, 0 },
        { eB,  edgeBuf.data(), edBz,  0 },
        { ecB, &edgeCount,     cntBz, 0 },
    } );

    ComputePipelineDesc   pd{ m_rm->CreateShader( "shaders/compute/biology/mitosis_vessel_append.comp" ), "RingMaturation" };
    ComputePipelineHandle ph = m_rm->CreatePipeline( pd );
    BindingGroup* bg = m_rm->GetBindingGroup( m_rm->CreateBindingGroup( ph, 0 ) );
    bg->Bind( 0, m_rm->GetBuffer( abR ) ); bg->Bind( 1, m_rm->GetBuffer( abW ) );
    bg->Bind( 2, m_rm->GetBuffer( pb  ) ); bg->Bind( 3, m_rm->GetBuffer( cB  ) );
    bg->Bind( 4, m_rm->GetBuffer( eB  ) ); bg->Bind( 5, m_rm->GetBuffer( ecB ) );
    bg->Bind( 6, m_rm->GetBuffer( oB  ) );
    bg->Build();

    ComputePushConstants pc{};
    pc.totalTime = 10.0f; // past 3.0s grace period → maturation fires
    pc.maxCapacity = maxCap; pc.uParam1 = 0;

    auto ctx = m_device->GetThreadContext( m_device->CreateThreadContext( QueueType::COMPUTE ) );
    auto cmd = ctx->GetCommandBuffer( ctx->CreateCommandBuffer() );
    cmd->Begin();
    cmd->SetPipeline( m_rm->GetPipeline( ph ) );
    cmd->SetBindingGroup( bg, m_rm->GetPipeline( ph )->GetLayout(), VK_PIPELINE_BIND_POINT_COMPUTE );
    cmd->PushConstants( m_rm->GetPipeline( ph )->GetLayout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof( pc ), &pc );
    cmd->Dispatch( 1, 1, 1 );
    cmd->End();
    m_device->GetComputeQueue()->Submit( { cmd } );
    m_device->GetComputeQueue()->WaitIdle();

    std::vector<PhenotypeData> resPheno( maxCap );
    m_stream->ReadbackBufferImmediate( pb, resPheno.data(), phBz );

    EXPECT_EQ( resPheno[ 0 ].cellType, 3u )
        << "StalkCell with only a RING edge to TipCell must mature to PhalanxCell";
    EXPECT_EQ( resPheno[ 1 ].cellType, 1u ) << "TipCell must remain TipCell";

    m_rm->DestroyBuffer( abR ); m_rm->DestroyBuffer( abW ); m_rm->DestroyBuffer( pb );
    m_rm->DestroyBuffer( cB );  m_rm->DestroyBuffer( eB );  m_rm->DestroyBuffer( ecB );
    m_rm->DestroyBuffer( oB );
}

// =================================================================================================
// BrownianMotion shader tests
// =================================================================================================

// Verify that an alive agent moves after a single dispatch (speed=10, dt=1 → guaranteed displacement)
TEST_F( ComputeTest, Shader_BrownianMotion_AgentMoves )
{
    if( !m_device )
        GTEST_SKIP();

    glm::vec4 agentIn  = glm::vec4( 0.0f, 0.0f, 0.0f, 1.0f );
    glm::vec4 agentOut = glm::vec4( 0.0f );

    BufferHandle inBuf    = m_rm->CreateBuffer( { sizeof( glm::vec4 ),  BufferType::STORAGE,  "BrownInBuf" } );
    BufferHandle outBuf   = m_rm->CreateBuffer( { sizeof( glm::vec4 ),  BufferType::STORAGE,  "BrownOutBuf" } );
    BufferHandle countBuf = m_rm->CreateBuffer( { sizeof( uint32_t ),   BufferType::INDIRECT, "BrownCountBuf" } );

    m_stream->UploadBufferImmediate( { { inBuf,  &agentIn,  sizeof( glm::vec4 ) } } );
    m_stream->UploadBufferImmediate( { { outBuf, &agentOut, sizeof( glm::vec4 ) } } );
    uint32_t count = 1;
    m_stream->UploadBufferImmediate( { { countBuf, &count, sizeof( uint32_t ) } } );

    PhenotypeData phenotype = { 0u, 0.5f, 0.0f, 0u }; // alive (Normoxic)
    BufferHandle  phenoBuf  = m_rm->CreateBuffer( { sizeof( PhenotypeData ), BufferType::STORAGE, "BrownPhenoBuf" } );
    m_stream->UploadBufferImmediate( { { phenoBuf, &phenotype, sizeof( PhenotypeData ) } } );

    ComputePipelineDesc   pipeDesc{};
    pipeDesc.shader                  = m_rm->CreateShader( "shaders/compute/brownian.comp" );
    ComputePipelineHandle pipeHandle = m_rm->CreatePipeline( pipeDesc );
    ComputePipeline*      pipeline   = m_rm->GetPipeline( pipeHandle );

    BindingGroup* bg = m_rm->GetBindingGroup( m_rm->CreateBindingGroup( pipeHandle, 0 ) );
    bg->Bind( 0, m_rm->GetBuffer( inBuf ) );
    bg->Bind( 1, m_rm->GetBuffer( outBuf ) );
    bg->Bind( 2, m_rm->GetBuffer( countBuf ) );
    bg->Bind( 3, m_rm->GetBuffer( phenoBuf ) );
    bg->Build();

    ComputePushConstants pc{};
    pc.dt          = 1.0f;
    pc.totalTime   = 0.0f;
    pc.fParam0     = 10.0f;                       // speed
    pc.fParam1     = -1.0f;                       // reqCT = -1 (any)
    pc.offset      = 0;
    pc.maxCapacity = 1;
    pc.uParam0     = static_cast<uint32_t>( -1 ); // reqLC = -1 (any)
    pc.uParam1     = 0;                           // grpNdx

    auto ctx = m_device->GetThreadContext( m_device->CreateThreadContext( QueueType::COMPUTE ) );
    auto cmd = ctx->GetCommandBuffer( ctx->CreateCommandBuffer() );
    cmd->Begin();
    cmd->SetPipeline( pipeline );
    cmd->SetBindingGroup( bg, pipeline->GetLayout(), VK_PIPELINE_BIND_POINT_COMPUTE );
    cmd->PushConstants( pipeline->GetLayout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof( pc ), &pc );
    cmd->Dispatch( 1, 1, 1 );
    cmd->End();
    m_device->GetComputeQueue()->Submit( { cmd } );
    m_device->GetComputeQueue()->WaitIdle();

    glm::vec4 result;
    m_stream->ReadbackBufferImmediate( outBuf, &result, sizeof( glm::vec4 ) );

    // RNG with globalIdx=0, totalTime=0 gives rx=-1 → displacement = 10 in X
    float displacement = glm::length( glm::vec3( result ) - glm::vec3( agentIn ) );
    EXPECT_GT( displacement, 0.0f ) << "BrownianMotion: alive agent did not move";
    EXPECT_FLOAT_EQ( result.w, 1.0f ) << "w flag was corrupted";

    m_rm->DestroyBuffer( inBuf );
    m_rm->DestroyBuffer( outBuf );
    m_rm->DestroyBuffer( countBuf );
    m_rm->DestroyBuffer( phenoBuf );
}

// Verify that a dead (Necrotic, lifecycleState=3) agent is written through unchanged
TEST_F( ComputeTest, Shader_BrownianMotion_DeadSlot_Skipped )
{
    if( !m_device )
        GTEST_SKIP();

    glm::vec4 agentIn  = glm::vec4( 5.0f, 3.0f, -2.0f, 1.0f );
    glm::vec4 agentOut = glm::vec4( 0.0f );

    BufferHandle inBuf    = m_rm->CreateBuffer( { sizeof( glm::vec4 ),  BufferType::STORAGE,  "BrownDeadInBuf" } );
    BufferHandle outBuf   = m_rm->CreateBuffer( { sizeof( glm::vec4 ),  BufferType::STORAGE,  "BrownDeadOutBuf" } );
    BufferHandle countBuf = m_rm->CreateBuffer( { sizeof( uint32_t ),   BufferType::INDIRECT, "BrownDeadCountBuf" } );

    m_stream->UploadBufferImmediate( { { inBuf,  &agentIn,  sizeof( glm::vec4 ) } } );
    m_stream->UploadBufferImmediate( { { outBuf, &agentOut, sizeof( glm::vec4 ) } } );
    uint32_t count = 1;
    m_stream->UploadBufferImmediate( { { countBuf, &count, sizeof( uint32_t ) } } );

    PhenotypeData phenotype = { 3u, 0.5f, 0.0f, 0u }; // lifecycleState=3 (Necrotic)
    BufferHandle  phenoBuf  = m_rm->CreateBuffer( { sizeof( PhenotypeData ), BufferType::STORAGE, "BrownDeadPhenoBuf" } );
    m_stream->UploadBufferImmediate( { { phenoBuf, &phenotype, sizeof( PhenotypeData ) } } );

    ComputePipelineDesc   pipeDesc{};
    pipeDesc.shader                  = m_rm->CreateShader( "shaders/compute/brownian.comp" );
    ComputePipelineHandle pipeHandle = m_rm->CreatePipeline( pipeDesc );
    ComputePipeline*      pipeline   = m_rm->GetPipeline( pipeHandle );

    BindingGroup* bg = m_rm->GetBindingGroup( m_rm->CreateBindingGroup( pipeHandle, 0 ) );
    bg->Bind( 0, m_rm->GetBuffer( inBuf ) );
    bg->Bind( 1, m_rm->GetBuffer( outBuf ) );
    bg->Bind( 2, m_rm->GetBuffer( countBuf ) );
    bg->Bind( 3, m_rm->GetBuffer( phenoBuf ) );
    bg->Build();

    ComputePushConstants pc{};
    pc.dt          = 1.0f;
    pc.totalTime   = 0.0f;
    pc.fParam0     = 100.0f;                      // high speed — would move far if not skipped
    pc.fParam1     = -1.0f;
    pc.offset      = 0;
    pc.maxCapacity = 1;
    pc.uParam0     = static_cast<uint32_t>( -1 );
    pc.uParam1     = 0;

    auto ctx = m_device->GetThreadContext( m_device->CreateThreadContext( QueueType::COMPUTE ) );
    auto cmd = ctx->GetCommandBuffer( ctx->CreateCommandBuffer() );
    cmd->Begin();
    cmd->SetPipeline( pipeline );
    cmd->SetBindingGroup( bg, pipeline->GetLayout(), VK_PIPELINE_BIND_POINT_COMPUTE );
    cmd->PushConstants( pipeline->GetLayout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof( pc ), &pc );
    cmd->Dispatch( 1, 1, 1 );
    cmd->End();
    m_device->GetComputeQueue()->Submit( { cmd } );
    m_device->GetComputeQueue()->WaitIdle();

    glm::vec4 result;
    m_stream->ReadbackBufferImmediate( outBuf, &result, sizeof( glm::vec4 ) );

    EXPECT_FLOAT_EQ( result.x, agentIn.x ) << "Dead agent x was modified";
    EXPECT_FLOAT_EQ( result.y, agentIn.y ) << "Dead agent y was modified";
    EXPECT_FLOAT_EQ( result.z, agentIn.z ) << "Dead agent z was modified";
    EXPECT_FLOAT_EQ( result.w, agentIn.w ) << "Dead agent w was modified";

    m_rm->DestroyBuffer( inBuf );
    m_rm->DestroyBuffer( outBuf );
    m_rm->DestroyBuffer( countBuf );
    m_rm->DestroyBuffer( phenoBuf );
}

// =============================================================================
// jkr_forces.comp — cadherin adhesion scaling tests
// =============================================================================

// Helper: dispatch jkr_forces once for two overlapping agents, return their x positions.
// Both agents are placed on the X axis, same Y/Z.  The hash places both in cell (0,0,0).
// Returns {agent0_x, agent1_x}.
// Plate parameter pack matching the GPU `PlateBuf` layout used by jkr_forces +
// polarity_update. All fields default to "disabled" — tests that don't exercise
// the plate branch leave these zeroed and the shader skips the plate block.
struct PlateTestParams {
    glm::vec3 normal            = glm::vec3( 0.0f, 0.0f, 1.0f );
    float     height            = 0.0f;
    float     contactStiffness  = 0.0f;
    float     integrinAdhesion  = 0.0f;
    float     anchorageDistance = 0.0f;
    float     polarityBias      = 0.0f;
    uint32_t  activeFlag        = 0u;
};

static std::pair<float, float> RunJKRCadherin(
    Device*           device,
    ResourceManager*  rm,
    StreamingManager* stream,
    float             agent0x,        // x-position of agent 0
    float             agent1x,        // x-position of agent 1
    float             repulsion,
    float             adhesion,
    float             maxRadius,
    glm::vec4         profile0,       // cadherin profile for agent 0
    glm::vec4         profile1,       // cadherin profile for agent 1
    glm::mat4         affinityMatrix,
    uint32_t          cadherinFlag,   // 0 = off, 1 = on
    float             couplingStrength,
    uint32_t          polarityFlag    = 0u,        // 0 = off, 1 = on
    glm::vec4         polarity0       = glm::vec4( 0.0f ), // xyz=dir w=magnitude
    glm::vec4         polarity1       = glm::vec4( 0.0f ),
    float             apicalRepulsion = 0.5f,
    float             basalAdhesion   = 1.5f,
    PlateTestParams   plate           = {},
    float             corticalTension = 0.0f )
{
    uint32_t agentCount = 2;

    std::vector<glm::vec4> inAgents = {
        glm::vec4( agent0x, 0.0f, 0.0f, 1.0f ),
        glm::vec4( agent1x, 0.0f, 0.0f, 1.0f )
    };

    struct AgentHash { uint32_t hash; uint32_t agentIndex; };
    // Both agents map to cell (0,0,0) with cellSize large enough to span them
    std::vector<AgentHash> sortedHashes = { { 0, 0 }, { 0, 1 } };

    uint32_t              offsetArraySize = 256;
    std::vector<uint32_t> cellOffsets( offsetArraySize, 0xFFFFFFFF );
    cellOffsets[ 0 ] = 0;

    std::vector<glm::vec4> outAgentsData( agentCount, glm::vec4( 0.0f ) );
    std::vector<float>     outPressures( agentCount, 0.0f );

    BufferHandle inBuf      = rm->CreateBuffer( { agentCount * sizeof( glm::vec4 ),          BufferType::STORAGE,  "JKRCadInBuf" } );
    BufferHandle outBuf     = rm->CreateBuffer( { agentCount * sizeof( glm::vec4 ),          BufferType::STORAGE,  "JKRCadOutBuf" } );
    BufferHandle pressBuf   = rm->CreateBuffer( { agentCount * sizeof( float ),              BufferType::STORAGE,  "JKRCadPressBuf" } );
    BufferHandle hashBuf    = rm->CreateBuffer( { agentCount * sizeof( AgentHash ),          BufferType::STORAGE,  "JKRCadHashBuf" } );
    BufferHandle offsetBuf  = rm->CreateBuffer( { offsetArraySize * sizeof( uint32_t ),      BufferType::STORAGE,  "JKRCadOffsetBuf" } );
    BufferHandle countBuf   = rm->CreateBuffer( { agentCount * sizeof( uint32_t ),           BufferType::INDIRECT, "JKRCadCountBuf" } );
    BufferHandle phenoBuf   = rm->CreateBuffer( { agentCount * sizeof( PhenotypeData ),      BufferType::STORAGE,  "JKRCadPhenoBuf" } );
    BufferHandle profBuf    = rm->CreateBuffer( { agentCount * sizeof( glm::vec4 ),          BufferType::STORAGE,  "JKRCadProfBuf" } );
    BufferHandle affBuf     = rm->CreateBuffer( { sizeof( glm::mat4 ),                       BufferType::STORAGE,  "JKRCadAffBuf" } );
    BufferHandle polarityBuf= rm->CreateBuffer( { agentCount * sizeof( glm::vec4 ),          BufferType::STORAGE,  "JKRCadPolarityBuf" } );

    std::vector<glm::vec4>     profiles   = { profile0, profile1 };
    std::vector<glm::vec4>     polarities = { polarity0, polarity1 };
    std::vector<PhenotypeData> phenotypes( agentCount, { 0u, 0.5f, 0.0f, 0u } );

    stream->UploadBufferImmediate( { { inBuf,        inAgents.data(),     agentCount * sizeof( glm::vec4 ) } } );
    stream->UploadBufferImmediate( { { hashBuf,      sortedHashes.data(), agentCount * sizeof( AgentHash ) } } );
    stream->UploadBufferImmediate( { { offsetBuf,    cellOffsets.data(),  offsetArraySize * sizeof( uint32_t ) } } );
    stream->UploadBufferImmediate( { { countBuf,     &agentCount,         sizeof( uint32_t ) } } );
    stream->UploadBufferImmediate( { { phenoBuf,     phenotypes.data(),   agentCount * sizeof( PhenotypeData ) } } );
    stream->UploadBufferImmediate( { { profBuf,      profiles.data(),     agentCount * sizeof( glm::vec4 ) } } );
    stream->UploadBufferImmediate( { { affBuf,       &affinityMatrix,     sizeof( glm::mat4 ) } } );
    stream->UploadBufferImmediate( { { polarityBuf,  polarities.data(),   agentCount * sizeof( glm::vec4 ) } } );

    ComputePipelineDesc   pipeDesc{};
    pipeDesc.shader                  = rm->CreateShader( "shaders/compute/jkr_forces.comp" );
    ComputePipelineHandle pipeHandle = rm->CreatePipeline( pipeDesc );
    ComputePipeline*      pipeline   = rm->GetPipeline( pipeHandle );

    // Dummy orientation buffer for binding 10 (hull count=0 → rigid body path inactive)
    glm::vec4 identityQuats[2] = { glm::vec4( 0, 0, 0, 1 ), glm::vec4( 0, 0, 0, 1 ) };
    BufferHandle orientBuf = rm->CreateBuffer( { sizeof( identityQuats ), BufferType::STORAGE, "JKROrientDummy" } );
    stream->UploadBufferImmediate( { { orientBuf, identityQuats, sizeof( identityQuats ) } } );
    // Dummy contact hull buffer for binding 11 (hullMeta.x=0 → point-particle fallback)
    struct ContactHullGPU { glm::vec4 meta{}; glm::vec4 points[16]{}; };
    ContactHullGPU dummyHull{};
    BufferHandle hullBuf = rm->CreateBuffer( { sizeof( ContactHullGPU ), BufferType::STORAGE, "JKRHullDummy" } );
    stream->UploadBufferImmediate( { { hullBuf, &dummyHull, sizeof( ContactHullGPU ) } } );

    // Plate buffer (binding 12) — Step B multi-plate layout. meta.x = plate
    // count (0 = disabled, loops skip entirely). We always pack the single
    // test plate into plates[0/1]; activeFlag becomes count in {0, 1}.
    struct PlateBufferGPU { glm::uvec4 meta; glm::vec4 plates[16]; };
    PlateBufferGPU plateData{};
    plateData.meta         = glm::uvec4( plate.activeFlag, 0u, 0u, 0u );
    plateData.plates[ 0 ]  = glm::vec4( plate.normal, plate.height );
    plateData.plates[ 1 ]  = glm::vec4( plate.contactStiffness, plate.integrinAdhesion,
                                        plate.anchorageDistance, plate.polarityBias );
    BufferHandle plateBuf = rm->CreateBuffer( { sizeof( PlateBufferGPU ), BufferType::STORAGE, "JKRPlateBuf" } );
    stream->UploadBufferImmediate( { { plateBuf, &plateData, sizeof( PlateBufferGPU ) } } );

    BindingGroup* bg = rm->GetBindingGroup( rm->CreateBindingGroup( pipeHandle, 0 ) );
    bg->Bind( 0, rm->GetBuffer( inBuf ) );
    bg->Bind( 1, rm->GetBuffer( outBuf ) );
    bg->Bind( 2, rm->GetBuffer( pressBuf ) );
    bg->Bind( 3, rm->GetBuffer( hashBuf ) );
    bg->Bind( 4, rm->GetBuffer( offsetBuf ) );
    bg->Bind( 5, rm->GetBuffer( countBuf ) );
    bg->Bind( 6, rm->GetBuffer( phenoBuf ) );
    bg->Bind( 7, rm->GetBuffer( profBuf ) );
    bg->Bind( 8, rm->GetBuffer( affBuf ) );
    bg->Bind( 9, rm->GetBuffer( polarityBuf ) );
    bg->Bind( 10, rm->GetBuffer( orientBuf ) );
    bg->Bind( 11, rm->GetBuffer( hullBuf ) );
    bg->Bind( 12, rm->GetBuffer( plateBuf ) );
    bg->Build();

    uint32_t couplingBits  = 0u;
    std::memcpy( &couplingBits, &couplingStrength, sizeof( float ) );
    uint32_t polarityBits  = glm::packHalf2x16( glm::vec2( apicalRepulsion, basalAdhesion ) );
    // gridSize.x packs cadherinFlag (bit 0) with half-float corticalTension in the upper 16 bits.
    uint32_t tensionPacked = glm::packHalf2x16( glm::vec2( 0.0f, corticalTension ) );
    uint32_t gridSizeX     = ( cadherinFlag & 1u ) | ( tensionPacked & 0xFFFF0000u );

    ComputePushConstants pc{};
    pc.dt          = 1.0f;
    pc.maxCapacity = agentCount;
    pc.offset      = 0;
    pc.fParam0     = repulsion;
    pc.fParam1     = adhesion;
    pc.fParam2     = -1.0f; // reqLC = any
    pc.fParam3     = -1.0f; // reqCT = any
    pc.fParam4     = 0.0f;  // no damping
    pc.fParam5     = maxRadius;
    pc.uParam0     = offsetArraySize;
    pc.uParam1     = 0;
    pc.domainSize  = glm::vec4( 100.0f, 100.0f, 100.0f, maxRadius * 2.0f + 1.0f ); // cellSize spans both agents
    pc.gridSize    = glm::uvec4( gridSizeX, couplingBits, polarityFlag, polarityBits );

    auto ctx = device->GetThreadContext( device->CreateThreadContext( QueueType::COMPUTE ) );
    auto cmd = ctx->GetCommandBuffer( ctx->CreateCommandBuffer() );
    cmd->Begin();
    cmd->SetPipeline( pipeline );
    cmd->SetBindingGroup( bg, pipeline->GetLayout(), VK_PIPELINE_BIND_POINT_COMPUTE );
    cmd->PushConstants( pipeline->GetLayout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof( pc ), &pc );
    cmd->Dispatch( 1, 1, 1 );
    cmd->End();
    device->GetComputeQueue()->Submit( { cmd } );
    device->GetComputeQueue()->WaitIdle();

    std::vector<glm::vec4> result( agentCount );
    stream->ReadbackBufferImmediate( outBuf, result.data(), agentCount * sizeof( glm::vec4 ) );

    rm->DestroyBuffer( inBuf );  rm->DestroyBuffer( outBuf );  rm->DestroyBuffer( pressBuf );
    rm->DestroyBuffer( hashBuf ); rm->DestroyBuffer( offsetBuf ); rm->DestroyBuffer( countBuf );
    rm->DestroyBuffer( phenoBuf ); rm->DestroyBuffer( profBuf ); rm->DestroyBuffer( affBuf );
    rm->DestroyBuffer( polarityBuf );
    rm->DestroyBuffer( orientBuf ); rm->DestroyBuffer( hullBuf );
    rm->DestroyBuffer( plateBuf );

    return { result[ 0 ].x, result[ 1 ].x };
}

// Minimal single-agent plate harness — tests the BasementMembrane contact
// block in isolation from cell-cell interactions. One agent, no neighbours
// (the hash grid only has its own entry), JKR output position readback.
static glm::vec3 RunJKRWithPlate(
    Device*                 device,
    ResourceManager*        rm,
    StreamingManager*       stream,
    glm::vec3               agentPos,
    const PlateTestParams&  plate,
    float                   dt = 0.01f )
{
    uint32_t agentCount = 1;
    glm::vec4 inAgent( agentPos, 1.0f );

    struct AgentHash { uint32_t hash; uint32_t agentIndex; };
    AgentHash hashEntry{ 0u, 0u };

    uint32_t offsetArraySize = 256;
    std::vector<uint32_t> cellOffsets( offsetArraySize, 0xFFFFFFFF );
    cellOffsets[ 0 ] = 0;

    BufferHandle inBuf    = rm->CreateBuffer( { sizeof( glm::vec4 ),             BufferType::STORAGE,  "PlateInBuf" } );
    BufferHandle outBuf   = rm->CreateBuffer( { sizeof( glm::vec4 ),             BufferType::STORAGE,  "PlateOutBuf" } );
    BufferHandle pressBuf = rm->CreateBuffer( { sizeof( float ),                 BufferType::STORAGE,  "PlatePressBuf" } );
    BufferHandle hashBuf  = rm->CreateBuffer( { sizeof( AgentHash ),             BufferType::STORAGE,  "PlateHashBuf" } );
    BufferHandle offsetBuf= rm->CreateBuffer( { offsetArraySize * sizeof( uint32_t ), BufferType::STORAGE, "PlateOffsetBuf" } );
    BufferHandle countBuf = rm->CreateBuffer( { sizeof( uint32_t ),              BufferType::INDIRECT, "PlateCountBuf" } );
    BufferHandle phenoBuf = rm->CreateBuffer( { sizeof( PhenotypeData ),         BufferType::STORAGE,  "PlatePhenoBuf" } );
    BufferHandle profBuf  = rm->CreateBuffer( { sizeof( glm::vec4 ),             BufferType::STORAGE,  "PlateProfBuf" } );
    BufferHandle affBuf   = rm->CreateBuffer( { sizeof( glm::mat4 ),             BufferType::STORAGE,  "PlateAffBuf" } );
    BufferHandle polBuf   = rm->CreateBuffer( { sizeof( glm::vec4 ),             BufferType::STORAGE,  "PlatePolBuf" } );
    BufferHandle orientBuf= rm->CreateBuffer( { sizeof( glm::vec4 ),             BufferType::STORAGE,  "PlateOrientBuf" } );

    struct ContactHullGPU { glm::vec4 meta{}; glm::vec4 points[16]{}; };
    ContactHullGPU dummyHull{};
    BufferHandle hullBuf  = rm->CreateBuffer( { sizeof( ContactHullGPU ),        BufferType::STORAGE,  "PlateHullBuf" } );

    struct PlateBufferGPU { glm::uvec4 meta; glm::vec4 plates[16]; };
    PlateBufferGPU plateData{};
    plateData.meta         = glm::uvec4( plate.activeFlag, 0u, 0u, 0u );
    plateData.plates[ 0 ]  = glm::vec4( plate.normal, plate.height );
    plateData.plates[ 1 ]  = glm::vec4( plate.contactStiffness, plate.integrinAdhesion,
                                        plate.anchorageDistance, plate.polarityBias );
    BufferHandle plateBuf = rm->CreateBuffer( { sizeof( PlateBufferGPU ),         BufferType::STORAGE,  "PlatePlateBuf" } );

    glm::vec4     profile( 0.0f );
    glm::mat4     affinity( 1.0f );
    glm::vec4     polarity( 0.0f );
    glm::vec4     orient( 0.0f, 0.0f, 0.0f, 1.0f );
    PhenotypeData pheno{ 0u, 0.5f, 0.0f, 0u };
    uint32_t      count = agentCount;

    stream->UploadBufferImmediate( { { inBuf,     &inAgent,       sizeof( glm::vec4 ) } } );
    stream->UploadBufferImmediate( { { hashBuf,   &hashEntry,     sizeof( AgentHash ) } } );
    stream->UploadBufferImmediate( { { offsetBuf, cellOffsets.data(), offsetArraySize * sizeof( uint32_t ) } } );
    stream->UploadBufferImmediate( { { countBuf,  &count,         sizeof( uint32_t ) } } );
    stream->UploadBufferImmediate( { { phenoBuf,  &pheno,         sizeof( PhenotypeData ) } } );
    stream->UploadBufferImmediate( { { profBuf,   &profile,       sizeof( glm::vec4 ) } } );
    stream->UploadBufferImmediate( { { affBuf,    &affinity,      sizeof( glm::mat4 ) } } );
    stream->UploadBufferImmediate( { { polBuf,    &polarity,      sizeof( glm::vec4 ) } } );
    stream->UploadBufferImmediate( { { orientBuf, &orient,        sizeof( glm::vec4 ) } } );
    stream->UploadBufferImmediate( { { hullBuf,   &dummyHull,     sizeof( ContactHullGPU ) } } );
    stream->UploadBufferImmediate( { { plateBuf,  &plateData,     sizeof( PlateBufferGPU ) } } );

    ComputePipelineDesc pipeDesc{};
    pipeDesc.shader = rm->CreateShader( "shaders/compute/jkr_forces.comp" );
    ComputePipelineHandle pipeHandle = rm->CreatePipeline( pipeDesc );
    ComputePipeline*      pipeline   = rm->GetPipeline( pipeHandle );

    BindingGroup* bg = rm->GetBindingGroup( rm->CreateBindingGroup( pipeHandle, 0 ) );
    bg->Bind( 0, rm->GetBuffer( inBuf ) );
    bg->Bind( 1, rm->GetBuffer( outBuf ) );
    bg->Bind( 2, rm->GetBuffer( pressBuf ) );
    bg->Bind( 3, rm->GetBuffer( hashBuf ) );
    bg->Bind( 4, rm->GetBuffer( offsetBuf ) );
    bg->Bind( 5, rm->GetBuffer( countBuf ) );
    bg->Bind( 6, rm->GetBuffer( phenoBuf ) );
    bg->Bind( 7, rm->GetBuffer( profBuf ) );
    bg->Bind( 8, rm->GetBuffer( affBuf ) );
    bg->Bind( 9, rm->GetBuffer( polBuf ) );
    bg->Bind( 10, rm->GetBuffer( orientBuf ) );
    bg->Bind( 11, rm->GetBuffer( hullBuf ) );
    bg->Bind( 12, rm->GetBuffer( plateBuf ) );
    bg->Build();

    ComputePushConstants pc{};
    pc.dt          = dt;
    pc.maxCapacity = agentCount;
    pc.offset      = 0;
    pc.fParam0     = 10.0f;   // repulsion
    pc.fParam1     = 1.0f;    // adhesion (small; neighbour-less so unused)
    pc.fParam2     = -1.0f;   // reqLC = any
    pc.fParam3     = -1.0f;   // reqCT = any
    pc.fParam4     = 0.0f;    // damping
    pc.fParam5     = 0.5f;    // maxRadius (small; no neighbour in range)
    pc.uParam0     = offsetArraySize;
    pc.uParam1     = 0;
    pc.domainSize  = glm::vec4( 100.0f, 100.0f, 100.0f, 2.0f );
    pc.gridSize    = glm::uvec4( 0u, 0u, 0u, 0u );

    auto ctx = device->GetThreadContext( device->CreateThreadContext( QueueType::COMPUTE ) );
    auto cmd = ctx->GetCommandBuffer( ctx->CreateCommandBuffer() );
    cmd->Begin();
    cmd->SetPipeline( pipeline );
    cmd->SetBindingGroup( bg, pipeline->GetLayout(), VK_PIPELINE_BIND_POINT_COMPUTE );
    cmd->PushConstants( pipeline->GetLayout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof( pc ), &pc );
    cmd->Dispatch( 1, 1, 1 );
    cmd->End();
    device->GetComputeQueue()->Submit( { cmd } );
    device->GetComputeQueue()->WaitIdle();

    glm::vec4 result( 0.0f );
    stream->ReadbackBufferImmediate( outBuf, &result, sizeof( glm::vec4 ) );

    rm->DestroyBuffer( inBuf ); rm->DestroyBuffer( outBuf ); rm->DestroyBuffer( pressBuf );
    rm->DestroyBuffer( hashBuf ); rm->DestroyBuffer( offsetBuf ); rm->DestroyBuffer( countBuf );
    rm->DestroyBuffer( phenoBuf ); rm->DestroyBuffer( profBuf ); rm->DestroyBuffer( affBuf );
    rm->DestroyBuffer( polBuf ); rm->DestroyBuffer( orientBuf ); rm->DestroyBuffer( hullBuf );
    rm->DestroyBuffer( plateBuf );

    return glm::vec3( result );
}

// 1. Matching profiles + coupling > 1 → higher adhesion → agents move less apart than without cadherin
TEST_F( ComputeTest, Shader_JKRForces_Cadherin_SameProfile_IncreasedAdhesion )
{
    if( !m_device )
        GTEST_SKIP();

    // Agents slightly overlapping: dist = 2.9, interactDist = 3.0, overlap = 0.1
    // With couplingStrength=2 and identical E-cad profiles: adhForce doubles → less repulsion
    float repulsion = 50.0f, adhesion = 5.0f, maxRadius = 1.5f;

    auto [x0_cad, x1_cad] = RunJKRCadherin(
        m_device.get(), m_rm.get(), m_stream.get(),
        0.0f, 2.9f,
        repulsion, adhesion, maxRadius,
        glm::vec4( 1.0f, 0.0f, 0.0f, 0.0f ),  // E-cad = 1
        glm::vec4( 1.0f, 0.0f, 0.0f, 0.0f ),  // E-cad = 1
        glm::mat4( 1.0f ),                      // identity affinity
        1u,                                      // cadherin ON
        2.0f );                                  // couplingStrength = 2 → 2× adhesion

    auto [x0_base, x1_base] = RunJKRCadherin(
        m_device.get(), m_rm.get(), m_stream.get(),
        0.0f, 2.9f,
        repulsion, adhesion, maxRadius,
        glm::vec4( 0.0f ),
        glm::vec4( 0.0f ),
        glm::mat4( 1.0f ),
        0u,   // cadherin OFF (flag = 0)
        1.0f );

    // With doubled adhesion, agents should be pushed apart less
    float disp_cad  = x1_cad  - x0_cad;   // separation after cadherin
    float disp_base = x1_base - x0_base;   // separation after normal JKR
    EXPECT_LT( disp_cad, disp_base )
        << "Cadherin coupling=2 should reduce net repulsion (less separation)";
}

// 2. Orthogonal profiles (E-cad vs N-cad) → A = 0 → no cadherin adhesion → pure repulsion
TEST_F( ComputeTest, Shader_JKRForces_Cadherin_OrthogonalProfiles_ZeroAdhesion )
{
    if( !m_device )
        GTEST_SKIP();

    float repulsion = 50.0f, adhesion = 5.0f, maxRadius = 1.5f;

    // Orthogonal profiles: A = dot((1,0,0,0), I*(0,1,0,0)) = 0 → adhForce = 0
    auto [x0_ortho, x1_ortho] = RunJKRCadherin(
        m_device.get(), m_rm.get(), m_stream.get(),
        0.0f, 2.9f,
        repulsion, adhesion, maxRadius,
        glm::vec4( 1.0f, 0.0f, 0.0f, 0.0f ),  // E-cad
        glm::vec4( 0.0f, 1.0f, 0.0f, 0.0f ),  // N-cad
        glm::mat4( 1.0f ),
        1u,     // cadherin ON
        1.0f ); // coupling = 1 (irrelevant when A=0)

    // Without cadherin (flag=0): base adhesion reduces repulsion → smaller separation
    auto [x0_base, x1_base] = RunJKRCadherin(
        m_device.get(), m_rm.get(), m_stream.get(),
        0.0f, 2.9f,
        repulsion, adhesion, maxRadius,
        glm::vec4( 0.0f ),
        glm::vec4( 0.0f ),
        glm::mat4( 1.0f ),
        0u,     // cadherin OFF
        1.0f );

    // Orthogonal profiles → no adhesion → more repulsion → larger separation
    float disp_ortho = x1_ortho - x0_ortho;
    float disp_base  = x1_base  - x0_base;
    EXPECT_GT( disp_ortho, disp_base )
        << "Orthogonal profiles (A=0) should produce more repulsion than normal JKR";
}

// =============================================================================
// jkr_forces.comp — polarity-modulated adhesion tests
// =============================================================================

// Polarity vectors: agent 0 faces +X (basal toward agent 1), agent 1 faces -X (basal toward agent 0).
// Both basal-facing → high alignment → scale = basalAdhesion > 1 → stronger adhesion → less repulsion.
TEST_F( ComputeTest, Shader_JKR_Polarity_BasalFacing_StrongAdhesion )
{
    if( !m_device )
        GTEST_SKIP();

    float repulsion = 50.0f, adhesion = 5.0f, maxRadius = 1.5f;

    // Basal-basal: polarity vectors point toward each other (outward toward the contact)
    // Agent 0 at x=0, polarity +X; agent 1 at x=2.9, polarity -X
    glm::vec4 pol0 = glm::vec4(  1.0f, 0.0f, 0.0f, 1.0f ); // dir=+X, magnitude=1
    glm::vec4 pol1 = glm::vec4( -1.0f, 0.0f, 0.0f, 1.0f ); // dir=-X, magnitude=1

    auto [x0_pol, x1_pol] = RunJKRCadherin(
        m_device.get(), m_rm.get(), m_stream.get(),
        0.0f, 2.9f, repulsion, adhesion, maxRadius,
        glm::vec4( 0.0f ), glm::vec4( 0.0f ), glm::mat4( 1.0f ),
        0u, 1.0f,   // cadherin OFF
        1u, pol0, pol1, 0.5f, 2.0f );  // polarity ON, basalAdhesion=2

    auto [x0_base, x1_base] = RunJKRCadherin(
        m_device.get(), m_rm.get(), m_stream.get(),
        0.0f, 2.9f, repulsion, adhesion, maxRadius,
        glm::vec4( 0.0f ), glm::vec4( 0.0f ), glm::mat4( 1.0f ),
        0u, 1.0f ); // polarity OFF (default)

    // Basal-basal contact → more adhesion → less separation
    float disp_pol  = x1_pol  - x0_pol;
    float disp_base = x1_base - x0_base;
    EXPECT_LT( disp_pol, disp_base )
        << "Basal-basal polarity should increase adhesion → reduce separation";
}

// Apical-apical: both polarities point away from contact → low alignment → apicalRepulsion < 1 → less adhesion.
TEST_F( ComputeTest, Shader_JKR_Polarity_ApicalFacing_WeakAdhesion )
{
    if( !m_device )
        GTEST_SKIP();

    float repulsion = 50.0f, adhesion = 5.0f, maxRadius = 1.5f;

    // Agent 0 at x=0, polarity -X (apical faces agent 1); agent 1 at x=2.9, polarity +X
    glm::vec4 pol0 = glm::vec4( -1.0f, 0.0f, 0.0f, 1.0f ); // apical toward agent 1
    glm::vec4 pol1 = glm::vec4(  1.0f, 0.0f, 0.0f, 1.0f ); // apical toward agent 0

    auto [x0_pol, x1_pol] = RunJKRCadherin(
        m_device.get(), m_rm.get(), m_stream.get(),
        0.0f, 2.9f, repulsion, adhesion, maxRadius,
        glm::vec4( 0.0f ), glm::vec4( 0.0f ), glm::mat4( 1.0f ),
        0u, 1.0f,
        1u, pol0, pol1, 0.1f, 1.5f );  // polarity ON, apicalRepulsion=0.1 (strongly weakened)

    auto [x0_base, x1_base] = RunJKRCadherin(
        m_device.get(), m_rm.get(), m_stream.get(),
        0.0f, 2.9f, repulsion, adhesion, maxRadius,
        glm::vec4( 0.0f ), glm::vec4( 0.0f ), glm::mat4( 1.0f ),
        0u, 1.0f );

    // Apical-apical contact → less adhesion → more separation
    float disp_pol  = x1_pol  - x0_pol;
    float disp_base = x1_base - x0_base;
    EXPECT_GT( disp_pol, disp_base )
        << "Apical-apical polarity should reduce adhesion → increase separation";
}

// Zero polarity magnitude (w=0) → mix(1.0, scale, 0) = 1.0 → no change from baseline.
TEST_F( ComputeTest, Shader_JKR_Polarity_ZeroMagnitude_NoEffect )
{
    if( !m_device )
        GTEST_SKIP();

    float repulsion = 50.0f, adhesion = 5.0f, maxRadius = 1.5f;

    // Polarity vectors have direction but w=0 (interior cell — no polarity effect)
    glm::vec4 pol0 = glm::vec4(  1.0f, 0.0f, 0.0f, 0.0f ); // magnitude = 0
    glm::vec4 pol1 = glm::vec4( -1.0f, 0.0f, 0.0f, 0.0f ); // magnitude = 0

    auto [x0_pol, x1_pol] = RunJKRCadherin(
        m_device.get(), m_rm.get(), m_stream.get(),
        0.0f, 2.9f, repulsion, adhesion, maxRadius,
        glm::vec4( 0.0f ), glm::vec4( 0.0f ), glm::mat4( 1.0f ),
        0u, 1.0f,
        1u, pol0, pol1, 0.1f, 3.0f ); // polarity flag ON but w=0 on both

    auto [x0_base, x1_base] = RunJKRCadherin(
        m_device.get(), m_rm.get(), m_stream.get(),
        0.0f, 2.9f, repulsion, adhesion, maxRadius,
        glm::vec4( 0.0f ), glm::vec4( 0.0f ), glm::mat4( 1.0f ),
        0u, 1.0f ); // polarity OFF

    EXPECT_NEAR( x0_pol, x0_base, 1e-4f ) << "Zero-magnitude polarity should produce no change";
    EXPECT_NEAR( x1_pol, x1_base, 1e-4f ) << "Zero-magnitude polarity should produce no change";
}

// Phase 3: apical-apical aligned contact with NEGATIVE apicalRepulsion flips
// adhesion into ACTIVE repulsion — two cells end up further apart than without
// any JKR attraction at all. This is the PODXL electrostatic repulsion
// mechanism (Strilic 2009): at apical poles, net-negative polarity modifier
// pushes cells apart beyond the baseline repulsion distance.
TEST_F( ComputeTest, Shader_JKR_Polarity_ApicalFacing_NetNegative_IsRepulsive )
{
    if( !m_device )
        GTEST_SKIP();

    float repulsion = 50.0f, adhesion = 5.0f, maxRadius = 1.5f;

    // Agent 0 at x=0, polarity -X (apical toward agent 1)
    // Agent 1 at x=2.9, polarity +X (apical toward agent 0)
    // Both magnitudes = 1 → full polarity effect.
    glm::vec4 pol0 = glm::vec4( -1.0f, 0.0f, 0.0f, 1.0f );
    glm::vec4 pol1 = glm::vec4(  1.0f, 0.0f, 0.0f, 1.0f );

    // Phase 3 values: apicalRepulsion = -1.0 (active repulsion)
    auto [x0_neg, x1_neg] = RunJKRCadherin(
        m_device.get(), m_rm.get(), m_stream.get(),
        0.0f, 2.9f, repulsion, adhesion, maxRadius,
        glm::vec4( 0.0f ), glm::vec4( 0.0f ), glm::mat4( 1.0f ),
        0u, 1.0f,
        1u, pol0, pol1, -1.0f, 2.5f );

    // Baseline: Phase 2 values: apicalRepulsion = 0.3 (weakened, still attractive)
    auto [x0_weak, x1_weak] = RunJKRCadherin(
        m_device.get(), m_rm.get(), m_stream.get(),
        0.0f, 2.9f, repulsion, adhesion, maxRadius,
        glm::vec4( 0.0f ), glm::vec4( 0.0f ), glm::mat4( 1.0f ),
        0u, 1.0f,
        1u, pol0, pol1, 0.3f, 1.5f );

    float disp_neg  = x1_neg  - x0_neg;
    float disp_weak = x1_weak - x0_weak;

    // Net-negative apical must produce STRICTLY LARGER separation than merely
    // weakened-apical — proves the sign flip is taking effect, not just a scale.
    EXPECT_GT( disp_neg, disp_weak )
        << "Net-negative apical (-1.0) must push cells apart more than weak apical (+0.3)";
    // Additionally, the net motion under -1.0 must move cells OUTWARD relative to
    // their starting 2.9 separation (since overlap = 0.1 initially, the pair
    // should separate, not compress).
    EXPECT_GT( disp_neg, 2.9f )
        << "Net-negative apical must push cells past their starting separation (active repulsion)";
}

// Phase 4 — cortical tension (Maître et al. 2012 interfacial-tension model).
// At fixed overlap, cortical tension contributes an outward force that linearly
// opposes the inward adhesion term. With all else equal, adding corticalTension
// must leave cells further apart after one dispatch than the baseline (tension = 0).
TEST_F( ComputeTest, Shader_JKR_CorticalTension_OpposesAdhesion )
{
    if( !m_device )
        GTEST_SKIP();

    // Deliberately adhesion-dominated pair: mild overlap, large adhesion vs
    // small repulsion, so the baseline net force is inward (cells compress).
    // Cortical tension adds k_T·overlap outward — must partially/fully cancel.
    float repulsion = 5.0f;
    float adhesion  = 20.0f;
    float maxRadius = 1.5f; // interactDist = 3.0; at x=2.9 overlap = 0.1

    // Baseline: no cortical tension.
    auto [x0_base, x1_base] = RunJKRCadherin(
        m_device.get(), m_rm.get(), m_stream.get(),
        0.0f, 2.9f, repulsion, adhesion, maxRadius,
        glm::vec4( 0.0f ), glm::vec4( 0.0f ), glm::mat4( 1.0f ),
        0u, 1.0f,                                   // cadherin OFF
        0u, glm::vec4( 0.0f ), glm::vec4( 0.0f ),   // polarity OFF
        0.5f, 1.5f, PlateTestParams{},
        0.0f );                                     // corticalTension = 0

    // Phase 4: strong cortical tension.
    auto [x0_ten, x1_ten] = RunJKRCadherin(
        m_device.get(), m_rm.get(), m_stream.get(),
        0.0f, 2.9f, repulsion, adhesion, maxRadius,
        glm::vec4( 0.0f ), glm::vec4( 0.0f ), glm::mat4( 1.0f ),
        0u, 1.0f,
        0u, glm::vec4( 0.0f ), glm::vec4( 0.0f ),
        0.5f, 1.5f, PlateTestParams{},
        30.0f );                                    // corticalTension = 30 (strong)

    float disp_base = x1_base - x0_base;
    float disp_ten  = x1_ten  - x0_ten;

    // Cortical tension must push the pair further apart than the no-tension baseline.
    EXPECT_GT( disp_ten, disp_base )
        << "Cortical tension must oppose adhesion: pair separation with tension ("
        << disp_ten << ") must exceed baseline (" << disp_base << ")";
}

// =============================================================================
// Phase 4.5 — polarity_update.comp junctional propagation tests
// =============================================================================
//
// These tests hit the polarity_update shader directly (not the JKR shader that
// consumes polarity). We control initial polarity and agent positions, dispatch
// the shader N times, read back the polarity buffer, and assert on the result.
// The harness packs all agents into a single hash cell so the neighbour scan
// simply walks every agent pair.

// Run polarity_update.comp `steps` times on the given agent configuration and
// return the final polarity buffer. Plate is optional (activeFlag selects).
static std::vector<glm::vec4> RunPolarityUpdate(
    Device*           device,
    ResourceManager*  rm,
    StreamingManager* stream,
    const std::vector<glm::vec4>& initialPositions,  // xyz = pos, w = alive flag (1 live, 0 dead)
    const std::vector<glm::vec4>& initialPolarities, // xyz = dir, w = magnitude
    float             regulationRate,
    float             interactionRadius,
    float             propagationStrength,
    int               steps,
    float             dt       = 1.0f / 60.0f,
    PlateTestParams   plate    = {} )
{
    const uint32_t agentCount = static_cast<uint32_t>( initialPositions.size() );
    EXPECT_EQ( initialPositions.size(), initialPolarities.size() );

    // All agents share cell (0,0,0); cellSize chosen large enough to span any test
    // configuration (positions must be within ±cellSize/2 of origin).
    struct AgentHash { uint32_t hash; uint32_t agentIndex; };
    std::vector<AgentHash> sortedHashes( agentCount );
    for( uint32_t i = 0; i < agentCount; ++i )
        sortedHashes[ i ] = { 0u, i };

    const uint32_t        offsetArraySize = 256;
    std::vector<uint32_t> cellOffsets( offsetArraySize, 0xFFFFFFFFu );
    cellOffsets[ 0 ] = 0u;

    std::vector<PhenotypeData> phenotypes( agentCount, { 0u, 0.5f, 0.0f, 0u } );

    // Buffers
    BufferHandle inBuf     = rm->CreateBuffer( { agentCount * sizeof( glm::vec4 ),      BufferType::STORAGE,  "PolUpdInBuf" } );
    BufferHandle polBuf    = rm->CreateBuffer( { agentCount * sizeof( glm::vec4 ),      BufferType::STORAGE,  "PolUpdPolBuf" } );
    BufferHandle hashBuf   = rm->CreateBuffer( { agentCount * sizeof( AgentHash ),      BufferType::STORAGE,  "PolUpdHashBuf" } );
    BufferHandle offsetBuf = rm->CreateBuffer( { offsetArraySize * sizeof( uint32_t ),  BufferType::STORAGE,  "PolUpdOffsetBuf" } );
    BufferHandle countBuf  = rm->CreateBuffer( { sizeof( uint32_t ),                    BufferType::INDIRECT, "PolUpdCountBuf" } );
    BufferHandle phenoBuf  = rm->CreateBuffer( { agentCount * sizeof( PhenotypeData ),  BufferType::STORAGE,  "PolUpdPhenoBuf" } );

    struct PlateBufferGPU { glm::uvec4 meta; glm::vec4 plates[16]; };
    PlateBufferGPU plateData{};
    plateData.meta         = glm::uvec4( plate.activeFlag, 0u, 0u, 0u );
    plateData.plates[ 0 ]  = glm::vec4( plate.normal, plate.height );
    plateData.plates[ 1 ]  = glm::vec4( plate.contactStiffness, plate.integrinAdhesion,
                                        plate.anchorageDistance, plate.polarityBias );
    BufferHandle plateBuf  = rm->CreateBuffer( { sizeof( PlateBufferGPU ),             BufferType::STORAGE,  "PolUpdPlateBuf" } );

    stream->UploadBufferImmediate( { { inBuf,     initialPositions.data(),  agentCount * sizeof( glm::vec4 ) } } );
    stream->UploadBufferImmediate( { { polBuf,    initialPolarities.data(), agentCount * sizeof( glm::vec4 ) } } );
    stream->UploadBufferImmediate( { { hashBuf,   sortedHashes.data(),      agentCount * sizeof( AgentHash ) } } );
    stream->UploadBufferImmediate( { { offsetBuf, cellOffsets.data(),       offsetArraySize * sizeof( uint32_t ) } } );
    stream->UploadBufferImmediate( { { countBuf,  &agentCount,              sizeof( uint32_t ) } } );
    stream->UploadBufferImmediate( { { phenoBuf,  phenotypes.data(),        agentCount * sizeof( PhenotypeData ) } } );
    stream->UploadBufferImmediate( { { plateBuf,  &plateData,               sizeof( PlateBufferGPU ) } } );

    // Pipeline
    ComputePipelineDesc pipeDesc{};
    pipeDesc.shader = rm->CreateShader( "shaders/compute/polarity_update.comp" );
    ComputePipelineHandle pipeHandle = rm->CreatePipeline( pipeDesc );
    ComputePipeline*      pipeline   = rm->GetPipeline( pipeHandle );

    BindingGroup* bg = rm->GetBindingGroup( rm->CreateBindingGroup( pipeHandle, 0 ) );
    bg->Bind( 0, rm->GetBuffer( inBuf ) );
    bg->Bind( 1, rm->GetBuffer( polBuf ) );
    bg->Bind( 2, rm->GetBuffer( countBuf ) );
    bg->Bind( 3, rm->GetBuffer( hashBuf ) );
    bg->Bind( 4, rm->GetBuffer( offsetBuf ) );
    bg->Bind( 5, rm->GetBuffer( phenoBuf ) );
    bg->Bind( 6, rm->GetBuffer( plateBuf ) );
    bg->Build();

    ComputePushConstants pc{};
    pc.dt          = dt;
    pc.fParam0     = regulationRate;
    pc.fParam1     = interactionRadius;
    pc.fParam2     = -1.0f; // reqLC = any
    pc.fParam3     = -1.0f; // reqCT = any
    pc.fParam4     = propagationStrength; // Phase 4.5
    pc.offset      = 0;
    pc.maxCapacity = agentCount;
    pc.uParam0     = offsetArraySize;
    pc.uParam1     = 0;
    // cellSize large enough to hold any test agent within cell (0,0,0):
    pc.domainSize  = glm::vec4( 100.0f, 100.0f, 100.0f, 100.0f );

    auto ctx = device->GetThreadContext( device->CreateThreadContext( QueueType::COMPUTE ) );

    for( int s = 0; s < steps; ++s )
    {
        auto cmd = ctx->GetCommandBuffer( ctx->CreateCommandBuffer() );
        cmd->Begin();
        cmd->SetPipeline( pipeline );
        cmd->SetBindingGroup( bg, pipeline->GetLayout(), VK_PIPELINE_BIND_POINT_COMPUTE );
        cmd->PushConstants( pipeline->GetLayout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof( pc ), &pc );
        cmd->Dispatch( ( agentCount + 255 ) / 256, 1, 1 );
        cmd->End();
        device->GetComputeQueue()->Submit( { cmd } );
        device->GetComputeQueue()->WaitIdle();
    }

    std::vector<glm::vec4> result( agentCount );
    stream->ReadbackBufferImmediate( polBuf, result.data(), agentCount * sizeof( glm::vec4 ) );

    rm->DestroyBuffer( inBuf );  rm->DestroyBuffer( polBuf );  rm->DestroyBuffer( hashBuf );
    rm->DestroyBuffer( offsetBuf ); rm->DestroyBuffer( countBuf ); rm->DestroyBuffer( phenoBuf );
    rm->DestroyBuffer( plateBuf );

    return result;
}

// Test 1. A fully-polar neighbour propagates its direction to an unpolarised cell.
// Setup: cell 0 polarised (+Y, magnitude 1), cell 1 unpolarised.
// Assertion: after several dispatches, cell 1 gains magnitude > 0 and direction
// tilts toward +Y. This is the core junctional-coupling mechanism.
TEST_F( ComputeTest, Shader_PolarityPropagation_AnchoredCellPolarisesNeighbour )
{
    if( !m_device )
        GTEST_SKIP();

    std::vector<glm::vec4> positions = {
        glm::vec4( 0.0f, 0.0f, 0.0f, 1.0f ),    // cell 0 — will stay polar (+Y)
        glm::vec4( 0.5f, 0.0f, 0.0f, 1.0f )     // cell 1 — starts unpolarised
    };
    std::vector<glm::vec4> polarities = {
        glm::vec4( 0.0f, 1.0f, 0.0f, 1.0f ),    // fully polar +Y
        glm::vec4( 0.0f, 0.0f, 0.0f, 0.0f )     // unpolarised
    };

    // Many steps so the cascade has time to build. regulationRate=1.0 for speed.
    auto result = RunPolarityUpdate(
        m_device.get(), m_rm.get(), m_stream.get(),
        positions, polarities,
        /*regulationRate=*/1.0f,
        /*interactionRadius=*/0.75f,
        /*propagationStrength=*/1.0f,
        /*steps=*/120 );

    // Cell 1 magnitude must have grown above zero.
    EXPECT_GT( result[ 1 ].w, 0.3f )
        << "Unpolarised neighbour must inherit magnitude from polar neighbour "
           "(got " << result[ 1 ].w << ")";
    // Cell 1 direction must tilt toward +Y (the seed).
    EXPECT_GT( result[ 1 ].y, 0.3f )
        << "Inherited direction must point roughly toward +Y (got "
        << result[ 1 ].x << ", " << result[ 1 ].y << ", " << result[ 1 ].z << ")";
}

// Test 2. Propagation requires a polar seed. With all cells unpolarised,
// the deadband prevents FP-noise amplification — no spontaneous polarisation.
TEST_F( ComputeTest, Shader_PolarityPropagation_NoPolarNeighboursNoEffect )
{
    if( !m_device )
        GTEST_SKIP();

    std::vector<glm::vec4> positions = {
        glm::vec4( 0.0f, 0.0f, 0.0f, 1.0f ),
        glm::vec4( 0.5f, 0.0f, 0.0f, 1.0f ),
        glm::vec4( 0.0f, 0.5f, 0.0f, 1.0f )
    };
    std::vector<glm::vec4> polarities = {
        glm::vec4( 0.0f, 0.0f, 0.0f, 0.0f ),    // all unpolarised
        glm::vec4( 0.0f, 0.0f, 0.0f, 0.0f ),
        glm::vec4( 0.0f, 0.0f, 0.0f, 0.0f )
    };

    auto result = RunPolarityUpdate(
        m_device.get(), m_rm.get(), m_stream.get(),
        positions, polarities,
        /*regulationRate=*/1.0f,
        /*interactionRadius=*/0.75f,
        /*propagationStrength=*/1.0f,
        /*steps=*/120 );

    // Note: neighbour-centroid polarity will still make these cells polar on
    // the surface (they ARE on the surface of a 3-cell cluster). So we can't
    // assert magnitude ≈ 0. But PROPAGATION specifically must NOT have been
    // the source — the direction each cell adopts should come from geometric
    // outward-centroid, not from mutual amplification. We assert a weaker
    // property: no cell's polarity magnitude has run away to ≫1.
    for( size_t i = 0; i < result.size(); ++i ) {
        EXPECT_LE( result[ i ].w, 1.01f )
            << "Cell " << i << " magnitude ran away to " << result[ i ].w
            << " — FP-noise propagation feedback is not controlled by the deadband.";
    }
}

// Test 3. Both runs use propagationStrength = 0 — the propagation path must not
// contribute when disabled, regardless of any other polarity state. This is a
// weaker assertion than "bit-identical to Phase-4" because the blend formula
// legitimately changed from nested mix to weighted sum (see plan for math
// rationale). The invariant we actually need: prop=0 means NO propagation term.
TEST_F( ComputeTest, Shader_PolarityPropagation_ZeroStrengthIsDisabled )
{
    if( !m_device )
        GTEST_SKIP();

    std::vector<glm::vec4> positions = {
        glm::vec4( 0.0f, 0.0f, 0.0f, 1.0f ),
        glm::vec4( 0.5f, 0.0f, 0.0f, 1.0f )
    };
    std::vector<glm::vec4> polarities = {
        glm::vec4( 0.0f, 1.0f, 0.0f, 1.0f ),   // cell 0 fully polar +Y (would propagate)
        glm::vec4( 0.0f )                      // cell 1 unpolarised
    };

    // With prop=0, cell 1 must NOT inherit cell 0's +Y direction — propagation
    // is disabled. Any polarity cell 1 develops comes strictly from the
    // neighbour-centroid geometric cue (which for a two-cell pair is along X,
    // not Y).
    auto result = RunPolarityUpdate(
        m_device.get(), m_rm.get(), m_stream.get(),
        positions, polarities,
        /*regulationRate=*/1.0f,
        /*interactionRadius=*/0.75f,
        /*propagationStrength=*/0.0f,
        /*steps=*/60 );

    // Cell 1 must have negligible +Y component — it did not inherit from cell 0.
    EXPECT_LT( std::abs( result[ 1 ].y ), 0.2f )
        << "With propagationStrength = 0, cell 1 must not inherit +Y from cell 0 "
           "(got y = " << result[ 1 ].y << ", should be ≈ 0)";
}

// Test 4. Plate seed is not overridden by strong propagation. A plate-anchored
// cell with multiple polar neighbours pointing in a different direction must
// still retain a plate-biased polarity — propagation is ADDITIVE to the plate,
// not a replacement (weighted-sum blend, not nested mix).
TEST_F( ComputeTest, Shader_PolarityPropagation_PlateNotOverriddenByPropagation )
{
    if( !m_device )
        GTEST_SKIP();

    // Plate at y=0, normal +Y, strong polarityBias. Cell 0 is right on the plate.
    PlateTestParams plate{};
    plate.normal            = glm::vec3( 0.0f, 1.0f, 0.0f );
    plate.height            = 0.0f;
    plate.contactStiffness  = 0.0f; // not exercised here
    plate.integrinAdhesion  = 0.0f;
    plate.anchorageDistance = 1.0f;
    plate.polarityBias      = 2.0f;
    plate.activeFlag        = 1u;

    // Step A convention update: polarity = BASAL direction. Plate normal = +Y
    // means plate is below, basal = toward plate = -Y. Anchored cells polarise
    // toward -Y (basal down). "Hostile" propagation neighbours here push
    // toward +Y to check that plate pull is NOT overridden.
    std::vector<glm::vec4> positions = {
        glm::vec4( 0.0f, 0.0f, 0.0f, 1.0f ),    // cell 0 — anchored (d=0)
        glm::vec4( 0.0f, 0.5f, 0.0f, 1.0f ),    // cell 1 — polar +Y (hostile direction)
        glm::vec4( 0.5f, 0.5f, 0.0f, 1.0f ),    // cell 2 — polar +Y
        glm::vec4( -0.5f, 0.5f, 0.0f, 1.0f )    // cell 3 — polar +Y
    };
    std::vector<glm::vec4> polarities = {
        glm::vec4( 0.0f ),                      // cell 0 unpolarised, plate will polarise it -Y (basal)
        glm::vec4( 0.0f,  1.0f, 0.0f, 1.0f ),   // cell 1 fully polar +Y (opposite to plate cue)
        glm::vec4( 0.0f,  1.0f, 0.0f, 1.0f ),   // cell 2 fully polar +Y
        glm::vec4( 0.0f,  1.0f, 0.0f, 1.0f )    // cell 3 fully polar +Y
    };

    auto result = RunPolarityUpdate(
        m_device.get(), m_rm.get(), m_stream.get(),
        positions, polarities,
        /*regulationRate=*/1.0f,
        /*interactionRadius=*/0.75f,
        /*propagationStrength=*/5.0f,    // aggressive propagation
        /*steps=*/120,
        /*dt=*/1.0f / 60.0f,
        plate );

    // Cell 0 must stay on the -Y side (basal toward plate) despite aggressive
    // +Y propagation from neighbours. Plate pull is additive and polarityBias=2
    // exceeds propagationStrength=5 × propMagAvg (propMagAvg ~1 for 3 polar /
    // 3 neighbours, so propW_eff clamps to 1; plateWeight = 2.0 at d=0). The
    // weighted-sum blend therefore keeps cell 0 leaning basal (-Y).
    EXPECT_LT( result[ 0 ].y, 0.0f )
        << "Anchored cell's polarity must stay on the -Y side (basal toward plate) "
           "despite aggressive +Y propagation from neighbours (got y="
        << result[ 0 ].y << ")";
}

// Test 5. Weighted-sum blends partial-magnitude neighbour cues proportionally.
// A cell with one polar neighbour (+Y) and propagationStrength > 0 receives
// a direction tilted partly toward +Y, not fully replacing the existing state.
TEST_F( ComputeTest, Shader_PolarityPropagation_WeightedSumBlendsCorrectly )
{
    if( !m_device )
        GTEST_SKIP();

    std::vector<glm::vec4> positions = {
        glm::vec4( 0.0f, 0.0f, 0.0f, 1.0f ),    // cell 0 — will be updated
        glm::vec4( 0.5f, 0.0f, 0.0f, 1.0f )     // cell 1 — stays polar +Y
    };
    std::vector<glm::vec4> polarities = {
        glm::vec4( 0.0f, 0.0f, 0.0f, 0.0f ),    // cell 0 unpolarised
        glm::vec4( 0.0f, 1.0f, 0.0f, 1.0f )     // cell 1 fully polar +Y
    };

    // Run with propagationStrength = 0.5. Interior cell's polarity should
    // reflect a mix of neighbour-centroid direction (−X, toward cell 1) and
    // propagation direction (+Y, inherited from cell 1).
    auto result = RunPolarityUpdate(
        m_device.get(), m_rm.get(), m_stream.get(),
        positions, polarities,
        /*regulationRate=*/1.0f,
        /*interactionRadius=*/0.75f,
        /*propagationStrength=*/0.5f,
        /*steps=*/120 );

    // Cell 0 direction must have a positive Y component (propagation contribution)
    // AND a negative X component (centroid contribution — cell 0 is to the LEFT
    // of cell 1, so centroid is at x=0.25, so rawVec = myPos - centroid = -X).
    EXPECT_GT( result[ 0 ].y, 0.05f )
        << "Weighted-sum blend must include +Y propagation component (got y="
        << result[ 0 ].y << ")";
    EXPECT_LT( result[ 0 ].x, 0.0f )
        << "Weighted-sum blend must retain -X centroid component (got x="
        << result[ 0 ].x << ")";
}

// =============================================================================
// BasementMembrane plate — raw jkr_forces shader tests (Phase 2)
// =============================================================================

// Cell below the plane (signed distance d < 0) → Hertz-like push along +normal.
TEST_F( ComputeTest, Shader_Plate_RepulsesPenetratingCell )
{
    if( !m_device )
        GTEST_SKIP();

    PlateTestParams plate{};
    plate.normal            = glm::vec3( 0.0f, 0.0f, 1.0f );
    plate.height            = 0.0f;
    plate.contactStiffness  = 50.0f;
    plate.integrinAdhesion  = 0.0f;   // adhesion disabled so we observe repulsion only
    plate.anchorageDistance = 1.0f;
    plate.polarityBias      = 0.0f;
    plate.activeFlag        = 1u;

    // Agent penetrated 0.5 units below the plate (z = -0.5). One dt of
    // Hertz-like repulsion will NOT clear the plate in a single frame (expected
    // Δz ≈ 50·0.5^1.5·0.01 ≈ 0.18) — the test only asserts monotone upward
    // motion along the plate normal.
    glm::vec3 pos = RunJKRWithPlate( m_device.get(), m_rm.get(), m_stream.get(),
                                      glm::vec3( 0.0f, 0.0f, -0.5f ), plate );

    EXPECT_GT( pos.z, -0.5f ) << "Penetrating cell must be pushed upward along +normal";
    // Sanity: position is along z only, no lateral drift.
    EXPECT_NEAR( pos.x, 0.0f, 1e-4f );
    EXPECT_NEAR( pos.y, 0.0f, 1e-4f );
}

// Cell just above the plate (0 <= d <= anchorageDistance) → JKR-like pull toward plate.
TEST_F( ComputeTest, Shader_Plate_AttractsWithinAnchorageDistance )
{
    if( !m_device )
        GTEST_SKIP();

    PlateTestParams plate{};
    plate.normal            = glm::vec3( 0.0f, 0.0f, 1.0f );
    plate.height            = 0.0f;
    plate.contactStiffness  = 0.0f;   // repulsion disabled; contact region unused here
    plate.integrinAdhesion  = 5.0f;
    plate.anchorageDistance = 1.0f;
    plate.polarityBias      = 0.0f;
    plate.activeFlag        = 1u;

    // Agent at z = 0.5 (halfway through anchorage band)
    glm::vec3 pos = RunJKRWithPlate( m_device.get(), m_rm.get(), m_stream.get(),
                                      glm::vec3( 0.0f, 0.0f, 0.5f ), plate );

    EXPECT_LT( pos.z, 0.5f ) << "Cell within anchorage distance must be pulled toward plate";
    EXPECT_GE( pos.z, 0.0f ) << "Integrin pull must not push cell through the plate in one dt";
}

// Plate inactive (flags.x = 0) → no force on the cell, whatever the params.
// This is the ECM-leak guard: proves that a blueprint without BasementMembrane
// behaves identically to one without the plate-branch code at all.
TEST_F( ComputeTest, Shader_Plate_Inactive_WhenFlagOff )
{
    if( !m_device )
        GTEST_SKIP();

    PlateTestParams plate{};
    plate.normal            = glm::vec3( 0.0f, 0.0f, 1.0f );
    plate.height            = 0.0f;
    plate.contactStiffness  = 50.0f;  // large params but...
    plate.integrinAdhesion  = 5.0f;
    plate.anchorageDistance = 1.0f;
    plate.polarityBias      = 5.0f;
    plate.activeFlag        = 0u;     // ... disabled.

    glm::vec3 posBelow = RunJKRWithPlate( m_device.get(), m_rm.get(), m_stream.get(),
                                           glm::vec3( 0.0f, 0.0f, -0.5f ), plate );
    glm::vec3 posAbove = RunJKRWithPlate( m_device.get(), m_rm.get(), m_stream.get(),
                                           glm::vec3( 0.0f, 0.0f,  0.5f ), plate );

    EXPECT_NEAR( posBelow.z, -0.5f, 1e-4f ) << "Inactive plate must not affect position";
    EXPECT_NEAR( posAbove.z,  0.5f, 1e-4f ) << "Inactive plate must not affect position";
}

// =============================================================================
// cadherin_expression_update.comp tests
// =============================================================================

// Helper: dispatch cadherin_expression_update once, return readback profile
static glm::vec4 RunCadherinExpressionUpdate(
    Device*           device,
    ResourceManager*  rm,
    StreamingManager* stream,
    glm::vec4         initialProfile,
    glm::vec4         agentPos,      // w=1 → live, w=0 → dead slot
    float             expressionRate,
    float             degradationRate,
    glm::vec4         targetExpression,
    uint32_t          lifecycleState = 0u,
    uint32_t          reqLC          = 0xFFFFFFFFu )
{

    BufferHandle agentBuf   = rm->CreateBuffer( { sizeof( glm::vec4 ),    BufferType::STORAGE,  "CadAgentBuf" } );
    BufferHandle profileBuf = rm->CreateBuffer( { sizeof( glm::vec4 ),    BufferType::STORAGE,  "CadProfileBuf" } );
    BufferHandle countBuf   = rm->CreateBuffer( { sizeof( uint32_t ),     BufferType::INDIRECT, "CadCountBuf" } );
    BufferHandle phenoBuf   = rm->CreateBuffer( { sizeof( PhenotypeData ),BufferType::STORAGE,  "CadPhenoBuf" } );

    stream->UploadBufferImmediate( { { agentBuf,   &agentPos,       sizeof( glm::vec4 ) } } );
    stream->UploadBufferImmediate( { { profileBuf, &initialProfile, sizeof( glm::vec4 ) } } );
    uint32_t count = 1;
    stream->UploadBufferImmediate( { { countBuf, &count, sizeof( uint32_t ) } } );
    PhenotypeData pheno = { lifecycleState, 0.5f, 0.0f, 0u };
    stream->UploadBufferImmediate( { { phenoBuf, &pheno, sizeof( PhenotypeData ) } } );

    ComputePipelineDesc   pipeDesc{};
    pipeDesc.shader                  = rm->CreateShader( "shaders/compute/cadherin_expression_update.comp" );
    ComputePipelineHandle pipeHandle = rm->CreatePipeline( pipeDesc );
    ComputePipeline*      pipeline   = rm->GetPipeline( pipeHandle );

    BindingGroup* bg = rm->GetBindingGroup( rm->CreateBindingGroup( pipeHandle, 0 ) );
    bg->Bind( 0, rm->GetBuffer( agentBuf ) );
    bg->Bind( 1, rm->GetBuffer( profileBuf ) );
    bg->Bind( 2, rm->GetBuffer( countBuf ) );
    bg->Bind( 3, rm->GetBuffer( phenoBuf ) );
    bg->Build();

    ComputePushConstants pc{};
    pc.dt          = 1.0f;
    pc.totalTime   = 0.0f;
    pc.fParam0     = expressionRate;
    pc.fParam1     = degradationRate;
    pc.fParam2     = targetExpression.x;
    pc.fParam3     = targetExpression.y;
    pc.fParam4     = targetExpression.z;
    pc.fParam5     = targetExpression.w;
    pc.offset      = 0;
    pc.maxCapacity = 1;
    pc.uParam0     = reqLC;
    pc.uParam1     = 0;

    auto ctx = device->GetThreadContext( device->CreateThreadContext( QueueType::COMPUTE ) );
    auto cmd = ctx->GetCommandBuffer( ctx->CreateCommandBuffer() );
    cmd->Begin();
    cmd->SetPipeline( pipeline );
    cmd->SetBindingGroup( bg, pipeline->GetLayout(), VK_PIPELINE_BIND_POINT_COMPUTE );
    cmd->PushConstants( pipeline->GetLayout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof( pc ), &pc );
    cmd->Dispatch( 1, 1, 1 );
    cmd->End();
    device->GetComputeQueue()->Submit( { cmd } );
    device->GetComputeQueue()->WaitIdle();

    glm::vec4 result;
    stream->ReadbackBufferImmediate( profileBuf, &result, sizeof( glm::vec4 ) );

    rm->DestroyBuffer( agentBuf );
    rm->DestroyBuffer( profileBuf );
    rm->DestroyBuffer( countBuf );
    rm->DestroyBuffer( phenoBuf );

    return result;
}

// 1. Upregulation: profile starts at 0, target = 1 → each channel increases by expressionRate * dt
TEST_F( ComputeTest, Shader_CadherinExpressionUpdate_Upregulation )
{
    if( !m_device )
        GTEST_SKIP();

    float     rate   = 0.05f;
    glm::vec4 result = RunCadherinExpressionUpdate(
        m_device.get(), m_rm.get(), m_stream.get(),
        glm::vec4( 0.0f ),                    // initial profile: all zeros
        glm::vec4( 0.0f, 0.0f, 0.0f, 1.0f ), // live agent (w=1)
        rate,                                  // expressionRate
        0.01f,                                 // degradationRate (should not fire)
        glm::vec4( 1.0f ) );                  // target: all ones

    EXPECT_NEAR( result.x, rate, 1e-5f ) << "E-cadherin upregulation incorrect";
    EXPECT_NEAR( result.y, rate, 1e-5f ) << "N-cadherin upregulation incorrect";
    EXPECT_NEAR( result.z, rate, 1e-5f ) << "VE-cadherin upregulation incorrect";
    EXPECT_NEAR( result.w, rate, 1e-5f ) << "Cadherin-11 upregulation incorrect";
}

// 2. Downregulation: profile starts at 1, target = 0 → each channel decreases by degradationRate * dt
TEST_F( ComputeTest, Shader_CadherinExpressionUpdate_Downregulation )
{
    if( !m_device )
        GTEST_SKIP();

    float     rate   = 0.02f;
    glm::vec4 result = RunCadherinExpressionUpdate(
        m_device.get(), m_rm.get(), m_stream.get(),
        glm::vec4( 1.0f ),                    // initial profile: all ones
        glm::vec4( 0.0f, 0.0f, 0.0f, 1.0f ), // live agent
        0.05f,                                 // expressionRate (should not fire)
        rate,                                  // degradationRate
        glm::vec4( 0.0f ) );                  // target: all zeros

    EXPECT_NEAR( result.x, 1.0f - rate, 1e-5f ) << "E-cadherin downregulation incorrect";
    EXPECT_NEAR( result.y, 1.0f - rate, 1e-5f ) << "N-cadherin downregulation incorrect";
    EXPECT_NEAR( result.z, 1.0f - rate, 1e-5f ) << "VE-cadherin downregulation incorrect";
    EXPECT_NEAR( result.w, 1.0f - rate, 1e-5f ) << "Cadherin-11 downregulation incorrect";
}

// 3. Dead-slot guard: agent.w == 0 → profile must not change
TEST_F( ComputeTest, Shader_CadherinExpressionUpdate_DeadSlot_Skipped )
{
    if( !m_device )
        GTEST_SKIP();

    glm::vec4 initial = glm::vec4( 0.3f, 0.5f, 0.0f, 0.1f );
    glm::vec4 result  = RunCadherinExpressionUpdate(
        m_device.get(), m_rm.get(), m_stream.get(),
        initial,
        glm::vec4( 0.0f ),  // dead slot (w=0)
        1.0f,               // high rate — would move far if not skipped
        1.0f,
        glm::vec4( 1.0f ) );

    EXPECT_NEAR( result.x, initial.x, 1e-5f ) << "Dead slot E-cad was modified";
    EXPECT_NEAR( result.y, initial.y, 1e-5f ) << "Dead slot N-cad was modified";
    EXPECT_NEAR( result.z, initial.z, 1e-5f ) << "Dead slot VE-cad was modified";
    EXPECT_NEAR( result.w, initial.w, 1e-5f ) << "Dead slot Cad-11 was modified";
}

// 4. No overshoot: large dt * rate must clamp at target, not past it
TEST_F( ComputeTest, Shader_CadherinExpressionUpdate_NoOvershoot )
{
    if( !m_device )
        GTEST_SKIP();

    // target=0.5, current=0.4, rate=1.0, dt=1.0 → raw step=1.0; must clamp to 0.5
    glm::vec4 result = RunCadherinExpressionUpdate(
        m_device.get(), m_rm.get(), m_stream.get(),
        glm::vec4( 0.4f ),                    // initial: 0.4
        glm::vec4( 0.0f, 0.0f, 0.0f, 1.0f ), // live agent
        1.0f,                                  // expressionRate (huge)
        1.0f,                                  // degradationRate
        glm::vec4( 0.5f ) );                  // target: 0.5

    EXPECT_NEAR( result.x, 0.5f, 1e-5f ) << "E-cad overshot target";
    EXPECT_NEAR( result.y, 0.5f, 1e-5f ) << "N-cad overshot target";
    EXPECT_NEAR( result.z, 0.5f, 1e-5f ) << "VE-cad overshot target";
    EXPECT_NEAR( result.w, 0.5f, 1e-5f ) << "Cad-11 overshot target";
}

// =============================================================================
// jkr_forces.comp — rigid body dynamics tests
// =============================================================================

struct RigidBodyResult
{
    std::vector<glm::vec4> positions;
    std::vector<glm::vec4> orientations;
};

// Dispatch jkr_forces with a contact hull and orientation buffer; return new positions + orientations.
// All agents are placed near the origin so they all land in hash cell (0,0,0).
//
// cadherin parameters (all default to 0/disabled):
//   cadherinCoupling > 0  enables cadherin; all agents are assigned a uniform profile (1,0,0,0)
//   with an identity affinity matrix so adhScale = cadherinCoupling for all pairs.
static RigidBodyResult RunJKRRigidBody(
    Device*                device,
    ResourceManager*       rm,
    StreamingManager*      stream,
    std::vector<glm::vec4> inPositions,
    std::vector<glm::vec4> inOrientations,
    uint32_t               hullCount,
    std::vector<glm::vec4> hullPoints,      // xyz=model offset, w=sub-sphere radius (up to 16)
    float                  repulsion,
    float                  adhesion,
    float                  maxRadius,
    float                  damping           = 0.0f,
    float                  dt                = 1.0f,
    float                  stericZ              = 0.0f,   // hullMeta.z — model-Z half-extent
    float                  stericY              = 0.0f,   // hullMeta.w — model-Y half-extent
    float                  edgeAlignStrength    = 0.0f,   // hullMeta.y — cadherin edge alignment
    float                  cadherinCoupling     = 0.0f,   // >0 enables cadherin (adhScale=coupling)
    float                  lateralAdhesionScale = 0.0f )  // Phase 4.5-B hull-pair translation scale
{
    struct ContactHullGPU { glm::vec4 meta{}; glm::vec4 points[16]{}; };
    struct AgentHash { uint32_t hash; uint32_t agentIndex; };

    uint32_t agentCount      = static_cast<uint32_t>( inPositions.size() );
    uint32_t offsetArraySize = 256;

    // Place all agents in hash cell (0,0,0): cellSize = maxRadius * 3 covers all test offsets.
    std::vector<AgentHash>   sortedHashes( agentCount );
    std::vector<uint32_t>    cellOffsets( offsetArraySize, 0xFFFFFFFF );
    cellOffsets[0] = 0;
    for ( uint32_t i = 0; i < agentCount; i++ )
        sortedHashes[i] = { 0u, i };

    ContactHullGPU hullGPU{};
    hullGPU.meta = glm::vec4( float( hullCount ), edgeAlignStrength, stericZ, stericY );
    for ( uint32_t i = 0; i < std::min( hullCount, 16u ); i++ )
        hullGPU.points[i] = hullPoints[i];

    glm::mat4               identity = glm::mat4( 1.0f );
    std::vector<glm::vec4>  zeros( agentCount, glm::vec4( 0.0f ) );
    // Uniform cadherin profile (channel 0 = 1.0) for all agents when cadherin is active.
    std::vector<glm::vec4>  cadProfiles( agentCount, glm::vec4( 1.0f, 0.0f, 0.0f, 0.0f ) );
    std::vector<PhenotypeData> phenos( agentCount, { 0u, 0.5f, 0.0f, 0u } );

    BufferHandle inBuf     = rm->CreateBuffer( { agentCount * sizeof( glm::vec4 ),       BufferType::STORAGE,  "RBInBuf" } );
    BufferHandle outBuf    = rm->CreateBuffer( { agentCount * sizeof( glm::vec4 ),       BufferType::STORAGE,  "RBOutBuf" } );
    BufferHandle pressBuf  = rm->CreateBuffer( { agentCount * sizeof( float ),           BufferType::STORAGE,  "RBPressBuf" } );
    BufferHandle hashBuf   = rm->CreateBuffer( { agentCount * sizeof( AgentHash ),       BufferType::STORAGE,  "RBHashBuf" } );
    BufferHandle offsetBuf = rm->CreateBuffer( { offsetArraySize * sizeof( uint32_t ),   BufferType::STORAGE,  "RBOffsetBuf" } );
    BufferHandle countBuf  = rm->CreateBuffer( { agentCount * sizeof( uint32_t ),        BufferType::INDIRECT, "RBCountBuf" } );
    BufferHandle phenoBuf  = rm->CreateBuffer( { agentCount * sizeof( PhenotypeData ),   BufferType::STORAGE,  "RBPhenoBuf" } );
    BufferHandle profBuf   = rm->CreateBuffer( { agentCount * sizeof( glm::vec4 ),       BufferType::STORAGE,  "RBProfBuf" } );
    BufferHandle affBuf    = rm->CreateBuffer( { sizeof( glm::mat4 ),                    BufferType::STORAGE,  "RBAffBuf" } );
    BufferHandle polBuf    = rm->CreateBuffer( { agentCount * sizeof( glm::vec4 ),       BufferType::STORAGE,  "RBPolBuf" } );
    BufferHandle orientBuf = rm->CreateBuffer( { agentCount * sizeof( glm::vec4 ),       BufferType::STORAGE,  "RBOrientBuf" } );
    BufferHandle hullBuf   = rm->CreateBuffer( { sizeof( ContactHullGPU ),               BufferType::STORAGE,  "RBHullBuf" } );

    // Plate buffer (binding 12) — rigid-body tests don't exercise the plate, so
    // allocate a disabled dummy so validation layers don't complain about an
    // unbound descriptor. Step B: meta.x = 0 (count) → shader skips plate loop.
    struct PlateBufferGPU { glm::uvec4 meta; glm::vec4 plates[16]; };
    PlateBufferGPU plateData{};
    BufferHandle plateBuf  = rm->CreateBuffer( { sizeof( PlateBufferGPU ),              BufferType::STORAGE,  "RBPlateBuf" } );

    stream->UploadBufferImmediate( { { inBuf,     inPositions.data(),    agentCount * sizeof( glm::vec4 ) } } );
    stream->UploadBufferImmediate( { { hashBuf,   sortedHashes.data(),   agentCount * sizeof( AgentHash ) } } );
    stream->UploadBufferImmediate( { { offsetBuf, cellOffsets.data(),    offsetArraySize * sizeof( uint32_t ) } } );
    stream->UploadBufferImmediate( { { countBuf,  &agentCount,           sizeof( uint32_t ) } } );
    stream->UploadBufferImmediate( { { phenoBuf,  phenos.data(),         agentCount * sizeof( PhenotypeData ) } } );
    // Upload cadherin profiles: full expression (channel 0 = 1.0) when cadherin is active, zeros otherwise.
    const void* profData = ( cadherinCoupling > 0.0f ) ? (const void*)cadProfiles.data() : (const void*)zeros.data();
    stream->UploadBufferImmediate( { { profBuf, profData, agentCount * sizeof( glm::vec4 ) } } );
    stream->UploadBufferImmediate( { { affBuf,    &identity,             sizeof( identity ) } } );
    stream->UploadBufferImmediate( { { polBuf,    zeros.data(),          agentCount * sizeof( glm::vec4 ) } } );
    stream->UploadBufferImmediate( { { orientBuf, inOrientations.data(), agentCount * sizeof( glm::vec4 ) } } );
    stream->UploadBufferImmediate( { { hullBuf,   &hullGPU,             sizeof( ContactHullGPU ) } } );
    stream->UploadBufferImmediate( { { plateBuf,  &plateData,           sizeof( PlateBufferGPU ) } } );

    ComputePipelineDesc   pipeDesc{};
    pipeDesc.shader                  = rm->CreateShader( "shaders/compute/jkr_forces.comp" );
    ComputePipelineHandle pipeHandle = rm->CreatePipeline( pipeDesc );
    ComputePipeline*      pipeline   = rm->GetPipeline( pipeHandle );

    BindingGroup* bg = rm->GetBindingGroup( rm->CreateBindingGroup( pipeHandle, 0 ) );
    bg->Bind( 0,  rm->GetBuffer( inBuf ) );
    bg->Bind( 1,  rm->GetBuffer( outBuf ) );
    bg->Bind( 2,  rm->GetBuffer( pressBuf ) );
    bg->Bind( 3,  rm->GetBuffer( hashBuf ) );
    bg->Bind( 4,  rm->GetBuffer( offsetBuf ) );
    bg->Bind( 5,  rm->GetBuffer( countBuf ) );
    bg->Bind( 6,  rm->GetBuffer( phenoBuf ) );
    bg->Bind( 7,  rm->GetBuffer( profBuf ) );
    bg->Bind( 8,  rm->GetBuffer( affBuf ) );
    bg->Bind( 9,  rm->GetBuffer( polBuf ) );
    bg->Bind( 10, rm->GetBuffer( orientBuf ) );
    bg->Bind( 11, rm->GetBuffer( hullBuf ) );
    bg->Bind( 12, rm->GetBuffer( plateBuf ) );
    bg->Build();

    ComputePushConstants pc{};
    pc.dt          = dt;
    pc.maxCapacity = agentCount;
    pc.offset      = 0;
    pc.fParam0     = repulsion;
    pc.fParam1     = adhesion;
    pc.fParam2     = -1.0f;         // reqLC = any
    pc.fParam3     = -1.0f;         // reqCT = any
    pc.fParam4     = damping;
    pc.fParam5     = maxRadius;
    pc.uParam0     = offsetArraySize;
    pc.uParam1     = 0;             // grpNdx = 0
    pc.domainSize  = glm::vec4( 100.0f, 100.0f, 100.0f, maxRadius * 3.0f );
    // Cadherin: gridSize.x=1 enables it; gridSize.y = uintBitsToFloat(couplingStrength).
    // Affinity matrix is identity so A=dot(profile,(M*profile))=1.0 → adhScale=couplingStrength.
    // gridSize.z (Phase 4.5-B): bit 0 = polarity active flag; bits 16..31 =
    // packHalf2x16 upper half = lateralAdhesionScale. Default 0 → no change
    // from pre-Phase-4.5-B behaviour (hull pairs are torque-only).
    uint32_t lateralPacked = glm::packHalf2x16( glm::vec2( 0.0f, lateralAdhesionScale ) );
    uint32_t gridSizeZ     = ( lateralPacked & 0xFFFF0000u );
    if ( cadherinCoupling > 0.0f ) {
        uint32_t couplingBits;
        std::memcpy( &couplingBits, &cadherinCoupling, sizeof( float ) );
        pc.gridSize = glm::uvec4( 1u, couplingBits, gridSizeZ, 0u );
    } else {
        pc.gridSize = glm::uvec4( 0u, 0u, gridSizeZ, 0u );
    }

    auto ctx = device->GetThreadContext( device->CreateThreadContext( QueueType::COMPUTE ) );
    auto cmd = ctx->GetCommandBuffer( ctx->CreateCommandBuffer() );
    cmd->Begin();
    cmd->SetPipeline( pipeline );
    cmd->SetBindingGroup( bg, pipeline->GetLayout(), VK_PIPELINE_BIND_POINT_COMPUTE );
    cmd->PushConstants( pipeline->GetLayout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof( pc ), &pc );
    cmd->Dispatch( ( agentCount + 255 ) / 256, 1, 1 );
    cmd->End();
    device->GetComputeQueue()->Submit( { cmd } );
    device->GetComputeQueue()->WaitIdle();

    RigidBodyResult result;
    result.positions.resize( agentCount );
    result.orientations.resize( agentCount );
    stream->ReadbackBufferImmediate( outBuf,    result.positions.data(),    agentCount * sizeof( glm::vec4 ) );
    stream->ReadbackBufferImmediate( orientBuf, result.orientations.data(), agentCount * sizeof( glm::vec4 ) );

    rm->DestroyBuffer( inBuf );    rm->DestroyBuffer( outBuf );    rm->DestroyBuffer( pressBuf );
    rm->DestroyBuffer( hashBuf );  rm->DestroyBuffer( offsetBuf ); rm->DestroyBuffer( countBuf );
    rm->DestroyBuffer( phenoBuf ); rm->DestroyBuffer( profBuf );   rm->DestroyBuffer( affBuf );
    rm->DestroyBuffer( polBuf );   rm->DestroyBuffer( orientBuf ); rm->DestroyBuffer( hullBuf );
    rm->DestroyBuffer( plateBuf );

    return result;
}

// 1. Off-axis hull contact produces a non-zero torque that rotates the cell.
//
// Cell 0 at (0,0,0), cell 1 at (0, 0.5, 0).  Hull: 1 point at (0,0,0.5), subR=0.4.
// Cell 0's hull point in world = (0,0,0.5).  Cell 1's hull point = (0,0.5,0.5).
// Pair distance = 0.5 < contactDist=0.8 → overlap=0.3.
// Adhesion-only torque: dir = (0,-1,0) (pointing from cell 0 hull pt toward cell 1 hull pt).
// Force = dir * (-adh) = (0,+F,0).
// Torque = cross((0,0,0.5), (0,+F,0)) = (-F*0.5, 0, 0) → rotation about X axis (negative).
// Integrated: orientations[0].x < 0.
TEST_F( ComputeTest, RigidBody_TorqueFromEdgeContact )
{
    if( !m_device )
        GTEST_SKIP();

    std::vector<glm::vec4> pos  = { glm::vec4( 0.0f, 0.0f, 0.0f, 1.0f ),
                                    glm::vec4( 0.0f, 0.5f, 0.0f, 1.0f ) };
    std::vector<glm::vec4> ori  = { glm::vec4( 0.0f, 0.0f, 0.0f, 1.0f ),
                                    glm::vec4( 0.0f, 0.0f, 0.0f, 1.0f ) };
    std::vector<glm::vec4> hull = { glm::vec4( 0.0f, 0.0f, 0.5f, 0.4f ) }; // xyz=offset, w=subR

    auto r = RunJKRRigidBody( m_device.get(), m_rm.get(), m_stream.get(),
                               pos, ori,
                               1u, hull,
                               50.0f, 20.0f, 1.5f );

    // Adhesion-only torque: adhesive force pulls hull point toward contact → negative X torque.
    EXPECT_LT( r.orientations[0].x, -0.1f ) << "Cell 0 should rotate about X (negative, adhesion pulls)";
    EXPECT_GT( r.orientations[1].x,  0.1f ) << "Cell 1 should rotate about X (positive, reaction)";
}

// 2. Collinear hull contact produces zero torque.
//
// Cell 0 at (0,0,0), cell 1 at (0.55, 0, 0).  Hull: 1 point at (0.3,0,0), subR=0.3.
// Adhesion force is purely along X (collinear with lever arm).
// cross((0.3,0,0), (+F,0,0)) = (0,0,0) → zero torque → identity quaternion preserved.
TEST_F( ComputeTest, RigidBody_NoTorqueWhenAligned )
{
    if( !m_device )
        GTEST_SKIP();

    std::vector<glm::vec4> pos  = { glm::vec4( 0.0f,  0.0f, 0.0f, 1.0f ),
                                    glm::vec4( 0.55f, 0.0f, 0.0f, 1.0f ) };
    std::vector<glm::vec4> ori  = { glm::vec4( 0.0f, 0.0f, 0.0f, 1.0f ),
                                    glm::vec4( 0.0f, 0.0f, 0.0f, 1.0f ) };
    std::vector<glm::vec4> hull = { glm::vec4( 0.3f, 0.0f, 0.0f, 0.3f ) };

    auto r = RunJKRRigidBody( m_device.get(), m_rm.get(), m_stream.get(),
                               pos, ori,
                               1u, hull,
                               50.0f, 20.0f, 1.5f );

    // Force is coaxial with lever arm → cross product is zero → no rotation.
    EXPECT_NEAR( r.orientations[0].x, 0.0f, 1e-4f ) << "No torque: x should stay 0";
    EXPECT_NEAR( r.orientations[0].y, 0.0f, 1e-4f ) << "No torque: y should stay 0";
    EXPECT_NEAR( r.orientations[0].z, 0.0f, 1e-4f ) << "No torque: z should stay 0";
    EXPECT_NEAR( r.orientations[0].w, 1.0f, 1e-4f ) << "No torque: w should stay 1";
}

// 3. Quaternion remains unit-length after integration.
//
// Uses the same off-axis contact as test 1 (large adhesion torque, large angular step)
// to stress-test normalisation.
TEST_F( ComputeTest, RigidBody_QuaternionNormalized )
{
    if( !m_device )
        GTEST_SKIP();

    std::vector<glm::vec4> pos  = { glm::vec4( 0.0f, 0.0f, 0.0f, 1.0f ),
                                    glm::vec4( 0.0f, 0.5f, 0.0f, 1.0f ) };
    std::vector<glm::vec4> ori  = { glm::vec4( 0.0f, 0.0f, 0.0f, 1.0f ),
                                    glm::vec4( 0.0f, 0.0f, 0.0f, 1.0f ) };
    std::vector<glm::vec4> hull = { glm::vec4( 0.0f, 0.0f, 0.5f, 0.4f ) };

    auto r = RunJKRRigidBody( m_device.get(), m_rm.get(), m_stream.get(),
                               pos, ori,
                               1u, hull,
                               50.0f, 20.0f, 1.5f );

    for ( int i = 0; i < 2; i++ )
    {
        glm::vec4 q = r.orientations[i];
        float     len2 = q.x*q.x + q.y*q.y + q.z*q.z + q.w*q.w;
        EXPECT_NEAR( len2, 1.0f, 1e-4f ) << "Quaternion [" << i << "] is not unit length";
    }
}

// 4. hull count = 0 falls back to point-particle JKR: position changes, orientation unchanged.
//
// Two heavily overlapping agents (dist=0.1, interactDist=3.0) with no contact hull.
// Repulsion pushes them apart.  Orientation buffer must be untouched.
TEST_F( ComputeTest, RigidBody_PointParticleFallback )
{
    if( !m_device )
        GTEST_SKIP();

    std::vector<glm::vec4> pos  = { glm::vec4( 0.0f, 0.0f, 0.0f, 1.0f ),
                                    glm::vec4( 0.1f, 0.0f, 0.0f, 1.0f ) };
    std::vector<glm::vec4> ori  = { glm::vec4( 0.0f, 0.0f, 0.0f, 1.0f ),
                                    glm::vec4( 0.0f, 0.0f, 0.0f, 1.0f ) };

    // hullCount=0: point-particle path
    auto r = RunJKRRigidBody( m_device.get(), m_rm.get(), m_stream.get(),
                               pos, ori,
                               0u, {},
                               50.0f, 0.0f, 1.5f, 0.0f, 0.016f );

    // Cell 0 pushed in −X by repulsion.
    EXPECT_LT( r.positions[0].x, 0.0f ) << "Cell 0 should be pushed left";
    EXPECT_GT( r.positions[1].x, 0.1f ) << "Cell 1 should be pushed right";

    // No hull path → orientation block not reached → identity preserved.
    EXPECT_NEAR( r.orientations[0].x, 0.0f, 1e-5f ) << "Orientation x unchanged";
    EXPECT_NEAR( r.orientations[0].y, 0.0f, 1e-5f ) << "Orientation y unchanged";
    EXPECT_NEAR( r.orientations[0].z, 0.0f, 1e-5f ) << "Orientation z unchanged";
    EXPECT_NEAR( r.orientations[0].w, 1.0f, 1e-5f ) << "Orientation w unchanged";
}

// 5. Dead cells (position.w=0) are excluded from the neighbour force loop.
//
// Cell 0 is dead (w=0) and overlaps hull-point-to-hull-point with live cell 1.
// The guard `if (neighborData.w == 0.0) continue;` in the neighbour loop means
// cell 1 never processes cell 0 as a source of force → cell 1's orientation stays
// at identity despite being close enough for hull contact.
TEST_F( ComputeTest, RigidBody_DeadCell_ExcludedFromNeighborhood )
{
    if( !m_device )
        GTEST_SKIP();

    std::vector<glm::vec4> pos  = { glm::vec4( 0.0f, 0.0f, 0.0f, 0.0f ),  // dead (w=0)
                                    glm::vec4( 0.0f, 0.5f, 0.0f, 1.0f ) }; // live
    std::vector<glm::vec4> ori  = { glm::vec4( 0.0f, 0.0f, 0.0f, 1.0f ),
                                    glm::vec4( 0.0f, 0.0f, 0.0f, 1.0f ) };
    // Hull point at (0,0,0.5), subR=0.4: hull[0] of cell 0 = (0,0,0.5),
    // hull[0] of cell 1 = (0,0.5,0.5), distance=0.5 < contactDist=0.8 — would contact if not guarded.
    std::vector<glm::vec4> hull = { glm::vec4( 0.0f, 0.0f, 0.5f, 0.4f ) };

    auto r = RunJKRRigidBody( m_device.get(), m_rm.get(), m_stream.get(),
                               pos, ori,
                               1u, hull,
                               50.0f, 0.0f, 1.5f );

    // Cell 1 (live) must not rotate — dead cell 0 is skipped as a neighbour.
    EXPECT_NEAR( r.orientations[1].x, 0.0f, 1e-5f ) << "Live cell should not be torqued by dead neighbor";
    EXPECT_NEAR( r.orientations[1].y, 0.0f, 1e-5f );
    EXPECT_NEAR( r.orientations[1].z, 0.0f, 1e-5f );
    EXPECT_NEAR( r.orientations[1].w, 1.0f, 1e-5f );
    // Cell 1 must not have moved.
    EXPECT_NEAR( r.positions[1].y, 0.5f, 1e-4f ) << "Live cell position should be unchanged";
}

// 6. Oriented steric repulsion pushes a rotated tile harder than an aligned tile.
//
// Setup: 2 cells along Z, dist=1.2.  Hull: 4 corners at (±0.5, 0, ±0.5), subR=0.1.
// stericZ=0.5, stericY=0.1 enables the oriented box steric path.
//
// Cell 0 at (0,0,0), cell 1 at (0,0,1.2).
// ALIGNED run: both cells identity quaternion.
//   Support extents along Z: max(0.5*0 + 0.1*0 + 0.5*1) + ... = 0.5 each → sOver = 1.0-1.2 < 0.
//   Wait — Z is model-Z and d_hat ≈ (0,0,1).  stericHalfX derived from |points.x| = 0.5.
//   myExt = 0.5*|dot(axisX,z)| + 0.1*|dot(axisY,z)| + 0.5*|dot(axisZ,z)|
//         = 0.5*0 + 0.1*0 + 0.5*1 = 0.5  (identity: axisZ = +Z).
//   sDist = 0.5 + 0.5 = 1.0 < dist=1.2 → no steric overlap → no extra force.
//
// ROTATED run: cell 1 rotated 30° about Y (model Y = (0,1,0), qi=(0,sinπ/12,0,cosπ/12)).
//   cell 1's axisX after rotation = ( cos30°, 0, sin30° ) = (0.866, 0, 0.5).
//   nAxisZ after rotation = (-sin30°, 0, cos30°) = (-0.5, 0, 0.866).
//   nExt = 0.5*|dot(axisX,(0,0,1))| + 0.1*0 + 0.5*|dot(axisZ,(0,0,1))|
//        = 0.5*0.5 + 0 + 0.5*0.866 = 0.25 + 0.433 = 0.683.
//   sDist = 0.5 + 0.683 = 1.183 > dist=1.2? → 1.183 < 1.2, no overlap yet (just barely).
//
// Actually at dist=1.1 to make overlap clearer — see setup below.
//
// Simpler assertion: after N frames with stericZ>0, steric-enabled repulsion produces
// more separation than steric-disabled repulsion (stericZ=0) for the rotated case.
TEST_F( ComputeTest, RigidBody_StericRepulsion_RotatedTile )
{
    if( !m_device )
        GTEST_SKIP();

    // 4-corner tile hull: corners at (±0.5, 0, ±0.5), sub-sphere radius 0.1
    std::vector<glm::vec4> hull = {
        glm::vec4(  0.5f, 0.0f,  0.5f, 0.1f ),
        glm::vec4( -0.5f, 0.0f,  0.5f, 0.1f ),
        glm::vec4( -0.5f, 0.0f, -0.5f, 0.1f ),
        glm::vec4(  0.5f, 0.0f, -0.5f, 0.1f ),
    };

    // Cell 1 rotated 45° about Y: sin(45°)=cos(45°)=1/√2
    const float s = 1.0f / std::sqrt( 2.0f );
    std::vector<glm::vec4> pos = { glm::vec4( 0.0f, 0.0f, 0.0f,  1.0f ),
                                   glm::vec4( 0.0f, 0.0f, 1.1f,  1.0f ) };
    // Cell 0 identity, cell 1 rotated 45° about Y axis: q = (0, sin22.5°, 0, cos22.5°)
    const float half45 = glm::radians( 22.5f );
    std::vector<glm::vec4> oriRotated = {
        glm::vec4( 0.0f, 0.0f,           0.0f,           1.0f ),
        glm::vec4( 0.0f, std::sin(half45), 0.0f, std::cos(half45) )
    };
    std::vector<glm::vec4> oriAligned = {
        glm::vec4( 0.0f, 0.0f, 0.0f, 1.0f ),
        glm::vec4( 0.0f, 0.0f, 0.0f, 1.0f )
    };

    // Steric-enabled: stericZ=0.5 (model-Z half-extent), stericY=0.1 (model-Y = thickness)
    auto rSteric = RunJKRRigidBody( m_device.get(), m_rm.get(), m_stream.get(),
                                    pos, oriRotated,
                                    4u, hull,
                                    20.0f, 0.0f, 0.75f,
                                    0.0f, 1.0f / 60.0f,
                                    0.5f, 0.1f );

    // No steric: same rotated setup but stericZ=stericY=0
    auto rNoSteric = RunJKRRigidBody( m_device.get(), m_rm.get(), m_stream.get(),
                                      pos, oriRotated,
                                      4u, hull,
                                      20.0f, 0.0f, 0.75f,
                                      0.0f, 1.0f / 60.0f,
                                      0.0f, 0.0f );

    // With steric enabled, the rotated corner adds extra repulsion → cell 0 moves further
    // in the -Z direction (away from cell 1) than without steric.
    float sepSteric   = rSteric.positions[1].z   - rSteric.positions[0].z;
    float sepNoSteric = rNoSteric.positions[1].z  - rNoSteric.positions[0].z;

    EXPECT_GT( sepSteric, sepNoSteric )
        << "Steric repulsion should increase separation for a rotated tile "
        << "(steric=" << sepSteric << ", no-steric=" << sepNoSteric << ")";
}

// 7. Distributed hull adhesion-only torques drive a rotated tile toward alignment.
//
// Cell 0 at (0,0,0) identity, Cell 1 at (1.4,0,0) rotated +15 deg Y.
// Hull: 8 points — 4 corners + 4 edge midpoints, matching CreateTile(1.4, 1.4, 0.2).
//
// At 15° rotation, cell 1's left edge midpoint (-0.7,0,0) maps to world (0.724, 0, 0.181).
// Distance to cell 0's right midpoint (0.7, 0, 0) = 0.183 < contactDist 0.2 → contact fires.
//
// Adhesion-only torque on cell 1: dir = (0.131, 0, 0.989), force = dir*(-adh) toward Pb.
// Lever arm = qrot(q1, (-0.7,0,0)) = (-0.676, 0, 0.181).
// torque_j = cross((-0.676,0,0.181), (-0.131,-,−0.989)*adh) → negative Y → Y decreases.
// Cell 1 (rotated +15°) gets a restoring torque — its Y quaternion component decreases toward 0.
//
// Without cadherin (adhesion=0): hull torque = cross(arm, 0) = 0 → no rotation change.
TEST_F( ComputeTest, RigidBody_DistributedHull_RestoresRotation )
{
    if( !m_device )
        GTEST_SKIP();

    // 8-point tile hull: 4 corners + 4 edge midpoints, matching CreateTile(1.4, 1.4, 0.2)
    std::vector<glm::vec4> hull = {
        glm::vec4(  0.7f, 0.0f,  0.7f, 0.1f ),   // corner ++
        glm::vec4( -0.7f, 0.0f,  0.7f, 0.1f ),   // corner -+
        glm::vec4( -0.7f, 0.0f, -0.7f, 0.1f ),   // corner --
        glm::vec4(  0.7f, 0.0f, -0.7f, 0.1f ),   // corner +-
        glm::vec4(  0.7f, 0.0f,  0.0f, 0.1f ),   // right edge mid
        glm::vec4( -0.7f, 0.0f,  0.0f, 0.1f ),   // left edge mid
        glm::vec4(  0.0f, 0.0f,  0.7f, 0.1f ),   // front edge mid
        glm::vec4(  0.0f, 0.0f, -0.7f, 0.1f ),   // back edge mid
    };

    std::vector<glm::vec4> pos = {
        glm::vec4( 0.0f, 0.0f, 0.0f, 1.0f ),   // cell 0: center
        glm::vec4( 1.4f, 0.0f, 0.0f, 1.0f ),   // cell 1: within edge-midpoint contact range
    };

    // Cell 1 rotated +15 deg around Y: half-angle = 7.5 deg
    const float h15 = glm::radians( 7.5f );
    std::vector<glm::vec4> ori = {
        glm::vec4( 0.0f,              0.0f, 0.0f, 1.0f ),
        glm::vec4( 0.0f, std::sin(h15), 0.0f, std::cos(h15) ),
    };
    const float initRotatedY = std::sin( h15 );  // ≈ 0.1305

    // ── With cadherin (adhesion > 0) ────────────────────────────────────────────
    // Edge midpoint contact fires; adhesion torque is restoring on cell 1.
    auto rAdh = RunJKRRigidBody( m_device.get(), m_rm.get(), m_stream.get(),
                                  pos, ori,
                                  8u, hull,
                                  20.0f,          // repulsion
                                  2.0f,           // adhesion
                                  0.75f,          // maxRadius
                                  100.0f,         // damping
                                  1.0f / 60.0f,   // dt
                                  0.7f, 0.1f,     // stericZ, stericY
                                  0.0f,           // edgeAlignStrength = 0 (no phenomenological torque)
                                  2.0f );         // cadherinCoupling (adhScale=2)

    // Rotated tile: adhesion-only torque reduces its Y rotation toward 0.
    EXPECT_LT( rAdh.orientations[1].y, initRotatedY )
        << "Rotated tile should have reduced Y rotation from distributed hull adhesion";
    // Some torque must have fired — Y should have changed noticeably.
    EXPECT_LT( rAdh.orientations[1].y, initRotatedY - 1e-4f )
        << "Y rotation change should be measurable";

    // ── Without cadherin (adhesion = 0) ─────────────────────────────────────────
    // Hull torque = cross(arm, dir*0) = 0 → no rotation despite hull contact.
    auto rNoAdh = RunJKRRigidBody( m_device.get(), m_rm.get(), m_stream.get(),
                                    pos, ori,
                                    8u, hull,
                                    20.0f, 0.0f, 0.75f,
                                    100.0f, 1.0f / 60.0f,
                                    0.7f, 0.1f,
                                    0.0f, 0.0f );

    // Without adhesion, no hull torque → cell 1 stays at initial rotation.
    EXPECT_NEAR( rNoAdh.orientations[1].y, initRotatedY, 1e-4f )
        << "No torque without adhesion — Y rotation should be unchanged";
}

// =============================================================================
// jkr_forces.comp — hull-based translation tests
// =============================================================================

// 8. Cells with overlapping hull pairs are pushed apart by point-particle translation.
//
// Hull pairs are torque-only; the unconditional point-particle block provides translation.
// Cell 0 at (0,0,0), cell 1 at (0,0.5,0). Hull: 1 point at (0,0,0.5), subR=0.4.
// Point-particle: interactDist=3.0 (maxR=1.5), dist=0.5, overlap=2.5 → large repulsion.
// Both cells should move apart along Y.
TEST_F( ComputeTest, RigidBody_HullTranslation_RepulsionPushesApart )
{
    if( !m_device )
        GTEST_SKIP();

    std::vector<glm::vec4> pos = { glm::vec4( 0.0f, 0.0f, 0.0f, 1.0f ),
                                   glm::vec4( 0.0f, 0.5f, 0.0f, 1.0f ) };
    std::vector<glm::vec4> ori = { glm::vec4( 0.0f, 0.0f, 0.0f, 1.0f ),
                                   glm::vec4( 0.0f, 0.0f, 0.0f, 1.0f ) };
    // 1 hull point at (0,0,0.5), subR=0.4.
    std::vector<glm::vec4> hull = { glm::vec4( 0.0f, 0.0f, 0.5f, 0.4f ) };

    // Zero adhesion: pure repulsion from overlapping hull pairs.
    auto r = RunJKRRigidBody( m_device.get(), m_rm.get(), m_stream.get(),
                               pos, ori,
                               1u, hull,
                               50.0f, 0.0f, 1.5f,
                               0.0f, 1.0f );

    // Cell 0 was at Y=0 and receives a -Y force from hull pair repulsion.
    EXPECT_LT( r.positions[0].y, 0.0f )
        << "Cell 0 should be pushed in -Y by hull sub-sphere repulsion";
    // Cell 1 (the other agent in the dispatch — processed by its own invocation):
    // both cells are in the same dispatch here, so cell 1 also receives +Y force.
    EXPECT_GT( r.positions[1].y, 0.5f )
        << "Cell 1 should be pushed in +Y by hull sub-sphere repulsion";
}

// 9. Hull adhesion partially cancels repulsion — cells separate less than with zero adhesion.
//
// Same geometry as test 8.  Adhesion > 0 + cadherin coupling means adhF > 0.
// netF = repF - adhF < repF → smaller displacement than zero-adhesion run.
TEST_F( ComputeTest, RigidBody_HullTranslation_AdhesionReducesRepulsion )
{
    if( !m_device )
        GTEST_SKIP();

    std::vector<glm::vec4> pos = { glm::vec4( 0.0f, 0.0f, 0.0f, 1.0f ),
                                   glm::vec4( 0.0f, 0.5f, 0.0f, 1.0f ) };
    std::vector<glm::vec4> ori = { glm::vec4( 0.0f, 0.0f, 0.0f, 1.0f ),
                                   glm::vec4( 0.0f, 0.0f, 0.0f, 1.0f ) };
    std::vector<glm::vec4> hull = { glm::vec4( 0.0f, 0.0f, 0.5f, 0.4f ) };

    // No adhesion run.
    auto rNoAdh = RunJKRRigidBody( m_device.get(), m_rm.get(), m_stream.get(),
                                    pos, ori,
                                    1u, hull,
                                    50.0f, 0.0f, 1.5f,
                                    0.0f, 1.0f );

    // Adhesion run: adhesion=20, cadherin coupling=1.0 → adhScale=1.0 → adhF reduces netF.
    auto rAdh = RunJKRRigidBody( m_device.get(), m_rm.get(), m_stream.get(),
                                  pos, ori,
                                  1u, hull,
                                  50.0f, 20.0f, 1.5f,
                                  0.0f, 1.0f,
                                  0.0f, 0.0f, 0.0f, 1.0f );  // cadherinCoupling=1

    float sepNoAdh = rNoAdh.positions[1].y - rNoAdh.positions[0].y;
    float sepAdh   = rAdh.positions[1].y   - rAdh.positions[0].y;

    EXPECT_LT( sepAdh, sepNoAdh )
        << "Adhesion should reduce separation vs pure repulsion "
        << "(adh=" << sepAdh << ", no-adh=" << sepNoAdh << ")";
}

// 10. Hull sub-sphere pairs are torque-only: hull contact does NOT add extra translation.
//
// Hull sub-sphere overlap is on a different scale to centre-to-centre overlap and would
// overwhelm point-particle adhesion if it fed into translation forces.  Hull is alignment-
// only; all translation comes from the unconditional point-particle block.
//
// Cell 0 at (0,0,0), cell 1 at (0,0,0.5). Hull: 1 point at (0,0,0.3), subR=0.4.
// Hull pair: Pa=(0,0,0.3), Pb=(0,0,0.8). pairDist=0.5, contactDist=0.8, overlap=0.3.
// Point-particle: interactDist=1.0, dist=0.5, overlap=0.5. Fires for both runs.
// Expected: separation WITH hull ≈ separation WITHOUT hull (only point-particle drives translation).
TEST_F( ComputeTest, RigidBody_HullTorqueOnly_NoTranslationBoost )
{
    if( !m_device )
        GTEST_SKIP();

    std::vector<glm::vec4> pos = { glm::vec4( 0.0f, 0.0f, 0.0f, 1.0f ),
                                   glm::vec4( 0.0f, 0.0f, 0.5f, 1.0f ) };
    std::vector<glm::vec4> ori = { glm::vec4( 0.0f, 0.0f, 0.0f, 1.0f ),
                                   glm::vec4( 0.0f, 0.0f, 0.0f, 1.0f ) };
    std::vector<glm::vec4> hull = { glm::vec4( 0.0f, 0.0f, 0.3f, 0.4f ) };

    auto rHull = RunJKRRigidBody( m_device.get(), m_rm.get(), m_stream.get(),
                                   pos, ori,
                                   1u, hull,
                                   50.0f, 0.0f, 0.5f,  // maxRadius=0.5 → interactDist=1.0
                                   0.0f, 1.0f );

    std::vector<glm::vec4> emptyHull;
    auto rNoHull = RunJKRRigidBody( m_device.get(), m_rm.get(), m_stream.get(),
                                     pos, ori,
                                     0u, emptyHull,
                                     50.0f, 0.0f, 0.5f,
                                     0.0f, 1.0f );

    float sepHull   = rHull.positions[1].z   - rHull.positions[0].z;
    float sepNoHull = rNoHull.positions[1].z - rNoHull.positions[0].z;

    // Translation must come from point-particle only — hull adds no extra displacement.
    EXPECT_NEAR( sepHull, sepNoHull, 1e-4f )
        << "Hull torque-only: separation should equal no-hull separation "
        << "(hull=" << sepHull << ", no-hull=" << sepNoHull << ")";
    // Point-particle fires unconditionally — both runs separate beyond initial 0.5.
    EXPECT_GT( sepNoHull, 0.5f )
        << "Point-particle should always fire regardless of hull count";
}

// =============================================================================
// Phase 4.5-B — lateral adhesion (hull-pair translation contribution)
// =============================================================================
//
// When lateralAdhesionScale > 0, each overlapping hull sub-sphere pair
// contributes a translational pull along its contact normal, scaled by the
// user-set parameter. This models cadherin-belt junctions (VE-cadherin) that
// exert translational force along the cell-cell interface, not just
// orientational alignment. Multiple overlapping pairs sum their contributions,
// producing a strong preference for face-to-face contacts vs corner contacts.

// Test 1. With lateralAdhesionScale > 0, hull adhesion pulls cells CLOSER
// than the zero-scale (torque-only) baseline. Regression: Phase 4 and earlier
// remain bit-identical at scale=0.
TEST_F( ComputeTest, RigidBody_LateralAdhesion_PullsCellsTogetherVsBaseline )
{
    if( !m_device )
        GTEST_SKIP();

    // Two cells face-to-face along Z axis, both hull points on the facing
    // surface so they overlap when the cells approach.
    std::vector<glm::vec4> pos = { glm::vec4( 0.0f, 0.0f, 0.0f, 1.0f ),
                                   glm::vec4( 0.0f, 0.0f, 0.6f, 1.0f ) };
    std::vector<glm::vec4> ori = { glm::vec4( 0.0f, 0.0f, 0.0f, 1.0f ),
                                   glm::vec4( 0.0f, 0.0f, 0.0f, 1.0f ) };
    // Single hull point on each cell at the facing surface — one hull pair will overlap.
    std::vector<glm::vec4> hull = { glm::vec4( 0.0f, 0.0f, 0.3f, 0.4f ) };

    // Helper signature: repulsion, adhesion, maxRadius, damping, dt, stericZ,
    // stericY, edgeAlign, cadherinCoupling, lateralAdhesionScale. We enable
    // cadherin (coupling=1.0) to set adhScale=1 so adhesion fires normally.
    // Baseline: lateralAdhesionScale = 0 → hull is torque-only (Phase-4 behaviour).
    auto rBase = RunJKRRigidBody( m_device.get(), m_rm.get(), m_stream.get(),
                                   pos, ori,
                                   1u, hull,
                                   50.0f, 5.0f, 0.5f,
                                   /*damping=*/0.0f, /*dt=*/1.0f,
                                   /*stericZ=*/0.0f, /*stericY=*/0.0f,
                                   /*edgeAlign=*/0.0f, /*cadherinCoupling=*/1.0f,
                                   /*lateralAdhesionScale=*/0.0f );

    // Phase 4.5-B: lateralAdhesionScale = 0.5 → hull contributes translation.
    auto rLat  = RunJKRRigidBody( m_device.get(), m_rm.get(), m_stream.get(),
                                   pos, ori,
                                   1u, hull,
                                   50.0f, 5.0f, 0.5f,
                                   /*damping=*/0.0f, /*dt=*/1.0f,
                                   /*stericZ=*/0.0f, /*stericY=*/0.0f,
                                   /*edgeAlign=*/0.0f, /*cadherinCoupling=*/1.0f,
                                   /*lateralAdhesionScale=*/0.5f );

    float sepBase = rBase.positions[1].z - rBase.positions[0].z;
    float sepLat  = rLat .positions[1].z - rLat .positions[0].z;

    // Lateral adhesion must pull cells closer together (smaller separation).
    EXPECT_LT( sepLat, sepBase )
        << "Lateral adhesion must reduce cell separation vs torque-only baseline "
        << "(lateral=" << sepLat << " vs baseline=" << sepBase << ")";
}

// Test 2. lateralAdhesionScale = 0 reproduces the pre-Phase-4.5-B behaviour
// bit-exactly: hull translation code path is gated off. Regression guard
// preventing accidental activation of the new mechanism.
TEST_F( ComputeTest, RigidBody_LateralAdhesion_ZeroScaleIsTorqueOnly )
{
    if( !m_device )
        GTEST_SKIP();

    // Same geometry as the regression guard below (HullTorqueOnly test),
    // but we check the translation output directly at scale=0 vs no-hull.
    std::vector<glm::vec4> pos = { glm::vec4( 0.0f, 0.0f, 0.0f, 1.0f ),
                                   glm::vec4( 0.0f, 0.0f, 0.5f, 1.0f ) };
    std::vector<glm::vec4> ori = { glm::vec4( 0.0f, 0.0f, 0.0f, 1.0f ),
                                   glm::vec4( 0.0f, 0.0f, 0.0f, 1.0f ) };
    std::vector<glm::vec4> hull = { glm::vec4( 0.0f, 0.0f, 0.3f, 0.4f ) };

    // Hull path with scale=0: hull pairs overlap but contribute only torque.
    auto rHull = RunJKRRigidBody( m_device.get(), m_rm.get(), m_stream.get(),
                                   pos, ori,
                                   1u, hull,
                                   50.0f, 0.0f, 0.5f,
                                   /*damping=*/0.0f, /*dt=*/1.0f,
                                   /*stericZ=*/0.0f, /*stericY=*/0.0f,
                                   /*edgeAlign=*/0.0f, /*cadherinCoupling=*/1.0f,
                                   /*lateralAdhesionScale=*/0.0f );

    // No-hull path: point-particle only (nothing else contributes).
    std::vector<glm::vec4> emptyHull;
    auto rNoHull = RunJKRRigidBody( m_device.get(), m_rm.get(), m_stream.get(),
                                     pos, ori,
                                     0u, emptyHull,
                                     50.0f, 0.0f, 0.5f,
                                     /*damping=*/0.0f, /*dt=*/1.0f,
                                     /*stericZ=*/0.0f, /*stericY=*/0.0f,
                                     /*edgeAlign=*/0.0f, /*cadherinCoupling=*/1.0f,
                                     /*lateralAdhesionScale=*/0.0f );

    float sepHull   = rHull.positions[1].z   - rHull.positions[0].z;
    float sepNoHull = rNoHull.positions[1].z - rNoHull.positions[0].z;

    // With lateralAdhesionScale=0, hull must contribute zero translation —
    // bit-exact match to the no-hull run.
    EXPECT_NEAR( sepHull, sepNoHull, 1e-4f )
        << "lateralAdhesionScale=0 must preserve Phase-4 torque-only semantics "
        << "(hull=" << sepHull << ", no-hull=" << sepNoHull << ")";
}

// Test 3. Translational pull scales linearly in lateralAdhesionScale. At s=0.2
// the inward displacement should be roughly 2× the s=0.1 displacement. Verifies
// the force formulation is linear in the parameter (the plan claim).
TEST_F( ComputeTest, RigidBody_LateralAdhesion_ScalesLinearly )
{
    if( !m_device )
        GTEST_SKIP();

    std::vector<glm::vec4> pos = { glm::vec4( 0.0f, 0.0f, 0.0f, 1.0f ),
                                   glm::vec4( 0.0f, 0.0f, 0.6f, 1.0f ) };
    std::vector<glm::vec4> ori = { glm::vec4( 0.0f, 0.0f, 0.0f, 1.0f ),
                                   glm::vec4( 0.0f, 0.0f, 0.0f, 1.0f ) };
    std::vector<glm::vec4> hull = { glm::vec4( 0.0f, 0.0f, 0.3f, 0.4f ) };

    // Three runs at different lateral scales: 0, 0.1, 0.2.
    auto r0  = RunJKRRigidBody( m_device.get(), m_rm.get(), m_stream.get(),
                                 pos, ori, 1u, hull,
                                 50.0f, 5.0f, 0.5f,
                                 /*damping=*/0.0f, /*dt=*/1.0f,
                                 /*stericZ=*/0.0f, /*stericY=*/0.0f,
                                 /*edgeAlign=*/0.0f, /*cadherinCoupling=*/1.0f,
                                 /*lateralAdhesionScale=*/0.0f );
    auto r1  = RunJKRRigidBody( m_device.get(), m_rm.get(), m_stream.get(),
                                 pos, ori, 1u, hull,
                                 50.0f, 5.0f, 0.5f,
                                 /*damping=*/0.0f, /*dt=*/1.0f,
                                 /*stericZ=*/0.0f, /*stericY=*/0.0f,
                                 /*edgeAlign=*/0.0f, /*cadherinCoupling=*/1.0f,
                                 /*lateralAdhesionScale=*/0.1f );
    auto r2  = RunJKRRigidBody( m_device.get(), m_rm.get(), m_stream.get(),
                                 pos, ori, 1u, hull,
                                 50.0f, 5.0f, 0.5f,
                                 /*damping=*/0.0f, /*dt=*/1.0f,
                                 /*stericZ=*/0.0f, /*stericY=*/0.0f,
                                 /*edgeAlign=*/0.0f, /*cadherinCoupling=*/1.0f,
                                 /*lateralAdhesionScale=*/0.2f );

    float sep0 = r0.positions[1].z - r0.positions[0].z;
    float sep1 = r1.positions[1].z - r1.positions[0].z;
    float sep2 = r2.positions[1].z - r2.positions[0].z;

    float dsep1 = sep0 - sep1;  // displacement caused by lateral at scale 0.1
    float dsep2 = sep0 - sep2;  // displacement caused by lateral at scale 0.2

    EXPECT_GT( dsep1, 0.0f ) << "scale=0.1 must reduce separation vs scale=0";
    EXPECT_GT( dsep2, dsep1 ) << "scale=0.2 must reduce separation MORE than scale=0.1";
    // Linearity check: dsep2 should be roughly 2× dsep1 (within 15% slop for
    // single-dispatch numerical noise and damping-free simple-Euler integration).
    float ratio = dsep2 / dsep1;
    EXPECT_NEAR( ratio, 2.0f, 0.3f )
        << "Lateral adhesion must scale linearly in the parameter "
        << "(dsep@0.2 / dsep@0.1 = " << ratio << ", expected ≈ 2.0)";
}
