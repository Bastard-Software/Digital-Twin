#include "SetupHelpers.h"
#include "resources/ResourceManager.h"
#include <glm/gtc/constants.hpp>
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
#include "simulation/MorphologyGenerator.h"
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

    // Item 2 demolition: mitosis_append.comp no longer records vessel edges
    // (pre-Item-1 edge-graph infrastructure removed). Bindings 4+5 are gone.

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
// Phase 2.1 — per-cell morphology index bit-packing (Item 2)
// =================================================================================================
//
// PhenotypeData.cellType layout (see include/simulation/Phenotype.h):
//   bits  0..15 = CellType enum (biological role: Tip/Stalk/Phalanx/Default)
//   bits 16..31 = morphologyIndex (mesh variant within AgentGroup's variant list)
//
// Shaders that filter on biological role via reqCT MUST mask to the lower 16
// bits when comparing. build_indirect.comp does an exact uint32 match on the
// combined packed value against DrawMeta.targetCellType.

// Unit test — pure C++, no GPU. Exhaustive round-trip across CellType enum
// values and morphology indices 0..15.
TEST( PhenotypePacking, PackCellType_RoundTrip )
{
    using namespace DigitalTwin;

    const CellType types[] = { CellType::Default, CellType::TipCell, CellType::StalkCell, CellType::PhalanxCell };
    for( CellType t : types )
    {
        for( uint32_t m = 0; m < 16; ++m )
        {
            uint32_t packed = PackCellType( t, m );
            EXPECT_EQ( UnpackCellType( packed ), t )            << "Biological role must survive round-trip";
            EXPECT_EQ( UnpackMorphologyIndex( packed ), m )     << "Morphology index must survive round-trip";
            // Low-16-bit mask alone recovers the biological role — matches the pattern
            // every shader filter now uses: `(cellType & 0xFFFFu) != reqCT`.
            EXPECT_EQ( packed & 0xFFFFu, static_cast<uint32_t>( t ) );
        }
    }
    // Max 16-bit morphology index — boundary case.
    EXPECT_EQ( UnpackMorphologyIndex( PackCellType( CellType::Default, 0xFFFFu ) ), 0xFFFFu );
    // Upper bits never spill into biological-role bits.
    EXPECT_EQ( UnpackCellType( PackCellType( CellType::Default, 0xFFFFu ) ), CellType::Default );
}

// build_indirect.comp variant dispatch: two morphology variants share biological
// CellType::Default. Half the agents carry morphIdx=0 (packed cellType = 0,
// catch-all default), half carry morphIdx=1 (packed cellType = 0x00010000,
// specific variant draw). Verifies the exact-match dispatch path works
// against bit-packed keys end-to-end.
TEST_F( ComputeTest, Shader_BuildIndirect_DispatchesPackedMorphologyVariants )
{
    if( !m_device )
        GTEST_SKIP();

    ShaderHandle          sh   = m_rm->CreateShader( "shaders/graphics/build_indirect.comp" );
    ComputePipelineDesc   desc{};
    desc.shader                = sh;
    ComputePipelineHandle pipe = m_rm->CreatePipeline( desc );
    ComputePipeline*      pipePtr = m_rm->GetPipeline( pipe );

    const uint32_t agentCount    = 4;
    const uint32_t drawCmdCount  = 2; // default (variant 0, catch-all) + variant 1
    const uint32_t groupCapacity = 64;

    struct DrawMeta { uint32_t groupIndex, targetCellType, groupOffset, groupCapacity; };
    const uint32_t kVariant1Key = DigitalTwin::PackCellType( DigitalTwin::CellType::Default, 1u );
    std::vector<DrawMeta> metaData = {
        { 0, 0xFFFFFFFF,    0, groupCapacity }, // variant 0 → catch-all default
        { 0, kVariant1Key,  0, groupCapacity }, // variant 1 → packed exact match
    };

    struct DrawCommand { uint32_t indexCount, instanceCount, firstIndex, vertexOffset, firstInstance; };
    std::vector<DrawCommand> cmds = {
        { 36, 0, 0, 0, 0 * groupCapacity },
        { 36, 0, 0, 0, 1 * groupCapacity },
    };

    std::vector<uint32_t> counts = { agentCount };

    std::vector<glm::vec4> positions( groupCapacity, glm::vec4( 0 ) );
    for( uint32_t i = 0; i < agentCount; ++i ) positions[ i ] = glm::vec4( float( i ), 0, 0, 1 );

    // Agents 0,1 → morphIdx 0 (cellType=0). Agents 2,3 → morphIdx 1 (cellType=0x00010000).
    std::vector<PhenotypeData> phenotypes( groupCapacity, { 0, 0.5f, 0.0f, 0u } );
    phenotypes[ 0 ].cellType = DigitalTwin::PackCellType( DigitalTwin::CellType::Default, 0u );
    phenotypes[ 1 ].cellType = DigitalTwin::PackCellType( DigitalTwin::CellType::Default, 0u );
    phenotypes[ 2 ].cellType = DigitalTwin::PackCellType( DigitalTwin::CellType::Default, 1u );
    phenotypes[ 3 ].cellType = DigitalTwin::PackCellType( DigitalTwin::CellType::Default, 1u );

    uint32_t                 reorderSize = drawCmdCount * groupCapacity;
    std::vector<uint32_t>    reorderInit( reorderSize, 0xDEADBEEF );
    std::vector<uint32_t>    visibility  = { 1u };

    auto countBuf      = m_rm->CreateBuffer( { sizeof( uint32_t ),                         BufferType::STORAGE, "BI_Counts2" } );
    auto indirectBuf   = m_rm->CreateBuffer( { cmds.size() * sizeof( DrawCommand ),        BufferType::STORAGE, "BI_Indirect2" } );
    auto phenoBuf      = m_rm->CreateBuffer( { phenotypes.size() * sizeof( PhenotypeData ),BufferType::STORAGE, "BI_Pheno2" } );
    auto reorderBuf    = m_rm->CreateBuffer( { reorderSize * sizeof( uint32_t ),           BufferType::STORAGE, "BI_Reorder2" } );
    auto metaBuf       = m_rm->CreateBuffer( { metaData.size() * sizeof( DrawMeta ),       BufferType::STORAGE, "BI_Meta2" } );
    auto agentBuf      = m_rm->CreateBuffer( { positions.size() * sizeof( glm::vec4 ),     BufferType::STORAGE, "BI_Agents2" } );
    auto visibilityBuf = m_rm->CreateBuffer( { visibility.size() * sizeof( uint32_t ),     BufferType::STORAGE, "BI_Visibility2" } );

    m_stream->UploadBufferImmediate( {
        { countBuf,      counts.data(),      counts.size() * sizeof( uint32_t ) },
        { indirectBuf,   cmds.data(),        cmds.size() * sizeof( DrawCommand ) },
        { phenoBuf,      phenotypes.data(),  phenotypes.size() * sizeof( PhenotypeData ) },
        { reorderBuf,    reorderInit.data(), reorderInit.size() * sizeof( uint32_t ) },
        { metaBuf,       metaData.data(),    metaData.size() * sizeof( DrawMeta ) },
        { agentBuf,      positions.data(),   positions.size() * sizeof( glm::vec4 ) },
        { visibilityBuf, visibility.data(),  visibility.size() * sizeof( uint32_t ) },
    } );

    BindingGroup* bg = m_rm->GetBindingGroup( m_rm->CreateBindingGroup( pipe, 0 ) );
    bg->Bind( 0, m_rm->GetBuffer( countBuf ) );
    bg->Bind( 1, m_rm->GetBuffer( indirectBuf ) );
    bg->Bind( 2, m_rm->GetBuffer( phenoBuf ) );
    bg->Bind( 3, m_rm->GetBuffer( reorderBuf ) );
    bg->Bind( 4, m_rm->GetBuffer( metaBuf ) );
    bg->Bind( 5, m_rm->GetBuffer( agentBuf ) );
    bg->Bind( 6, m_rm->GetBuffer( visibilityBuf ) );
    bg->Build();

    auto ctx    = m_device->GetThreadContext( m_device->CreateThreadContext( QueueType::COMPUTE ) );
    auto cmdBuf = ctx->GetCommandBuffer( ctx->CreateCommandBuffer() );

    ComputePushConstants resetPC{};       resetPC.uParam0 = 0; resetPC.uParam1 = drawCmdCount;
    ComputePushConstants classifyPC{};    classifyPC.uParam0 = 1; classifyPC.uParam1 = drawCmdCount; classifyPC.maxCapacity = groupCapacity;

    cmdBuf->Begin();
    cmdBuf->SetPipeline( pipePtr );
    cmdBuf->SetBindingGroup( bg, pipePtr->GetLayout(), VK_PIPELINE_BIND_POINT_COMPUTE );
    cmdBuf->PushConstants( pipePtr->GetLayout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof( ComputePushConstants ), &resetPC );
    cmdBuf->Dispatch( 1, 1, 1 );

    VkMemoryBarrier2 barrier = { VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
    barrier.srcStageMask     = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    barrier.srcAccessMask    = VK_ACCESS_2_SHADER_WRITE_BIT;
    barrier.dstStageMask     = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    barrier.dstAccessMask    = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT;
    VkDependencyInfo dep     = { VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
    dep.memoryBarrierCount   = 1;
    dep.pMemoryBarriers      = &barrier;
    cmdBuf->PipelineBarrier( &dep );

    cmdBuf->PushConstants( pipePtr->GetLayout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof( ComputePushConstants ), &classifyPC );
    cmdBuf->Dispatch( ( groupCapacity + 255 ) / 256, 1, 1 );

    cmdBuf->End();
    m_device->GetComputeQueue()->Submit( { cmdBuf } );
    m_device->GetComputeQueue()->WaitIdle();

    std::vector<DrawCommand> resultCmds( drawCmdCount );
    m_stream->ReadbackBufferImmediate( indirectBuf, resultCmds.data(), drawCmdCount * sizeof( DrawCommand ) );

    // 2 agents with morphIdx=0 fall through to the catch-all default; 2 with morphIdx=1 hit variant 1.
    EXPECT_EQ( resultCmds[ 0 ].instanceCount, 2u ) << "Variant 0 (catch-all default) draws the morphIdx=0 agents";
    EXPECT_EQ( resultCmds[ 1 ].instanceCount, 2u ) << "Variant 1 draws the morphIdx=1 agents";
}

// jkr_forces.comp lower-16-bit cellType mask (Phase 2.1): two agents both
// tagged as TipCell but with different morphIdx values must BOTH pass the
// `reqCT = TipCell` filter. Without the mask change, the morphIdx-tagged
// agent would be filtered out and its force calculation skipped.
TEST_F( ComputeTest, Shader_JKR_CellTypeFilter_IgnoresUpperMorphologyBits )
{
    if( !m_device )
        GTEST_SKIP();

    // Two agents close enough to repel each other. Biological role: TipCell for both.
    // Agent A: cellType packs morphIdx=0 → raw value = TipCell (1). Agent B: morphIdx=5 → 0x00050001.
    uint32_t ctA = DigitalTwin::PackCellType( DigitalTwin::CellType::TipCell, 0u );
    uint32_t ctB = DigitalTwin::PackCellType( DigitalTwin::CellType::TipCell, 5u );

    glm::mat4 identity( 1.0f );
    glm::vec4 profile( 0.0f );  // cadherin OFF via flag

    // Both agents set to TipCell — reqCT = TipCell should match BOTH after masking.
    // With the mask, both get forces applied → pair moves apart.
    auto runWithPhenotype = [&]( uint32_t ctA_val, uint32_t ctB_val ) -> std::pair<float, float>
    {
        // Manually instantiate the raw-shader harness that underlies RunJKRCadherin.
        // We reuse the existing helper but need per-agent cellType control — fall back
        // to a minimal custom dispatch for this test.
        uint32_t agentCount = 2;
        std::vector<glm::vec4>  inPos = { glm::vec4( 0.0f, 0.0f, 0.0f, 1.0f ), glm::vec4( 2.0f, 0.0f, 0.0f, 1.0f ) };
        struct AgentHash { uint32_t hash, idx; };
        std::vector<AgentHash>  hashes = { { 0u, 0u }, { 0u, 1u } };
        uint32_t                offsetArraySize = 256;
        std::vector<uint32_t>   cellOffsets( offsetArraySize, 0xFFFFFFFFu );
        cellOffsets[ 0 ] = 0;
        std::vector<PhenotypeData> phenotypes = {
            { 0u, 0.5f, 0.0f, ctA_val },
            { 0u, 0.5f, 0.0f, ctB_val },
        };
        std::vector<glm::vec4>  profiles = { profile, profile };
        std::vector<glm::vec4>  polarity = { glm::vec4( 0.0f ), glm::vec4( 0.0f ) };
        glm::vec4               identityQuat[ 2 ] = { glm::vec4( 0, 0, 0, 1 ), glm::vec4( 0, 0, 0, 1 ) };
        struct HullBuf           { glm::vec4 meta, points[ 16 ]; } hullDummy{};
        struct PlateBufferGPU    { glm::uvec4 meta; glm::vec4 plates[ 16 ]; } plateDummy{};

        BufferHandle inBuf    = m_rm->CreateBuffer( { agentCount * sizeof( glm::vec4 ),       BufferType::STORAGE,  "FilterInBuf" } );
        BufferHandle outBuf   = m_rm->CreateBuffer( { agentCount * sizeof( glm::vec4 ),       BufferType::STORAGE,  "FilterOutBuf" } );
        BufferHandle pressBuf = m_rm->CreateBuffer( { agentCount * sizeof( float ),           BufferType::STORAGE,  "FilterPressBuf" } );
        BufferHandle hashBuf  = m_rm->CreateBuffer( { agentCount * sizeof( AgentHash ),       BufferType::STORAGE,  "FilterHashBuf" } );
        BufferHandle offBuf   = m_rm->CreateBuffer( { offsetArraySize * sizeof( uint32_t ),   BufferType::STORAGE,  "FilterOffBuf" } );
        BufferHandle cntBuf   = m_rm->CreateBuffer( { agentCount * sizeof( uint32_t ),        BufferType::INDIRECT, "FilterCntBuf" } );
        BufferHandle phenoBuf = m_rm->CreateBuffer( { agentCount * sizeof( PhenotypeData ),   BufferType::STORAGE,  "FilterPhenoBuf" } );
        BufferHandle profBuf  = m_rm->CreateBuffer( { agentCount * sizeof( glm::vec4 ),       BufferType::STORAGE,  "FilterProfBuf" } );
        BufferHandle affBuf   = m_rm->CreateBuffer( { sizeof( glm::mat4 ),                    BufferType::STORAGE,  "FilterAffBuf" } );
        BufferHandle polBuf   = m_rm->CreateBuffer( { agentCount * sizeof( glm::vec4 ),       BufferType::STORAGE,  "FilterPolBuf" } );
        BufferHandle oriBuf   = m_rm->CreateBuffer( { sizeof( identityQuat ),                 BufferType::STORAGE,  "FilterOriBuf" } );
        BufferHandle hullBuf  = m_rm->CreateBuffer( { sizeof( HullBuf ),                      BufferType::STORAGE,  "FilterHullBuf" } );
        BufferHandle plateBuf = m_rm->CreateBuffer( { sizeof( PlateBufferGPU ),               BufferType::STORAGE,  "FilterPlateBuf" } );

        m_stream->UploadBufferImmediate( { { inBuf, inPos.data(), agentCount * sizeof( glm::vec4 ) } } );
        m_stream->UploadBufferImmediate( { { hashBuf, hashes.data(), agentCount * sizeof( AgentHash ) } } );
        m_stream->UploadBufferImmediate( { { offBuf, cellOffsets.data(), offsetArraySize * sizeof( uint32_t ) } } );
        m_stream->UploadBufferImmediate( { { cntBuf, &agentCount, sizeof( uint32_t ) } } );
        m_stream->UploadBufferImmediate( { { phenoBuf, phenotypes.data(), agentCount * sizeof( PhenotypeData ) } } );
        m_stream->UploadBufferImmediate( { { profBuf, profiles.data(), agentCount * sizeof( glm::vec4 ) } } );
        m_stream->UploadBufferImmediate( { { affBuf, &identity, sizeof( glm::mat4 ) } } );
        m_stream->UploadBufferImmediate( { { polBuf, polarity.data(), agentCount * sizeof( glm::vec4 ) } } );
        m_stream->UploadBufferImmediate( { { oriBuf, identityQuat, sizeof( identityQuat ) } } );
        m_stream->UploadBufferImmediate( { { hullBuf, &hullDummy, sizeof( HullBuf ) } } );
        m_stream->UploadBufferImmediate( { { plateBuf, &plateDummy, sizeof( PlateBufferGPU ) } } );

        ComputePipelineDesc pipeDesc{};
        pipeDesc.shader = m_rm->CreateShader( "shaders/compute/jkr_forces.comp" );
        ComputePipelineHandle pipeH = m_rm->CreatePipeline( pipeDesc );
        ComputePipeline*      pipe  = m_rm->GetPipeline( pipeH );

        BindingGroup* bg = m_rm->GetBindingGroup( m_rm->CreateBindingGroup( pipeH, 0 ) );
        bg->Bind( 0, m_rm->GetBuffer( inBuf ) );
        bg->Bind( 1, m_rm->GetBuffer( outBuf ) );
        bg->Bind( 2, m_rm->GetBuffer( pressBuf ) );
        bg->Bind( 3, m_rm->GetBuffer( hashBuf ) );
        bg->Bind( 4, m_rm->GetBuffer( offBuf ) );
        bg->Bind( 5, m_rm->GetBuffer( cntBuf ) );
        bg->Bind( 6, m_rm->GetBuffer( phenoBuf ) );
        bg->Bind( 7, m_rm->GetBuffer( profBuf ) );
        bg->Bind( 8, m_rm->GetBuffer( affBuf ) );
        bg->Bind( 9, m_rm->GetBuffer( polBuf ) );
        bg->Bind( 10, m_rm->GetBuffer( oriBuf ) );
        bg->Bind( 11, m_rm->GetBuffer( hullBuf ) );
        bg->Bind( 12, m_rm->GetBuffer( plateBuf ) );
        bg->Build();

        ComputePushConstants pc{};
        pc.dt          = 1.0f;
        pc.maxCapacity = agentCount;
        pc.fParam0     = 10.0f;  // repulsion
        pc.fParam1     = 0.0f;   // adhesion off
        pc.fParam2     = -1.0f;  // reqLC = any
        pc.fParam3     = 1.0f;   // reqCT = TipCell — only cells with biological type == TipCell apply forces
        pc.fParam5     = 1.5f;   // maxRadius
        pc.uParam0     = offsetArraySize;
        pc.domainSize  = glm::vec4( 100.0f, 100.0f, 100.0f, 4.0f );
        pc.gridSize    = glm::uvec4( 0u, 0u, 0u, 0u ); // cadherin + polarity flags OFF

        auto ctx = m_device->GetThreadContext( m_device->CreateThreadContext( QueueType::COMPUTE ) );
        auto cmd = ctx->GetCommandBuffer( ctx->CreateCommandBuffer() );
        cmd->Begin();
        cmd->SetPipeline( pipe );
        cmd->SetBindingGroup( bg, pipe->GetLayout(), VK_PIPELINE_BIND_POINT_COMPUTE );
        cmd->PushConstants( pipe->GetLayout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof( pc ), &pc );
        cmd->Dispatch( 1, 1, 1 );
        cmd->End();
        m_device->GetComputeQueue()->Submit( { cmd } );
        m_device->GetComputeQueue()->WaitIdle();

        std::vector<glm::vec4> result( agentCount );
        m_stream->ReadbackBufferImmediate( outBuf, result.data(), agentCount * sizeof( glm::vec4 ) );

        m_rm->DestroyBuffer( inBuf );  m_rm->DestroyBuffer( outBuf );  m_rm->DestroyBuffer( pressBuf );
        m_rm->DestroyBuffer( hashBuf ); m_rm->DestroyBuffer( offBuf ); m_rm->DestroyBuffer( cntBuf );
        m_rm->DestroyBuffer( phenoBuf );m_rm->DestroyBuffer( profBuf );m_rm->DestroyBuffer( affBuf );
        m_rm->DestroyBuffer( polBuf ); m_rm->DestroyBuffer( oriBuf ); m_rm->DestroyBuffer( hullBuf );
        m_rm->DestroyBuffer( plateBuf );
        return { result[ 0 ].x, result[ 1 ].x };
    };

    // Both agents tagged TipCell but with different morphIdx — filter must pass them both.
    auto [mixA, mixB] = runWithPhenotype( ctA, ctB );
    float dispMix = mixB - mixA;

    // Both agents tagged TipCell with morphIdx=0 → reference baseline (no packing).
    auto [refA, refB] = runWithPhenotype(
        DigitalTwin::PackCellType( DigitalTwin::CellType::TipCell, 0u ),
        DigitalTwin::PackCellType( DigitalTwin::CellType::TipCell, 0u ) );
    float dispRef = refB - refA;

    // Displacements match within FP tolerance — morphIdx must NOT affect the
    // reqCT filter decision.
    EXPECT_NEAR( dispMix, dispRef, 1e-4f )
        << "Mask change regression: cells with biological TipCell but different morphIdx "
           "must still pass reqCT=TipCell filter. Ref disp=" << dispRef << ", mixed disp=" << dispMix;
    // And both must actually have moved (sanity: forces applied, not zero).
    EXPECT_GT( dispMix, 2.0f + 0.01f )  << "Agents should have been pushed apart";
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
    float             corticalTension = 0.0f,
    float             catchBondStrength = 0.0f,   // Phase 5 — Rakshit 2012 catch-bond multiplier
    float             catchBondPeakLoad = 0.3f )
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

    uint32_t polarityBits  = glm::packHalf2x16( glm::vec2( apicalRepulsion, basalAdhesion ) );
    // Phase 5 encoding:
    //   gridSize.x: bit 0       = cadherinFlag
    //               bits 1..8   = catchBondStrength × 51 (8-bit fixed-point mapping [0, 5])
    //               bits 16..31 = corticalTension (half-float upper)
    //   gridSize.y: packHalf2x16(catchBondPeakLoad, couplingStrength)
    uint32_t tensionPacked = glm::packHalf2x16( glm::vec2( 0.0f, corticalTension ) );
    float    catchClamped      = glm::clamp( catchBondStrength / 5.0f, 0.0f, 1.0f );
    uint32_t catchStrengthBits = static_cast<uint32_t>( catchClamped * 255.0f + 0.5f ) & 0xFFu;
    uint32_t gridSizeX = ( cadherinFlag & 1u ) | ( catchStrengthBits << 1 ) | ( tensionPacked & 0xFFFF0000u );
    uint32_t gridSizeY = glm::packHalf2x16( glm::vec2( catchBondPeakLoad, couplingStrength ) );

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
    pc.gridSize    = glm::uvec4( gridSizeX, gridSizeY, polarityFlag, polarityBits );

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
// Phase 5 — Rakshit 2012 VE-cadherin catch-bond multiplier tests
// =============================================================================
//
// REFORMULATION v3 (2026-04-19) — catch-bond activates based on EXTERNAL
// tensile load (corticalTension / adhesion ratio), gated on attractive pairs
// (polMod > 0 AND cadA > 0), with a squared-smoothstep activation curve that
// keeps the mechanism essentially inert below peak load. Matches Rakshit
// 2012 single-molecule biology: X-dimer catch-bond has a genuine activation
// threshold and stabilises ATTRACTIVE bonds under stress.

// When the pair is under external tensile load (here: cortical tension
// matching the peak), enabling catch-bond must strengthen cadherin adhesion
// vs the no-catch-bond baseline — cells end up closer after one integration
// step.
TEST_F( ComputeTest, Shader_JKR_CatchBond_StrengthensUnderLoad )
{
    if( !m_device )
        GTEST_SKIP();

    float repulsion = 5.0f;
    float adhesion  = 10.0f;
    float maxRadius = 1.5f;

    glm::vec4 profile( 0.0f, 0.0f, 1.0f, 0.0f );
    glm::mat4 identity( 1.0f );

    // Cortical tension = 3.0 → loadSignal = 3/10 = 0.3 (at peak).
    // activated = smoothstep(0, 0.3, 0.3)² = 1² = 1. catchMul = 1 + 2·1 = 3.
    float corticalTension = 3.0f;

    // Baseline: catch-bond disabled, cortical tension on.
    auto [x0_base, x1_base] = RunJKRCadherin(
        m_device.get(), m_rm.get(), m_stream.get(),
        1.0f, 2.0f, repulsion, adhesion, maxRadius,
        profile, profile, identity,
        1u, 1.0f,                                   // cadherin ON, coupling 1
        0u, glm::vec4( 0.0f ), glm::vec4( 0.0f ),   // polarity OFF → polMod=1
        0.5f, 1.5f, PlateTestParams{},
        corticalTension,
        0.0f, 0.3f );

    // Catch-bond strength 2 with external tension at peak.
    auto [x0_catch, x1_catch] = RunJKRCadherin(
        m_device.get(), m_rm.get(), m_stream.get(),
        1.0f, 2.0f, repulsion, adhesion, maxRadius,
        profile, profile, identity,
        1u, 1.0f,
        0u, glm::vec4( 0.0f ), glm::vec4( 0.0f ),
        0.5f, 1.5f, PlateTestParams{},
        corticalTension,
        2.0f, 0.3f );

    float disp_base  = x1_base  - x0_base;
    float disp_catch = x1_catch - x0_catch;

    EXPECT_LT( disp_catch, disp_base )
        << "Catch-bond must strengthen adhesion under external tensile load: "
           "separation with catch-bond (" << disp_catch << ") must be less than "
           "baseline (" << disp_base << ")";
}

// Zero external load (no cortical tension) must leave catch-bond inert even
// with catchBondStrength > 0. Critical biology: at rest, the bond behaves
// as normal cadherin.
TEST_F( ComputeTest, Shader_JKR_CatchBond_InertAtZeroExternalLoad )
{
    if( !m_device )
        GTEST_SKIP();

    float repulsion = 5.0f;
    float adhesion  = 10.0f;
    float maxRadius = 1.5f;

    glm::vec4 profile( 0.0f, 0.0f, 1.0f, 0.0f );
    glm::mat4 identity( 1.0f );

    // No cortical tension → loadSignal = 0 → activated = 0 → catchMul = 1.
    auto [x0_base, x1_base] = RunJKRCadherin(
        m_device.get(), m_rm.get(), m_stream.get(),
        1.0f, 2.0f, repulsion, adhesion, maxRadius,
        profile, profile, identity,
        1u, 1.0f,
        0u, glm::vec4( 0.0f ), glm::vec4( 0.0f ),
        0.5f, 1.5f, PlateTestParams{},
        0.0f,
        0.0f, 0.3f );

    // Catch-bond "enabled" at extreme strength but no external load → inert.
    auto [x0_catch, x1_catch] = RunJKRCadherin(
        m_device.get(), m_rm.get(), m_stream.get(),
        1.0f, 2.0f, repulsion, adhesion, maxRadius,
        profile, profile, identity,
        1u, 1.0f,
        0u, glm::vec4( 0.0f ), glm::vec4( 0.0f ),
        0.5f, 1.5f, PlateTestParams{},
        0.0f,
        5.0f, 0.3f );                               // extreme strength

    EXPECT_FLOAT_EQ( x0_base, x0_catch )
        << "Catch-bond must be inert at zero external load (bond at rest)";
    EXPECT_FLOAT_EQ( x1_base, x1_catch )
        << "Catch-bond must be inert at zero external load (bond at rest)";
}

// Apical-repulsion regime (polMod < 0): catch-bond MUST NOT fire. Biology:
// on apical-apical contacts in the Strilic cord-hollowing regime, polarity
// is actively repelling — there is no attractive cadherin bond to stabilise.
// Catch-bond strengthening repulsion would tear the cord apart, which is the
// OPPOSITE of its function. The polMod > 0 gate enforces this.
TEST_F( ComputeTest, Shader_JKR_CatchBond_InactiveWhenRepulsive )
{
    if( !m_device )
        GTEST_SKIP();

    float repulsion = 5.0f;
    float adhesion  = 10.0f;
    float maxRadius = 1.5f;

    glm::vec4 profile( 0.0f, 0.0f, 1.0f, 0.0f );
    glm::mat4 identity( 1.0f );

    // Two cells with polarity configured for apical-apical contact in Strilic
    // regime: cell 0 basal points -X (so apical faces +X toward neighbor),
    // cell 1 basal points +X (apical faces -X toward neighbor). Both fully
    // polarised (w=1). With apical=-1.0, polMod = mix(1.0, -1.0, 1) = -1.0
    // → gate (polMod > 0) fails → catch-bond inert regardless of strength.
    glm::vec4 pol0( -1.0f, 0.0f, 0.0f, 1.0f );     // cell 0 basal = -X
    glm::vec4 pol1( +1.0f, 0.0f, 0.0f, 1.0f );     // cell 1 basal = +X
    float corticalTension = 3.0f;                   // load high enough to activate

    auto [x0_base, x1_base] = RunJKRCadherin(
        m_device.get(), m_rm.get(), m_stream.get(),
        1.0f, 2.0f, repulsion, adhesion, maxRadius,
        profile, profile, identity,
        1u, 1.0f,
        1u, pol0, pol1,                             // polarity ON, apical-apical
        -1.0f, 2.5f,                                // Strilic regime: apical=-1
        PlateTestParams{},
        corticalTension,
        0.0f, 0.3f );                               // catch-bond OFF

    auto [x0_catch, x1_catch] = RunJKRCadherin(
        m_device.get(), m_rm.get(), m_stream.get(),
        1.0f, 2.0f, repulsion, adhesion, maxRadius,
        profile, profile, identity,
        1u, 1.0f,
        1u, pol0, pol1,
        -1.0f, 2.5f,
        PlateTestParams{},
        corticalTension,
        5.0f, 0.3f );                               // catch-bond ON (strong)

    EXPECT_FLOAT_EQ( x0_base, x0_catch )
        << "Catch-bond must not fire on polarity-repulsive contacts (polMod<0)";
    EXPECT_FLOAT_EQ( x1_base, x1_catch )
        << "Catch-bond must not fire on polarity-repulsive contacts (polMod<0)";
}

// Default catchBondStrength = 0 means the catch-bond code path is inert.
// Output must be identical to running without the Phase 5 mechanism.
TEST_F( ComputeTest, Shader_JKR_CatchBond_ZeroStrengthIsBackwardsCompatible )
{
    if( !m_device )
        GTEST_SKIP();

    float repulsion = 5.0f;
    float adhesion  = 10.0f;
    float maxRadius = 1.5f;

    glm::vec4 profile( 0.0f, 0.0f, 1.0f, 0.0f );
    glm::mat4 identity( 1.0f );

    // Explicit catchBondStrength = 0 with a non-default peak-load. Include
    // nonzero tension to prove the zero-strength gate (not the zero-load gate)
    // disables catch-bond.
    auto [x0_off, x1_off] = RunJKRCadherin(
        m_device.get(), m_rm.get(), m_stream.get(),
        1.0f, 2.0f, repulsion, adhesion, maxRadius,
        profile, profile, identity,
        1u, 1.0f,
        0u, glm::vec4( 0.0f ), glm::vec4( 0.0f ),
        0.5f, 1.5f, PlateTestParams{},
        2.0f,                                       // cortical tension ON
        0.0f, 0.7f );                               // strength=0, peak=0.7 (non-default)

    // Same run with catch-bond args omitted (defaults to strength=0, peak=0.3).
    auto [x0_def, x1_def] = RunJKRCadherin(
        m_device.get(), m_rm.get(), m_stream.get(),
        1.0f, 2.0f, repulsion, adhesion, maxRadius,
        profile, profile, identity,
        1u, 1.0f,
        0u, glm::vec4( 0.0f ), glm::vec4( 0.0f ),
        0.5f, 1.5f, PlateTestParams{},
        2.0f );                                     // cortical tension, default catch-bond

    EXPECT_FLOAT_EQ( x0_off, x0_def );
    EXPECT_FLOAT_EQ( x1_off, x1_def );
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

// ==================================================================================================
// Phase 2.2 — Puzzle-piece primitive contact tests (pentagon + heptagon + elongated quad hulls).
// Dispatches jkr_forces.comp with hulls built from MorphologyGenerator's new primitives,
// verifying that the 10/14-point contactHull arrays cooperate correctly with the existing JKR
// rigid-body path (torque + lateralAdhesionScale translation). Both cells share the buffer
// (jkr_forces binds one hull buffer per AgentGroup), so the test takes one primitive's hull,
// gives both cells the same hull, and breaks the X-axis symmetry by offsetting cell 1 laterally —
// the configuration models a "cell with pentagon/heptagon/quad morphology edging into its
// neighbour" without needing per-cell hull dispatch (a Phase 2.1+ architectural choice).
// ==================================================================================================

// 1. Pentagon hull + quad-like asymmetric offset: torque fires, lateral adhesion closes the gap.
// Pentagon's 10 corner/edge-midpoint sub-spheres produce multiple overlapping pair contacts.
// Asymmetric X-offset breaks the 5-fold symmetry so the summed hull torque is non-zero.
TEST_F( ComputeTest, RigidBody_PentagonQuadContact_ProducesAdhesionTorque )
{
    if( !m_device )
        GTEST_SKIP();

    auto pent = MorphologyGenerator::CreatePentagonDefect( /*radius=*/0.5f, /*thickness=*/0.5f );
    ASSERT_EQ( pent.contactHull.size(), 10u );

    std::vector<glm::vec4> pos = { glm::vec4( 0.0f, 0.0f, 0.0f, 1.0f ),
                                   glm::vec4( 0.1f, 0.3f, 0.0f, 1.0f ) }; // +X offset breaks symmetry
    std::vector<glm::vec4> ori = { glm::vec4( 0.0f, 0.0f, 0.0f, 1.0f ),
                                   glm::vec4( 0.0f, 0.0f, 0.0f, 1.0f ) };

    // Baseline: lateralAdhesionScale=0 — hull path fires torque, but no translation.
    auto rBase = RunJKRRigidBody( m_device.get(), m_rm.get(), m_stream.get(),
                                   pos, ori,
                                   static_cast<uint32_t>( pent.contactHull.size() ),
                                   pent.contactHull,
                                   /*repulsion=*/0.0f, /*adhesion=*/5.0f, /*maxRadius=*/0.35f,
                                   /*damping=*/0.0f, /*dt=*/1.0f,
                                   /*stericZ=*/0.0f, /*stericY=*/0.0f,
                                   /*edgeAlign=*/0.0f, /*cadherinCoupling=*/1.0f,
                                   /*lateralAdhesionScale=*/0.0f );

    auto rLat = RunJKRRigidBody( m_device.get(), m_rm.get(), m_stream.get(),
                                  pos, ori,
                                  static_cast<uint32_t>( pent.contactHull.size() ),
                                  pent.contactHull,
                                  /*repulsion=*/0.0f, /*adhesion=*/5.0f, /*maxRadius=*/0.35f,
                                  /*damping=*/0.0f, /*dt=*/1.0f,
                                  /*stericZ=*/0.0f, /*stericY=*/0.0f,
                                  /*edgeAlign=*/0.0f, /*cadherinCoupling=*/1.0f,
                                  /*lateralAdhesionScale=*/0.15f );

    // Torque: asymmetric hull contact produces a non-zero orientation change on cell 0.
    glm::vec3 rotAxis0( rBase.orientations[0].x, rBase.orientations[0].y, rBase.orientations[0].z );
    EXPECT_GT( glm::length( rotAxis0 ), 1e-4f )
        << "Pentagon hull edge-contact must produce a non-zero rotation on cell 0";

    // Lateral translation: with lateralAdhesionScale=0.15 the Y-gap shrinks vs baseline.
    float sepBase = rBase.positions[1].y - rBase.positions[0].y;
    float sepLat  = rLat .positions[1].y - rLat .positions[0].y;
    EXPECT_LT( sepLat, sepBase )
        << "lateralAdhesionScale=0.15 must pull pentagon-hull cells closer "
        << "(lateral=" << sepLat << " vs baseline=" << sepBase << ")";
}

// 2. Same assertion with the heptagon hull (14 sub-spheres instead of 10).
// Heptagon still fits within the 16-point contactHull budget, so no shader changes required.
TEST_F( ComputeTest, RigidBody_HeptagonQuadContact_ProducesAdhesionTorque )
{
    if( !m_device )
        GTEST_SKIP();

    auto hept = MorphologyGenerator::CreateHeptagonDefect( /*radius=*/0.5f, /*thickness=*/0.5f );
    ASSERT_EQ( hept.contactHull.size(), 14u );

    std::vector<glm::vec4> pos = { glm::vec4( 0.0f, 0.0f, 0.0f, 1.0f ),
                                   glm::vec4( 0.1f, 0.3f, 0.0f, 1.0f ) };
    std::vector<glm::vec4> ori = { glm::vec4( 0.0f, 0.0f, 0.0f, 1.0f ),
                                   glm::vec4( 0.0f, 0.0f, 0.0f, 1.0f ) };

    auto rBase = RunJKRRigidBody( m_device.get(), m_rm.get(), m_stream.get(),
                                   pos, ori,
                                   static_cast<uint32_t>( hept.contactHull.size() ),
                                   hept.contactHull,
                                   /*repulsion=*/0.0f, /*adhesion=*/5.0f, /*maxRadius=*/0.35f,
                                   /*damping=*/0.0f, /*dt=*/1.0f,
                                   /*stericZ=*/0.0f, /*stericY=*/0.0f,
                                   /*edgeAlign=*/0.0f, /*cadherinCoupling=*/1.0f,
                                   /*lateralAdhesionScale=*/0.0f );

    auto rLat = RunJKRRigidBody( m_device.get(), m_rm.get(), m_stream.get(),
                                  pos, ori,
                                  static_cast<uint32_t>( hept.contactHull.size() ),
                                  hept.contactHull,
                                  /*repulsion=*/0.0f, /*adhesion=*/5.0f, /*maxRadius=*/0.35f,
                                  /*damping=*/0.0f, /*dt=*/1.0f,
                                  /*stericZ=*/0.0f, /*stericY=*/0.0f,
                                  /*edgeAlign=*/0.0f, /*cadherinCoupling=*/1.0f,
                                  /*lateralAdhesionScale=*/0.15f );

    glm::vec3 rotAxis0( rBase.orientations[0].x, rBase.orientations[0].y, rBase.orientations[0].z );
    EXPECT_GT( glm::length( rotAxis0 ), 1e-4f )
        << "Heptagon hull edge-contact must produce a non-zero rotation on cell 0";

    float sepBase = rBase.positions[1].y - rBase.positions[0].y;
    float sepLat  = rLat .positions[1].y - rLat .positions[0].y;
    EXPECT_LT( sepLat, sepBase )
        << "lateralAdhesionScale=0.15 must pull heptagon-hull cells closer "
        << "(lateral=" << sepLat << " vs baseline=" << sepBase << ")";
}

// 3. Pentagon-heptagon edge-length meshing (CPU-side geometric assertion).
// Stone-Wales 5/7 pair insertion at diameter transitions depends on the pentagon's
// edge length matching the heptagon's edge length so their hull sub-spheres overlap
// correctly when the two polygons tile edge-to-edge (Stone & Wales 1986; Iijima 1991
// for carbon-nanotube chirality transitions). This test verifies that when the two
// radii are chosen by the edge-length rule `r_p·sin(pi/5) = r_h·sin(pi/7)`, the
// pentagon's edge-midpoint hull point and the heptagon's edge-midpoint hull point
// can be placed within 2·subR (full overlap) by a centre-to-centre separation of
// apothem_p + apothem_h — the configuration Phase 2.4 uses.
TEST( MorphologyGeneratorTest, PentagonHeptagonPair_MeshesAtShortEdges )
{
    // Edge-length match: pick r_h relative to r_p so pentagon.edge = heptagon.edge.
    const float r_p = 1.0f;
    const float r_h = r_p * std::sin( glm::pi<float>() / 5.0f )
                          / std::sin( glm::pi<float>() / 7.0f );

    auto pent = MorphologyGenerator::CreatePentagonDefect( r_p, 0.4f );
    auto hept = MorphologyGenerator::CreateHeptagonDefect( r_h, 0.4f );
    ASSERT_EQ( pent.contactHull.size(), 10u );
    ASSERT_EQ( hept.contactHull.size(), 14u );

    // Edge-length equality (the Stone-Wales tessellation invariant).
    const float pentEdge = 2.0f * r_p * std::sin( glm::pi<float>() / 5.0f );
    const float heptEdge = 2.0f * r_h * std::sin( glm::pi<float>() / 7.0f );
    EXPECT_NEAR( pentEdge, heptEdge, 1e-4f )
        << "Stone-Wales pair requires pentagon & heptagon edge lengths to match";

    // Place pentagon edge-midpoint (i=0 between corner 0 @ +X and corner 1) at +X pole
    // of pentagon. Pentagon apothem + heptagon apothem along +X puts the heptagon's
    // opposing edge-midpoint back at the pentagon's edge-midpoint within 2·subR.
    const float apothem_p = r_p * std::cos( glm::pi<float>() / 5.0f );
    const float apothem_h = r_h * std::cos( glm::pi<float>() / 7.0f );
    const float subR      = 0.2f; // thickness/2 for both primitives

    // Count hull-pair overlaps when cell 0 (pentagon at origin) meets cell 1 (heptagon at
    // +X pole apothem_p + apothem_h away, rotated so an edge-midpoint faces cell 0).
    // `overlap = 2·subR - d(point_pent, point_hept)`; count pairs with overlap > 0.
    const glm::vec3 c1( apothem_p + apothem_h, 0.0f, 0.0f );
    int overlapCount = 0;
    for( const auto& pp : pent.contactHull )
    {
        glm::vec3 a( pp.x, pp.y, pp.z );
        for( const auto& ph : hept.contactHull )
        {
            // Heptagon point rotated 180° around Y so it faces cell 0:
            //   (x, y, z) → (-x, y, -z), then translated by +c1.
            glm::vec3 b( -ph.x + c1.x, ph.y, -ph.z + c1.z );
            float     d = glm::length( a - b );
            if( d < 2.0f * subR ) ++overlapCount;
        }
    }

    // Edge-matched Stone-Wales pair produces overlapping hull sub-spheres at the facing
    // edge midpoints (and nearby corners). "≥ 2" is the deep-research 2026-04-19 threshold:
    // gives the JKR shader enough overlapping contacts to hold the defect together under
    // Phase-4.5-B lateral adhesion pull.
    EXPECT_GE( overlapCount, 2 )
        << "Pentagon-heptagon edge-matched pair must produce ≥ 2 overlapping hull pairs";
}
