#include "SetupHelpers.h"
#include "resources/ResourceManager.h"
#include "resources/StreamingManager.h"
#include "rhi/BindingGroup.h"
#include "rhi/Buffer.h"
#include "rhi/Device.h"
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
std::vector<float> ReadbackGrid( DigitalTwin::SimulationState& state, uint32_t fieldIndex, uint32_t textureIndexToRead,
                                 DigitalTwin::ResourceManager* rm, DigitalTwin::Device* device )
{
    auto&        field       = state.gridFields[ fieldIndex ];
    size_t       size        = field.width * field.height * field.depth * sizeof( float );
    BufferHandle readbackBuf = rm->CreateBuffer( { size, BufferType::READBACK, "Readback" } );

    auto ctxHandle = device->CreateThreadContext( QueueType::GRAPHICS );
    auto ctx       = device->GetThreadContext( ctxHandle );
    auto cmd       = ctx->GetCommandBuffer( ctx->CreateCommandBuffer() );

    cmd->Begin();
    VkBufferImageCopy region{};
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.layerCount = 1;
    region.imageExtent                 = { field.width, field.height, field.depth };

    cmd->CopyImageToBuffer( rm->GetTexture( field.textures[ textureIndexToRead ] ), VK_IMAGE_LAYOUT_GENERAL, rm->GetBuffer( readbackBuf ), 1,
                            &region );
    cmd->End();

    device->GetGraphicsQueue()->Submit( { cmd } );
    device->GetGraphicsQueue()->WaitIdle();

    std::vector<float> data( field.width * field.height * field.depth );
    rm->GetBuffer( readbackBuf )->Read( data.data(), size, 0 );
    rm->DestroyBuffer( readbackBuf );
    return data;
}

class ComputeEngineTest : public ::testing::Test
{
protected:
    Scope<MemorySystem>    m_memory;
    Scope<FileSystem>      m_fileSystem;
    Scope<RHI>             m_rhi;
    Scope<Device>          m_device;
    Scope<ResourceManager> m_rm;
    std::string            m_testShader = "test_compute_graph.comp";

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

        std::ofstream out( m_testShader );
        out << "#version 450\n"
            << "layout(local_size_x=256) in;\n"
            << "layout(set=0, binding=0) buffer P { float pos[]; };\n"
            << "layout(push_constant) uniform PC { float dt; float totalTime; float speed; uint offset; uint count; } pc;\n"
            << "void main() { if(gl_GlobalInvocationID.x < pc.count) { pos[pc.offset + gl_GlobalInvocationID.x] += pc.speed; } }";
        out.close();
    }

    void TearDown() override
    {
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

        std::filesystem::remove( m_testShader );
        std::filesystem::remove( m_testShader + ".spv" );
    }
};

// Verify that the task frequency limiter works correctly
TEST_F( ComputeEngineTest, TaskFrequencyExecution )
{
    ComputePushConstants pc{};
    ComputeTask          task( nullptr, nullptr, nullptr, 10.0f /* 10 Hz */, pc, glm::uvec3( 1 ) );

    EXPECT_FALSE( task.ShouldExecute( 0.05f ) );
    EXPECT_TRUE( task.ShouldExecute( 0.06f ) );
    EXPECT_FALSE( task.ShouldExecute( 0.05f ) );
}

// Full graph execution test via Dispatcher
TEST_F( ComputeEngineTest, GraphExecution )
{
    if( !m_device )
        GTEST_SKIP();

    ShaderHandle shaderHandle = m_rm->CreateShader( m_testShader );

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

    ComputePushConstants pc{ 0.16f, 1.0f, 1.0f, 0, 100 };
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
}

TEST_F( ComputeEngineTest, GridFieldBuilderAllocationAndUpload )
{
    // 1. Setup Blueprint with Domain and GridFields
    SimulationBlueprint blueprint;

    // Domain: 100x100x100 micrometers. Voxel size: 2 micrometers
    // Expected resolution: 50x50x50 voxels
    blueprint.SetDomainSize( glm::vec3( 100.0f ), 2.0f );

    blueprint.AddGridField( "Oxygen" )
        .SetInitializer( DigitalTwin::GridInitializer::Gaussian( glm::vec3( 0 ), 10.0f, 100.0f ) )
        .SetDiffusionCoefficient( 0.5f )
        .SetComputeHz( 120.0f );

    // We also need StreamingManager for initial texture upload
    auto streamingManager = CreateScope<StreamingManager>( m_device.get(), m_rm.get() );
    ASSERT_EQ( streamingManager->Initialize(), Result::SUCCESS );

    // 2. Build state
    DigitalTwin::SimulationBuilder builder( m_rm.get(), streamingManager.get() );
    SimulationState                state = builder.Build( blueprint );

    // 3. Verify allocations
    ASSERT_EQ( state.gridFields.size(), 1 );

    auto& oxygenState = state.gridFields[ 0 ];
    ASSERT_EQ( oxygenState.width, 50 );
    ASSERT_EQ( oxygenState.height, 50 );
    ASSERT_EQ( oxygenState.depth, 50 );

    // Ensure ping-pong 3D textures are valid and allocated on device
    ASSERT_TRUE( oxygenState.textures[ 0 ].IsValid() );
    ASSERT_TRUE( oxygenState.textures[ 1 ].IsValid() );

    // Cleanup GPU resources gracefully
    state.Destroy( m_rm.get() );
}

TEST_F( ComputeEngineTest, PhysicsAndInteractions_DiffusionConsumption )
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

    auto streamingManager = CreateScope<StreamingManager>( m_device.get(), m_rm.get() );
    ASSERT_EQ( streamingManager->Initialize(), Result::SUCCESS );

    DigitalTwin::SimulationBuilder builder( m_rm.get(), streamingManager.get() );
    SimulationState                state = builder.Build( blueprint );

    // Grab execution graph
    auto compCtxHandle = m_device->CreateThreadContext( QueueType::COMPUTE );
    auto compCtx       = m_device->GetThreadContext( compCtxHandle );
    auto compCmd       = compCtx->GetCommandBuffer( compCtx->CreateCommandBuffer() );

    GraphDispatcher dispatcher;

    // Ping
    compCmd->Begin();
    dispatcher.Dispatch( &state.computeGraph, compCmd, nullptr, 1.0f, 1.0f, 0 );
    dispatcher.Dispatch( &state.computeGraph, compCmd, nullptr, 1.0f, 2.0f, 1 );
    compCmd->End();

    m_device->GetComputeQueue()->Submit( { compCmd } );
    m_device->GetComputeQueue()->WaitIdle();

    // Readback the grid
    std::vector<float> gridData = ReadbackGrid( state, 0, 0, m_rm.get(), m_device.get() );

    // Center is at 5,5,5 in a 10x10x10 grid (using 0-based index)
    uint32_t centerIdx = 5 + 5 * 10 + 5 * 100;

    // Original was 100.0. After 1 effective integration of consumption, it should be heavily depleted
    // (It won't be exactly 90.0 because of diffusion spreading it slightly, but definitely less than 95.0)
    EXPECT_LT( gridData[ centerIdx ], 95.0f );

    // A voxel far away (e.g. 0,0,0) should still be relatively untouched (close to 100)
    EXPECT_GT( gridData[ 0 ], 99.0f );

    state.Destroy( m_rm.get() );
}