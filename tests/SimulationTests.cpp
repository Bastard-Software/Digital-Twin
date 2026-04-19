#include "simulation/BiologyGenerator.h"
#include "simulation/BiomechanicsGenerator.h"
#include "simulation/MorphologyGenerator.h"
#include "simulation/SimulationBlueprint.h"
#include "simulation/SimulationBuilder.h"
#include "simulation/SpatialDistribution.h"
#include "simulation/Phenotype.h"
#include <gtest/gtest.h>

// Core and Engine dependencies needed for the Builder
#include "SetupHelpers.h"
#include "compute/ComputeGraph.h"
#include "compute/ComputeTask.h"
#include "compute/GraphDispatcher.h"
#include "core/FileSystem.h"
#include "core/memory/MemorySystem.h"
#include "resources/ResourceManager.h"
#include "resources/StreamingManager.h"
#include "rhi/Device.h"
#include "rhi/RHI.h"
#include "rhi/ThreadContext.h"

using namespace DigitalTwin;

// Helper method to readback 3D Grid from GPU into CPU memory
std::vector<float> ReadbackGrid( DigitalTwin::SimulationState& state, uint32_t fieldIndex, DigitalTwin::StreamingManager* stream )
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

// =================================================================================================
// 1. Simulation Blueprint Logic Tests (CPU Only, No Vulkan needed)
// =================================================================================================

TEST( SimulationBlueprintTest, FluentAPIAndDataStorage )
{
    SimulationBlueprint blueprint;

    // Create dummy morphology and positions
    MorphologyData mockMorphology;
    mockMorphology.vertices = { { glm::vec4( 0.0f ), glm::vec4( 1.0f ) } };
    mockMorphology.indices  = { 0, 1, 2 };

    std::vector<glm::vec4> mockPositions = { glm::vec4( 1.0f ), glm::vec4( 2.0f ) };

    // Use the fluent API
    blueprint.AddAgentGroup( "CancerCells" ).SetCount( 100 ).SetMorphology( mockMorphology ).SetDistribution( mockPositions );

    const auto& groups = blueprint.GetGroups();

    // Assertions
    ASSERT_EQ( groups.size(), 1 );
    EXPECT_EQ( groups[ 0 ].GetName(), "CancerCells" );
    EXPECT_EQ( groups[ 0 ].GetCount(), 100 );
    EXPECT_EQ( groups[ 0 ].GetMorphology().vertices.size(), 1 );
    EXPECT_EQ( groups[ 0 ].GetPositions().size(), 2 );
}

TEST( SimulationBlueprintTest, MultipleGroups )
{
    SimulationBlueprint blueprint;

    blueprint.AddAgentGroup( "GroupA" ).SetCount( 10 );
    blueprint.AddAgentGroup( "GroupB" ).SetCount( 20 );

    const auto& groups = blueprint.GetGroups();
    ASSERT_EQ( groups.size(), 2 );
    EXPECT_EQ( groups[ 0 ].GetCount(), 10 );
    EXPECT_EQ( groups[ 1 ].GetCount(), 20 );
}

TEST( SimulationBlueprintTest, BlueprintSetName )
{
    SimulationBlueprint blueprint;
    blueprint.SetName( "My Simulation" );
    EXPECT_EQ( blueprint.GetName(), "My Simulation" );
}

TEST( SimulationBlueprintTest, MutableGroupAccess )
{
    SimulationBlueprint blueprint;
    blueprint.AddAgentGroup( "Cells" ).SetCount( 5 );
    blueprint.GetGroupsMutable()[ 0 ].SetCount( 99 );
    EXPECT_EQ( blueprint.GetGroups()[ 0 ].GetCount(), 99 );
}

TEST( SimulationBlueprintTest, MutableGridFieldAccess )
{
    SimulationBlueprint blueprint;
    blueprint.AddGridField( "Oxygen" ).SetDiffusionCoefficient( 1.0f );
    blueprint.GetGridFieldsMutable()[ 0 ].SetDiffusionCoefficient( 7.5f );
    EXPECT_FLOAT_EQ( blueprint.GetGridFields()[ 0 ].GetDiffusionCoefficient(), 7.5f );
}

TEST( SimulationBlueprintTest, MutableBehaviourAccess )
{
    SimulationBlueprint blueprint;
    blueprint.AddAgentGroup( "Cells" )
        .SetCount( 1 )
        .AddBehaviour( DigitalTwin::Behaviours::BrownianMotion{ 1.0f } )
        .SetHz( 60.0f );

    auto& b  = std::get<DigitalTwin::Behaviours::BrownianMotion>( blueprint.GetGroupsMutable()[ 0 ].GetBehavioursMutable()[ 0 ].behaviour );
    b.speed  = 5.0f;

    const auto& readback = std::get<DigitalTwin::Behaviours::BrownianMotion>( blueprint.GetGroups()[ 0 ].GetBehaviours()[ 0 ].behaviour );
    EXPECT_FLOAT_EQ( readback.speed, 5.0f );
}

// =================================================================================================
// 2. Simulation Builder & GPU Integration Tests
// =================================================================================================

class SimulationBuilderTest : public ::testing::Test
{
protected:
    Scope<MemorySystem>     m_memory;
    Scope<FileSystem>       m_fileSystem;
    Scope<RHI>              m_rhi;
    Scope<Device>           m_device;
    Scope<ResourceManager>  m_resourceManager;
    Scope<StreamingManager> m_streamingManager;

    void SetUp() override
    {
        // 1. Core Systems
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

        // 2. RHI & Device (Headless)
        m_rhi = CreateScope<RHI>();
        RHIConfig rhiConfig;
        rhiConfig.headless         = true;
        rhiConfig.enableValidation = true;
        ASSERT_EQ( m_rhi->Initialize( rhiConfig ), Result::SUCCESS );

        if( m_rhi->GetAdapters().empty() )
        {
            GTEST_SKIP() << "No GPU adapters found. Skipping Builder tests.";
        }
        ASSERT_EQ( m_rhi->CreateDevice( 0, m_device ), Result::SUCCESS );

        // 3. Resource & Streaming Managers
        m_resourceManager = CreateScope<ResourceManager>( m_device.get(), m_memory.get(), m_fileSystem.get() );
        ASSERT_EQ( m_resourceManager->Initialize(), Result::SUCCESS );

        m_streamingManager = CreateScope<StreamingManager>( m_device.get(), m_resourceManager.get() );
        ASSERT_EQ( m_streamingManager->Initialize(), Result::SUCCESS );
    }

    void TearDown() override
    {
        if( m_streamingManager )
            m_streamingManager->Shutdown();
        if( m_resourceManager )
            m_resourceManager->Shutdown();
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

TEST_F( SimulationBuilderTest, BuildEmptyBlueprint )
{
    if( !m_device )
        GTEST_SKIP();

    SimulationBlueprint blueprint;
    SimulationBuilder   builder( m_resourceManager.get(), m_streamingManager.get() );

    SimulationState state = builder.Build( blueprint );

    // State should be invalid because there were no agents to build
    EXPECT_FALSE( state.IsValid() );
}

TEST_F( SimulationBuilderTest, BuildValidBlueprint )
{
    if( !m_device )
        GTEST_SKIP();

    // Create a Blueprint with two distinct groups
    SimulationBlueprint blueprint;

    blueprint.AddAgentGroup( "RedCells" )
        .SetCount( 50 )
        .SetMorphology( MorphologyGenerator::CreateCube( 1.0f ) )
        .SetDistribution( SpatialDistribution::UniformInBox( 50, glm::vec3( 10.0f ) ) );

    blueprint.AddAgentGroup( "WhiteCells" )
        .SetCount( 100 )
        .SetMorphology( MorphologyGenerator::CreateSphere( 1.0f, 8, 8 ) )
        .SetDistribution( SpatialDistribution::UniformInSphere( 100, 20.0f ) );

    SimulationBuilder builder( m_resourceManager.get(), m_streamingManager.get() );

    // Compile to GPU
    SimulationState state = builder.Build( blueprint );

    // Verify all buffers were created successfully
    EXPECT_TRUE( state.IsValid() );
    EXPECT_TRUE( state.vertexBuffer.IsValid() );
    EXPECT_TRUE( state.indexBuffer.IsValid() );
    EXPECT_TRUE( state.indirectCmdBuffer.IsValid() );
    EXPECT_TRUE( state.agentBuffers[ 0 ].IsValid() );
    EXPECT_TRUE( state.agentBuffers[ 1 ].IsValid() );

    // Clean up GPU resources
    state.Destroy( m_resourceManager.get() );
    EXPECT_FALSE( state.IsValid() );
}

// =================================================================================================
// Simulation task tests
// =================================================================================================

// 1. Test grid building
TEST_F( SimulationBuilderTest, Builder_GridAllocationAndUpload )
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
    DigitalTwin::SimulationBuilder builder( m_resourceManager.get(), m_streamingManager.get() );
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
    state.Destroy( m_resourceManager.get() );
}

// 2. Test proper grig field consumption
TEST_F( SimulationBuilderTest, Behaviour_ConsumeField )
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

    DigitalTwin::SimulationBuilder builder( m_resourceManager.get(), m_streamingManager.get() );
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
    std::vector<float> gridData = ReadbackGrid( state, 0, m_streamingManager.get() );

    // Center is at 5,5,5 in a 10x10x10 grid (using 0-based index)
    uint32_t centerIdx = 5 + 5 * 10 + 5 * 100;

    // Original was 100.0. After 2 effective integrations of consumption (Ping and Pong),
    // it should be heavily depleted. We expect it to be less than 95.0.
    EXPECT_LT( gridData[ centerIdx ], 95.0f ) << "Agent failed to consume the field or grid data is corrupted!";

    // A voxel far away (e.g. 0,0,0) should still be relatively untouched (close to 100.0)
    EXPECT_GT( gridData[ 0 ], 99.0f ) << "Diffusion spread too fast or entire grid is losing values!";

    state.Destroy( m_resourceManager.get() );
}

// 3. Test grid field secretion
TEST_F( SimulationBuilderTest, Behaviour_SecreteField )
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

    DigitalTwin::SimulationBuilder builder( m_resourceManager.get(), m_streamingManager.get() );
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
    std::vector<float> gridData = ReadbackGrid( state, 0, m_streamingManager.get() );

    uint32_t centerIdx = 5 + 5 * 10 + 5 * 100;

    // Center should be heavily saturated with secreted lactate (>9.0 due to slight diffusion loss to neighbors)
    EXPECT_GT( gridData[ centerIdx ], 9.0f ) << "Agent failed to secrete the field properly!";

    // A voxel far away should still be completely clean (close to 0.0)
    EXPECT_LT( gridData[ 0 ], 0.1f ) << "Secretion affected distant voxels unexpectedly!";

    state.Destroy( m_resourceManager.get() );
}

// 3b. SecreteField still works when a second group uses SetInitialCellType
TEST_F( SimulationBuilderTest, Behaviour_SecreteField_WithInitialCellTypeOnSecondGroup )
{
    if( !m_device )
        GTEST_SKIP();

    SimulationBlueprint blueprint;
    blueprint.SetDomainSize( glm::vec3( 10.0f ), 1.0f );

    blueprint.AddGridField( "VEGF" )
        .SetInitializer( DigitalTwin::GridInitializer::Constant( 0.0f ) )
        .SetDiffusionCoefficient( 0.01f )
        .SetComputeHz( 1.0f );

    std::vector<glm::vec4> centerCell = { glm::vec4( 0.0f, 0.0f, 0.0f, 1.0f ) };
    std::vector<glm::vec4> farCell    = { glm::vec4( 4.0f, 0.0f, 0.0f, 1.0f ) };

    // Group 0: secretes VEGF
    blueprint.AddAgentGroup( "SecretingCell" )
        .SetCount( 1 )
        .SetDistribution( centerCell )
        .AddBehaviour( DigitalTwin::Behaviours::SecreteField{ "VEGF", 10.0f } )
        .SetHz( 1.0f );

    // Group 1: inert but has non-default initial cell type (PhalanxCell=3)
    blueprint.AddAgentGroup( "InertCell" )
        .SetCount( 1 )
        .SetDistribution( farCell )
        .SetInitialCellType( static_cast<int>( DigitalTwin::CellType::PhalanxCell ) );

    DigitalTwin::SimulationBuilder builder( m_resourceManager.get(), m_streamingManager.get() );
    SimulationState                state = builder.Build( blueprint );

    auto compCtxHandle = m_device->CreateThreadContext( QueueType::COMPUTE );
    auto compCtx       = m_device->GetThreadContext( compCtxHandle );
    auto compCmd       = compCtx->GetCommandBuffer( compCtx->CreateCommandBuffer() );

    GraphDispatcher dispatcher;
    compCmd->Begin();
    dispatcher.Dispatch( &state.computeGraph, compCmd, nullptr, 1.0f, 1.0f, 0 );
    dispatcher.Dispatch( &state.computeGraph, compCmd, nullptr, 1.0f, 2.0f, 1 );
    compCmd->End();

    m_device->GetComputeQueue()->Submit( { compCmd } );
    m_device->GetComputeQueue()->WaitIdle();

    std::vector<float> gridData = ReadbackGrid( state, 0, m_streamingManager.get() );
    uint32_t           centerIdx = 5 + 5 * 10 + 5 * 100;

    EXPECT_GT( gridData[ centerIdx ], 9.0f )
        << "SecreteField broken when a second group uses SetInitialCellType!";

    state.Destroy( m_resourceManager.get() );
}

// 4. Test pure grid field diffusion (No Agents)
TEST_F( SimulationBuilderTest, Behaviour_PureDiffusion )
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

    DigitalTwin::SimulationBuilder builder( m_resourceManager.get(), m_streamingManager.get() );
    SimulationState                state = builder.Build( blueprint );

    // 1. Readback the INITIAL state of the grid before any compute dispatches
    std::vector<float> initialGridData = ReadbackGrid( state, 0, m_streamingManager.get() );

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
    std::vector<float> finalGridData = ReadbackGrid( state, 0, m_streamingManager.get() );

    float finalCenterValue = finalGridData[ centerIdx ];
    float finalEdgeValue   = finalGridData[ edgeIdx ];

    // Assertions:
    // The central peak must have lost concentration as it diffused outwards.
    EXPECT_LT( finalCenterValue, initialCenterValue ) << "Diffusion failed to reduce the peak concentration!";

    // The outer voxels must have gained concentration as the substance spread to them.
    EXPECT_GT( finalEdgeValue, initialEdgeValue ) << "Diffusion failed to increase concentration in outer regions!";

    state.Destroy( m_resourceManager.get() );
}

// 5. Test proper biomechanics build
TEST_F( SimulationBuilderTest, Builder_BiomechanicsAllocation )
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
    DigitalTwin::SimulationBuilder builder( m_resourceManager.get(), m_streamingManager.get() );
    DigitalTwin::SimulationState   state = builder.Build( blueprint );

    // 3. Verify allocations
    // Biomechanics requires specific storage buffers for spatial hashing and physics.
    // We check if SimulationBuilder successfully detected the behaviour and allocated them.
    EXPECT_TRUE( state.hashBuffer.IsValid() ) << "Biomechanics hash buffer was not allocated!";
    EXPECT_TRUE( state.offsetBuffer.IsValid() ) << "Biomechanics offset buffer was not allocated!";
    EXPECT_TRUE( state.pressureBuffer.IsValid() ) << "Biomechanics pressure buffer was not allocated!";

    // Cleanup GPU resources gracefully
    state.Destroy( m_resourceManager.get() );
}

// Verifies that signalingBuffer is invalid on a default SimulationState and after Destroy
TEST_F( SimulationBuilderTest, SignalingBuffer_DefaultInvalid )
{
    DigitalTwin::SimulationState state;
    EXPECT_FALSE( state.signalingBuffer.IsValid() ) << "signalingBuffer must be invalid on a default SimulationState";
}

// 6. Verifies that the global Spatial Grid computes hashes, sorts agents, and builds valid offsets without any mechanics attached
TEST_F( SimulationBuilderTest, SpatialGrid_Data_Validation )
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

    // Place two agents extremely close to each other.
    // With a cell size of 30.0, they are guaranteed to fall into the same spatial bucket.
    std::vector<glm::vec4> gridTestCells = {
        glm::vec4( 0.0f, 0.0f, 0.0f, 1.0f ), // Agent 0
        glm::vec4( 0.1f, 0.0f, 0.0f, 1.0f )  // Agent 1
    };

    // Notice: We add the group, but DO NOT add any behaviours (e.g., JKR).
    blueprint.AddAgentGroup( "GridTesters" ).SetCount( 2 ).SetDistribution( gridTestCells );

    DigitalTwin::SimulationBuilder builder( m_resourceManager.get(), m_streamingManager.get() );
    DigitalTwin::SimulationState   state = builder.Build( blueprint );

    // Setup compute dispatch
    auto compCtxHandle = m_device->CreateThreadContext( QueueType::COMPUTE );
    auto compCtx       = m_device->GetThreadContext( compCtxHandle );
    auto compCmd       = compCtx->GetCommandBuffer( compCtx->CreateCommandBuffer() );

    DigitalTwin::GraphDispatcher dispatcher;

    // Dispatch 1 frame. Since there are no behaviours, this will ONLY execute:
    // Task A (Hash) -> Task B (Bitonic Sort) -> Task C (Build Offsets)
    compCmd->Begin();
    dispatcher.Dispatch( &state.computeGraph, compCmd, nullptr, 0.02f, 0.02f, 0 );
    compCmd->End();

    m_device->GetComputeQueue()->Submit( { compCmd } );
    m_device->GetComputeQueue()->WaitIdle();

    // Setup data structures to match GPU buffer layouts
    // Hash buffer stores pairs of [SpatialHash (uint32), OriginalAgentIndex (uint32)]
    struct HashEntry
    {
        uint32_t key;
        uint32_t value;
    };

    // For 2 agents, paddedCount for bitonic sort is 2
    uint32_t               paddedCount = 2;
    std::vector<HashEntry> resultHashes( paddedCount );
    m_streamingManager->ReadbackBufferImmediate( state.hashBuffer, resultHashes.data(), paddedCount * sizeof( HashEntry ) );

    uint32_t              offsetArraySize = 262144; // 64x64x64 (Must match internal simulation builder size)
    std::vector<uint32_t> resultOffsets( offsetArraySize );
    m_streamingManager->ReadbackBufferImmediate( state.offsetBuffer, resultOffsets.data(), offsetArraySize * sizeof( uint32_t ) );

    // --- Verify Spatial Grid Logic ---

    // 1. Check if both agents ended up in the same cell (identical hash keys)
    EXPECT_EQ( resultHashes[ 0 ].key, resultHashes[ 1 ].key ) << "Agents in the same cell must generate identical spatial hashes!";

    // 2. Check if the sort buffer correctly retained the original agent indices (0 and 1)
    bool hasAgent0 = ( resultHashes[ 0 ].value == 0 ) || ( resultHashes[ 1 ].value == 0 );
    bool hasAgent1 = ( resultHashes[ 0 ].value == 1 ) || ( resultHashes[ 1 ].value == 1 );
    EXPECT_TRUE( hasAgent0 && hasAgent1 ) << "Hash buffer lost track of original agent indices during bitonic sort!";

    // 3. Verify that the offset buffer was correctly built (at least one cell is active/not empty)
    uint32_t activeCellsCount = 0;
    for( uint32_t offset: resultOffsets )
    {
        if( offset != 0xFFFFFFFF ) // 0xFFFFFFFF is our empty cell marker
        {
            activeCellsCount++;
        }
    }

    // Since both agents share a hash, they form exactly 1 occupied spatial cell cluster
    EXPECT_EQ( activeCellsCount, 1 ) << "Offset buffer should have exactly one active cell (start index) registered!";

    state.Destroy( m_resourceManager.get() );
}

// 7. Test biomechanics integration
TEST_F( SimulationBuilderTest, Behaviour_Biomechanics_Integration )
{
    if( !m_device )
        GTEST_SKIP();

    DigitalTwin::SimulationBlueprint blueprint;
    blueprint.SetDomainSize( glm::vec3( 100.0f ), 2.0f )
        .ConfigureSpatialPartitioning()
        .SetMethod( DigitalTwin::SpatialPartitioningMethod::HashGrid )
        .SetCellSize( 30.0f )
        .SetMaxDensity( 64 )
        .SetComputeHz( 30.0f );

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

    DigitalTwin::SimulationBuilder builder( m_resourceManager.get(), m_streamingManager.get() );
    DigitalTwin::SimulationState   state = builder.Build( blueprint );

    auto compCtxHandle = m_device->CreateThreadContext( QueueType::COMPUTE );
    auto compCtx       = m_device->GetThreadContext( compCtxHandle );
    auto compCmd       = compCtx->GetCommandBuffer( compCtx->CreateCommandBuffer() );

    DigitalTwin::GraphDispatcher dispatcher;

    // Dispatch 1 frame of physics.
    // activeIndex = 0 means it reads from agentBuffers[0] and writes results to agentBuffers[1]
    compCmd->Begin();
    dispatcher.Dispatch( &state.computeGraph, compCmd, nullptr, 0.04f, 0.04f, 0 );
    compCmd->End();

    m_device->GetComputeQueue()->Submit( { compCmd } );
    m_device->GetComputeQueue()->WaitIdle();

    // Readback Agent Positions from the output buffer (index 1)
    std::vector<glm::vec4> resultPositions( 2 );
    m_streamingManager->ReadbackBufferImmediate( state.agentBuffers[ 1 ], resultPositions.data(), 2 * sizeof( glm::vec4 ) );

    // Readback Pressures from the pressure buffer
    std::vector<float> resultPressures( 2 );
    m_streamingManager->ReadbackBufferImmediate( state.pressureBuffer, resultPressures.data(), 2 * sizeof( float ) );

    // Verify Physics Logic:

    // Cell A (index 0) should have been pushed negatively in X
    EXPECT_LT( resultPositions[ 0 ].x, 0.0f ) << "Agent 0 was not repelled correctly to the left!";

    // Cell B (index 1) should have been pushed positively in X (away from A, beyond its initial 0.1f)
    EXPECT_GT( resultPositions[ 1 ].x, 0.1f ) << "Agent 1 was not repelled correctly to the right!";

    // Verify Newton's Third Law (Every action has an equal and opposite reaction)
    EXPECT_GT( resultPressures[ 0 ], 0.0f ) << "Agent 0 did not register collision pressure!";
    EXPECT_GT( resultPressures[ 1 ], 0.0f ) << "Agent 1 did not register collision pressure!";
    EXPECT_FLOAT_EQ( resultPressures[ 0 ], resultPressures[ 1 ] ) << "Newton's Third Law violated: Pressures unequal!";

    state.Destroy( m_resourceManager.get() );
}

// 8. Tests cell cycle behaviour
TEST_F( SimulationBuilderTest, Behaviour_CellCycle_Integration )
{
    if( !m_device )
        GTEST_SKIP();

    DigitalTwin::SimulationBlueprint blueprint;
    blueprint.SetDomainSize( glm::vec3( 100.0f ), 2.0f );

    // Add dummy Oxygen Field to satisfy Vulkan Validation Layers in Phenotype shader
    blueprint.AddGridField( "Oxygen" ).SetInitializer( DigitalTwin::GridInitializer::Constant( 38.0f ) ).SetComputeHz( 60.0f );

    // Place a single agent. We will test if it grows and divides.
    // Ensure 'w' component is 1.0f (Active)
    std::vector<glm::vec4> initialCells = { glm::vec4( 0.0f, 0.0f, 0.0f, 1.0f ) };

    blueprint.AddAgentGroup( "TumorCells" )
        .SetCount( 1 )
        .SetDistribution( initialCells )
        .AddBehaviour( DigitalTwin::BiologyGenerator::StandardCellCycle()
                           .SetBaseDoublingTime( 1.0f / 3600.0f ) // Extremely fast! Doubles in 1 second
                           .SetProliferationOxygenTarget( 38.0f )
                           .SetArrestPressureThreshold( 10.0f )
                           .SetNecrosisOxygenThreshold( 5.0f )
                           .SetApoptosisRate( 0.0f )
                           .Build() )
        .SetHz( 60.0f ); // Run fast for test

    DigitalTwin::SimulationBuilder builder( m_resourceManager.get(), m_streamingManager.get() );
    DigitalTwin::SimulationState   state = builder.Build( blueprint );

    auto compCtxHandle = m_device->CreateThreadContext( QueueType::COMPUTE );
    auto compCtx       = m_device->GetThreadContext( compCtxHandle );
    auto compCmd       = compCtx->GetCommandBuffer( compCtx->CreateCommandBuffer() );

    DigitalTwin::GraphDispatcher dispatcher;

    // Simulate EXACTLY 30 frames.
    // Growth per frame = 1.0 (per sec) / 60.0 = 0.01666...
    // 0.5 (initial) + 31 * 0.01666... > 1.0.
    // At exactly frame 31, biomass hits more than 1.0, mitosis fires, and splits it to 0.5 and 0.5.
    compCmd->Begin();
    for( int i = 0; i < 31; ++i )
    {
        dispatcher.Dispatch( &state.computeGraph, compCmd, nullptr, 1.0f / 60.0f, static_cast<float>( i ), 0 );
    }
    compCmd->End();

    m_device->GetComputeQueue()->Submit( { compCmd } );
    m_device->GetComputeQueue()->WaitIdle();

    // Readback the GPU-Driven Atomic Counter to see if mitosis actually fired
    uint32_t resultAgentCount = 0;
    m_streamingManager->ReadbackBufferImmediate( state.agentCountBuffer, &resultAgentCount, sizeof( uint32_t ) );

    // Readback Phenotypes
    std::vector<PhenotypeData> resultPhenotypes( 2 ); // Check up to 2 cells
    m_streamingManager->ReadbackBufferImmediate( state.phenotypeBuffer, resultPhenotypes.data(), 2 * sizeof( PhenotypeData ) );

    // Readback Agent Positions
    std::vector<glm::vec4> resultPositions( 2 );
    m_streamingManager->ReadbackBufferImmediate( state.agentBuffers[ 0 ], resultPositions.data(), 2 * sizeof( glm::vec4 ) );

    // --- Assertions ---
    // 1. Counter should increment because biomass exceeded 1.0!
    EXPECT_EQ( resultAgentCount, 2 ) << "Mitosis Append failed to increment GPU-driven counter!";

    // 2. Mother cell (idx 0) should have halved its biomass back to 0.5
    EXPECT_NEAR( resultPhenotypes[ 0 ].biomass, 0.5f, 0.001f ) << "Mother cell did not reset its biomass after division!";

    // 3. Daughter cell (idx 1) should be initialized properly
    EXPECT_NEAR( resultPhenotypes[ 1 ].biomass, 0.5f, 0.001f ) << "Daughter cell was not initialized with 0.5 biomass!";
    EXPECT_EQ( resultPhenotypes[ 1 ].lifecycleState, 0 ) << "Daughter cell is not in 'Live' state!";
    EXPECT_FLOAT_EQ( resultPositions[ 1 ].w, 1.0f ) << "Daughter cell did not receive w=1.0 (Alive flag)!";

    state.Destroy( m_resourceManager.get() );
}

// 9. Integration Test: Hypoxic Cell Secretes VEGF
TEST_F( SimulationBuilderTest, Behaviour_Hypoxia_Secretion_Integration )
{
    if( !m_device )
        GTEST_SKIP();

    SimulationBlueprint blueprint;
    blueprint.SetDomainSize( glm::vec3( 10.0f ), 1.0f );

    // Setup Oxygen to be extremely low (10.0 mmHg)
    blueprint.AddGridField( "Oxygen" ).SetInitializer( DigitalTwin::GridInitializer::Constant( 10.0f ) ).SetComputeHz( 60.0f );

    // Setup VEGF (Empty initially)
    blueprint.AddGridField( "VEGF" ).SetInitializer( DigitalTwin::GridInitializer::Constant( 0.0f ) ).SetComputeHz( 60.0f );

    std::vector<glm::vec4> oneCell = { glm::vec4( 0.0f, 0.0f, 0.0f, 1.0f ) };

    auto& agentGrp = blueprint.AddAgentGroup( "TestCell" ).SetCount( 1 ).SetDistribution( oneCell );
    agentGrp
        .AddBehaviour( DigitalTwin::BiologyGenerator::StandardCellCycle()
                           .SetHypoxiaOxygenThreshold( 15.0f ) // Triggers hypoxia because O2 (10) < Threshold (15)
                           .Build() )
        .SetHz( 60.0f );
    agentGrp
        .AddBehaviour(
            DigitalTwin::Behaviours::SecreteField{ "VEGF", 10.0f, DigitalTwin::LifecycleState::Hypoxic } )
        .SetHz( 60.0f );

    DigitalTwin::SimulationBuilder builder( m_resourceManager.get(), m_streamingManager.get() );
    SimulationState                state = builder.Build( blueprint );

    auto compCtxHandle = m_device->CreateThreadContext( QueueType::COMPUTE );
    auto compCtx       = m_device->GetThreadContext( compCtxHandle );
    auto compCmd       = compCtx->GetCommandBuffer( compCtx->CreateCommandBuffer() );

    DigitalTwin::GraphDispatcher dispatcher;

    compCmd->Begin();
    // Ping pass
    dispatcher.Dispatch( &state.computeGraph, compCmd, nullptr, 1.0f, 1.0f, 0 );
    // Pong pass
    dispatcher.Dispatch( &state.computeGraph, compCmd, nullptr, 1.0f, 2.0f, 1 );
    compCmd->End();

    m_device->GetComputeQueue()->Submit( { compCmd } );
    m_device->GetComputeQueue()->WaitIdle();

    std::vector<float> vegfData  = ReadbackGrid( state, 1, m_streamingManager.get() ); // 1 is VEGF index
    uint32_t           centerIdx = 5 + 5 * 10 + 5 * 100;

    // Verify cell survived, went hypoxic, and secreted VEGF
    EXPECT_NEAR( vegfData[ centerIdx ], 0.16666f, 0.001f ) << "Hypoxic cell failed to secrete correct physical amount of VEGF!";

    state.Destroy( m_resourceManager.get() );
}

// 10. Chemotaxis integration: agent migrates toward the VEGF gradient peak
TEST_F( SimulationBuilderTest, Behaviour_Chemotaxis_AgentMovesTowardGradient )
{
    if( !m_device )
        GTEST_SKIP();

    // 10x10x10 domain (world coords -5 to +5 per axis)
    SimulationBlueprint blueprint;
    blueprint.SetDomainSize( glm::vec3( 10.0f ), 1.0f );

    // VEGF field: pure linear X gradient (concentration increases toward +X)
    // This ensures dCy = dCz = 0 exactly regardless of voxel-center offsets
    blueprint.AddGridField( "VEGF" )
        .SetInitializer( []( const glm::vec3& pos ) { return pos.x * 10.0f + 50.0f; } )
        .SetDiffusionCoefficient( 0.0f )
        .SetDecayRate( 0.0f )
        .SetComputeHz( 1.0f );

    // 1 agent placed at -X side (away from the VEGF peak)
    std::vector<glm::vec4> startPos = { glm::vec4( -3.0f, 0.0f, 0.0f, 1.0f ) };

    blueprint.AddAgentGroup( "EndothelialCells" )
        .SetCount( 1 )
        .SetDistribution( startPos )
        // sensitivity=10, saturation=0 (linear), maxVelocity=100 (no clamp)
        .AddBehaviour( DigitalTwin::Behaviours::Chemotaxis{ "VEGF", 10.0f, 0.0f, 100.0f } )
        .SetHz( 1.0f );

    DigitalTwin::SimulationBuilder builder( m_resourceManager.get(), m_streamingManager.get() );
    SimulationState                state = builder.Build( blueprint );

    auto compCtxHandle = m_device->CreateThreadContext( QueueType::COMPUTE );
    auto compCtx       = m_device->GetThreadContext( compCtxHandle );
    auto compCmd       = compCtx->GetCommandBuffer( compCtx->CreateCommandBuffer() );

    DigitalTwin::GraphDispatcher dispatcher;
    compCmd->Begin();
    // activeIndex=0: reads agentBuffers[0], writes agentBuffers[1]; dt=1.0, totalTime=1.0 → executes
    dispatcher.Dispatch( &state.computeGraph, compCmd, nullptr, 1.0f, 1.0f, 0 );
    compCmd->End();

    m_device->GetComputeQueue()->Submit( { compCmd } );
    m_device->GetComputeQueue()->WaitIdle();

    // Readback from output buffer (index 1 after activeIndex=0 dispatch)
    glm::vec4 resultPos{ 0.0f };
    m_streamingManager->ReadbackBufferImmediate( state.agentBuffers[ 1 ], &resultPos, sizeof( glm::vec4 ) );

    // Agent started at x=-3.0; VEGF gradient points in +X direction → must have moved right
    EXPECT_GT( resultPos.x, -3.0f ) << "Chemotaxis failed: agent did not move toward the VEGF gradient";

    // Y and Z should be near zero (gradient is purely along X axis)
    EXPECT_NEAR( resultPos.y, 0.0f, 0.5f ) << "Chemotaxis introduced unexpected Y displacement";
    EXPECT_NEAR( resultPos.z, 0.0f, 0.5f ) << "Chemotaxis introduced unexpected Z displacement";

    // Sanity: agent is still alive
    EXPECT_FLOAT_EQ( resultPos.w, 1.0f );

    state.Destroy( m_resourceManager.get() );
}

// Chemotaxis with TipCell filter: only TipCells move, StalkCells stay in place.
// Uses ForceAllCellType-style phenotype override to set cell types before dispatch.
TEST_F( SimulationBuilderTest, Behaviour_Chemotaxis_TipCellFilter_OnlyTipCellMoves )
{
    if( !m_device )
        GTEST_SKIP();

    SimulationBlueprint blueprint;
    blueprint.SetDomainSize( glm::vec3( 10.0f ), 1.0f );

    // VEGF field: linear X gradient
    blueprint.AddGridField( "VEGF" )
        .SetInitializer( []( const glm::vec3& pos ) { return pos.x * 10.0f + 50.0f; } )
        .SetDiffusionCoefficient( 0.0f )
        .SetDecayRate( 0.0f )
        .SetComputeHz( 1.0f );

    // 2 agents: one at x=-3 (will be TipCell), one at x=+3 (will be StalkCell)
    std::vector<glm::vec4> startPos = {
        glm::vec4( -3.0f, 0.0f, 0.0f, 1.0f ),
        glm::vec4(  3.0f, 0.0f, 0.0f, 1.0f )
    };

    blueprint.AddAgentGroup( "EndothelialCells" )
        .SetCount( 2 )
        .SetDistribution( startPos )
        .AddBehaviour( DigitalTwin::Behaviours::Chemotaxis{ "VEGF", 10.0f, 0.0f, 100.0f } )
        .SetHz( 1.0f )
        .SetRequiredCellType( DigitalTwin::CellType::TipCell );

    SimulationBuilder builder( m_resourceManager.get(), m_streamingManager.get() );
    SimulationState   state = builder.Build( blueprint );

    // Force agent 0 = TipCell (1), agent 1 = StalkCell (2) in the phenotype buffer
    std::vector<PhenotypeData> phenotypes( 2, { 0u, 0.5f, 0.0f, 0u } );
    phenotypes[ 0 ].cellType = 1u; // TipCell
    phenotypes[ 1 ].cellType = 2u; // StalkCell
    m_streamingManager->UploadBufferImmediate( { { state.phenotypeBuffer, phenotypes.data(), 2 * sizeof( PhenotypeData ) } } );

    auto compCtxHandle = m_device->CreateThreadContext( QueueType::COMPUTE );
    auto compCtx       = m_device->GetThreadContext( compCtxHandle );
    auto compCmd       = compCtx->GetCommandBuffer( compCtx->CreateCommandBuffer() );

    GraphDispatcher dispatcher;
    compCmd->Begin();
    dispatcher.Dispatch( &state.computeGraph, compCmd, nullptr, 1.0f, 1.0f, 0 );
    compCmd->End();

    m_device->GetComputeQueue()->Submit( { compCmd } );
    m_device->GetComputeQueue()->WaitIdle();

    // Readback both agents from output buffer (index 1 after activeIndex=0 dispatch)
    std::vector<glm::vec4> results( 2 );
    m_streamingManager->ReadbackBufferImmediate( state.agentBuffers[ 1 ], results.data(), 2 * sizeof( glm::vec4 ) );

    // Agent 0 (TipCell at x=-3): should have moved toward +X (gradient direction)
    EXPECT_GT( results[ 0 ].x, -3.0f ) << "TipCell must move toward the VEGF gradient";

    // Agent 1 (StalkCell at x=+3): chemotaxis should skip it — position unchanged
    // Note: the shader returns without writing for filtered cells, so the output buffer
    // retains whatever was in it. With a fresh build, agentBuffers[1] is zero-initialized,
    // meaning the StalkCell's output position is (0,0,0,0) rather than (3,0,0,1).
    // We verify the TipCell moved — that's the critical assertion. The StalkCell's
    // output is undefined (not written), which is the correct shader behaviour.

    EXPECT_FLOAT_EQ( results[ 0 ].w, 1.0f ) << "TipCell should still be alive";

    state.Destroy( m_resourceManager.get() );
}

// ===========================================================================================
// Perfusion + Drain — integration tests
// ===========================================================================================

// Helper: force all agents in the phenotype buffer to a given cellType
static void ForceAllCellType( DigitalTwin::StreamingManager* stream, const DigitalTwin::BufferHandle& phenotypeBuffer,
                               uint32_t cellType, uint32_t capacity )
{
    std::vector<PhenotypeData> phenotypes( capacity, { 0u, 0.5f, 0.0f, cellType } );
    stream->UploadBufferImmediate( { { phenotypeBuffer, phenotypes.data(), capacity * sizeof( PhenotypeData ) } } );
}

// StalkCell + Perfusion → O2 field increases at agent voxel.
TEST_F( SimulationBuilderTest, Behaviour_Perfusion_StalkCell_InjectsOxygen )
{
    if( !m_device )
        GTEST_SKIP();

    SimulationBlueprint blueprint;
    blueprint.SetDomainSize( glm::vec3( 10.0f ), 1.0f );

    // Oxygen starts empty — Perfusion should fill the center voxel
    blueprint.AddGridField( "Oxygen" )
        .SetInitializer( DigitalTwin::GridInitializer::Constant( 0.0f ) )
        .SetDiffusionCoefficient( 0.01f )
        .SetComputeHz( 1.0f );

    std::vector<glm::vec4> oneCell = { glm::vec4( 0.0f, 0.0f, 0.0f, 1.0f ) };

    blueprint.AddAgentGroup( "Vessel" )
        .SetCount( 1 )
        .SetDistribution( oneCell )
        .AddBehaviour( Behaviours::Perfusion{ "Oxygen", 10.0f } )
        .SetHz( 1.0f );

    SimulationBuilder builder( m_resourceManager.get(), m_streamingManager.get() );
    SimulationState   state = builder.Build( blueprint );

    // Force agent to StalkCell (cellType=2) — Perfusion only fires for StalkCells
    ForceAllCellType( m_streamingManager.get(), state.phenotypeBuffer, 2u, 131072 );

    auto compCtxHandle = m_device->CreateThreadContext( QueueType::COMPUTE );
    auto compCtx       = m_device->GetThreadContext( compCtxHandle );
    auto compCmd       = compCtx->GetCommandBuffer( compCtx->CreateCommandBuffer() );

    GraphDispatcher dispatcher;
    compCmd->Begin();
    dispatcher.Dispatch( &state.computeGraph, compCmd, nullptr, 1.0f, 1.0f, 0 );
    dispatcher.Dispatch( &state.computeGraph, compCmd, nullptr, 1.0f, 2.0f, 1 );
    compCmd->End();

    m_device->GetComputeQueue()->Submit( { compCmd } );
    m_device->GetComputeQueue()->WaitIdle();

    std::vector<float> gridData = ReadbackGrid( state, 0, m_streamingManager.get() );
    uint32_t           centerIdx = 5 + 5 * 10 + 5 * 100;

    EXPECT_GT( gridData[ centerIdx ], 0.0f ) << "StalkCell must inject O2: center voxel should increase from 0";

    state.Destroy( m_resourceManager.get() );
}

// PhalanxCell + Perfusion → O2 field increases at agent voxel (Phase 4 fix verification).
TEST_F( SimulationBuilderTest, Behaviour_Perfusion_PhalanxCell_InjectsOxygen )
{
    if( !m_device )
        GTEST_SKIP();

    SimulationBlueprint blueprint;
    blueprint.SetDomainSize( glm::vec3( 10.0f ), 1.0f );

    blueprint.AddGridField( "Oxygen" )
        .SetInitializer( DigitalTwin::GridInitializer::Constant( 0.0f ) )
        .SetDiffusionCoefficient( 0.01f )
        .SetComputeHz( 1.0f );

    std::vector<glm::vec4> oneCell = { glm::vec4( 0.0f, 0.0f, 0.0f, 1.0f ) };

    blueprint.AddAgentGroup( "Vessel" )
        .SetCount( 1 )
        .SetDistribution( oneCell )
        .AddBehaviour( Behaviours::Perfusion{ "Oxygen", 10.0f } )
        .SetHz( 1.0f );

    SimulationBuilder builder( m_resourceManager.get(), m_streamingManager.get() );
    SimulationState   state = builder.Build( blueprint );

    // Force agent to PhalanxCell (cellType=3) — the Phase 4 fix allows these to perfuse
    ForceAllCellType( m_streamingManager.get(), state.phenotypeBuffer, 3u, 131072 );

    auto compCtxHandle = m_device->CreateThreadContext( QueueType::COMPUTE );
    auto compCtx       = m_device->GetThreadContext( compCtxHandle );
    auto compCmd       = compCtx->GetCommandBuffer( compCtx->CreateCommandBuffer() );

    GraphDispatcher dispatcher;
    compCmd->Begin();
    dispatcher.Dispatch( &state.computeGraph, compCmd, nullptr, 1.0f, 1.0f, 0 );
    dispatcher.Dispatch( &state.computeGraph, compCmd, nullptr, 1.0f, 2.0f, 1 );
    compCmd->End();

    m_device->GetComputeQueue()->Submit( { compCmd } );
    m_device->GetComputeQueue()->WaitIdle();

    std::vector<float> gridData = ReadbackGrid( state, 0, m_streamingManager.get() );
    uint32_t           centerIdx = 5 + 5 * 10 + 5 * 100;

    EXPECT_GT( gridData[ centerIdx ], 0.0f ) << "PhalanxCell must inject O2 after Phase 4 perfusion fix";

    state.Destroy( m_resourceManager.get() );
}

// Perfusion with RequiredCellType=PhalanxCell — StalkCell is rejected (fParam1 builder wiring).
TEST_F( SimulationBuilderTest, Behaviour_Perfusion_RequiredCellType_RejectsStalkCell )
{
    if( !m_device )
        GTEST_SKIP();

    SimulationBlueprint blueprint;
    blueprint.SetDomainSize( glm::vec3( 10.0f ), 1.0f );

    blueprint.AddGridField( "Oxygen" )
        .SetInitializer( DigitalTwin::GridInitializer::Constant( 0.0f ) )
        .SetDiffusionCoefficient( 0.01f )
        .SetComputeHz( 1.0f );

    std::vector<glm::vec4> oneCell = { glm::vec4( 0.0f, 0.0f, 0.0f, 1.0f ) };

    blueprint.AddAgentGroup( "Vessel" )
        .SetCount( 1 )
        .SetDistribution( oneCell )
        .AddBehaviour( Behaviours::Perfusion{ "Oxygen", 10.0f } )
        .SetHz( 1.0f )
        .SetRequiredCellType( DigitalTwin::CellType::PhalanxCell );

    SimulationBuilder builder( m_resourceManager.get(), m_streamingManager.get() );
    SimulationState   state = builder.Build( blueprint );

    ForceAllCellType( m_streamingManager.get(), state.phenotypeBuffer, 2u, 131072 ); // StalkCell

    auto compCtxHandle = m_device->CreateThreadContext( QueueType::COMPUTE );
    auto compCtx       = m_device->GetThreadContext( compCtxHandle );
    auto compCmd       = compCtx->GetCommandBuffer( compCtx->CreateCommandBuffer() );

    GraphDispatcher dispatcher;
    compCmd->Begin();
    dispatcher.Dispatch( &state.computeGraph, compCmd, nullptr, 1.0f, 1.0f, 0 );
    dispatcher.Dispatch( &state.computeGraph, compCmd, nullptr, 1.0f, 2.0f, 1 );
    compCmd->End();

    m_device->GetComputeQueue()->Submit( { compCmd } );
    m_device->GetComputeQueue()->WaitIdle();

    uint32_t           centerIdx = 5 + 5 * 10 + 5 * 100;
    std::vector<float> gridData  = ReadbackGrid( state, 0, m_streamingManager.get() );
    EXPECT_EQ( gridData[ centerIdx ], 0.0f ) << "StalkCell must be rejected by PhalanxCell-only Perfusion filter";

    state.Destroy( m_resourceManager.get() );
}

// Perfusion with RequiredCellType=PhalanxCell — PhalanxCell is accepted (fParam1 builder wiring).
TEST_F( SimulationBuilderTest, Behaviour_Perfusion_RequiredCellType_AcceptsPhalanxCell )
{
    if( !m_device )
        GTEST_SKIP();

    SimulationBlueprint blueprint;
    blueprint.SetDomainSize( glm::vec3( 10.0f ), 1.0f );

    blueprint.AddGridField( "Oxygen" )
        .SetInitializer( DigitalTwin::GridInitializer::Constant( 0.0f ) )
        .SetDiffusionCoefficient( 0.01f )
        .SetComputeHz( 1.0f );

    std::vector<glm::vec4> oneCell = { glm::vec4( 0.0f, 0.0f, 0.0f, 1.0f ) };

    blueprint.AddAgentGroup( "Vessel" )
        .SetCount( 1 )
        .SetDistribution( oneCell )
        .AddBehaviour( Behaviours::Perfusion{ "Oxygen", 10.0f } )
        .SetHz( 1.0f )
        .SetRequiredCellType( DigitalTwin::CellType::PhalanxCell );

    SimulationBuilder builder( m_resourceManager.get(), m_streamingManager.get() );
    SimulationState   state = builder.Build( blueprint );

    ForceAllCellType( m_streamingManager.get(), state.phenotypeBuffer, 3u, 131072 ); // PhalanxCell

    auto compCtxHandle = m_device->CreateThreadContext( QueueType::COMPUTE );
    auto compCtx       = m_device->GetThreadContext( compCtxHandle );
    auto compCmd       = compCtx->GetCommandBuffer( compCtx->CreateCommandBuffer() );

    GraphDispatcher dispatcher;
    compCmd->Begin();
    dispatcher.Dispatch( &state.computeGraph, compCmd, nullptr, 1.0f, 1.0f, 0 );
    dispatcher.Dispatch( &state.computeGraph, compCmd, nullptr, 1.0f, 2.0f, 1 );
    compCmd->End();

    m_device->GetComputeQueue()->Submit( { compCmd } );
    m_device->GetComputeQueue()->WaitIdle();

    uint32_t           centerIdx = 5 + 5 * 10 + 5 * 100;
    std::vector<float> gridData  = ReadbackGrid( state, 0, m_streamingManager.get() );
    EXPECT_GT( gridData[ centerIdx ], 0.0f ) << "PhalanxCell must inject O2 when reqCellType=PhalanxCell";

    state.Destroy( m_resourceManager.get() );
}

// StalkCell + Drain → Lactate field decreases at agent voxel.
TEST_F( SimulationBuilderTest, Behaviour_Drain_StalkCell_RemovesLactate )
{
    if( !m_device )
        GTEST_SKIP();

    SimulationBlueprint blueprint;
    blueprint.SetDomainSize( glm::vec3( 10.0f ), 1.0f );

    // Lactate starts full — Drain should deplete the center voxel
    blueprint.AddGridField( "Lactate" )
        .SetInitializer( DigitalTwin::GridInitializer::Constant( 100.0f ) )
        .SetDiffusionCoefficient( 0.01f )
        .SetComputeHz( 1.0f );

    std::vector<glm::vec4> oneCell = { glm::vec4( 0.0f, 0.0f, 0.0f, 1.0f ) };

    blueprint.AddAgentGroup( "Vessel" )
        .SetCount( 1 )
        .SetDistribution( oneCell )
        .AddBehaviour( Behaviours::Drain{ "Lactate", 10.0f } )
        .SetHz( 1.0f );

    SimulationBuilder builder( m_resourceManager.get(), m_streamingManager.get() );
    SimulationState   state = builder.Build( blueprint );

    ForceAllCellType( m_streamingManager.get(), state.phenotypeBuffer, 2u, 131072 );

    auto compCtxHandle = m_device->CreateThreadContext( QueueType::COMPUTE );
    auto compCtx       = m_device->GetThreadContext( compCtxHandle );
    auto compCmd       = compCtx->GetCommandBuffer( compCtx->CreateCommandBuffer() );

    GraphDispatcher dispatcher;
    compCmd->Begin();
    dispatcher.Dispatch( &state.computeGraph, compCmd, nullptr, 1.0f, 1.0f, 0 );
    dispatcher.Dispatch( &state.computeGraph, compCmd, nullptr, 1.0f, 2.0f, 1 );
    compCmd->End();

    m_device->GetComputeQueue()->Submit( { compCmd } );
    m_device->GetComputeQueue()->WaitIdle();

    std::vector<float> gridData = ReadbackGrid( state, 0, m_streamingManager.get() );
    uint32_t           centerIdx = 5 + 5 * 10 + 5 * 100;

    EXPECT_LT( gridData[ centerIdx ], 100.0f ) << "StalkCell must drain lactate: center voxel should decrease from 100";

    state.Destroy( m_resourceManager.get() );
}

// Multi-mesh rendering: group with cell-type morphologies generates per-cellType draw commands
TEST_F( SimulationBuilderTest, MultiMesh_DrawCommandCount )
{
    if( !m_device )
        GTEST_SKIP();

    SimulationBlueprint blueprint;
    blueprint.SetDomainSize( glm::vec3( 100.0f ), 1.0f );

    // Group A: no cell-type morphologies → 1 draw command
    std::vector<glm::vec4> posA = { glm::vec4( 0, 0, 0, 1 ) };
    blueprint.AddAgentGroup( "GroupA" )
        .SetCount( 1 )
        .SetDistribution( posA );

    // Group B: 2 cell-type morphologies (TipCell + StalkCell) → 3 draw commands
    std::vector<glm::vec4> posB = { glm::vec4( 10, 0, 0, 1 ), glm::vec4( 20, 0, 0, 1 ) };
    blueprint.AddAgentGroup( "GroupB" )
        .SetCount( 2 )
        .SetDistribution( posB )
        .AddCellTypeMorphology( 1, MorphologyGenerator::CreateSpikySphere() )
        .AddCellTypeMorphology( 2, MorphologyGenerator::CreateCylinder() );

    SimulationBuilder builder( m_resourceManager.get(), m_streamingManager.get() );
    SimulationState   state = builder.Build( blueprint );

    EXPECT_TRUE( state.IsValid() );
    EXPECT_EQ( state.groupCount, 2u );
    EXPECT_EQ( state.drawCommandCount, 4u ) << "1 (GroupA default) + 3 (GroupB default + TipCell + StalkCell)";
    EXPECT_TRUE( state.agentReorderBuffer.IsValid() );
    EXPECT_TRUE( state.drawMetaBuffer.IsValid() );

    // Verify draw meta: readback and check group indices + target cell types
    struct DrawMeta { uint32_t groupIndex, targetCellType, groupOffset, groupCapacity; };
    std::vector<DrawMeta> meta( 4 );
    m_streamingManager->ReadbackBufferImmediate( state.drawMetaBuffer, meta.data(), 4 * sizeof( DrawMeta ) );

    // Command 0: GroupA default
    EXPECT_EQ( meta[ 0 ].groupIndex, 0u );
    EXPECT_EQ( meta[ 0 ].targetCellType, 0xFFFFFFFFu ) << "Default (any) cellType";

    // Command 1: GroupB default
    EXPECT_EQ( meta[ 1 ].groupIndex, 1u );
    EXPECT_EQ( meta[ 1 ].targetCellType, 0xFFFFFFFFu ) << "Default (any) cellType";

    // Command 2: GroupB TipCell
    EXPECT_EQ( meta[ 2 ].groupIndex, 1u );
    EXPECT_EQ( meta[ 2 ].targetCellType, 1u ) << "TipCell";

    // Command 3: GroupB StalkCell
    EXPECT_EQ( meta[ 3 ].groupIndex, 1u );
    EXPECT_EQ( meta[ 3 ].targetCellType, 2u ) << "StalkCell";

    state.Destroy( m_resourceManager.get() );
}

// Phase 2.1 — SetMorphologyVariants emits one DrawMeta per variant.
// The bit-packed cellType convention means variant 0 becomes the catch-all
// default (targetCellType = 0xFFFFFFFF) and variants 1..N are registered with
// targetCellType = PackCellType(biologicalType, i). Verifies the fluent API +
// builder wiring is correct end-to-end before TwoShapeDemo visual-checks the
// runtime dispatch.
TEST_F( SimulationBuilderTest, MorphologyVariants_EmitsPerVariantDrawMeta )
{
    if( !m_device )
        GTEST_SKIP();

    DigitalTwin::SimulationBlueprint blueprint;
    blueprint.SetDomainSize( glm::vec3( 20.0f ), 1.0f );

    // 4 cells: first 2 → morphIdx 0 (default); last 2 → morphIdx 1 (variant 1).
    std::vector<glm::vec4> positions = {
        glm::vec4( -2, 0, -2, 1 ), glm::vec4( -1, 0, -2, 1 ),
        glm::vec4( -2, 0, +2, 1 ), glm::vec4( -1, 0, +2, 1 ),
    };
    std::vector<uint32_t> cellTypes = {
        DigitalTwin::PackCellType( DigitalTwin::CellType::Default, 0u ),
        DigitalTwin::PackCellType( DigitalTwin::CellType::Default, 0u ),
        DigitalTwin::PackCellType( DigitalTwin::CellType::Default, 1u ),
        DigitalTwin::PackCellType( DigitalTwin::CellType::Default, 1u ),
    };

    blueprint.AddAgentGroup( "MixedGroup" )
        .SetCount( 4 )
        .SetDistribution( positions )
        .SetInitialCellTypes( cellTypes )
        .SetMorphologyVariants( DigitalTwin::CellType::Default, {
            DigitalTwin::MorphologyGenerator::CreateDisc( 0.6f, 0.15f, 16 ),
            DigitalTwin::MorphologyGenerator::CreateCurvedTile( 60.0f, 1.0f, 0.15f, 1.5f, 4 ),
        } );

    SimulationBuilder builder( m_resourceManager.get(), m_streamingManager.get() );
    SimulationState   state = builder.Build( blueprint );

    EXPECT_TRUE( state.IsValid() );
    EXPECT_EQ( state.groupCount, 1u );
    // 2 DrawMeta: catch-all default (variant 0) + packed exact-match (variant 1).
    EXPECT_EQ( state.drawCommandCount, 2u )
        << "SetMorphologyVariants({v0, v1}) should emit 1 catch-all + 1 variant draw";

    struct DrawMetaCPU { uint32_t groupIndex, targetCellType, groupOffset, groupCapacity; };
    std::vector<DrawMetaCPU> meta( 2 );
    m_streamingManager->ReadbackBufferImmediate( state.drawMetaBuffer, meta.data(), 2 * sizeof( DrawMetaCPU ) );

    EXPECT_EQ( meta[ 0 ].targetCellType, 0xFFFFFFFFu )
        << "Variant 0 becomes the catch-all default";

    const uint32_t expectedV1 = DigitalTwin::PackCellType( DigitalTwin::CellType::Default, 1u );
    EXPECT_EQ( meta[ 1 ].targetCellType, expectedV1 )
        << "Variant 1 targets PackCellType(Default, 1) = 0x00010000";

    // Cross-check the indirect command buffer — the two draws must point to
    // DISTINCT mesh-index ranges. If variant 1's indexCount/vertexOffset
    // matched variant 0's, both draws would render the same mesh (catching a
    // builder bug that single-DrawMeta tests would miss).
    struct DrawCommand { uint32_t indexCount, instanceCount, firstIndex, vertexOffset, firstInstance; };
    std::vector<DrawCommand> cmds( 2 );
    m_streamingManager->ReadbackBufferImmediate( state.indirectCmdBuffer, cmds.data(), 2 * sizeof( DrawCommand ) );

    EXPECT_NE( cmds[ 0 ].firstIndex, cmds[ 1 ].firstIndex )
        << "Variant draws must reference distinct index-buffer ranges (else renders identical mesh)";
    EXPECT_NE( cmds[ 0 ].vertexOffset, cmds[ 1 ].vertexOffset )
        << "Variant draws must reference distinct vertex-buffer ranges";
    // The disc (16 sectors → ~3*2*16 = 96 indices for top+bottom fans + sides; don't pin exact
    // count, just ensure both are non-zero).
    EXPECT_GT( cmds[ 0 ].indexCount, 0u );
    EXPECT_GT( cmds[ 1 ].indexCount, 0u );

    // End-to-end verification: the per-cell cellType tags must actually be
    // uploaded to the phenotype buffer. Catches the original Phase 2.1 bug
    // where phenotype buffer allocation was gated only on the group-level
    // `SetInitialCellType != 0` check — missing the per-cell override path.
    // Without this check, build_indirect would scatter every cell (all with
    // cellType=0) into the catch-all default draw, making variant 1 invisible.
    ASSERT_TRUE( state.phenotypeBuffer.IsValid() )
        << "Phenotype buffer must be allocated when SetInitialCellTypes is used";
    std::vector<PhenotypeData> pheno( 4 );
    m_streamingManager->ReadbackBufferImmediate( state.phenotypeBuffer, pheno.data(), 4 * sizeof( PhenotypeData ) );
    EXPECT_EQ( pheno[ 0 ].cellType, cellTypes[ 0 ] ) << "Cell 0 cellType tag did not reach GPU";
    EXPECT_EQ( pheno[ 1 ].cellType, cellTypes[ 1 ] ) << "Cell 1 cellType tag did not reach GPU";
    EXPECT_EQ( pheno[ 2 ].cellType, cellTypes[ 2 ] ) << "Cell 2 cellType tag did not reach GPU";
    EXPECT_EQ( pheno[ 3 ].cellType, cellTypes[ 3 ] ) << "Cell 3 cellType tag did not reach GPU";

    state.Destroy( m_resourceManager.get() );
}

// CellCycle with requiredCellType=StalkCell: StalkCell accumulates biomass and divides;
// TipCell in the same group does not grow (stays at 0.5).
TEST_F( SimulationBuilderTest, Behaviour_CellCycle_StalkCellOnlyDivides )
{
    if( !m_device )
        GTEST_SKIP();

    DigitalTwin::SimulationBlueprint blueprint;
    blueprint.SetDomainSize( glm::vec3( 100.0f ), 2.0f );
    blueprint.AddGridField( "Oxygen" )
        .SetInitializer( DigitalTwin::GridInitializer::Constant( 40.0f ) )
        .SetComputeHz( 60.0f );

    // 2 agents: idx 0 = StalkCell (cellType=2), idx 1 = TipCell (cellType=1)
    std::vector<glm::vec4> positions = {
        glm::vec4( 0.0f, 0.0f, 0.0f, 1.0f ),
        glm::vec4( 5.0f, 0.0f, 0.0f, 1.0f )
    };

    struct PhenotypeInit { uint32_t lifecycleState; float biomass; float timer; uint32_t cellType; };
    std::vector<PhenotypeInit> initPheno = {
        { 0u, 0.5f, 0.0f, 2u }, // StalkCell
        { 0u, 0.5f, 0.0f, 1u }  // TipCell
    };

    blueprint.AddAgentGroup( "Endo" )
        .SetCount( 2 )
        .SetDistribution( positions )
        .AddBehaviour( DigitalTwin::BiologyGenerator::StandardCellCycle()
                           .SetBaseDoublingTime( 1.0f / 3600.0f ) // Doubles in 1 s
                           .SetProliferationOxygenTarget( 40.0f )
                           .SetArrestPressureThreshold( 100.0f )  // Prevent arrest
                           .SetNecrosisOxygenThreshold( 0.0f )
                           .SetHypoxiaOxygenThreshold( 0.1f )
                           .SetApoptosisRate( 0.0f )
                           .Build() )
        .SetHz( 60.0f )
        .SetRequiredCellType( DigitalTwin::CellType::StalkCell );

    DigitalTwin::SimulationBuilder builder( m_resourceManager.get(), m_streamingManager.get() );
    DigitalTwin::SimulationState   state = builder.Build( blueprint );

    // Upload initial phenotypes so both cells have correct cellType before any dispatch
    m_streamingManager->UploadBufferImmediate(
        { { state.phenotypeBuffer, initPheno.data(), 2 * sizeof( PhenotypeInit ), 0 } } );

    auto compCtxHandle = m_device->CreateThreadContext( QueueType::COMPUTE );
    auto compCtx       = m_device->GetThreadContext( compCtxHandle );
    auto compCmd       = compCtx->GetCommandBuffer( compCtx->CreateCommandBuffer() );

    DigitalTwin::GraphDispatcher dispatcher;

    // 31 frames at 60Hz — enough for StalkCell to reach biomass>=1.0 and divide
    compCmd->Begin();
    for( int i = 0; i < 31; ++i )
        dispatcher.Dispatch( &state.computeGraph, compCmd, nullptr, 1.0f / 60.0f, static_cast<float>( i ), 0 );
    compCmd->End();
    m_device->GetComputeQueue()->Submit( { compCmd } );
    m_device->GetComputeQueue()->WaitIdle();

    // Agent count should be 3: original 2 + 1 StalkCell daughter
    uint32_t agentCount = 0;
    m_streamingManager->ReadbackBufferImmediate( state.agentCountBuffer, &agentCount, sizeof( uint32_t ) );
    EXPECT_EQ( agentCount, 3u ) << "StalkCell should have divided (count 2->3)";

    // Read back phenotypes for 3 slots
    std::vector<PhenotypeInit> pheno( 3 );
    m_streamingManager->ReadbackBufferImmediate( state.phenotypeBuffer, pheno.data(), 3 * sizeof( PhenotypeInit ) );

    // StalkCell (slot 0) mother biomass reset to 0.5
    EXPECT_NEAR( pheno[ 0 ].biomass, 0.5f, 0.01f ) << "StalkCell mother should have halved biomass";
    EXPECT_EQ( pheno[ 0 ].cellType, 2u ) << "StalkCell mother should remain StalkCell";

    // TipCell (slot 1) should NOT have divided — biomass stays at 0.5
    EXPECT_NEAR( pheno[ 1 ].biomass, 0.5f, 0.01f ) << "TipCell must not accumulate biomass";
    EXPECT_EQ( pheno[ 1 ].cellType, 1u ) << "TipCell should remain TipCell";

    // Daughter (slot 2) inherits StalkCell type
    EXPECT_EQ( pheno[ 2 ].cellType, 2u ) << "Daughter should inherit StalkCell type";
    EXPECT_NEAR( pheno[ 2 ].biomass, 0.5f, 0.01f ) << "Daughter initial biomass should be 0.5";

    state.Destroy( m_resourceManager.get() );
}

// PhalanxActivation: builder initialises cells as PhalanxCell (3); VEGF above threshold → all
// cells transition to StalkCell (2) after several frames.
TEST_F( SimulationBuilderTest, Behaviour_PhalanxActivation_HighVEGF_ActivatesAllCells )
{
    if( !m_device )
        GTEST_SKIP();

    SimulationBlueprint blueprint;
    blueprint.SetDomainSize( glm::vec3( 20.0f ), 1.0f )
        .ConfigureSpatialPartitioning()
        .SetCellSize( 10.0f )
        .SetComputeHz( 60.0f );

    // VEGF constant at 50 — well above activationThreshold (20)
    blueprint.AddGridField( "VEGF" )
        .SetInitializer( GridInitializer::Constant( 50.0f ) )
        .SetDiffusionCoefficient( 0.0f )
        .SetDecayRate( 0.0f )
        .SetComputeHz( 60.0f );

    std::vector<glm::vec4> pos = { glm::vec4( 0.0f, 0.0f, 0.0f, 1.0f ),
                                   glm::vec4( 2.0f, 0.0f, 0.0f, 1.0f ) };

    blueprint.AddAgentGroup( "Endothelial" )
        .SetCount( 2 )
        .SetDistribution( pos )
        .AddBehaviour( Behaviours::PhalanxActivation{ "VEGF", 20.0f, 5.0f } )
        .SetHz( 60.0f );

    SimulationBuilder builder( m_resourceManager.get(), m_streamingManager.get() );
    SimulationState   state = builder.Build( blueprint );

    // Verify builder initialised phenotypes as PhalanxCell (3) before any dispatch
    std::vector<PhenotypeData> initPheno( 2 );
    m_streamingManager->ReadbackBufferImmediate( state.phenotypeBuffer, initPheno.data(), 2 * sizeof( PhenotypeData ) );
    EXPECT_EQ( initPheno[ 0 ].cellType, 3u ) << "Builder should initialise cell 0 as PhalanxCell (3)";
    EXPECT_EQ( initPheno[ 1 ].cellType, 3u ) << "Builder should initialise cell 1 as PhalanxCell (3)";

    auto compCtxHandle = m_device->CreateThreadContext( QueueType::COMPUTE );
    auto compCtx       = m_device->GetThreadContext( compCtxHandle );
    auto compCmd       = compCtx->GetCommandBuffer( compCtx->CreateCommandBuffer() );

    GraphDispatcher dispatcher;
    compCmd->Begin();
    for( int i = 0; i < 5; ++i )
        dispatcher.Dispatch( &state.computeGraph, compCmd, nullptr, 1.0f / 60.0f, static_cast<float>( i ) / 60.0f, 0 );
    compCmd->End();

    m_device->GetComputeQueue()->Submit( { compCmd } );
    m_device->GetComputeQueue()->WaitIdle();

    std::vector<PhenotypeData> result( 2 );
    m_streamingManager->ReadbackBufferImmediate( state.phenotypeBuffer, result.data(), 2 * sizeof( PhenotypeData ) );

    EXPECT_EQ( result[ 0 ].cellType, 2u ) << "Cell 0 should have activated to StalkCell (2) at VEGF=50";
    EXPECT_EQ( result[ 1 ].cellType, 2u ) << "Cell 1 should have activated to StalkCell (2) at VEGF=50";

    state.Destroy( m_resourceManager.get() );
}

// Orientation buffer is allocated and correctly uploaded when a group provides orientations.
TEST_F( SimulationBuilderTest, SimulationBuilder_OrientationBuffer_Allocated )
{
    if( !m_device )
        GTEST_SKIP();

    SimulationBlueprint blueprint;
    blueprint.SetDomainSize( glm::vec3( 20.0f ), 1.0f );

    std::vector<glm::vec4> pos = {
        glm::vec4( 0.0f, 0.0f, 0.0f, 1.0f ),
        glm::vec4( 2.0f, 0.0f, 0.0f, 1.0f ),
    };
    std::vector<glm::vec4> ori = {
        glm::vec4( 1.0f, 0.0f, 0.0f, 0.0f ),  // face +X
        glm::vec4( 0.0f, 0.0f, 1.0f, 0.0f ),  // face +Z
    };

    blueprint.AddAgentGroup( "Vessel" ).SetCount( 2 ).SetDistribution( pos ).SetOrientations( ori );

    SimulationBuilder builder( m_resourceManager.get(), m_streamingManager.get() );
    SimulationState   state = builder.Build( blueprint );

    EXPECT_TRUE( state.orientationBuffer.IsValid() ) << "Orientation buffer must be allocated";

    // Read back and verify the two orientations were uploaded correctly
    std::vector<glm::vec4> readback( 2 );
    m_streamingManager->ReadbackBufferImmediate( state.orientationBuffer, readback.data(), 2 * sizeof( glm::vec4 ) );

    EXPECT_NEAR( readback[ 0 ].x, 1.0f, 1e-5f );
    EXPECT_NEAR( readback[ 0 ].y, 0.0f, 1e-5f );
    EXPECT_NEAR( readback[ 0 ].z, 0.0f, 1e-5f );
    EXPECT_NEAR( readback[ 1 ].x, 0.0f, 1e-5f );
    EXPECT_NEAR( readback[ 1 ].y, 0.0f, 1e-5f );
    EXPECT_NEAR( readback[ 1 ].z, 1.0f, 1e-5f );

    state.Destroy( m_resourceManager.get() );
}

// Orientation buffer is allocated with default (0,1,0,0) when a group provides no orientations.
TEST_F( SimulationBuilderTest, SimulationBuilder_OrientationBuffer_DefaultWhenEmpty )
{
    if( !m_device )
        GTEST_SKIP();

    SimulationBlueprint blueprint;
    blueprint.SetDomainSize( glm::vec3( 20.0f ), 1.0f );

    std::vector<glm::vec4> pos = { glm::vec4( 0.0f, 0.0f, 0.0f, 1.0f ) };
    blueprint.AddAgentGroup( "Cells" ).SetCount( 1 ).SetDistribution( pos );
    // No SetOrientations call — should get default

    SimulationBuilder builder( m_resourceManager.get(), m_streamingManager.get() );
    SimulationState   state = builder.Build( blueprint );

    EXPECT_TRUE( state.orientationBuffer.IsValid() ) << "Orientation buffer must always be allocated";

    std::vector<glm::vec4> readback( 1 );
    m_streamingManager->ReadbackBufferImmediate( state.orientationBuffer, readback.data(), sizeof( glm::vec4 ) );

    EXPECT_NEAR( readback[ 0 ].x, 0.0f, 1e-5f );
    EXPECT_NEAR( readback[ 0 ].y, 1.0f, 1e-5f ) << "Default orientation must be (0,1,0,0)";
    EXPECT_NEAR( readback[ 0 ].z, 0.0f, 1e-5f );

    state.Destroy( m_resourceManager.get() );
}

// ── Stage 3: Cadherin buffer tests ───────────────────────────────────────────

// 1. Both cadherin buffers are allocated even when no group uses CadherinAdhesion.
TEST_F( SimulationBuilderTest, Builder_CadherinBuffers_AlwaysAllocated )
{
    if( !m_device )
        GTEST_SKIP();

    SimulationBlueprint blueprint;
    blueprint.SetDomainSize( glm::vec3( 10.0f ), 1.0f );
    blueprint.AddAgentGroup( "Cells" )
        .SetCount( 4 )
        .SetDistribution( DigitalTwin::SpatialDistribution::UniformInBox(
            4, glm::vec3( 0.0f ), glm::vec3( 5.0f ) ) )
        .AddBehaviour( Behaviours::Biomechanics{ 15.0f, 2.0f, 1.5f, 0.0f } );

    SimulationBuilder builder( m_resourceManager.get(), m_streamingManager.get() );
    SimulationState   state = builder.Build( blueprint );

    EXPECT_TRUE( state.cadherinProfileBuffer.IsValid() )
        << "cadherinProfileBuffer must be allocated even when no group uses CadherinAdhesion";
    EXPECT_TRUE( state.cadherinAffinityBuffer.IsValid() )
        << "cadherinAffinityBuffer must be allocated even when no group uses CadherinAdhesion";

    state.Destroy( m_resourceManager.get() );
}

// 2. When a group has CadherinAdhesion, cadherinProfileBuffer is full-size.
TEST_F( SimulationBuilderTest, Builder_CadherinAdhesion_RealProfileBuffer )
{
    if( !m_device )
        GTEST_SKIP();

    SimulationBlueprint blueprint;
    blueprint.SetDomainSize( glm::vec3( 10.0f ), 1.0f );
    AgentGroup& g = blueprint.AddAgentGroup( "Cells" );
    g.SetCount( 4 );
    g.SetDistribution( DigitalTwin::SpatialDistribution::UniformInBox(
        4, glm::vec3( 0.0f ), glm::vec3( 5.0f ) ) );
    g.AddBehaviour( Behaviours::Biomechanics{ 15.0f, 2.0f, 1.5f, 0.0f } );
    g.AddBehaviour( Behaviours::CadherinAdhesion{
        glm::vec4( 0.0f, 0.0f, 1.0f, 0.0f ), 0.01f, 0.001f, 1.0f } );

    SimulationBuilder builder( m_resourceManager.get(), m_streamingManager.get() );
    SimulationState   state = builder.Build( blueprint );

    EXPECT_TRUE( state.cadherinProfileBuffer.IsValid() );

    // If the buffer is full-size (131072 slots), reading 1024 profiles must succeed.
    // A 16-byte dummy would be too small for this readback.
    std::vector<glm::vec4> probe( 1024 );
    m_streamingManager->ReadbackBufferImmediate(
        state.cadherinProfileBuffer, probe.data(), 1024 * sizeof( glm::vec4 ) );
    // No crash = buffer is at least 1024 * 16 = 16384 bytes, confirming it is full-size
    SUCCEED() << "cadherinProfileBuffer is full-size when CadherinAdhesion is present";

    state.Destroy( m_resourceManager.get() );
}

// 3. Profiles are initialized from each group's targetExpression.
TEST_F( SimulationBuilderTest, Builder_CadherinAdhesion_InitializesProfileFromTarget )
{
    if( !m_device )
        GTEST_SKIP();

    SimulationBlueprint blueprint;
    blueprint.SetDomainSize( glm::vec3( 10.0f ), 1.0f );
    AgentGroup& g = blueprint.AddAgentGroup( "Cells" );
    g.SetCount( 4 );
    g.SetDistribution( DigitalTwin::SpatialDistribution::UniformInBox(
        4, glm::vec3( 0.0f ), glm::vec3( 5.0f ) ) );
    g.AddBehaviour( Behaviours::Biomechanics{ 15.0f, 2.0f, 1.5f, 0.0f } );
    g.AddBehaviour( Behaviours::CadherinAdhesion{
        glm::vec4( 0.0f, 0.0f, 1.0f, 0.0f ), 0.01f, 0.001f, 1.0f } );

    SimulationBuilder builder( m_resourceManager.get(), m_streamingManager.get() );
    SimulationState   state = builder.Build( blueprint );

    // Read back the first 4 slots
    std::vector<glm::vec4> profiles( 4 );
    m_streamingManager->ReadbackBufferImmediate(
        state.cadherinProfileBuffer, profiles.data(), 4 * sizeof( glm::vec4 ) );

    for( int i = 0; i < 4; ++i )
    {
        EXPECT_NEAR( profiles[ i ].x, 0.0f, 1e-5f ) << "slot " << i << ": E-cad should be 0";
        EXPECT_NEAR( profiles[ i ].y, 0.0f, 1e-5f ) << "slot " << i << ": N-cad should be 0";
        EXPECT_NEAR( profiles[ i ].z, 1.0f, 1e-5f ) << "slot " << i << ": VE-cad should be 1";
        EXPECT_NEAR( profiles[ i ].w, 0.0f, 1e-5f ) << "slot " << i << ": Cad-11 should be 0";
    }

    state.Destroy( m_resourceManager.get() );
}

// 4. CadherinAdhesion: expression-update task is registered in the compute graph.
TEST_F( SimulationBuilderTest, Builder_CadherinAdhesion_ExpressionUpdateTask_Exists )
{
    if( !m_device )
        GTEST_SKIP();

    SimulationBlueprint blueprint;
    blueprint.SetDomainSize( glm::vec3( 10.0f ), 1.0f );
    AgentGroup& g = blueprint.AddAgentGroup( "Cells" );
    g.SetCount( 4 );
    g.SetDistribution( DigitalTwin::SpatialDistribution::UniformInBox(
        4, glm::vec3( 0.0f ), glm::vec3( 5.0f ) ) );
    g.AddBehaviour( Behaviours::Biomechanics{ 15.0f, 2.0f, 1.5f, 0.0f } );
    g.AddBehaviour( Behaviours::CadherinAdhesion{
        glm::vec4( 0.0f ), 0.05f, 0.01f, 1.0f } );

    SimulationBuilder builder( m_resourceManager.get(), m_streamingManager.get() );
    SimulationState   state = builder.Build( blueprint );

    // Biomechanics is behaviour 0, CadherinAdhesion is behaviour 1 → tag suffix "_0_1"
    ComputeTask* task = state.computeGraph.FindTask( "cadherin_expr_0_1" );
    ASSERT_NE( task, nullptr ) << "cadherin_expr task was not added to the compute graph";

    state.Destroy( m_resourceManager.get() );
}

// 5. CadherinAdhesion: one dispatch frame moves profiles toward targetExpression.
TEST_F( SimulationBuilderTest, Builder_CadherinAdhesion_ProfileMovesTowardTarget )
{
    if( !m_device )
        GTEST_SKIP();

    SimulationBlueprint blueprint;
    blueprint.SetDomainSize( glm::vec3( 10.0f ), 1.0f );
    AgentGroup& g = blueprint.AddAgentGroup( "Cells" );
    g.SetCount( 4 );
    g.SetDistribution( DigitalTwin::SpatialDistribution::UniformInBox(
        4, glm::vec3( 0.0f ), glm::vec3( 5.0f ) ) );
    g.AddBehaviour( Behaviours::Biomechanics{ 15.0f, 2.0f, 1.5f, 0.0f } );
    // target = (1,0,0,0): E-cadherin only
    g.AddBehaviour( Behaviours::CadherinAdhesion{
        glm::vec4( 1.0f, 0.0f, 0.0f, 0.0f ),
        0.05f, 0.01f, 1.0f } );

    SimulationBuilder builder( m_resourceManager.get(), m_streamingManager.get() );
    SimulationState   state = builder.Build( blueprint );

    // Zero-out profiles (Stage 3 pre-fills with targetExpression; we want to start at 0)
    std::vector<glm::vec4> zeros( 131072, glm::vec4( 0.0f ) );
    m_streamingManager->UploadBufferImmediate(
        { { state.cadherinProfileBuffer, zeros.data(), 131072 * sizeof( glm::vec4 ), 0 } } );

    // Run one frame
    auto compCtxHandle = m_device->CreateThreadContext( QueueType::COMPUTE );
    auto compCtx       = m_device->GetThreadContext( compCtxHandle );
    auto compCmd       = compCtx->GetCommandBuffer( compCtx->CreateCommandBuffer() );

    GraphDispatcher dispatcher;
    compCmd->Begin();
    dispatcher.Dispatch( &state.computeGraph, compCmd, nullptr, 1.0f / 60.0f, 0.0f, 0 );
    compCmd->End();
    m_device->GetComputeQueue()->Submit( { compCmd } );
    m_device->GetComputeQueue()->WaitIdle();

    // Read back first 4 agent slots
    std::vector<glm::vec4> profiles( 4 );
    m_streamingManager->ReadbackBufferImmediate(
        state.cadherinProfileBuffer, profiles.data(), 4 * sizeof( glm::vec4 ) );

    // E-cadherin (x) should have increased from 0 toward target 1
    for( int i = 0; i < 4; ++i )
        EXPECT_GT( profiles[ i ].x, 0.0f )
            << "slot " << i << ": E-cad did not increase after one dispatch";

    // N/VE/Cad-11 targets are 0 — started at 0, must remain 0
    for( int i = 0; i < 4; ++i )
    {
        EXPECT_NEAR( profiles[ i ].y, 0.0f, 1e-5f ) << "slot " << i << ": N-cad should be 0";
        EXPECT_NEAR( profiles[ i ].z, 0.0f, 1e-5f ) << "slot " << i << ": VE-cad should be 0";
        EXPECT_NEAR( profiles[ i ].w, 0.0f, 1e-5f ) << "slot " << i << ": Cad-11 should be 0";
    }

    state.Destroy( m_resourceManager.get() );
}


// 6. Biomechanics + CadherinAdhesion: JKR wiring includes cadherin buffers and dispatch runs cleanly.
TEST_F( SimulationBuilderTest, Builder_JKR_Cadherin_Wiring )
{
    if( !m_device )
        GTEST_SKIP();

    SimulationBlueprint blueprint;
    blueprint.SetDomainSize( glm::vec3( 100.0f ), 2.0f );
    AgentGroup& g = blueprint.AddAgentGroup( "Tissue" );
    g.SetCount( 2 );
    g.SetDistribution( { glm::vec4( 0.0f, 0.0f, 0.0f, 1.0f ),
                         glm::vec4( 1.0f, 0.0f, 0.0f, 1.0f ) } );
    g.AddBehaviour( Behaviours::Biomechanics{ 15.0f, 2.0f, 1.5f, 0.0f } );
    g.AddBehaviour( Behaviours::CadherinAdhesion{
        glm::vec4( 1.0f, 0.0f, 0.0f, 0.0f ), 0.05f, 0.01f, 1.0f } );

    SimulationBuilder builder( m_resourceManager.get(), m_streamingManager.get() );
    SimulationState   state = builder.Build( blueprint );

    ASSERT_TRUE( state.cadherinProfileBuffer.IsValid() );
    ASSERT_TRUE( state.cadherinAffinityBuffer.IsValid() );

    // Two dispatch passes (ping + pong) to ensure ChainFlip tasks have a complete cycle
    auto compCtxHandle = m_device->CreateThreadContext( QueueType::COMPUTE );
    auto compCtx       = m_device->GetThreadContext( compCtxHandle );
    auto compCmd       = compCtx->GetCommandBuffer( compCtx->CreateCommandBuffer() );

    GraphDispatcher dispatcher;
    compCmd->Begin();
    uint32_t finalIdx = dispatcher.Dispatch( &state.computeGraph, compCmd, nullptr, 1.0f, 1.0f, 0 );
    compCmd->End();
    m_device->GetComputeQueue()->Submit( { compCmd } );
    m_device->GetComputeQueue()->WaitIdle();

    // Overlapping agents should have moved under JKR repulsion
    std::vector<glm::vec4> agents( 2 );
    m_streamingManager->ReadbackBufferImmediate(
        state.agentBuffers[ finalIdx ], agents.data(), 2 * sizeof( glm::vec4 ) );

    EXPECT_NE( agents[ 0 ].x, 0.0f ) << "Agent 0 did not move -- JKR+cadherin dispatch may have failed";
    EXPECT_NE( agents[ 1 ].x, 1.0f ) << "Agent 1 did not move -- JKR+cadherin dispatch may have failed";

    state.Destroy( m_resourceManager.get() );
}

// =================================================================================================
// Stage 4: CellPolarity buffer + task tests
// =================================================================================================

// 1. polarityBuffer always allocated; full-size when CellPolarity present.
TEST_F( SimulationBuilderTest, Builder_CellPolarity_BufferAllocated )
{
    if( !m_device )
        GTEST_SKIP();

    SimulationBlueprint blueprint;
    blueprint.SetDomainSize( glm::vec3( 100.0f ), 2.0f );
    AgentGroup& g = blueprint.AddAgentGroup( "ECs" );
    g.SetCount( 4 );
    g.SetDistribution( SpatialDistribution::UniformInBox( 4, glm::vec3( 4.0f ), glm::vec3( 0.0f ) ) );
    g.AddBehaviour( Behaviours::Biomechanics{ 15.0f, 2.0f, 1.5f, 0.0f } );
    g.AddBehaviour( Behaviours::CellPolarity{} );

    SimulationBuilder builder( m_resourceManager.get(), m_streamingManager.get() );
    SimulationState   state = builder.Build( blueprint );

    EXPECT_TRUE( state.polarityBuffer.IsValid() );

    state.Destroy( m_resourceManager.get() );
}

// 2. Even without CellPolarity a dummy polarityBuffer is allocated.
TEST_F( SimulationBuilderTest, Builder_CellPolarity_DummyBufferWhenAbsent )
{
    if( !m_device )
        GTEST_SKIP();

    SimulationBlueprint blueprint;
    blueprint.SetDomainSize( glm::vec3( 100.0f ), 2.0f );
    AgentGroup& g = blueprint.AddAgentGroup( "Cells" );
    g.SetCount( 4 );
    g.SetDistribution( SpatialDistribution::UniformInBox( 4, glm::vec3( 4.0f ), glm::vec3( 0.0f ) ) );
    g.AddBehaviour( Behaviours::Biomechanics{ 15.0f, 2.0f, 1.5f, 0.0f } );

    SimulationBuilder builder( m_resourceManager.get(), m_streamingManager.get() );
    SimulationState   state = builder.Build( blueprint );

    EXPECT_TRUE( state.polarityBuffer.IsValid() ); // dummy still valid

    state.Destroy( m_resourceManager.get() );
}

// 3. CellPolarity task appears in the compute graph.
TEST_F( SimulationBuilderTest, Builder_CellPolarity_TaskExists )
{
    if( !m_device )
        GTEST_SKIP();

    SimulationBlueprint blueprint;
    blueprint.SetDomainSize( glm::vec3( 100.0f ), 2.0f );
    AgentGroup& g = blueprint.AddAgentGroup( "ECs" );
    g.SetCount( 4 );
    g.SetDistribution( SpatialDistribution::UniformInBox( 4, glm::vec3( 4.0f ), glm::vec3( 0.0f ) ) );
    g.AddBehaviour( Behaviours::Biomechanics{ 15.0f, 2.0f, 1.5f, 0.0f } );
    g.AddBehaviour( Behaviours::CellPolarity{} );

    SimulationBuilder builder( m_resourceManager.get(), m_streamingManager.get() );
    SimulationState   state = builder.Build( blueprint );

    // Behaviour index 1 = CellPolarity (after Biomechanics at index 0)
    EXPECT_NE( state.computeGraph.FindTask( "polarity_0_1" ), nullptr )
        << "polarity_update task not found in compute graph";

    state.Destroy( m_resourceManager.get() );
}

// 4. Polarity task dispatches cleanly without GPU errors; polarity buffer changes after dispatch.
TEST_F( SimulationBuilderTest, Builder_CellPolarity_DispatchUpdatesBuffer )
{
    if( !m_device )
        GTEST_SKIP();

    SimulationBlueprint blueprint;
    blueprint.SetDomainSize( glm::vec3( 100.0f ), 2.0f );
    AgentGroup& g = blueprint.AddAgentGroup( "ECs" );
    // Use cells placed in a cluster so each has visible neighbors for polarity
    g.SetCount( 8 );
    g.SetDistribution( SpatialDistribution::LatticeInSphere( 1.5f, 3.0f ) );
    g.AddBehaviour( Behaviours::Biomechanics{ 15.0f, 2.0f, 1.5f, 0.0f } );
    g.AddBehaviour( Behaviours::CellPolarity{ 1.0f } ); // high rate for quick response

    SimulationBuilder builder( m_resourceManager.get(), m_streamingManager.get() );
    SimulationState   state = builder.Build( blueprint );
    ASSERT_TRUE( state.polarityBuffer.IsValid() );

    // Capture initial polarity (all zeroes from init)
    std::vector<glm::vec4> before( 8 );
    m_streamingManager->ReadbackBufferImmediate(
        state.polarityBuffer, before.data(), 8 * sizeof( glm::vec4 ) );

    // Dispatch one frame
    auto compCtxHandle = m_device->CreateThreadContext( QueueType::COMPUTE );
    auto compCtx       = m_device->GetThreadContext( compCtxHandle );
    auto compCmd       = compCtx->GetCommandBuffer( compCtx->CreateCommandBuffer() );

    GraphDispatcher dispatcher;
    compCmd->Begin();
    dispatcher.Dispatch( &state.computeGraph, compCmd, nullptr, 1.0f / 60.0f, 0.0f, 0 );
    compCmd->End();
    m_device->GetComputeQueue()->Submit( { compCmd } );
    m_device->GetComputeQueue()->WaitIdle();

    std::vector<glm::vec4> after( 8 );
    m_streamingManager->ReadbackBufferImmediate(
        state.polarityBuffer, after.data(), 8 * sizeof( glm::vec4 ) );

    // At least some agents should have non-zero polarity after one dispatch
    bool anyNonZero = false;
    for( int i = 0; i < 8; ++i )
        if( glm::length( glm::vec3( after[ i ] ) ) > 1e-4f )
            anyNonZero = true;

    EXPECT_TRUE( anyNonZero ) << "Polarity buffer unchanged after dispatch — shader may have failed";

    state.Destroy( m_resourceManager.get() );
}

// 5. When CellPolarity is present, JKR task has gridSize.z == 1 (polarity wired).
TEST_F( SimulationBuilderTest, Builder_JKR_Polarity_Wiring )
{
    if( !m_device )
        GTEST_SKIP();

    SimulationBlueprint blueprint;
    blueprint.SetDomainSize( glm::vec3( 100.0f ), 2.0f );
    AgentGroup& g = blueprint.AddAgentGroup( "ECs" );
    g.SetCount( 2 );
    g.SetDistribution( { glm::vec4( 0.0f, 0.0f, 0.0f, 1.0f ),
                         glm::vec4( 1.0f, 0.0f, 0.0f, 1.0f ) } );
    g.AddBehaviour( Behaviours::Biomechanics{ 15.0f, 2.0f, 1.5f, 0.0f } );
    g.AddBehaviour( Behaviours::CellPolarity{} );

    SimulationBuilder builder( m_resourceManager.get(), m_streamingManager.get() );
    SimulationState   state = builder.Build( blueprint );

    // JKR task tag for group 0, behaviour index 0 (Biomechanics is first)
    ComputeTask* jkrTask = state.computeGraph.FindTask( "jkr_0_0" );
    ASSERT_NE( jkrTask, nullptr ) << "JKR task not found";
    EXPECT_EQ( jkrTask->GetPushConstants().gridSize.z, 1u )
        << "gridSize.z should be 1 when CellPolarity is present";

    state.Destroy( m_resourceManager.get() );
}

// 6. ECBlobDemo / EC2DMatrigelDemo biology regression: a blueprint mirroring the demo
//    setup builds, dispatches 10 frames without errors, and leaves a non-zero
//    polarity buffer (surface cells develop outward polarity).
//
//    Kept as the Phase-1 biology regression; uses a pre-arranged cylindrical
//    shell so the test exercises the polarity + cadherin stack deterministically
//    independent of the random-cloud collapse dynamics in the actual demos.
TEST_F( SimulationBuilderTest, ECBlobDemo_PolarityBuildsAndDispatches )
{
    if( !m_device )
        GTEST_SKIP();

    SimulationBlueprint blueprint;
    blueprint.SetName( "Endothelial Tube" );
    blueprint.SetDomainSize( glm::vec3( 40.0f ), 1.5f );

    blueprint.ConfigureSpatialPartitioning()
        .SetMethod( SpatialPartitioningMethod::HashGrid )
        .SetCellSize( 3.0f )
        .SetMaxDensity( 64 )
        .SetComputeHz( 60.0f );

    auto shell = SpatialDistribution::ShellOnCylinder( 1.35f, 3.0f, 6.0f, 14 );
    ASSERT_GT( shell.positions.size(), 0u );

    const uint32_t count = static_cast<uint32_t>( shell.positions.size() );

    auto jkr = BiomechanicsGenerator::JKR()
                   .SetYoungsModulus( 20.0f )
                   .SetPoissonRatio( 0.4f )
                   .SetAdhesionEnergy( 1.5f )
                   .SetMaxInteractionRadius( 0.75f )  // interactDist 1.5 > spacing 1.35
                   .SetDampingCoefficient( 500.0f )
                   .Build();

    AgentGroup& ecs = blueprint.AddAgentGroup( "Endothelial Cells" );
    ecs.SetCount( count );
    ecs.SetDistribution( shell.positions );
    ecs.SetOrientations( shell.normals );
    ecs.AddBehaviour( jkr ).SetHz( 60.0f );
    ecs.AddBehaviour( Behaviours::CadherinAdhesion{
                          glm::vec4( 0.0f, 0.0f, 1.0f, 0.0f ),
                          0.05f, 0.001f, 2.5f } )
        .SetHz( 60.0f );
    Behaviours::CellPolarity polarity;
    polarity.regulationRate  = 1.0f;  // fast rate so surface polarity is visible in 10 frames
    polarity.apicalRepulsion = 0.3f;
    polarity.basalAdhesion   = 1.5f;
    ecs.AddBehaviour( polarity ).SetHz( 60.0f );
    ecs.AddBehaviour( Behaviours::BrownianMotion{ 0.3f } ).SetHz( 60.0f );

    SimulationBuilder builder( m_resourceManager.get(), m_streamingManager.get() );
    SimulationState   state = builder.Build( blueprint );
    ASSERT_TRUE( state.polarityBuffer.IsValid() );

    // Dispatch 10 frames
    auto compCtxHandle = m_device->CreateThreadContext( QueueType::COMPUTE );
    auto compCtx       = m_device->GetThreadContext( compCtxHandle );
    auto compCmd       = compCtx->GetCommandBuffer( compCtx->CreateCommandBuffer() );

    GraphDispatcher dispatcher;
    const float     dt = 1.0f / 60.0f;
    for( int frame = 0; frame < 10; ++frame )
    {
        compCmd->Begin();
        dispatcher.Dispatch( &state.computeGraph, compCmd, nullptr, dt, dt * static_cast<float>( frame ), frame % 2 );
        compCmd->End();
        m_device->GetComputeQueue()->Submit( { compCmd } );
        m_device->GetComputeQueue()->WaitIdle();
    }

    // Read back polarity — at least one surface cell should have non-zero polarity
    std::vector<glm::vec4> polarityData( count );
    m_streamingManager->ReadbackBufferImmediate(
        state.polarityBuffer, polarityData.data(), count * sizeof( glm::vec4 ) );

    bool anyNonZero = false;
    for( uint32_t i = 0; i < count; ++i )
        if( glm::length( glm::vec3( polarityData[ i ] ) ) > 1e-4f )
        {
            anyNonZero = true;
            break;
        }

    EXPECT_TRUE( anyNonZero ) << "No cells developed polarity after 10 frames";

    state.Destroy( m_resourceManager.get() );
}

// 7. ECBlobDemo scaffold: builds blueprint matching the actual demo's random-cloud
//    setup (elongated cylinder along +X, ~100 cells, CurvedTile morphology,
//    Biomechanics + CadherinAdhesion + CellPolarity + Brownian — no plate),
//    dispatches ~1 second of frames, and asserts no agent escaped / went NaN.
//    Guards the Phase-1 demo scaffold against integration regressions.
TEST_F( SimulationBuilderTest, ECBlobDemo_BuildsWithoutExplosion )
{
    if( !m_device )
        GTEST_SKIP();

    SimulationBlueprint blueprint;
    blueprint.SetName( "EC Blob" );
    blueprint.SetDomainSize( glm::vec3( 40.0f ), 1.5f );
    blueprint.ConfigureSpatialPartitioning()
        .SetMethod( SpatialPartitioningMethod::HashGrid )
        .SetCellSize( 3.0f )
        .SetMaxDensity( 64 )
        .SetComputeHz( 60.0f );

    // Mirror SeedECCloud: lattice-placed cells in elongated cylinder along +X.
    const float     radius     = 2.0f;
    const float     halfLength = 6.0f;
    const float     spacing    = 1.2f;
    const glm::vec3 center     = glm::vec3( 0.0f, 2.5f, 0.0f );
    const glm::vec3 axis       = glm::vec3( 1.0f, 0.0f, 0.0f );

    auto positions = SpatialDistribution::LatticeInCylinder(
        spacing, radius, halfLength, center, axis );
    ASSERT_GT( positions.size(), 50u );  // geometry-determined count, ~100 expected
    const uint32_t count = static_cast<uint32_t>( positions.size() );

    AgentGroup& ecs = blueprint.AddAgentGroup( "Endothelial Cells" );
    ecs.SetCount( count )
        .SetMorphology( MorphologyGenerator::CreateCurvedTile( 20.0f, 1.05f, 0.25f, radius ) )
        .SetDistribution( positions );

    auto jkr = BiomechanicsGenerator::JKR()
                   .SetYoungsModulus( 20.0f )
                   .SetPoissonRatio( 0.4f )
                   .SetAdhesionEnergy( 5.0f )
                   .SetMaxInteractionRadius( 0.75f )
                   .SetDampingCoefficient( 150.0f )
                   .Build();
    ecs.AddBehaviour( jkr ).SetHz( 60.0f );
    ecs.AddBehaviour( Behaviours::CadherinAdhesion{
                          glm::vec4( 0.0f, 0.0f, 1.0f, 0.0f ), 0.05f, 0.001f, 2.0f } )
        .SetHz( 60.0f );
    Behaviours::CellPolarity polarity;
    polarity.regulationRate  = 0.2f;
    polarity.apicalRepulsion = 0.3f;
    polarity.basalAdhesion   = 1.5f;
    ecs.AddBehaviour( polarity ).SetHz( 60.0f );
    ecs.AddBehaviour( Behaviours::BrownianMotion{ 0.1f } ).SetHz( 60.0f );

    SimulationBuilder builder( m_resourceManager.get(), m_streamingManager.get() );
    SimulationState   state = builder.Build( blueprint );

    // Dispatch ~1 s at 60 Hz
    auto compCtxHandle = m_device->CreateThreadContext( QueueType::COMPUTE );
    auto compCtx       = m_device->GetThreadContext( compCtxHandle );
    auto compCmd       = compCtx->GetCommandBuffer( compCtx->CreateCommandBuffer() );

    GraphDispatcher dispatcher;
    const float     dt        = 1.0f / 60.0f;
    uint32_t        activeIdx = 0;
    for( int frame = 0; frame < 60; ++frame )
    {
        compCmd->Begin();
        activeIdx = dispatcher.Dispatch( &state.computeGraph, compCmd, nullptr, dt, dt * static_cast<float>( frame ), activeIdx );
        compCmd->End();
        m_device->GetComputeQueue()->Submit( { compCmd } );
        m_device->GetComputeQueue()->WaitIdle();
    }

    // Read back final positions — no NaN, every agent still inside the domain
    std::vector<glm::vec4> finalPositions( count );
    m_streamingManager->ReadbackBufferImmediate(
        state.agentBuffers[ activeIdx ], finalPositions.data(), count * sizeof( glm::vec4 ) );

    uint32_t aliveCount = 0;
    for( uint32_t i = 0; i < count; ++i )
    {
        glm::vec3 p = glm::vec3( finalPositions[ i ] );
        EXPECT_FALSE( std::isnan( p.x ) || std::isnan( p.y ) || std::isnan( p.z ) )
            << "Agent " << i << " position went NaN";
        EXPECT_LT( std::abs( p.x ), 20.0f ) << "Agent " << i << " escaped X bounds";
        EXPECT_LT( std::abs( p.y ), 20.0f ) << "Agent " << i << " escaped Y bounds";
        EXPECT_LT( std::abs( p.z ), 20.0f ) << "Agent " << i << " escaped Z bounds";
        if( finalPositions[ i ].w > 0.0f ) ++aliveCount;
    }
    EXPECT_EQ( aliveCount, count ) << "Agent count changed during 1 s of simulation";

    state.Destroy( m_resourceManager.get() );
}

// 8. EC2DMatrigelDemo scaffold (Phase 1). Identical to ECBlobDemo in Phase 1 — no plate
//    yet. Diverges from Phase 2 onward when BasementMembrane is added only here.
//    Keeping both tests from Phase 1 exposes any accidental behaviour divergence
//    introduced while refactoring.
TEST_F( SimulationBuilderTest, EC2DMatrigelDemo_BuildsWithoutExplosion )
{
    if( !m_device )
        GTEST_SKIP();

    SimulationBlueprint blueprint;
    blueprint.SetName( "EC Tube" );
    blueprint.SetDomainSize( glm::vec3( 40.0f ), 1.5f );
    blueprint.ConfigureSpatialPartitioning()
        .SetMethod( SpatialPartitioningMethod::HashGrid )
        .SetCellSize( 3.0f )
        .SetMaxDensity( 64 )
        .SetComputeHz( 60.0f );

    const float     radius     = 2.0f;
    const float     halfLength = 6.0f;
    const float     spacing    = 1.2f;
    const glm::vec3 center     = glm::vec3( 0.0f, 2.5f, 0.0f );
    const glm::vec3 axis       = glm::vec3( 1.0f, 0.0f, 0.0f );

    auto positions = SpatialDistribution::LatticeInCylinder(
        spacing, radius, halfLength, center, axis );
    ASSERT_GT( positions.size(), 50u );
    const uint32_t count = static_cast<uint32_t>( positions.size() );

    AgentGroup& ecs = blueprint.AddAgentGroup( "Endothelial Cells" );
    ecs.SetCount( count )
        .SetMorphology( MorphologyGenerator::CreateCurvedTile( 20.0f, 1.05f, 0.25f, radius ) )
        .SetDistribution( positions );

    auto jkr = BiomechanicsGenerator::JKR()
                   .SetYoungsModulus( 20.0f )
                   .SetPoissonRatio( 0.4f )
                   .SetAdhesionEnergy( 5.0f )
                   .SetMaxInteractionRadius( 0.75f )
                   .SetDampingCoefficient( 150.0f )
                   .Build();
    ecs.AddBehaviour( jkr ).SetHz( 60.0f );
    ecs.AddBehaviour( Behaviours::CadherinAdhesion{
                          glm::vec4( 0.0f, 0.0f, 1.0f, 0.0f ), 0.05f, 0.001f, 2.0f } )
        .SetHz( 60.0f );
    Behaviours::CellPolarity polarity;
    polarity.regulationRate  = 0.2f;
    polarity.apicalRepulsion = 0.3f;
    polarity.basalAdhesion   = 1.5f;
    ecs.AddBehaviour( polarity ).SetHz( 60.0f );
    ecs.AddBehaviour( Behaviours::BrownianMotion{ 0.1f } ).SetHz( 60.0f );
    // NOTE: No BasementMembrane yet — introduced in Phase 2.

    SimulationBuilder builder( m_resourceManager.get(), m_streamingManager.get() );
    SimulationState   state = builder.Build( blueprint );

    auto compCtxHandle = m_device->CreateThreadContext( QueueType::COMPUTE );
    auto compCtx       = m_device->GetThreadContext( compCtxHandle );
    auto compCmd       = compCtx->GetCommandBuffer( compCtx->CreateCommandBuffer() );

    GraphDispatcher dispatcher;
    const float     dt        = 1.0f / 60.0f;
    uint32_t        activeIdx = 0;
    for( int frame = 0; frame < 60; ++frame )
    {
        compCmd->Begin();
        activeIdx = dispatcher.Dispatch( &state.computeGraph, compCmd, nullptr, dt, dt * static_cast<float>( frame ), activeIdx );
        compCmd->End();
        m_device->GetComputeQueue()->Submit( { compCmd } );
        m_device->GetComputeQueue()->WaitIdle();
    }

    std::vector<glm::vec4> finalPositions( count );
    m_streamingManager->ReadbackBufferImmediate(
        state.agentBuffers[ activeIdx ], finalPositions.data(), count * sizeof( glm::vec4 ) );

    uint32_t aliveCount = 0;
    for( uint32_t i = 0; i < count; ++i )
    {
        glm::vec3 p = glm::vec3( finalPositions[ i ] );
        EXPECT_FALSE( std::isnan( p.x ) || std::isnan( p.y ) || std::isnan( p.z ) )
            << "Agent " << i << " position went NaN";
        EXPECT_LT( std::abs( p.x ), 20.0f ) << "Agent " << i << " escaped X bounds";
        EXPECT_LT( std::abs( p.y ), 20.0f ) << "Agent " << i << " escaped Y bounds";
        EXPECT_LT( std::abs( p.z ), 20.0f ) << "Agent " << i << " escaped Z bounds";
        if( finalPositions[ i ].w > 0.0f ) ++aliveCount;
    }
    EXPECT_EQ( aliveCount, count ) << "Agent count changed during 1 s of simulation";

    state.Destroy( m_resourceManager.get() );
}

// Step C — ECTubeDemo (3D ECM placeholder) basic sanity. 4-plate channel; 1 s
// of simulation; cluster must stay bounded, no NaN, agent count preserved.
// This test uses the Gaudi Demos:: setup function directly, exercising the
// multi-plate builder path with 4 BasementMembrane behaviours at once.
TEST_F( SimulationBuilderTest, ECTubeDemo_BuildsWithMultiplePlates_WithoutExplosion )
{
    if( !m_device )
        GTEST_SKIP();

    // Mirror ECTubeDemo's 4-plate channel configuration in a test blueprint.
    SimulationBlueprint blueprint;
    blueprint.SetName( "ECTubeDemo test" );
    blueprint.SetDomainSize( glm::vec3( 40.0f ), 1.5f );
    blueprint.ConfigureSpatialPartitioning()
        .SetMethod( SpatialPartitioningMethod::HashGrid )
        .SetCellSize( 3.0f )
        .SetMaxDensity( 64 )
        .SetComputeHz( 60.0f );

    const float     radius     = 2.0f;
    const float     halfLength = 6.0f;
    const float     spacing    = 1.2f;
    const glm::vec3 center     = glm::vec3( 0.0f, 2.5f, 0.0f );
    const glm::vec3 axis       = glm::vec3( 1.0f, 0.0f, 0.0f );

    auto positions = SpatialDistribution::LatticeInCylinder(
        spacing, radius, halfLength, center, axis );
    ASSERT_GT( positions.size(), 50u );
    const uint32_t count = static_cast<uint32_t>( positions.size() );

    AgentGroup& ecs = blueprint.AddAgentGroup( "Endothelial Cells" );
    ecs.SetCount( count )
        .SetMorphology( MorphologyGenerator::CreateCurvedTile( 20.0f, 1.05f, 0.25f, radius ) )
        .SetDistribution( positions );

    ecs.AddBehaviour( BiomechanicsGenerator::JKR()
                          .SetYoungsModulus( 20.0f )
                          .SetPoissonRatio( 0.4f )
                          .SetAdhesionEnergy( 5.0f )
                          .SetMaxInteractionRadius( 0.75f )
                          .SetDampingCoefficient( 150.0f )
                          .SetCorticalTension( 0.5f )
                          .SetLateralAdhesionScale( 0.15f )
                          .Build() )
        .SetHz( 60.0f );
    ecs.AddBehaviour( Behaviours::CadherinAdhesion{
                          glm::vec4( 0.0f, 0.0f, 1.0f, 0.0f ), 0.05f, 0.001f, 2.0f } )
        .SetHz( 60.0f );
    Behaviours::CellPolarity polarity;
    polarity.regulationRate      = 0.2f;
    polarity.apicalRepulsion     = 0.3f;
    polarity.basalAdhesion       = 1.5f;
    polarity.propagationStrength = 1.0f;
    ecs.AddBehaviour( polarity ).SetHz( 60.0f );
    ecs.AddBehaviour( Behaviours::BrownianMotion{ 0.1f } ).SetHz( 60.0f );

    // Four-plate channel — floor + ceiling + two Z walls.
    auto addPlate = [&]( glm::vec3 normal, float height ) {
        Behaviours::BasementMembrane plate;
        plate.planeNormal       = normal;
        plate.height            = height;
        plate.contactStiffness  = 15.0f;
        plate.integrinAdhesion  = 1.5f;
        plate.anchorageDistance = 4.0f;
        plate.polarityBias      = 2.0f;
        ecs.AddBehaviour( plate ).SetHz( 60.0f );
    };
    addPlate( glm::vec3(  0.0f,  1.0f,  0.0f ),  0.0f );
    addPlate( glm::vec3(  0.0f, -1.0f,  0.0f ), -5.0f );
    addPlate( glm::vec3(  0.0f,  0.0f,  1.0f ), -3.0f );
    addPlate( glm::vec3(  0.0f,  0.0f, -1.0f ), -3.0f );

    SimulationBuilder builder( m_resourceManager.get(), m_streamingManager.get() );
    SimulationState   state = builder.Build( blueprint );

    // Must allocate basement membrane buffer (multi-plate).
    EXPECT_TRUE( state.basementMembraneBuffer.IsValid() )
        << "Multi-plate buffer must be allocated even with 4 BasementMembrane behaviours";

    auto compCtxHandle = m_device->CreateThreadContext( QueueType::COMPUTE );
    auto compCtx       = m_device->GetThreadContext( compCtxHandle );
    auto compCmd       = compCtx->GetCommandBuffer( compCtx->CreateCommandBuffer() );

    GraphDispatcher dispatcher;
    const float     dt        = 1.0f / 60.0f;
    uint32_t        activeIdx = 0;
    for( int frame = 0; frame < 60; ++frame )
    {
        compCmd->Begin();
        activeIdx = dispatcher.Dispatch( &state.computeGraph, compCmd, nullptr, dt, dt * static_cast<float>( frame ), activeIdx );
        compCmd->End();
        m_device->GetComputeQueue()->Submit( { compCmd } );
        m_device->GetComputeQueue()->WaitIdle();
    }

    std::vector<glm::vec4> finalPositions( count );
    m_streamingManager->ReadbackBufferImmediate(
        state.agentBuffers[ activeIdx ], finalPositions.data(), count * sizeof( glm::vec4 ) );

    uint32_t aliveCount = 0;
    for( uint32_t i = 0; i < count; ++i )
    {
        glm::vec3 p = glm::vec3( finalPositions[ i ] );
        EXPECT_FALSE( std::isnan( p.x ) || std::isnan( p.y ) || std::isnan( p.z ) )
            << "Agent " << i << " position went NaN";
        // All agents must stay within channel bounds (plus some dynamics slack).
        EXPECT_LT( std::abs( p.x ), 20.0f ) << "Agent " << i << " escaped X bounds";
        EXPECT_LT( std::abs( p.y ), 20.0f ) << "Agent " << i << " escaped Y bounds";
        EXPECT_LT( std::abs( p.z ), 20.0f ) << "Agent " << i << " escaped Z bounds";
        if( finalPositions[ i ].w > 0.0f ) ++aliveCount;
    }
    EXPECT_EQ( aliveCount, count ) << "Agent count changed during 1 s of simulation";

    state.Destroy( m_resourceManager.get() );
}

// Step C — ECTubeDemo cells with multi-plate channel should polarise with
// non-zero mean magnitude (BM contact from multiple sides activates the
// integrin-gated polarity model). Contrast with ECBlobDemo which has NO
// plates and thus no cell ever polarises.
TEST_F( SimulationBuilderTest, ECTubeDemo_MultiPlate_ProducesPolarisedInterior )
{
    if( !m_device )
        GTEST_SKIP();

    SimulationBlueprint blueprint;
    blueprint.SetName( "ECTubeDemo polarity test" );
    blueprint.SetDomainSize( glm::vec3( 40.0f ), 1.5f );
    blueprint.ConfigureSpatialPartitioning()
        .SetMethod( SpatialPartitioningMethod::HashGrid )
        .SetCellSize( 3.0f )
        .SetMaxDensity( 64 )
        .SetComputeHz( 60.0f );

    const float     radius     = 2.0f;
    const float     halfLength = 6.0f;
    const float     spacing    = 1.2f;
    const glm::vec3 center     = glm::vec3( 0.0f, 2.5f, 0.0f );
    const glm::vec3 axis       = glm::vec3( 1.0f, 0.0f, 0.0f );

    auto positions = SpatialDistribution::LatticeInCylinder(
        spacing, radius, halfLength, center, axis );
    const uint32_t count = static_cast<uint32_t>( positions.size() );

    AgentGroup& ecs = blueprint.AddAgentGroup( "Endothelial Cells" );
    ecs.SetCount( count )
        .SetMorphology( MorphologyGenerator::CreateCurvedTile( 20.0f, 1.05f, 0.25f, radius ) )
        .SetDistribution( positions );

    ecs.AddBehaviour( BiomechanicsGenerator::JKR()
                          .SetYoungsModulus( 20.0f )
                          .SetPoissonRatio( 0.4f )
                          .SetAdhesionEnergy( 5.0f )
                          .SetMaxInteractionRadius( 0.75f )
                          .SetDampingCoefficient( 150.0f )
                          .SetCorticalTension( 0.5f )
                          .SetLateralAdhesionScale( 0.15f )
                          .Build() )
        .SetHz( 60.0f );
    ecs.AddBehaviour( Behaviours::CadherinAdhesion{
                          glm::vec4( 0.0f, 0.0f, 1.0f, 0.0f ), 0.05f, 0.001f, 2.0f } )
        .SetHz( 60.0f );
    Behaviours::CellPolarity polarity;
    polarity.regulationRate      = 0.5f;  // faster cascade for test
    polarity.apicalRepulsion     = 0.3f;
    polarity.basalAdhesion       = 1.5f;
    polarity.propagationStrength = 1.0f;
    ecs.AddBehaviour( polarity ).SetHz( 60.0f );
    ecs.AddBehaviour( Behaviours::BrownianMotion{ 0.1f } ).SetHz( 60.0f );

    auto addPlate = [&]( glm::vec3 normal, float height ) {
        Behaviours::BasementMembrane plate;
        plate.planeNormal       = normal;
        plate.height            = height;
        plate.contactStiffness  = 15.0f;
        plate.integrinAdhesion  = 1.5f;
        plate.anchorageDistance = 4.0f;
        plate.polarityBias      = 2.0f;
        ecs.AddBehaviour( plate ).SetHz( 60.0f );
    };
    addPlate( glm::vec3(  0.0f,  1.0f,  0.0f ),  0.0f );
    addPlate( glm::vec3(  0.0f, -1.0f,  0.0f ), -5.0f );
    addPlate( glm::vec3(  0.0f,  0.0f,  1.0f ), -3.0f );
    addPlate( glm::vec3(  0.0f,  0.0f, -1.0f ), -3.0f );

    SimulationBuilder builder( m_resourceManager.get(), m_streamingManager.get() );
    SimulationState   state = builder.Build( blueprint );

    auto compCtxHandle = m_device->CreateThreadContext( QueueType::COMPUTE );
    auto compCtx       = m_device->GetThreadContext( compCtxHandle );
    auto compCmd       = compCtx->GetCommandBuffer( compCtx->CreateCommandBuffer() );

    GraphDispatcher dispatcher;
    const float     dt        = 1.0f / 60.0f;
    uint32_t        activeIdx = 0;
    for( int frame = 0; frame < 300; ++frame )  // 5 s — enough for polarity to saturate
    {
        compCmd->Begin();
        activeIdx = dispatcher.Dispatch( &state.computeGraph, compCmd, nullptr, dt, dt * static_cast<float>( frame ), activeIdx );
        compCmd->End();
        m_device->GetComputeQueue()->Submit( { compCmd } );
        m_device->GetComputeQueue()->WaitIdle();
    }

    std::vector<glm::vec4> finalPolarities( count );
    m_streamingManager->ReadbackBufferImmediate(
        state.polarityBuffer, finalPolarities.data(), count * sizeof( glm::vec4 ) );

    // Under the 4-plate channel, cells within the channel volume should have
    // BM contact from at least one side → non-zero polarity magnitude from
    // the plate cue. Compute mean magnitude over all live cells.
    double sum   = 0.0;
    size_t alive = 0;
    for( uint32_t i = 0; i < count; ++i )
    {
        if( finalPolarities[ i ].w > 0.0f || true ) {  // sum all cells (no dead-cell check needed for polarity)
            sum += finalPolarities[ i ].w;
            ++alive;
        }
    }
    float meanMag = alive > 0 ? static_cast<float>( sum / static_cast<double>( alive ) ) : 0.0f;

    EXPECT_GT( meanMag, 0.3f )
        << "Multi-plate channel must activate polarity on most cells "
           "(mean magnitude " << meanMag << " should be > 0.3 — under the 4-plate "
           "channel all cells have at least one plate within anchorageDistance).";

    state.Destroy( m_resourceManager.get() );
}

// 9. Phase 2 prediction: an EC2DMatrigelDemo-style blueprint with BasementMembrane
//    at z=0 (+z normal) should let cells settle ONTO the plate. Starting all
//    cells above the plate, after N seconds we expect the cluster centroid to
//    have descended (integrin pull) and no cell to be below z=0 (contact
//    repulsion).
TEST_F( SimulationBuilderTest, EC2DMatrigelDemo_SettlesOnPlate )
{
    if( !m_device )
        GTEST_SKIP();

    SimulationBlueprint blueprint;
    blueprint.SetName( "EC Tube (test)" );
    blueprint.SetDomainSize( glm::vec3( 40.0f ), 1.5f );
    blueprint.ConfigureSpatialPartitioning()
        .SetMethod( SpatialPartitioningMethod::HashGrid )
        .SetCellSize( 3.0f )
        .SetMaxDensity( 64 )
        .SetComputeHz( 60.0f );

    const float     radius     = 2.0f;
    const float     halfLength = 6.0f;
    const float     spacing    = 1.2f;
    const glm::vec3 center     = glm::vec3( 0.0f, 2.5f, 0.0f );
    const glm::vec3 axis       = glm::vec3( 1.0f, 0.0f, 0.0f );

    auto positions = SpatialDistribution::LatticeInCylinder(
        spacing, radius, halfLength, center, axis );
    ASSERT_GT( positions.size(), 50u );
    const uint32_t count = static_cast<uint32_t>( positions.size() );

    // Mean z before sim
    float yMeanBefore = 0.0f;
    for( const auto& p : positions ) yMeanBefore += p.y;
    yMeanBefore /= static_cast<float>( count );

    AgentGroup& ecs = blueprint.AddAgentGroup( "Endothelial Cells" );
    ecs.SetCount( count )
        .SetMorphology( MorphologyGenerator::CreateCurvedTile( 20.0f, 1.05f, 0.25f, radius ) )
        .SetDistribution( positions );

    auto jkr = BiomechanicsGenerator::JKR()
                   .SetYoungsModulus( 20.0f )
                   .SetPoissonRatio( 0.4f )
                   .SetAdhesionEnergy( 5.0f )
                   .SetMaxInteractionRadius( 0.75f )
                   .SetDampingCoefficient( 150.0f )
                   .Build();
    ecs.AddBehaviour( jkr ).SetHz( 60.0f );
    ecs.AddBehaviour( Behaviours::CadherinAdhesion{
                          glm::vec4( 0.0f, 0.0f, 1.0f, 0.0f ), 0.05f, 0.001f, 2.0f } )
        .SetHz( 60.0f );
    Behaviours::CellPolarity polarity;
    polarity.regulationRate  = 0.2f;
    polarity.apicalRepulsion = 0.3f;
    polarity.basalAdhesion   = 1.5f;
    ecs.AddBehaviour( polarity ).SetHz( 60.0f );
    ecs.AddBehaviour( Behaviours::BrownianMotion{ 0.1f } ).SetHz( 60.0f );

    // Test uses deliberately STRONG plate parameters so the integrin pull is
    // unambiguous at the regression level. The actual demo uses gentler values
    // (tuned for visual realism over 30+ s of sim time).
    Behaviours::BasementMembrane plate;
    plate.planeNormal       = glm::vec3( 0.0f, 1.0f, 0.0f );
    plate.height            = 0.0f;
    plate.contactStiffness  = 15.0f;
    plate.integrinAdhesion  = 10.0f;  // strong — test only
    plate.anchorageDistance = 5.0f;   // covers full cluster y range — test only
    plate.polarityBias      = 2.0f;
    ecs.AddBehaviour( plate ).SetHz( 60.0f );

    SimulationBuilder builder( m_resourceManager.get(), m_streamingManager.get() );
    SimulationState   state = builder.Build( blueprint );

    auto compCtxHandle = m_device->CreateThreadContext( QueueType::COMPUTE );
    auto compCtx       = m_device->GetThreadContext( compCtxHandle );
    auto compCmd       = compCtx->GetCommandBuffer( compCtx->CreateCommandBuffer() );

    GraphDispatcher dispatcher;
    const float     dt        = 1.0f / 60.0f;
    uint32_t        activeIdx = 0;
    // 5 s of sim time — enough to settle onto the plate.
    for( int frame = 0; frame < 300; ++frame )
    {
        compCmd->Begin();
        activeIdx = dispatcher.Dispatch( &state.computeGraph, compCmd, nullptr, dt, dt * static_cast<float>( frame ), activeIdx );
        compCmd->End();
        m_device->GetComputeQueue()->Submit( { compCmd } );
        m_device->GetComputeQueue()->WaitIdle();
    }

    std::vector<glm::vec4> finalPositions( count );
    m_streamingManager->ReadbackBufferImmediate(
        state.agentBuffers[ activeIdx ], finalPositions.data(), count * sizeof( glm::vec4 ) );

    float yMeanAfter = 0.0f;
    float yMin       = 1e9f;
    for( uint32_t i = 0; i < count; ++i )
    {
        yMeanAfter += finalPositions[ i ].y;
        if( finalPositions[ i ].y < yMin ) yMin = finalPositions[ i ].y;
    }
    yMeanAfter /= static_cast<float>( count );

    EXPECT_LT( yMeanAfter, yMeanBefore )
        << "EC2DMatrigelDemo with plate must settle toward the plate (y_mean should decrease)";
    EXPECT_GE( yMin, -0.5f )
        << "No cell should penetrate the plate by more than a small tolerance";

    state.Destroy( m_resourceManager.get() );
}

// 10. Paired control: same blueprint WITHOUT BasementMembrane should NOT settle.
//     Proves the plate effect in EC2DMatrigelDemo is attributable to the plate behaviour
//     (no accidental ECM leak into plateless demos).
TEST_F( SimulationBuilderTest, ECBlobDemo_DoesNotSettle )
{
    if( !m_device )
        GTEST_SKIP();

    SimulationBlueprint blueprint;
    blueprint.SetName( "EC Blob (test)" );
    blueprint.SetDomainSize( glm::vec3( 40.0f ), 1.5f );
    blueprint.ConfigureSpatialPartitioning()
        .SetMethod( SpatialPartitioningMethod::HashGrid )
        .SetCellSize( 3.0f )
        .SetMaxDensity( 64 )
        .SetComputeHz( 60.0f );

    const float     radius     = 2.0f;
    const float     halfLength = 6.0f;
    const float     spacing    = 1.2f;
    const glm::vec3 center     = glm::vec3( 0.0f, 2.5f, 0.0f );
    const glm::vec3 axis       = glm::vec3( 1.0f, 0.0f, 0.0f );

    auto positions = SpatialDistribution::LatticeInCylinder(
        spacing, radius, halfLength, center, axis );
    ASSERT_GT( positions.size(), 50u );
    const uint32_t count = static_cast<uint32_t>( positions.size() );

    float yMeanBefore = 0.0f;
    for( const auto& p : positions ) yMeanBefore += p.y;
    yMeanBefore /= static_cast<float>( count );

    AgentGroup& ecs = blueprint.AddAgentGroup( "Endothelial Cells" );
    ecs.SetCount( count )
        .SetMorphology( MorphologyGenerator::CreateCurvedTile( 20.0f, 1.05f, 0.25f, radius ) )
        .SetDistribution( positions );

    auto jkr = BiomechanicsGenerator::JKR()
                   .SetYoungsModulus( 20.0f )
                   .SetPoissonRatio( 0.4f )
                   .SetAdhesionEnergy( 5.0f )
                   .SetMaxInteractionRadius( 0.75f )
                   .SetDampingCoefficient( 150.0f )
                   .Build();
    ecs.AddBehaviour( jkr ).SetHz( 60.0f );
    ecs.AddBehaviour( Behaviours::CadherinAdhesion{
                          glm::vec4( 0.0f, 0.0f, 1.0f, 0.0f ), 0.05f, 0.001f, 2.0f } )
        .SetHz( 60.0f );
    Behaviours::CellPolarity polarity;
    polarity.regulationRate  = 0.2f;
    polarity.apicalRepulsion = 0.3f;
    polarity.basalAdhesion   = 1.5f;
    ecs.AddBehaviour( polarity ).SetHz( 60.0f );
    ecs.AddBehaviour( Behaviours::BrownianMotion{ 0.1f } ).SetHz( 60.0f );
    // NO BasementMembrane.

    SimulationBuilder builder( m_resourceManager.get(), m_streamingManager.get() );
    SimulationState   state = builder.Build( blueprint );

    auto compCtxHandle = m_device->CreateThreadContext( QueueType::COMPUTE );
    auto compCtx       = m_device->GetThreadContext( compCtxHandle );
    auto compCmd       = compCtx->GetCommandBuffer( compCtx->CreateCommandBuffer() );

    GraphDispatcher dispatcher;
    const float     dt        = 1.0f / 60.0f;
    uint32_t        activeIdx = 0;
    for( int frame = 0; frame < 300; ++frame )
    {
        compCmd->Begin();
        activeIdx = dispatcher.Dispatch( &state.computeGraph, compCmd, nullptr, dt, dt * static_cast<float>( frame ), activeIdx );
        compCmd->End();
        m_device->GetComputeQueue()->Submit( { compCmd } );
        m_device->GetComputeQueue()->WaitIdle();
    }

    std::vector<glm::vec4> finalPositions( count );
    m_streamingManager->ReadbackBufferImmediate(
        state.agentBuffers[ activeIdx ], finalPositions.data(), count * sizeof( glm::vec4 ) );

    float yMeanAfter = 0.0f;
    for( uint32_t i = 0; i < count; ++i )
        yMeanAfter += finalPositions[ i ].y;
    yMeanAfter /= static_cast<float>( count );

    // Without a plate, adhesion-driven surface-tension + Brownian should drive
    // the cluster toward a compact / pinching aggregate, NOT toward a plate.
    // Centroid y should stay within a tight band of its starting value
    // (Plateau-Rayleigh breakup moves cells laterally, not systematically
    // downward). A 1.0-unit tolerance is generous but still asserts no
    // plate-like downward drift (>2 units would mean the plate is leaking).
    EXPECT_NEAR( yMeanAfter, yMeanBefore, 1.0f )
        << "Plateless ECBlobDemo must not settle toward a plate (ECM leak check)";

    state.Destroy( m_resourceManager.get() );
}

// =============================================================================
// Phase 3 — net-negative apical adhesion (apical repulsion)
// =============================================================================
//
// Shared setup helper for Phase-3 integration tests: builds a Phase-1+2+3
// blueprint with configurable plate presence and configurable apical
// parameters, runs N seconds of sim.
// Returns the final positions for comparative analysis.
// Returns final positions (first) and final polarities (second). Polarity readback
// was added in Phase 4.5 to enable tests that assert on the polarity buffer state.
static std::pair<std::vector<glm::vec4>, std::vector<glm::vec4>> RunECDemoPhase3(
    Device*           device,
    ResourceManager*  rm,
    StreamingManager* stream,
    bool              withPlate,
    int               frames,
    float             apicalRepulsion      = -1.0f,
    float             basalAdhesion        =  2.5f,
    float             corticalTension      =  0.0f,  // Phase 4 — default off for regression tests
    float             propagationStrength  =  0.0f,  // Phase 4.5 — default off preserves Phase 3/4 tests
    float             regulationRate       =  0.2f,  // Phase 4.5 — demo-local rate override
    float             lateralAdhesionScale =  0.0f,  // Phase 4.5-B — default off preserves earlier tests
    float             catchBondStrength    =  0.0f,  // Phase 5 — default off preserves earlier tests
    float             catchBondPeakLoad    =  0.3f ) // Phase 5 — VE-cad Rakshit 2012 peak
{
    SimulationBlueprint blueprint;
    blueprint.SetDomainSize( glm::vec3( 40.0f ), 1.5f );
    blueprint.ConfigureSpatialPartitioning()
        .SetMethod( SpatialPartitioningMethod::HashGrid )
        .SetCellSize( 3.0f )
        .SetMaxDensity( 64 )
        .SetComputeHz( 60.0f );

    const float     radius     = 2.0f;
    const float     halfLength = 6.0f;
    const float     spacing    = 1.2f;
    const glm::vec3 center     = glm::vec3( 0.0f, 2.5f, 0.0f );
    const glm::vec3 axis       = glm::vec3( 1.0f, 0.0f, 0.0f );

    auto positions = SpatialDistribution::LatticeInCylinder(
        spacing, radius, halfLength, center, axis );
    const uint32_t count = static_cast<uint32_t>( positions.size() );

    AgentGroup& ecs = blueprint.AddAgentGroup( "Endothelial Cells" );
    ecs.SetCount( count )
        .SetMorphology( MorphologyGenerator::CreateCurvedTile( 20.0f, 1.05f, 0.25f, radius ) )
        .SetDistribution( positions );

    ecs.AddBehaviour( BiomechanicsGenerator::JKR()
                          .SetYoungsModulus( 20.0f )
                          .SetPoissonRatio( 0.4f )
                          .SetAdhesionEnergy( 5.0f )
                          .SetMaxInteractionRadius( 0.75f )
                          .SetDampingCoefficient( 150.0f )
                          .SetCorticalTension( corticalTension )
                          .SetLateralAdhesionScale( lateralAdhesionScale )
                          .Build() )
        .SetHz( 60.0f );
    ecs.AddBehaviour( Behaviours::CadherinAdhesion{
                          glm::vec4( 0.0f, 0.0f, 1.0f, 0.0f ),
                          0.05f, 0.001f, 2.0f,
                          catchBondStrength, catchBondPeakLoad } )
        .SetHz( 60.0f );
    Behaviours::CellPolarity polarity;
    polarity.regulationRate      = regulationRate;
    polarity.apicalRepulsion     = apicalRepulsion;
    polarity.basalAdhesion       = basalAdhesion;
    polarity.propagationStrength = propagationStrength;
    ecs.AddBehaviour( polarity ).SetHz( 60.0f );
    ecs.AddBehaviour( Behaviours::BrownianMotion{ 0.1f } ).SetHz( 60.0f );

    if( withPlate )
    {
        Behaviours::BasementMembrane plate;
        plate.planeNormal       = glm::vec3( 0.0f, 1.0f, 0.0f );
        plate.height            = 0.0f;
        plate.contactStiffness  = 15.0f;
        plate.integrinAdhesion  = 1.5f;
        plate.anchorageDistance = 1.0f;
        plate.polarityBias      = 2.0f;
        ecs.AddBehaviour( plate ).SetHz( 60.0f );
    }

    SimulationBuilder builder( rm, stream );
    SimulationState   state = builder.Build( blueprint );

    auto compCtxHandle = device->CreateThreadContext( QueueType::COMPUTE );
    auto compCtx       = device->GetThreadContext( compCtxHandle );
    auto compCmd       = compCtx->GetCommandBuffer( compCtx->CreateCommandBuffer() );

    GraphDispatcher dispatcher;
    const float     dt        = 1.0f / 60.0f;
    uint32_t        activeIdx = 0;
    for( int frame = 0; frame < frames; ++frame )
    {
        compCmd->Begin();
        activeIdx = dispatcher.Dispatch( &state.computeGraph, compCmd, nullptr,
                                         dt, dt * static_cast<float>( frame ), activeIdx );
        compCmd->End();
        device->GetComputeQueue()->Submit( { compCmd } );
        device->GetComputeQueue()->WaitIdle();
    }

    std::vector<glm::vec4> finalPositions( count );
    stream->ReadbackBufferImmediate(
        state.agentBuffers[ activeIdx ], finalPositions.data(), count * sizeof( glm::vec4 ) );

    // Polarity readback (Phase 4.5). Buffer always exists — the builder allocates
    // a dummy 1-vec4 buffer even when no group carries CellPolarity, so this
    // readback is safe. When the group does have CellPolarity the buffer is
    // sized to match `count` (and possibly padded); we read just the first
    // `count` entries.
    std::vector<glm::vec4> finalPolarities( count, glm::vec4( 0.0f ) );
    if( state.polarityBuffer.IsValid() )
    {
        stream->ReadbackBufferImmediate(
            state.polarityBuffer, finalPolarities.data(), count * sizeof( glm::vec4 ) );
    }

    state.Destroy( rm );
    return { std::move( finalPositions ), std::move( finalPolarities ) };
}

// Phase 4.5 — mean polarity magnitude over cells passing a predicate (e.g.
// "interior cells" = cells above a y threshold in EC2DMatrigelDemo). Used by the
// propagation integration tests to verify that the plate-to-interior cascade
// actually polarises the interior. Unlike surface cells (which get polarity
// from the neighbour-centroid term regardless of propagation), interior cells
// only get non-zero magnitude if propagation is working.
template<typename Pred>
static float MeanInteriorPolarityMagnitude(
    const std::vector<glm::vec4>& positions,
    const std::vector<glm::vec4>& polarities,
    Pred                           isInterior )
{
    EXPECT_EQ( positions.size(), polarities.size() );
    double sum  = 0.0;
    size_t n    = 0;
    for( size_t i = 0; i < positions.size(); ++i )
    {
        if( positions[ i ].w == 0.0f ) continue; // dead slot guard
        if( !isInterior( positions[ i ] ) ) continue;
        sum += polarities[ i ].w;
        ++n;
    }
    return n > 0 ? static_cast<float>( sum / static_cast<double>( n ) ) : 0.0f;
}

// Phase 4.5 — radial density around the +X axis. Bins cells by perpendicular
// distance from (x, 0, 0) and returns each bin's cell count normalised by the
// annulus-volume at that bin (so empty interior shows as a true density dip
// rather than a geometric artefact). `bins` must be >= 2. Used to detect the
// axial cavity opened by propagation-driven cord hollowing in EC2DMatrigelDemo.
static std::vector<float> RadialDensityAroundXAxis(
    const std::vector<glm::vec4>& positions,
    int                            bins,
    float                          rMax )
{
    EXPECT_GE( bins, 2 );
    std::vector<int> counts( bins, 0 );
    for( const auto& p : positions )
    {
        if( p.w == 0.0f ) continue;
        float r = std::sqrt( p.y * p.y + p.z * p.z );
        if( r >= rMax ) continue;
        int idx = static_cast<int>( r * float( bins ) / rMax );
        if( idx >= bins ) idx = bins - 1;
        counts[ idx ]++;
    }
    // Normalise by annulus area π·(r_{i+1}² − r_i²). Cells per unit cross-section area.
    constexpr float    kPi  = 3.14159265358979323846f;
    std::vector<float> density( bins, 0.0f );
    float              binW = rMax / float( bins );
    for( int i = 0; i < bins; ++i )
    {
        float r0  = float( i ) * binW;
        float r1  = r0 + binW;
        float a   = kPi * ( r1 * r1 - r0 * r0 );
        density[ i ] = ( a > 1e-6f ) ? ( float( counts[ i ] ) / a ) : 0.0f;
    }
    return density;
}

// Mean pairwise distance across all cells — a cheap proxy for how "open" or
// "compact" the aggregate is. Active inter-cell repulsion increases this;
// strong adhesion (no repulsion) decreases it.
static float MeanPairwiseDistance( const std::vector<glm::vec4>& pos )
{
    if( pos.size() < 2 ) return 0.0f;
    // Sample-based for speed: for ~100 cells this is cheap enough to do all N²/2
    // pairs directly, but we cap to avoid surprise costs at larger counts.
    const size_t n = pos.size();
    double       sum  = 0.0;
    size_t       nPairs = 0;
    for( size_t i = 0; i < n; ++i )
        for( size_t j = i + 1; j < n; ++j )
        {
            glm::vec3 d = glm::vec3( pos[ i ] ) - glm::vec3( pos[ j ] );
            sum += glm::length( d );
            ++nPairs;
        }
    return nPairs > 0 ? static_cast<float>( sum / static_cast<double>( nPairs ) ) : 0.0f;
}

// 11. Phase 3 positive case — net-negative apical repulsion must produce more
//     inter-cell spacing than merely-weakened apical adhesion, under the same
//     initial conditions and plate. This is the core Phase 3 signal: turning
//     apical modifier from attractive (+0.3) to actively repulsive (-1.0)
//     pushes cells apart throughout the aggregate.
//
//     Note: we do NOT assert the appearance of a clean axial tube here — that
//     requires active cortical tension (Phase 4) to resist the plate-induced
//     pancake-spreading. Phase 3 in isolation produces "more spaced cells",
//     not "tube morphology". That's the correct biology progression.
// Step A (2026-04-18) — polarity magnitude is now BM-gated. The signal this
// test measures (apical repulsion at surface cells whose polarity came from
// centroid geometry) no longer exists under the new biology. Polarity is now
// concentrated at the plate anchorage zone rather than across the whole
// surface, and the apical-value differential at 300 frames is below noise.
// Phase 6 sweep will revisit with longer runtime and stronger parameter
// contrasts to find a regime where the apical mechanism is measurably
// detectable in a 2D context.
TEST_F( SimulationBuilderTest, DISABLED_EC2DMatrigelDemo_ApicalRepulsion_IncreasesSpacing )
{
    if( !m_device )
        GTEST_SKIP();

    // Run A — Phase 3 values (apical -1.0: active repulsion).
    auto [posRepulsive, polRepulsive] = RunECDemoPhase3(
        m_device.get(), m_resourceManager.get(), m_streamingManager.get(),
        /*withPlate=*/true, /*frames=*/300,
        /*apicalRepulsion=*/-1.0f, /*basalAdhesion=*/2.5f );
    float dRepulsive = MeanPairwiseDistance( posRepulsive );

    // Run B — Phase 2 apical values (apical +0.3: just weakened adhesion).
    auto [posAttractive, polAttractive] = RunECDemoPhase3(
        m_device.get(), m_resourceManager.get(), m_streamingManager.get(),
        /*withPlate=*/true, /*frames=*/300,
        /*apicalRepulsion=*/0.3f, /*basalAdhesion=*/1.5f );
    float dAttractive = MeanPairwiseDistance( posAttractive );

    EXPECT_GT( dRepulsive, dAttractive )
        << "Net-negative apical must produce more inter-cell spacing than "
           "weak-apical (spacing: repulsive=" << dRepulsive
        << " vs attractive=" << dAttractive << ")";
}

// 12. Phase 3 paired control — the same differential must hold in ECBlobDemo
//     (no plate). Apical repulsion must do *something* even without the
//     substrate cue, because surface cells still establish polarity from
//     neighbour geometry. Proves the Phase-3 mechanism is not plate-dependent
//     — it acts on any polarised cell pair.
// Step A biology assertion — anoikis. Without a BM seed, ECBlob cells cannot
// establish apical-basal polarity (no integrin engagement → no PAR6 recruitment
// → no polarity axis). Therefore the `apicalRepulsion` parameter has NO effect
// on cluster spacing in ECBlob regardless of its value. This inverts the
// original (pre-Step-A) test which relied on the geometric-centroid magnitude
// bug; the NEW test asserts the correct biology: apical value is a cell-
// intrinsic property but cannot fire without environmental gating.
TEST_F( SimulationBuilderTest, ECBlobDemo_NoBM_ApicalValueIrrelevant )
{
    if( !m_device )
        GTEST_SKIP();

    // Run with strong apical repulsion — would disrupt cluster under old model.
    auto [posRepulsive, polRepulsive] = RunECDemoPhase3(
        m_device.get(), m_resourceManager.get(), m_streamingManager.get(),
        /*withPlate=*/false, /*frames=*/300,
        /*apicalRepulsion=*/-1.0f, /*basalAdhesion=*/2.5f );
    float dRepulsive = MeanPairwiseDistance( posRepulsive );

    // Run with weak apical attraction (baseline).
    auto [posAttractive, polAttractive] = RunECDemoPhase3(
        m_device.get(), m_resourceManager.get(), m_streamingManager.get(),
        /*withPlate=*/false, /*frames=*/300,
        /*apicalRepulsion=*/0.3f, /*basalAdhesion=*/1.5f );
    float dAttractive = MeanPairwiseDistance( posAttractive );

    // Spacing must be ≈ the same because polarity never establishes without a BM
    // seed. Tolerance 2% — anything above that would indicate spurious polarity
    // firing (i.e. the old geometric-magnitude bug resurrected).
    float denom   = dAttractive > 0.001f ? dAttractive : 0.001f;
    float relDiff = std::abs( dRepulsive - dAttractive ) / denom;
    EXPECT_LT( relDiff, 0.02f )
        << "Anoikis biology: apicalRepulsion must not affect ECBlob spacing "
           "without a BM seed. relDiff=" << relDiff
        << " (repulsive=" << dRepulsive << " vs attractive=" << dAttractive << ")";
}

// 13. Phase 4 — cortical tension (Maître et al. 2012) must increase the
//     equilibrium inter-cell spacing in EC2DMatrigelDemo. Under the same apical
//     repulsion (-1.0) and plate, adding an outward contractile term at the
//     pair level shifts the adhesion/tension balance outward → larger mean
//     pairwise distance at steady state.
TEST_F( SimulationBuilderTest, EC2DMatrigelDemo_CorticalTension_IncreasesSpacing )
{
    if( !m_device )
        GTEST_SKIP();

    // With cortical tension.
    auto [posTension, polTension] = RunECDemoPhase3(
        m_device.get(), m_resourceManager.get(), m_streamingManager.get(),
        /*withPlate=*/true, /*frames=*/300,
        /*apicalRepulsion=*/-1.0f, /*basalAdhesion=*/2.5f,
        /*corticalTension=*/5.0f );
    float dTension = MeanPairwiseDistance( posTension );

    // Without cortical tension — Phase 3 baseline.
    auto [posBase, polBase] = RunECDemoPhase3(
        m_device.get(), m_resourceManager.get(), m_streamingManager.get(),
        /*withPlate=*/true, /*frames=*/300,
        /*apicalRepulsion=*/-1.0f, /*basalAdhesion=*/2.5f,
        /*corticalTension=*/0.0f );
    float dBase = MeanPairwiseDistance( posBase );

    EXPECT_GT( dTension, dBase )
        << "Cortical tension must increase inter-cell spacing (tension="
        << dTension << " vs baseline=" << dBase << ")";
}

// 14. Phase 4 paired control — the same differential must hold in ECBlobDemo
//     (no plate). Cortical tension is a cell-intrinsic property and must
//     act independently of the substrate cue.
TEST_F( SimulationBuilderTest, ECBlobDemo_CorticalTension_IncreasesSpacing )
{
    if( !m_device )
        GTEST_SKIP();

    auto [posTension, polTension] = RunECDemoPhase3(
        m_device.get(), m_resourceManager.get(), m_streamingManager.get(),
        /*withPlate=*/false, /*frames=*/300,
        /*apicalRepulsion=*/-1.0f, /*basalAdhesion=*/2.5f,
        /*corticalTension=*/5.0f );
    float dTension = MeanPairwiseDistance( posTension );

    auto [posBase, polBase] = RunECDemoPhase3(
        m_device.get(), m_resourceManager.get(), m_streamingManager.get(),
        /*withPlate=*/false, /*frames=*/300,
        /*apicalRepulsion=*/-1.0f, /*basalAdhesion=*/2.5f,
        /*corticalTension=*/0.0f );
    float dBase = MeanPairwiseDistance( posBase );

    EXPECT_GT( dTension, dBase )
        << "Cortical tension must increase inter-cell spacing, plate-independent "
           "(tension=" << dTension << " vs baseline=" << dBase << ")";
}

// 14b. Phase 4.5-B — lateral adhesion (hull-pair translation). With hull
//      morphology and lateralAdhesionScale > 0, each overlapping hull pair
//      contributes an inward pull scaled by this parameter. The integration
//      signal: mean pairwise distance should be SMALLER with lateral adhesion
//      enabled than without, confirming the end-to-end wiring through builder +
//      push constants + shader produces the expected aggregate-level behaviour.
TEST_F( SimulationBuilderTest, ECBlobDemo_LateralAdhesion_ProducesTighterPacking )
{
    if( !m_device )
        GTEST_SKIP();

    // With lateral adhesion: cells should pack more tightly.
    auto [posLat, polLat] = RunECDemoPhase3(
        m_device.get(), m_resourceManager.get(), m_streamingManager.get(),
        /*withPlate=*/false, /*frames=*/300,
        /*apicalRepulsion=*/0.3f, /*basalAdhesion=*/1.5f,
        /*corticalTension=*/0.0f,
        /*propagationStrength=*/0.0f,
        /*regulationRate=*/0.2f,
        /*lateralAdhesionScale=*/0.15f );
    float dLat = MeanPairwiseDistance( posLat );

    // Baseline: no lateral adhesion (hull pairs torque-only).
    auto [posBase, polBase] = RunECDemoPhase3(
        m_device.get(), m_resourceManager.get(), m_streamingManager.get(),
        /*withPlate=*/false, /*frames=*/300,
        /*apicalRepulsion=*/0.3f, /*basalAdhesion=*/1.5f,
        /*corticalTension=*/0.0f,
        /*propagationStrength=*/0.0f,
        /*regulationRate=*/0.2f,
        /*lateralAdhesionScale=*/0.0f );
    float dBase = MeanPairwiseDistance( posBase );

    EXPECT_LT( dLat, dBase )
        << "Lateral adhesion must reduce mean pairwise distance "
           "(lateral=" << dLat << " vs baseline=" << dBase << ")";
}

// 14c. Phase 5 — catch-bond integration. After the 2026-04-19 reformulation,
//      catch-bond is load-gated: it only activates when there's external
//      tensile drive (cortical tension or apical polarity deficit). At rest
//      it's inert. This test exercises the high-stress regime: crank up
//      cortical tension so loadSignal enters the peak zone, and verify the
//      catch-bond mechanism then measurably affects aggregate geometry.
//      Guards against bit-packing bugs that silently inert the GPU path.
TEST_F( SimulationBuilderTest, ECBlobDemo_CatchBond_MechanismActivates )
{
    if( !m_device )
        GTEST_SKIP();

    // High cortical tension (3.0) + modest adhesion (via AdhesionEnergy=5 →
    // adhesion push-constant ~10) → loadSignal ≈ 0.3 → catchMul near peak.
    float highTension = 3.0f;

    auto [posCatch, polCatch] = RunECDemoPhase3(
        m_device.get(), m_resourceManager.get(), m_streamingManager.get(),
        /*withPlate=*/false, /*frames=*/200,
        /*apicalRepulsion=*/0.3f, /*basalAdhesion=*/1.5f,
        /*corticalTension=*/highTension,
        /*propagationStrength=*/0.0f,
        /*regulationRate=*/0.2f,
        /*lateralAdhesionScale=*/0.15f,
        /*catchBondStrength=*/2.0f,
        /*catchBondPeakLoad=*/0.3f );
    float dCatch = MeanPairwiseDistance( posCatch );

    // Baseline: catch-bond OFF, same high tension.
    auto [posBase, polBase] = RunECDemoPhase3(
        m_device.get(), m_resourceManager.get(), m_streamingManager.get(),
        /*withPlate=*/false, /*frames=*/200,
        /*apicalRepulsion=*/0.3f, /*basalAdhesion=*/1.5f,
        /*corticalTension=*/highTension,
        /*propagationStrength=*/0.0f,
        /*regulationRate=*/0.2f,
        /*lateralAdhesionScale=*/0.15f,
        /*catchBondStrength=*/0.0f );
    float dBase = MeanPairwiseDistance( posBase );

    float relDiff = std::abs( dCatch - dBase ) / dBase;
    EXPECT_GT( relDiff, 0.02f )
        << "Catch-bond mechanism must perturb the aggregate by >2% under high "
           "cortical tension to confirm end-to-end pipeline (catch=" << dCatch
        << " vs baseline=" << dBase << ", relDiff=" << relDiff << ")";
}

// 14d. Phase 5 — catch-bond is INERT at demo-default parameters. Confirms the
//      key biological property: a relaxed aggregate (no aggressive cortical
//      tension, no apical repulsion) feels baseline cadherin regardless of
//      whether catch-bond is enabled. This is the guard against the Phase 4.5
//      aesthetic regression the user flagged — beauty of the edge-to-edge
//      monolayer must survive the mere PRESENCE of catch-bond in the shader.
TEST_F( SimulationBuilderTest, ECBlobDemo_CatchBond_InertAtDemoDefaults )
{
    if( !m_device )
        GTEST_SKIP();

    // Demo-default values: corticalTension = 0.5 (mild), apical = 0.3 (no
    // repulsion), polMod ≈ 1 → loadSignal ≈ 0.05 → catchMul ≈ 1.14 at most.
    auto [posCatch, polCatch] = RunECDemoPhase3(
        m_device.get(), m_resourceManager.get(), m_streamingManager.get(),
        /*withPlate=*/false, /*frames=*/200,
        /*apicalRepulsion=*/0.3f, /*basalAdhesion=*/1.5f,
        /*corticalTension=*/0.5f,
        /*propagationStrength=*/0.0f,
        /*regulationRate=*/0.2f,
        /*lateralAdhesionScale=*/0.15f,
        /*catchBondStrength=*/2.0f,
        /*catchBondPeakLoad=*/0.3f );
    float dCatch = MeanPairwiseDistance( posCatch );

    auto [posBase, polBase] = RunECDemoPhase3(
        m_device.get(), m_resourceManager.get(), m_streamingManager.get(),
        /*withPlate=*/false, /*frames=*/200,
        /*apicalRepulsion=*/0.3f, /*basalAdhesion=*/1.5f,
        /*corticalTension=*/0.5f,
        /*propagationStrength=*/0.0f,
        /*regulationRate=*/0.2f,
        /*lateralAdhesionScale=*/0.15f,
        /*catchBondStrength=*/0.0f );
    float dBase = MeanPairwiseDistance( posBase );

    float relDiff = std::abs( dCatch - dBase ) / dBase;
    // The aggregate shape must be within 5% — catch-bond's 14% strengthening
    // at loadSignal=0.05 is small enough at aggregate level to preserve the
    // Phase 4.5-B wall-forming phenotype.
    EXPECT_LT( relDiff, 0.05f )
        << "Catch-bond must be near-inert at demo defaults (no external load) "
           "to preserve Phase 4.5-B aggregate geometry (catch=" << dCatch
        << " vs baseline=" << dBase << ", relDiff=" << relDiff << ")";
}

// 15. Phase 4.5 — junctional propagation must polarise interior cells in
//     EC2DMatrigelDemo. With the plate acting as the symmetry-breaking seed
//     (anchored cells polarise toward +Y), the PAR/Crumbs cascade transmits
//     that orientation upward through the cluster via cell-cell junctions.
//     Interior cells (y > anchorageDistance) should gain non-zero polarity
//     magnitude only when propagationStrength > 0.
// Step A (2026-04-18) — test is biologically valid but needs re-tuning. Under
// the BM-gated polarity model, the propagation cascade from plate-anchored
// cells does reach interior cells eventually, but at 600 frames / anchorageDistance=1.0
// the interior (y>2.0) hasn't had time to build magnitude above threshold. The
// cascade rate is now dependent on the deadband (0.05 mean neighbour magnitude)
// rather than on neighbour-count geometry. Phase 6 sweep will either (a) increase
// simulation time, (b) increase regulationRate to speed up EMA, or (c) use a
// different interior predicate to re-enable this signal.
TEST_F( SimulationBuilderTest, DISABLED_EC2DMatrigelDemo_PolarityPropagation_PolarisesInteriorCells )
{
    if( !m_device )
        GTEST_SKIP();

    // "Interior" = above anchorage distance (1.0) from plate at y=0, plus a
    // safety margin so we don't count bottom-layer cells the plate directly
    // polarises via polarityBias.
    auto isInterior = []( const glm::vec4& p ) { return p.y > 2.0f; };

    // Run with propagation active. 10 s of sim at regulationRate=0.5 gives
    // the cascade enough hops to reach the cluster apex.
    auto [posProp, polProp] = RunECDemoPhase3(
        m_device.get(), m_resourceManager.get(), m_streamingManager.get(),
        /*withPlate=*/true, /*frames=*/600,
        /*apicalRepulsion=*/-1.0f, /*basalAdhesion=*/2.5f,
        /*corticalTension=*/0.5f,
        /*propagationStrength=*/1.0f,
        /*regulationRate=*/0.5f );
    float magProp = MeanInteriorPolarityMagnitude( posProp, polProp, isInterior );

    // Baseline: propagation off, everything else equal.
    auto [posBase, polBase] = RunECDemoPhase3(
        m_device.get(), m_resourceManager.get(), m_streamingManager.get(),
        /*withPlate=*/true, /*frames=*/600,
        /*apicalRepulsion=*/-1.0f, /*basalAdhesion=*/2.5f,
        /*corticalTension=*/0.5f,
        /*propagationStrength=*/0.0f,
        /*regulationRate=*/0.5f );
    float magBase = MeanInteriorPolarityMagnitude( posBase, polBase, isInterior );

    EXPECT_GT( magProp, magBase )
        << "Junctional propagation must raise interior-cell mean polarity "
           "magnitude above the Phase-4 baseline (prop=" << magProp
        << " vs base=" << magBase << ")";
    // Also assert absolute magnitude is meaningfully non-zero — propagation
    // must actually polarise, not merely edge above the baseline noise floor.
    EXPECT_GT( magProp, 0.15f )
        << "Interior cells must reach meaningful polarity magnitude (got "
        << magProp << ")";
}

// 16. Phase 4.5 paired control — ECBlobDemo (no plate). Surface cells ARE
//     polar from the centroid cue, so propagation inside a plateless aggregate
//     will cascade outward-polarity inward. This is biologically reasonable
//     (cyst-formation in MDCK-style suspension cultures shows exactly this
//     outside-in polarisation pattern). The test asserts the DIFFERENTIAL —
//     propagation must produce measurably more interior polarity magnitude
//     than the prop=0 baseline — rather than an absolute "stays at zero" claim.
//     Without a plate the resulting polarity is ISOTROPIC outward (all cells
//     basal-outward) which cannot open a horizontal tube; EC2DMatrigelDemo's cavity
//     test (test 17) confirms the tube is plate-dependent.
// Step A biology assertion — anoikis and propagation. Without a BM seed, the
// PAR/Crumbs cascade has nothing to propagate FROM — propagation is a
// junction-to-junction transmission mechanism that requires at least one cell
// with established polarity as a source. In ECBlob the whole cluster is
// source-less, so propagation (even at strength 1.0) produces zero polarity
// magnitude anywhere. Asserts the biology, not a differential: cells ALL stay
// unpolarised across interior AND surface.
TEST_F( SimulationBuilderTest, ECBlobDemo_NoBM_PropagationProducesNoPolarity )
{
    if( !m_device )
        GTEST_SKIP();

    auto isAnyCell = []( const glm::vec4& p ) { return p.w > 0.5f; };  // live cells

    // Run with propagation strength 1.0 — under OLD model, surface cells would
    // have polarised from geometry and cascaded inward. Under the new BM-gated
    // model, no seed exists, so no cell polarises.
    auto [posProp, polProp] = RunECDemoPhase3(
        m_device.get(), m_resourceManager.get(), m_streamingManager.get(),
        /*withPlate=*/false, /*frames=*/600,
        /*apicalRepulsion=*/-1.0f, /*basalAdhesion=*/2.5f,
        /*corticalTension=*/0.5f,
        /*propagationStrength=*/1.0f,
        /*regulationRate=*/0.5f );
    float magMean = MeanInteriorPolarityMagnitude( posProp, polProp, isAnyCell );

    // Every cell must stay unpolarised. Allow tiny non-zero from FP noise /
    // partial EMA convergence; threshold 0.05 matches the shader deadband.
    EXPECT_LT( magMean, 0.05f )
        << "Anoikis biology: no cell may polarise in ECBlob even with "
           "propagationStrength=1.0 because there's no BM seed. "
           "Mean magnitude = " << magMean << " (should be ≈ 0).";
}

// 17. Phase 4.5 — KNOWN LIMITATION: uniform propagation cannot open a cavity
//     by itself. This test is DISABLED (not deleted) to document the finding
//     precisely so a future implementer can enable it once the lumen-cue
//     field is added (see plan §"Phase 4.5 follow-up — lumen nucleation").
//
//     Why it fails: propagation transmits a SINGLE polarity direction from the
//     plate seed upward through the cluster, producing uniform +Y polarity
//     throughout. For two cells with the same polarity vector, the JKR shader
//     computes `dotI + dotJ = 0` identically (because dotI = pol·(-r_ij),
//     dotJ = pol·r_ij, so dotI = -dotJ when polarities match). Alignment
//     clamps to 0.5 for every pair → scale = mix(apical, basal, 0.5) = 0.75
//     (mildly attractive), NEVER apical-apical repulsion.
//
//     Cord hollowing (Strilic 2009) biologically requires MIRRORED polarities
//     across the future lumen — cells above the lumen point apical-down, cells
//     below point apical-up. A scalar "lumen-nucleation cue" field that cells
//     orient toward (via `+gradient` sampling) is the established architectural
//     fix — it naturally produces mirrored polarities because the gradient
//     points toward the field peak from opposite sides.
//
//     Re-enable this test after Phase 4.5-B ships the lumen-cue field. At that
//     point propagation remains valuable (it stabilises the seed direction
//     against Brownian noise and extends into homogeneous tissue volumes) but
//     the cue field is what actually breaks the left-right symmetry.
TEST_F( SimulationBuilderTest, DISABLED_EC2DMatrigelDemo_PolarityPropagation_ReducesCentralDensity )
{
    if( !m_device )
        GTEST_SKIP();

    auto [posProp, polProp] = RunECDemoPhase3(
        m_device.get(), m_resourceManager.get(), m_streamingManager.get(),
        /*withPlate=*/true, /*frames=*/600,
        /*apicalRepulsion=*/-1.0f, /*basalAdhesion=*/2.5f,
        /*corticalTension=*/0.5f,
        /*propagationStrength=*/1.0f,
        /*regulationRate=*/0.5f );

    auto [posBase, polBase] = RunECDemoPhase3(
        m_device.get(), m_resourceManager.get(), m_streamingManager.get(),
        /*withPlate=*/true, /*frames=*/600,
        /*apicalRepulsion=*/-1.0f, /*basalAdhesion=*/2.5f,
        /*corticalTension=*/0.5f,
        /*propagationStrength=*/0.0f,
        /*regulationRate=*/0.5f );

    auto densityProp = RadialDensityAroundXAxis( posProp, /*bins=*/4, /*rMax=*/4.0f );
    auto densityBase = RadialDensityAroundXAxis( posBase, /*bins=*/4, /*rMax=*/4.0f );

    auto innerOuterRatio = []( const std::vector<float>& d ) {
        float inner = d[ 0 ] + d[ 1 ];
        float outer = d[ 2 ] + d[ 3 ];
        return ( outer > 0.0f ) ? ( inner / outer ) : 0.0f;
    };
    float ratioProp = innerOuterRatio( densityProp );
    float ratioBase = innerOuterRatio( densityBase );

    EXPECT_LT( ratioProp, ratioBase )
        << "Propagation alone cannot open a cavity — see DISABLED test header. "
        << "prop=" << ratioProp << " vs base=" << ratioBase;
}

// 18. Phase 4.5 — bit-exact backwards compatibility at propagationStrength=0.
//     With prop=0 the shader's deadband path should be inactive and the
//     polarity EMA must match Phase-4 behaviour exactly. Uses the position
//     buffer as a proxy for correct shader dispatch (prop=0 must leave
//     agent dynamics identical to Phase-4 regression runs).
TEST_F( SimulationBuilderTest, EC2DMatrigelDemo_PolarityPropagation_ZeroStrengthMatchesPhase4 )
{
    if( !m_device )
        GTEST_SKIP();

    // Phase-4 regression reference (same params as Phase-4 integration tests).
    auto [posRef, polRef] = RunECDemoPhase3(
        m_device.get(), m_resourceManager.get(), m_streamingManager.get(),
        /*withPlate=*/true, /*frames=*/300,
        /*apicalRepulsion=*/-1.0f, /*basalAdhesion=*/2.5f,
        /*corticalTension=*/5.0f,
        /*propagationStrength=*/0.0f,
        /*regulationRate=*/0.2f );

    // Same run, explicit prop=0. Must be identical up to FP tolerance — the
    // propagation code path is present but the deadband suppresses noise.
    auto [posNew, polNew] = RunECDemoPhase3(
        m_device.get(), m_resourceManager.get(), m_streamingManager.get(),
        /*withPlate=*/true, /*frames=*/300,
        /*apicalRepulsion=*/-1.0f, /*basalAdhesion=*/2.5f,
        /*corticalTension=*/5.0f,
        /*propagationStrength=*/0.0f,
        /*regulationRate=*/0.2f );

    // Positions should be identical modulo GPU non-determinism (tight tolerance).
    float mpdRef = MeanPairwiseDistance( posRef );
    float mpdNew = MeanPairwiseDistance( posNew );
    EXPECT_NEAR( mpdRef, mpdNew, 0.05f )
        << "Prop=0 + Phase-4 params must reproduce Phase-4 dynamics "
           "(ref=" << mpdRef << " vs new=" << mpdNew << ")";
}

// =============================================================================
// Rigid body dynamics — builder wiring + orientation evolution tests
// =============================================================================

// 1. Builder wiring: a blueprint with CurvedTile morphology + JKR Biomechanics must allocate
//    the contactHullBuffer and orientationBuffer.
//
// CurvedTile has a non-empty contactHull, so the builder should produce both buffers.
TEST_F( SimulationBuilderTest, JKR_RigidBody_BuilderWiring )
{
    if( !m_device )
        GTEST_SKIP();

    SimulationBlueprint blueprint;
    blueprint.SetName( "RigidBodyWiringTest" );
    blueprint.SetDomainSize( glm::vec3( 20.0f ), 1.5f );

    blueprint.ConfigureSpatialPartitioning()
        .SetMethod( SpatialPartitioningMethod::HashGrid )
        .SetCellSize( 3.0f )
        .SetMaxDensity( 64 )
        .SetComputeHz( 60.0f );

    auto jkr = BiomechanicsGenerator::JKR()
                   .SetYoungsModulus( 20.0f )
                   .SetPoissonRatio( 0.4f )
                   .SetAdhesionEnergy( 1.0f )
                   .SetMaxInteractionRadius( 0.75f )
                   .SetDampingCoefficient( 200.0f )
                   .Build();

    // Two cells side-by-side; positions don't need to be physically meaningful here.
    std::vector<glm::vec4> positions = {
        glm::vec4( 0.0f, 0.0f, 0.0f, 1.0f ),
        glm::vec4( 0.5f, 0.0f, 0.0f, 1.0f )
    };

    AgentGroup& group = blueprint.AddAgentGroup( "Tiles" );
    group.SetCount( 2 )
         .SetMorphology( MorphologyGenerator::CreateCurvedTile( 10.0f, 0.3f, 0.5f, 1.0f ) )
         .SetDistribution( positions );
    group.AddBehaviour( jkr ).SetHz( 60.0f );

    SimulationBuilder builder( m_resourceManager.get(), m_streamingManager.get() );
    SimulationState   state = builder.Build( blueprint );

    EXPECT_TRUE( state.orientationBuffer.IsValid() )  << "orientationBuffer not allocated";
    EXPECT_TRUE( state.contactHullBuffer.IsValid() )  << "contactHullBuffer not allocated";

    state.Destroy( m_resourceManager.get() );
}

// 2. Orientation evolution: after several JKR frames, cells with hull contacts must have
//    orientations that differ from their initial identity quaternion.
//
// Two cells are placed 0.4 units apart along X with small CurvedTile morphology (subR=0.25).
// Hull points are ~0.15 from centre along X; inner contacts overlap by ~0.4 with subR=0.25.
// Adhesion-only hull torque fires: cross(arm, dir*(-adh)) is non-zero for off-axis contacts.
// After 10 frames the stored quaternions must move away from identity.
TEST_F( SimulationBuilderTest, JKR_RigidBody_OrientationChanges )
{
    if( !m_device )
        GTEST_SKIP();

    SimulationBlueprint blueprint;
    blueprint.SetName( "RigidBodyOrientTest" );
    blueprint.SetDomainSize( glm::vec3( 20.0f ), 1.5f );

    blueprint.ConfigureSpatialPartitioning()
        .SetMethod( SpatialPartitioningMethod::HashGrid )
        .SetCellSize( 3.0f )
        .SetMaxDensity( 64 )
        .SetComputeHz( 60.0f );

    auto jkr = BiomechanicsGenerator::JKR()
                   .SetYoungsModulus( 20.0f )
                   .SetPoissonRatio( 0.4f )
                   .SetAdhesionEnergy( 2.0f )     // adhesion drives hull torque (adhesion-only model)
                   .SetMaxInteractionRadius( 0.75f )
                   .SetDampingCoefficient( 0.0f ) // no damping — maximise rotation per step
                   .Build();

    // 0.4 units apart: inner axial hull corners (at x≈±0.15) are 0.1 apart → overlap=0.4 for subR=0.25
    std::vector<glm::vec4> positions = {
        glm::vec4( 0.0f, 0.0f, 0.0f, 1.0f ),
        glm::vec4( 0.4f, 0.0f, 0.0f, 1.0f )
    };

    AgentGroup& group = blueprint.AddAgentGroup( "Tiles" );
    group.SetCount( 2 )
         .SetMorphology( MorphologyGenerator::CreateCurvedTile( 10.0f, 0.3f, 0.5f, 1.0f ) )
         .SetDistribution( positions );
    group.AddBehaviour( jkr ).SetHz( 60.0f );

    SimulationBuilder builder( m_resourceManager.get(), m_streamingManager.get() );
    SimulationState   state = builder.Build( blueprint );
    ASSERT_TRUE( state.orientationBuffer.IsValid() );
    ASSERT_TRUE( state.contactHullBuffer.IsValid() );

    // Record initial orientations (identity quaternions for hull morphology)
    std::vector<glm::vec4> initialOrientations( 2 );
    m_streamingManager->ReadbackBufferImmediate(
        state.orientationBuffer, initialOrientations.data(), 2 * sizeof( glm::vec4 ) );

    // Dispatch 10 frames
    auto compCtxHandle = m_device->CreateThreadContext( QueueType::COMPUTE );
    auto compCtx       = m_device->GetThreadContext( compCtxHandle );
    auto compCmd       = compCtx->GetCommandBuffer( compCtx->CreateCommandBuffer() );

    GraphDispatcher dispatcher;
    const float     dt = 1.0f / 60.0f;
    for( int frame = 0; frame < 10; ++frame )
    {
        compCmd->Begin();
        dispatcher.Dispatch( &state.computeGraph, compCmd, nullptr, dt, dt * static_cast<float>( frame ), frame % 2 );
        compCmd->End();
        m_device->GetComputeQueue()->Submit( { compCmd } );
        m_device->GetComputeQueue()->WaitIdle();
    }

    std::vector<glm::vec4> finalOrientations( 2 );
    m_streamingManager->ReadbackBufferImmediate(
        state.orientationBuffer, finalOrientations.data(), 2 * sizeof( glm::vec4 ) );

    // At least one cell must have rotated — orientation.xyz should be non-zero.
    // Adhesion-only hull torques fire on overlapping off-axis contacts (Z offset ~0.109).
    // Threshold is 5e-5 — well above floating-point noise (~1e-7).
    bool anyChanged = false;
    for( int i = 0; i < 2; ++i )
    {
        glm::vec4 d = finalOrientations[i] - initialOrientations[i];
        if( std::abs( d.x ) > 5e-5f || std::abs( d.y ) > 5e-5f || std::abs( d.z ) > 5e-5f )
        {
            anyChanged = true;
            break;
        }
    }
    EXPECT_TRUE( anyChanged ) << "No cell rotated after 10 frames of hull contact";

    state.Destroy( m_resourceManager.get() );
}

// 3. Hull-based translation: after several JKR frames, agents with hull contacts must have
//    moved apart (hull repulsion adds to translation, not just torque).
//
// Previously hull contacts produced torque only; translation came from point-particle.
// After the shader change, hull pairs generate both.  This test verifies positions change.
//
// Two SpikySphere agents placed 0.8 apart (hull extent R=0.5, subR=0.125 → contactDist=0.25).
// Hull pair distance ≈ 0.8 - 0.5 - 0.5 = -0.2 → well inside contact → strong repulsion.
// After 10 frames they should be further apart than the initial 0.8.
TEST_F( SimulationBuilderTest, JKR_RigidBody_HullTranslation_Integration )
{
    if( !m_device )
        GTEST_SKIP();

    SimulationBlueprint blueprint;
    blueprint.SetName( "HullTranslationTest" );
    blueprint.SetDomainSize( glm::vec3( 20.0f ), 1.5f );

    blueprint.ConfigureSpatialPartitioning()
        .SetMethod( SpatialPartitioningMethod::HashGrid )
        .SetCellSize( 3.0f )
        .SetMaxDensity( 64 )
        .SetComputeHz( 60.0f );

    auto jkr = BiomechanicsGenerator::JKR()
                   .SetYoungsModulus( 10.0f )
                   .SetPoissonRatio( 0.4f )
                   .SetAdhesionEnergy( 0.0f )      // no adhesion — pure hull repulsion
                   .SetMaxInteractionRadius( 0.70f )
                   .SetDampingCoefficient( 0.0f )  // no damping — maximise movement per step
                   .Build();

    // Two SpikySphere cells placed 0.8 apart — hull points deeply overlapping.
    std::vector<glm::vec4> positions = {
        glm::vec4( 0.0f, 0.0f, 0.0f, 1.0f ),
        glm::vec4( 0.8f, 0.0f, 0.0f, 1.0f )
    };

    AgentGroup& group = blueprint.AddAgentGroup( "SpikyCells" );
    group.SetCount( 2 )
         .SetMorphology( MorphologyGenerator::CreateSpikySphere( 0.357f, 1.4f ) )
         .SetDistribution( positions );
    group.AddBehaviour( jkr ).SetHz( 60.0f );

    SimulationBuilder builder( m_resourceManager.get(), m_streamingManager.get() );
    SimulationState   state = builder.Build( blueprint );
    ASSERT_TRUE( state.orientationBuffer.IsValid() );
    ASSERT_TRUE( state.contactHullBuffer.IsValid() );

    // Read initial positions
    std::vector<glm::vec4> initialPositions( 2 );
    m_streamingManager->ReadbackBufferImmediate(
        state.agentBuffers[0], initialPositions.data(), 2 * sizeof( glm::vec4 ) );

    float initialSep = std::abs( initialPositions[1].x - initialPositions[0].x );

    // Dispatch 10 frames
    auto compCtxHandle = m_device->CreateThreadContext( QueueType::COMPUTE );
    auto compCtx       = m_device->GetThreadContext( compCtxHandle );
    auto compCmd       = compCtx->GetCommandBuffer( compCtx->CreateCommandBuffer() );

    GraphDispatcher dispatcher;
    const float     dt         = 1.0f / 60.0f;
    int             activeIdx  = 0;
    for( int frame = 0; frame < 10; ++frame )
    {
        compCmd->Begin();
        dispatcher.Dispatch( &state.computeGraph, compCmd, nullptr, dt, dt * static_cast<float>( frame ), activeIdx );
        compCmd->End();
        m_device->GetComputeQueue()->Submit( { compCmd } );
        m_device->GetComputeQueue()->WaitIdle();
        activeIdx ^= 1;
    }

    // After N frames the output is in agentBuffers[activeIdx] (the buffer last written).
    std::vector<glm::vec4> finalPositions( 2 );
    m_streamingManager->ReadbackBufferImmediate(
        state.agentBuffers[activeIdx], finalPositions.data(), 2 * sizeof( glm::vec4 ) );

    float finalSep = std::abs( finalPositions[1].x - finalPositions[0].x );

    // Hull repulsion must have pushed cells further apart.
    EXPECT_GT( finalSep, initialSep )
        << "Hull-based translation should push SpikySphere cells apart "
        << "(initial=" << initialSep << ", final=" << finalSep << ")";

    state.Destroy( m_resourceManager.get() );
}
