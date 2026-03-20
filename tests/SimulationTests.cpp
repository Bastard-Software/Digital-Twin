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
    struct PhenotypeData
    {
        uint32_t lifecycleState;
        float    biomass;
        float    timer;
        uint32_t cellType;
    };
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
            DigitalTwin::Behaviours::SecreteField{ "VEGF", 10.0f, static_cast<int>( DigitalTwin::LifecycleState::Hypoxic ) } )
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

// NotchDll4: isolated agent has no Dll4 suppression from neighbors → Dll4 rises → becomes TipCell
TEST_F( SimulationBuilderTest, Behaviour_NotchDll4_IsolatedAgentBecomesTipCell )
{
    if( !m_device )
        GTEST_SKIP();

    SimulationBlueprint blueprint;
    blueprint.SetDomainSize( glm::vec3( 100.0f ), 1.0f )
        .ConfigureSpatialPartitioning()
        .SetCellSize( 30.0f )
        .SetComputeHz( 60.0f );

    // Single isolated agent — no neighbors to suppress its Dll4
    std::vector<glm::vec4> pos = { glm::vec4( 0.0f, 0.0f, 0.0f, 1.0f ) };

    blueprint.AddAgentGroup( "Endothelial" )
        .SetCount( 1 )
        .SetDistribution( pos )
        .AddBehaviour( Behaviours::NotchDll4{
            /* dll4ProductionRate   */ 1.0f,
            /* dll4DecayRate        */ 0.1f,
            /* notchInhibitionGain  */ 1.0f,
            /* vegfr2BaseExpression */ 1.0f,
            /* tipThreshold         */ 0.8f,
            /* stalkThreshold       */ 0.3f } )
        .SetHz( 60.0f );

    SimulationBuilder builder( m_resourceManager.get(), m_streamingManager.get() );
    SimulationState   state = builder.Build( blueprint );

    auto compCtxHandle = m_device->CreateThreadContext( QueueType::COMPUTE );
    auto compCtx       = m_device->GetThreadContext( compCtxHandle );
    auto compCmd       = compCtx->GetCommandBuffer( compCtx->CreateCommandBuffer() );

    GraphDispatcher dispatcher;
    compCmd->Begin();
    for( int i = 0; i < 100; ++i )
        dispatcher.Dispatch( &state.computeGraph, compCmd, nullptr, 1.0f / 60.0f, static_cast<float>( i ) / 60.0f, 0 );
    compCmd->End();

    m_device->GetComputeQueue()->Submit( { compCmd } );
    m_device->GetComputeQueue()->WaitIdle();

    struct PhenotypeData { uint32_t lifecycleState; float biomass; float timer; uint32_t cellType; };
    PhenotypeData result{};
    m_streamingManager->ReadbackBufferImmediate( state.phenotypeBuffer, &result, sizeof( PhenotypeData ) );

    EXPECT_EQ( result.cellType, 1u ) << "Isolated agent should become TipCell (cellType=1) with no Dll4 suppression";

    state.Destroy( m_resourceManager.get() );
}

// NotchDll4: two adjacent agents undergo lateral inhibition — one becomes TipCell, the other StalkCell
TEST_F( SimulationBuilderTest, Behaviour_NotchDll4_LateralInhibition )
{
    if( !m_device )
        GTEST_SKIP();

    SimulationBlueprint blueprint;
    blueprint.SetDomainSize( glm::vec3( 100.0f ), 1.0f )
        .ConfigureSpatialPartitioning()
        .SetCellSize( 30.0f )
        .SetComputeHz( 60.0f );

    // Two agents close together — within signaling radius (cellSize/2 = 15µm)
    std::vector<glm::vec4> pos = {
        glm::vec4( 0.0f, 0.0f, 0.0f, 1.0f ),
        glm::vec4( 5.0f, 0.0f, 0.0f, 1.0f )
    };

    blueprint.AddAgentGroup( "Endothelial" )
        .SetCount( 2 )
        .SetDistribution( pos )
        .AddBehaviour( Behaviours::NotchDll4{
            /* dll4ProductionRate   */ 1.0f,
            /* dll4DecayRate        */ 0.1f,
            /* notchInhibitionGain  */ 5.0f, // Strong suppression to drive differentiation
            /* vegfr2BaseExpression */ 1.0f,
            /* tipThreshold         */ 0.8f,
            /* stalkThreshold       */ 0.3f } )
        .SetHz( 60.0f );

    SimulationBuilder builder( m_resourceManager.get(), m_streamingManager.get() );
    SimulationState   state = builder.Build( blueprint );

    auto compCtxHandle = m_device->CreateThreadContext( QueueType::COMPUTE );
    auto compCtx       = m_device->GetThreadContext( compCtxHandle );
    auto compCmd       = compCtx->GetCommandBuffer( compCtx->CreateCommandBuffer() );

    GraphDispatcher dispatcher;
    compCmd->Begin();
    for( int i = 0; i < 200; ++i )
        dispatcher.Dispatch( &state.computeGraph, compCmd, nullptr, 1.0f / 60.0f, static_cast<float>( i ) / 60.0f, 0 );
    compCmd->End();

    m_device->GetComputeQueue()->Submit( { compCmd } );
    m_device->GetComputeQueue()->WaitIdle();

    struct PhenotypeData { uint32_t lifecycleState; float biomass; float timer; uint32_t cellType; };
    std::vector<PhenotypeData> results( 2 );
    m_streamingManager->ReadbackBufferImmediate( state.phenotypeBuffer, results.data(), 2 * sizeof( PhenotypeData ) );

    // Both agents start at dll4=0.5 (symmetric IC) — they mutually suppress each other equally.
    // Symmetric equilibrium with these params converges to dll4=1.0 (clamped), so both become TipCell.
    // This validates: signaling buffer is shared, neighbours are queried, cellType is updated.
    EXPECT_EQ( results[ 0 ].cellType, 1u ) << "Agent 0 should be TipCell after symmetric Notch-Dll4 convergence";
    EXPECT_EQ( results[ 1 ].cellType, 1u ) << "Agent 1 should be TipCell after symmetric Notch-Dll4 convergence";

    state.Destroy( m_resourceManager.get() );
}

// ===========================================================================================
// SpatialDistribution::VesselLine — CPU-only, no GPU fixture needed
// ===========================================================================================

TEST( SpatialDistribution_VesselLine, EvenSpacing )
{
    const uint32_t  count = 5;
    const glm::vec3 start( -10.0f, 5.0f, 0.0f );
    const glm::vec3 end( 10.0f, 5.0f, 0.0f );

    auto positions = SpatialDistribution::VesselLine( count, start, end );

    ASSERT_EQ( positions.size(), count );

    // First position should be start
    EXPECT_FLOAT_EQ( positions.front().x, start.x );
    EXPECT_FLOAT_EQ( positions.front().y, start.y );
    EXPECT_FLOAT_EQ( positions.front().z, start.z );

    // Last position should be end
    EXPECT_FLOAT_EQ( positions.back().x, end.x );
    EXPECT_FLOAT_EQ( positions.back().y, end.y );
    EXPECT_FLOAT_EQ( positions.back().z, end.z );

    // All w components should be 1.0
    for( const auto& p: positions )
        EXPECT_FLOAT_EQ( p.w, 1.0f );

    // Positions should be evenly spaced along the segment
    float expectedSpacing = glm::length( end - start ) / static_cast<float>( count - 1 );
    for( uint32_t i = 1; i < count; ++i )
    {
        float dist = glm::length( glm::vec3( positions[ i ] ) - glm::vec3( positions[ i - 1 ] ) );
        EXPECT_NEAR( dist, expectedSpacing, 1e-5f );
    }
}

TEST( SpatialDistribution_VesselLine, SingleAgent )
{
    const glm::vec3 start( -5.0f, 0.0f, 0.0f );
    const glm::vec3 end( 5.0f, 0.0f, 0.0f );

    auto positions = SpatialDistribution::VesselLine( 1, start, end );

    ASSERT_EQ( positions.size(), 1u );
    EXPECT_FLOAT_EQ( positions[ 0 ].x, 0.0f ); // midpoint
    EXPECT_FLOAT_EQ( positions[ 0 ].y, 0.0f );
    EXPECT_FLOAT_EQ( positions[ 0 ].z, 0.0f );
    EXPECT_FLOAT_EQ( positions[ 0 ].w, 1.0f );
}

TEST( SpatialDistribution_VesselLine, FixedSpacing )
{
    const glm::vec3 start( 0.0f, 0.0f, 0.0f );
    const glm::vec3 end( 10.0f, 0.0f, 0.0f );

    // Request 100 agents with spacing=3.0 — line length is 10, so only 4 fit (0,3,6,9)
    auto positions = SpatialDistribution::VesselLine( 100, start, end, 3.0f );

    ASSERT_EQ( positions.size(), 4u );
    EXPECT_FLOAT_EQ( positions[ 0 ].x, 0.0f );
    EXPECT_FLOAT_EQ( positions[ 1 ].x, 3.0f );
    EXPECT_FLOAT_EQ( positions[ 2 ].x, 6.0f );
    EXPECT_FLOAT_EQ( positions[ 3 ].x, 9.0f );
}