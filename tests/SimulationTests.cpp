#include "simulation/MorphologyGenerator.h"
#include "simulation/SimulationBlueprint.h"
#include "simulation/SimulationBuilder.h"
#include "simulation/SpatialDistribution.h"
#include <gtest/gtest.h>

// Core and Engine dependencies needed for the Builder
#include "core/FileSystem.h"
#include "core/memory/MemorySystem.h"
#include "resources/ResourceManager.h"
#include "resources/StreamingManager.h"
#include "rhi/Device.h"
#include "rhi/RHI.h"

using namespace DigitalTwin;

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

// =================================================================================================
// 2. Simulation Builder & GPU Integration Tests
// =================================================================================================

class SimulationBuilderTest : public ::testing::Test
{
protected:
    Scope<MemorySystem>     m_memory;
    Scope<FileSystem>       m_fs;
    Scope<RHI>              m_rhi;
    Scope<Device>           m_device;
    Scope<ResourceManager>  m_resourceManager;
    Scope<StreamingManager> m_streamingManager;

    void SetUp() override
    {
        // 1. Core Systems
        m_memory = CreateScope<MemorySystem>();
        m_memory->Initialize();

        m_fs = CreateScope<FileSystem>( m_memory.get() );
        m_fs->Initialize( std::filesystem::current_path(), std::filesystem::current_path() );

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
        m_resourceManager = CreateScope<ResourceManager>( m_device.get(), m_memory.get(), m_fs.get() );
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
        if( m_fs )
            m_fs->Shutdown();
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