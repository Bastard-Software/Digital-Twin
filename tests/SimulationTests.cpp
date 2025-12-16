#include "rhi/RHI.hpp"
#include "runtime/Engine.hpp"
#include "simulation/SimulationContext.hpp"
#include "simulation/Types.hpp"
#include "resources/StreamingManager.hpp"
#include <gtest/gtest.h>

using namespace DigitalTwin;

class SimulationTests : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Manual RHI Init for Headless Environment
        if( RHI::IsInitialized() )
            RHI::Shutdown();

        RHIConfig config;
        config.headless         = true;
        config.enableValidation = true;
        RHI::Init( config );
    }

    void TearDown() override { RHI::Shutdown(); }
};

TEST_F( SimulationTests, UploadAndVerifyCellsAndCounter )
{
    if( RHI::GetAdapterCount() == 0 )
        GTEST_SKIP() << "No GPU adapter found";

    auto device   = RHI::CreateDevice( 0 );
    auto streamer = CreateRef<StreamingManager>( device );
    ASSERT_EQ( streamer->Init(), Result::SUCCESS );

    // 1. Create Context
    SimulationContext ctx( device );
    uint32_t          maxCount    = 100;
    uint32_t          activeCount = 10; // We will upload 10 cells
    ctx.Init( maxCount );

    ASSERT_NE( ctx.GetCellBuffer(), nullptr );
    ASSERT_NE( ctx.GetCounterBuffer(), nullptr );

    // 2. Prepare Data
    std::vector<Cell> cells( activeCount );
    for( uint32_t i = 0; i < activeCount; ++i )
    {
        cells[ i ].position = glm::vec4( ( float )i, 0.f, 0.f, 1.f );
        cells[ i ].color    = glm::vec4( 1.f );
    }

    // 3. Upload State (Cells + Counter)
    streamer->BeginFrame( 0 );
    ctx.UploadState( streamer.get(), cells );
    streamer->EndFrame();
    streamer->WaitForTransferComplete();

    // 4. Verify Cells (Readback)
    streamer->BeginFrame( 1 );

    // Read Cells
    auto cellAlloc = streamer->CaptureBuffer( ctx.GetCellBuffer(), activeCount * sizeof( Cell ) );

    // Read Atomic Counter (4 bytes)
    auto counterAlloc = streamer->CaptureBuffer( ctx.GetCounterBuffer(), sizeof( uint32_t ) );

    streamer->EndFrame();
    streamer->WaitForTransferComplete();

    // 5. Assertions
    // Check Cell Data
    Cell* data = ( Cell* )cellAlloc.mappedData;
    EXPECT_FLOAT_EQ( data[ 5 ].position.x, 5.0f );

    // Check Counter Data
    uint32_t* counterValue = ( uint32_t* )counterAlloc.mappedData;
    EXPECT_EQ( *counterValue, activeCount ) << "Atomic counter should match uploaded cell count";

    // Cleanup
    ctx.Shutdown();
    streamer->Shutdown();
    // Device is destroyed by Ref<>
}