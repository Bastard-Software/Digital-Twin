#include "rhi/Device.hpp"
#include "rhi/RHI.hpp"
#include "streaming/StreamingManager.hpp"
#include <array>
#include <cstring>
#include <gtest/gtest.h>

using namespace DigitalTwin;

class StreamingTests : public ::testing::Test
{
protected:
    RHIConfig             config;
    Ref<Device>           device;
    Ref<StreamingManager> streamer;

    void SetUp() override
    {
        if( RHI::IsInitialized() )
            RHI::Shutdown();

        config.headless         = true;
        config.enableValidation = false;
        ASSERT_EQ( RHI::Init( config ), Result::SUCCESS );

        // We assume at least one adapter exists for tests to run
        if( RHI::GetAdapterCount() > 0 )
        {
            device = RHI::CreateDevice( 0 );
        }

        streamer = CreateRef<StreamingManager>( device );
        streamer->Init();
    }

    void TearDown() override
    {
        if( streamer )
        {
            streamer->Shutdown();
            streamer.reset();
        }

        if( device )
        {
            RHI::DestroyDevice( device );
            device.reset();
        }
        RHI::Shutdown();
    }
};

TEST_F( StreamingTests, UploadAndReadbackLoop )
{
    if( !device )
        GTEST_SKIP() << "Device not initialized";

    // 1. Create a Device Local buffer (Persistent Storage)
    BufferDesc bufDesc;
    bufDesc.size            = 1024;
    bufDesc.type            = BufferType::STORAGE;
    bufDesc.additionalUsage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

    auto gpuBuffer = device->CreateBuffer( bufDesc );
    ASSERT_NE( gpuBuffer, nullptr );

    // 2. Prepare Data
    std::array<float, 4> sendData = { 1.1f, 2.2f, 3.3f, 4.4f };
    uint64_t             frameNum = 0;

    // --- Begin Frame ---
    streamer->BeginFrame( frameNum );

    // Upload Data (CPU -> GPU)
    streamer->UploadToBuffer( gpuBuffer, sendData.data(), sizeof( sendData ) );

    // Capture Data (GPU -> CPU) - in the same batch for testing
    auto readToken = streamer->CaptureBuffer( gpuBuffer, sizeof( sendData ) );

    // Submit
    auto syncPoint = streamer->EndFrame();

    // --- Wait ---
    streamer->WaitForTransferComplete();

    // --- Verify ---
    float* readbackData = static_cast<float*>( readToken.mappedData );

    // Use near comparison for floats
    EXPECT_NEAR( readbackData[ 0 ], 1.1f, 0.0001f );
    EXPECT_NEAR( readbackData[ 3 ], 4.4f, 0.0001f );
}

TEST_F( StreamingTests, RingBufferCycling )
{
    if( !device )
        GTEST_SKIP() << "Device not initialized";

    // Test cycling more frames than the buffer size (FRAMES_IN_FLIGHT)
    // to ensure fences are waited on correctly.
    constexpr int TEST_FRAMES = 10;

    for( int i = 0; i < TEST_FRAMES; i++ )
    {
        streamer->BeginFrame( i );

        auto alloc = streamer->AllocateUpload( 128 );
        std::memset( alloc.mappedData, 0xFF, 128 ); // Write garbage

        streamer->EndFrame();
    }

    streamer->WaitForTransferComplete(); // Wait for the last frame
    SUCCEED();
}