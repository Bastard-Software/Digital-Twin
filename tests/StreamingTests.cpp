#include "SetupHelpers.h"
#include "core/FileSystem.h"
#include "core/jobs/JobSystem.h"
#include "core/memory/MemorySystem.h"
#include "resources/ResourceManager.h"
#include "resources/StreamingManager.h"
#include "rhi/Buffer.h"
#include "rhi/Device.h"
#include "rhi/Queue.h"
#include "rhi/RHI.h"
#include "rhi/Texture.h"
#include <gtest/gtest.h>
#include <numeric>
#include <vector>

using namespace DigitalTwin;

class StreamingTest : public ::testing::Test
{
protected:
    Scope<RHI>              m_rhi;
    Scope<Device>           m_device;
    Scope<MemorySystem>     m_memory;
    Scope<FileSystem>       m_fileSystem;
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
        RHIConfig config;
        config.headless         = true;
        config.enableValidation = true;
        m_rhi->Initialize( config );

        if( !m_rhi->GetAdapters().empty() )
        {
            m_rhi->CreateDevice( 0, m_device );
            m_rm = CreateScope<ResourceManager>( m_device.get(), m_memory.get(), m_fileSystem.get() );
            m_rm->Initialize();
            m_stream = CreateScope<StreamingManager>( m_device.get(), m_rm.get() );
            m_stream->Initialize();
        }
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
        if( m_memory )
            m_memory->Shutdown();
    }
};

// 1. Tests immediate data upload to a buffer and its immediate readback to verify data integrity
TEST_F( StreamingTest, ImmediateUploadReadback )
{
    if( !m_device )
        GTEST_SKIP();

    const size_t     size = 1024;
    std::vector<int> srcData( size / sizeof( int ) );
    std::iota( srcData.begin(), srcData.end(), 0 );

    BufferDesc   desc{ size, BufferType::STORAGE };
    BufferHandle handle = m_rm->CreateBuffer( desc );

    m_stream->UploadBufferImmediate( handle, srcData.data(), size );

    std::vector<int> dstData( size / sizeof( int ) );
    m_stream->ReadbackBufferImmediate( handle, dstData.data(), size );

    EXPECT_EQ( srcData, dstData );
}

// 2. Verifies concurrent buffer upload operations across multiple threads using the JobSystem
TEST_F( StreamingTest, MultithreadedUpload )
{
    if( !m_device )
        GTEST_SKIP();

    JobSystem         jobs;
    JobSystem::Config jobConfig;
    jobConfig.workerCount = 4;
    jobs.Initialize( jobConfig );

    const int                     bufferCount = 10;
    const size_t                  size        = 256;
    std::vector<BufferHandle>     handles( bufferCount );
    std::vector<std::vector<int>> data( bufferCount );

    for( int i = 0; i < bufferCount; ++i )
    {
        handles[ i ] = m_rm->CreateBuffer( { size, BufferType::STORAGE } );
        data[ i ].resize( size / sizeof( int ) );
        std::fill( data[ i ].begin(), data[ i ].end(), i );
    }

    // Dispatch Uploads
    jobs.Dispatch( bufferCount, [ & ]( uint32_t i ) { m_stream->UploadBufferImmediate( handles[ i ], data[ i ].data(), size ); } );
    jobs.Wait();

    // Verify
    for( int i = 0; i < bufferCount; ++i )
    {
        std::vector<int> readback( size / sizeof( int ) );
        m_stream->ReadbackBufferImmediate( handles[ i ], readback.data(), size );
        EXPECT_EQ( data[ i ], readback );
    }

    jobs.Shutdown();
}

// 3. Tests immediate data upload to a single texture
TEST_F( StreamingTest, TextureUploadImmediate )
{
    if( !m_device )
        GTEST_SKIP();

    TextureDesc desc;
    desc.width  = 64;
    desc.height = 64;
    desc.format = VK_FORMAT_R8G8B8A8_UNORM;
    desc.usage  = TextureUsage::TRANSFER_DST | TextureUsage::SAMPLED;

    TextureHandle tex = m_rm->CreateTexture( desc );

    std::vector<uint32_t> data( 64 * 64, 0xFF0000FF ); // Red
    m_stream->UploadTextureImmediate( tex, data.data(), data.size() * 4 );

    m_rm->DestroyTexture( tex );
}

// 4. Tests deferred buffer readback by queuing the transfer and sync
TEST_F( StreamingTest, DeferredReadback )
{
    if( !m_device )
        GTEST_SKIP();

    // 1. Create Source Buffer
    const size_t size = 256;
    BufferHandle src  = m_rm->CreateBuffer( { size, BufferType::STORAGE } );

    std::vector<int> initData( size / sizeof( int ), 42 );
    m_stream->UploadBufferImmediate( src, initData.data(), size );

    // 2. Create Readback Buffer
    BufferHandle readback = m_rm->CreateBuffer( { size, BufferType::READBACK } );

    // 3. Queue Deferred Readback (Renamed method)
    m_stream->BeginFrame();
    m_stream->ReadbackBuffer( src, readback, size );
    uint64_t sync = m_stream->EndFrame();

    // 4. Wait for Transfer Queue using Device API
    VkSemaphore sem = m_device->GetTransferQueue()->GetTimelineSemaphore();
    m_device->WaitForSemaphores( { sem }, { sync } );

    // 5. Read
    std::vector<int> result( size / sizeof( int ) );
    m_rm->GetBuffer( readback )->Read( result.data(), size, 0 );

    EXPECT_EQ( initData, result );
}

// 5. Tests deferred texture readback by queuing the transfer and waiting for completion
TEST_F( StreamingTest, DeferredTextureReadback )
{
    if( !m_device )
        GTEST_SKIP();

    // 1. Create Texture
    TextureDesc desc;
    desc.width        = 32;
    desc.height       = 32;
    desc.format       = VK_FORMAT_R8G8B8A8_UNORM;
    desc.usage        = TextureUsage::TRANSFER_DST | TextureUsage::TRANSFER_SRC | TextureUsage::SAMPLED;
    TextureHandle tex = m_rm->CreateTexture( desc );

    const size_t          size = 32 * 32 * 4;
    std::vector<uint32_t> initData( 32 * 32, 0xAABBCCDD );

    // Upload immediately to populate data (leaves texture in TRANSFER_DST)
    m_stream->UploadTextureImmediate( tex, initData.data(), size );

    // 2. Create Readback Buffer
    BufferHandle readback = m_rm->CreateBuffer( { size, BufferType::READBACK } );

    // 3. Queue Deferred Readback
    m_stream->BeginFrame();
    m_stream->ReadbackTexture( tex, readback, size );
    uint64_t sync = m_stream->EndFrame();

    // 4. Wait for Transfer Queue using Device API
    VkSemaphore sem = m_device->GetTransferQueue()->GetTimelineSemaphore();
    m_device->WaitForSemaphores( { sem }, { sync } );

    // 5. Read and Verify
    std::vector<uint32_t> result( 32 * 32 );
    m_rm->GetBuffer( readback )->Read( result.data(), size, 0 );

    EXPECT_EQ( initData, result );
}

// 6. Verifies the immediate batched upload of multiple buffers in a single request list
TEST_F( StreamingTest, BatchedUploadImmediate )
{
    if( !m_device )
        GTEST_SKIP();

    const size_t size1 = 256;
    const size_t size2 = 512;

    std::vector<int> srcData1( size1 / sizeof( int ), 42 );
    std::vector<int> srcData2( size2 / sizeof( int ), 84 );

    // Optional: test the debug name assignment if you added it to BufferDesc
    BufferDesc desc1{ size1, BufferType::STORAGE, "BatchDst1" };
    BufferDesc desc2{ size2, BufferType::STORAGE, "BatchDst2" };

    BufferHandle dst1 = m_rm->CreateBuffer( desc1 );
    BufferHandle dst2 = m_rm->CreateBuffer( desc2 );

    // Build the request list
    std::vector<BufferUploadRequest> requests = { { dst1, srcData1.data(), size1, 0 }, { dst2, srcData2.data(), size2, 0 } };

    // Execute Batched Upload
    m_stream->UploadBufferImmediate( requests );

    // Verify via individual readbacks
    std::vector<int> readback1( size1 / sizeof( int ) );
    std::vector<int> readback2( size2 / sizeof( int ) );

    m_stream->ReadbackBufferImmediate( dst1, readback1.data(), size1 );
    m_stream->ReadbackBufferImmediate( dst2, readback2.data(), size2 );

    EXPECT_EQ( srcData1, readback1 );
    EXPECT_EQ( srcData2, readback2 );

    m_rm->DestroyBuffer( dst1 );
    m_rm->DestroyBuffer( dst2 );
}

// 7. Verifies the immediate batched readback of multiple buffers simultaneously
TEST_F( StreamingTest, BatchedReadbackImmediate )
{
    if( !m_device )
        GTEST_SKIP();

    const size_t sizeA = 128;
    const size_t sizeB = 1024;

    std::vector<int> initDataA( sizeA / sizeof( int ), 7 );
    std::vector<int> initDataB( sizeB / sizeof( int ), 9 );

    BufferHandle srcA = m_rm->CreateBuffer( { sizeA, BufferType::STORAGE } );
    BufferHandle srcB = m_rm->CreateBuffer( { sizeB, BufferType::STORAGE } );

    // Setup initial data on GPU
    std::vector<BufferUploadRequest> uploads = { { srcA, initDataA.data(), sizeA, 0 }, { srcB, initDataB.data(), sizeB, 0 } };
    m_stream->UploadBufferImmediate( uploads );

    // Prepare readback destinations
    std::vector<int> readDataA( sizeA / sizeof( int ), 0 );
    std::vector<int> readDataB( sizeB / sizeof( int ), 0 );

    // Build Batched Readback Request
    std::vector<BufferReadbackRequest> readbacks = { { srcA, readDataA.data(), sizeA, 0 }, { srcB, readDataB.data(), sizeB, 0 } };

    // Execute Batched Readback
    m_stream->ReadbackBufferImmediate( readbacks );

    EXPECT_EQ( initDataA, readDataA );
    EXPECT_EQ( initDataB, readDataB );

    m_rm->DestroyBuffer( srcA );
    m_rm->DestroyBuffer( srcB );
}

// 8. Tests immediate data readback from a single texture directly to CPU memory
TEST_F( StreamingTest, SingleTextureUploadAndReadbackImmediate )
{
    if( !m_device )
        GTEST_SKIP();

    // Setup dimensions for a standard 2D slice (e.g., biological field data)
    const uint32_t width     = 64;
    const uint32_t height    = 64;
    const size_t   numPixels = width * height;
    const size_t   dataSize  = numPixels * sizeof( uint32_t );

    // Initialize mock data (e.g., specific biological marker concentrations)
    std::vector<uint32_t> uploadData( numPixels, 0x1A2B3C4D );
    std::vector<uint32_t> readbackData( numPixels, 0 );

    TextureDesc desc = {};
    desc.width       = width;
    desc.height      = height;
    desc.depth       = 1;
    desc.format      = VK_FORMAT_R8G8B8A8_UNORM;
    desc.usage       = TextureUsage::TRANSFER_SRC | TextureUsage::TRANSFER_DST | TextureUsage::SAMPLED;

    TextureHandle texture = m_rm->CreateTexture( desc );

    // 1. Upload the data immediately
    m_stream->UploadTextureImmediate( texture, uploadData.data(), dataSize );

    // 2. Readback the data immediately
    m_stream->ReadbackTextureImmediate( texture, readbackData.data(), dataSize );

    // 3. Verify data integrity
    EXPECT_EQ( uploadData, readbackData ) << "Texture readback data does not match the uploaded data!";

    // Cleanup
    m_rm->DestroyTexture( texture );
}

// 9. Verifies the immediate batched upload and readback of multiple textures simultaneously
TEST_F( StreamingTest, BatchedTextureUploadAndReadbackImmediate )
{
    if( !m_device )
        GTEST_SKIP();

    // Create asymmetrical dimensions to ensure pitch/strides are handled correctly internally
    const uint32_t widthA = 32, heightA = 32;
    const uint32_t widthB = 128, heightB = 64;

    const size_t sizeA = widthA * heightA * sizeof( uint32_t );
    const size_t sizeB = widthB * heightB * sizeof( uint32_t );

    std::vector<uint32_t> uploadDataA( widthA * heightA, 0xAAAAAAAA );
    std::vector<uint32_t> uploadDataB( widthB * heightB, 0xBBBBBBBB );

    std::vector<uint32_t> readbackDataA( widthA * heightA, 0 );
    std::vector<uint32_t> readbackDataB( widthB * heightB, 0 );

    TextureDesc descA = {};
    descA.width       = widthA;
    descA.height      = heightA;
    descA.depth       = 1;
    descA.format      = VK_FORMAT_R8G8B8A8_UNORM;
    descA.usage       = TextureUsage::TRANSFER_SRC | TextureUsage::TRANSFER_DST | TextureUsage::SAMPLED;

    TextureDesc descB = descA;
    descB.width       = widthB;
    descB.height      = heightB;

    TextureHandle texA = m_rm->CreateTexture( descA );
    TextureHandle texB = m_rm->CreateTexture( descB );

    // 1. Batched Upload
    std::vector<TextureUploadRequest> uploads = { { texA, uploadDataA.data(), sizeA }, { texB, uploadDataB.data(), sizeB } };
    m_stream->UploadTextureImmediate( uploads );

    // 2. Batched Readback
    std::vector<TextureReadbackRequest> readbacks = { { texA, readbackDataA.data(), sizeA }, { texB, readbackDataB.data(), sizeB } };
    m_stream->ReadbackTextureImmediate( readbacks );

    // 3. Verify memory
    EXPECT_EQ( uploadDataA, readbackDataA ) << "Batched readback failed for Texture A";
    EXPECT_EQ( uploadDataB, readbackDataB ) << "Batched readback failed for Texture B";

    // Cleanup
    m_rm->DestroyTexture( texA );
    m_rm->DestroyTexture( texB );
}