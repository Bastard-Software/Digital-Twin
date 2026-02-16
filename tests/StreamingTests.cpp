#include "core/jobs/JobSystem.h"
#include "core/memory/MemorySystem.h"
#include "resources/ResourceManager.h"
#include "resources/StreamingManager.h"
#include "rhi/Device.h"
#include "rhi/Queue.h"
#include "rhi/RHI.h"
#include "rhi/Buffer.h"
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
    Scope<ResourceManager>  m_rm;
    Scope<StreamingManager> m_stream;

    void SetUp() override
    {
        m_memory = CreateScope<MemorySystem>();
        m_memory->Initialize();

        m_rhi = CreateScope<RHI>();
        RHIConfig config;
        config.headless         = true;
        config.enableValidation = true;
        m_rhi->Initialize( config );

        if( !m_rhi->GetAdapters().empty() )
        {
            m_rhi->CreateDevice( 0, m_device );
            m_rm = CreateScope<ResourceManager>( m_device.get(), m_memory.get() );
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