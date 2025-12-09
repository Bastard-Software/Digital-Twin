#include "StreamingManager.hpp"

#include "core/Log.hpp"

namespace DigitalTwin
{
    // Default heap sizes (adjust based on simulation scale)
    constexpr VkDeviceSize UPLOAD_HEAP_SIZE   = 64 * 1024 * 1024; // 64MB
    constexpr VkDeviceSize READBACK_HEAP_SIZE = 32 * 1024 * 1024; // 32MB

    StreamingManager::StreamingManager( Ref<Device> device )
        : m_device( device )
    {
    }

    StreamingManager::~StreamingManager()
    {
        Shutdown();
    }

    Result StreamingManager::Init()
    {
        DT_CORE_INFO( "[Streaming] Initializing StreamingManager..." );

        for( uint32_t i = 0; i < FRAMES_IN_FLIGHT; i++ )
        {
            auto& f                 = m_frames[ i ];
            m_frameFenceValues[ i ] = 0;

            // 1. Create Upload Heap (CPU Write -> GPU Read)
            BufferDesc uploadDesc{};
            uploadDesc.size = UPLOAD_HEAP_SIZE;
            uploadDesc.type = BufferType::UPLOAD;
            // Usage: Transfer Source (for staging), Uniforms, Storage, Device Address
            uploadDesc.additionalUsage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                         VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;

            f.uploadHeap = m_device->CreateBuffer( uploadDesc );
            if( !f.uploadHeap )
                return Result::FAIL;
            f.uploadMappedPtr = f.uploadHeap->Map();

            // 2. Create Readback Heap (GPU Write -> CPU Read)
            BufferDesc readbackDesc{};
            readbackDesc.size            = READBACK_HEAP_SIZE;
            readbackDesc.type            = BufferType::READBACK;
            readbackDesc.additionalUsage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;

            f.readbackHeap = m_device->CreateBuffer( readbackDesc );
            if( !f.readbackHeap )
                return Result::FAIL;
            f.readbackMappedPtr = f.readbackHeap->Map();

            // 3. Create Transfer Command Buffer
            f.transferCmd = m_device->CreateCommandBuffer( QueueType::TRANSFER );
        }

        return Result::SUCCESS;
    }

    void StreamingManager::Shutdown()
    {
        if( m_device )
        {
            // Wait for all operations to finish before destroying resources
            m_device->GetAPI().vkDeviceWaitIdle( m_device->GetHandle() );

            for( uint32_t i = 0; i < FRAMES_IN_FLIGHT; i++ )
            {
                auto& f = m_frames[ i ];
                f.uploadHeap.reset();
                f.readbackHeap.reset();
                f.transferCmd.reset();
            }
        }
    }

    void StreamingManager::BeginFrame( uint64_t frameNumber )
    {
        m_currentFrameNumber = frameNumber;
        m_frameIndex         = frameNumber % FRAMES_IN_FLIGHT;
        auto& f              = m_frames[ m_frameIndex ];

        // 1. Synchronization: Check if we are overwriting a slot that is still in use by GPU.
        // We look at the fence value stored the last time we used this index.
        uint64_t waitValue = m_frameFenceValues[ m_frameIndex ];

        if( waitValue > 0 )
        {
            auto transferQueue = m_device->GetTransferQueue();

            // Optimization: Avoid heavy syscall if already completed
            if( !transferQueue->IsValueCompleted( waitValue ) )
            {
                // Wait on CPU for the specific value on the Transfer Queue
                Result res = m_device->WaitForQueue( transferQueue, waitValue );
                if( res != Result::SUCCESS )
                {
                    DT_CORE_ERROR( "[Streaming] Timeout waiting for Transfer Queue!" );
                }
            }
        }

        // 2. Reset Linear Allocators
        f.uploadOffset   = 0;
        f.readbackOffset = 0;

        // 3. Begin Recording Transfer Commands
        f.transferCmd->Begin( VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT );
    }

    SyncPoint StreamingManager::EndFrame()
    {
        auto& f = m_frames[ m_frameIndex ];
        f.transferCmd->End();

        uint64_t signalValue   = 0;
        auto     transferQueue = m_device->GetTransferQueue();

        // 1. Submit to Queue
        // The Queue class handles the timeline semaphore signal internally.
        // We get back the 'signalValue' that indicates when this batch is done.
        Result res = transferQueue->Submit( f.transferCmd->GetHandle(), signalValue );

        if( res != Result::SUCCESS )
        {
            DT_CORE_ERROR( "[Streaming] Failed to submit transfer batch!" );
        }

        // 2. Store the value to wait for when this slot cycles back
        m_frameFenceValues[ m_frameIndex ] = signalValue;

        // 3. Return the SyncPoint so other queues can depend on this transfer
        return { transferQueue->GetTimelineSemaphore(), signalValue };
    }

    void StreamingManager::WaitForTransferComplete()
    {
        // Wait for the operations of the CURRENT frame
        uint64_t waitValue = m_frameFenceValues[ m_frameIndex ];
        if( waitValue > 0 )
        {
            m_device->WaitForQueue( m_device->GetTransferQueue(), waitValue );
        }

        auto& f = m_frames[ m_frameIndex ];
        if( f.readbackHeap )
        {
            f.readbackHeap->Invalidate( f.readbackOffset, 0 );
        }
    }

    TransientAllocation StreamingManager::AllocateUpload( VkDeviceSize size, VkDeviceSize alignment )
    {
        auto& f = m_frames[ m_frameIndex ];

        // Align the offset
        VkDeviceSize alignedOffset = ( f.uploadOffset + alignment - 1 ) & ~( alignment - 1 );

        // Check for overflow
        if( alignedOffset + size > f.uploadHeap->GetSize() )
        {
            DT_CORE_ERROR( "[Streaming] Upload Heap OOM! Frame: {}", m_currentFrameNumber );
            DT_DEBUGBREAK();
            return {};
        }

        // Advance offset
        f.uploadOffset = alignedOffset + size;

        // Create allocation info
        TransientAllocation alloc;
        alloc.buffer        = f.uploadHeap->GetHandle();
        alloc.offset        = alignedOffset;
        alloc.size          = size;
        alloc.mappedData    = static_cast<uint8_t*>( f.uploadMappedPtr ) + alignedOffset;
        alloc.deviceAddress = f.uploadHeap->GetDeviceAddress() + alignedOffset;

        return alloc;
    }

    void StreamingManager::UploadToBuffer( Ref<Buffer> dstBuffer, const void* data, VkDeviceSize size, VkDeviceSize dstOffset )
    {
        // 1. Allocate in staging (Upload Heap)
        auto staging = AllocateUpload( size, 4 );
        if( !staging.mappedData )
            return;

        // 2. Copy to mapped memory
        memcpy( staging.mappedData, data, size );

        // 3. Record copy command
        auto&        f = m_frames[ m_frameIndex ];
        VkBufferCopy region{};
        region.srcOffset = staging.offset;
        region.dstOffset = dstOffset;
        region.size      = size;

        f.transferCmd->CopyBuffer( f.uploadHeap, dstBuffer, region );
    }

    TransientAllocation StreamingManager::CaptureBuffer( Ref<Buffer> srcBuffer, VkDeviceSize size, VkDeviceSize srcOffset )
    {
        auto& f = m_frames[ m_frameIndex ];

        // Align offset (Readback often requires alignment)
        VkDeviceSize alignedOffset = ( f.readbackOffset + 256 - 1 ) & ~( 256 - 1 );

        if( alignedOffset + size > f.readbackHeap->GetSize() )
        {
            DT_CORE_ERROR( "[Streaming] Readback Heap OOM! Frame: {}", m_currentFrameNumber );
            return {};
        }

        f.readbackOffset = alignedOffset + size;

        // We must ensure that any previous writes to srcBuffer (e.g. from UploadToBuffer)
        // are visible to the transfer read stage before we copy from it.
        VkBufferMemoryBarrier barrier = { VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER };
        barrier.srcAccessMask         = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask         = VK_ACCESS_TRANSFER_READ_BIT;
        barrier.srcQueueFamilyIndex   = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex   = VK_QUEUE_FAMILY_IGNORED;
        barrier.buffer                = srcBuffer->GetHandle();
        barrier.offset                = 0; // Protect whole buffer for simplicity
        barrier.size                  = VK_WHOLE_SIZE;

        f.transferCmd->PipelineBarrier( VK_PIPELINE_STAGE_TRANSFER_BIT, // srcStage
                                        VK_PIPELINE_STAGE_TRANSFER_BIT, // dstStage
                                        0,                              // dependencyFlags
                                        {},                             // memoryBarriers
                                        { barrier },                    // bufferBarriers
                                        {}                              // imageBarriers
        );

        // Record copy command (Device -> Readback Heap)
        VkBufferCopy region{};
        region.srcOffset = srcOffset;
        region.dstOffset = alignedOffset;
        region.size      = size;

        f.transferCmd->CopyBuffer( srcBuffer, f.readbackHeap, region );

        // Prepare result
        TransientAllocation alloc;
        alloc.buffer     = f.readbackHeap->GetHandle();
        alloc.offset     = alignedOffset;
        alloc.size       = size;
        alloc.mappedData = static_cast<uint8_t*>( f.readbackMappedPtr ) + alignedOffset;

        return alloc;
    }
} // namespace DigitalTwin