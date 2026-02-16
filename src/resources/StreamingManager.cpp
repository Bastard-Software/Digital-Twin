#include "resources/StreamingManager.h"

#include "core/Log.h"
#include "resources/ResourceManager.h"
#include "rhi/Buffer.h"
#include "rhi/CommandBuffer.h"
#include "rhi/Device.h"
#include "rhi/Queue.h"
#include "rhi/Texture.h"
#include "rhi/ThreadContext.h"
#include <thread>

namespace DigitalTwin
{

    static size_t GetThreadHash()
    {
        return std::hash<std::thread::id>{}( std::this_thread::get_id() );
    }

    StreamingManager::StreamingManager( Device* device, ResourceManager* resourceManager )
        : m_device( device )
        , m_resourceManager( resourceManager )
    {
    }

    StreamingManager::~StreamingManager()
    {
    }

    Result StreamingManager::Initialize()
    {
        DT_INFO( "[StreamingManager] Initializing..." );

        m_mainTransferCtxHandle = m_device->CreateThreadContext( QueueType::TRANSFER );
        m_mainTransferCtx       = m_device->GetThreadContext( m_mainTransferCtxHandle );

        if( !m_mainTransferCtx )
        {
            DT_ERROR( "Failed to create Transfer ThreadContext" );
            return Result::FAIL;
        }

        return Result::SUCCESS;
    }

    void StreamingManager::Shutdown()
    {
        if( m_device && m_device->GetTransferQueue() )
        {
            m_device->GetTransferQueue()->WaitIdle();
        }

        // Cleanup pending staging buffers
        {
            std::lock_guard<std::mutex> lock( m_queueMutex );
            for( const auto& req: m_pendingTransfers )
            {
                m_resourceManager->DestroyBuffer( req.stagingHandle );
            }
            m_pendingTransfers.clear();
        }

        m_mainTransferCtx = nullptr;
    }

    void StreamingManager::BeginFrame()
    {
        if( m_mainTransferCtx )
        {
            m_mainTransferCtx->Reset();
            m_currentCmdHandle = m_mainTransferCtx->CreateCommandBuffer();
            m_currentCmd       = m_mainTransferCtx->GetCommandBuffer( m_currentCmdHandle );

            if( m_currentCmd )
            {
                m_currentCmd->Begin();
            }
        }
    }

    uint64_t StreamingManager::EndFrame()
    {
        if( !m_currentCmd )
            return 0;

        bool                         hasWork = false;
        std::vector<TransferRequest> transfers;

        // 1. Fetch Queue
        {
            std::lock_guard<std::mutex> lock( m_queueMutex );
            if( !m_pendingTransfers.empty() )
            {
                transfers.swap( m_pendingTransfers );
                hasWork = true;
            }
        }

        // 2. Record
        if( hasWork )
        {
            for( const auto& req: transfers )
            {
                if( req.type == TransferType::UploadBuffer )
                {
                    Buffer* src = m_resourceManager->GetBuffer( req.stagingHandle );
                    Buffer* dst = m_resourceManager->GetBuffer( req.targetBuffer );
                    if( src && dst )
                    {
                        m_currentCmd->CopyBuffer( src, dst, req.size, 0, req.bufferOffset );
                    }
                }
                else if( req.type == TransferType::ReadbackBuffer )
                {
                    Buffer* src = m_resourceManager->GetBuffer( req.targetBuffer );
                    Buffer* dst = m_resourceManager->GetBuffer( req.stagingHandle ); // Staging is destination here
                    if( src && dst )
                    {
                        m_currentCmd->CopyBuffer( src, dst, req.size, req.bufferOffset, 0 );
                    }
                }
                else if( req.type == TransferType::UploadTexture )
                {
                    Buffer*  src = m_resourceManager->GetBuffer( req.stagingHandle );
                    Texture* dst = m_resourceManager->GetTexture( req.targetTexture );
                    if( src && dst )
                    {
                        // Transition Undefined -> Transfer Dst
                        VkImageMemoryBarrier2 barrier = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
                        barrier.srcStageMask          = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
                        barrier.srcAccessMask         = 0;
                        barrier.dstStageMask          = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
                        barrier.dstAccessMask         = VK_ACCESS_2_TRANSFER_WRITE_BIT;
                        barrier.oldLayout             = VK_IMAGE_LAYOUT_UNDEFINED;
                        barrier.newLayout             = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
                        barrier.image                 = dst->GetHandle();
                        barrier.subresourceRange      = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
                        m_currentCmd->PipelineBarrier( 0, 0, 0, 0, nullptr, 0, nullptr, 1, &barrier );

                        VkBufferImageCopy region{};
                        region.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
                        region.imageExtent      = dst->GetExtent();

                        vkCmdCopyBufferToImage( m_currentCmd->GetHandle(), src->GetHandle(), dst->GetHandle(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                                1, &region );
                    }
                }
                else if( req.type == TransferType::ReadbackTexture )
                {
                    Buffer*  dst = m_resourceManager->GetBuffer( req.stagingHandle );
                    Texture* src = m_resourceManager->GetTexture( req.targetTexture );
                    if( src && dst )
                    {

                        // NOTE: We assume the texture was recently uploaded or is in a compatible state.
                        // Ideally we should track state. Here we assume TRANSFER_DST (if just uploaded)
                        // or SHADER_READ_ONLY. We transition from DST to SRC for safety in tests.
                        // WARNING: oldLayout=UNDEFINED invalidates content! We must guess a safe layout or track it.
                        // For this implementation (streaming manager), we assume we own it and it's likely TRANSFER_DST_OPTIMAL.

                        VkImageMemoryBarrier2 barrier = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
                        barrier.srcStageMask          = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT; // Wait for anything
                        barrier.srcAccessMask         = VK_ACCESS_2_MEMORY_WRITE_BIT;
                        barrier.dstStageMask          = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
                        barrier.dstAccessMask         = VK_ACCESS_2_TRANSFER_READ_BIT;
                        // Use TRANSFER_DST_OPTIMAL as oldLayout since this is mostly used after Upload.
                        // If used after rendering, this might be invalid without state tracking.
                        barrier.oldLayout        = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
                        barrier.newLayout        = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
                        barrier.image            = src->GetHandle();
                        barrier.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
                        m_currentCmd->PipelineBarrier( 0, 0, 0, 0, nullptr, 0, nullptr, 1, &barrier );

                        VkBufferImageCopy region{};
                        region.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
                        region.imageExtent      = src->GetExtent();

                        vkCmdCopyImageToBuffer( m_currentCmd->GetHandle(), src->GetHandle(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dst->GetHandle(),
                                                1, &region );
                    }
                }
            }

            DT_INFO( "[StreamingManager] EndFrame: Submitted {} transfers.", transfers.size() );
        }

        m_currentCmd->End();

        uint64_t signalValue = 0;
        if( hasWork )
        {
            Queue* queue = m_device->GetTransferQueue();
            if( queue )
            {
                queue->Submit( { m_currentCmd } );
                signalValue = queue->GetLastSubmittedValue();
            }
        }

        // Cleanup upload staging buffers (Readback buffers are owned by user)
        for( const auto& req: transfers )
        {
            if( req.type == TransferType::UploadBuffer || req.type == TransferType::UploadTexture )
            {
                m_resourceManager->DestroyBuffer( req.stagingHandle );
            }
        }

        return signalValue;
    }

    void StreamingManager::UploadBuffer( BufferHandle dstHandle, const void* data, size_t size, size_t dstOffset )
    {
        if( !data || size == 0 )
            return;

        BufferDesc   stageDesc{ size, BufferType::UPLOAD };
        BufferHandle stageHandle = m_resourceManager->CreateBuffer( stageDesc );
        Buffer*      stage       = m_resourceManager->GetBuffer( stageHandle );

        if( !stage )
        {
            DT_ERROR( "StreamingManager: Failed to allocate staging buffer." );
            return;
        }

        stage->Write( data, size, 0 );

        std::lock_guard<std::mutex> lock( m_queueMutex );
        TransferRequest             req;
        req.type          = TransferType::UploadBuffer;
        req.stagingHandle = stageHandle;
        req.targetBuffer  = dstHandle;
        req.size          = size;
        req.bufferOffset  = dstOffset;
        m_pendingTransfers.push_back( req );

        DT_INFO( "[StreamingManager] Queued Buffer Upload: {} bytes (Thread: {})", size, GetThreadHash() );
    }

    void StreamingManager::UploadTexture( TextureHandle dstTexture, const void* data, size_t size )
    {
        if( !data || size == 0 )
            return;

        BufferDesc   stageDesc{ size, BufferType::UPLOAD };
        BufferHandle stageHandle = m_resourceManager->CreateBuffer( stageDesc );
        Buffer*      stage       = m_resourceManager->GetBuffer( stageHandle );

        if( stage )
        {
            stage->Write( data, size, 0 );

            std::lock_guard<std::mutex> lock( m_queueMutex );
            TransferRequest             req;
            req.type          = TransferType::UploadTexture;
            req.stagingHandle = stageHandle;
            req.targetTexture = dstTexture;
            req.size          = size;
            m_pendingTransfers.push_back( req );
        }

        DT_INFO( "[StreamingManager] Queued TExture Upload: {} bytes (Thread: {})", size, GetThreadHash() );
    }

    void StreamingManager::ReadbackBuffer( BufferHandle srcBuffer, BufferHandle dstReadbackBuffer, size_t size, size_t srcOffset )
    {
        std::lock_guard<std::mutex> lock( m_queueMutex );
        TransferRequest             req;
        req.type          = TransferType::ReadbackBuffer;
        req.stagingHandle = dstReadbackBuffer; // Destination
        req.targetBuffer  = srcBuffer;         // Source
        req.size          = size;
        req.bufferOffset  = srcOffset;
        m_pendingTransfers.push_back( req );

        DT_INFO( "[StreamingManager] Queued Buffer Readback: {} bytes (Thread: {})", size, GetThreadHash() );
    }

    void StreamingManager::ReadbackTexture( TextureHandle srcTexture, BufferHandle dstReadbackBuffer, size_t size )
    {
        std::lock_guard<std::mutex> lock( m_queueMutex );
        TransferRequest             req;
        req.type          = TransferType::ReadbackTexture;
        req.stagingHandle = dstReadbackBuffer; // Destination
        req.targetTexture = srcTexture;        // Source
        req.size          = size;
        m_pendingTransfers.push_back( req );

        DT_INFO( "[StreamingManager] Queued Texture Readback: {} bytes (Thread: {})", size, GetThreadHash() );
    }

    void StreamingManager::ExecuteImmediateTransfer( std::function<void( CommandBuffer* )> recordCallback )
    {
        // Transient context for immediate execution on any thread
        ThreadContextHandle tempCtxHandle = m_device->CreateThreadContext( QueueType::TRANSFER );
        ThreadContext*      tempCtx       = m_device->GetThreadContext( tempCtxHandle );

        if( tempCtx )
        {
            CommandBufferHandle cmdHandle = tempCtx->CreateCommandBuffer();
            CommandBuffer*      cmd       = tempCtx->GetCommandBuffer( cmdHandle );

            cmd->Begin();
            recordCallback( cmd );
            cmd->End();

            Queue* queue = m_device->GetTransferQueue();
            queue->Submit( { cmd } );
            queue->WaitIdle(); // Blocking
        }
        // Device owns context, will be cleaned up on Shutdown (or we need DestroyThreadContext API)
    }

    void StreamingManager::UploadBufferImmediate( BufferHandle dstHandle, const void* data, size_t size, size_t dstOffset )
    {
        Buffer* dst = m_resourceManager->GetBuffer( dstHandle );
        if( !dst )
            return;

        BufferDesc   stageDesc{ size, BufferType::UPLOAD };
        BufferHandle stageHandle = m_resourceManager->CreateBuffer( stageDesc );
        Buffer*      stage       = m_resourceManager->GetBuffer( stageHandle );

        if( stage )
        {
            stage->Write( data, size, 0 );
            ExecuteImmediateTransfer( [ & ]( CommandBuffer* cmd ) { cmd->CopyBuffer( stage, dst, size, 0, dstOffset ); } );
            DT_INFO( "[StreamingManager] Immediate Buffer Upload SUCCESS: {} bytes (Thread: {})", size, GetThreadHash() );
        }
        m_resourceManager->DestroyBuffer( stageHandle );
    }

    void StreamingManager::UploadTextureImmediate( TextureHandle dstHandle, const void* data, size_t size )
    {
        Texture* dst = m_resourceManager->GetTexture( dstHandle );
        if( !dst )
            return;

        BufferDesc   stageDesc{ size, BufferType::UPLOAD };
        BufferHandle stageHandle = m_resourceManager->CreateBuffer( stageDesc );
        Buffer*      stage       = m_resourceManager->GetBuffer( stageHandle );

        if( stage )
        {
            stage->Write( data, size, 0 );
            ExecuteImmediateTransfer( [ & ]( CommandBuffer* cmd ) {
                VkImageMemoryBarrier2 barrier = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
                barrier.srcStageMask          = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
                barrier.srcAccessMask         = 0;
                barrier.dstStageMask          = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
                barrier.dstAccessMask         = VK_ACCESS_2_TRANSFER_WRITE_BIT;
                barrier.oldLayout             = VK_IMAGE_LAYOUT_UNDEFINED;
                barrier.newLayout             = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
                barrier.image                 = dst->GetHandle();
                barrier.subresourceRange      = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
                cmd->PipelineBarrier( 0, 0, 0, 0, nullptr, 0, nullptr, 1, &barrier );

                VkBufferImageCopy region{};
                region.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
                region.imageExtent      = dst->GetExtent();
                vkCmdCopyBufferToImage( cmd->GetHandle(), stage->GetHandle(), dst->GetHandle(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region );
            } );
            DT_INFO( "[StreamingManager] Immediate Texture Upload SUCCESS: {} bytes (Thread: {})", size, GetThreadHash() );
        }
        m_resourceManager->DestroyBuffer( stageHandle );
    }

    void StreamingManager::ReadbackBufferImmediate( BufferHandle srcHandle, void* outData, size_t size, size_t srcOffset )
    {
        Buffer* src = m_resourceManager->GetBuffer( srcHandle );
        if( !src )
            return;

        BufferDesc   stageDesc{ size, BufferType::READBACK };
        BufferHandle stageHandle = m_resourceManager->CreateBuffer( stageDesc );
        Buffer*      stage       = m_resourceManager->GetBuffer( stageHandle );

        if( stage )
        {
            ExecuteImmediateTransfer( [ & ]( CommandBuffer* cmd ) { cmd->CopyBuffer( src, stage, size, srcOffset, 0 ); } );
            stage->Read( outData, size, 0 );
            DT_INFO( "[StreamingManager] Immediate Buffer Readback SUCCESS: {} bytes (Thread: {})", size, GetThreadHash() );
        }
        m_resourceManager->DestroyBuffer( stageHandle );
    }

} // namespace DigitalTwin