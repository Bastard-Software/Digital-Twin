#pragma once
#include "DigitalTwinTypes.h"
#include "rhi/RHITypes.h"

#include "core/Core.h"
#include <mutex>
#include <vector>

namespace DigitalTwin
{
    class ResourceManager;

    enum class TransferType
    {
        UploadBuffer,
        UploadTexture,
        ReadbackBuffer,
        ReadbackTexture,
    };

    struct TransferRequest
    {
        TransferType type;

        // Handles
        BufferHandle  stagingHandle; // For Upload: Source. For Readback: Destination.
        BufferHandle  targetBuffer;  // Dst for upload, Src for readback
        TextureHandle targetTexture; // Dst for upload / Src for readback

        size_t size;
        size_t bufferOffset; // Offset in the target/source buffer

        // Texture specifics
        uint32_t width = 0, height = 0, depth = 0;
        uint32_t mipLevel = 0, arrayLayer = 0;
    };

    /**
     * @brief Manages data transfer between CPU and GPU.
     * Supports deferred uploads (batched at EndFrame) and immediate blocking uploads.
     * Thread-safe.
     */
    class StreamingManager
    {
    public:
        StreamingManager( Device* device, ResourceManager* resourceManager );
        ~StreamingManager();

        Result Initialize();
        void   Shutdown();

        void     BeginFrame();
        uint64_t EndFrame();

        /**
         * @brief Queues buffer data for upload.
         * Copies data to staging immediately, records copy command at EndFrame.
         */
        void UploadBuffer( BufferHandle dstBuffer, const void* data, size_t size, size_t dstOffset = 0 );

        /**
         * @brief Queues texture data for upload.
         * Note: Texture will remain in VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL after upload.
         */
        void UploadTexture( TextureHandle dstTexture, const void* data, size_t size );

        /**
         * @brief Queues a readback from GPU buffer to a READBACK buffer.
         * @param dstReadbackBuffer Must be created with BufferType::READBACK.
         */
        void ReadbackBuffer( BufferHandle srcBuffer, BufferHandle dstReadbackBuffer, size_t size, size_t srcOffset = 0 );

        /**
         * @brief Queues a readback from GPU Texture to a READBACK buffer.
         * @param dstReadbackBuffer Must be created with BufferType::READBACK.
         * @param size Size in bytes (must match texture dimensions * format size).
         */
        void ReadbackTexture( TextureHandle srcTexture, BufferHandle dstReadbackBuffer, size_t size );

        void UploadBufferImmediate( BufferHandle dstBuffer, const void* data, size_t size, size_t dstOffset = 0 );
        void UploadTextureImmediate( TextureHandle dstTexture, const void* data, size_t size );
        void ReadbackBufferImmediate( BufferHandle srcBuffer, void* outData, size_t size, size_t srcOffset = 0 );

    private:
        // Helper to execute commands immediately on a transient context
        void ExecuteImmediateTransfer( std::function<void( CommandBuffer* )> recordCallback );

    private:
        Device*          m_device;
        ResourceManager* m_resourceManager;

        // Main Thread Transfer Context (used in Begin/End Frame)
        ThreadContextHandle m_mainTransferCtxHandle;
        ThreadContext*      m_mainTransferCtx = nullptr;

        CommandBufferHandle m_currentCmdHandle;
        CommandBuffer*      m_currentCmd = nullptr;

        // Deferred Queue
        std::vector<TransferRequest> m_pendingTransfers;
        std::mutex                   m_queueMutex;
    };

} // namespace DigitalTwin