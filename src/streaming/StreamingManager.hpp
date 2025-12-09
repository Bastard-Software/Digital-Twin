#pragma once
#include "core/Base.hpp"
#include "rhi/Buffer.hpp"
#include "rhi/CommandBuffer.hpp"
#include "rhi/Device.hpp"
#include <vector>

namespace DigitalTwin
{
    // Number of frames processed in parallel (Double Buffering)
    constexpr uint32_t FRAMES_IN_FLIGHT = 2;

    /**
     * @brief Represents a synchronization dependency.
     * Contains the semaphore and the specific counter value to wait for.
     */
    struct SyncPoint
    {
        VkSemaphore semaphore; // The underlying timeline semaphore
        uint64_t    value;     // The counter value that indicates completion
    };

    /**
     * @brief A temporary allocation within the linear ring buffer.
     */
    struct TransientAllocation
    {
        VkBuffer        buffer        = VK_NULL_HANDLE;
        VkDeviceSize    offset        = 0;
        VkDeviceSize    size          = 0;
        void*           mappedData    = nullptr; // Host pointer for writing/reading
        VkDeviceAddress deviceAddress = 0;       // GPU address (for buffer_reference)
    };

    /**
     * @brief Internal resources dedicated to a single frame slot.
     */
    struct FrameContext
    {
        // CPU -> GPU Ring Buffer (Host Visible)
        Ref<Buffer>  uploadHeap;
        VkDeviceSize uploadOffset    = 0;
        void*        uploadMappedPtr = nullptr;

        // GPU -> CPU Ring Buffer (Host Visible, Cached recommended)
        Ref<Buffer>  readbackHeap;
        VkDeviceSize readbackOffset    = 0;
        void*        readbackMappedPtr = nullptr;

        // Command buffer for transfer operations
        Ref<CommandBuffer> transferCmd;
    };

    /**
     * @brief Manages asynchronous data streaming between CPU and GPU.
     * Implements a Ring Buffer strategy to avoid pipeline stalls.
     * Uses the Transfer Queue's internal Timeline Semaphore for synchronization.
     */
    class StreamingManager
    {
    public:
        StreamingManager( Ref<Device> device );
        ~StreamingManager();

        Result Init();
        void   Shutdown();

        // --- Frame Lifecycle ---

        /**
         * @brief Prepares resources for a new frame.
         * If the ring buffer wrapped around, it waits for the GPU to finish using the slot.
         * @param frameNumber Monotonic frame counter from the engine.
         */
        void BeginFrame( uint64_t frameNumber );

        /**
         * @brief Finalizes and submits transfer commands.
         * @return SyncPoint that Compute/Graphics queues MUST wait on before execution.
         */
        SyncPoint EndFrame();

        /**
         * @brief Blocks CPU until the current frame's transfer is fully complete.
         * Useful for immediate readback in RL steps.
         */
        void WaitForTransferComplete();

        // --- Data API ---

        /**
         * @brief Allocates space in the Upload Heap.
         * @return Allocation with a mapped pointer to write data to.
         */
        TransientAllocation AllocateUpload( VkDeviceSize size, VkDeviceSize alignment = 256 );

        /**
         * @brief Helper to upload data to a specific Device Local buffer.
         * Uses internal staging buffer.
         */
        void UploadToBuffer( Ref<Buffer> dstBuffer, const void* data, VkDeviceSize size, VkDeviceSize dstOffset = 0 );

        /**
         * @brief Commands the GPU to copy data from a buffer into the Readback Heap.
         * Data will be available in mappedData AFTER the frame completes.
         */
        TransientAllocation CaptureBuffer( Ref<Buffer> srcBuffer, VkDeviceSize size, VkDeviceSize srcOffset = 0 );

    private:
        Ref<Device>  m_device;
        FrameContext m_frames[ FRAMES_IN_FLIGHT ];

        uint32_t m_frameIndex         = 0; // Current slot index (0..FRAMES_IN_FLIGHT-1)
        uint64_t m_currentFrameNumber = 0;

        // Stores the fence value (from Transfer Queue) associated with the completion
        // of the work submitted for each frame slot.
        uint64_t m_frameFenceValues[ FRAMES_IN_FLIGHT ] = { 0 };
    };
} // namespace DigitalTwin