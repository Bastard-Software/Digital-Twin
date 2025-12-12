#pragma once
#include "compute/ComputeGraph.hpp"
#include "rhi/CommandBuffer.hpp"
#include "rhi/DescriptorAllocator.hpp"
#include "rhi/Device.hpp"
#include <deque>

namespace DigitalTwin
{
    class ComputeEngine
    {
    public:
        ComputeEngine( Ref<Device> device );
        ~ComputeEngine();

        void Init();
        void Shutdown();

        /**
         * @brief Creates a command buffer, records the graph, and submits to the compute queue.
         * Keeps the CommandBuffer alive until execution completes.
         * @return The timeline semaphore value to wait for.
         */
        uint64_t ExecuteGraph( ComputeGraph& graph, uint32_t agentCount );

        // Helper to block CPU until task completes (for testing/sync)
        void WaitForTask( uint64_t taskID );

    private:
        // Checks completed fences and releases held command buffers
        void GarbageCollect();

    private:
        Ref<Device> m_device;

        struct InFlightWork
        {
            uint64_t           fenceValue;
            Ref<CommandBuffer> cmd;
        };

        // Queue of active tasks. We release CommandBuffers only after GPU finishes them.
        std::deque<InFlightWork> m_inflightWork;
    };
} // namespace DigitalTwin