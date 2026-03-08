#pragma once
#include "rhi/RHITypes.h"

#include "compute/ComputeGraph.h"

namespace DigitalTwin
{
    class ResourceManager;

    /**
     * @brief Represents the compiled, GPU-side state of a simulation.
     * Contains handles to the heavily optimized SoA buffers and mega-buffers.
     * This class is internal to the engine and should not be exposed to the user.
     */
    struct SimulationState
    {
        BufferHandle vertexBuffer;
        BufferHandle indexBuffer;
        BufferHandle indirectCmdBuffer;
        BufferHandle groupDataBuffer;

        // Ping-pong buffers for compute shader integration (Read/Write swapping)
        BufferHandle agentBuffers[ 2 ];
        uint32_t     currentReadIndex = 0;

        // How many distinct agent groups exist (used for Indirect Draw Count)
        uint32_t groupCount = 0;

        // Simulation logic
        ComputeGraph computeGraph;

        bool IsValid() const { return vertexBuffer.IsValid(); }

        /**
         * @brief Safely destroys all GPU resources associated with this state.
         */
        void Destroy( ResourceManager* resourceManager );
    };

} // namespace DigitalTwin