#pragma once
#include "rhi/RHITypes.h"

namespace DigitalTwin
{
    /**
     * @brief Container for all scene data required by the renderer.
     * Separates the simulation state from the rendering logic.
     */
    class Scene
    {
    public:
        BufferHandle vertexBuffer;
        BufferHandle indexBuffer;

        BufferHandle indirectCmdBuffer;

        BufferHandle agentBuffers[ 2 ];

        /**
         * @brief Helper to get the correct agent buffer for reading in the current frame.
         * @param flightIndex The current frame-in-flight index (0 or 1).
         * @return Handle to the buffer that should be read by the vertex shader.
         */
        BufferHandle GetAgentReadBuffer( uint32_t flightIndex ) const
        {
            // Simple ping-pong logic based on flight index
            return agentBuffers[ flightIndex % 2 ];
        }
    };

} // namespace DigitalTwin