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
        BufferHandle groupDataBuffer;
        BufferHandle agentBuffers[ 2 ];
        BufferHandle agentCountBuffer;
        uint32_t     drawCount = 0;

        /**
         * @brief Helper to get the correct agent buffer for reading in the current frame.
         * @param flightIndex The current frame-in-flight index (0 or 1).
         * @return Handle to the buffer that should be read by the vertex shader.
         */
        BufferHandle GetAgentReadBuffer( uint32_t flightIndex ) const { return agentBuffers[ flightIndex ]; }
    };

} // namespace DigitalTwin