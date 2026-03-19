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
        BufferHandle phenotypeBuffer;
        uint32_t     drawCount  = 0;
        uint32_t     readIndex  = 0; // Which ping-pong buffer holds the latest valid agent positions

        BufferHandle GetAgentReadBuffer() const { return agentBuffers[ readIndex ]; }
    };

} // namespace DigitalTwin