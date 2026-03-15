#pragma once
#include "rhi/RHITypes.h"

#include "compute/ComputeGraph.h"

namespace DigitalTwin
{
    class ResourceManager;

    /**
     * @brief Holds the ping-pong 3D textures for a single PDE grid field.
     */
    struct GridFieldState
    {
        std::string   name;                   // Needed to match behaviours to grids
        BufferHandle  interactionDeltaBuffer; // SSBO for atomic agent interactions
        TextureHandle textures[ 2 ];          // 0: Read, 1: Write (swapped each tick)
        uint32_t      currentReadIndex = 0;

        // Dimensions in voxels
        uint32_t width  = 0;
        uint32_t height = 0;
        uint32_t depth  = 0;
    };

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

        BufferHandle agentCountBuffer;
        BufferHandle phenotypeBuffer;

        // Biomechanics
        BufferHandle hashBuffer;
        BufferHandle offsetBuffer;
        BufferHandle pressureBuffer;

        // Grig fields
        std::vector<GridFieldState> gridFields;

        // Domain size
        glm::vec3 domainSize = glm::vec3( 1000.0f );

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