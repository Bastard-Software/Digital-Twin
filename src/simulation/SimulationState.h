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
        uint32_t     currentReadIndex   = 0; // Drives activeIndex for ALL tasks (field + agent)
        uint32_t     latestAgentBuffer  = 0; // Final buffer written by position tasks this frame (for renderer)

        BufferHandle agentCountBuffer;
        BufferHandle phenotypeBuffer;

        // Per-agent cadherin expression profile (vec4 per slot, same global capacity as phenotypeBuffer).
        // Full-size when any group has CadherinAdhesion; 16-byte dummy otherwise.
        // x=E-cad  y=N-cad  z=VE-cad  w=Cad-11
        BufferHandle cadherinProfileBuffer;

        // 64-byte UBO holding the blueprint's 4x4 affinity matrix.
        // Always allocated — identity dummy when no group uses cadherin.
        BufferHandle cadherinAffinityBuffer;

        // Per-agent orientation normals (xyz=outward normal, w=0). Static — written once at init.
        // Groups without orientations get default (0,1,0,0). Read by geometry.vert to orient meshes.
        BufferHandle orientationBuffer;

        // Angiogenesis signaling (Notch-Dll4 pathway state per agent)
        BufferHandle signalingBuffer;

        // Anastomosis — vessel edge graph
        BufferHandle vesselEdgeBuffer;       // VesselEdge[paddedCount]: {agentA, agentB, dist, flags}
        BufferHandle vesselEdgeCountBuffer;  // uint32_t: count of recorded edges

        // Vessel connected components — uint32_t label per agent, labels[i]=i initially
        BufferHandle vesselComponentBuffer;

        // Multi-mesh rendering — per-cellType draw commands
        BufferHandle agentReorderBuffer;  // uint32_t per reorder slot: maps instance → agent index
        BufferHandle drawMetaBuffer;      // DrawMeta per draw command: {groupIndex, targetCellType, groupOffset, groupCapacity}
        uint32_t     drawCommandCount = 0;
        uint32_t     totalPaddedAgents = 0;

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