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

        // Per-agent polarity vector (xyz=outward normal, w=magnitude 0-1).
        // Full-size when any group has CellPolarity; 16-byte dummy otherwise.
        BufferHandle polarityBuffer;

        // Basement-membrane plate parameters — single global plate per simulation.
        // Always allocated (48 bytes). Shader branches on plateFlags.x.
        //   layout: vec4 normalHeight (xyz=normal, w=height)
        //           vec4 params       (x=contactStiffness, y=integrinAdhesion,
        //                               z=anchorageDistance, w=polarityBias)
        //           uvec4 flags       (x=active 0/1; yzw reserved)
        BufferHandle basementMembraneBuffer;

        // Per-agent orientation quaternions (xyzw). Dynamic when rigid body active — updated by JKR
        // each frame. Groups without a contact hull store (0,1,0,0) (shortest-arc mode in geometry.vert).
        // Groups with a contact hull store identity quaternion (0,0,0,1) (quaternion mode).
        BufferHandle orientationBuffer;

        // Per-group contact hull for rigid body dynamics (144 bytes: vec4 meta + 8 × vec4 points).
        // Allocated when any group has Biomechanics. hullMeta.x=0 for sphere/spiky-sphere groups.
        BufferHandle contactHullBuffer;

        // Angiogenesis signaling (Notch-Dll4 pathway state per agent)
        BufferHandle signalingBuffer;

        // Multi-mesh rendering — per-cellType draw commands
        BufferHandle agentReorderBuffer;  // uint32_t per reorder slot: maps instance → agent index
        BufferHandle drawMetaBuffer;      // DrawMeta per draw command: {groupIndex, targetCellType, groupOffset, groupCapacity}
        BufferHandle visibilityBuffer;    // uint32_t per group: 1=visible, 0=hidden (UPLOAD, CPU-writable)
        uint32_t     drawCommandCount = 0;
        // Phase 2.6.5.c — subset of drawCommandCount rendered via the dynamic-
        // topology (voronoi_fan) pipeline. Dynamic DrawMetas are placed at
        // the END of the indirect/meta buffers so GeometryPass can issue two
        // distinct DrawIndexedIndirect calls: [0, staticCount) on the static
        // pipeline, [staticCount, drawCount) on the dynamic pipeline.
        uint32_t     dynamicDrawCommandCount = 0;
        uint32_t     totalPaddedAgents = 0;

        // Biomechanics
        BufferHandle hashBuffer;
        BufferHandle offsetBuffer;
        BufferHandle pressureBuffer;
        // Phase 2.6.5.a — shared per-agent neighbour list, consumed by
        // jkr_forces.comp (and future voronoi_cell_polygon.comp). Layout per agent:
        //   uint count; uint indices[24]; uint _pad[7];  // 128 bytes aligned stride
        BufferHandle neighborListBuffer;
        // Phase 2.6.5.b — per-cell Voronoi polygon buffer (written each frame by
        // voronoi_cell_polygon.comp). Layout per agent (208 bytes, 16-aligned):
        //   uint count; uint _pad[3]; vec4 vertices[12];
        // Rendering consumption deferred to Phase 2.6.5.c; Phase 2.6.5.b only
        // populates the buffer + CPU-side verification via ComputeTests.
        BufferHandle polygonBuffer;

        // Grig fields
        std::vector<GridFieldState> gridFields;

        // Domain size
        glm::vec3 domainSize = glm::vec3( 1000.0f );

        // How many distinct agent groups exist (used for Indirect Draw Count)
        uint32_t groupCount = 0;

        // Simulation logic
        ComputeGraph computeGraph;

        bool IsValid() const
        {
            // Phase 2.6.5.c: a dynamic-topology-only blueprint has no static mesh
            // (vertexBuffer remains invalid) but is still renderable via the
            // Voronoi path. Treat the state as valid whenever it has agents to
            // draw through EITHER pipeline.
            return vertexBuffer.IsValid() || polygonBuffer.IsValid();
        }

        /**
         * @brief Safely destroys all GPU resources associated with this state.
         */
        void Destroy( ResourceManager* resourceManager );
    };

} // namespace DigitalTwin