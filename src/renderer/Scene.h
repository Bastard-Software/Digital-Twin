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
        BufferHandle orientationBuffer; // per-agent outward normals for mesh rotation (may be invalid)
        BufferHandle agentReorderBuffer;
        BufferHandle drawMetaBuffer;
        BufferHandle visibilityBuffer; // per-group uint32: 1=visible, 0=hidden (UPLOAD, CPU-writable)

        // Phase 2.6.5.c — per-cell Voronoi polygon data written by
        // voronoi_cell_polygon.comp. Consumed by voronoi_fan.vert via SSBO lookup.
        // Invalid handle → no AgentGroup opted into dynamic topology; static pipeline
        // covers every draw and the dynamic path is skipped in GeometryPass.
        BufferHandle polygonBuffer;

        // Phase 2.6.5.c.2 Step 1 — per-agent surface info (same SSBO the Voronoi
        // compute consumes). The voronoi_fan VS reads this to compute the TRUE
        // cylinder-surface normal at each polygon vertex. Adjacent cells'
        // shared-edge vertices then agree on outward normal → smooth shading
        // across cell boundaries → removes the per-cell flat-shading creases
        // visible on the camera-facing side in earlier Phase 2.6.5.c screenshots.
        BufferHandle surfaceInfoBuffer;

        // Phase 2.6.5.c.2 Step D.2 — JKR contact hull (272 B per group, model-
        // space sub-sphere offsets rotated by each agent's orientation).
        // Read by the debug-markers pass to render radial lines from each
        // cell center to each hull point, making both visible at once.
        BufferHandle contactHullBuffer;

        // Phase 2.6.5.c.2 Step D.3 — extra debug overlays.
        // `polarityBuffer` = per-agent polarity vec4 (xyz=direction, w=magnitude)
        //   used to draw red arrows along each cell's polarity direction.
        // `initialPositionsBuffer` = snapshot of positions at build time, used
        //   to draw yellow lines showing drift from the original placement.
        BufferHandle polarityBuffer;
        BufferHandle initialPositionsBuffer;

        // Phase 2.6.5.c.2 Step D — dynamic-topology debug flags forwarded as a
        // push constant to the voronoi_fan pipeline. Bit 0 = wireframe outline,
        // bit 1 = vertex-count tint. Zero = normal rendering (bit-identical).
        uint32_t     debugFlags       = 0;

        uint32_t     drawCount        = 0;  // total draw-command count (static + dynamic)
        uint32_t     dynamicDrawCount = 0;  // subset rendered via voronoi_fan pipeline;
                                            //   placed AT THE END of indirectCmdBuffer
                                            //   (static draws at [0, drawCount - dynamicDrawCount),
                                            //    dynamic at [drawCount - dynamicDrawCount, drawCount))
        uint32_t     totalPaddedAgents = 0;
        uint32_t     readIndex        = 0; // Which ping-pong buffer holds the latest valid agent positions

        BufferHandle GetAgentReadBuffer() const { return agentBuffers[ readIndex ]; }
        uint32_t     StaticDrawCount()   const { return drawCount - dynamicDrawCount; }
    };

} // namespace DigitalTwin