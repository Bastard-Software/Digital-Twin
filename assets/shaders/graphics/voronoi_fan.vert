#version 450
#extension GL_ARB_shader_draw_parameters : require

// Item 2 Phase 2.6.5.c — variable-vertex-count per-instance rendering for
// dynamic-topology cells. Reads the per-cell Voronoi polygon from
// PolygonBuffer (populated by voronoi_cell_polygon.comp in Phase 2.6.5.b)
// and emits a MAX_POLY_VERTS-triangle fan per instance. Unused fan slots
// (triangle index ≥ polygon.count) collapse to the cell centre, producing
// zero-area triangles that the rasterizer discards at no fill-rate cost.

#define MAX_POLY_VERTS 12u

layout(set = 0, binding = 0) uniform CameraData {
    mat4 viewProj;
} camera;

struct Agent { vec4 position; };
layout(std430, set = 0, binding = 2) readonly buffer Agents {
    Agent agents[];
};

layout(std140, set = 0, binding = 3) readonly buffer GroupData {
    vec4 colors[];
} groupData;

struct PhenotypeData {
    uint  lifecycleState;
    float biomass;
    float timer;
    uint  cellType;
};
layout(std430, set = 0, binding = 4) readonly buffer Phenotypes {
    PhenotypeData data[];
} phenotypes;

layout(std430, set = 0, binding = 5) readonly buffer ReorderBuffer {
    uint indices[];
} reorderBuffer;

layout(std430, set = 0, binding = 6) readonly buffer Orientations {
    vec4 data[];
} orientations;

struct CellPolygon {
    uint  count;
    // Phase 2.6.5.c.2 Step 4a.g — outward normal (radialOut) written by the
    // compute shader; used to extrude the polygon ±thickness/2 for 3D biprism.
    float normal_x;
    float normal_y;
    float normal_z;
    vec4  vertices[12];
};
layout(std430, set = 0, binding = 7) readonly buffer Polygons {
    CellPolygon data[];
} polygons;

// Phase 2.6.5.c.2 Step 1 — per-agent surface info (same layout as the compute
// shader's binding 4). x = surfaceRadius (R); y = sizeScale; zw reserved.
// Used here to compute the per-vertex cylinder-surface normal so shared-edge
// vertices between adjacent cells agree on outward direction → smooth shading
// across cell boundaries → visible seams from flat per-cell normals disappear.
layout(std430, set = 0, binding = 8) readonly buffer SurfaceInfo {
    vec4 data[];
} surfaceInfo;

// gl_DrawIDARB starts at 0 per DrawIndexedIndirect call. Dynamic draws come
// AFTER static draws in the indirect command buffer, so we offset the color
// lookup by the push-constant `drawIdOffset` (set on the CPU side to the
// static draw count).
//
// Phase 2.6.5.c.2 Step D — `debugFlags` forwarded to the fragment shader.
// Bit 0 = wireframe outline, bit 1 = vertex-count tint. Zero = normal
// rendering (bit-identical to pre-Step-D output).
layout(push_constant) uniform PushConstants {
    uint  drawIdOffset;
    uint  debugFlags;
    float thickness;       // Phase 2.6.5.c.2 Step 4a.g — biprism extrusion amount
} pc;

layout(location = 0) out vec3 outNormal;
layout(location = 1) out vec4 outColor;
// Phase 2.6.5.c.2 Step D — barycentric coord + debug forwarding.
layout(location = 2) out vec3 outBary;
layout(location = 3) flat out uint outDebugFlags;
layout(location = 4) flat out uint outPolyCount;

// Rotate a vector by a quaternion (q.xyz + q.w). Matches geometry.vert's
// `qrot` used elsewhere in the engine.
vec3 qrot(vec4 q, vec3 v) {
    return v + 2.0 * cross(q.xyz, cross(q.xyz, v) + q.w * v);
}

void main() {
    // Reorder buffer: draw-command instance → global agent index.
    uint agentIdx = reorderBuffer.indices[gl_InstanceIndex];

    vec3        cellCenter = agents[agentIdx].position.xyz;
    CellPolygon poly       = polygons.data[agentIdx];

    // Phase 2.6.5.c.2 Step 4a.g — BIPRISM triangulation: 48 triangles per cell
    // = 144 indices total, laid out as:
    //   triangleId [0,  12): TOP fan         — perimeter + centre, +thickness/2
    //   triangleId [12, 24): BOTTOM fan      — perimeter + centre, -thickness/2 (flipped winding)
    //   triangleId [24, 48): SIDE quads      — 12 quads × 2 triangles, connecting top and bottom
    //
    // Unused slots (sub-index ≥ poly.count) collapse to cell centre → zero-area
    // triangle → rasterizer discards at no fill-rate cost.
    //
    // Phase 2.6.5.c.2 Step 4a.h — perimeter vertex inflation. Each polygon
    // vertex is pushed 8 % outward from the cell centre so adjacent cells
    // overlap slightly at shared edges. Closes the visible gaps between
    // flow-aligned rhomboids at diagonal junctions under ±15% placeJitter.
    // Applied in the VS only (not the compute shader) so the raw polygon
    // buffer stays bit-identical for tests that assert exact vertex positions.
    // The biprism body conceals the per-cell overlap region inside the shell.
    const float kInflate = 1.08;
    vec3  biprismNormal = normalize(vec3(poly.normal_x, poly.normal_y, poly.normal_z));
    float halfThickness = pc.thickness * 0.5;
    vec3  topOffset     = +halfThickness * biprismNormal;
    vec3  botOffset     = -halfThickness * biprismNormal;

    uint triangleId = gl_VertexIndex / 3u;
    uint vertInTri  = gl_VertexIndex % 3u;

    vec3 worldPos;
    vec3 vertexNormal = biprismNormal; // default; overridden per region
    bool isFanCenter  = false;
    bool isDegenerate = (poly.count < 3u);

    if (isDegenerate) {
        // Polygon has too few vertices — collapse entire biprism to cell centre.
        worldPos = cellCenter;
    }
    else if (triangleId < MAX_POLY_VERTS) {
        // --- TOP FAN ---
        uint subTri = triangleId;
        if (subTri >= poly.count) {
            worldPos = cellCenter + topOffset;
        } else if (vertInTri == 0u) {
            worldPos     = cellCenter + topOffset;
            isFanCenter  = true;
        } else if (vertInTri == 1u) {
            vec3 pv  = cellCenter + (poly.vertices[subTri].xyz - cellCenter) * kInflate;
            worldPos = pv + topOffset;
        } else {
            uint nextIdx = (subTri + 1u) % poly.count;
            vec3 pv  = cellCenter + (poly.vertices[nextIdx].xyz - cellCenter) * kInflate;
            worldPos = pv + topOffset;
        }
        vertexNormal = biprismNormal;
    }
    else if (triangleId < 2u * MAX_POLY_VERTS) {
        // --- BOTTOM FAN --- (flip winding order so the bottom faces outward)
        uint subTri = triangleId - MAX_POLY_VERTS;
        if (subTri >= poly.count) {
            worldPos = cellCenter + botOffset;
        } else if (vertInTri == 0u) {
            worldPos    = cellCenter + botOffset;
            isFanCenter = true;
        } else if (vertInTri == 1u) {
            // Swapped with vertInTri==2 to flip the winding (vs top fan).
            uint nextIdx = (subTri + 1u) % poly.count;
            vec3 pv  = cellCenter + (poly.vertices[nextIdx].xyz - cellCenter) * kInflate;
            worldPos = pv + botOffset;
        } else {
            vec3 pv  = cellCenter + (poly.vertices[subTri].xyz - cellCenter) * kInflate;
            worldPos = pv + botOffset;
        }
        vertexNormal = -biprismNormal;
    }
    else {
        // --- SIDE QUADS --- 24 triangles total (2 per quad, 12 quads).
        uint quadTriId = triangleId - 2u * MAX_POLY_VERTS;
        uint quadId    = quadTriId / 2u;
        uint triInQuad = quadTriId % 2u;

        if (quadId >= poly.count) {
            worldPos = cellCenter;
        } else {
            uint nextIdx = (quadId + 1u) % poly.count;
            vec3 pvCurr  = cellCenter + (poly.vertices[quadId].xyz  - cellCenter) * kInflate;
            vec3 pvNext  = cellCenter + (poly.vertices[nextIdx].xyz - cellCenter) * kInflate;
            vec3 vTop    = pvCurr + topOffset;
            vec3 vNxtTop = pvNext + topOffset;
            vec3 vBot    = pvCurr + botOffset;
            vec3 vNxtBot = pvNext + botOffset;

            // Quad vertices laid out CCW when viewed from outside:
            //   vBot → vNxtBot → vNxtTop → vTop  (going around)
            // Split into 2 triangles:
            //   Tri 0: vBot,    vNxtBot, vNxtTop
            //   Tri 1: vBot,    vNxtTop, vTop
            if (triInQuad == 0u) {
                if      (vertInTri == 0u) worldPos = vBot;
                else if (vertInTri == 1u) worldPos = vNxtBot;
                else                      worldPos = vNxtTop;
            } else {
                if      (vertInTri == 0u) worldPos = vBot;
                else if (vertInTri == 1u) worldPos = vNxtTop;
                else                      worldPos = vTop;
            }

            // Side-face outward normal: perpendicular to the edge, tangent to
            // the surface, pointing away from cell centre. Approximated as the
            // in-tangent-plane projection of (edgeMidpoint - cellCenter).
            vec3 edgeMid = 0.5 * (poly.vertices[quadId].xyz + poly.vertices[nextIdx].xyz);
            vec3 outward = edgeMid - cellCenter;
            outward      = outward - dot(outward, biprismNormal) * biprismNormal;
            float oLen   = length(outward);
            vertexNormal = (oLen > 1e-4) ? (outward / oLen) : biprismNormal;
        }
    }

    gl_Position = camera.viewProj * vec4(worldPos, 1.0);

    // Phase 2.6.5.c.2 Step 4a.g — biprism per-region normals.
    // Top-fan perimeter vertices get the TRUE cylinder-surface outward normal
    // (computed by projecting onto the cell's local axis) for smooth shading
    // across cell boundaries. Top-fan CENTRE vertex, bottom-fan vertices, and
    // side-quad vertices use the region-specific `vertexNormal` already set
    // above (biprismNormal, -biprismNormal, or tangent-outward respectively).
    float surfaceRadius = surfaceInfo.data[agentIdx * 2u].x; // extended SurfaceInfo layout
    bool  isTopFanPerim = (triangleId < MAX_POLY_VERTS)
                          && !isFanCenter && !isDegenerate
                          && (triangleId < poly.count);

    if (surfaceRadius > 0.0 && isTopFanPerim) {
        // Use cell's orientation to get the local tangent axis; then project
        // the vertex onto the cylinder's axis line to get the exact outward
        // direction. Prevents the "hourglass pinched middle" seam artefact
        // documented in prior sessions.
        vec4 orient = orientations.data[agentIdx];
        vec3 tangentX;
        if (abs(orient.w) > 0.001) {
            tangentX = qrot(orient, vec3(1.0, 0.0, 0.0));
        } else {
            tangentX = vec3(1.0, 0.0, 0.0);
        }
        vec3  axisRef     = cellCenter - surfaceRadius * biprismNormal;
        float axialOffset = dot(worldPos - axisRef, tangentX);
        vec3  axisPoint   = axisRef + axialOffset * tangentX;
        outNormal         = normalize(worldPos - axisPoint);
    } else {
        outNormal = vertexNormal;
    }

    // Phase 2.6.5.c.2 Step D — per-vertex barycentric coordinate. The
    // polygon-boundary edge of each fan triangle is the edge OPPOSITE the
    // fan-centre vertex (perimeter → perimeter). In barycentric terms, a
    // fragment on that edge has `bary.x == 0`. Assigning (1,0,0) to the
    // fan-centre vertex and (0,1,0) / (0,0,1) to the two perimeter vertices
    // makes `bary.x` interpolate linearly from 1 at the centre to 0 along
    // the polygon edge — ready for the frag shader to detect.
    if      (vertInTri == 0u) outBary = vec3(1.0, 0.0, 0.0);
    else if (vertInTri == 1u) outBary = vec3(0.0, 1.0, 0.0);
    else                      outBary = vec3(0.0, 0.0, 1.0);
    outDebugFlags = pc.debugFlags;
    outPolyCount  = poly.count;

    // Lifecycle-state modulation identical to geometry.vert so dynamic-topology
    // cells shade consistently with the static-pipeline cells.
    vec4 baseColor = groupData.colors[gl_DrawIDARB + pc.drawIdOffset];
    uint state     = phenotypes.data[agentIdx].lifecycleState;
    if (state == 0u)      outColor = baseColor;
    else if (state == 1u) outColor = vec4(baseColor.rgb * 0.6, baseColor.a);
    else if (state == 2u) outColor = mix(baseColor, vec4(0.3, 0.0, 0.8, 1.0), 0.7);
    else if (state == 3u) outColor = mix(baseColor, vec4(1.0, 0.6, 0.0, 1.0), 0.8);
    else if (state == 4u) outColor = vec4(0.15, 0.12, 0.12, 1.0);
    else                  outColor = baseColor;
}
