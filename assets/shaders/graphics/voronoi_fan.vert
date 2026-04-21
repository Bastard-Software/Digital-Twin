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
    uint count;
    uint pad0;
    uint pad1;
    uint pad2;
    vec4 vertices[12];
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
    uint drawIdOffset;
    uint debugFlags;
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

    // Fan triangulation layout: MAX_POLY_VERTS triangles, 3 vertices each.
    //   triangle i:
    //     vert 0 = cell centre
    //     vert 1 = polygon.vertices[i]
    //     vert 2 = polygon.vertices[(i+1) % count]
    // If i ≥ count (polygon has fewer vertices than MAX_POLY_VERTS) the
    // triangle collapses to the cell centre → zero-area → rasterizer-discarded.
    uint triangleId = gl_VertexIndex / 3u;
    uint vertInTri  = gl_VertexIndex % 3u;

    vec3 worldPos;
    if (poly.count < 3u || triangleId >= poly.count) {
        worldPos = cellCenter;
    } else if (vertInTri == 0u) {
        worldPos = cellCenter;
    } else if (vertInTri == 1u) {
        worldPos = poly.vertices[triangleId].xyz;
    } else {
        uint nextIdx = (triangleId + 1u) % poly.count;
        worldPos = poly.vertices[nextIdx].xyz;
    }

    gl_Position = camera.viewProj * vec4(worldPos, 1.0);

    // Outward normal — per-vertex, NOT per-cell. For cell-center fan vertex the
    // normal is the cell's radialOut (identical to the old per-cell behaviour).
    // For polygon vertices we compute the true cylinder-surface normal.
    //
    // The cylinder axis is a LINE (through `cellCenter - R·radialOut` in the
    // `tangentX` direction), not a point. For a vertex with axial offset u from
    // the cell, the closest axis point slides along `tangentX` by that same u.
    // A naïve `normalize(worldPos - fixedAxisPoint)` treats the axis as a point
    // and mixes an axial component into the normal — exactly the error that
    // made the camera-facing side look hourglass-pinched in earlier screenshots
    // (the tilted normals fought the lighting direction along the tube axis).
    //
    //   axisPoint_fixed = cellCenter - R · radialOut          // at cell's axial coord
    //   axialOffset     = (worldPos - axisPoint_fixed) · tangentX
    //   axisPoint_local = axisPoint_fixed + axialOffset · tangentX
    //   normal          = normalize(worldPos - axisPoint_local)
    //
    // With the axis projected properly, pure-axial offsets produce a normal
    // equal to radialOut (correct for a cylinder — normal has no axial
    // component), and pure-circumferential offsets produce the rotated
    // `cos(θ)·radialOut + sin(θ)·tangentZ`. Adjacent cells sharing an edge
    // still agree on the normal → smooth shading across cell boundaries.
    vec4 orient = orientations.data[agentIdx];
    vec3 radialOut;
    if (abs(orient.w) > 0.001) {
        radialOut = qrot(orient, vec3(0.0, 1.0, 0.0));
    } else if (length(orient.xyz) > 0.5) {
        radialOut = normalize(orient.xyz);
    } else {
        radialOut = vec3(0.0, 1.0, 0.0);
    }

    float surfaceRadius = surfaceInfo.data[agentIdx].x;
    bool  isFanCenter   = (vertInTri == 0u) || (poly.count < 3u) || (triangleId >= poly.count);
    if (surfaceRadius > 0.0 && !isFanCenter) {
        vec3  tangentX    = qrot(orient, vec3(1.0, 0.0, 0.0));
        vec3  axisRef     = cellCenter - surfaceRadius * radialOut;
        float axialOffset = dot(worldPos - axisRef, tangentX);
        vec3  axisPoint   = axisRef + axialOffset * tangentX;
        outNormal = normalize(worldPos - axisPoint);
    } else {
        outNormal = radialOut;
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
