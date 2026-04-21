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

// gl_DrawIDARB starts at 0 per DrawIndexedIndirect call. Dynamic draws come
// AFTER static draws in the indirect command buffer, so we offset the color
// lookup by the push-constant `drawIdOffset` (set on the CPU side to the
// static draw count).
layout(push_constant) uniform PushConstants {
    uint drawIdOffset;
} pc;

layout(location = 0) out vec3 outNormal;
layout(location = 1) out vec4 outColor;

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

    // Outward normal = agent's radial-out direction derived from the orientation
    // quaternion (+Y local). Mirrors geometry.vert's dual-mode convention
    // (w ≈ 0 → legacy shortest-arc; w ≠ 0 → full quaternion).
    vec4 orient = orientations.data[agentIdx];
    vec3 outward;
    if (abs(orient.w) > 0.001) {
        outward = qrot(orient, vec3(0.0, 1.0, 0.0));
    } else if (length(orient.xyz) > 0.5) {
        outward = normalize(orient.xyz);
    } else {
        outward = vec3(0.0, 1.0, 0.0);
    }
    outNormal = outward;

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
