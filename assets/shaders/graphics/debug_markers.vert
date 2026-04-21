#version 450
#extension GL_ARB_shader_draw_parameters : require

// Phase 2.6.5.c.2 Step D.2 — debug marker pass.
//
// Draws, per agent, 8 radial LINES from the cell center to each of the
// 8 contact-hull sub-sphere positions (transformed by the cell's
// orientation quaternion). The shared cell-center endpoint makes the
// cell position visible as the star's hub; the opposite endpoints show
// where JKR thinks the cell's contact boundary sits.
//
// Draw call: vkCmdDraw(16, N_agents, 0, 0) on a LINE_LIST pipeline.
//   - 16 vertices per instance = 8 lines × 2 endpoints.
//   - Vertex 2k   (k ∈ [0,8)) → cell center (all 8 converge here).
//   - Vertex 2k+1 → hull point k in world space.

layout(set = 0, binding = 0) uniform CameraData {
    mat4 viewProj;
} camera;

struct Agent { vec4 position; };
layout(std430, set = 0, binding = 1) readonly buffer Agents {
    Agent agents[];
};

layout(std430, set = 0, binding = 2) readonly buffer Orientations {
    vec4 data[];
} orientations;

// Matches ContactHullGPU struct in SimulationBuilder (272 bytes):
//   vec4 meta        (x = hullCount)
//   vec4 points[16]  (xyz = model-space offset from cell center, w = radius)
layout(std430, set = 0, binding = 3) readonly buffer ContactHull {
    vec4 meta;
    vec4 points[16];
} hull;

layout(location = 0) out vec3 outColor;

vec3 qrot(vec4 q, vec3 v) {
    return v + 2.0 * cross(q.xyz, cross(q.xyz, v) + q.w * v);
}

void main() {
    uint agentIdx = uint(gl_InstanceIndex);
    uint vertIdx  = uint(gl_VertexIndex);

    vec4 agentData = agents[agentIdx].position;
    // Skip dead agents (w == 0) by collapsing all 16 verts to origin
    // (degenerate — no visible line).
    if (agentData.w == 0.0) {
        gl_Position = vec4(0.0);
        outColor    = vec3(0.0);
        return;
    }

    vec3 cellCenter = agentData.xyz;
    vec4 orient     = orientations.data[agentIdx];
    if (abs(orient.w) < 0.001 && dot(orient.xyz, orient.xyz) < 0.001) {
        orient = vec4(0.0, 0.0, 0.0, 1.0); // identity
    }

    uint pairIdx = vertIdx / 2u;   // which hull line (0..7)
    uint endIdx  = vertIdx % 2u;   // 0 = center, 1 = hull point

    vec3 worldPos;
    if (endIdx == 0u) {
        worldPos = cellCenter;
        // Warm yellow at the center — makes the converging hub obvious.
        outColor = vec3(1.0, 0.95, 0.3);
    } else {
        vec3 hullOffset = hull.points[pairIdx].xyz;
        worldPos = cellCenter + qrot(orient, hullOffset);
        // Cool cyan at the hull point — contrast against the centre and
        // against the vessel's pink/magenta base colour so the debug
        // overlay is readable at any camera angle.
        outColor = vec3(0.2, 0.95, 1.0);
    }

    gl_Position = camera.viewProj * vec4(worldPos, 1.0);
}
