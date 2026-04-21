#version 450

// Phase 2.6.5.c.2 Step D.3 — per-agent debug vectors.
//
// Two modes selected by `pc.mode`:
//   0 = POLARITY  — red line from cell center along polarity direction.
//                   Endpoint = center + polarity.xyz · polarity.w · scale.
//                   Reveals whether polarity converged radial-outward uniformly
//                   or if some cells have drifted polarity directions
//                   (a cause of asymmetric drift).
//   1 = DRIFT     — yellow line from cell center to its initial position
//                   (stored at build time). Long line = cell drifted far
//                   from placement; short line = still in place.
//                   Directly reveals which side of the tube has physical
//                   drift and in what direction.
//
// Draw call per mode: vkCmdDraw(2, N_agents, 0, 0). 2 vertices = 1 line
// per instance (endpoint 0 = center, endpoint 1 = target).

layout(set = 0, binding = 0) uniform CameraData {
    mat4 viewProj;
} camera;

struct Agent { vec4 position; };
layout(std430, set = 0, binding = 1) readonly buffer Agents {
    Agent agents[];
};

layout(std430, set = 0, binding = 2) readonly buffer Polarity {
    vec4 data[];  // xyz = direction, w = magnitude (0..1)
} polarity;

layout(std430, set = 0, binding = 3) readonly buffer InitialPositions {
    vec4 data[];  // xyz = initial position, w = alive flag
} initialPos;

layout(push_constant) uniform PushConstants {
    uint  mode;           // 0 = polarity, 1 = drift
    float polarityScale;  // world-unit length for a polarity magnitude of 1
} pc;

layout(location = 0) out vec3 outColor;

void main() {
    uint agentIdx = uint(gl_InstanceIndex);
    uint endIdx   = uint(gl_VertexIndex);

    vec4 agentData = agents[agentIdx].position;
    if (agentData.w == 0.0) {
        gl_Position = vec4(0.0);
        outColor    = vec3(0.0);
        return;
    }
    vec3 cellCenter = agentData.xyz;

    vec3 worldPos;
    if (pc.mode == 0u) {
        // Polarity mode — red
        if (endIdx == 0u) {
            worldPos = cellCenter;
        } else {
            vec4 pol = polarity.data[agentIdx];
            worldPos = cellCenter + pol.xyz * pol.w * pc.polarityScale;
        }
        outColor = vec3(1.0, 0.25, 0.25);
    } else {
        // Drift mode — yellow
        if (endIdx == 0u) {
            worldPos = cellCenter;
        } else {
            worldPos = initialPos.data[agentIdx].xyz;
        }
        outColor = vec3(1.0, 0.9, 0.15);
    }

    gl_Position = camera.viewProj * vec4(worldPos, 1.0);
}
