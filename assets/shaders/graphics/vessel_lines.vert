#version 450

layout(set = 0, binding = 0) uniform CameraData {
    mat4 viewProj;
    mat4 invViewProj;
} camera;

struct Agent {
    vec4 position;
};

layout(std430, set = 0, binding = 1) readonly buffer Agents {
    Agent agents[];
};

struct VesselEdge {
    uint  agentA;
    uint  agentB;
    float dist;
    uint  flags;
};

layout(std430, set = 0, binding = 2) readonly buffer Edges {
    VesselEdge data[];
} edges;

layout(std430, set = 0, binding = 3) readonly buffer EdgeCountBuffer {
    uint count;
} edgeCount;

layout(push_constant) uniform PushConstants {
    vec4 lineColor;
} pc;

layout(location = 0) out vec4 outColor;

void main()
{
    uint edgeIdx  = gl_VertexIndex / 2;
    uint endpoint = gl_VertexIndex & 1;

    if (edgeIdx >= edgeCount.count) {
        gl_Position = vec4(0.0, 0.0, -2.0, 1.0);
        outColor = vec4(0.0);
        return;
    }

    VesselEdge e = edges.data[edgeIdx];
    uint agentIdx = (endpoint == 0) ? e.agentA : e.agentB;
    vec3 pos = agents[agentIdx].position.xyz;

    gl_Position = camera.viewProj * vec4(pos, 1.0);
    outColor = pc.lineColor;
}
