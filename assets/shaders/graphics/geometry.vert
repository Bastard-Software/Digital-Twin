#version 450

#extension GL_ARB_shader_draw_parameters : require

layout(set = 0, binding = 0) uniform CameraData {
    mat4 viewProj;
} camera;

struct Vertex {
    vec4 pos;
    vec4 normal;
};

layout(std430, set = 0, binding = 1) readonly buffer Geometry {
    Vertex vertices[];
};

struct Agent {
    vec4 position;
};

layout(std430, set = 0, binding = 2) readonly buffer Agents {
    Agent agents[];
};

layout(std140, binding = 3) readonly buffer GroupData {
    vec4 colors[];
} groupData;

layout(location = 0) out vec3 outNormal;
layout(location = 1) out vec4 outColor;

void main() 
{
    Vertex v = vertices[gl_VertexIndex];
    Agent a = agents[gl_InstanceIndex];

    vec3 scale = vec3(a.position.w);
    vec3 worldPos = (v.pos.xyz * scale) + a.position.xyz;

    gl_Position = camera.viewProj * vec4(worldPos, 1.0);

    outNormal = v.normal.xyz;
outColor = groupData.colors[gl_DrawIDARB];
}