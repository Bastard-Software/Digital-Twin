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

struct PhenotypeData {
    uint  lifecycleState;
    float biomass;
    float timer;
    uint  cellType;
};
layout(std430, set = 0, binding = 4) readonly buffer Phenotypes {
    PhenotypeData data[];
} phenotypes;

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

    vec4 baseColor = groupData.colors[gl_DrawIDARB];
    uint state = phenotypes.data[gl_InstanceIndex].lifecycleState;

    if (state == 0) {
        outColor = baseColor;
    } 
    else if (state == 1) {
        outColor = vec4(baseColor.rgb * 0.6, baseColor.a);
    } 
    else if (state == 2) {
        outColor = mix(baseColor, vec4(0.3, 0.0, 0.8, 1.0), 0.7);
    } 
    else if (state == 3) {
        outColor = mix(baseColor, vec4(1.0, 0.6, 0.0, 1.0), 0.8);
    } 
    else if (state == 4) {
        outColor = vec4(0.15, 0.12, 0.12, 1.0);
    } 
    else {
        outColor = baseColor;
    }
}