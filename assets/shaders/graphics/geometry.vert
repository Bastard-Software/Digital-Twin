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

layout(std430, set = 0, binding = 5) readonly buffer ReorderBuffer {
    uint indices[];
} reorderBuffer;

layout(std430, set = 0, binding = 6) readonly buffer Orientations {
    vec4 orientations[];
};

layout(location = 0) out vec3 outNormal;
layout(location = 1) out vec4 outColor;

void main()
{
    // Reorder buffer maps draw-command instance index → global agent index
    uint agentIdx = reorderBuffer.indices[gl_InstanceIndex];

    Vertex v = vertices[gl_VertexIndex];
    Agent a = agents[agentIdx];

    vec3 scale = vec3(a.position.w);

    // Per-cell orientation (dual mode):
    //   w == 0  → legacy: shortest-arc from +Y to stored xyz normal (no twist control)
    //   w != 0  → full unit quaternion (qx,qy,qz,qw) — controls direction AND axial twist
    vec4 orient  = orientations[agentIdx];
    vec3 meshPos = v.pos.xyz;
    vec3 meshNrm = v.normal.xyz;

    if (abs(orient.w) > 0.001)
    {
        // Full quaternion mode: apply directly — used by sprout daughter cells
        vec4 q = orient;
        meshPos = meshPos + 2.0 * cross(q.xyz, cross(q.xyz, meshPos) + q.w * meshPos);
        meshNrm = meshNrm + 2.0 * cross(q.xyz, cross(q.xyz, meshNrm) + q.w * meshNrm);
    }
    else
    {
        // Legacy shortest-arc mode — used by parent tube cells and non-vessel agents
        vec3  storedNormal = orient.xyz;
        float nLen         = length(storedNormal);
        if (nLen > 0.5)
        {
            vec3  n    = storedNormal / nLen;
            vec3  from = vec3(0.0, 1.0, 0.0);
            float cosA = dot(from, n);

            if (cosA < -0.9999)
            {
                meshPos = vec3( meshPos.x, -meshPos.y, -meshPos.z);
                meshNrm = vec3( meshNrm.x, -meshNrm.y, -meshNrm.z);
            }
            else if (cosA < 0.9999)
            {
                vec4 q = normalize(vec4(cross(from, n), 1.0 + cosA));
                meshPos = meshPos + 2.0 * cross(q.xyz, cross(q.xyz, meshPos) + q.w * meshPos);
                meshNrm = meshNrm + 2.0 * cross(q.xyz, cross(q.xyz, meshNrm) + q.w * meshNrm);
            }
        }
    }

    vec3 worldPos = (meshPos * scale) + a.position.xyz;

    gl_Position = camera.viewProj * vec4(worldPos, 1.0);
    outNormal = meshNrm;

    vec4 baseColor = groupData.colors[gl_DrawIDARB];
    uint state = phenotypes.data[agentIdx].lifecycleState;

    // Lifecycle state modulation
    if (state == 0u) {
        outColor = baseColor;
    }
    else if (state == 1u) {
        outColor = vec4(baseColor.rgb * 0.6, baseColor.a);
    }
    else if (state == 2u) {
        outColor = mix(baseColor, vec4(0.3, 0.0, 0.8, 1.0), 0.7);
    }
    else if (state == 3u) {
        outColor = mix(baseColor, vec4(1.0, 0.6, 0.0, 1.0), 0.8);
    }
    else if (state == 4u) {
        outColor = vec4(0.15, 0.12, 0.12, 1.0);
    }
    else {
        outColor = baseColor;
    }
}
