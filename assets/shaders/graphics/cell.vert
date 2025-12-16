#version 450
#extension GL_EXT_buffer_reference : require

// --- SET 0: Instance Data ---
struct Cell {
    vec4 position; // xyz, w = radius
    vec4 velocity;
    vec4 color;
    // Metadata block
    uint meshID;
    uint pad0;
    uint pad1;
    uint pad2;
};

layout(std430, set = 0, binding = 0) readonly buffer CellBuffer {
    Cell cells[];
} population;

// --- SET 1: Geometry Data ---
struct Vertex {
    vec4 position;
    vec4 normal;
    vec4 color;
};

layout(std430, set = 1, binding = 0) readonly buffer MeshBuffer {
    Vertex vertices[];
} mesh;

// --- Push Constants ---
layout(push_constant) uniform Constants {
    mat4 viewProj;
    uint targetMeshID; // Tells shader which mesh we are currently drawing
} pc;

layout(location = 0) out vec4 fragColor;
layout(location = 1) out vec3 outNormal;

void main() {
    // 1. Fetch Instance
    Cell cell = population.cells[gl_InstanceIndex];

    // 2. ID FILTERING (The Trick)
    // If this instance (Cell) does not use the mesh we are currently drawing, discard it.
    // By setting position to NaN/inf, the GPU clips it cleanly.
    if (cell.meshID != pc.targetMeshID) {
        gl_Position = vec4(0.0/0.0); // Generate NaN to discard
        return;
    }

    // 3. Geometry
    Vertex v = mesh.vertices[gl_VertexIndex];
    float scale = cell.position.w; 
    vec3 worldPos = (v.position.xyz * scale) + cell.position.xyz;

    gl_Position = pc.viewProj * vec4(worldPos, 1.0);
    fragColor = cell.color;
    outNormal = v.normal.xyz;
}