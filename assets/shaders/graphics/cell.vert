#version 450
#extension GL_EXT_buffer_reference : require

// --- SET 0: Instance Data ---
struct Cell {
    vec4 position; // xyz, w = radius
    vec4 velocity;
    vec4 color;
};

layout(std430, set = 0, binding = 0) readonly buffer CellBuffer {
    Cell cells[];
} population;

// --- SET 1: Geometry Data (Unified Buffer) ---
struct Vertex {
    vec4 position; // xyz
    vec4 normal;   // xyz
    vec4 color;    // rgba
};

layout(std430, set = 1, binding = 0) readonly buffer MeshBuffer {
    Vertex vertices[];
} mesh;

// --- Push Constants ---
layout(push_constant) uniform Constants {
    mat4 viewProj;
} pc;

// --- Outputs ---
layout(location = 0) out vec4 fragColor;
layout(location = 1) out vec3 outNormal; // Pass normal to frag for simple lighting

void main() {
    // 1. Fetch Instance Data
    Cell cell = population.cells[gl_InstanceIndex];

    // 2. Fetch Vertex Data (Programmable Pulling)
    Vertex v = mesh.vertices[gl_VertexIndex];

    // 3. Transform
    // Sphere Generator creates unit sphere (radius 1.0) or diam 1.0 depending on params.
    // Assuming generated sphere has radius 1.0. 
    // Cell.w is radius.
    // Final Scale = Cell.w (if mesh is radius 1)
    
    vec3 localPos = v.position.xyz;
    float scale = cell.position.w; // Radius
    
    vec3 worldPos = (localPos * scale) + cell.position.xyz;

    gl_Position = pc.viewProj * vec4(worldPos, 1.0);

    // Pass data to fragment shader
    fragColor = cell.color;
    outNormal = v.normal.xyz;
}