#version 450
#extension GL_EXT_buffer_reference : require

/**
 * @brief Vertex shader for rendering cells.
 * * DESIGN:
 * - Set 0: Instance Data (Population) - Bound once per frame.
 * - Set 1: Geometry Data (Mesh) - Bound per draw call.
 * - Push Constants: Camera matrices and current MeshID.
 */

// --- SET 0: Instance Data ---

struct Cell {
    vec4 position; // xyz = position, w = radius
    vec4 velocity;
    vec4 color;
    uint meshID;
    uint pad0;
    uint pad1;
    uint pad2;
};

// FIXED: Removed GlobalData. Population is now at Binding 0 to simplify C++ binding logic.
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
    uint targetMeshID; // Used for filtering instances
} pc;

layout(location = 0) out vec4 fragColor;
layout(location = 1) out vec3 outNormal;

void main() {
    // 1. Fetch Instance Data from SSBO
    // This reads from the simulation buffer updated by the Compute Shader.
    Cell cell = population.cells[gl_InstanceIndex];

    // 2. ID Filtering
    // If this cell is meant to be rendered with a different mesh (e.g., Sphere vs Cube),
    // discard this vertex by emitting NaN. The GPU will clip it.
    if (cell.meshID != pc.targetMeshID) {
        gl_Position = vec4(0.0/0.0); 
        return;
    }

    // 3. Fetch Geometry
    Vertex v = mesh.vertices[gl_VertexIndex];
    float scale = cell.position.w;
    
    // 4. Calculate Final Position
    // Model Matrix is effectively (Position + Scale), Rotation is omitted for now.
    vec3 worldPos = (v.position.xyz * scale) + cell.position.xyz;
    gl_Position = pc.viewProj * vec4(worldPos, 1.0);
    
    // 5. Pass attributes to fragment shader
    fragColor = cell.color;
    outNormal = v.normal.xyz;
}