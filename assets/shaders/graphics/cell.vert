#version 450

// Data structure matching the C++ definition
struct Cell {
    vec4 position; // .xyz = position, .w = radius
    vec4 velocity; // .xyz = velocity
    vec4 color;    // .rgba = color
};

// Read-only access to the storage buffer containing cell data
// Binding 0, Set 0 must match C++ BindingGroup configuration
layout(std430, set = 0, binding = 0) readonly buffer CellBuffer {
    Cell cells[];
} population;

// Push Constants for the View-Projection matrix
layout(push_constant) uniform Constants {
    mat4 viewProj;
} pc;

// Output to Fragment Shader
layout(location = 0) out vec4 fragColor;

// Procedural Cube Geometry (Unit size: -0.5 to 0.5)
// 36 vertices for 12 triangles (Standard Triangle List)
const vec3 CUBE[36] = vec3[](
    // Front face
    vec3(-0.5,-0.5, 0.5), vec3( 0.5,-0.5, 0.5), vec3( 0.5, 0.5, 0.5),
    vec3( 0.5, 0.5, 0.5), vec3(-0.5, 0.5, 0.5), vec3(-0.5,-0.5, 0.5),
    // Back face
    vec3(-0.5,-0.5,-0.5), vec3(-0.5, 0.5,-0.5), vec3( 0.5, 0.5,-0.5),
    vec3( 0.5, 0.5,-0.5), vec3( 0.5,-0.5,-0.5), vec3(-0.5,-0.5,-0.5),
    // Top face
    vec3(-0.5, 0.5,-0.5), vec3(-0.5, 0.5, 0.5), vec3( 0.5, 0.5, 0.5),
    vec3( 0.5, 0.5, 0.5), vec3( 0.5, 0.5,-0.5), vec3(-0.5, 0.5,-0.5),
    // Bottom face
    vec3(-0.5,-0.5,-0.5), vec3( 0.5,-0.5,-0.5), vec3( 0.5,-0.5, 0.5),
    vec3( 0.5,-0.5, 0.5), vec3(-0.5,-0.5, 0.5), vec3(-0.5,-0.5,-0.5),
    // Right face
    vec3( 0.5,-0.5,-0.5), vec3( 0.5, 0.5,-0.5), vec3( 0.5, 0.5, 0.5),
    vec3( 0.5, 0.5, 0.5), vec3( 0.5,-0.5, 0.5), vec3( 0.5,-0.5,-0.5),
    // Left face
    vec3(-0.5,-0.5,-0.5), vec3(-0.5,-0.5, 0.5), vec3(-0.5, 0.5, 0.5),
    vec3(-0.5, 0.5, 0.5), vec3(-0.5, 0.5,-0.5), vec3(-0.5,-0.5,-0.5)
);

void main() {
    // 1. Fetch cell data based on Instance Index (one instance per cell)
    Cell cell = population.cells[gl_InstanceIndex];
    
    // 2. Get local vertex position from the constant array
    vec3 localPos = CUBE[gl_VertexIndex]; 
    
    // 3. Scale the cube
    // The CUBE is size 1.0 (-0.5 to 0.5).
    // The physical size should be (2 * Radius).
    float diameter = 2.0 * cell.position.w;
    
    // 4. Calculate World Position
    vec3 worldPos = (localPos * diameter) + cell.position.xyz;
    
    // 5. Output transformed position
    gl_Position = pc.viewProj * vec4(worldPos, 1.0);
    
    // 6. Pass color to fragment shader
    fragColor = cell.color;
}