#version 450

// Fullscreen triangle trick without vertex buffers
layout(location = 0) out vec2 v_UV;

void main() 
{
    v_UV = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    gl_Position = vec4(v_UV * 2.0f - 1.0f, 0.0f, 1.0f);
}