#version 450

// Input from Vertex Shader
layout(location = 0) in vec4 inColor;

// Final Output Color
layout(location = 0) out vec4 outColor;

void main() {
    // Just output the interpolated color directly.
    // No lighting, no discard, no transparency logic.
    // This is the safest way to ensure something draws on screen.
    outColor = inColor;
}