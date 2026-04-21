#version 450

// Phase 2.6.5.c.2 Step D.2 — debug marker pass fragment shader.
// Pass-through colour set by the vertex shader. Solid opaque lines.

layout(location = 0) in vec3 inColor;
layout(location = 0) out vec4 outColor;

void main() {
    outColor = vec4(inColor, 1.0);
}
