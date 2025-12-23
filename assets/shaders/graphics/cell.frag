#version 450

/**
 * @brief Fragment shader for basic cell shading.
 * Applies simple directional lighting based on normals.
 */

layout(location = 0) in vec4 inColor;
layout(location = 1) in vec3 inNormal;

layout(location = 0) out vec4 outColor;

void main() {
    // Simple fake lighting (N dot L)
    // Light coming from top-right-front
    vec3 lightDir = normalize(vec3(0.5, 1.0, 0.5));
    
    // Calculate diffuse component with minimum ambient light of 0.2
    float diff = max(dot(normalize(inNormal), lightDir), 0.2); 

    outColor = vec4(inColor.rgb * diff, inColor.a);
}