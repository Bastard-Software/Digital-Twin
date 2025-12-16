#version 450

layout(location = 0) in vec4 inColor;
layout(location = 1) in vec3 inNormal;

layout(location = 0) out vec4 outColor;

void main() {
    // Simple fake lighting (N dot L)
    vec3 lightDir = normalize(vec3(0.5, 1.0, 0.5));
    float diff = max(dot(normalize(inNormal), lightDir), 0.2); // 0.2 ambient

    outColor = vec4(inColor.rgb * diff, inColor.a);
}