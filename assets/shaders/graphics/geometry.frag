#version 450

layout(location = 0) in vec3 inNormal;
layout(location = 1) in vec4 inColor;

layout(location = 0) out vec4 outColor;

void main()
{
    vec3 lightDir = normalize(vec3(0.5, 1.0, 0.3));
    float diff = max(dot(inNormal, lightDir), 0.2); 
    
    outColor = vec4(inColor.rgb * diff, inColor.a);
}