#version 450

layout(location = 0) in vec3 inNormal;
layout(location = 0) out vec4 outColor;

void main()
{
vec3 lightDir = normalize(vec3(0.5, 1.0, 0.3));
    float diff = max(dot(inNormal, lightDir), 0.2); 

    vec3 baseColor = vec3(0.8, 0.7, 0.1); 
    
    outColor = vec4(baseColor * diff, 1.0);
}