#version 450

layout(location = 0) in vec3 inNormal;
layout(location = 1) in vec4 inColor;

layout(location = 0) out vec4 outColor;

void main()
{
    // Two-light setup with stronger fill for bottom-of-tube readability
    // (user observation: fill at 0.4 made bottom rhombuses look deformed
    // even though the geometry was correct — 0.34 vs 0.85 brightness was
    // a 2.5× asymmetry). Bumped fill to 0.75 and ambient floor to 0.25 so
    // back-facing cells retain cell shape while key-light contrast still
    // provides depth cues on top.
    vec3 keyDir  = normalize(vec3( 0.5,  1.0,  0.3));
    vec3 fillDir = normalize(vec3(-0.5, -1.0, -0.3));

    float key  = max(dot(inNormal, keyDir),  0.0);
    float fill = max(dot(inNormal, fillDir), 0.0) * 0.75;

    float diff = max(key + fill, 0.25);

    outColor = vec4(inColor.rgb * diff, inColor.a);
}