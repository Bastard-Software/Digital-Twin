#version 450

layout(location = 0) in vec2 v_UV;
layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform CameraUBO 
{
    mat4 viewProj;
    mat4 invViewProj;
} camera;

layout(set = 0, binding = 1) uniform sampler3D u_Grid;

layout(push_constant) uniform PC
{
    int   mode;       // 0 = Volumetric Cloud, 1 = Slice 2D
    float sliceZ;
    float opacitySlice;
    float opacityCloud;
    vec3  domainSize;
    float normalizationScale; // Concentration divided by this to get [0,1] range
} pc;

vec2 intersectAABB(vec3 ro, vec3 rd, vec3 boxMin, vec3 boxMax) 
{
    vec3 tMin = (boxMin - ro) / rd;
    vec3 tMax = (boxMax - ro) / rd;
    vec3 t1 = min(tMin, tMax);
    vec3 t2 = max(tMin, tMax);
    float tNear = max(max(t1.x, t1.y), t1.z);
    float tFar = min(min(t2.x, t2.y), t2.z);
    return vec2(tNear, tFar);
}

float sampleGrid(vec3 uvw) 
{
    // ivec3 texSize = imageSize(u_Grid);
    // ivec3 coord = clamp(ivec3(uvw * vec3(texSize)), ivec3(0), texSize - ivec3(1));
    // return imageLoad(u_Grid, coord).r;
    return texture(u_Grid, uvw).r;
}

// --- MEDICAL TRANSFER FUNCTION (Heatmap) ---
vec4 getTransferFunction(float concentration, float globalOpacity)
{
    float t = clamp(concentration / pc.normalizationScale, 0.0, 1.0);

    // If there is almost no oxygen, it's completely transparent
    if (t <= 0.01) return vec4(0.0);

    // Color gradient: Deep Blue -> Red -> Bright Yellow
    vec3 cold = vec3(0.0, 0.1, 0.5);
    vec3 mid  = vec3(0.8, 0.1, 0.2);
    vec3 hot  = vec3(1.0, 0.9, 0.1);

    vec3 color = (t < 0.5) ? mix(cold, mid, t * 2.0) : mix(mid, hot, (t - 0.5) * 2.0);
    
    // Smooth alpha ramp based on concentration
    float alpha = smoothstep(0.01, 0.3, t) * globalOpacity;

    return vec4(color, alpha);
}

void main() 
{
    vec4 ndcFar  = vec4(v_UV * 2.0 - 1.0, 1.0, 1.0);
    vec4 ndcNear = vec4(v_UV * 2.0 - 1.0, 0.0, 1.0);
    
    vec4 worldFar  = camera.invViewProj * ndcFar;
    vec4 worldNear = camera.invViewProj * ndcNear;
    worldFar  /= worldFar.w;
    worldNear /= worldNear.w;

    vec3 ro = worldNear.xyz;
    vec3 rd = normalize(worldFar.xyz - worldNear.xyz);

    vec3 boxMin = -pc.domainSize * 0.5;
    vec3 boxMax =  pc.domainSize * 0.5;

    if (pc.mode == 1) 
    {
        // --- 2D SLICE MODE ---
        float targetZ = mix(boxMin.z, boxMax.z, pc.sliceZ);
        
        if (abs(rd.z) < 1e-5) { outColor = vec4(0.0); return; }
        
        float t = (targetZ - ro.z) / rd.z;
        if (t < 0.0) { outColor = vec4(0.0); return; }

        vec3 p = ro + rd * t;
        if (any(lessThan(p, boxMin)) || any(greaterThan(p, boxMax))) { outColor = vec4(0.0); return; }

        vec3 uvw = (p - boxMin) / pc.domainSize;
        float val = sampleGrid(uvw);
        
        outColor = getTransferFunction(val, pc.opacitySlice);
    } 
    else 
    {
        // --- VOLUMETRIC CLOUD MODE ---
        vec2 hit = intersectAABB(ro, rd, boxMin, boxMax);
        if (hit.x > hit.y || hit.y < 0.0) { outColor = vec4(0.0); return; }

        float tmin = max(hit.x, 0.0);
        float tmax = hit.y;

        int steps = 64;
        float stepSize = (tmax - tmin) / float(steps);
        float currentT = tmin;

        vec4 accum = vec4(0.0);

        for (int i = 0; i < steps; i++) 
        {
            vec3 p = ro + rd * currentT;
            vec3 uvw = (p - boxMin) / pc.domainSize;
            float val = sampleGrid(uvw);

            vec4 sampleColor = getTransferFunction(val, pc.opacityCloud);
            
            if (sampleColor.a > 0.0) 
            {
                sampleColor.rgb *= sampleColor.a;
                accum += sampleColor * (1.0 - accum.a);
                if (accum.a >= 0.95) break;
            }
            currentT += stepSize;
        }

        outColor = accum;
    }
}