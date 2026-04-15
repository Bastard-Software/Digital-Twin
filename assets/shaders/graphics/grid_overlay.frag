#version 450

layout(location = 0) in vec2 v_UV;
layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform CameraUBO
{
    mat4 viewProj;
    mat4 invViewProj;
} camera;

layout(set = 0, binding = 1) uniform sampler3D u_Grid;

// Push constant layout (offsets must match C++ VisPushConstants exactly):
//   0  int   mode
//   4  float sliceZ
//   8  float opacitySlice
//  12  float opacityCloud
//  16  vec3  domainSize     (std430: vec3 aligned to 16)
//  28  float minValue
//  32  float maxValue
//  36  float alphaCutoff
//  40  int   colormap
//  44  float gamma
//  48  vec4  customLow      (std430: vec4 aligned to 16, 48 is multiple of 16)
//  64  vec4  customMid
//  80  vec4  customHigh
// Total: 96 bytes
layout(push_constant) uniform PC
{
    int   mode;
    float sliceZ;
    float opacitySlice;
    float opacityCloud;
    vec3  domainSize;
    float minValue;
    float maxValue;
    float alphaCutoff;
    int   colormap;
    float gamma;
    vec4  customLow;
    vec4  customMid;
    vec4  customHigh;
} pc;

vec2 intersectAABB(vec3 ro, vec3 rd, vec3 boxMin, vec3 boxMax)
{
    vec3 tMin = (boxMin - ro) / rd;
    vec3 tMax = (boxMax - ro) / rd;
    vec3 t1 = min(tMin, tMax);
    vec3 t2 = max(tMin, tMax);
    float tNear = max(max(t1.x, t1.y), t1.z);
    float tFar  = min(min(t2.x, t2.y), t2.z);
    return vec2(tNear, tFar);
}

float sampleGrid(vec3 uvw)
{
    return texture(u_Grid, uvw).r;
}

// ── Colormap helpers ──────────────────────────────────────────────────────────

// 3-stop piecewise linear gradient: c0 at t=0, c1 at t=0.5, c2 at t=1
vec3 threeStop(vec3 c0, vec3 c1, vec3 c2, float t)
{
    if (t < 0.5) return mix(c0, c1, t * 2.0);
    return mix(c1, c2, (t - 0.5) * 2.0);
}

// JET: blue → cyan → green → yellow → red
vec3 applyJet(float t)
{
    float r = clamp(1.5 - abs(4.0 * t - 3.0), 0.0, 1.0);
    float g = clamp(1.5 - abs(4.0 * t - 2.0), 0.0, 1.0);
    float b = clamp(1.5 - abs(4.0 * t - 1.0), 0.0, 1.0);
    return vec3(r, g, b);
}

// OXYGEN: dark blue (hypoxic) → cyan → yellow → red (normoxic)
vec3 applyOxygen(float t)
{
    vec3 c0 = vec3(0.0, 0.0,  0.4);
    vec3 c1 = vec3(0.0, 0.8,  0.8);
    vec3 c2 = vec3(0.9, 0.9,  0.0);
    vec3 c3 = vec3(0.8, 0.0,  0.0);
    if (t < 0.333) return mix(c0, c1, t / 0.333);
    if (t < 0.667) return mix(c1, c2, (t - 0.333) / 0.334);
    return mix(c2, c3, (t - 0.667) / 0.333);
}

// HOT: black → red → yellow → white  (Fiji "Fire" / fluorescence standard)
vec3 applyHot(float t)
{
    vec3 c0 = vec3(0.0, 0.0, 0.0);
    vec3 c1 = vec3(1.0, 0.0, 0.0);
    vec3 c2 = vec3(1.0, 1.0, 0.0);
    vec3 c3 = vec3(1.0, 1.0, 1.0);
    if (t < 0.333) return mix(c0, c1, t / 0.333);
    if (t < 0.667) return mix(c1, c2, (t - 0.333) / 0.334);
    return mix(c2, c3, (t - 0.667) / 0.333);
}

// PLASMA: dark purple → magenta → orange → yellow  (Matplotlib plasma, perceptually uniform)
vec3 applyPlasma(float t)
{
    vec3 c0 = vec3(0.05, 0.03, 0.53);
    vec3 c1 = vec3(0.49, 0.01, 0.66);
    vec3 c2 = vec3(0.88, 0.31, 0.30);
    vec3 c3 = vec3(0.99, 0.91, 0.14);
    if (t < 0.333) return mix(c0, c1, t / 0.333);
    if (t < 0.667) return mix(c1, c2, (t - 0.333) / 0.334);
    return mix(c2, c3, (t - 0.667) / 0.333);
}

// VEGF: near-black → yellow → orange → magenta
vec3 applyVegf(float t)
{
    vec3 c0 = vec3(0.05, 0.0, 0.1);
    vec3 c1 = vec3(0.8,  0.8, 0.0);
    vec3 c2 = vec3(0.9,  0.4, 0.0);
    vec3 c3 = vec3(0.9,  0.0, 0.7);
    if (t < 0.333) return mix(c0, c1, t / 0.333);
    if (t < 0.667) return mix(c1, c2, (t - 0.333) / 0.334);
    return mix(c2, c3, (t - 0.667) / 0.333);
}

// ── Transfer function ─────────────────────────────────────────────────────────

vec4 getTransferFunction(float val, float globalOpacity)
{
    float range = max(pc.maxValue - pc.minValue, 1e-5);
    float t = clamp((val - pc.minValue) / range, 0.0, 1.0);

    if (t <= pc.alphaCutoff) return vec4(0.0);

    // Apply gamma ramp (gamma < 1 lifts weak gradients; > 1 focuses on peaks)
    t = pow(t, max(pc.gamma, 0.01));

    vec3 color;
    if      (pc.colormap == 1) color = applyOxygen(t);
    else if (pc.colormap == 2) color = applyHot(t);
    else if (pc.colormap == 3) color = applyPlasma(t);
    else if (pc.colormap == 4) color = applyVegf(t);
    else if (pc.colormap == 5) color = threeStop(pc.customLow.rgb, pc.customMid.rgb, pc.customHigh.rgb, t);
    else                       color = applyJet(t);

    float alpha = smoothstep(pc.alphaCutoff, pc.alphaCutoff + 0.2, t) * globalOpacity;
    return vec4(color, alpha);
}

// ── Main ──────────────────────────────────────────────────────────────────────

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

        int   steps    = 64;
        float stepSize = (tmax - tmin) / float(steps);
        float currentT = tmin;

        vec4 accum = vec4(0.0);

        for (int i = 0; i < steps; i++)
        {
            vec3  p   = ro + rd * currentT;
            vec3  uvw = (p - boxMin) / pc.domainSize;
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
