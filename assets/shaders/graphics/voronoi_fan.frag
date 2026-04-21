#version 450

// Phase 2.6.5.c.2 Step D — dedicated fragment shader for the Voronoi-fan
// dynamic-topology pipeline. Layout and lighting identical to
// `geometry.frag`, with TWO additional debug paths gated by push-constant
// flags forwarded through the VS:
//
//   bit 0 (DTOP_DEBUG_WIREFRAME)       → highlight polygon-boundary edges
//                                        using the barycentric coordinate the
//                                        VS emits (b.x = 0 on the edge
//                                        opposite the fan-centre vertex, i.e.
//                                        the perimeter-to-perimeter polygon
//                                        edge we actually want to outline).
//                                        Fan-interior edges (centre → vertex)
//                                        are NOT highlighted — they're
//                                        triangulation artefacts, not the
//                                        Voronoi polygon boundary.
//
//   bit 1 (DTOP_DEBUG_VERTEX_COUNT)    → tint by the cell's polygon vertex
//                                        count so 7-sided "heptagons above
//                                        hexagons" show as red at a glance:
//                                        3 → bright green (degenerate)
//                                        4 → green (rhomboid — normal at
//                                            dual-seam capillaries)
//                                        5 → blue (pentagon — normal at
//                                            ring-count transitions / carinas)
//                                        6 → no tint (clean hexagon)
//                                        7 → red (heptagon — the symptom)
//                                        8 → yellow (octagon — unusual)
//                                        > 8 → magenta (error).

layout(location = 0) in vec3 inNormal;
layout(location = 1) in vec4 inColor;
layout(location = 2) in vec3 inBary;

layout(location = 3) flat in uint inDebugFlags;
layout(location = 4) flat in uint inPolyCount;

layout(location = 0) out vec4 outColor;

const uint DTOP_DEBUG_WIREFRAME     = 1u << 0;
const uint DTOP_DEBUG_VERTEX_COUNT  = 1u << 1;

void main()
{
    // Base lighting — two-light setup identical to geometry.frag.
    // Fill strength 0.75 (up from 0.4) so bottom-of-tube cells retain
    // their rhombus shape in rendering; ambient floor 0.25 (up from 0.15).
    // The 2.5× top/bottom asymmetry from the 0.4 fill was making valid
    // rhombuses LOOK deformed on the back side of curved surfaces.
    vec3 keyDir  = normalize(vec3( 0.5,  1.0,  0.3));
    vec3 fillDir = normalize(vec3(-0.5, -1.0, -0.3));
    float key  = max(dot(inNormal, keyDir),  0.0);
    float fill = max(dot(inNormal, fillDir), 0.0) * 0.75;
    float diff = max(key + fill, 0.25);

    vec3 shaded = inColor.rgb * diff;

    // Vertex-count tint. Applied BEFORE wireframe so the wireframe darkening
    // is visible on top of tinted regions (wireframe overrides tint).
    if ((inDebugFlags & DTOP_DEBUG_VERTEX_COUNT) != 0u) {
        vec3 tint;
        float tintMix = 0.5;
        if      (inPolyCount == 3u) { tint = vec3(0.30, 1.00, 0.30); tintMix = 0.85; } // bright green — degenerate
        else if (inPolyCount == 4u) { tint = vec3(0.40, 0.90, 0.40); }                 // green — rhomboid
        else if (inPolyCount == 5u) { tint = vec3(0.45, 0.55, 1.00); }                 // blue — pentagon
        else if (inPolyCount == 6u) { tint = shaded; tintMix = 0.0; }                  // hexagon — no tint
        else if (inPolyCount == 7u) { tint = vec3(1.00, 0.30, 0.30); tintMix = 0.65; } // red — heptagon (symptom)
        else if (inPolyCount == 8u) { tint = vec3(1.00, 0.95, 0.30); }                 // yellow — octagon
        else                        { tint = vec3(1.00, 0.30, 1.00); tintMix = 0.85; } // magenta — error / > 8
        shaded = mix(shaded, tint, tintMix);
    }

    // Wireframe — darken pixels near the polygon-boundary edge. baryc.x is
    // 0 on the perimeter-to-perimeter edge of each fan triangle. Uses screen-
    // space derivatives so the line thickness is roughly constant in pixels
    // rather than in world units (avoids the line thinning out at distance).
    if ((inDebugFlags & DTOP_DEBUG_WIREFRAME) != 0u) {
        float d   = inBary.x;
        float dd  = fwidth(d);
        // 1.5 pixels of line, smoothstep edge for anti-aliasing.
        float line = 1.0 - smoothstep(0.0, 1.5 * dd, d);
        shaded = mix(shaded, vec3(0.04, 0.04, 0.04), line);
    }

    outColor = vec4(shaded, inColor.a);
}
