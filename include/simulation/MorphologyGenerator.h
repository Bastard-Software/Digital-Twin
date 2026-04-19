#pragma once
#include "simulation/SimulationTypes.h"

#include "core/Core.h"

namespace DigitalTwin
{
    /**
     * @brief Utility class for generating procedural morphologies for agents.
     */
    class DT_API MorphologyGenerator
    {
    public:
        /**
         * @brief Creates a cubic morphology (flat-shaded).
         * @param size The length of the cube's edges.
         */
        static MorphologyData CreateCube( float size = 1.0f );

        /**
         * @brief Creates a spherical morphology (smooth-shaded UV sphere).
         * @param radius The radius of the sphere.
         * @param sectors Number of horizontal slices.
         * @param stacks Number of vertical slices.
         */
        static MorphologyData CreateSphere( float radius = 1.0f, uint32_t sectors = 36, uint32_t stacks = 18 );

        /**
         * @brief Creates a cylindrical morphology aligned along the Y axis.
         *        Suitable for StalkCell vessel tube representation.
         * @param radius Cylinder radius.
         * @param height Cylinder height.
         * @param sectors Number of angular segments.
         */
        static MorphologyData CreateCylinder( float radius = 1.0f, float height = 2.0f, uint32_t sectors = 18 );

        /**
         * @brief Creates a sphere with alternating spike vertices for a spiky/irregular look.
         *        Suitable for TipCell filopodial representation.
         * @param radius Base sphere radius.
         * @param spikeScale Radial scale factor for every other vertex (>1 creates spikes).
         * @param sectors Number of horizontal segments.
         * @param stacks Number of vertical segments.
         */
        static MorphologyData CreateSpikySphere( float radius = 1.0f, float spikeScale = 1.4f, uint32_t sectors = 16, uint32_t stacks = 8 );

        /**
         * @brief Creates a flat disc morphology aligned on the XZ plane (normal along +Y).
         *        Suitable for plate-like endothelial cells in 2D tube vessel topology.
         * @param radius Disc radius.
         * @param thickness Disc thickness (height along Y).
         * @param sectors Number of angular segments around the circumference.
         */
        static MorphologyData CreateDisc( float radius = 0.8f, float thickness = 0.2f, uint32_t sectors = 16 );

        /**
         * @brief Creates a flat rectangular tile morphology aligned on the XZ plane (normal along +Y).
         *        Suitable for brick-like endothelial cells in vessel tube topology.
         *        Width maps to the circumferential (X) direction, height to the axial (Z) direction.
         * @param width  Tile extent along X.
         * @param height Tile extent along Z.
         * @param thickness Tile thickness along Y.
         */
        static MorphologyData CreateTile( float width = 1.4f, float height = 1.4f, float thickness = 0.2f );

        /**
         * @brief Creates a curved tile morphology: a section of a cylindrical shell.
         *        At rest (no orientation applied) the outer face normal points in +Y, matching
         *        the orientation pipeline convention. When oriented outward by the per-cell
         *        quaternion, the tile conforms to the vessel lumen producing a circular cross-section.
         * @param arcAngleDeg  Arc angle subtended by one cell (degrees). Use 360/ringSize.
         * @param height       Tile height along the vessel axis (Z in local space).
         * @param thickness    Shell thickness (radial depth of the tile).
         * @param innerRadius  Vessel inner radius — sets the curvature of the tile.
         * @param sectors      Number of arc subdivisions (smoothness of the curved face).
         */
        static MorphologyData CreateCurvedTile( float arcAngleDeg = 60.0f, float height = 1.35f,
                                                float thickness = 0.25f, float innerRadius = 1.5f,
                                                uint32_t sectors = 4 );

        /**
         * @brief Flow-aligned elongated rhomboidal tile for arterial / arteriolar ECs.
         *
         * Arterial ECs under laminar flow elongate along the flow axis via
         * microtubule acetylation and VE-cadherin / PECAM-1 / VEGFR2 mechanosensing
         * (Davies 2009 DOI 10.1038/nrcardio.2009.14; Baeyens 2016 DOI 10.7554/eLife.16533).
         * Rhomboid geometry interlocks staggered (brick-pattern), preventing the
         * longitudinal-seam instability that planar rectangles would exhibit under
         * JKR + VE-cadherin catch-bond loads.
         *
         * Orientation at rest (no quaternion applied): outward face normal = +Y.
         * Local coordinate convention matches CreateCurvedTile:
         *   X = axial (flow direction, length extent)
         *   Y = radial (thickness extent, outward face normal points +Y)
         *   Z = circumferential (width extent, perpendicular to flow)
         *
         * Hull points: 8 (4 corners + 4 edge midpoints) on the Y=0 mid-plane,
         * compatible with the 16-point jkr_forces.comp contactHull buffer.
         *
         * @param length          Tile length along the axial (flow) X direction.
         * @param width           Tile width along the circumferential Z direction.
         * @param thickness       Tile thickness along the radial Y direction.
         * @param curvatureRadius Vessel inner radius (reserved for Phase 2.5 surface-conforming
         *                        placement; Phase 2.2 ships flat tiles).
         */
        static MorphologyData CreateElongatedQuad( float length = 8.0f, float width = 1.0f,
                                                   float thickness = 0.2f, float curvatureRadius = 0.0f );

        /**
         * @brief Pentagon Stone-Wales defect primitive (5-sided tile).
         *
         * Carries +pi/3 Gaussian curvature (Stone & Wales 1986 DOI 10.1016/0009-2614(86)80661-3).
         * Sits on the NARROWER side of a diameter transition or at daughter-branch
         * root rings where Murray's-law circumference expansion needs absorbing
         * (Murray 1926 DOI 10.1073/pnas.12.3.207). Biologically justified at
         * bifurcations by the cobblestone EC morphology observed at flow dividers
         * (Chiu & Chien 2011 DOI 10.1152/physrev.00047.2009; van der Heiden 2013).
         *
         * Orientation at rest: outward face normal = +Y. Polygon lies in the X-Z
         * plane with corner 0 pointing +X (axial/flow direction).
         *
         * Hull points: 10 (5 corners + 5 edge midpoints).
         *
         * @param radius          Circumscribed-circle radius of the pentagon.
         * @param thickness       Tile thickness along the radial Y direction.
         * @param curvatureRadius Vessel inner radius (reserved for Phase 2.5 surface-conforming
         *                        placement; Phase 2.2 ships flat tiles).
         */
        static MorphologyData CreatePentagonDefect( float radius = 1.0f, float thickness = 0.2f,
                                                    float curvatureRadius = 0.0f );

        /**
         * @brief Heptagon Stone-Wales defect primitive (7-sided tile).
         *
         * Carries -pi/3 Gaussian curvature. Sits on the WIDER side of a diameter
         * transition or at the carina of a Y-bifurcation where the Gauss-Bonnet
         * theorem mandates negative defects. Biologically: carina ECs are
         * cobblestone polygonal cells under multi-directional WSS
         * (Chiu & Chien 2011 DOI 10.1152/physrev.00047.2009; van der Heiden 2013).
         *
         * Orientation at rest: outward face normal = +Y. Polygon lies in the X-Z
         * plane with corner 0 pointing +X (axial/flow direction).
         *
         * Hull points: 14 (7 corners + 7 edge midpoints), fits the 16-point
         * contactHull buffer with no shader changes.
         *
         * @param radius          Circumscribed-circle radius of the heptagon.
         * @param thickness       Tile thickness along the radial Y direction.
         * @param curvatureRadius Vessel inner radius (reserved for Phase 2.5 surface-conforming
         *                        placement; Phase 2.2 ships flat tiles).
         */
        static MorphologyData CreateHeptagonDefect( float radius = 1.0f, float thickness = 0.2f,
                                                    float curvatureRadius = 0.0f );

        /**
         * @brief Creates a spheroid morphology (ellipsoid of revolution).
         *        Prolate (radiusXZ < radiusY): elongated along Y — suitable for fibroblasts,
         *        neurons, and spindle-shaped mesenchymal cells.
         *        Oblate (radiusXZ > radiusY): flattened — suitable for red blood cells,
         *        squamous cells, or pancaked tumour cells.
         * @param radiusXZ  Equatorial radius (X and Z axes — circular cross-section).
         * @param radiusY   Polar radius (Y axis — elongation/flattening direction).
         * @param sectors   Horizontal (azimuthal) subdivisions.
         * @param stacks    Vertical (latitudinal) subdivisions.
         */
        static MorphologyData CreateEllipsoid( float radiusXZ = 0.5f, float radiusY = 1.0f,
                                               uint32_t sectors = 24, uint32_t stacks = 12 );
    };

} // namespace DigitalTwin