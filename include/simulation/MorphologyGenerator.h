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
    };

} // namespace DigitalTwin