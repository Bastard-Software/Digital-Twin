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
    };

} // namespace DigitalTwin