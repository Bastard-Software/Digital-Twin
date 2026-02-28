#pragma once
#include "simulation/SimulationTypes.h"

namespace DigitalTwin
{
    /**
     * @brief Utility class for generating procedural morphologies for agents.
     */
    class MorphologyGenerator
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
    };

} // namespace DigitalTwin