#pragma once
#include "core/Core.h"
#include "simulation/GridField.h"
#include <glm/glm.hpp>
#include <vector>

namespace DigitalTwin
{
    /**
     * @brief Result type for distributions that produce both positions and per-cell outward normals.
     * Follows the same pattern as VesselTreeResult.
     */
    struct DT_API CylinderShellResult
    {
        std::vector<glm::vec4> positions;  // xyz = position, w = 1.0
        std::vector<glm::vec4> normals;    // xyz = outward radial normal, w = 0.0
    };

    /**
     * @brief Mathematical strategies for spatial distribution (seeding) of agents in the simulation volume.
     */
    class DT_API SpatialDistribution
    {
    public:
        /**
         * @brief Distributes agents uniformly inside a 3D spherical volume.
         * @param count Number of agents to seed.
         * @param radius The radius of the bounding sphere.
         * @param center The central coordinate of the distribution.
         * @return std::vector<glm::vec4> (xyz = position, w = 1.0f status flag)
         */
        static std::vector<glm::vec4> UniformInSphere( uint32_t count, float radius, const glm::vec3& center = glm::vec3( 0.0f ) );

        /**
         * @brief Distributes agents uniformly inside a 3D bounding box.
         * @param count Number of agents to seed.
         * @param extents Half-dimensions of the box (width, height, depth).
         * @param center The central coordinate of the distribution.
         */
        static std::vector<glm::vec4> UniformInBox( uint32_t count, const glm::vec3& extents, const glm::vec3& center = glm::vec3( 0.0f ) );

        /**
         * @brief Distributes agents evenly along a line segment from start to end.
         * @param count Number of agents to place.
         * @param start The start point of the line segment.
         * @param end The end point of the line segment.
         * @param spacing If > 0, fixed distance between agents (count may be capped by line length). If 0, evenly spaced.
         * @return std::vector<glm::vec4> (xyz = position, w = 1.0f status flag)
         */
        static std::vector<glm::vec4> VesselLine( uint32_t count, const glm::vec3& start, const glm::vec3& end, float spacing = 0.0f );

        /**
         * @brief Fills a sphere with agents placed on a regular cubic lattice.
         * @param spacing Distance between adjacent lattice points (should be >= cell diameter).
         * @param radius  Radius of the bounding sphere.
         * @param center  Center of the sphere.
         * @return All lattice points that fall inside the sphere. Count is determined by
         *         geometry — no RNG, fully deterministic.
         */
        static std::vector<glm::vec4> LatticeInSphere( float spacing, float radius, const glm::vec3& center = glm::vec3( 0.0f ) );

        /**
         * @brief Fills a cylinder with agents placed on a regular cubic lattice.
         * @param spacing    Distance between adjacent lattice points.
         * @param radius     Radius of the cylinder (distance from axis).
         * @param halfLength Half-length along the cylinder axis.
         * @param center     Center of the cylinder.
         * @param axis       Unit vector defining the cylinder axis (default Y-up).
         * @return All lattice points where dist_from_axis <= radius AND |projection| <= halfLength.
         */
        static std::vector<glm::vec4> LatticeInCylinder( float spacing, float radius, float halfLength,
                                                          const glm::vec3& center = glm::vec3( 0.0f ),
                                                          const glm::vec3& axis   = glm::vec3( 0.0f, 1.0f, 0.0f ) );

        /**
         * @brief Distributes agents uniformly inside a cylindrical volume.
         * @param count       Number of agents to seed.
         * @param radius      Outer radius of the cylinder.
         * @param halfLength  Half-length along the cylinder axis.
         * @param center      Center of the cylinder.
         * @param axis        Unit vector defining the cylinder axis (default Y-up).
         * @param innerRadius Inner radius cutout (0 = solid cylinder, >0 = hollow annular volume).
         * @param seed        RNG seed for deterministic placement.
         * @return std::vector<glm::vec4> (xyz = position, w = 1.0f status flag)
         */
        static std::vector<glm::vec4> UniformInCylinder(
            uint32_t count, float radius, float halfLength,
            const glm::vec3& center = glm::vec3( 0.0f ),
            const glm::vec3& axis   = glm::vec3( 0.0f, 1.0f, 0.0f ),
            float    innerRadius    = 0.0f,
            uint32_t seed           = 42 );

        /**
         * @brief Places agents in rings on the surface of a cylindrical shell (hollow tube).
         *
         * Returns both positions and per-cell outward radial normals, which can be passed
         * directly to AgentGroup::SetOrientations() to orient curved-tile morphologies outward.
         *
         * @param axialSpacing  Distance between successive rings along the cylinder axis.
         * @param radius        Radius of the cylinder shell.
         * @param halfLength    Half-length along the cylinder axis.
         * @param cellsPerRing  Number of cells in each circumferential ring.
         * @param center        Center of the cylinder.
         * @param axis          Unit vector defining the cylinder axis (default Y-up).
         * @param jitter        Random angular + axial displacement as a fraction of spacing (0 = perfect grid).
         * @param seed          RNG seed for deterministic jitter.
         */
        static CylinderShellResult ShellOnCylinder(
            float axialSpacing, float radius, float halfLength,
            uint32_t cellsPerRing,
            const glm::vec3& center = glm::vec3( 0.0f ),
            const glm::vec3& axis   = glm::vec3( 0.0f, 1.0f, 0.0f ),
            float    jitter         = 0.0f,
            uint32_t seed           = 42 );
    };

} // namespace DigitalTwin