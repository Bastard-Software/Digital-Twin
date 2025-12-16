#pragma once
#include "resources/Mesh.hpp"

namespace DigitalTwin
{
    class ShapeGenerator
    {
    public:
        static Mesh CreateCube();
        static Mesh CreateSphere( float radius, int stacks, int slices );
    };
} // namespace DigitalTwin