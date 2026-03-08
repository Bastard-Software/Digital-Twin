#pragma once

namespace DigitalTwin::Behaviours
{
    /**
     * @brief Represents random thermal motion of particles.
     * Pure data structure (AST node) for the Simulation Compiler.
     */
    struct BrownianMotion
    {
        float speed = 1.0f;
    };

    // Future mechanisms: Chemotaxis, ConsumeField, Proliferate, etc.
} // namespace DigitalTwin::Behaviours