#pragma once
#include <variant>

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

namespace DigitalTwin
{
    // A variant holding all possible behaviours the engine understands
    using BehaviourVariant = std::variant<Behaviours::BrownianMotion>;

    // Wrapper to attach execution parameters (like frequency) to a behaviour
    struct BehaviourRecord
    {
        BehaviourVariant behaviour;
        float            targetHz = 60.0f; // Default to 60 executions per second

        // Fluent API for frequency
        BehaviourRecord& SetHz( float hz )
        {
            targetHz = hz;
            return *this;
        }
    };
} // namespace DigitalTwin