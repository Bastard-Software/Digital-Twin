#pragma once
#include <string>
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

    struct ConsumeField
    {
        std::string fieldName;
        float       rate = 1.0f;
    };

    struct SecreteField
    {
        std::string fieldName;
        float       rate = 1.0f;
    };

    /**
     * @brief Raw data structure for the GPU Compute Compiler.
     * Contains pre-calculated mathematical constants derived from biological parameters.
     */
    struct Biomechanics
    {
        float repulsionStiffness = 15.0f; // How hard cells push each other apart (Hertz force)
        float adhesionStrength   = 2.0f;  // How strongly cells stick together (JKR force)
        float maxRadius          = 1.5f;  // The interaction radius for collision detection
    };

    /**
     * @brief Raw data structure for the GPU Compute Compiler.
     * Contains highly optimized rates (per second) pre-calculated from biological hours/days.
     */
    struct CellCycle
    {
        float growthRatePerSec;    // Fraction of biomass gained per second
        float targetO2;            // Required O2 for optimal proliferation
        float arrestPressure;      // Threshold for Contact Inhibition (Quiescence)
        float necrosisO2;          // Threshold for starvation death
        float apoptosisProbPerSec; // Base probability of death per second
    };

} // namespace DigitalTwin::Behaviours

namespace DigitalTwin
{
    // A variant holding all possible behaviours the engine understands
    using BehaviourVariant =
        std::variant<Behaviours::BrownianMotion, Behaviours::ConsumeField, Behaviours::SecreteField, Behaviours::Biomechanics, Behaviours::CellCycle>;

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