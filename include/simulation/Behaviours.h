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
        float       rate                  = 1.0f;
        int         requiredLifecycleState = -1; // -1 no requirements
    };

    struct SecreteField
    {
        std::string fieldName;
        float       rate                  = 1.0f;
        int         requiredLifecycleState = -1; // -1 no requirements
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
        float hypoxiaO2;           // Threshold for starvation
        float apoptosisProbPerSec; // Base probability of death per second
    };

    struct Chemotaxis
    {
        std::string fieldName;
        float       chemotacticSensitivity = 1.0f;   // Bias strength (µm²/s per unit gradient)
        float       receptorSaturation     = 0.01f;  // Kd-like constant; 0 = linear (no saturation)
        float       maxVelocity            = 5.0f;   // Hard clamp on resultant speed (µm/s)
    };

    struct NotchDll4
    {
        float dll4ProductionRate   = 1.0f;
        float dll4DecayRate        = 0.1f;
        float notchInhibitionGain  = 1.0f;
        float vegfr2BaseExpression = 1.0f;
        float tipThreshold         = 0.8f;
        float stalkThreshold       = 0.3f;
    };

    struct Anastomosis
    {
        float contactDistance = 3.0f;
    };

    struct Perfusion
    {
        std::string fieldName;
        float       baseFlowRate = 1.0f;
    };

} // namespace DigitalTwin::Behaviours

namespace DigitalTwin
{
    // A variant holding all possible behaviours the engine understands
    using BehaviourVariant = std::variant<
        Behaviours::BrownianMotion,
        Behaviours::ConsumeField,
        Behaviours::SecreteField,
        Behaviours::Biomechanics,
        Behaviours::CellCycle,
        Behaviours::Chemotaxis,
        Behaviours::NotchDll4,
        Behaviours::Anastomosis,
        Behaviours::Perfusion>;

    // Wrapper to attach execution parameters (like frequency) to a behaviour
    struct BehaviourRecord
    {
        BehaviourVariant behaviour;
        float            targetHz              = 60.0f; // Default to 60 executions per second
        int              requiredLifecycleState = -1;   // -1 = any lifecycle state
        int              requiredCellType       = -1;   // -1 = any cell type

        // Fluent API for frequency
        BehaviourRecord& SetHz( float hz )
        {
            targetHz = hz;
            return *this;
        }

        BehaviourRecord& SetRequiredLifecycleState( int s )
        {
            requiredLifecycleState = s;
            return *this;
        }

        BehaviourRecord& SetRequiredCellType( int t )
        {
            requiredCellType = t;
            return *this;
        }
    };

} // namespace DigitalTwin
