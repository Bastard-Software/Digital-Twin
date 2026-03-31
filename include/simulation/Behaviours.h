#pragma once
#include <string>
#include <variant>
#include <vector>

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
        float repulsionStiffness  = 15.0f; // How hard cells push each other apart (Hertz force)
        float adhesionStrength    = 2.0f;  // How strongly cells stick together (JKR force)
        float maxRadius           = 1.5f;  // The interaction radius for collision detection
        float dampingCoefficient  = 0.0f;  // Velocity drag (dashpot). 0 = no damping.
    };

    /**
     * @brief Raw data structure for the GPU Compute Compiler.
     * Contains highly optimized rates (per second) pre-calculated from biological hours/days.
     */
    struct CellCycle
    {
        float growthRatePerSec;            // Fraction of biomass gained per second
        float targetO2;                    // Required O2 for optimal proliferation
        float arrestPressure;              // Threshold for Contact Inhibition (Quiescence)
        float necrosisO2;                  // Threshold for starvation death
        float hypoxiaO2;                   // Threshold for starvation
        float apoptosisProbPerSec;         // Base probability of death per second
        bool  directedMitosis = false;     // Place daughter along vessel axis (StalkCell sprouting)
    };

    struct Chemotaxis
    {
        std::string fieldName;
        float       chemotacticSensitivity    = 1.0f;  // Bias strength (µm²/s per unit gradient)
        float       receptorSaturation        = 0.01f; // Kd-like constant; 0 = linear (no saturation)
        float       maxVelocity               = 5.0f;  // Hard clamp on resultant speed (µm/s)
        float       contactInhibitionDensity  = 0.0f;  // 0 = disabled; >0 = neighbor count for full stop
    };

    struct NotchDll4
    {
        float       dll4ProductionRate   = 1.0f;
        float       dll4DecayRate        = 0.1f;
        float       notchInhibitionGain  = 1.0f;
        float       vegfr2BaseExpression = 1.0f;
        float       tipThreshold         = 0.8f;
        float       stalkThreshold       = 0.3f;
        std::string vegfFieldName;  // empty = no VEGF gating (vegfr2 unmodulated)
        uint32_t    subSteps             = 1;   // ODE iterations per frame — more = faster convergence
    };

    struct Anastomosis
    {
        float contactDistance = 3.0f;
        bool  allowTipToStalk = true; // TipCell can fuse with StalkCell from a different vessel component
    };

    // VEGF-gated Phalanx ↔ Stalk transitions.
    // PhalanxCells (quiescent) activate to StalkCells when local VEGF exceeds activationThreshold.
    // StalkCells re-quiesce to PhalanxCells when local VEGF drops below deactivationThreshold.
    // TipCells are never affected. Must be added BEFORE NotchDll4 in the behaviour list.
    struct PhalanxActivation
    {
        std::string vegfFieldName;
        float       activationThreshold   = 20.0f; // VEGF level to wake PhalanxCell → StalkCell
        float       deactivationThreshold = 5.0f;  // VEGF level to re-quiesce StalkCell → PhalanxCell
    };

    // Hooke's Law spring forces along vessel edges, keeping the tube coherent as TipCells migrate.
    struct VesselSpring
    {
        float springStiffness    = 5.0f;  // Hooke's k — force per unit stretch per second
        float restingLength      = 2.0f;  // Equilibrium cell-cell distance
        float dampingCoefficient = 0.0f;  // Implicit Euler velocity drag. 0 = no damping.
        // When true (default), PhalanxCells are exempt from spring forces (quiescent vessel wall).
        // Set false for static vessel demos where all cells are PhalanxCells and springs must hold them.
        bool  anchorPhalanxCells = true;
    };

    // Seeds initial vessel edges at build time — no GPU shader. Two modes:
    //   explicitEdges non-empty: upload exactly those edges (supports 2D ring topology from VesselTreeGenerator).
    //   explicitEdges empty:     fall back to consecutive-pair chains within each segmentCounts segment.
    struct VesselSeed
    {
        std::vector<uint32_t>                     segmentCounts; // one entry per branch (used for validation + fallback)
        std::vector<std::pair<uint32_t,uint32_t>> explicitEdges; // if non-empty, overrides segmentCounts for edge upload
    };

    // Vessel injects a substance into the field (O2, glucose). Rate > 0.
    struct Perfusion
    {
        std::string fieldName;
        float       rate = 1.0f;
    };

    // Vessel removes a substance from the field (lactate, CO2). Rate > 0 (sign negated by builder).
    struct Drain
    {
        std::string fieldName;
        float       rate = 1.0f;
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
        Behaviours::PhalanxActivation,
        Behaviours::NotchDll4,
        Behaviours::Anastomosis,
        Behaviours::Perfusion,
        Behaviours::Drain,
        Behaviours::VesselSeed,
        Behaviours::VesselSpring>;

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
