#pragma once
#include "simulation/Phenotype.h"
#include <glm/glm.hpp>
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
        std::string  fieldName;
        float        rate                  = 1.0f;
        LifecycleState requiredLifecycleState = LifecycleState::Any;
    };

    struct SecreteField
    {
        std::string  fieldName;
        float        rate                  = 1.0f;
        LifecycleState requiredLifecycleState = LifecycleState::Any;
    };

    /**
     * @brief Apical-basal polarity for tissue organisation and lumen formation.
     * Each cell develops a polarity vector pointing outward (basal direction) based
     * on local neighbor geometry. Polarity modulates JKR adhesion: basal-basal contacts
     * are strengthened, apical-apical contacts are weakened or repulsive, driving
     * cavity/lumen formation from initially solid cell aggregates.
     *
     * Works in combination with Biomechanics (required).
     *
     * Biological relevance:
     *   - Endothelial cells:    strong polarity → vessel lumen formation
     *   - Tumour cells:         regulationRate ≈ 0 → polarity loss (EMT, invasion)
     *   - Epithelial cells:     strong polarity → organized tissue layers / acini
     *   - Immune/white cells:   do not use this (front-rear migration polarity is Chemotaxis)
     */
    struct CellPolarity
    {
        float regulationRate  = 0.1f;  // EMA rate for polarity adaptation (1/s)
        float apicalRepulsion = 0.5f;  // adhesion multiplier when apical faces apical (< 1 = weaker)
        float basalAdhesion   = 1.5f;  // adhesion multiplier when basal faces basal (> 1 = stronger)
    };

    /**
     * @brief Differential adhesion through cadherin expression profiles.
     * Works in combination with Biomechanics: when both are present on a group,
     * the JKR adhesion term is scaled by the per-pair affinity
     *   A(i,j) = dot(profile_i, M * profile_j)
     * where M is the blueprint-level 4x4 affinity matrix.
     *
     * Expression channels (cadherinProfile vec4):
     *   x = E-cadherin  (epithelial, liver)
     *   y = N-cadherin  (mesenchymal, neural, invasive tumour)
     *   z = VE-cadherin (vascular endothelial)
     *   w = Cadherin-11 (osteoblast, bone metastasis)
     */
    struct CadherinAdhesion
    {
        glm::vec4 targetExpression = glm::vec4( 0.0f ); // Genetic target per channel (0-1)
        float     expressionRate   = 0.01f;              // Up-regulation speed (1/s)
        float     degradationRate  = 0.001f;             // Down-regulation speed (1/s)
        float     couplingStrength = 1.0f;               // Global force scale
        float     _pad             = 0.0f;               // Pad to 32 bytes
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
        std::vector<uint32_t>                     edgeFlags;     // parallel to explicitEdges: RING=0x1, AXIAL=0x2, JUNCTION=0x4
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
        Behaviours::CellPolarity,
        Behaviours::CadherinAdhesion,
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
        float            targetHz              = 60.0f;
        LifecycleState   requiredLifecycleState = LifecycleState::Any;
        CellType         requiredCellType       = CellType::Any;

        // Fluent API for frequency
        BehaviourRecord& SetHz( float hz )
        {
            targetHz = hz;
            return *this;
        }

        BehaviourRecord& SetRequiredLifecycleState( LifecycleState s )
        {
            requiredLifecycleState = s;
            return *this;
        }

        BehaviourRecord& SetRequiredCellType( CellType t )
        {
            requiredCellType = t;
            return *this;
        }
    };

} // namespace DigitalTwin
