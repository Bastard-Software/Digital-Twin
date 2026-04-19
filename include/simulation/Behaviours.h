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
        // Adhesion multiplier when apical faces apical.
        //   > 1.0  = stronger adhesion (rare biologically)
        //   0 < x < 1 = weakened adhesion (attenuates VE-cad at apical pole)
        //   = 0   = no apical-apical adhesion
        //   < 0   = actively REPULSIVE at apical pole (PODXL electrostatic
        //           repulsion, cord-hollowing mechanism; Strilic 2009)
        // Net pair adhesion reads `adhScale = mix(apicalRepulsion, basalAdhesion, alignment)`
        // times cadherin affinity, so a negative value here flips adhesion to
        // repulsion for fully-apical-aligned contacts.
        float apicalRepulsion = 0.5f;
        float basalAdhesion   = 1.5f;  // adhesion multiplier when basal faces basal (> 1 = stronger)
        // Phase 4.5 — weight of the apical-basal cue propagated from polar
        // neighbours via cell-cell junctional coupling (St Johnston & Ahringer
        // 2010; Bryant et al. 2010 PAR/Crumbs cascade). Solves the interior-
        // polarity gap: without propagation, interior cells have
        // neighbour-centroid magnitude ≈ 0 and never feel apical-basal
        // adhesion scaling; with propagation, an anchored seed (plate-
        // polarised cells) transmits orientation cell-to-cell through the
        // aggregate, enabling cord hollowing from the interior outward.
        //   0.0  = disabled — interior cells stay unpolarised (Phase 4 behaviour)
        //   0.5-1.0 = biologically plausible for tight-junction coupling
        //   >1.0 = dominates geometric centroid (sweep values for research use)
        // Sub-threshold neighbour polarity (mean magnitude < 0.05) is treated
        // as zero — a biologically-motivated deadband (junctional PAR
        // recruitment has a cooperativity threshold) that also prevents
        // FP-noise feedback amplification in non-seeded clusters.
        float propagationStrength = 0.0f;
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
        // Phase 5 — Rakshit et al. 2012 (PNAS) catch-bond multiplier. Under
        // tensile load a VE-cadherin X-dimer bond transiently STRENGTHENS
        // (conformational switch to a high-affinity state) until a peak
        // ~30 pN, then rapidly weakens and ruptures (slip-bond tail).
        // Modelled here as a multiplier on the cadherin affinity A:
        //   loadNorm = dist / interactDist  (0 = compressed, 1 = just breaking)
        //   catchMul = 1 + catchBondStrength * smoothstep(0, peak, loadNorm)
        //   if loadNorm > peak: catchMul *= (1 - smoothstep(peak, 1, loadNorm))
        //   A *= catchMul
        // Biological effect: vessel walls, cords, and monolayers resist tearing
        // under active mechanical stress (cord hollowing, sprout migration,
        // flow shear) without needing a stiffer baseline adhesion, which would
        // instead produce overly-compact aggregates. Disabled (= 0) preserves
        // pre-Phase-5 behaviour bit-exact.
        float     catchBondStrength = 0.0f;              // peak multiplier under tensile load (2.0 = VE-cad X-dimer calibrated)
        float     catchBondPeakLoad = 0.3f;              // normalised load at peak (0-1); beyond this, slip-bond rupture tail
    };

    /**
     * @brief Raw data structure for the GPU Compute Compiler.
     * Contains pre-calculated mathematical constants derived from biological parameters.
     *
     * corticalTension models the actomyosin cortex contractility that opposes
     * cell-cell adhesion (Maître et al. 2012 "Adhesion functions in cell sorting
     * by mechanically coupling the cortices of adhering cells"). In the
     * interfacial-tension framework the per-pair interfacial tension is
     *   γ_ij = (T_i + T_j) - W_ij
     * where T is cortical tension and W is adhesion work. The effect in this
     * model: for each overlapping neighbour pair an outward force
     *   F_tension = corticalTension * overlap * dir_ij
     * is added, linearly opposing the adhesive inward pull. Raising tension
     * thus stiffens the aggregate (rounder spheroids, smoother boundaries)
     * and shifts the adhesion/tension balance toward sorting-like compaction.
     * Default 0.0f preserves pre-Phase-4 behaviour.
     */
    struct Biomechanics
    {
        float repulsionStiffness  = 15.0f; // How hard cells push each other apart (Hertz force)
        float adhesionStrength    = 2.0f;  // How strongly cells stick together (JKR force)
        float maxRadius           = 1.5f;  // The interaction radius for collision detection
        float dampingCoefficient  = 0.0f;  // Velocity drag (dashpot). 0 = no damping.
        float corticalTension     = 0.0f;  // Actomyosin contractile term opposing adhesion (0 = off)
        // Phase 4.5-B — lateral (edge-to-edge) junctional adhesion. Hull
        // sub-sphere pairs contribute a translational force scaled by this
        // factor, pulling cells into face-to-face contact in proportion to the
        // number of hull-point pairs that overlap. Biological analog: cadherin
        // belt junctions (VE-cad, E-cad adherens belts) exert a translational
        // pull along the contact surface, not just a torque. Without this
        // mechanism, hull pairs only ALIGN tiles rotationally; tiles can drift
        // corner-to-corner without committing to a full edge-to-edge contact.
        // Scale ~0.05-0.2 is biologically plausible (lateral adhesion is a
        // fraction of the omnidirectional point-particle adhesion). 0 = off.
        float lateralAdhesionScale = 0.0f;
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

    /**
     * @brief Basement-membrane plate — static infinite plane anchorage surface.
     *
     * Supplies the two biologically essential ECM cues for endothelial
     * tubulogenesis without the cost of a full 3D ECM field:
     *   1. Contact repulsion — cells cannot penetrate the plane (integrin-rich
     *      basal surface is impermeable at the cell's mechanical scale).
     *   2. Integrin adhesion — cells within `anchorageDistance` of the plane
     *      are attracted toward it (focal-adhesion / integrin engagement).
     *   3. Polarity bias — anchored cells' polarity target shifts toward
     *      `planeNormal` (basal-toward-membrane polarity cue from integrin
     *      signalling).
     *
     * Biological analog: in vitro Matrigel tube formation assay (Kubota 1988,
     * Arnaoutova 2009) — a 2D basement-membrane-like gel substrate on which
     * ECs are seeded. In vivo analog: a pre-existing basal lamina that
     * endothelial cells aggregate upon.
     *
     * Does NOT model:
     *   - ECs depositing their own BM (phalanx-cell activity, later work).
     *   - MMP-driven BM degradation during sprouting (roadmap item 5).
     *   - Full 3D interstitial ECM (roadmap item 5).
     *
     * Global per simulation: one plate supported. All groups carrying this
     * behaviour share the same plate parameters (builder picks the first).
     */
    struct BasementMembrane
    {
        glm::vec3 planeNormal       = { 0.0f, 0.0f, 1.0f }; // unit outward normal (away from plate into bulk)
        float     height            = 0.0f;                 // plane origin along planeNormal (dot(planeOrigin, planeNormal))
        float     contactStiffness  = 15.0f;                // Hertz-like repulsion preventing penetration
        float     integrinAdhesion  = 1.5f;                 // JKR-like adhesion pulling cells toward plate
        float     anchorageDistance = 1.0f;                 // max distance at which integrin adhesion + polarity bias apply
        float     polarityBias      = 2.0f;                 // weight of plate polarity cue (overrides neighbour-centroid when strong)
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
        Behaviours::VesselSpring,
        Behaviours::BasementMembrane>;

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
