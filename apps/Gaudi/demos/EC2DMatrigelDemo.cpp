#include "Demos.h"

#include <simulation/BiomechanicsGenerator.h>

#include <glm/glm.hpp>

namespace Gaudi::Demos
{
    void SetupEC2DMatrigelDemo( DigitalTwin::SimulationBlueprint& blueprint )
    {
        // ── EC on 2D Matrigel substrate (Kubota 1988, Arnaoutova 2009) ────────
        // Paired with ECBlobDemo (hanging-drop / ULA culture, no substrate).
        // Identical initial cell distribution via the shared SeedECCloud helper;
        // the ONLY divergence is the BasementMembrane plate and its anchorage
        // parameters, matching real experimental practice where the same cell
        // suspension is pipetted onto either plain medium (suspension) or a
        // Matrigel-coated dish.
        //
        // Expected phenotype (4-24 h in real assay): cells settle onto the
        // Matrigel surface, establish basal polarity toward the BM, and form
        // a MONOLAYER or cord-like network through lateral cadherin junctions.
        // NOT a hollow tube with lumen — that morphology requires 3D collagen
        // or fibrin gel (roadmap item 5), where cells are embedded in ECM on
        // all sides of a nascent cord, letting cord hollowing (Strilic 2009)
        // converge apical markers to an interior lumen.
        //
        // The demo is DELIBERATELY named EC2DMatrigelDemo (not ECTubeDemo) so
        // the biological expectation — monolayer/cord, not tube — is clear
        // from the name. A separate ECTubeDemo will land when 3D ECM arrives.

        blueprint.SetName( "EC 2D Matrigel" );
        blueprint.SetDomainSize( glm::vec3( 40.0f ), 1.5f );

        blueprint.ConfigureSpatialPartitioning()
            .SetMethod( DigitalTwin::SpatialPartitioningMethod::HashGrid )
            .SetCellSize( 3.0f )
            .SetMaxDensity( 64 )
            .SetComputeHz( 60.0f );

        auto& ecs = blueprint.AddAgentGroup( "Endothelial Cells" );
        SeedECCloud( ecs, 42 );

        // ── Biomechanics ───────────────────────────────────────────────────────
        // See ECBlobDemo.cpp for JKR basin-width rationale. corticalTension
        // matches ECBlobDemo (0.5) so the ONLY physics divergence between the
        // two demos remains the BasementMembrane plate — that's the positive/
        // negative control design. Tension is a cell-intrinsic mechanical
        // property that applies regardless of the ECM context.
        // Phase 4.5-B lateral adhesion matched to ECBlobDemo for the "only the
        // plate differs" invariant.
        auto jkr = DigitalTwin::BiomechanicsGenerator::JKR()
                       .SetYoungsModulus( 20.0f )
                       .SetPoissonRatio( 0.4f )
                       .SetAdhesionEnergy( 5.0f )
                       .SetMaxInteractionRadius( 0.75f )
                       .SetDampingCoefficient( 150.0f )
                       .SetCorticalTension( 0.5f )
                       .SetLateralAdhesionScale( 0.15f )
                       .Build();
        ecs.AddBehaviour( jkr ).SetHz( 60.0f );

        // Phase 5 — catch-bond parameters unified across all three demos.
        // Same VE-cadherin X-dimer kinetics (Rakshit 2012). In EC2DMatrigelDemo
        // the integrin pull produces tensile load on bottom-layer lateral
        // junctions as cells spread on the plate; the catch-bond strengthens
        // VE-cad there, stabilising the monolayer against basal-pull-induced
        // fracture while keeping the compressed-state baseline identical to
        // ECBlobDemo.
        ecs.AddBehaviour( DigitalTwin::Behaviours::CadherinAdhesion{
                              glm::vec4( 0.0f, 0.0f, 1.0f, 0.0f ),
                              0.05f,
                              0.001f,
                              2.0f,
                              2.0f,   // catchBondStrength
                              0.3f    // catchBondPeakLoad
                          } )
            .SetHz( 60.0f );

        // ── CellPolarity (Step A — unified cell behaviours across all demos) ──
        //
        // SAME CellPolarity parameters as ECBlobDemo and future ECTubeDemo.
        // Biological principle: ECs have intrinsic PAR/Crumbs + VE-cadherin
        // polarity machinery regardless of substrate. What differs between
        // demos is the ENVIRONMENT, not the cell's molecular kit.
        //
        // In EC2DMatrigelDemo specifically: plate at y=0 provides a BM cue →
        // cells within anchorageDistance polarise basal-toward-plate. With
        // `propagationStrength = 1`, that polarity cascades upward through
        // the cluster via cell-cell junctions (PAR/Crumbs cascade,
        // St Johnston 2010). Post-Step-A the polarity magnitude is gated
        // by plate+propagation only (no spurious magnitude from geometric
        // centroid) — this is what makes the identical-cell design work:
        // in ECBlob the lack of plate seed means no cell ever polarises,
        // while here the plate seeds a cascade.
        //
        // Apical = 0.3 (conservative 2D value) keeps monolayer cohesive.
        // Phase 6 sweep will tune apical/basal for the 3D demo.
        DigitalTwin::Behaviours::CellPolarity polarity;
        polarity.regulationRate      = 0.2f;
        polarity.apicalRepulsion     = 0.3f;
        polarity.basalAdhesion       = 1.5f;
        polarity.propagationStrength = 1.0f;
        ecs.AddBehaviour( polarity ).SetHz( 60.0f );

        ecs.AddBehaviour( DigitalTwin::Behaviours::BrownianMotion{ 0.1f } ).SetHz( 60.0f );

        // ── Basement membrane (Phase 2, anchorage range revised Phase 4.5-B) ──
        // 2D Matrigel plate lying in the XZ plane at y=0, outward normal +Y.
        //
        // Mechanism provided:
        //   1. Contact repulsion — cells cannot penetrate the substrate.
        //   2. Integrin adhesion — cells within anchorageDistance are pulled
        //      toward the plate. Parabolic shape peaking at aD/2.
        //   3. Polarity bias — anchored cells' apico-basal axis aligns with
        //      +Y (basal side toward plate).
        //
        // Phase 4.5-B: anchorageDistance 1.0 → 4.0. In real biology Matrigel's
        // BM-mimetic substrate emits soluble laminin peptides, sequestered
        // VEGF, FGF, HGF, etc. — a chemoattractive zone that extends several
        // cell diameters into the medium above the gel (Kleinman &
        // Martin 2005; Hughes et al. 2010 Matrigel proteomic profile). Cells
        // in a freshly pipetted drop feel the BM signal throughout the drop
        // volume, not only at physical contact. With aD = 4.0 the whole
        // initial cluster (y ≈ 0.5 to 4.5) is within the integrin-chemotactic
        // zone from frame 1 → cells settle onto the substrate smoothly rather
        // than first forming a 3D blob above it. The parabolic integrin pull
        // still peaks at aD/2 = 2.0 (physical-contact region stays strongest);
        // cells higher up feel a weaker but non-zero attraction, modelling
        // the diffused chemokine gradient.
        DigitalTwin::Behaviours::BasementMembrane plate;
        plate.planeNormal       = glm::vec3( 0.0f, 1.0f, 0.0f );
        plate.height            = 0.0f;
        plate.contactStiffness  = 15.0f;
        plate.integrinAdhesion  = 1.5f;
        plate.anchorageDistance = 2.8f;  // Covers the whole initial cluster
        plate.polarityBias      = 2.0f;
        ecs.AddBehaviour( plate ).SetHz( 60.0f );
    }
} // namespace Gaudi::Demos
