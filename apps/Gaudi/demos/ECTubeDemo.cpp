#include "Demos.h"

#include <simulation/BiomechanicsGenerator.h>

#include <glm/glm.hpp>

namespace Gaudi::Demos
{
    void SetupECTubeDemo( DigitalTwin::SimulationBlueprint& blueprint )
    {
        // ── EC in 3D-ECM placeholder channel (Step C) ─────────────────────────
        // Third demo of the biological trilogy:
        //   - ECBlobDemo      — hanging drop / suspension (no BM) → spheroid
        //   - EC2DMatrigelDemo — 2D Matrigel (one plate)          → monolayer/cord
        //   - ECTubeDemo       — 3D-ECM placeholder (4 plates)    → cord/tube candidate
        //
        // All three demos use IDENTICAL cell behaviours (same initial seed, same
        // Biomechanics, Cadherin, CellPolarity parameters). Phenotype divergence
        // emerges purely from the ENVIRONMENT — here, four plates forming a
        // rectangular channel along +X around the cluster. This is the closest
        // we can get to 3D collagen gel biology with the Phase-2 BasementMembrane
        // primitive. Cells inside the channel have BM contact on 2-4 sides,
        // providing the geometric context for cord hollowing (Strilic 2009):
        // apical polarity converges to the channel interior, basal polarity
        // points outward to the nearest wall.
        //
        // This demo is DELIBERATELY imperfect at this phase. The plates are flat
        // rather than forming a true cylindrical BM; the channel is rectangular
        // rather than circular; Phase 5 (catch-bond) has not yet landed to
        // stabilise the cord under stress. Expected phenotype: elongated mass
        // along +X with cells pressed against channel walls, potentially with
        // partial interior polarity convergence. Phase 6 sweep will tune the
        // parameters (apical repulsion, cortical tension, catch-bond strength)
        // to produce a recognisable cord/tube morphology.
        //
        // A true `ECCollagenGel3DDemo` with volumetric ECM (roadmap item 5)
        // will eventually replace this placeholder.

        blueprint.SetName( "EC Tube (3D ECM placeholder)" );
        blueprint.SetDomainSize( glm::vec3( 40.0f ), 1.5f );

        blueprint.ConfigureSpatialPartitioning()
            .SetMethod( DigitalTwin::SpatialPartitioningMethod::HashGrid )
            .SetCellSize( 3.0f )
            .SetMaxDensity( 64 )
            .SetComputeHz( 60.0f );

        auto& ecs = blueprint.AddAgentGroup( "Endothelial Cells" );
        SeedECCloud( ecs, 42 );

        // ── Biomechanics (identical to ECBlob / EC2DMatrigel) ─────────────────
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

        // VE-cadherin — identical to other demos
        ecs.AddBehaviour( DigitalTwin::Behaviours::CadherinAdhesion{
                              glm::vec4( 0.0f, 0.0f, 1.0f, 0.0f ),
                              0.05f,
                              0.001f,
                              2.0f
                          } )
            .SetHz( 60.0f );

        // CellPolarity — identical to other demos. Conservative apical=0.3 for
        // now. Phase 6 sweep will explore -1.0 (Strilic cord-hollowing value)
        // against the 4-plate context to see if the 3D ECM placeholder is
        // sufficient to produce cavity opening without Phase 5 catch-bond.
        DigitalTwin::Behaviours::CellPolarity polarity;
        polarity.regulationRate      = 0.2f;
        polarity.apicalRepulsion     = 0.3f;
        polarity.basalAdhesion       = 1.5f;
        polarity.propagationStrength = 1.0f;
        ecs.AddBehaviour( polarity ).SetHz( 60.0f );

        ecs.AddBehaviour( DigitalTwin::Behaviours::BrownianMotion{ 0.1f } ).SetHz( 60.0f );

        // ── 4-plate channel forming a 3D-ECM placeholder ──────────────────────
        // Cluster geometry (from SeedECCloud): elongated cylinder along +X,
        // r=2.0, halfLength=6.0, centred at y=2.5. Cells span x ∈ [-6, 6],
        // y ∈ [0.5, 4.5], z ∈ [-2, 2].
        //
        // Channel: four flat plates framing the cluster on Y and Z axes, OPEN
        // along X so the cord has a direction to form. Each plate has
        // anchorageDistance large enough to reach cluster centre (Matrigel
        // chemoattractive zone, Kleinman & Martin 2005) and polarityBias
        // strong enough to orient anchored cells' basal vectors outward.
        //
        // Plate 1 (floor):     y =  0, normal = +Y
        // Plate 2 (ceiling):   y =  5, normal = -Y
        // Plate 3 (left  Z-):  z = -3, normal = +Z
        // Plate 4 (right Z+):  z =  3, normal = -Z
        //
        // A cell at cluster centre (0, 2.5, 0) has d = 2.5 to each of Y plates
        // (floor above, ceiling below in metric), d = 3.0 to each Z plate.
        // With anchorageDistance = 4.0, all plates reach all cluster cells.
        // Opposing plates' basal cues CANCEL direction at cluster centre
        // (floor pulls -Y, ceiling pulls +Y) while both contribute to
        // magnitude — so interior cells end up polarised (w > 0) but with
        // zero direction, waiting for propagation / neighbour influence to
        // resolve. Cells closer to one wall feel that wall more strongly and
        // polarise basal-toward-that-wall, correctly.

        auto addPlate = [&]( glm::vec3 normal, float height )
        {
            DigitalTwin::Behaviours::BasementMembrane plate;
            plate.planeNormal       = normal;
            plate.height            = height;
            plate.contactStiffness  = 15.0f;
            plate.integrinAdhesion  = 1.5f;
            plate.anchorageDistance = 4.0f;
            plate.polarityBias      = 2.0f;
            ecs.AddBehaviour( plate ).SetHz( 60.0f );
        };

        addPlate( glm::vec3(  0.0f,  1.0f,  0.0f ),  0.0f );  // floor
        addPlate( glm::vec3(  0.0f, -1.0f,  0.0f ), -5.0f );  // ceiling (plane y = 5)
        addPlate( glm::vec3(  0.0f,  0.0f,  1.0f ), -3.0f );  // z- wall  (plane z = -3)
        addPlate( glm::vec3(  0.0f,  0.0f, -1.0f ), -3.0f );  // z+ wall  (plane z = +3)
    }
} // namespace Gaudi::Demos
