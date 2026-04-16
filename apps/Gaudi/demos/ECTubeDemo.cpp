#include "Demos.h"

#include <simulation/BiomechanicsGenerator.h>

#include <glm/glm.hpp>

namespace Gaudi::Demos
{
    void SetupECTubeDemo( DigitalTwin::SimulationBlueprint& blueprint )
    {
        // ── EC tube, with ECM cue (plate lands in Phase 2) ────────────────────
        // Phase 1 scaffold. Identical to ECBlobDemo except for the name — the
        // two demos share SeedECCloud so they start from the exact same cloud
        // with the same random seed. From Phase 2 onward this demo will have
        // a BasementMembrane behaviour (supplying basal polarity cue + integrin
        // anchorage) and its behaviour evolves into a biological tube via cord
        // hollowing. In Phase 1 the demos are deliberately indistinguishable —
        // the visual divergence introduced by the plate in the next phase is
        // the scientific claim the demo pair exists to make.
        //
        // Biological analog (from Phase 2): the 2D Matrigel tube-formation assay
        // (Kubota 1988, Arnaoutova 2009) — the most widely used in-vitro
        // angiogenesis model.

        blueprint.SetName( "EC Tube" );
        blueprint.SetDomainSize( glm::vec3( 40.0f ), 1.5f );

        blueprint.ConfigureSpatialPartitioning()
            .SetMethod( DigitalTwin::SpatialPartitioningMethod::HashGrid )
            .SetCellSize( 3.0f )
            .SetMaxDensity( 64 )
            .SetComputeHz( 60.0f );

        auto& ecs = blueprint.AddAgentGroup( "Endothelial Cells" );
        SeedECCloud( ecs, 42 );

        // ── Biomechanics ───────────────────────────────────────────────────────
        // See ECBlobDemo.cpp for the JKR basin-width rationale (adhesion 5.0).
        auto jkr = DigitalTwin::BiomechanicsGenerator::JKR()
                       .SetYoungsModulus( 20.0f )
                       .SetPoissonRatio( 0.4f )
                       .SetAdhesionEnergy( 5.0f )
                       .SetMaxInteractionRadius( 0.75f )
                       .SetDampingCoefficient( 150.0f )
                       .Build();
        ecs.AddBehaviour( jkr ).SetHz( 60.0f );

        ecs.AddBehaviour( DigitalTwin::Behaviours::CadherinAdhesion{
                              glm::vec4( 0.0f, 0.0f, 1.0f, 0.0f ),
                              0.05f,
                              0.001f,
                              2.0f
                          } )
            .SetHz( 60.0f );

        DigitalTwin::Behaviours::CellPolarity polarity;
        polarity.regulationRate  = 0.2f;
        polarity.apicalRepulsion = 0.3f;
        polarity.basalAdhesion   = 1.5f;
        ecs.AddBehaviour( polarity ).SetHz( 60.0f );

        ecs.AddBehaviour( DigitalTwin::Behaviours::BrownianMotion{ 0.1f } ).SetHz( 60.0f );

        // NOTE: No BasementMembrane behaviour yet — introduced in Phase 2.
    }
} // namespace Gaudi::Demos
