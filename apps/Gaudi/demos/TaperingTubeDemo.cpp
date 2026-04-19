#include "Demos.h"

#include <simulation/BiomechanicsGenerator.h>
#include <simulation/MorphologyGenerator.h>
#include <simulation/Phenotype.h>
#include <simulation/SpatialDistribution.h>
#include <simulation/VesselTreeGenerator.h>

namespace Gaudi::Demos
{
    // Phase 2.4.5 — gentle continuous taper rendered with rhombus tiles only.
    //
    // Previous Phase 2.4 attempt inserted Stone-Wales 5/7 defect polygons at every
    // ring-count transition; the result looked like "randomly placed heptagons" on
    // the tube surface (user visual verification 2026-04-19) because 5/7 defects
    // at continuous-taper transitions aren't genuine topological features — they're
    // just differently-shaped meshes at regular lattice positions that can't bridge
    // the real angular misalignment between consecutive rings.
    //
    // Phase 2.4.5 ships the honest interim solution: rhombus (diamond) tiles only,
    // gentle taper, `SetStoneWalesAtTaperTransitions(false)` default. Small angular-
    // misalignment gaps at each ring-count transition remain visible but are of the
    // order of ecWidth * 0.05 — much less intrusive than oversized polygon decorations.
    // The proper fix is Phase 2.5 dynamic topology (Voronoi tessellation from
    // neighbour contacts), which will close the gaps naturally.
    //
    // Biology: real arterial endothelial cells ARE rhomboid in shape (Davies 2009
    // DOI 10.1038/nrcardio.2009.14 — in vivo SEM shows fish-scale diamond
    // tessellation, not rectangular brick tiling). Stone-Wales 5/7 heptagons belong
    // at bifurcation carinas (Phase 2.5), not along continuous tapers.
    void SetupTaperingTubeDemo( DigitalTwin::SimulationBlueprint& blueprint )
    {
        blueprint.SetName( "Tapering Tube" );
        blueprint.SetDomainSize( glm::vec3( 40.0f ), 1.5f );

        blueprint.ConfigureSpatialPartitioning()
            .SetMethod( DigitalTwin::SpatialPartitioningMethod::HashGrid )
            .SetCellSize( 3.0f )
            .SetMaxDensity( 64 )
            .SetComputeHz( 60.0f );

        const float ecWidth          = 1.0f;
        const float aspect           = 1.5f;
        const float tubeRadiusWide   = 3.0f;  // ring ≈ round(2π·3.0) = 19
        const float tubeRadiusNarrow = 2.0f;  // ring ≈ round(2π·2.0) = 13
        const float length           = 36.0f; // axialStep = 1.5 → 25 rings
        // Medium taper over a long tube: 33% radius reduction (19 → 13 cells per ring,
        // ~6 transitions spread over 25 rings). Radial step per ring ≈ 0.042 — the two
        // known static-primitive-mesh limitations (gap bands at transitions, overlapping-
        // diamond bumpiness at per-ring radial steps) remain visible but are spread over
        // enough length that the overall vessel shape reads as a clean cone. Both
        // artefacts close automatically under Phase 2.5 dynamic topology.

        auto tree = DigitalTwin::VesselTreeGenerator::BranchingTree()
                        .SetOrigin( glm::vec3( -length * 0.5f, 0.0f, 0.0f ) )
                        .SetDirection( glm::vec3( 1.0f, 0.0f, 0.0f ) )
                        .SetLength( length )
                        .SetTubeRadius( tubeRadiusWide )
                        .SetTubeRadiusEnd( tubeRadiusNarrow )
                        .SetECCircumferentialWidth( ecWidth )
                        .SetCellAspectRatio( aspect )
                        .SetBranchingDepth( 0 )
                        .SetSeed( 42 )
                        .Build();

        auto& ecs = blueprint.AddAgentGroup( "Tapering Vessel Endothelium" );
        ecs.SetVesselTree( tree )
           .SetColor( glm::vec4( 0.88f, 0.42f, 0.46f, 1.0f ) );

        // Phase 2.4.5 — single rhombus variant for the entire tube (pentagon / heptagon
        // registration removed; defect cells never emitted while the flag is off).
        ecs.SetMorphologyVariants(
            DigitalTwin::CellType::Default,
            {
                DigitalTwin::MorphologyGenerator::CreateRhombus(
                    /*longDiagonal=*/ecWidth * aspect * 2.0f,
                    /*shortDiagonal=*/ecWidth,
                    /*thickness=*/0.2f ),
            } );

        auto jkr = DigitalTwin::BiomechanicsGenerator::JKR()
                       .SetYoungsModulus( 20.0f )
                       .SetPoissonRatio( 0.4f )
                       .SetAdhesionEnergy( 5.0f )
                       .SetMaxInteractionRadius( 0.75f )
                       .SetDampingCoefficient( 500.0f ) // quiescent-tube regime (Phase 2.3)
                       .SetCorticalTension( 0.5f )
                       .SetLateralAdhesionScale( 0.15f )
                       .Build();
        ecs.AddBehaviour( jkr ).SetHz( 60.0f );

        ecs.AddBehaviour( DigitalTwin::Behaviours::CadherinAdhesion{
                              glm::vec4( 0.0f, 0.0f, 1.0f, 0.0f ),
                              0.05f,  // expressionRate
                              0.001f, // degradationRate
                              2.0f,   // couplingStrength
                              2.0f,   // catchBondStrength (Rakshit 2012 VE-cad X-dimer)
                              0.3f    // catchBondPeakLoad
                          } )
            .SetHz( 60.0f );

        DigitalTwin::Behaviours::CellPolarity polarity;
        polarity.regulationRate      = 0.2f;
        polarity.apicalRepulsion     = 0.3f;
        polarity.basalAdhesion       = 1.5f;
        polarity.propagationStrength = 1.0f;
        ecs.AddBehaviour( polarity ).SetHz( 60.0f );

        ecs.AddBehaviour( DigitalTwin::Behaviours::BrownianMotion{ 0.02f } ).SetHz( 60.0f );
    }
} // namespace Gaudi::Demos
