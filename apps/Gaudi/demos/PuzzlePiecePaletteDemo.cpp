#include "Demos.h"

#include <simulation/MorphologyGenerator.h>
#include <simulation/Phenotype.h>
#include <simulation/SpatialDistribution.h>

namespace Gaudi::Demos
{
    // Phase 2.2 visual palette. One AgentGroup, 4 cells laid out along X, each rendering
    // a different puzzle-piece primitive via the Phase 2.1 per-cell morphology-index
    // dispatch:
    //   x = -4.5   variant 0   CurvedTile      (Item 1 reference — arc-shaped tile)
    //   x = -1.5   variant 1   ElongatedQuad   (Davies 2009 — flow-aligned arterial EC)
    //   x = +1.5   variant 2   PentagonDefect  (Stone & Wales 1986 — +pi/3 curvature; narrower-side transitions)
    //   x = +4.5   variant 3   HeptagonDefect  (Stone & Wales 1986 — -pi/3 curvature; wider-side + bifurcation carinas)
    //
    // No physics, no behaviours — this is a static render verification that all four
    // primitives build valid meshes and dispatch correctly through build_indirect.comp.
    // Phases 2.3+ compose these variants onto vessel surfaces with adaptive ring counts
    // (Aird 2007) and Stone-Wales defect insertion at diameter transitions + bifurcations.
    void SetupPuzzlePiecePaletteDemo( DigitalTwin::SimulationBlueprint& blueprint )
    {
        blueprint.SetName( "Puzzle Piece Palette" );
        blueprint.SetDomainSize( glm::vec3( 20.0f ), 1.0f );

        blueprint.ConfigureSpatialPartitioning()
            .SetMethod( DigitalTwin::SpatialPartitioningMethod::HashGrid )
            .SetCellSize( 3.0f )
            .SetMaxDensity( 16 )
            .SetComputeHz( 60.0f );

        constexpr uint32_t kCount = 4;

        // 4 cells along X at Y=0, Z=0. Spacing of 3.0 keeps them visually separated.
        std::vector<glm::vec4> positions = {
            glm::vec4( -4.5f, 0.0f, 0.0f, 1.0f ), // variant 0 — CurvedTile
            glm::vec4( -1.5f, 0.0f, 0.0f, 1.0f ), // variant 1 — ElongatedQuad
            glm::vec4( +1.5f, 0.0f, 0.0f, 1.0f ), // variant 2 — PentagonDefect
            glm::vec4( +4.5f, 0.0f, 0.0f, 1.0f ), // variant 3 — HeptagonDefect
        };

        // Per-cell morphology indices packed into the upper 16 bits of cellType
        // (Phase 2.1 bit-packing convention; see Phenotype.h PackCellType).
        std::vector<uint32_t> cellTypes;
        cellTypes.reserve( kCount );
        for( uint32_t i = 0; i < kCount; ++i )
            cellTypes.push_back( DigitalTwin::PackCellType( DigitalTwin::CellType::Default, i ) );

        auto& group = blueprint.AddAgentGroup( "Puzzle-piece palette" )
                          .SetCount( kCount )
                          .SetDistribution( positions )
                          .SetInitialCellTypes( cellTypes )
                          .SetColor( glm::vec4( 0.85f, 0.65f, 0.40f, 1.0f ) );

        // Register four mesh variants under CellType::Default. Variant 0 becomes the
        // catch-all default; variants 1..3 are dispatched via PackCellType(Default, i).
        // Each primitive uses a visually similar footprint (≈ 1.2 x 1.2 units) so the
        // palette reads as a uniform row with four distinct shapes.
        group.SetMorphologyVariants(
            DigitalTwin::CellType::Default,
            {
                DigitalTwin::MorphologyGenerator::CreateCurvedTile( /*arc=*/60.0f, /*height=*/1.2f,
                                                                   /*thickness=*/0.2f, /*innerR=*/1.2f, /*sectors=*/4 ),
                DigitalTwin::MorphologyGenerator::CreateElongatedQuad( /*length=*/1.6f, /*width=*/0.8f,
                                                                       /*thickness=*/0.2f ),
                DigitalTwin::MorphologyGenerator::CreatePentagonDefect( /*radius=*/0.7f, /*thickness=*/0.2f ),
                DigitalTwin::MorphologyGenerator::CreateHeptagonDefect( /*radius=*/0.7f, /*thickness=*/0.2f ),
            } );
    }
} // namespace Gaudi::Demos
