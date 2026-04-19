#include "Demos.h"

#include <simulation/MorphologyGenerator.h>
#include <simulation/Phenotype.h>
#include <simulation/SpatialDistribution.h>

namespace Gaudi::Demos
{
    // Phase 2.1 plumbing verification demo. Single AgentGroup containing 50
    // cells split 50/50 across two mesh variants — half render as CreateDisc,
    // half as CreateCurvedTile — proving the bit-packed morphology-index
    // dispatch pipeline end-to-end (Phenotype.h PackCellType → AgentGroup
    // SetMorphologyVariants → SimulationBuilder multi-DrawMeta emission →
    // build_indirect.comp variant matching).
    //
    // Biologically this is nothing — it's engine plumbing only. The proper
    // per-cell morphology biology (elongated rhomboid arterial ECs + pentagon/
    // heptagon Stone-Wales defects at diameter transitions and bifurcations;
    // Aird 2007, Davies 2009, Stone & Wales 1986) lands in Phases 2.2+.
    void SetupTwoShapeDemo( DigitalTwin::SimulationBlueprint& blueprint )
    {
        blueprint.SetName( "Two Shape" );
        blueprint.SetDomainSize( glm::vec3( 20.0f ), 1.0f );

        blueprint.ConfigureSpatialPartitioning()
            .SetMethod( DigitalTwin::SpatialPartitioningMethod::HashGrid )
            .SetCellSize( 3.0f )
            .SetMaxDensity( 64 )
            .SetComputeHz( 60.0f );

        constexpr uint32_t kCount = 50;

        // Layout: two parallel rows along X. Row 1 (z = -2) = variant 0 (disc),
        // row 2 (z = +2) = variant 1 (curved tile). Visual separation makes the
        // dispatch correctness obvious at a glance.
        std::vector<glm::vec4> positions;
        positions.reserve( kCount );
        for( uint32_t i = 0; i < kCount; ++i )
        {
            const float x  = -8.0f + static_cast<float>( i % 25 ) * 0.7f;
            const float z  = ( i < 25 ) ? -2.0f : +2.0f;
            positions.push_back( glm::vec4( x, 0.0f, z, 1.0f ) );
        }

        // Per-cell bit-packed cellType tags (Phase 2.1 plumbing):
        //   first 25 cells (z = -2 row)  → morphIdx 0 (flat disc, cellType = 0)
        //   next  25 cells (z = +2 row)  → morphIdx 1 (curved tile)
        std::vector<uint32_t> cellTypes;
        cellTypes.reserve( kCount );
        for( uint32_t i = 0; i < kCount; ++i )
        {
            uint32_t morphIdx = ( i < 25 ) ? 0u : 1u;
            cellTypes.push_back( DigitalTwin::PackCellType( DigitalTwin::CellType::Default, morphIdx ) );
        }

        auto& group = blueprint.AddAgentGroup( "Mixed-morphology cells" )
                          .SetCount( kCount )
                          .SetDistribution( positions )
                          .SetInitialCellTypes( cellTypes )
                          .SetColor( glm::vec4( 0.65f, 0.75f, 0.95f, 1.0f ) );

        // Register two mesh variants for CellType::Default. Variant 0 becomes the
        // catch-all default (rendered for cells whose cellType packs morphIdx=0);
        // variant 1 is dispatched to cells whose cellType = PackCellType(Default, 1).
        group.SetMorphologyVariants(
            DigitalTwin::CellType::Default,
            {
                DigitalTwin::MorphologyGenerator::CreateDisc( 0.6f, 0.15f, 16 ),                    // variant 0 — flat disc
                DigitalTwin::MorphologyGenerator::CreateCurvedTile( 60.0f, 1.0f, 0.15f, 1.5f, 4 ), // variant 1 — curved shell
            } );
    }
} // namespace Gaudi::Demos
