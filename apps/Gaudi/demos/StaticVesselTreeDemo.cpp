#include "Demos.h"

#include <simulation/MorphologyGenerator.h>
#include <simulation/Phenotype.h>
#include <simulation/VesselTreeGenerator.h>

namespace Gaudi::Demos
{
    void SetupStaticVesselTreeDemo( DigitalTwin::SimulationBlueprint& blueprint )
    {
        // Static branching vessel tree: 2D ring topology, no biology, no VEGF.
        // Purpose: visually verify tree shape, ring connectivity, and structural stability.
        //
        // VesselTreeGenerator builds a trunk + 2-level branching tree. VesselSeed uploads
        // the full edge list (circumferential + axial + junction). VesselSpring keeps the
        // rings coherent. Biomechanics ensures vessel cells collide with each other.

        blueprint.SetName( "Static Vessel Tree" );
        blueprint.SetDomainSize( glm::vec3( 80.0f ), 2.0f );

        blueprint.ConfigureSpatialPartitioning()
            .SetMethod( DigitalTwin::SpatialPartitioningMethod::HashGrid )
            .SetCellSize( 4.0f )
            .SetMaxDensity( 64 )
            .SetComputeHz( 60.0f );

        // Trunk: ring=6, radius=1.5, spacing=1.5 → cell width ≈ 1.57 (auto-derived).
        // Child branches adapt ring size: depth-1 ≈ ring=4, depth-2 ≈ ring=3.
        // All cells are uniform scale=1.0; thickness comes from fewer cells per ring.
        // Per-edge rest lengths seeded from initial geometry → no global mismatch.
        auto tree = DigitalTwin::VesselTreeGenerator::BranchingTree()
            .SetOrigin( glm::vec3( -25.0f, 0.0f, 0.0f ) )
            .SetDirection( glm::vec3( 1.0f, 0.0f, 0.0f ) )
            .SetLength( 20.0f )
            .SetCellSpacing( 1.5f )
            .SetRingSize( 6 )
            .SetTubeRadius( 1.5f )
            .SetBranchingAngle( 35.0f )
            .SetBranchingDepth( 2 )
            .SetLengthFalloff( 0.65f )
            .SetAngleJitter( 8.0f )
            .SetTubeRadiusFalloff( 0.65f )
            .SetCurvature( 0.15f )
            .SetBranchTwist( 3.0f )
            .SetSeed( 42 )
            .Build();

        DigitalTwin::Behaviours::VesselSeed seed;
        seed.segmentCounts = tree.segmentCounts;
        seed.explicitEdges = tree.edges;

        auto& vessel = blueprint.AddAgentGroup( "VesselTree" )
            .SetCount( tree.totalCells )
            .SetMorphology( DigitalTwin::MorphologyGenerator::CreateCurvedTile( 60.0f, 1.35f, 0.25f, 1.5f ) )
            .SetDistribution( tree.positions )
            .SetOrientations( tree.normals )
            .SetColor( glm::vec4( 0.6f, 0.1f, 0.1f, 1.0f ) )
            .SetInitialCellType( static_cast<int>( DigitalTwin::CellType::PhalanxCell ) );

        vessel.AddBehaviour( seed );
        // anchorPhalanxCells=true: PhalanxCells are frozen. Per-edge rest lengths seeded
        // from geometry handle the mixed ring sizes. No JKR: re-added in Phase 3.

        DigitalTwin::Behaviours::VesselSpring spring{};
        spring.springStiffness    = 15.0f;
        spring.restingLength      = 1.5f; // fallback for runtime-appended edges
        spring.dampingCoefficient = 10.0f;
        spring.anchorPhalanxCells = true;
        vessel.AddBehaviour( spring ).SetHz( 60.0f );
    }
} // namespace Gaudi::Demos
