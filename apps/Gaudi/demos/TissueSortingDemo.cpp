#include "Demos.h"

#include <simulation/BiomechanicsGenerator.h>
#include <simulation/MorphologyGenerator.h>
#include <simulation/SpatialDistribution.h>

#include <algorithm>
#include <random>

namespace Gaudi::Demos
{
    void SetupTissueSortingDemo( DigitalTwin::SimulationBlueprint& blueprint )
    {
        // Two cell populations — Epithelial (E-cadherin) and Mesenchymal (N-cadherin) —
        // start randomly mixed on a dense lattice. Homophilic cadherin adhesion (identity
        // affinity matrix) drives spontaneous sorting: same-type cells stick together,
        // different types don't. Over ~30-60s the mixed cluster segregates into homotypic
        // domains with a sharp boundary.
        //
        // Based on the Differential Adhesion Hypothesis (Steinberg, 1963).
        blueprint.SetName( "Tissue Sorting" );
        blueprint.SetDomainSize( glm::vec3( 60.0f ), 1.5f );

        blueprint.ConfigureSpatialPartitioning()
            .SetMethod( DigitalTwin::SpatialPartitioningMethod::HashGrid )
            .SetCellSize( 3.0f )
            .SetMaxDensity( 64 )
            .SetComputeHz( 60.0f );

        // ── Generate lattice positions inside a sphere ──────────────────────────
        // spacing must be >= 2×maxInteractionRadius so cells start with zero overlap.
        const float cellRadius  = 0.6f;
        const float physRadius  = 0.65f;  // JKR interaction radius (≈ cell render radius)
        const float spacing     = 1.35f;  // > 2 × physRadius → tiny gap, no initial repulsion

        auto allPositions = DigitalTwin::SpatialDistribution::LatticeInSphere( spacing, 8.0f );

        // Shuffle and split evenly between the two groups.
        std::mt19937 rng( 42 );
        std::shuffle( allPositions.begin(), allPositions.end(), rng );

        uint32_t half = static_cast<uint32_t>( allPositions.size() ) / 2;
        std::vector<glm::vec4> epiPositions( allPositions.begin(), allPositions.begin() + half );
        std::vector<glm::vec4> mesPositions( allPositions.begin() + half, allPositions.begin() + 2 * half );

        // maxInteractionRadius matches cell size (0.65) → two cells overlap only
        // when distance < 1.3, which the lattice avoids. Adhesion pulls neighbors
        // into gentle contact; cadherin coupling (×3) makes same-type stick.
        auto jkr = DigitalTwin::BiomechanicsGenerator::JKR()
                       .SetYoungsModulus( 15.0f )
                       .SetPoissonRatio( 0.4f )
                       .SetAdhesionEnergy( 2.0f )
                       .SetMaxInteractionRadius( physRadius )
                       .SetDampingCoefficient( 200.0f )
                       .Build();

        // ── Epithelial cells (red-orange) — express E-cadherin (channel x) ──────────────
        auto& epi = blueprint.AddAgentGroup( "Epithelial" )
                        .SetCount( half )
                        .SetMorphology( DigitalTwin::MorphologyGenerator::CreateSphere( cellRadius ) )
                        .SetDistribution( epiPositions )
                        .SetColor( glm::vec4( 0.95f, 0.35f, 0.15f, 1.0f ) );

        epi.AddBehaviour( jkr ).SetHz( 60.0f );
        epi.AddBehaviour( DigitalTwin::Behaviours::CadherinAdhesion{
                              glm::vec4( 1.0f, 0.0f, 0.0f, 0.0f ),
                              0.05f, 0.001f, 3.0f } )
            .SetHz( 60.0f );
        epi.AddBehaviour( DigitalTwin::Behaviours::BrownianMotion{ 0.5f } ).SetHz( 60.0f );

        // ── Mesenchymal cells (blue) — express N-cadherin (channel y) ───────────────────
        auto& mes = blueprint.AddAgentGroup( "Mesenchymal" )
                        .SetCount( half )
                        .SetMorphology( DigitalTwin::MorphologyGenerator::CreateSphere( cellRadius ) )
                        .SetDistribution( mesPositions )
                        .SetColor( glm::vec4( 0.2f, 0.45f, 1.0f, 1.0f ) );

        mes.AddBehaviour( jkr ).SetHz( 60.0f );
        mes.AddBehaviour( DigitalTwin::Behaviours::CadherinAdhesion{
                              glm::vec4( 0.0f, 1.0f, 0.0f, 0.0f ),
                              0.05f, 0.001f, 3.0f } )
            .SetHz( 60.0f );
        mes.AddBehaviour( DigitalTwin::Behaviours::BrownianMotion{ 0.4f } ).SetHz( 60.0f );
    }
} // namespace Gaudi::Demos
