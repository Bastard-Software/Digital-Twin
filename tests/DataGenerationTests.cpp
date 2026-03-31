#include "simulation/AgentGroup.h"
#include "simulation/MorphologyGenerator.h"
#include "simulation/SpatialDistribution.h"
#include "simulation/VesselTreeGenerator.h"
#include <gtest/gtest.h>
#include <glm/glm.hpp>
#include <algorithm>
#include <numeric>

using namespace DigitalTwin;

// =================================================================================================
// SpatialDistribution — CPU-only
// =================================================================================================

TEST( SpatialDistribution_VesselLine, EvenSpacing )
{
    const uint32_t  count = 5;
    const glm::vec3 start( -10.0f, 5.0f, 0.0f );
    const glm::vec3 end( 10.0f, 5.0f, 0.0f );

    auto positions = SpatialDistribution::VesselLine( count, start, end );

    ASSERT_EQ( positions.size(), count );

    EXPECT_FLOAT_EQ( positions.front().x, start.x );
    EXPECT_FLOAT_EQ( positions.front().y, start.y );
    EXPECT_FLOAT_EQ( positions.front().z, start.z );

    EXPECT_FLOAT_EQ( positions.back().x, end.x );
    EXPECT_FLOAT_EQ( positions.back().y, end.y );
    EXPECT_FLOAT_EQ( positions.back().z, end.z );

    for( const auto& p : positions )
        EXPECT_FLOAT_EQ( p.w, 1.0f );

    float expectedSpacing = glm::length( end - start ) / static_cast<float>( count - 1 );
    for( uint32_t i = 1; i < count; ++i )
    {
        float dist = glm::length( glm::vec3( positions[ i ] ) - glm::vec3( positions[ i - 1 ] ) );
        EXPECT_NEAR( dist, expectedSpacing, 1e-5f );
    }
}

TEST( SpatialDistribution_VesselLine, SingleAgent )
{
    const glm::vec3 start( -5.0f, 0.0f, 0.0f );
    const glm::vec3 end( 5.0f, 0.0f, 0.0f );

    auto positions = SpatialDistribution::VesselLine( 1, start, end );

    ASSERT_EQ( positions.size(), 1u );
    EXPECT_FLOAT_EQ( positions[ 0 ].x, 0.0f ); // midpoint
    EXPECT_FLOAT_EQ( positions[ 0 ].y, 0.0f );
    EXPECT_FLOAT_EQ( positions[ 0 ].z, 0.0f );
    EXPECT_FLOAT_EQ( positions[ 0 ].w, 1.0f );
}

TEST( SpatialDistribution_VesselLine, FixedSpacing )
{
    const glm::vec3 start( 0.0f, 0.0f, 0.0f );
    const glm::vec3 end( 10.0f, 0.0f, 0.0f );

    // spacing=3.0 on a 10-unit line fits 4 cells at x=0,3,6,9
    auto positions = SpatialDistribution::VesselLine( 100, start, end, 3.0f );

    ASSERT_EQ( positions.size(), 4u );
    EXPECT_FLOAT_EQ( positions[ 0 ].x, 0.0f );
    EXPECT_FLOAT_EQ( positions[ 1 ].x, 3.0f );
    EXPECT_FLOAT_EQ( positions[ 2 ].x, 6.0f );
    EXPECT_FLOAT_EQ( positions[ 3 ].x, 9.0f );
}

// =================================================================================================
// MorphologyGenerator — CPU-only
// =================================================================================================

TEST( MorphologyGeneratorTest, CreateCylinder_ValidMesh )
{
    MorphologyData m = MorphologyGenerator::CreateCylinder( 1.0f, 2.0f, 18 );
    ASSERT_FALSE( m.vertices.empty() );
    ASSERT_FALSE( m.indices.empty() );
    EXPECT_EQ( m.indices.size() % 3, 0u ) << "Indices must be triangle list";
    for( uint32_t idx : m.indices )
        EXPECT_LT( idx, static_cast<uint32_t>( m.vertices.size() ) ) << "Index out of range";
    for( const auto& v : m.vertices )
    {
        EXPECT_FLOAT_EQ( v.pos.w, 1.0f );
        EXPECT_FLOAT_EQ( v.normal.w, 0.0f );
    }
}

TEST( MorphologyGeneratorTest, CreateSpikySphere_ValidMesh )
{
    MorphologyData m = MorphologyGenerator::CreateSpikySphere( 1.0f, 1.4f, 16, 8 );
    ASSERT_FALSE( m.vertices.empty() );
    ASSERT_FALSE( m.indices.empty() );
    EXPECT_EQ( m.indices.size() % 3, 0u ) << "Indices must be triangle list";
    for( uint32_t idx : m.indices )
        EXPECT_LT( idx, static_cast<uint32_t>( m.vertices.size() ) ) << "Index out of range";
    bool hasSpike = false;
    for( const auto& v : m.vertices )
    {
        if( glm::length( glm::vec3( v.pos ) ) > 1.05f )
            hasSpike = true;
    }
    EXPECT_TRUE( hasSpike ) << "CreateSpikySphere must produce vertices beyond base radius";
}

// =================================================================================================
// AgentGroup cell-type morphologies — CPU-only
// =================================================================================================

TEST( AgentGroupTest, AddCellTypeMorphology_StoresEntries )
{
    AgentGroup group( "EndothelialCells" );
    group.AddCellTypeMorphology( 1, MorphologyGenerator::CreateSpikySphere() );  // TipCell
    group.AddCellTypeMorphology( 2, MorphologyGenerator::CreateCylinder() );     // StalkCell

    const auto& entries = group.GetCellTypeMorphologies();
    ASSERT_EQ( entries.size(), 2u );
    EXPECT_EQ( entries[ 0 ].cellTypeIndex, 1 );
    EXPECT_FALSE( entries[ 0 ].mesh.vertices.empty() );
    EXPECT_EQ( entries[ 1 ].cellTypeIndex, 2 );
    EXPECT_FALSE( entries[ 1 ].mesh.vertices.empty() );
}

// Per-cell-type color: the 3-argument overload stores the color; the 2-argument overload
// defaults to color.x < 0 (sentinel meaning "use base group color").
TEST( AgentGroupTest, AddCellTypeMorphology_WithColor_StoresColor )
{
    AgentGroup group( "EndothelialCells" );
    glm::vec4 tipGreen( 0.1f, 0.9f, 0.2f, 1.0f );
    glm::vec4 stalkYellow( 1.0f, 0.8f, 0.1f, 1.0f );

    group.AddCellTypeMorphology( 1, MorphologyGenerator::CreateSpikySphere(), tipGreen );   // with color
    group.AddCellTypeMorphology( 2, MorphologyGenerator::CreateCylinder() );                 // no color → sentinel

    const auto& entries = group.GetCellTypeMorphologies();
    ASSERT_EQ( entries.size(), 2u );

    // Explicit color is stored correctly
    EXPECT_FLOAT_EQ( entries[ 0 ].color.r, tipGreen.r );
    EXPECT_FLOAT_EQ( entries[ 0 ].color.g, tipGreen.g );
    EXPECT_FLOAT_EQ( entries[ 0 ].color.b, tipGreen.b );
    EXPECT_GE( entries[ 0 ].color.x, 0.0f ) << "Positive color.x means override is active";

    // No color → sentinel value (color.x < 0 → use base group color)
    EXPECT_LT( entries[ 1 ].color.x, 0.0f ) << "Default color must be sentinel (< 0)";
}

// =================================================================================================
// MorphologyGenerator::CreateDisc — CPU-only
// =================================================================================================

TEST( MorphologyGeneratorTest, CreateDisc_ValidMesh )
{
    MorphologyData m = MorphologyGenerator::CreateDisc( 0.8f, 0.2f, 16 );
    EXPECT_FALSE( m.vertices.empty() ) << "Disc must have vertices";
    EXPECT_FALSE( m.indices.empty() )  << "Disc must have indices";
    EXPECT_EQ( m.indices.size() % 3, 0u ) << "Index count must be a multiple of 3 (triangles)";

    // All w components of positions must be 1.0
    for( const auto& v : m.vertices )
        EXPECT_FLOAT_EQ( v.pos.w, 1.0f );
}

TEST( MorphologyGeneratorTest, CreateDisc_NormalsUnitLength )
{
    MorphologyData m = MorphologyGenerator::CreateDisc( 0.8f, 0.2f, 16 );
    for( const auto& v : m.vertices )
    {
        float len = glm::length( glm::vec3( v.normal ) );
        EXPECT_NEAR( len, 1.0f, 1e-5f ) << "All normals must be unit length";
    }
}

TEST( MorphologyGeneratorTest, CreateTile_ValidMesh )
{
    MorphologyData m = MorphologyGenerator::CreateTile( 1.4f, 1.2f, 0.2f );
    EXPECT_EQ( m.vertices.size(), 24u ) << "Tile must have 24 vertices (4 per face × 6 faces)";
    EXPECT_EQ( m.indices.size(),  36u ) << "Tile must have 36 indices (2 triangles per face × 6 faces)";
    EXPECT_EQ( m.indices.size() % 3, 0u );
    for( uint32_t idx : m.indices )
        EXPECT_LT( idx, static_cast<uint32_t>( m.vertices.size() ) ) << "Index out of range";
    for( const auto& v : m.vertices )
        EXPECT_FLOAT_EQ( v.pos.w, 1.0f );
}

TEST( MorphologyGeneratorTest, CreateTile_Dimensions )
{
    const float w = 1.6f, h = 1.2f, t = 0.3f;
    MorphologyData m = MorphologyGenerator::CreateTile( w, h, t );

    float minX = 1e9f, maxX = -1e9f;
    float minY = 1e9f, maxY = -1e9f;
    float minZ = 1e9f, maxZ = -1e9f;
    for( const auto& v : m.vertices )
    {
        minX = std::min( minX, v.pos.x ); maxX = std::max( maxX, v.pos.x );
        minY = std::min( minY, v.pos.y ); maxY = std::max( maxY, v.pos.y );
        minZ = std::min( minZ, v.pos.z ); maxZ = std::max( maxZ, v.pos.z );
    }
    EXPECT_NEAR( maxX - minX, w, 1e-5f ) << "Tile width (X extent) must match parameter";
    EXPECT_NEAR( maxY - minY, t, 1e-5f ) << "Tile thickness (Y extent) must match parameter";
    EXPECT_NEAR( maxZ - minZ, h, 1e-5f ) << "Tile height (Z extent) must match parameter";
}

TEST( MorphologyGeneratorTest, CreateTile_NormalsUnitLength )
{
    MorphologyData m = MorphologyGenerator::CreateTile( 1.4f, 1.2f, 0.2f );
    for( const auto& v : m.vertices )
    {
        float len = glm::length( glm::vec3( v.normal ) );
        EXPECT_NEAR( len, 1.0f, 1e-5f );
    }
}

// =================================================================================================
// VesselTreeGenerator — CPU-only
// =================================================================================================

// Single trunk (depth=0, no branching) — exactly numRings*ringSize cells, one segment count
TEST( VesselTreeGeneratorTest, SingleTrunk_NoBranching )
{
    auto result = VesselTreeGenerator::BranchingTree()
        .SetOrigin( { 0, 0, 0 } )
        .SetDirection( { 1, 0, 0 } )
        .SetLength( 8.0f )
        .SetCellSpacing( 2.0f )     // 5 rings (0,2,4,6,8)
        .SetRingSize( 6 )
        .SetTubeRadius( 1.5f )
        .SetBranchingDepth( 0 )
        .SetSeed( 1 )
        .Build();

    const uint32_t expectedRings = 5u; // length/spacing + 1 = 8/2 + 1 = 5
    const uint32_t expectedCells = expectedRings * 6u;

    EXPECT_EQ( result.totalCells, expectedCells );
    EXPECT_EQ( result.positions.size(), expectedCells );
    EXPECT_EQ( result.normals.size(), expectedCells );
    EXPECT_EQ( result.segmentCounts.size(), 1u )     << "One segment per branch";
    EXPECT_EQ( result.segmentCounts[ 0 ], expectedCells );
}

// Depth=1 branching: trunk + 2 children = 3 segment count entries
TEST( VesselTreeGeneratorTest, OneLevelBranch_ThreeSegments )
{
    auto result = VesselTreeGenerator::BranchingTree()
        .SetLength( 10.0f )
        .SetCellSpacing( 2.0f )
        .SetRingSize( 4 )
        .SetBranchingDepth( 1 )
        .SetBranchProbability( 1.0f )
        .SetSeed( 42 )
        .Build();

    EXPECT_EQ( result.segmentCounts.size(), 3u ) << "Trunk + 2 children = 3 segments";
    EXPECT_GT( result.totalCells, 0u );

    uint32_t sum = std::accumulate( result.segmentCounts.begin(), result.segmentCounts.end(), 0u );
    EXPECT_EQ( sum, result.totalCells ) << "segmentCounts must sum to totalCells";
}

// Segment counts sum invariant holds for depth=2
TEST( VesselTreeGeneratorTest, SegmentCountsSumToTotal )
{
    auto result = VesselTreeGenerator::BranchingTree()
        .SetLength( 8.0f )
        .SetCellSpacing( 2.0f )
        .SetRingSize( 6 )
        .SetBranchingDepth( 2 )
        .SetSeed( 7 )
        .Build();

    uint32_t sum = std::accumulate( result.segmentCounts.begin(), result.segmentCounts.end(), 0u );
    EXPECT_EQ( sum, result.totalCells );
}

// Circumferential edges form closed loops: every ring has exactly ringSize edges
TEST( VesselTreeGeneratorTest, RingTopology_CircumferentialEdges )
{
    const uint32_t ringSize = 6u;
    auto result = VesselTreeGenerator::BranchingTree()
        .SetLength( 4.0f )
        .SetCellSpacing( 2.0f ) // 3 rings = 18 cells
        .SetRingSize( ringSize )
        .SetBranchingDepth( 0 )
        .SetSeed( 1 )
        .Build();

    // Count edges where both endpoints are in the same ring (both within [base, base+ringSize))
    uint32_t circumEdges = 0;
    uint32_t numRings    = result.totalCells / ringSize;
    for( const auto& [a, b] : result.edges )
    {
        uint32_t ra = a / ringSize;
        uint32_t rb = b / ringSize;
        if( ra == rb )
            ++circumEdges;
    }
    EXPECT_EQ( circumEdges, numRings * ringSize ) << "Each ring must have exactly ringSize circumferential edges";
}

// Axial edges connect ring[r][j] to ring[r+1][j] for all r, j
TEST( VesselTreeGeneratorTest, AxialEdges_ConnectAdjacentRings )
{
    const uint32_t ringSize = 6u;
    auto result = VesselTreeGenerator::BranchingTree()
        .SetLength( 4.0f )
        .SetCellSpacing( 2.0f ) // 3 rings
        .SetRingSize( ringSize )
        .SetBranchingDepth( 0 )
        .SetSeed( 1 )
        .Build();

    uint32_t numRings  = result.totalCells / ringSize;
    uint32_t axialEdges = 0;
    for( const auto& [a, b] : result.edges )
    {
        uint32_t ra = a / ringSize;
        uint32_t rb = b / ringSize;
        if( ra != rb && a % ringSize == b % ringSize )
            ++axialEdges;
    }
    EXPECT_EQ( axialEdges, ( numRings - 1 ) * ringSize ) << "Axial edges: (numRings-1)*ringSize";
}

// Normals point outward: dot(normal, radial direction from centerline) must be > 0
TEST( VesselTreeGeneratorTest, Normals_PointOutward )
{
    auto result = VesselTreeGenerator::BranchingTree()
        .SetOrigin( { 0, 0, 0 } )
        .SetDirection( { 1, 0, 0 } )
        .SetLength( 4.0f )
        .SetCellSpacing( 2.0f )
        .SetRingSize( 6 )
        .SetTubeRadius( 1.5f )
        .SetBranchingDepth( 0 )
        .SetSeed( 1 )
        .Build();

    // Ring centers are on the X axis; radial direction from (x, 0, 0) to cell (x, y, z) must match normal
    for( uint32_t i = 0; i < result.totalCells; ++i )
    {
        glm::vec3 pos    = glm::vec3( result.positions[ i ] );
        glm::vec3 normal = glm::vec3( result.normals[ i ] );
        // Radial component (perpendicular to X axis)
        glm::vec3 radial = glm::vec3( 0.0f, pos.y, pos.z );
        if( glm::length( radial ) > 1e-4f )
        {
            float dot = glm::dot( normal, glm::normalize( radial ) );
            EXPECT_GT( dot, 0.9f ) << "Normal at cell " << i << " must point radially outward";
        }
    }
}

// Adaptive ring size: trunk ring count matches SetRingSize() (backward compat)
TEST( VesselTreeGeneratorTest, AdaptiveRingSize_TrunkMatchesLegacy )
{
    // Trunk-only (depth=0): adaptive formula should produce same ring count as SetRingSize(6).
    const uint32_t expectedRingSize = 6;
    auto result = VesselTreeGenerator::BranchingTree()
        .SetRingSize( expectedRingSize )
        .SetTubeRadius( 1.5f )
        .SetCellSpacing( 1.5f )
        .SetLength( 4.0f ) // 3 rings
        .SetBranchingDepth( 0 )
        .Build();

    // 3 rings × 6 cells = 18 cells
    EXPECT_EQ( result.totalCells, 18u );
}

// Adaptive ring size: child branches have fewer cells per ring than the trunk
TEST( VesselTreeGeneratorTest, AdaptiveRingSize_ChildSmallerRing )
{
    auto result = VesselTreeGenerator::BranchingTree()
        .SetRingSize( 6 )
        .SetTubeRadius( 1.5f )
        .SetTubeRadiusFalloff( 0.65f )
        .SetCellSpacing( 1.5f )
        .SetLength( 6.0f ) // 5 rings per branch
        .SetBranchingDepth( 1 )
        .SetSeed( 42 )
        .Build();

    // Trunk has 5 rings × 6 cells = 30. Children have ring=4 (round(2π×0.975/1.57)=4).
    // 2 children × 5 rings × 4 cells = 40. Total = 70.
    // (Segment count invariant: sum(segmentCounts) == totalCells)
    uint32_t sumSegments = 0;
    for( uint32_t s : result.segmentCounts ) sumSegments += s;
    EXPECT_EQ( sumSegments, result.totalCells );

    // Trunk is the first segment; should have more cells per ring than child segments
    ASSERT_GE( result.segmentCounts.size(), 3u );
    uint32_t trunkCells = result.segmentCounts[ 0 ]; // numRings × ringSize for trunk
    uint32_t childCells = result.segmentCounts[ 1 ]; // same numRings × childRingSize
    // Since ringSize shrinks, childCells < trunkCells (same numRings but fewer per ring)
    EXPECT_LT( childCells, trunkCells );
}

// Adaptive ring size: all position.w values must be 1.0 (uniform cell scale)
TEST( VesselTreeGeneratorTest, AdaptiveRingSize_UniformCellScale )
{
    auto result = VesselTreeGenerator::BranchingTree()
        .SetRingSize( 6 )
        .SetTubeRadius( 1.5f )
        .SetTubeRadiusFalloff( 0.65f )
        .SetBranchingDepth( 2 )
        .SetSeed( 42 )
        .Build();

    for( const auto& p : result.positions )
        EXPECT_FLOAT_EQ( p.w, 1.0f );
}

// Junction edges when parent and child have different ring sizes
TEST( VesselTreeGeneratorTest, JunctionEdges_DifferentRingSizes )
{
    auto result = VesselTreeGenerator::BranchingTree()
        .SetRingSize( 6 )
        .SetTubeRadius( 1.5f )
        .SetTubeRadiusFalloff( 0.65f ) // child ring=4
        .SetCellSpacing( 1.5f )
        .SetLength( 6.0f )
        .SetBranchingDepth( 1 )
        .SetSeed( 42 )
        .Build();

    // Total cells = 5×6 + 2×(5×4) = 30 + 40 = 70
    // Every cell index referenced in edges must be in range
    for( const auto& [a, b] : result.edges )
    {
        EXPECT_LT( a, result.totalCells );
        EXPECT_LT( b, result.totalCells );
    }

    // Junction edges exist: at least ringSize edges cross the parent-child boundary.
    // Trunk last ring = cells [24..29]; child first rings start after that.
    uint32_t trunkEnd = result.segmentCounts[ 0 ]; // 30
    uint32_t junctionEdgeCount = 0;
    for( const auto& [a, b] : result.edges )
    {
        bool aInTrunk = a < trunkEnd;
        bool bInTrunk = b < trunkEnd;
        if( aInTrunk != bInTrunk ) // one endpoint in trunk, one in child
            junctionEdgeCount++;
    }
    EXPECT_GT( junctionEdgeCount, 0u );
}

// Curvature=0: all ring centers lie on the branch axis (straight line)
TEST( VesselTreeGeneratorTest, CurvedBranch_ZeroCurvature_StraightLine )
{
    const glm::vec3 origin( 0.0f, 0.0f, 0.0f );
    const glm::vec3 direction( 1.0f, 0.0f, 0.0f );
    const uint32_t  ringSize = 6;
    const float     spacing  = 2.0f;

    auto result = VesselTreeGenerator::BranchingTree()
        .SetOrigin( origin ).SetDirection( direction )
        .SetLength( 8.0f ).SetCellSpacing( spacing ).SetRingSize( ringSize )
        .SetBranchingDepth( 0 ).SetCurvature( 0.0f ).SetSeed( 42 )
        .Build();

    // With curvature=0 each ring center = average of its cells, must lie on the X axis
    uint32_t numRings = result.totalCells / ringSize;
    for( uint32_t r = 0; r < numRings; ++r )
    {
        glm::vec3 center( 0.0f );
        for( uint32_t j = 0; j < ringSize; ++j )
            center += glm::vec3( result.positions[ r * ringSize + j ] );
        center /= static_cast<float>( ringSize );
        // Y and Z should be zero (ring centered on axis)
        EXPECT_NEAR( center.y, 0.0f, 1e-4f ) << "Ring " << r << " center Y off-axis";
        EXPECT_NEAR( center.z, 0.0f, 1e-4f ) << "Ring " << r << " center Z off-axis";
    }
}

// Curvature>0: ring centers deviate from the straight axis
TEST( VesselTreeGeneratorTest, CurvedBranch_NonZeroCurvature_CellsDeviate )
{
    const glm::vec3 origin( 0.0f, 0.0f, 0.0f );
    const glm::vec3 direction( 1.0f, 0.0f, 0.0f );
    const uint32_t  ringSize = 6;

    auto result = VesselTreeGenerator::BranchingTree()
        .SetOrigin( origin ).SetDirection( direction )
        .SetLength( 10.0f ).SetCellSpacing( 2.0f ).SetRingSize( ringSize )
        .SetBranchingDepth( 0 ).SetCurvature( 0.4f ).SetSeed( 7 )
        .Build();

    // Compute max deviation of ring centers from the straight axis
    uint32_t numRings  = result.totalCells / ringSize;
    float    maxDevSq  = 0.0f;
    for( uint32_t r = 0; r < numRings; ++r )
    {
        glm::vec3 center( 0.0f );
        for( uint32_t j = 0; j < ringSize; ++j )
            center += glm::vec3( result.positions[ r * ringSize + j ] );
        center /= static_cast<float>( ringSize );
        // Deviation = distance from X axis (Y² + Z²)
        maxDevSq = std::max( maxDevSq, center.y * center.y + center.z * center.z );
    }
    EXPECT_GT( maxDevSq, 0.01f ) << "Curvature=0.4 should produce visible deviation from straight axis";
}

// =================================================================================================
// MorphologyGenerator::CreateCurvedTile
// =================================================================================================

// Basic validity: non-zero vertices/indices, all indices in range
TEST( MorphologyGeneratorTest, CreateCurvedTile_ValidMesh )
{
    auto data = MorphologyGenerator::CreateCurvedTile( 60.0f, 1.35f, 0.25f, 1.5f, 4 );
    ASSERT_FALSE( data.vertices.empty() );
    ASSERT_FALSE( data.indices.empty() );
    EXPECT_EQ( data.indices.size() % 3, 0u ) << "Indices must be a multiple of 3";
    for( uint32_t idx : data.indices )
        EXPECT_LT( idx, static_cast<uint32_t>( data.vertices.size() ) ) << "Index out of range";
}

// At theta=0 (arc center), outer face normal must point in +Y (compatible with orientation pipeline)
TEST( MorphologyGeneratorTest, CreateCurvedTile_OuterCenterNormalPlusY )
{
    auto      data    = MorphologyGenerator::CreateCurvedTile( 60.0f, 1.35f, 0.25f, 1.5f, 4 );
    glm::vec4 nrm     = data.vertices[ 4 ].normal; // arc center = sector 2 of 4, outer top vertex
    // Sector 2 of 4 corresponds to theta = -30 + 2*15 = 0 degrees → top vertex (even index)
    uint32_t  centerV = 4; // outerBase + 2*2 = 4
    nrm               = data.vertices[ centerV ].normal;
    EXPECT_NEAR( nrm.x, 0.0f, 1e-5f );
    EXPECT_NEAR( nrm.y, 1.0f, 1e-5f );
    EXPECT_NEAR( nrm.z, 0.0f, 1e-5f );
}

// All normals on outer face must be unit length
TEST( MorphologyGeneratorTest, CreateCurvedTile_NormalsUnitLength )
{
    auto data = MorphologyGenerator::CreateCurvedTile( 60.0f, 1.35f, 0.25f, 1.5f, 4 );
    for( const auto& v : data.vertices )
    {
        float len = glm::length( glm::vec3( v.normal ) );
        EXPECT_NEAR( len, 1.0f, 1e-5f ) << "Normal not unit length";
    }
}

// Outer surface points must be at radius R_out from the vessel axis (Y offset = R_out - R_mid)
TEST( MorphologyGeneratorTest, CreateCurvedTile_OuterRadiusCorrect )
{
    const float arc = 60.0f, h = 1.35f, t = 0.25f, ri = 1.5f;
    auto        data   = MorphologyGenerator::CreateCurvedTile( arc, h, t, ri, 4 );
    const float R_out  = ri + t;
    const float R_mid  = ri + t * 0.5f;
    // Outer face: vertices 0 .. (sectors+1)*2-1
    for( uint32_t i = 0; i <= 4; ++i )
    {
        glm::vec3 pos( data.vertices[ i * 2 ].pos );
        // Arc is in the YZ plane; radial distance = sqrt((y+R_mid)^2 + z^2)
        float radial = sqrtf( ( pos.y + R_mid ) * ( pos.y + R_mid ) + pos.z * pos.z );
        EXPECT_NEAR( radial, R_out, 1e-4f ) << "Outer vertex " << i << " radius mismatch";
    }
}

// Reproducibility: same seed must produce identical results
TEST( VesselTreeGeneratorTest, Reproducible_SameSeed )
{
    auto build = [&]()
    {
        return VesselTreeGenerator::BranchingTree()
            .SetLength( 10.0f ).SetRingSize( 6 ).SetBranchingDepth( 2 ).SetSeed( 99 ).Build();
    };
    auto r1 = build();
    auto r2 = build();

    ASSERT_EQ( r1.totalCells, r2.totalCells );
    for( uint32_t i = 0; i < r1.totalCells; ++i )
    {
        EXPECT_NEAR( r1.positions[ i ].x, r2.positions[ i ].x, 1e-5f );
        EXPECT_NEAR( r1.positions[ i ].y, r2.positions[ i ].y, 1e-5f );
        EXPECT_NEAR( r1.positions[ i ].z, r2.positions[ i ].z, 1e-5f );
    }
}

TEST( VesselTreeGeneratorTest, EdgeFlags_SizeMatchesEdges )
{
    auto tree = VesselTreeGenerator::BranchingTree()
        .SetLength( 10.0f ).SetRingSize( 6 ).SetBranchingDepth( 1 ).SetSeed( 42 ).Build();

    ASSERT_EQ( tree.edgeFlags.size(), tree.edges.size() )
        << "edgeFlags must be parallel to edges";
}

TEST( VesselTreeGeneratorTest, EdgeFlags_RingEdgesTaggedCorrectly )
{
    // Single straight trunk, no branches — all edges are ring or axial
    auto tree = VesselTreeGenerator::BranchingTree()
        .SetLength( 10.0f ).SetRingSize( 6 ).SetBranchingDepth( 0 ).SetSeed( 42 ).Build();

    ASSERT_EQ( tree.edgeFlags.size(), tree.edges.size() );

    int ringCount  = 0;
    int axialCount = 0;
    for( uint32_t f : tree.edgeFlags )
    {
        if( f == 0x1u ) ++ringCount;
        else if( f == 0x2u ) ++axialCount;
        else ADD_FAILURE() << "Unexpected flag " << f << " on trunk-only tree";
    }

    EXPECT_GT( ringCount,  0 ) << "Expected ring edges";
    EXPECT_GT( axialCount, 0 ) << "Expected axial edges";
    EXPECT_EQ( ringCount + axialCount, static_cast<int>( tree.edges.size() ) );
}

TEST( VesselTreeGeneratorTest, EdgeFlags_JunctionEdgesTaggedCorrectly )
{
    auto tree = VesselTreeGenerator::BranchingTree()
        .SetLength( 10.0f ).SetRingSize( 6 ).SetBranchingDepth( 1 ).SetSeed( 42 ).Build();

    ASSERT_EQ( tree.edgeFlags.size(), tree.edges.size() );

    bool hasJunction = false;
    for( uint32_t f : tree.edgeFlags )
    {
        if( f == 0x4u ) { hasJunction = true; break; }
    }
    EXPECT_TRUE( hasJunction ) << "Branching tree must have junction edges (0x4)";
}
