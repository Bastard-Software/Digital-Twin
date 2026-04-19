#include "simulation/AgentGroup.h"
#include "simulation/MorphologyGenerator.h"
#include "simulation/SpatialDistribution.h"
#include "simulation/VesselTreeGenerator.h"
#include <gtest/gtest.h>
#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>
#include <glm/gtc/quaternion.hpp>
#include <algorithm>
#include <cmath>
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
// SpatialDistribution::LatticeInSphere — CPU-only
// =================================================================================================

TEST( SpatialDistribution_LatticeInSphere, AllInsideSphere )
{
    const float radius = 5.0f;
    auto positions = SpatialDistribution::LatticeInSphere( 1.0f, radius );

    ASSERT_FALSE( positions.empty() );
    for( const auto& p : positions )
    {
        float dist = glm::length( glm::vec3( p ) );
        EXPECT_LE( dist, radius + 1e-4f ) << "Point outside sphere radius";
    }
}

TEST( SpatialDistribution_LatticeInSphere, CorrectSpacing )
{
    const float spacing = 1.5f;
    auto positions = SpatialDistribution::LatticeInSphere( spacing, 4.0f );

    ASSERT_FALSE( positions.empty() );
    // Every point must have at least one neighbor at exactly `spacing` distance.
    for( uint32_t i = 0; i < positions.size(); ++i )
    {
        float minDist = std::numeric_limits<float>::max();
        for( uint32_t j = 0; j < positions.size(); ++j )
        {
            if( i == j ) continue;
            float d = glm::length( glm::vec3( positions[ i ] ) - glm::vec3( positions[ j ] ) );
            minDist = std::min( minDist, d );
        }
        EXPECT_NEAR( minDist, spacing, 1e-4f ) << "Point " << i << " has no neighbor at expected spacing";
    }
}

TEST( SpatialDistribution_LatticeInSphere, StatusFlagAlive )
{
    auto positions = SpatialDistribution::LatticeInSphere( 1.0f, 3.0f );
    for( const auto& p : positions )
        EXPECT_FLOAT_EQ( p.w, 1.0f );
}

TEST( SpatialDistribution_LatticeInSphere, CenterOffset )
{
    const glm::vec3 center( 10.0f, -5.0f, 3.0f );
    const float     radius = 3.0f;
    auto positions = SpatialDistribution::LatticeInSphere( 1.0f, radius, center );

    ASSERT_FALSE( positions.empty() );
    for( const auto& p : positions )
    {
        float dist = glm::length( glm::vec3( p ) - center );
        EXPECT_LE( dist, radius + 1e-4f ) << "Offset point outside sphere";
    }
}

TEST( SpatialDistribution_LatticeInSphere, Deterministic )
{
    auto a = SpatialDistribution::LatticeInSphere( 1.2f, 4.0f );
    auto b = SpatialDistribution::LatticeInSphere( 1.2f, 4.0f );

    ASSERT_EQ( a.size(), b.size() );
    for( uint32_t i = 0; i < a.size(); ++i )
    {
        EXPECT_FLOAT_EQ( a[ i ].x, b[ i ].x );
        EXPECT_FLOAT_EQ( a[ i ].y, b[ i ].y );
        EXPECT_FLOAT_EQ( a[ i ].z, b[ i ].z );
    }
}

// =================================================================================================
// SpatialDistribution::LatticeInCylinder — CPU-only
// =================================================================================================

TEST( SpatialDistribution_LatticeInCylinder, AllInsideCylinder )
{
    const float radius     = 4.0f;
    const float halfLength = 6.0f;
    auto        positions  = SpatialDistribution::LatticeInCylinder( 1.0f, radius, halfLength );

    ASSERT_FALSE( positions.empty() );
    for( const auto& p : positions )
    {
        // Axis is Y by default
        float radDist = std::sqrt( p.x * p.x + p.z * p.z );
        EXPECT_LE( radDist, radius + 1e-4f );
        EXPECT_LE( std::abs( p.y ), halfLength + 1e-4f );
    }
}

TEST( SpatialDistribution_LatticeInCylinder, CorrectSpacing )
{
    const float spacing    = 1.5f;
    auto        positions  = SpatialDistribution::LatticeInCylinder( spacing, 3.0f, 4.0f );

    ASSERT_FALSE( positions.empty() );
    // No two points should be closer than spacing (minus floating-point tolerance)
    for( size_t i = 0; i < positions.size(); ++i )
        for( size_t j = i + 1; j < positions.size(); ++j )
        {
            glm::vec3 a( positions[ i ] ), b( positions[ j ] );
            EXPECT_GE( glm::length( b - a ), spacing - 1e-3f );
        }
}

TEST( SpatialDistribution_LatticeInCylinder, StatusFlagAlive )
{
    auto positions = SpatialDistribution::LatticeInCylinder( 1.0f, 3.0f, 4.0f );
    for( const auto& p : positions )
        EXPECT_FLOAT_EQ( p.w, 1.0f );
}

TEST( SpatialDistribution_LatticeInCylinder, CenterOffset )
{
    const glm::vec3 center( 5.0f, -3.0f, 2.0f );
    auto            positions = SpatialDistribution::LatticeInCylinder( 1.0f, 2.0f, 3.0f, center );

    ASSERT_FALSE( positions.empty() );
    for( const auto& p : positions )
    {
        float radDist = std::sqrt( ( p.x - center.x ) * ( p.x - center.x ) +
                                   ( p.z - center.z ) * ( p.z - center.z ) );
        EXPECT_LE( radDist, 2.0f + 1e-4f );
        EXPECT_LE( std::abs( p.y - center.y ), 3.0f + 1e-4f );
    }
}

TEST( SpatialDistribution_LatticeInCylinder, CustomAxis )
{
    // Cylinder aligned with X axis
    auto positions = SpatialDistribution::LatticeInCylinder( 1.0f, 2.0f, 4.0f,
                                                              glm::vec3( 0.0f ),
                                                              glm::vec3( 1.0f, 0.0f, 0.0f ) );

    ASSERT_FALSE( positions.empty() );
    for( const auto& p : positions )
    {
        float radDist = std::sqrt( p.y * p.y + p.z * p.z );
        EXPECT_LE( radDist, 2.0f + 1e-4f );
        EXPECT_LE( std::abs( p.x ), 4.0f + 1e-4f );
    }
}

// =================================================================================================
// SpatialDistribution::UniformInCylinder — CPU-only
// =================================================================================================

TEST( SpatialDistribution_UniformInCylinder, CorrectCount )
{
    auto positions = SpatialDistribution::UniformInCylinder( 100, 3.0f, 6.0f );
    EXPECT_EQ( positions.size(), 100u );
}

TEST( SpatialDistribution_UniformInCylinder, AllInsideCylinder )
{
    const float radius     = 3.0f;
    const float halfLength = 6.0f;

    auto positions = SpatialDistribution::UniformInCylinder( 200, radius, halfLength );
    ASSERT_FALSE( positions.empty() );

    for( const auto& p : positions )
    {
        float radDist = std::sqrt( p.x * p.x + p.z * p.z );
        EXPECT_LE( radDist, radius + 1e-4f ) << "Point outside outer radius";
        EXPECT_LE( std::abs( p.y ), halfLength + 1e-4f ) << "Point outside half-length";
    }
}

TEST( SpatialDistribution_UniformInCylinder, HollowCylinder )
{
    const float radius      = 4.0f;
    const float innerRadius = 2.0f;
    const float halfLength  = 5.0f;

    auto positions = SpatialDistribution::UniformInCylinder( 200, radius, halfLength,
                                                              glm::vec3( 0.0f ),
                                                              glm::vec3( 0.0f, 1.0f, 0.0f ),
                                                              innerRadius );
    ASSERT_FALSE( positions.empty() );

    for( const auto& p : positions )
    {
        float radDist = std::sqrt( p.x * p.x + p.z * p.z );
        EXPECT_GE( radDist, innerRadius - 1e-4f ) << "Point inside inner radius cutout";
        EXPECT_LE( radDist, radius + 1e-4f )      << "Point outside outer radius";
    }
}

TEST( SpatialDistribution_UniformInCylinder, StatusFlagAlive )
{
    auto positions = SpatialDistribution::UniformInCylinder( 50, 2.0f, 4.0f );
    for( const auto& p : positions )
        EXPECT_FLOAT_EQ( p.w, 1.0f );
}

TEST( SpatialDistribution_UniformInCylinder, Deterministic )
{
    auto p1 = SpatialDistribution::UniformInCylinder( 50, 3.0f, 5.0f,
                                                       glm::vec3( 0.0f ),
                                                       glm::vec3( 0.0f, 1.0f, 0.0f ),
                                                       0.0f, 42 );
    auto p2 = SpatialDistribution::UniformInCylinder( 50, 3.0f, 5.0f,
                                                       glm::vec3( 0.0f ),
                                                       glm::vec3( 0.0f, 1.0f, 0.0f ),
                                                       0.0f, 42 );
    ASSERT_EQ( p1.size(), p2.size() );
    for( size_t i = 0; i < p1.size(); ++i )
    {
        EXPECT_FLOAT_EQ( p1[ i ].x, p2[ i ].x );
        EXPECT_FLOAT_EQ( p1[ i ].y, p2[ i ].y );
        EXPECT_FLOAT_EQ( p1[ i ].z, p2[ i ].z );
    }
}

TEST( SpatialDistribution_UniformInCylinder, DifferentSeeds )
{
    auto p1 = SpatialDistribution::UniformInCylinder( 50, 3.0f, 5.0f,
                                                       glm::vec3( 0.0f ),
                                                       glm::vec3( 0.0f, 1.0f, 0.0f ),
                                                       0.0f, 42 );
    auto p2 = SpatialDistribution::UniformInCylinder( 50, 3.0f, 5.0f,
                                                       glm::vec3( 0.0f ),
                                                       glm::vec3( 0.0f, 1.0f, 0.0f ),
                                                       0.0f, 99 );
    ASSERT_EQ( p1.size(), p2.size() );
    bool anyDiffers = false;
    for( size_t i = 0; i < p1.size(); ++i )
        if( p1[ i ].x != p2[ i ].x || p1[ i ].y != p2[ i ].y || p1[ i ].z != p2[ i ].z )
            anyDiffers = true;
    EXPECT_TRUE( anyDiffers ) << "Different seeds must produce different positions";
}

TEST( SpatialDistribution_UniformInCylinder, CustomAxis )
{
    // Cylinder aligned with X axis
    auto positions = SpatialDistribution::UniformInCylinder( 100, 2.0f, 4.0f,
                                                              glm::vec3( 0.0f ),
                                                              glm::vec3( 1.0f, 0.0f, 0.0f ) );
    ASSERT_FALSE( positions.empty() );
    for( const auto& p : positions )
    {
        float radDist = std::sqrt( p.y * p.y + p.z * p.z );
        EXPECT_LE( radDist, 2.0f + 1e-4f ) << "Point outside radial bound (X-axis cylinder)";
        EXPECT_LE( std::abs( p.x ), 4.0f + 1e-4f ) << "Point outside axial bound";
    }
}

TEST( SpatialDistribution_UniformInCylinder, CenterOffset )
{
    const glm::vec3 center( 5.0f, -3.0f, 2.0f );
    const float     radius     = 2.0f;
    const float     halfLength = 3.0f;

    auto positions = SpatialDistribution::UniformInCylinder( 100, radius, halfLength, center );
    ASSERT_FALSE( positions.empty() );
    for( const auto& p : positions )
    {
        // Radial distance from Y axis through center
        float dx = p.x - center.x;
        float dz = p.z - center.z;
        float radDist = std::sqrt( dx * dx + dz * dz );
        EXPECT_LE( radDist, radius + 1e-4f );
        EXPECT_LE( std::abs( p.y - center.y ), halfLength + 1e-4f );
    }
}

// =================================================================================================
// SpatialDistribution::ShellOnCylinder — CPU-only
// =================================================================================================

TEST( SpatialDistribution_ShellOnCylinder, AllOnSurface )
{
    // All positions should lie on the cylinder surface (within jitter tolerance).
    const float radius   = 3.0f;
    const float jitter   = 0.3f;
    // Max radial displacement from jitter: angle_jitter * radius ≈ 0.3 * 2π/10 * 3 ≈ 0.57
    const float pi           = 3.14159265f;
    const float maxRadialErr = jitter * ( 2.0f * pi / 10.0f ) * radius + 0.01f;

    auto result = SpatialDistribution::ShellOnCylinder( 1.35f, radius, 6.0f, 10,
                                                         glm::vec3( 0.0f ),
                                                         glm::vec3( 0.0f, 1.0f, 0.0f ),
                                                         jitter, 42 );
    ASSERT_FALSE( result.positions.empty() );

    for( const auto& p : result.positions )
    {
        // Radial distance from Y axis
        float radDist = std::sqrt( p.x * p.x + p.z * p.z );
        EXPECT_NEAR( radDist, radius, maxRadialErr )
            << "Position not on cylinder surface: radial distance = " << radDist;
    }
}

TEST( SpatialDistribution_ShellOnCylinder, NormalsPointOutward )
{
    // Normals must point radially outward (no jitter on normals).
    auto result = SpatialDistribution::ShellOnCylinder( 1.35f, 3.0f, 6.0f, 10,
                                                         glm::vec3( 0.0f ),
                                                         glm::vec3( 0.0f, 1.0f, 0.0f ),
                                                         0.3f, 42 );
    ASSERT_EQ( result.positions.size(), result.normals.size() );

    for( size_t i = 0; i < result.normals.size(); ++i )
    {
        glm::vec3 n( result.normals[ i ] );
        EXPECT_NEAR( glm::length( n ), 1.0f, 1e-4f ) << "Normal not unit length at index " << i;
        // Normal w should be 0
        EXPECT_FLOAT_EQ( result.normals[ i ].w, 0.0f );
        // Normal must point radially outward (Y component should be ~0, XZ should match radial dir)
        EXPECT_NEAR( n.y, 0.0f, 1e-4f ) << "Normal has Y component for Y-axis cylinder";
    }
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

// Phase 2.3: VesselTreeGenerator is now a pure cell placer. These tests target the
// adaptive-ring-count formula (Aird 2007 morphometry), radial-outward polarity seed
// (Mellman & Nelson 2008), staggered brick pattern (Davies 2009 flow-aligned ECs),
// and quaternion orientation that aligns each cell's local +Y with the radial-outward
// vessel surface normal.

// Dual-seam capillary lower bound: r=2 and ECWidth=12 would yield 1 cell/ring, clamped
// to the 2-cell minimum (Bär 1984 — 1-cell autocellular capillaries not modelled at this
// point-agent scale).
TEST( VesselTreeGeneratorTest, RingCount_DualSeamCapillary )
{
    auto result = VesselTreeGenerator::BranchingTree()
        .SetOrigin( { 0, 0, 0 } ).SetDirection( { 1, 0, 0 } )
        .SetLength( 4.0f ).SetTubeRadius( 2.0f )
        .SetECCircumferentialWidth( 12.0f ).SetBranchingDepth( 0 ).SetSeed( 1 )
        .Build();

    ASSERT_GT( result.totalCells, 0u );
    // Count cells at the first ring's axial position (X ≈ origin.x).
    const float firstX = result.cells[ 0 ].position.x;
    uint32_t firstRingCount = 0;
    for( const auto& c : result.cells )
        if( std::fabs( c.position.x - firstX ) < 1e-3f )
            ++firstRingCount;
    EXPECT_EQ( firstRingCount, 2u ) << "Dual-seam capillary minimum";
}

// Post-capillary venule: r=10, ECWidth=15 (venule-typical per Aird 2007) → ring ≈ 4.
TEST( VesselTreeGeneratorTest, RingCount_Venule )
{
    auto result = VesselTreeGenerator::BranchingTree()
        .SetOrigin( { 0, 0, 0 } ).SetDirection( { 1, 0, 0 } )
        .SetLength( 4.0f ).SetTubeRadius( 10.0f )
        .SetECCircumferentialWidth( 15.0f ).SetBranchingDepth( 0 ).SetSeed( 1 )
        .Build();

    const float firstX = result.cells[ 0 ].position.x;
    uint32_t firstRingCount = 0;
    for( const auto& c : result.cells )
        if( std::fabs( c.position.x - firstX ) < 1e-3f )
            ++firstRingCount;
    EXPECT_GE( firstRingCount, 3u );
    EXPECT_LE( firstRingCount, 5u );
}

// Arteriole: r=25, ECWidth=9 (arteriole-typical per Aird 2007) → ring ≈ 17.
TEST( VesselTreeGeneratorTest, RingCount_Arteriole )
{
    auto result = VesselTreeGenerator::BranchingTree()
        .SetOrigin( { 0, 0, 0 } ).SetDirection( { 1, 0, 0 } )
        .SetLength( 50.0f ).SetTubeRadius( 25.0f )
        .SetECCircumferentialWidth( 9.0f ).SetBranchingDepth( 0 ).SetSeed( 1 )
        .Build();

    const float firstX = result.cells[ 0 ].position.x;
    uint32_t firstRingCount = 0;
    for( const auto& c : result.cells )
        if( std::fabs( c.position.x - firstX ) < 1e-3f )
            ++firstRingCount;
    EXPECT_GE( firstRingCount, 8u );
    EXPECT_LE( firstRingCount, 20u );
}

// Muscular artery: r=100, ECWidth=12 → ring ≈ 52.
TEST( VesselTreeGeneratorTest, RingCount_Artery )
{
    auto result = VesselTreeGenerator::BranchingTree()
        .SetOrigin( { 0, 0, 0 } ).SetDirection( { 1, 0, 0 } )
        .SetLength( 200.0f ).SetTubeRadius( 100.0f )
        .SetECCircumferentialWidth( 12.0f ).SetBranchingDepth( 0 ).SetSeed( 1 )
        .Build();

    const float firstX = result.cells[ 0 ].position.x;
    uint32_t firstRingCount = 0;
    for( const auto& c : result.cells )
        if( std::fabs( c.position.x - firstX ) < 1e-3f )
            ++firstRingCount;
    EXPECT_GE( firstRingCount, 40u );
}

// Pre-seeded polarity: every cell's polaritySeed.xyz must be a unit vector pointing
// radially outward from the vessel axis (basal direction per Item 1 shader convention).
// Magnitude = 1.0 so junctional propagation (Phase 4.5) sustains polarity without BM contact.
TEST( VesselTreeGeneratorTest, PolaritySeed_PointsRadialOutward )
{
    auto result = VesselTreeGenerator::BranchingTree()
        .SetOrigin( { 0, 0, 0 } ).SetDirection( { 1, 0, 0 } )
        .SetLength( 10.0f ).SetTubeRadius( 3.0f )
        .SetECCircumferentialWidth( 1.0f ).SetBranchingDepth( 0 ).SetSeed( 1 )
        .Build();

    ASSERT_GT( result.totalCells, 0u );
    for( uint32_t i = 0; i < result.totalCells; ++i )
    {
        const auto& c       = result.cells[ i ];
        glm::vec3   seedDir = glm::vec3( c.polaritySeed );

        // Seed magnitude = 1.0 and direction ~ unit.
        EXPECT_NEAR( c.polaritySeed.w, 1.0f, 1e-5f ) << "Cell " << i << " polarity magnitude";
        EXPECT_NEAR( glm::length( seedDir ), 1.0f, 1e-4f )
            << "Cell " << i << " polarity direction not unit-length";

        // For a straight tube along +X with centreline at Y=Z=0, the radial
        // outward direction at a cell equals the (Y, Z) component of its position.
        glm::vec3 radialExpected = glm::normalize( glm::vec3( 0.0f, c.position.y, c.position.z ) );
        EXPECT_GT( glm::dot( seedDir, radialExpected ), 0.98f )
            << "Cell " << i << " polarity must align with radial outward";
    }
}

// Staggered brick pattern: alternate rings are circumferentially offset by half a cell-
// width (Davies 2009 — brick interlock prevents longitudinal-seam mechanical instability
// under JKR + VE-cad catch-bond loads).
TEST( VesselTreeGeneratorTest, StaggeredBricks_OddRingOffset )
{
    const float tubeRadius = 3.0f;
    const float ecWidth    = 1.0f;
    auto result = VesselTreeGenerator::BranchingTree()
        .SetOrigin( { 0, 0, 0 } ).SetDirection( { 1, 0, 0 } )
        .SetLength( 10.0f ).SetTubeRadius( tubeRadius )
        .SetECCircumferentialWidth( ecWidth ).SetBranchingDepth( 0 ).SetSeed( 1 )
        .Build();

    ASSERT_GT( result.totalCells, 0u );

    // Segregate cells into rings by axial position (X). Then compare the angular
    // phase of the first cell in ring 0 vs ring 1 — must differ by π/ringSize.
    std::vector<float> xs;
    for( const auto& c : result.cells )
    {
        bool seen = false;
        for( float x : xs ) if( std::fabs( x - c.position.x ) < 1e-3f ) { seen = true; break; }
        if( !seen ) xs.push_back( c.position.x );
    }
    std::sort( xs.begin(), xs.end() );
    ASSERT_GE( xs.size(), 2u );

    auto ringAngles = [&]( float x ) {
        std::vector<float> out;
        for( const auto& c : result.cells )
            if( std::fabs( c.position.x - x ) < 1e-3f )
                out.push_back( std::atan2( c.position.z, c.position.y ) );
        std::sort( out.begin(), out.end() );
        return out;
    };

    auto ring0 = ringAngles( xs[ 0 ] );
    auto ring1 = ringAngles( xs[ 1 ] );
    ASSERT_EQ( ring0.size(), ring1.size() );

    const float dAngle   = 2.0f * glm::pi<float>() / static_cast<float>( ring0.size() );
    const float expected = dAngle * 0.5f;
    float       diff     = std::fabs( ring1[ 0 ] - ring0[ 0 ] );
    diff                 = std::fmod( diff, dAngle ); // wrap into [0, dAngle)
    if( diff > dAngle * 0.5f ) diff = dAngle - diff;
    EXPECT_NEAR( diff, expected, dAngle * 0.1f )
        << "Ring 1 must be offset by half a cell-width circumferentially";
}

// Orientation quaternion: applying it to local +Y must produce the world radial-outward
// direction. This is the invariant Item 1's rendering + JKR-hull pipelines rely on.
TEST( VesselTreeGeneratorTest, Orientation_LocalYAlignsWithRadialOutward )
{
    auto result = VesselTreeGenerator::BranchingTree()
        .SetOrigin( { 0, 0, 0 } ).SetDirection( { 1, 0, 0 } )
        .SetLength( 6.0f ).SetTubeRadius( 2.0f )
        .SetECCircumferentialWidth( 1.0f ).SetBranchingDepth( 0 ).SetSeed( 1 )
        .Build();

    for( uint32_t i = 0; i < result.totalCells; ++i )
    {
        const auto& c = result.cells[ i ];
        glm::quat q( c.orientation.w, c.orientation.x, c.orientation.y, c.orientation.z );
        glm::vec3 worldY = q * glm::vec3( 0.0f, 1.0f, 0.0f );
        glm::vec3 radial = glm::normalize( glm::vec3( 0.0f, c.position.y, c.position.z ) );
        EXPECT_GT( glm::dot( worldY, radial ), 0.98f )
            << "Cell " << i << " orientation's local +Y must align with radial outward";
    }
}

// Branching generator still produces a tree of cells; the legacy segment/edge metadata is
// gone, so the test just asserts cell count grows with branching depth.
TEST( VesselTreeGeneratorTest, BranchingDepth_IncreasesCellCount )
{
    auto trunk = VesselTreeGenerator::BranchingTree()
        .SetLength( 10.0f ).SetTubeRadius( 2.0f )
        .SetECCircumferentialWidth( 1.0f ).SetBranchingDepth( 0 ).SetSeed( 42 )
        .Build();
    auto depth1 = VesselTreeGenerator::BranchingTree()
        .SetLength( 10.0f ).SetTubeRadius( 2.0f )
        .SetECCircumferentialWidth( 1.0f ).SetBranchingDepth( 1 )
        .SetBranchProbability( 1.0f ).SetSeed( 42 )
        .Build();
    EXPECT_GT( depth1.totalCells, trunk.totalCells )
        << "Branching depth 1 must add child-branch cells";
}

// Curvature=0: ring centres are collinear with the branch axis.
TEST( VesselTreeGeneratorTest, CurvedBranch_ZeroCurvature_RingCentresOnAxis )
{
    auto result = VesselTreeGenerator::BranchingTree()
        .SetOrigin( { 0, 0, 0 } ).SetDirection( { 1, 0, 0 } )
        .SetLength( 8.0f ).SetTubeRadius( 2.0f )
        .SetECCircumferentialWidth( 1.0f ).SetBranchingDepth( 0 ).SetCurvature( 0.0f )
        .SetSeed( 42 )
        .Build();

    std::vector<float> xs;
    for( const auto& c : result.cells )
    {
        bool seen = false;
        for( float x : xs ) if( std::fabs( x - c.position.x ) < 1e-3f ) { seen = true; break; }
        if( !seen ) xs.push_back( c.position.x );
    }
    for( float x : xs )
    {
        glm::vec3 center( 0.0f );
        uint32_t  n = 0;
        for( const auto& c : result.cells )
            if( std::fabs( c.position.x - x ) < 1e-3f )
            { center += glm::vec3( c.position ); ++n; }
        center /= static_cast<float>( n );
        // 1% symmetry-breaking jitter → mean of N cells sits within jitter/√N ≈ 0.005
        // of the axis. 0.05 tolerance covers the worst-case sample with margin while
        // still catching real centreline deviation (e.g. from curvature).
        EXPECT_NEAR( center.y, 0.0f, 0.05f );
        EXPECT_NEAR( center.z, 0.0f, 0.05f );
    }
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

// =================================================================================================
// MorphologyGenerator::CreateElongatedQuad / CreatePentagonDefect / CreateHeptagonDefect
// (Phase 2.2 puzzle-piece tessellation primitives — flow-aligned arterial rhomboid + Stone-Wales
// pentagon / heptagon defects for diameter transitions and bifurcation carinas; Davies 2009,
// Stone & Wales 1986, Chiu & Chien 2011.)
// =================================================================================================

// length=20, width=1, thickness=0.2 → corners at X=±10, Z=±0.5; hull extents reflect the
// radial (Y) and circumferential (Z) footprint per the CurvedTile coordinate convention
// (X=axial, Y=radial, Z=circumferential).
TEST( MorphologyGeneratorTest, CreateElongatedQuad_AspectRatio )
{
    const float length = 20.0f, width = 1.0f, thickness = 0.2f;
    auto m = MorphologyGenerator::CreateElongatedQuad( length, width, thickness );

    ASSERT_FALSE( m.vertices.empty() );
    EXPECT_EQ( m.indices.size() % 3, 0u );

    EXPECT_NEAR( m.hullExtentY, thickness * 0.5f, 1e-6f ) << "hullExtentY should be thickness/2";
    EXPECT_NEAR( m.hullExtentZ, width     * 0.5f, 1e-6f ) << "hullExtentZ should be width/2 (circumferential half-extent)";

    // Exactly 8 hull points: 4 corners + 4 edge midpoints on the Y=0 mid-plane.
    ASSERT_EQ( m.contactHull.size(), 8u );

    const float hl = length * 0.5f, hw = width * 0.5f;
    auto cornerSeen = [&]( float x, float z ) {
        for( const auto& p : m.contactHull )
            if( std::fabs( p.x - x ) < 1e-5f && std::fabs( p.y ) < 1e-5f && std::fabs( p.z - z ) < 1e-5f )
                return true;
        return false;
    };
    EXPECT_TRUE( cornerSeen( -hl, -hw ) );
    EXPECT_TRUE( cornerSeen( +hl, -hw ) );
    EXPECT_TRUE( cornerSeen( +hl, +hw ) );
    EXPECT_TRUE( cornerSeen( -hl, +hw ) );
    EXPECT_TRUE( cornerSeen( 0.0f, -hw ) );
    EXPECT_TRUE( cornerSeen( 0.0f, +hw ) );
    EXPECT_TRUE( cornerSeen( -hl, 0.0f ) );
    EXPECT_TRUE( cornerSeen( +hl, 0.0f ) );
}

// Pentagon fits within the 16-point contactHull budget and emits exactly
// 5 corners + 5 edge midpoints = 10 sub-sphere points.
TEST( MorphologyGeneratorTest, CreatePentagonDefect_HullPointCount )
{
    auto m = MorphologyGenerator::CreatePentagonDefect( 1.0f, 0.2f );
    EXPECT_EQ( m.contactHull.size(), 10u ) << "Pentagon: 5 corners + 5 edge midpoints";
    EXPECT_LE( m.contactHull.size(), 16u ) << "Must fit jkr_forces.comp contactHull buffer";

    // Sub-sphere radius uniform at thickness/2
    const float subR = 0.1f;
    for( const auto& p : m.contactHull )
        EXPECT_NEAR( p.w, subR, 1e-6f );
}

// Heptagon also fits within the 16-point budget (7 corners + 7 edge midpoints = 14).
TEST( MorphologyGeneratorTest, CreateHeptagonDefect_HullPointCount )
{
    auto m = MorphologyGenerator::CreateHeptagonDefect( 1.0f, 0.2f );
    EXPECT_EQ( m.contactHull.size(), 14u ) << "Heptagon: 7 corners + 7 edge midpoints";
    EXPECT_LE( m.contactHull.size(), 16u ) << "Must fit jkr_forces.comp contactHull buffer";

    const float subR = 0.1f;
    for( const auto& p : m.contactHull )
        EXPECT_NEAR( p.w, subR, 1e-6f );
}

// Pentagon hull points match corner (radius from center) + edge-midpoint (apothem from center)
// geometry. Corner 0 at angle 0 → points +X. Vertices rotate counter-clockwise around Y.
TEST( MorphologyGeneratorTest, CreatePentagonDefect_HullPointsMatchCornersAndMidpoints )
{
    const float radius = 1.5f;
    auto m = MorphologyGenerator::CreatePentagonDefect( radius, 0.2f );
    ASSERT_EQ( m.contactHull.size(), 10u );

    const uint32_t N   = 5;
    const float    two = 2.0f * glm::pi<float>();

    // First N entries are corners at full radius.
    for( uint32_t i = 0; i < N; ++i )
    {
        const auto& p = m.contactHull[ i ];
        float       d = std::sqrt( p.x * p.x + p.z * p.z );
        EXPECT_NEAR( d, radius, 1e-4f ) << "Corner " << i << " must lie on the circumscribed circle";
        EXPECT_NEAR( p.y, 0.0f, 1e-6f ) << "Corner " << i << " must lie on the Y=0 mid-plane";

        float expectedTheta = static_cast<float>( i ) * two / static_cast<float>( N );
        EXPECT_NEAR( p.x, radius * std::cos( expectedTheta ), 1e-4f );
        EXPECT_NEAR( p.z, radius * std::sin( expectedTheta ), 1e-4f );
    }

    // Next N entries are edge midpoints at apothem = radius * cos(pi/N).
    const float apothem = radius * std::cos( glm::pi<float>() / static_cast<float>( N ) );
    for( uint32_t i = 0; i < N; ++i )
    {
        const auto& p = m.contactHull[ N + i ];
        float       d = std::sqrt( p.x * p.x + p.z * p.z );
        EXPECT_NEAR( d, apothem, 1e-4f ) << "Edge midpoint " << i << " must lie on the apothem circle";
        EXPECT_NEAR( p.y, 0.0f, 1e-6f );
    }
}

// Same assertion for heptagon: corners on circumscribed circle, edge midpoints on apothem.
TEST( MorphologyGeneratorTest, CreateHeptagonDefect_HullPointsMatchCornersAndMidpoints )
{
    const float radius = 1.5f;
    auto m = MorphologyGenerator::CreateHeptagonDefect( radius, 0.2f );
    ASSERT_EQ( m.contactHull.size(), 14u );

    const uint32_t N   = 7;
    const float    two = 2.0f * glm::pi<float>();

    for( uint32_t i = 0; i < N; ++i )
    {
        const auto& p = m.contactHull[ i ];
        float       d = std::sqrt( p.x * p.x + p.z * p.z );
        EXPECT_NEAR( d, radius, 1e-4f );
        EXPECT_NEAR( p.y, 0.0f, 1e-6f );

        float expectedTheta = static_cast<float>( i ) * two / static_cast<float>( N );
        EXPECT_NEAR( p.x, radius * std::cos( expectedTheta ), 1e-4f );
        EXPECT_NEAR( p.z, radius * std::sin( expectedTheta ), 1e-4f );
    }

    const float apothem = radius * std::cos( glm::pi<float>() / static_cast<float>( N ) );
    for( uint32_t i = 0; i < N; ++i )
    {
        const auto& p = m.contactHull[ N + i ];
        float       d = std::sqrt( p.x * p.x + p.z * p.z );
        EXPECT_NEAR( d, apothem, 1e-4f );
        EXPECT_NEAR( p.y, 0.0f, 1e-6f );
    }
}

// Reproducibility: same seed must produce identical results
TEST( VesselTreeGeneratorTest, Reproducible_SameSeed )
{
    auto build = [&]()
    {
        return VesselTreeGenerator::BranchingTree()
            .SetLength( 10.0f ).SetTubeRadius( 2.0f )
            .SetECCircumferentialWidth( 1.0f ).SetBranchingDepth( 2 ).SetSeed( 99 ).Build();
    };
    auto r1 = build();
    auto r2 = build();

    ASSERT_EQ( r1.totalCells, r2.totalCells );
    for( uint32_t i = 0; i < r1.totalCells; ++i )
    {
        EXPECT_NEAR( r1.cells[ i ].position.x, r2.cells[ i ].position.x, 1e-5f );
        EXPECT_NEAR( r1.cells[ i ].position.y, r2.cells[ i ].position.y, 1e-5f );
        EXPECT_NEAR( r1.cells[ i ].position.z, r2.cells[ i ].position.z, 1e-5f );
    }
}
