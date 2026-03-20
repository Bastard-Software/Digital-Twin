#include "simulation/AgentGroup.h"
#include "simulation/MorphologyGenerator.h"
#include "simulation/SpatialDistribution.h"
#include <gtest/gtest.h>
#include <glm/glm.hpp>

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
