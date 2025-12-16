#include "resources/ResourceManager.hpp"
#include "resources/ShapeGenerator.hpp"
#include "resources/StreamingManager.hpp"
#include "rhi/Device.hpp"
#include "rhi/RHI.hpp"
#include <array>
#include <cstring>
#include <gtest/gtest.h>

using namespace DigitalTwin;

// =================================================================================================
// 1. Streaming Tests (Low Level Transfer)
// =================================================================================================

class StreamingTests : public ::testing::Test
{
protected:
    RHIConfig             config;
    Ref<Device>           device;
    Ref<StreamingManager> streamer;

    void SetUp() override
    {
        if( RHI::IsInitialized() )
            RHI::Shutdown();

        config.headless         = true;
        config.enableValidation = false;
        ASSERT_EQ( RHI::Init( config ), Result::SUCCESS );

        if( RHI::GetAdapterCount() > 0 )
        {
            device = RHI::CreateDevice( 0 );
        }

        streamer = CreateRef<StreamingManager>( device );
        streamer->Init();
    }

    void TearDown() override
    {
        if( streamer )
        {
            streamer->Shutdown();
            streamer.reset();
        }

        if( device )
        {
            RHI::DestroyDevice( device );
            device.reset();
        }
        RHI::Shutdown();
    }
};

TEST_F( StreamingTests, UploadAndReadbackLoop )
{
    if( !device )
        GTEST_SKIP() << "Device not initialized";

    // 1. Create a Device Local buffer
    BufferDesc bufDesc;
    bufDesc.size = 1024;
    bufDesc.type = BufferType::STORAGE;

    auto gpuBuffer = device->CreateBuffer( bufDesc );
    ASSERT_NE( gpuBuffer, nullptr );

    // 2. Prepare Data
    std::array<float, 4> sendData = { 1.1f, 2.2f, 3.3f, 4.4f };
    uint64_t             frameNum = 0;

    // --- Begin Frame ---
    streamer->BeginFrame( frameNum );

    streamer->UploadToBuffer( gpuBuffer, sendData.data(), sizeof( sendData ) );
    auto readToken = streamer->CaptureBuffer( gpuBuffer, sizeof( sendData ) );

    streamer->EndFrame();

    // --- Wait & Verify ---
    streamer->WaitForTransferComplete();

    float* readbackData = static_cast<float*>( readToken.mappedData );
    EXPECT_NEAR( readbackData[ 0 ], 1.1f, 0.0001f );
    EXPECT_NEAR( readbackData[ 3 ], 4.4f, 0.0001f );
}

// =================================================================================================
// 2. Shape Generator Tests (CPU Logic)
// =================================================================================================

TEST( ShapeGenerator, CubeGeometry )
{
    // A standard cube with flat shading (duplicate vertices for hard edges)
    // 6 faces * 4 vertices = 24 vertices
    // 6 faces * 2 triangles * 3 indices = 36 indices
    Mesh cube = ShapeGenerator::CreateCube();

    EXPECT_EQ( cube.vertices.size(), 24 );
    EXPECT_EQ( cube.indices.size(), 36 );
    EXPECT_EQ( cube.name, "Cube" );

    // Check bounds (roughly)
    EXPECT_FLOAT_EQ( cube.vertices[ 0 ].position.x, -0.5f ); // Should start at -0.5
}

TEST( ShapeGenerator, SphereGeometry )
{
    // Sphere with 32 stacks and 32 slices
    Mesh sphere = ShapeGenerator::CreateSphere( 1.0f, 32, 32 );

    EXPECT_GT( sphere.vertices.size(), 0 );
    EXPECT_GT( sphere.indices.size(), 0 );
    EXPECT_EQ( sphere.name, "Sphere" );
}

// =================================================================================================
// 3. Resource Manager Tests (Integration)
// =================================================================================================

class ResourceManagerTests : public ::testing::Test
{
protected:
    RHIConfig             config;
    Ref<Device>           device;
    Ref<StreamingManager> streamer;
    Ref<ResourceManager>  resources;

    void SetUp() override
    {
        if( RHI::IsInitialized() )
            RHI::Shutdown();

        config.headless = true;
        ASSERT_EQ( RHI::Init( config ), Result::SUCCESS );

        if( RHI::GetAdapterCount() > 0 )
        {
            device = RHI::CreateDevice( 0 );
        }

        streamer = CreateRef<StreamingManager>( device );
        streamer->Init();

        resources = CreateRef<ResourceManager>( device, streamer );
    }

    void TearDown() override
    {
        resources->Shutdown();
        resources.reset();
        streamer->Shutdown();
        streamer.reset();
        if( device )
            RHI::DestroyDevice( device );
        RHI::Shutdown();
    }
};

TEST_F( ResourceManagerTests, GetMeshCreatesValidBuffers )
{
    if( !device )
        GTEST_SKIP() << "Device not initialized";

    // 1. Request a Cube ID (Creates resource internally)
    AssetID cubeID = resources->GetMeshID( "Cube" );
    EXPECT_NE( cubeID, 0 );

    // 2. Retrieve Mesh by ID
    auto mesh = resources->GetMesh( cubeID );
    ASSERT_NE( mesh, nullptr );

    // 3. Verify Buffer Allocation
    EXPECT_NE( mesh->GetBuffer(), nullptr );
    EXPECT_GT( mesh->GetBuffer()->GetSize(), 0 );

    // Cube has 24 verts * 48 bytes + 36 indices * 4 bytes = 1296 bytes
    EXPECT_GE( mesh->GetBuffer()->GetSize(), 1296 );

    // 4. Verify Offsets
    EXPECT_EQ( mesh->GetIndexOffset(), 24 * sizeof( Vertex ) );
    EXPECT_EQ( mesh->GetIndexCount(), 36 );
}

TEST_F( ResourceManagerTests, CachingWorks )
{
    if( !device )
        GTEST_SKIP() << "Device not initialized";

    // 1. Request "Sphere" twice
    AssetID idA = resources->GetMeshID( "Sphere" );
    AssetID idB = resources->GetMeshID( "Sphere" );

    // IDs should be identical
    EXPECT_EQ( idA, idB );

    // 2. Pointers should be identical
    auto meshA = resources->GetMesh( idA );
    auto meshB = resources->GetMesh( idB );

    EXPECT_EQ( meshA, meshB );
    EXPECT_EQ( meshA->GetBuffer(), meshB->GetBuffer() );
}

TEST_F( ResourceManagerTests, DeferredUploadExecution )
{
    if( !device )
        GTEST_SKIP() << "Device not initialized";

    // 1. Request meshes (Queues upload lambda)
    AssetID cubeID   = resources->GetMeshID( "Cube" );
    AssetID sphereID = resources->GetMeshID( "Sphere" );

    // 2. Run a frame to execute the upload queue
    resources->BeginFrame( 0 );
    resources->EndFrame();

    // 3. Wait for GPU to confirm transfers completed
    streamer->WaitForTransferComplete();

    // If we reached here without validation errors or crashes, success.
    SUCCEED();
}