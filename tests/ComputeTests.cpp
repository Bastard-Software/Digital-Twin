#include "compute/ComputeEngine.hpp"
#include "compute/ComputeGraph.hpp"
#include "compute/ComputeKernel.hpp"
#include "core/Log.hpp"
#include "runtime/Engine.hpp"
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>

using namespace DigitalTwin;

class ComputeTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        Log::Init();
        EngineConfig config;
        config.headless = true;
        engine.Init( config );

        computeEngine = CreateRef<ComputeEngine>( engine.GetDevice() );
        computeEngine->Init();
    }

    void TearDown() override
    {
        computeEngine->Shutdown();
        engine.Shutdown();
    }

    std::string CreateTestShader()
    {
        std::string source = R"(
            #version 450
            layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

            layout(std430, set = 0, binding = 0) buffer DataBuffer {
                float values[];
            } data;

            void main() {
                uint index = gl_GlobalInvocationID.x;
                data.values[index] += 10.0;
            }
        )";

        std::string   filename = "TempComputeTest.comp";
        std::ofstream out( filename );
        out << source;
        out.close();
        return filename;
    }

    Engine             engine;
    Ref<ComputeEngine> computeEngine;
};

TEST_F( ComputeTest, BindingGroupWorkflow )
{
    auto device = engine.GetDevice();

    // 1. Create Buffer
    std::vector<float> initialData = { 1.0f, 2.0f, 3.0f, 4.0f };
    VkDeviceSize       dataSize    = initialData.size() * sizeof( float );

    BufferDesc bufDesc;
    bufDesc.size = dataSize;
    bufDesc.type = BufferType::STORAGE;
    auto buffer  = device->CreateBuffer( bufDesc );

    // Upload data (using StreamingManager)
    auto streamer = engine.GetStreamingManager();
    streamer->BeginFrame( 0 );
    streamer->UploadToBuffer( buffer, initialData.data(), dataSize, 0 );
    streamer->EndFrame();
    streamer->WaitForTransferComplete();

    // 2. Setup Kernel
    std::string shaderPath = CreateTestShader();
    auto        shader     = device->CreateShader( shaderPath );
    ASSERT_NE( shader, nullptr );

    ComputePipelineDesc pipeDesc;
    pipeDesc.shader = shader;
    auto pipeline   = device->CreateComputePipeline( pipeDesc );
    ASSERT_NE( pipeline, nullptr );

    auto kernel = CreateRef<ComputeKernel>( device, pipeline, "TestKernel" );
    kernel->SetGroupSize( 1, 1, 1 );

    // 3. Binding Group
    auto bindingGroup = kernel->CreateBindingGroup();
    ASSERT_NE( bindingGroup, nullptr );
    bindingGroup->Set( "data", buffer );
    bindingGroup->Build();

    // 4. Execute
    ComputeGraph graph;
    graph.AddTask( kernel, bindingGroup );

    uint64_t fence = computeEngine->ExecuteGraph( graph, 4 );

    // 5. Sync (This calls Device::WaitForQueue internally)
    computeEngine->WaitForTask( fence );

    // Cleanup
    std::filesystem::remove( shaderPath );

    ASSERT_TRUE( true );
}