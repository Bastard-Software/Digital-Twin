#include "core/Log.hpp"
#include "runtime/Engine.hpp"
#include "simulation/Simulation.hpp"

// Compute headers
#include "compute/ComputeEngine.hpp"
#include "compute/ComputeGraph.hpp"
#include "compute/ComputeKernel.hpp"
#include <filesystem>
#include <fstream>

using namespace DigitalTwin;

// Helper to create a simple compute shader for testing
void CreateTestShaderFile( const std::string& filename )
{
    std::string source = R"(
        #version 450
        layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

        struct Cell {
            vec4 position;
            vec4 velocity;
            vec4 color;
        };

        // Binding 0: The cell buffer from SimulationContext
        layout(std430, set = 0, binding = 0) buffer CellBuffer {
            Cell cells[];
        } population;

        void main() {
            uint index = gl_GlobalInvocationID.x;
            
            // Simple logic: Move cell slightly along X axis based on velocity
            // This allows us to verify on CPU that data is actually changing
            population.cells[index].position.x += population.cells[index].velocity.x * 0.1;
        }
    )";

    std::ofstream out( filename );
    out << source;
    out.close();
}

int main()
{
    // 0. Create shader asset for the test
    const std::string shaderPath = "MoveCells.comp";
    CreateTestShaderFile( shaderPath );

    // 1. Configure Engine
    EngineConfig config;
    config.headless = false; // Set to true if you don't need a window
    config.width    = 1280;
    config.height   = 720;

    // 2. Initialize Runtime
    Engine engine;
    if( engine.Init( config ) != Result::SUCCESS )
    {
        DT_CORE_CRITICAL( "Engine failed to initialize!" );
        return -1;
    }

    // 3. Configure Simulation
    Simulation sim( engine );
    sim.SetMicroenvironment( 0.5f, 9.81f );

    // Spawn 10 cells for easy verification
    DT_CORE_INFO( "Spawning initial population..." );
    for( int i = 0; i < 10; ++i )
    {
        sim.SpawnCell( { ( float )i * 1.0f, 0.0f, 0.0f }, // Position X = 0, 1, 2...
                       { 1.0f, 0.0f, 0.0f },              // Velocity X = 1.0
                       { 1.0f, 0.0f, 0.0f, 1.0f }         // Color
        );
    }

    // 4. Upload Config to GPU
    sim.InitializeGPU();

    // ==========================================================================================
    // COMPUTE SETUP
    // ==========================================================================================

    // A. Init Compute Engine
    auto computeEngine = CreateRef<ComputeEngine>( engine.GetDevice() );
    computeEngine->Init();

    // B. Create Pipeline & Kernel
    auto                shader = engine.GetDevice()->CreateShader( shaderPath );
    ComputePipelineDesc pipeDesc;
    pipeDesc.shader = shader;
    auto pipeline   = engine.GetDevice()->CreateComputePipeline( pipeDesc );

    auto kernel = CreateRef<ComputeKernel>( engine.GetDevice(), pipeline, "MoveCells" );
    kernel->SetGroupSize( 256, 1, 1 ); // Matches layout(local_size_x = 256)

    // C. Create Binding Group (Link Simulation Buffer to Shader)
    // Note: "population" matches the instance name in the shader source above
    auto bindings = kernel->CreateBindingGroup();
    bindings->Set( "population", sim.GetContext()->GetCellBuffer() );
    bindings->Build();

    // D. Create Graph
    ComputeGraph graph;
    graph.AddTask( kernel, bindings );

    // ==========================================================================================
    // MAIN LOOP
    // ==========================================================================================

    DT_CORE_INFO( "[Sandbox] Starting Main Loop with Readback..." );

    bool running   = true;
    int  stepCount = 0;

    // Free simulation resources before shutdown
    {
        while( running && stepCount < 5 ) // Run for 5 steps then exit (for demo)
        {
            // 1. Window Events
            if( engine.GetWindow() )
            {
                engine.GetWindow()->OnUpdate();
                if( engine.GetWindow()->IsClosed() )
                    running = false;
            }

            // 2. Execute Compute Graph
            // Executes the kernel that moves cells
            uint64_t fenceValue = computeEngine->ExecuteGraph( graph, sim.GetContext()->GetMaxCellCount() );

            // 3. Sync (Wait for Compute to finish)
            computeEngine->WaitForTask( fenceValue );

            // 4. READBACK (Capture data from GPU to CPU)
            auto streamer = engine.GetStreamingManager();

            // Start transfer frame
            streamer->BeginFrame( engine.GetFrameCount() );

            // Request copy from GPU buffer to Staging buffer
            VkDeviceSize dataSize   = 10 * sizeof( Cell ); // Read first 10 cells
            auto         allocation = streamer->CaptureBuffer( sim.GetContext()->GetCellBuffer(), dataSize );

            // Submit transfer
            streamer->EndFrame();

            // BLOCK CPU until transfer is done (Immediate Readback Mode)
            streamer->WaitForTransferComplete();

            // 5. Verify Data on CPU
            Cell* cpuData = static_cast<Cell*>( allocation.mappedData );
            DT_CORE_INFO( "--- Step {} ---", stepCount );
            for( int i = 0; i < 3; ++i ) // Print first 3 cells
            {
                // Expected: Position X should increase by 0.1 each step (Velocity 1.0 * 0.1 factor in shader)
                DT_CORE_INFO( "  Cell[{}] PosX: {:.4f}", i, cpuData[ i ].position.x );
            }

            stepCount++;
            std::this_thread::sleep_for( std::chrono::milliseconds( 100 ) );
        }
    }

    // Cleanup
    std::filesystem::remove( shaderPath );

    return 0;
}