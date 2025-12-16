#include "core/FileSystem.hpp"
#include "core/Log.hpp"
#include "runtime/Engine.hpp"
#include "simulation/Simulation.hpp"

// Compute
#include "compute/ComputeEngine.hpp"
#include "compute/ComputeGraph.hpp"
#include "compute/ComputeKernel.hpp"

// Renderer
#include "platform/Input.hpp"
#include "renderer/Renderer.hpp"
#include "renderer/Scene.hpp"
#include <filesystem>
#include <thread>

using namespace DigitalTwin;

int main()
{
    // 1. Set up config
    EngineConfig config;
    config.headless = false;
    config.width    = 1280;
    config.height   = 720;

    // 2. Initialize Engine (Window, Device, Vulkan)
    Engine engine;
    if( engine.Init( config ) != Result::SUCCESS )
        return -1;

    {
        // RAII Scope: Ensures Simulation/Renderer are destroyed BEFORE Engine shutdown

        // --- Simulation Setup ---
        // Initialize the simulation world
        Simulation sim( engine );
        auto       resMgr = engine.GetResourceManager();

        // Spawn 4 cells for testing collision logic.
        // Arguments:
        // 1. Position (Vec4): x, y, z, radius (w)
        // 2. Velocity (Vec3): x, y, z
        // 3. Color    (Vec4): r, g, b, a

        AssetID sphereID = resMgr->GetMeshID( "Sphere" );
        AssetID cubeID   = resMgr->GetMeshID( "Cube" );

        // Red Cell (moving right)
        sim.SpawnCell( sphereID, glm::vec4( -4.0f, 0.0f, 0.0f, 1.0f ), glm::vec3( 2.0f, 0.5f, 0.0f ), glm::vec4( 1.0f, 0.2f, 0.2f, 1.0f ) );

        // Green Cell (moving left)
        sim.SpawnCell( cubeID, glm::vec4( 4.0f, 0.0f, 0.0f, 1.0f ), glm::vec3( -2.0f, -0.5f, 0.0f ), glm::vec4( 0.2f, 1.0f, 0.2f, 1.0f ) );

        // Blue Cell (moving down)
        sim.SpawnCell( sphereID, glm::vec4( 0.0f, 4.0f, 0.0f, 1.0f ), glm::vec3( 0.1f, -2.0f, 0.0f ), glm::vec4( 0.2f, 0.2f, 1.0f, 1.0f ) );

        // Yellow Cell (moving up)
        sim.SpawnCell( cubeID, glm::vec4( 0.0f, -4.0f, 0.0f, 1.0f ), glm::vec3( -0.1f, 2.0f, 0.0f ), glm::vec4( 1.0f, 1.0f, 0.2f, 1.0f ) );

        // Upload initial data to GPU
        sim.InitializeGPU();

        // --- Compute Engine Setup ---
        auto computeEngine = CreateRef<ComputeEngine>( engine.GetDevice() );
        computeEngine->Init();

        // 1. Load the collision kernel
        auto collShader = engine.GetDevice()->CreateShader( FileSystem::GetPath( "shaders/compute/solve_collisions.comp" ).string() );
        if( !collShader )
        {
            DT_CORE_CRITICAL( "Failed to load solve_collisions.comp shader!" );
            return -1;
        }

        ComputePipelineDesc pipeDesc;
        pipeDesc.shader   = collShader;
        auto collPipeline = engine.GetDevice()->CreateComputePipeline( pipeDesc );

        auto collKernel = CreateRef<ComputeKernel>( engine.GetDevice(), collPipeline, "SolveCollisions" );
        collKernel->SetGroupSize( 256, 1, 1 );

        // Bind the Simulation Data (SSBO) to the Compute Shader
        auto collBindings = collKernel->CreateBindingGroup();
        collBindings->Set( "population", sim.GetContext()->GetCellBuffer() );
        collBindings->Build();

        // 2. Load the physics kernel
        auto moveShader = engine.GetDevice()->CreateShader( FileSystem::GetPath( "shaders/compute/move_cells.comp" ).string() );
        if( !moveShader )
        {
            DT_CORE_CRITICAL( "Failed to load move_cells.comp shader!" );
            return -1;
        }

        pipeDesc.shader   = moveShader;
        auto movePipeline = engine.GetDevice()->CreateComputePipeline( pipeDesc );

        auto moveKernel = CreateRef<ComputeKernel>( engine.GetDevice(), movePipeline, "MoveCells" );
        moveKernel->SetGroupSize( 256, 1, 1 );

        // Bind the Simulation Data (SSBO) to the Compute Shader
        auto moveBindings = moveKernel->CreateBindingGroup();
        moveBindings->Set( "population", sim.GetContext()->GetCellBuffer() );
        moveBindings->Build();

        ComputeGraph graph;
        graph.AddTask( collKernel, collBindings );
        graph.AddTask( moveKernel, moveBindings );

        // --- Renderer Setup ---
        Renderer renderer( engine );
        renderer.GetCamera().SetDistance( 20.0f ); // Zoom out to see the whole arena

        // Show the window now that everything is loaded
        engine.GetWindow()->Show();

        DT_CORE_INFO( "[Sandbox] Starting Interactive Loop... Close window to exit." );

        bool running = true;

        // --- Main Loop ---
        while( running )
        {
            // 1. Reset per-frame input states (e.g. scroll delta)
            Input::ResetScroll();

            // 2. Poll Window Events (Keyboard, Mouse, Resize, Close)
            if( engine.GetWindow() )
            {
                engine.GetWindow()->OnUpdate();
                if( engine.GetWindow()->IsClosed() )
                    running = false;
            }

            engine.GetResourceManager()->BeginFrame( engine.GetFrameCount() );

            // 3. Update Camera (Process Input for Orbit/Zoom)
            renderer.OnUpdate( 0.016f );

            // 4. Physics Step (Compute Shader)
            // ExecuteGraph returns a timeline value we can wait on
            uint64_t fenceValue = computeEngine->ExecuteGraph( graph, sim.GetContext()->GetMaxCellCount() );

            // 5. Render Step
            // Prepare scene description
            Scene scene;
            scene.camera         = &renderer.GetCamera();
            scene.instanceBuffer = sim.GetContext()->GetCellBuffer();
            scene.instanceCount  = sim.GetContext()->GetMaxCellCount();
            scene.activeMeshIDs  = sim.GetActiveMeshes();

            // Submit render commands, synchronizing with the Compute Queue
            auto      computeQueue = engine.GetDevice()->GetComputeQueue();
            SyncPoint resSync      = engine.GetResourceManager()->EndFrame();

            std::vector<VkSemaphore> waitSems = { computeQueue->GetTimelineSemaphore() };
            std::vector<uint64_t>    waitVals = { fenceValue };

            if( resSync.semaphore )
            {
                waitSems.push_back( resSync.semaphore );
                waitVals.push_back( resSync.value );
            }

            renderer.Render( scene, { computeQueue->GetTimelineSemaphore() }, { fenceValue } );

            // 6. Frame Pacing (~60 FPS cap)
            std::this_thread::sleep_for( std::chrono::milliseconds( 16 ) );
        }
    }

    return 0;
}