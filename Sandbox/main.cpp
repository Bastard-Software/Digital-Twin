#include "compute/ComputeKernel.hpp"
#include "core/FileSystem.hpp"
#include "core/Log.hpp"
#include "runtime/EntryPoint.hpp" // Hooks into the Engine's main()

using namespace DigitalTwin;

/**
 * @brief Experiment class based on the Application architecture.
 * Implements the exact logic from the working monolithic main.cpp.
 */
class Sandbox : public Application
{
public:
    Sandbox()
        : Application( { "Digital Twin Simulation", 1280, 720, false } )
    {
    }

    // --- PHASE 1: DATA SETUP ---
    // This runs before GPU initialization. We define the world here.
    void OnConfigureWorld() override
    {
        DT_CORE_INFO( "Setting up World Parameters & Agents (Logic from working main)..." );

        // 1. Get Assets
        // In Application, GetResourceManager returns a reference, not a pointer.
        AssetID sphereID = GetResourceManager().GetMeshID( "Sphere" );
        AssetID cubeID   = GetResourceManager().GetMeshID( "Cube" );

        // 2. Set Environment
        GetSimulation().SetMicroenvironment( 0.5f, 9.81f );

        // 3. Spawn Cells
        // Exact copy of the 4 cells from your working main.cpp

        // Red Cell (moving right)
        GetSimulation().SpawnCell( sphereID, glm::vec4( -4.0f, 0.0f, 0.0f, 1.0f ), glm::vec3( 2.0f, 0.5f, 0.0f ),
                                   glm::vec4( 1.0f, 0.2f, 0.2f, 1.0f ) );

        // Green Cell (moving left)
        GetSimulation().SpawnCell( cubeID, glm::vec4( 4.0f, 0.0f, 0.0f, 1.0f ), glm::vec3( -2.0f, -0.5f, 0.0f ),
                                   glm::vec4( 0.2f, 1.0f, 0.2f, 1.0f ) );

        // Blue Cell (moving down)
        GetSimulation().SpawnCell( sphereID, glm::vec4( 0.0f, 4.0f, 0.0f, 1.0f ), glm::vec3( 0.1f, -2.0f, 0.0f ),
                                   glm::vec4( 0.2f, 0.2f, 1.0f, 1.0f ) );

        // Yellow Cell (moving up)
        GetSimulation().SpawnCell( cubeID, glm::vec4( 0.0f, -4.0f, 0.0f, 1.0f ), glm::vec3( -0.1f, 2.0f, 0.0f ),
                                   glm::vec4( 1.0f, 1.0f, 0.2f, 1.0f ) );

        // NOTE: InitializeGPU() is called automatically by the base class immediately after this function.
    }

    // --- PHASE 2: PHYSICS SETUP ---
    // This runs after GPU buffers are created. We set up shaders here.
    void OnConfigurePhysics() override
    {
        DT_CORE_INFO( "Setting up Compute Pipelines (Logic from working main)..." );

        auto  device  = GetDevice();
        auto  context = GetSimulation().GetContext(); // Context is valid here
        auto& graph   = GetComputeGraph();

        // --- 1. Load Collision Kernel ---
        auto collShader = device->CreateShader( FileSystem::GetPath( "shaders/compute/solve_collisions.comp" ).string() );
        if( !collShader )
        {
            DT_CORE_CRITICAL( "Failed to load solve_collisions.comp shader!" );
            return;
        }

        ComputePipelineDesc collPipeDesc;
        collPipeDesc.shader = collShader;
        auto collPipeline   = device->CreateComputePipeline( collPipeDesc );

        auto collKernel = CreateRef<ComputeKernel>( device, collPipeline, "SolveCollisions" );
        collKernel->SetGroupSize( 256, 1, 1 );

        // Create Bindings
        // We bind ONLY "population" because that matches your working main.cpp.
        // We do NOT bind "meshInfo" to avoid validation errors if the shader doesn't expect it.
        auto collBindings = collKernel->CreateBindingGroup();
        if( collBindings )
        {
            collBindings->Set( "population", context->GetCellBuffer() );
            collBindings->Build();
        }
        else
        {
            DT_CORE_CRITICAL( "Failed to create bindings for Collision Kernel!" );
            return;
        }

        // --- 2. Load Movement Kernel ---
        auto moveShader = device->CreateShader( FileSystem::GetPath( "shaders/compute/move_cells.comp" ).string() );
        if( !moveShader )
        {
            DT_CORE_CRITICAL( "Failed to load move_cells.comp shader!" );
            return;
        }

        ComputePipelineDesc movePipeDesc;
        movePipeDesc.shader = moveShader;
        auto movePipeline   = device->CreateComputePipeline( movePipeDesc );

        auto moveKernel = CreateRef<ComputeKernel>( device, movePipeline, "MoveCells" );
        moveKernel->SetGroupSize( 256, 1, 1 );

        // Create Bindings
        auto moveBindings = moveKernel->CreateBindingGroup();
        if( moveBindings )
        {
            moveBindings->Set( "population", context->GetCellBuffer() );
            moveBindings->Build();
        }
        else
        {
            DT_CORE_CRITICAL( "Failed to create bindings for Movement Kernel!" );
            return;
        }

        // --- 3. Build the Compute Graph ---
        // Order: Solve Collisions -> Move Cells
        graph.AddTask( collKernel, collBindings );
        graph.AddTask( moveKernel, moveBindings );

        DT_CORE_INFO( "Compute Graph configured successfully." );
    }

    void OnGui() override
    {
        // Future UI code will go here
    }
};

// --- Entry Point Definition ---
// This factory function allows the Engine to instantiate our specific experiment.
DigitalTwin::Application* DigitalTwin::CreateApplication()
{
    return new Sandbox();
}