#include "compute/ComputeKernel.hpp"
#include "core/FileSystem.hpp"
#include "core/Log.hpp"
#include "runtime/EntryPoint.hpp" // Hooks into the Engine's main()
#include "simulation/Simulation.hpp"
#include "simulation/SimulationContext.hpp"

using namespace DigitalTwin;

/**
 * @brief Experiment class based on the NEW Simulation architecture.
 * Implements the exact logic from the working main.cpp but using the new API.
 */
class Sandbox : public Simulation
{
public:
    Sandbox() = default;

    // --- PHASE 1: DATA SETUP ---
    // This runs before GPU initialization. We define the world here.
    void OnConfigureWorld() override
    {
        DT_CORE_INFO( "Setting up World Parameters & Agents..." );

        // 1. Get Assets
        AssetID sphereID = GetMeshID( "Sphere" );
        AssetID cubeID   = GetMeshID( "Cube" );

        // 2. Set Environment
        SetMicroenvironment( 0.5f, 9.81f );

        // 3. Spawn Cells
        // Red Cell (moving right)
        SpawnCell( sphereID, glm::vec4( -4.0f, 0.0f, 0.0f, 1.0f ), glm::vec3( 2.0f, 0.5f, 0.0f ), glm::vec4( 1.0f, 0.2f, 0.2f, 1.0f ) );

        // Green Cell (moving left)
        SpawnCell( cubeID, glm::vec4( 4.0f, 0.0f, 0.0f, 1.0f ), glm::vec3( -2.0f, -0.5f, 0.0f ), glm::vec4( 0.2f, 1.0f, 0.2f, 1.0f ) );

        // Blue Cell (moving down)
        SpawnCell( sphereID, glm::vec4( 0.0f, 4.0f, 0.0f, 1.0f ), glm::vec3( 0.1f, -2.0f, 0.0f ), glm::vec4( 0.2f, 0.2f, 1.0f, 1.0f ) );

        // Yellow Cell (moving up)
        SpawnCell( cubeID, glm::vec4( 0.0f, -4.0f, 0.0f, 1.0f ), glm::vec3( -0.1f, 2.0f, 0.0f ), glm::vec4( 1.0f, 1.0f, 0.2f, 1.0f ) );
    }

    // --- PHASE 2: PHYSICS SETUP ---
    // This runs after GPU buffers are created. We set up shaders here.
    void OnConfigureSystems() override
    {
        DT_CORE_INFO( "Setting up Compute Pipelines..." );

        auto context   = GetContext();
        auto device    = context->GetDevice(); // Teraz to zadzia³a!
        auto globalUBO = GetGlobalUniformBuffer();

        ComputeGraph graph;

        // --- 1. Load Collision Kernel ---
        auto collShader = device->CreateShader( FileSystem::GetPath( "shaders/compute/solve_collisions.comp" ).string() );
        if( !collShader )
        {
            DT_CORE_CRITICAL( "Failed to load solve_collisions.comp!" );
            return;
        }

        ComputePipelineDesc collDesc;
        collDesc.shader = collShader;
        auto collPipe   = device->CreateComputePipeline( collDesc );
        auto collKernel = CreateRef<ComputeKernel>( device, collPipe, "Collisions" );

        auto collBind = collKernel->CreateBindingGroup();
        collBind->Set( "GlobalData", globalUBO );                // Set 0
        collBind->Set( "population", context->GetCellBuffer() ); // Set 1
        collBind->Build();

        graph.AddTask( collKernel, collBind );

        // --- 2. Load Movement Kernel ---
        auto moveShader = device->CreateShader( FileSystem::GetPath( "shaders/compute/move_cells.comp" ).string() );
        if( !moveShader )
        {
            DT_CORE_CRITICAL( "Failed to load move_cells.comp!" );
            return;
        }

        ComputePipelineDesc moveDesc;
        moveDesc.shader = moveShader;
        auto movePipe   = device->CreateComputePipeline( moveDesc );
        auto moveKernel = CreateRef<ComputeKernel>( device, movePipe, "Movement" );

        auto moveBind = moveKernel->CreateBindingGroup();
        moveBind->Set( "GlobalData", globalUBO );                // Set 0
        moveBind->Set( "population", context->GetCellBuffer() ); // Set 1
        moveBind->Build();

        graph.AddTask( moveKernel, moveBind );

        // Register the system to run at 60Hz
        RegisterSystem( "Physics", graph, 1.0f / 60.0f );
    }

    void OnUpdate( float dt ) override
    {
        // Example: Pause after 10 seconds
        static float timer = 0.0f;
        timer += dt;
    }

    void OnRenderGui() override
    {
        // ImGui::Text("Active Agents: %d", GetContext()->GetMaxCellCount());
    }
};

// --- Entry Point Definition ---
// This factory function allows the Engine to instantiate our specific experiment.
DigitalTwin::Simulation* DigitalTwin::CreateSimulation()
{
    return new Sandbox();
}