#include "core/FileSystem.hpp"
#include "core/Log.hpp"
#include "simulation/Simulation.hpp"
#include "simulation/SimulationContext.hpp"
#include "simulation/SystemBindings.hpp"

// CRITICAL: This header defines main() and starts the engine loop!
#include "runtime/EntryPoint.hpp"

using namespace DigitalTwin;

class Sandbox : public Simulation
{
public:
    void OnConfigureWorld() override
    {
        DT_CORE_INFO( "Sandbox: Spawning Cells..." );
        SetMicroenvironment( 0.5f, 9.81f );

        AssetID sphere = GetMeshID( "Sphere" );
        AssetID cube   = GetMeshID( "Cube" );

        SpawnCell( sphere, { -2.0f, 10.0f, 0.0f, 1.0f }, { 0, -2, 0 }, { 1.0f, 0.1f, 0.1f, 1.0f } );
        SpawnCell( sphere, { 2.0f, 12.0f, 0.0f, 1.0f }, { 0, -1, 0 }, { 0.1f, 1.0f, 0.1f, 1.0f } );
        SpawnCell( sphere, { 0.0f, 14.0f, -2.0f, 1.0f }, { 0, -3, 0 }, { 0.1f, 0.1f, 1.0f, 1.0f } );
        SpawnCell( cube, { 0.5f, 18.0f, 1.0f, 1.0f }, { 0, -5, 0 }, { 1.0f, 1.0f, 0.0f, 1.0f } );
    }

    void OnConfigureSystems() override
    {
        auto context = GetContext();
        auto device  = GetDevice();
        auto ubo     = GetGlobalUniformBuffer();

        // 1. Kernels
        auto collKernel = CreateKernel( "shaders/compute/solve_collisions.comp", "Collisions" );
        auto moveKernel = CreateKernel( "shaders/compute/move_cells.comp", "Movement" );

        // 2. Register
        RegisterSystem(
            "Physics",
            [ & ]( SimulationContext& ctx ) -> ComputeGraph {
                ComputeGraph graph;
                uint32_t     frameIdx = ctx.GetFrameIndex();

                // Task 1
                auto collBind = ctx.CreateSystemBindings( collKernel );
                collBind->SetUniform( "u_Global", ubo );
                collBind->SetInput( "InPopulation" );
                collBind->SetOutput( "OutPopulation" );
                collBind->Build();

                // EXTRACT RAW GROUP
                graph.AddTask( collKernel, collBind->Get( frameIdx ) );

                // Task 2
                auto moveBind = ctx.CreateSystemBindings( moveKernel );
                moveBind->SetUniform( "u_Global", ubo );
                moveBind->SetOutput( "OutPopulation" );
                moveBind->Build();

                // EXTRACT RAW GROUP
                graph.AddTask( moveKernel, moveBind->Get( frameIdx ) );

                return graph;
            },
            1.0f / 60.0f );
    }

private:
    Ref<ComputeKernel> CreateKernel( const std::string& path, const std::string& name )
    {
        auto dev    = GetDevice();
        auto shader = dev->CreateShader( FileSystem::GetPath( path ).string() );
        auto pipe   = dev->CreateComputePipeline( { shader } );
        return CreateRef<ComputeKernel>( dev, pipe, name );
    }
};

// Factory function expected by EntryPoint.hpp
Simulation* DigitalTwin::CreateSimulation()
{
    return new Sandbox();
}