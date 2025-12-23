#include "simulation/Simulation.hpp"

#include "core/Log.hpp"
#include "runtime/Engine.hpp"
#include "simulation/SimulationContext.hpp"
#include "simulation/SimulationScheduler.hpp"

namespace DigitalTwin
{
    Simulation::Simulation() = default;

    Simulation::~Simulation()
    {
        if( m_computeEngine )
        {
            m_computeEngine->Shutdown();
        }
        DT_CORE_INFO( "[Simulation] Shutting down..." );
        m_engineRef->GetDevice()->ResetDescriptorPools();
        m_scheduler.reset();
        m_context.reset();
    }

    void Simulation::InitializeRuntime( Engine& engine, Ref<ComputeEngine> computeEngine )
    {
        m_engineRef     = &engine;
        m_computeEngine = computeEngine;

        DT_CORE_INFO( "[Simulation] Initializing Runtime Environment..." );

        // 1. Create the Simulation Context (Vulkan Buffers Wrapper)
        m_context = CreateRef<SimulationContext>( engine.GetDevice() );

        // 2. Call User Code: Configure World
        // This populates m_initialCells and environment params on the CPU
        OnConfigureWorld();

        // 3. Initialize GPU Memory
        // Ensure at least 1024 cells capacity to prevent empty buffer errors
        uint32_t capacity = std::max( ( uint32_t )m_initialCells.size(), 1024u );
        m_context->Init( capacity );

        // 4. Create the Scheduler
        // The scheduler needs the Compute Engine to dispatch jobs and Context to access data
        m_scheduler = CreateScope<SimulationScheduler>( computeEngine, m_context );

        // 5. Upload Initial State to GPU
        auto resMgr   = engine.GetResourceManager();
        auto streamer = engine.GetStreamingManager();

        DT_CORE_INFO( "[Simulation] Uploading {} agents to GPU...", m_initialCells.size() );

        resMgr->BeginFrame( 0 );
        m_context->UploadState( streamer.get(), m_initialCells );
        resMgr->EndFrame();

        // Block until upload is complete to ensure validity before first tick
        streamer->WaitForTransferComplete();

        // 6. Call User Code: Configure Systems
        // User registers compute graphs here
        OnConfigureSystems();

        DT_CORE_INFO( "[Simulation] Initialization Complete." );
    }

    void Simulation::Tick( float realDt )
    {
        // 1. Execute User CPU Logic
        OnUpdate( realDt );

        // 2. Execute Scheduler (GPU Logic)
        if( m_scheduler )
        {
            m_scheduler->Tick( realDt );
        }
    }

    // --- Protected User API Implementation ---

    void Simulation::SetMicroenvironment( float viscosity, float gravity )
    {
        m_envParams.viscosity = viscosity;
        m_envParams.gravity   = gravity;
    }

    void Simulation::SpawnCell( uint32_t meshID, glm::vec4 pos, glm::vec3 vel, glm::vec4 color )
    {
        Cell c{};
        c.position = pos;
        c.velocity = glm::vec4( vel, 0.0f );
        c.color    = color;
        c.meshID   = meshID;

        m_initialCells.push_back( c );

        // Track active meshes for the Renderer
        bool exists = false;
        for( auto id: m_activeMeshes )
        {
            if( id == meshID )
                exists = true;
        }
        if( !exists )
            m_activeMeshes.push_back( meshID );
    }

    void Simulation::RegisterSystem( const std::string& name, ComputeGraph graph, float interval )
    {
        if( m_scheduler )
        {
            m_scheduler->AddSystem( name, graph, interval );
            DT_CORE_INFO( "[Simulation] System Registered: '{}' @ {:.2f} ms", name, interval * 1000.0f );
        }
        else
        {
            DT_CORE_ERROR( "[Simulation] Cannot register system '{}'. Scheduler is null!", name );
        }
    }

    void Simulation::SetTimeScale( float scale )
    {
        if( m_scheduler )
            m_scheduler->GetTimeController().SetTimeScale( scale );
    }

    float Simulation::GetTimeScale() const
    {
        return m_scheduler ? m_scheduler->GetTimeController().GetTimeScale() : 1.0f;
    }

    void Simulation::Pause()
    {
        SetTimeScale( 0.0f );
    }

    void Simulation::Resume()
    {
        SetTimeScale( 1.0f );
    }

    SimulationContext* Simulation::GetContext()
    {
        return m_context.get();
    }

    std::shared_ptr<Buffer> Simulation::GetGlobalUniformBuffer()
    {
        return m_scheduler ? m_scheduler->GetGlobalBuffer() : nullptr;
    }

    const std::vector<uint32_t>& Simulation::GetActiveMeshes() const
    {
        return m_activeMeshes;
    }

    uint64_t Simulation::GetComputeSignal() const
    {
        return m_scheduler ? m_scheduler->GetLastComputeSignal() : 0;
    }

    AssetID Simulation::GetMeshID( const std::string& name ) const
    {
        DT_CORE_ASSERT( m_engineRef, "Engine reference is null! Cannot access resources." );

        return m_engineRef->GetResourceManager()->GetMeshID( name );
    }

} // namespace DigitalTwin