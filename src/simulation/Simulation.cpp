#include "simulation/Simulation.hpp"

#include "core/Log.hpp"
#include "runtime/Engine.hpp"

namespace DigitalTwin
{
    Simulation::Simulation( Engine& engine )
        : m_engine( engine )
    {
        // Simulation owns the context
        m_context = CreateScope<SimulationContext>( engine.GetDevice() );
    }

    Simulation::~Simulation()
    {
        if( m_context )
            m_context->Shutdown();
    }

    void Simulation::SetMicroenvironment( float viscosity, float gravity )
    {
        m_envParams.viscosity = viscosity;
        m_envParams.gravity   = gravity;
    }

    void Simulation::SpawnCell( AssetID meshID, glm::vec4 position, glm::vec3 velocity, glm::vec4 phenotypeColor )
    {
        Cell cell{};
        cell.position = position; // w = radius
        cell.velocity = glm::vec4( velocity, 0.0f );
        cell.color    = phenotypeColor;
        cell.meshID   = meshID;

        m_initialCells.push_back( cell );

        bool exists = false;
        for( auto id: m_activeMeshes )
            if( id == meshID )
                exists = true;
        if( !exists )
            m_activeMeshes.push_back( meshID );
    }

    void Simulation::InitializeGPU()
    {
        DT_CORE_INFO( "[Simulation] Initializing GPU resources for {} cells...", m_initialCells.size() );

        // 1. Allocate GPU memory
        // Allocate at least 1024 or the size of initial cells to avoid 0-size buffers
        uint32_t capacity = std::max( ( uint32_t )m_initialCells.size(), 1024u );
        m_context->Init( capacity );

        // 2. Upload Initial State (Cells + Atomic Counter)
        auto resMgr   = m_engine.GetResourceManager();
        auto streamer = m_engine.GetStreamingManager();

        resMgr->BeginFrame( 0 ); // Use frame 0 slot for init
        m_context->UploadState( streamer.get(), m_initialCells );
        resMgr->EndFrame();

        // 3. Wait for upload to ensure GPU is ready before first compute dispatch
        streamer->WaitForTransferComplete();

        DT_CORE_INFO( "[Simulation] GPU Initialization Complete. Active Agents: {}", m_initialCells.size() );

        // Optional: clear CPU memory if not needed anymore
        // m_initialCells.clear();
    }

    void Simulation::Step( float dt )
    {
        // Placeholder for Compute Dispatch
        // In the next step (Compute Kernel), we will invoke the compute system here.
        // m_computeSystem->Dispatch(m_context, dt);
    }
} // namespace DigitalTwin