#include "simulation/Simulation.hpp"

#include "compute/ComputeEngine.hpp"
#include "core/Log.hpp"
#include "runtime/Engine.hpp"
#include "simulation/SimulationContext.hpp"

namespace DigitalTwin
{
    // Must match Shader layout (std140)
    struct GlobalData
    {
        float     dt;
        float     time;
        glm::vec2 resolution;
        float     viscosity;
        float     gravity;
        float     padding[ 2 ];
    };

    Simulation::Simulation() = default;

    Simulation::~Simulation()
    {
        // FIX: Wait for GPU to finish before destroying resources
        if( m_engineRef )
        {
            m_engineRef->GetDevice()->WaitIdle();
        }

        if( m_computeEngine )
            m_computeEngine->Shutdown();
        m_globalUBO.reset();
        m_context.reset();
    }

    void Simulation::InitializeRuntime( Engine& engine, Ref<ComputeEngine> computeEngine )
    {
        m_engineRef     = &engine;
        m_computeEngine = computeEngine;

        DT_CORE_INFO( "[Simulation] Initializing Runtime..." );

        m_context = CreateRef<SimulationContext>( engine.GetDevice() );

        // 1. Configure World
        OnConfigureWorld();

        // 2. Init Buffers
        uint32_t capacity = std::max( ( uint32_t )m_initialCells.size(), 1024u );
        m_context->Init( capacity );

        // 3. Create UBO
        BufferDesc uboDesc{};
        uboDesc.size = sizeof( GlobalData );
        uboDesc.type = BufferType::UNIFORM;
        m_globalUBO  = engine.GetDevice()->CreateBuffer( uboDesc );

        // 4. Upload Initial State
        // FIX: We must Start a Frame to record Copy Commands!
        auto resMgr   = engine.GetResourceManager();
        auto streamer = engine.GetStreamingManager();

        resMgr->BeginFrame( 0 ); // Start Recording CommandBuffer
        {
            m_context->UploadState( streamer.get(), m_initialCells );

            // Upload initial Global Data as well to avoid empty UBO validation errors
            UpdateGlobalData( 0.0f );
        }
        resMgr->EndFrame(); // Submit Recording

        streamer->WaitForTransferComplete();

        // 5. Configure Systems (Builds Graphs)
        OnConfigureSystems();

        DT_CORE_INFO( "[Simulation] Ready. Systems: {}", m_systems.size() );
    }

    void Simulation::ShutdownRuntime()
    {
    }

    void Simulation::RegisterSystem( const std::string& name, GraphBuilder builder, float interval )
    {
        DT_CORE_INFO( "[Simulation] Registering System: {}", name );

        SystemInstance sys;
        sys.name     = name;
        sys.interval = interval;

        // --- Double Build ---
        // Pass 0
        m_context->SetFrameIndex( 0 );
        sys.graphs[ 0 ] = builder( *m_context );

        // Pass 1
        m_context->SetFrameIndex( 1 );
        sys.graphs[ 1 ] = builder( *m_context );

        // Reset
        m_context->SetFrameIndex( 0 );

        m_systems.push_back( std::move( sys ) );
    }

    void Simulation::Tick( float realDt )
    {
        float dt = realDt * m_timeScale;
        OnUpdate( realDt );

        if( m_paused || !m_computeEngine )
            return;

        m_totalTime += dt;

        // Update UBO (Streamer inside handles command recording usually,
        // but if it uses vkCmdCopyBuffer directly it relies on the frame being active.
        // Application::Run() usually wraps Tick() in BeginFrame/EndFrame, so this is safe here.)
        UpdateGlobalData( dt );

        uint32_t frameIdx    = m_context->GetFrameIndex();
        bool     anyExecuted = false;

        for( auto& sys: m_systems )
        {
            sys.timer += dt;
            if( sys.timer >= sys.interval )
            {
                sys.timer -= sys.interval;

                ComputeGraph& activeGraph = sys.graphs[ frameIdx ];
                if( !activeGraph.IsEmpty() )
                {
                    m_lastComputeSignal = m_computeEngine->ExecuteGraph( activeGraph, m_context->GetMaxCellCount() );
                    anyExecuted         = true;
                }
            }
        }

        if( anyExecuted )
        {
            m_context->SwapBuffers();
        }
    }

    void Simulation::UpdateGlobalData( float dt )
    {
        if( !m_globalUBO )
            return;
        GlobalData data{};
        data.dt        = dt;
        data.time      = m_totalTime;
        data.viscosity = m_envParams.viscosity;
        data.gravity   = m_envParams.gravity;

        auto streamer = m_engineRef->GetStreamingManager();
        streamer->UploadToBuffer( m_globalUBO, &data, sizeof( GlobalData ), 0 );
    }

    // --- Helpers ---

    void Simulation::SetMicroenvironment( float v, float g )
    {
        m_envParams.viscosity = v;
        m_envParams.gravity   = g;
    }

    void Simulation::SpawnCell( uint32_t meshID, glm::vec4 pos, glm::vec3 vel, glm::vec4 color )
    {
        m_initialCells.push_back( { pos, glm::vec4( vel, 0.0f ), color, meshID } );
        bool found = false;
        for( auto id: m_activeMeshes )
            if( id == meshID )
                found = true;
        if( !found )
            m_activeMeshes.push_back( meshID );
    }

    SimulationContext* Simulation::GetContext()
    {
        return m_context.get();
    }
    Ref<Buffer> Simulation::GetGlobalUniformBuffer()
    {
        return m_globalUBO;
    }
    Ref<Device> Simulation::GetDevice()
    {
        return m_engineRef->GetDevice();
    }

    const std::vector<uint32_t>& Simulation::GetActiveMeshes() const
    {
        return m_activeMeshes;
    }
    uint64_t Simulation::GetComputeSignal() const
    {
        return m_lastComputeSignal;
    }

    AssetID Simulation::GetMeshID( const std::string& name ) const
    {
        return m_engineRef->GetResourceManager()->GetMeshID( name );
    }

    void Simulation::SetTimeScale( float scale )
    {
        m_timeScale = scale;
    }
    float Simulation::GetTimeScale() const
    {
        return m_timeScale;
    }
    void Simulation::Pause()
    {
        m_paused = true;
    }
    void Simulation::Resume()
    {
        m_paused = false;
    }
} // namespace DigitalTwin