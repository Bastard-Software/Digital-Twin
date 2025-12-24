#include "simulation/SimulationScheduler.hpp"

namespace DigitalTwin
{
    SimulationScheduler::SimulationScheduler( Ref<ComputeEngine> engine, Ref<SimulationContext> context )
        : m_computeEngine( engine )
        , m_context( context )
    {
        // Create the Global Uniform Buffer (Dynamic usage for frequent updates)
        BufferDesc desc{};
        desc.size = sizeof( GlobalContextData );
        desc.type = BufferType::UNIFORM;

        // Assuming device is accessible via engine or context
        m_globalUBO = engine->GetDevice()->CreateBuffer( desc );
    }

    void SimulationScheduler::AddSystem( const std::string& name, ComputeGraph graph, float interval )
    {
        SimulationPass pass;
        pass.name        = name;
        pass.graph       = graph;
        pass.interval    = interval;
        pass.accumulator = 0.0f;
        m_passes.push_back( pass );
    }

    void SimulationScheduler::Tick( float realDt )
    {
        // 1. Update time controller
        m_timeCtrl.Update( realDt );

        // If no agents or no passes, do nothing
        if( m_passes.empty() || m_context->GetMaxCellCount() == 0 )
            return;

        float availableSimTime = m_timeCtrl.GetSimDeltaTime();

        // 2. Iterate over all registered systems
        for( auto& pass: m_passes )
        {
            if( !pass.enabled )
                continue;

            pass.accumulator += availableSimTime;

            // 3. Fixed Update Loop: Catch up with the accumulated time
            while( pass.accumulator >= pass.interval )
            {
                // Update the Global UBO with the specific 'dt' for this system
                UpdateGlobalUBO( pass.interval );

                // Execute the Compute Graph for the current population
                uint32_t count      = m_context->GetMaxCellCount();
                m_lastComputeSignal = m_computeEngine->ExecuteGraph( pass.graph, count );

                // Consume time
                pass.accumulator -= pass.interval;
            }
        }
    }

    void SimulationScheduler::UpdateGlobalUBO( float currentStepDt )
    {
        GlobalContextData gpuData;
        gpuData.time      = m_timeCtrl.GetSimTime();
        gpuData.dt        = currentStepDt; // IMPORTANT: The shader sees the fixed step
        gpuData.timeScale = m_timeCtrl.GetTimeScale();
        gpuData.frame     = m_timeCtrl.GetFrameIndex();
        gpuData.worldSize = 20.0f; // TODO: Retrieve from Simulation config

        // Use the direct Write method as requested
        if( m_globalUBO )
        {
            m_globalUBO->Write( &gpuData, sizeof( GlobalContextData ), 0 );
        }
    }
} // namespace DigitalTwin
