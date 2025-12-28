#include "compute/ComputeEngine.hpp"

#include "core/Log.hpp"

namespace DigitalTwin
{
    ComputeEngine::ComputeEngine( Ref<Device> device )
        : m_device( device )
    {
    }

    ComputeEngine::~ComputeEngine()
    {
        Shutdown();
    }

    void ComputeEngine::Init()
    {
        DT_CORE_INFO( "[ComputeEngine] Initialized." );
    }

    void ComputeEngine::Shutdown()
    {
        // Force wait for all pending work before destroying command buffers
        if( !m_inflightWork.empty() )
        {
            uint64_t lastValue = m_inflightWork.back().fenceValue;
            WaitForTask( lastValue );
        }
        
        m_inflightWork.clear();
    }

    uint64_t ComputeEngine::ExecuteGraph( ComputeGraph& graph, uint32_t agentCount )
    {
        if( graph.IsEmpty() )
        {
            DT_CORE_WARN( "[ComputeEngine] Skipping execution of empty graph." );
            return 0;
        }

        // 0. Opportunistic cleanup of finished tasks
        GarbageCollect();

        // 1. Create Command Buffer
        Ref<CommandBuffer> cmd = m_device->CreateCommandBuffer( QueueType::COMPUTE );

        // 2. Record Commands
        cmd->Begin();
        graph.Record( *cmd, agentCount );
        cmd->End();

        // 3. Get Compute Queue
        Ref<Queue> queue = m_device->GetComputeQueue();
        if( !queue )
        {
            DT_CORE_CRITICAL( "[ComputeEngine] No Compute Queue available!" );
            return 0;
        }

        // 4. Submit
        uint64_t signalValue = 0;
        if( queue->Submit( cmd->GetHandle(), signalValue ) != Result::SUCCESS )
        {
            DT_CORE_ERROR( "[ComputeEngine] Submit failed!" );
            return 0;
        }

        // 5. Extend Lifetime: Store cmd until signalValue is reached
        m_inflightWork.push_back( { signalValue, cmd } );

        return signalValue;
    }

    void ComputeEngine::WaitForTask( uint64_t taskID )
    {
        Ref<Queue> queue = m_device->GetComputeQueue();
        if( queue )
        {
            // Block CPU
            m_device->WaitForQueue( queue, taskID );

            // Cleanup now that we know GPU is done
            GarbageCollect();
        }
    }

    void ComputeEngine::GarbageCollect()
    {
        Ref<Queue> queue = m_device->GetComputeQueue();
        if( !queue || m_inflightWork.empty() )
            return;

        // Check which buffers are finished and remove them
        // Timeline semaphores are monotonic, so we check from the front
        while( !m_inflightWork.empty() )
        {
            const auto& work = m_inflightWork.front();

            if( queue->IsValueCompleted( work.fenceValue ) )
            {
                // GPU finished this task.
                // Removing it from deque destroys the Ref<CommandBuffer>,
                // which calls vkFreeCommandBuffers. This is safe now.
                m_inflightWork.pop_front();
            }
            else
            {
                // If the oldest task isn't done, newer ones aren't either.
                break;
            }
        }
    }

} // namespace DigitalTwin