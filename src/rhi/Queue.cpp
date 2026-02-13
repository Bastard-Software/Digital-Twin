#include "rhi/Queue.h"

#include "core/Log.h"
#include "rhi/CommandBuffer.h"

namespace DigitalTwin
{
    Queue::Queue( VkDevice device, const VolkDeviceTable& api, uint32_t familyIndex, QueueType type )
        : m_device( device )
        , m_api( api )
        , m_queue( VK_NULL_HANDLE )
        , m_familyIndex( familyIndex )
        , m_type( type )
        , m_timelineSemaphore( VK_NULL_HANDLE )
    {
        // Retrieve the queue handle using the provided device table
        // We assume queue index 0 for now (simplification)
        m_api.vkGetDeviceQueue( m_device, m_familyIndex, 0, &m_queue );

        // Create the Timeline Semaphore
        VkSemaphoreTypeCreateInfo timelineCreateInfo = { VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO };
        timelineCreateInfo.semaphoreType             = VK_SEMAPHORE_TYPE_TIMELINE;
        timelineCreateInfo.initialValue              = 0;

        VkSemaphoreCreateInfo createInfo = { VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
        createInfo.pNext                 = &timelineCreateInfo;

        // Use the function pointer from the table
        VkResult res = m_api.vkCreateSemaphore( m_device, &createInfo, nullptr, &m_timelineSemaphore );
        if( res != VK_SUCCESS )
        {
            DT_ERROR( "Failed to create Timeline Semaphore for Queue!" );
        }
    }

    Queue::~Queue()
    {
        if( m_timelineSemaphore != VK_NULL_HANDLE )
        {
            m_api.vkDestroySemaphore( m_device, m_timelineSemaphore, nullptr );
        }
    }

    Result Queue::Submit( const std::vector<CommandBuffer*>& cmdBuffers, const std::vector<VkSemaphore>& waitSemaphores,
                          const std::vector<uint64_t>& waitValues, const std::vector<VkSemaphore>& signalSemaphores,
                          const std::vector<uint64_t>& signalValues )
    {
        // 1. Prepare Signals: Include internal Timeline Semaphore
        std::vector<VkSemaphore> finalSignalSemas = signalSemaphores;
        std::vector<uint64_t>    finalSignalVals  = signalValues;

        finalSignalSemas.push_back( m_timelineSemaphore );
        finalSignalVals.push_back( 666666 ); // Invalid - will be replaced in critical section with the correct value (m_nextValue)

        // 2. Prepare Command Buffers
        std::vector<VkCommandBuffer> vkCmds;
        vkCmds.reserve( cmdBuffers.size() );
        for( auto* cmd: cmdBuffers )
        {
#ifdef DT_DEBUG
            if( cmd->GetState() != CommandBuffer::State::Executable )
            {
                DT_ERROR( "Queue::Submit: Command Buffer is not in Executable state!" );
                return Result::FAIL;
            }
#endif
            vkCmds.push_back( cmd->GetHandle() );
        }

        // 3. Timeline Info
        VkTimelineSemaphoreSubmitInfo timelineInfo = { VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO };
        timelineInfo.waitSemaphoreValueCount       = ( uint32_t )waitValues.size();
        timelineInfo.pWaitSemaphoreValues          = waitValues.data();
        timelineInfo.signalSemaphoreValueCount     = ( uint32_t )finalSignalVals.size();
        timelineInfo.pSignalSemaphoreValues        = finalSignalVals.data();

        // 4. Submit Info
        VkSubmitInfo submitInfo = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
        submitInfo.pNext        = &timelineInfo;

        submitInfo.commandBufferCount = ( uint32_t )vkCmds.size();
        submitInfo.pCommandBuffers    = vkCmds.data();

        // Waits
        std::vector<VkPipelineStageFlags> waitStages( waitSemaphores.size(), VK_PIPELINE_STAGE_ALL_COMMANDS_BIT ); // Wait at top or all commands
        submitInfo.waitSemaphoreCount = ( uint32_t )waitSemaphores.size();
        submitInfo.pWaitSemaphores    = waitSemaphores.data();
        submitInfo.pWaitDstStageMask  = waitStages.data();

        // Signals
        submitInfo.signalSemaphoreCount = ( uint32_t )finalSignalSemas.size();
        submitInfo.pSignalSemaphores    = finalSignalSemas.data();

        {
            std::lock_guard<std::mutex> lock( m_submitMutex );
            finalSignalVals.back() = m_nextValue; // Update the timeline signal value to the current internal tracker
            if( m_api.vkQueueSubmit( m_queue, 1, &submitInfo, VK_NULL_HANDLE ) != VK_SUCCESS )
            {
                DT_ERROR( "Queue Submit Failed!" );
                return Result::FAIL;
            }

            // Advance internal CPU tracker
            m_nextValue++;
        }

        return Result::SUCCESS;
    }

    bool_t Queue::IsValueCompleted( uint64_t value ) const
    {
        if( value == 0 )
            return true; // Value 0 is always completed

        uint64_t completedValue = 0;
        VkResult result         = m_api.vkGetSemaphoreCounterValue( m_device, m_timelineSemaphore, &completedValue );

        if( result != VK_SUCCESS )
        {
            DT_ERROR( "Failed to get semaphore counter value!" );
            return false;
        }

        return completedValue >= value;
    }

    void Queue::WaitIdle() const
    {
        if( m_queue )
        {
            m_api.vkQueueWaitIdle( m_queue );
        }
    }
} // namespace DigitalTwin