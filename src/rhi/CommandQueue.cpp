#include "rhi/CommandQueue.hpp"

namespace DigitalTwin
{
    CommandQueue::CommandQueue( VkDevice device, const VolkDeviceTable& api, uint32_t queueFamilyIndex, QueueType type )
        : m_device( device )
        , m_api( api )
        , m_queue( VK_NULL_HANDLE )
        , m_queueFamilyIndex( queueFamilyIndex )
        , m_type( type )
        , m_timelineSemaphore( VK_NULL_HANDLE )
        , m_nextValue( 1 )
    {
        // Retrieve queue index 0 from the specified family
        m_api.vkGetDeviceQueue( m_device, m_queueFamilyIndex, 0, &m_queue );

        VkSemaphoreTypeCreateInfo timelineCreateInfo = { VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO };
        timelineCreateInfo.semaphoreType             = VK_SEMAPHORE_TYPE_TIMELINE;
        timelineCreateInfo.initialValue              = 0;

        VkSemaphoreCreateInfo createInfo = { VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
        createInfo.pNext                 = &timelineCreateInfo;

        VkResult result = m_api.vkCreateSemaphore( m_device, &createInfo, nullptr, &m_timelineSemaphore );
        if( result != VK_SUCCESS )
        {
            DT_CORE_CRITICAL( "Failed to create Timeline Semaphore for queue family {}! Error: {}", m_queueFamilyIndex, ( int )result );
        }
    }

    CommandQueue::~CommandQueue()
    {
        if( m_timelineSemaphore != VK_NULL_HANDLE )
        {
            m_api.vkDestroySemaphore( m_device, m_timelineSemaphore, nullptr );
        }
    }

    Result CommandQueue::Submit( VkCommandBuffer commandBuffer, uint64_t& outSignalValue )
    {
        std::lock_guard<std::mutex> lock( m_mutex );

        uint64_t signalValue = m_nextValue++;
        outSignalValue       = signalValue;

        VkTimelineSemaphoreSubmitInfo timelineInfo = { VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO };
        timelineInfo.signalSemaphoreValueCount     = 1;
        timelineInfo.pSignalSemaphoreValues        = &signalValue;

        VkSubmitInfo submitInfo         = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
        submitInfo.pNext                = &timelineInfo;
        submitInfo.commandBufferCount   = 1;
        submitInfo.pCommandBuffers      = &commandBuffer;
        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores    = &m_timelineSemaphore;

        VkResult result = m_api.vkQueueSubmit( m_queue, 1, &submitInfo, VK_NULL_HANDLE );

        if( result != VK_SUCCESS )
        {
            DT_CORE_CRITICAL( "Queue Submit failed! Error: {}", ( int )result );
            return Result::FAIL;
        }

        return Result::SUCCESS;
    }

    bool_t CommandQueue::IsValueCompleted( uint64_t fenceValue )
    {
        uint64_t completedValue = 0;
        VkResult result         = m_api.vkGetSemaphoreCounterValue( m_device, m_timelineSemaphore, &completedValue );

        if( result != VK_SUCCESS )
        {
            DT_CORE_ERROR( "GetSemaphoreCounterValue failed! Error: {}", ( int )result );
            return false;
        }

        return completedValue >= fenceValue;
    }

} // namespace DigitalTwin