#include "rhi/Queue.h"

#include "core/Log.h"

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