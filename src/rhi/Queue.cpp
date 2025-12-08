#include "rhi/Queue.hpp"

namespace DigitalTwin
{
    Queue::Queue( VkDevice device, const VolkDeviceTable& api, uint32_t queueFamilyIndex, QueueType type )
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

    Queue::~Queue()
    {
        if( m_timelineSemaphore != VK_NULL_HANDLE )
        {
            m_api.vkDestroySemaphore( m_device, m_timelineSemaphore, nullptr );
        }
    }

    Result Queue::Submit( const SubmitInfo& info, uint64_t* outSignalValue )
    {
        std::lock_guard<std::mutex> lock( m_mutex );

        // 1. Prepare Command Buffer Infos
        std::vector<VkCommandBufferSubmitInfo> cmdInfos;
        cmdInfos.reserve( info.commandBuffers.size() );

        for( auto cmd: info.commandBuffers )
        {
            VkCommandBufferSubmitInfo cmdInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO };
            cmdInfo.commandBuffer             = cmd;
            cmdInfos.push_back( cmdInfo );
        }

        // 2. Prepare Semaphore Wait Infos
        std::vector<VkSemaphoreSubmitInfo> waitInfos;
        waitInfos.reserve( info.waitSemaphores.size() );

        for( const auto& wait: info.waitSemaphores )
        {
            VkSemaphoreSubmitInfo semInfo = { VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO };
            semInfo.semaphore             = wait.semaphore;
            semInfo.value                 = wait.value;
            semInfo.stageMask             = wait.stageMask;
            semInfo.deviceIndex           = 0;
            waitInfos.push_back( semInfo );
        }

        // 3. Prepare Semaphore Signal Infos
        std::vector<VkSemaphoreSubmitInfo> signalInfos;
        signalInfos.reserve( info.signalSemaphores.size() + 1 );

        for( const auto& signal: info.signalSemaphores )
        {
            VkSemaphoreSubmitInfo semInfo = { VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO };
            semInfo.semaphore             = signal.semaphore;
            semInfo.value                 = signal.value;
            semInfo.stageMask             = signal.stageMask;
            semInfo.deviceIndex           = 0;
            signalInfos.push_back( semInfo );
        }

        // --- Append Internal Timeline Signal ---
        uint64_t currentSignalValue = m_nextValue++;

        VkSemaphoreSubmitInfo internalSignalInfo = { VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO };
        internalSignalInfo.semaphore             = m_timelineSemaphore;
        internalSignalInfo.value                 = currentSignalValue;
        internalSignalInfo.stageMask             = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        internalSignalInfo.deviceIndex           = 0;
        signalInfos.push_back( internalSignalInfo );

        if( outSignalValue )
        {
            *outSignalValue = currentSignalValue;
        }

        // 4. Construct Final Submit Info 2
        VkSubmitInfo2 submitInfo            = { VK_STRUCTURE_TYPE_SUBMIT_INFO_2 };
        submitInfo.commandBufferInfoCount   = static_cast<uint32_t>( cmdInfos.size() );
        submitInfo.pCommandBufferInfos      = cmdInfos.data();
        submitInfo.waitSemaphoreInfoCount   = static_cast<uint32_t>( waitInfos.size() );
        submitInfo.pWaitSemaphoreInfos      = waitInfos.data();
        submitInfo.signalSemaphoreInfoCount = static_cast<uint32_t>( signalInfos.size() );
        submitInfo.pSignalSemaphoreInfos    = signalInfos.data();

        // 5. Submit (No Fence!)
        VkResult result = m_api.vkQueueSubmit2( m_queue, 1, &submitInfo, VK_NULL_HANDLE );

        if( result != VK_SUCCESS )
        {
            DT_CORE_CRITICAL( "Queue Submit2 failed! Error: {}", ( int )result );
            return Result::FAIL;
        }

        return Result::SUCCESS;
    }

    Result Queue::Submit( VkCommandBuffer commandBuffer, uint64_t& outSignalValue )
    {
        SubmitInfo info;
        info.commandBuffers.push_back( commandBuffer );
        return Submit( info, &outSignalValue );
    }

    bool_t Queue::IsValueCompleted( uint64_t fenceValue )
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