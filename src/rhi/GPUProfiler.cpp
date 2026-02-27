#include "rhi/GPUProfiler.h"

#include "core/Log.h"
#include "rhi/CommandBuffer.h"
#include "rhi/Device.h"

namespace DigitalTwin
{
    GPUProfiler::GPUProfiler( Device* device, uint32_t maxZones )
        : m_device( device )
        , m_maxZones( maxZones )
    {
    }

    GPUProfiler::~GPUProfiler()
    {
    }

    Result GPUProfiler::Initialize()
    {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties( m_device->GetPhysicalDevice(), &props );
        m_timestampPeriod = props.limits.timestampPeriod;

        const auto& api = m_device->GetAPI();

        for( uint32_t i = 0; i < FRAMES_IN_FLIGHT; ++i )
        {
            // Timestamp pool (2 queries per zone: start and end)
            VkQueryPoolCreateInfo timestampInfo = { VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO };
            timestampInfo.queryType             = VK_QUERY_TYPE_TIMESTAMP;
            timestampInfo.queryCount            = m_maxZones * 2;

            if( api.vkCreateQueryPool( m_device->GetHandle(), &timestampInfo, nullptr, &m_timestampPools[ i ] ) != VK_SUCCESS )
                return Result::FAIL;

            // Stats pool (1 query per zone tracking 4 metrics)
            VkQueryPoolCreateInfo statInfo = { VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO };
            statInfo.queryType             = VK_QUERY_TYPE_PIPELINE_STATISTICS;
            statInfo.queryCount            = m_maxZones;
            statInfo.pipelineStatistics =
                VK_QUERY_PIPELINE_STATISTIC_VERTEX_SHADER_INVOCATIONS_BIT | VK_QUERY_PIPELINE_STATISTIC_CLIPPING_INVOCATIONS_BIT |
                VK_QUERY_PIPELINE_STATISTIC_FRAGMENT_SHADER_INVOCATIONS_BIT | VK_QUERY_PIPELINE_STATISTIC_COMPUTE_SHADER_INVOCATIONS_BIT;

            if( api.vkCreateQueryPool( m_device->GetHandle(), &statInfo, nullptr, &m_statPools[ i ] ) != VK_SUCCESS )
            {
                DT_ERROR( "Failed to create Stat Pool! Enable 'pipelineStatisticsQuery' in Device Features." );
                return Result::FAIL;
            }
        }
        return Result::SUCCESS;
    }

    void GPUProfiler::Shutdown()
    {
        const auto& api = m_device->GetAPI();
        for( uint32_t i = 0; i < FRAMES_IN_FLIGHT; ++i )
        {
            if( m_timestampPools[ i ] )
                api.vkDestroyQueryPool( m_device->GetHandle(), m_timestampPools[ i ], nullptr );
            if( m_statPools[ i ] )
                api.vkDestroyQueryPool( m_device->GetHandle(), m_statPools[ i ], nullptr );
            m_timestampPools[ i ] = VK_NULL_HANDLE;
            m_statPools[ i ]      = VK_NULL_HANDLE;
        }
    }

    void GPUProfiler::BeginFrame( CommandBuffer* cmd, uint32_t flightIndex )
    {
        const auto& api = m_device->GetAPI();

        // Reset pools using the command buffer for this frame
        cmd->ResetQueryPool( m_timestampPools[ flightIndex ], 0, m_maxZones * 2 );
        cmd->ResetQueryPool( m_statPools[ flightIndex ], 0, m_maxZones );

        if( m_currentZoneCount == 0 )
            return;

        // If previous frame finished recording, read results into host memory
        if( m_hasData[ flightIndex ] )
        {
            std::vector<uint64_t> timestamps( m_currentZoneCount * 2 );
            api.vkGetQueryPoolResults( m_device->GetHandle(), m_timestampPools[ flightIndex ], 0, m_currentZoneCount * 2,
                                       timestamps.size() * sizeof( uint64_t ), timestamps.data(), sizeof( uint64_t ), VK_QUERY_RESULT_64_BIT );

            std::vector<uint64_t> stats( m_currentZoneCount * 4 );
            api.vkGetQueryPoolResults( m_device->GetHandle(), m_statPools[ flightIndex ], 0, m_currentZoneCount, stats.size() * sizeof( uint64_t ),
                                       stats.data(), sizeof( uint64_t ) * 4, VK_QUERY_RESULT_64_BIT );

            for( const auto& [ name, index ]: m_zoneToIndex )
            {
                uint64_t startT = timestamps[ index * 2 ];
                uint64_t endT   = timestamps[ index * 2 + 1 ];

                GPUProfileData& data = m_results[ name ];
                data.timeMs          = ( endT > startT ) ? static_cast<float>( endT - startT ) * m_timestampPeriod * 1e-6f : 0.0f;

                data.vertexShaderInvocations   = stats[ index * 4 + 0 ];
                data.clippingInvocations       = stats[ index * 4 + 1 ];
                data.fragmentShaderInvocations = stats[ index * 4 + 2 ];
                data.computeShaderInvocations  = stats[ index * 4 + 3 ];
            }
        }

        m_hasData[ flightIndex ] = false;
    }

    void GPUProfiler::BeginZone( CommandBuffer* cmd, uint32_t flightIndex, const std::string& name )
    {
        if( m_zoneToIndex.find( name ) == m_zoneToIndex.end() )
        {
            if( m_currentZoneCount >= m_maxZones )
            {
                DT_WARN( "GPUProfiler max zones reached!" );
                return;
            }
            m_zoneToIndex[ name ] = m_currentZoneCount++;
        }

        uint32_t idx = m_zoneToIndex[ name ];
        cmd->WriteTimestamp( VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, m_timestampPools[ flightIndex ], idx * 2 );
        cmd->BeginQuery( m_statPools[ flightIndex ], idx );
    }

    void GPUProfiler::EndZone( CommandBuffer* cmd, uint32_t flightIndex, const std::string& name )
    {
        auto it = m_zoneToIndex.find( name );
        if( it == m_zoneToIndex.end() )
            return;

        uint32_t idx = it->second;
        cmd->EndQuery( m_statPools[ flightIndex ], idx );
        cmd->WriteTimestamp( VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, m_timestampPools[ flightIndex ], idx * 2 + 1 );

        m_hasData[ flightIndex ] = true;
    }
    GPUMemoryStats GPUProfiler::GetMemoryStats() const
    {
        GPUMemoryStats stats{};

        // Chain the budget structure into the memory properties request
        VkPhysicalDeviceMemoryBudgetPropertiesEXT budgetProps = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT };
        VkPhysicalDeviceMemoryProperties2         memProps    = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2 };
        memProps.pNext                                        = &budgetProps;

        // Query memory properties dynamically using volk
        vkGetPhysicalDeviceMemoryProperties2( m_device->GetPhysicalDevice(), &memProps );

        // Iterate over all memory heaps
        for( uint32_t i = 0; i < memProps.memoryProperties.memoryHeapCount; ++i )
        {
            // We only care about DEVICE_LOCAL memory (actual GPU VRAM)
            if( memProps.memoryProperties.memoryHeaps[ i ].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT )
            {
                stats.totalBudget += budgetProps.heapBudget[ i ];
                stats.currentUsage += budgetProps.heapUsage[ i ];
            }
        }

        return stats;
    }
} // namespace DigitalTwin