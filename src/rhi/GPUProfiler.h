#pragma once
#include "rhi/RHITypes.h"

#include <DigitalTwinTypes.h>
#include <string>
#include <unordered_map>
#include <vector>

namespace DigitalTwin
{

    /**
     * @brief Universal GPU Profiler owned by the Device.
     * Manages named zones and retrieves hardware performance metrics.
     */
    class GPUProfiler
    {
    public:
        // maxZones limits how many distinct regions we can profile
        GPUProfiler( Device* device, uint32_t maxZones = 16 );
        ~GPUProfiler();

        Result Initialize();
        void   Shutdown();

        /**
         * @brief Must be called at the start of rendering/compute for this flight index.
         * Fetches results from the previous frame and resets the pools.
         */
        void BeginFrame( CommandBuffer* cmd, uint32_t flightIndex );

        void BeginZone( CommandBuffer* cmd, uint32_t flightIndex, const std::string& name );
        void EndZone( CommandBuffer* cmd, uint32_t flightIndex, const std::string& name );

        const std::unordered_map<std::string, GPUProfileData>& GetResults() const { return m_results; }

        GPUMemoryStats GetMemoryStats() const;

    private:
        static const uint32_t FRAMES_IN_FLIGHT = 2;

        Device*  m_device;
        uint32_t m_maxZones;
        float    m_timestampPeriod = 1.0f;

        VkQueryPool m_timestampPools[ FRAMES_IN_FLIGHT ] = { VK_NULL_HANDLE, VK_NULL_HANDLE };
        VkQueryPool m_statPools[ FRAMES_IN_FLIGHT ]      = { VK_NULL_HANDLE, VK_NULL_HANDLE };

        std::unordered_map<std::string, uint32_t>       m_zoneToIndex;
        std::unordered_map<std::string, GPUProfileData> m_results;

        uint32_t m_currentZoneCount            = 0;
        bool     m_hasData[ FRAMES_IN_FLIGHT ] = { false, false };
    };
} // namespace DigitalTwin