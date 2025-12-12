#pragma once
#include "rhi/BindingGroup.hpp"
#include "rhi/CommandBuffer.hpp"
#include "rhi/DescriptorAllocator.hpp"
#include "rhi/Device.hpp"
#include "rhi/Pipeline.hpp"
#include <string>

namespace DigitalTwin
{
    /**
     * @brief Represents a specific compute operation (Shader + Configuration).
     * Acts as a factory for BindingGroups and handles dispatch logic.
     */
    class ComputeKernel
    {
    public:
        ComputeKernel( Ref<Device> device, Ref<ComputePipeline> pipeline, std::string name );

        void SetGroupSize( uint32_t x, uint32_t y, uint32_t z ) { m_groupSize = { x, y, z }; }

        /**
         * @brief Creates a new BindingGroup compatible with this kernel's shader layout.
         * Allocates a descriptor set from the provided allocator.
         */
        Ref<BindingGroup> CreateBindingGroup();

        /**
         * @brief Records the dispatch command using a pre-configured BindingGroup.
         */
        void Dispatch( CommandBuffer& cmd, Ref<BindingGroup> group, uint32_t elementCount );

        const std::string& GetName() const { return m_name; }

    private:
        Ref<Device>          m_device;
        Ref<ComputePipeline> m_pipeline;
        std::string          m_name;
        struct
        {
            uint32_t x = 1, y = 1, z = 1;
        } m_groupSize;
    };
} // namespace DigitalTwin