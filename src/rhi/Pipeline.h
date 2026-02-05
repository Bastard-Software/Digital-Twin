#pragma once
#include "rhi/RHITypes.h"

#include "core/Core.h"
#include "rhi/Shader.h"
#include <map>
#include <vector>
#include <volk.h>

namespace DigitalTwin
{

    namespace PipelineUtils
    {

        struct PipelineLayoutResult
        {
            VkPipelineLayout                          pipelineLayout = VK_NULL_HANDLE;
            std::map<uint32_t, VkDescriptorSetLayout> descriptorSetLayouts;

            ShaderReflectionData reflectionData;
        };

        // Analyzes shaders via reflection, merges resources, and creates Vulkan layouts
        PipelineLayoutResult CreatePipelineLayout( VkDevice device, const VolkDeviceTable* api, const std::vector<Shader*>& shaders );

        // Destroys pipeline layout and all descriptor set layouts
        void DestroyPipelineLayout( VkDevice device, const VolkDeviceTable* api, const PipelineLayoutResult& resources );

    } // namespace PipelineUtils

    class ComputePipeline
    {
    public:
        ComputePipeline( VkDevice device, const VolkDeviceTable* api );
        ~ComputePipeline();

        Result Create( const ComputePipelineNativeDesc& desc );
        void   Destroy();

        VkPipeline                  GetHandle() const { return m_pipeline; }
        VkPipelineLayout            GetLayout() const { return m_resources.pipelineLayout; }
        const ShaderReflectionData& GetReflectionData() const { return m_resources.reflectionData; }
        VkDescriptorSetLayout       GetDescriptorSetLayout( uint32_t set ) const;

    public:
        // Disable copying (RAII), allow moving
        ComputePipeline( const ComputePipeline& )            = delete;
        ComputePipeline& operator=( const ComputePipeline& ) = delete;
        ComputePipeline( ComputePipeline&& other ) noexcept;
        ComputePipeline& operator=( ComputePipeline&& other ) noexcept;

    private:
        VkDevice                            m_device;
        const VolkDeviceTable*              m_api;
        VkPipeline                          m_pipeline = VK_NULL_HANDLE;
        PipelineUtils::PipelineLayoutResult m_resources;
    };

    class GraphicsPipeline
    {
    public:
        GraphicsPipeline( VkDevice device, const VolkDeviceTable* api );
        ~GraphicsPipeline();

        Result Create( const GraphicsPipelineNativeDesc& desc );
        void   Destroy();

        VkPipeline                  GetHandle() const { return m_pipeline; }
        VkPipelineLayout            GetLayout() const { return m_resources.pipelineLayout; }
        const ShaderReflectionData& GetReflectionData() const { return m_resources.reflectionData; }
        VkDescriptorSetLayout       GetDescriptorSetLayout( uint32_t set ) const;

    public:
        // Disable copying (RAII), allow moving
        GraphicsPipeline( const GraphicsPipeline& )            = delete;
        GraphicsPipeline& operator=( const GraphicsPipeline& ) = delete;
        GraphicsPipeline( GraphicsPipeline&& other ) noexcept;
        GraphicsPipeline& operator=( GraphicsPipeline&& other ) noexcept;

    private:
        VkDevice                            m_device;
        const VolkDeviceTable*              m_api;
        VkPipeline                          m_pipeline = VK_NULL_HANDLE;
        PipelineUtils::PipelineLayoutResult m_resources;
    };

} // namespace DigitalTwin