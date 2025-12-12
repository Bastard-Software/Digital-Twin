#pragma once
#include "core/Base.hpp"
#include "rhi/Shader.hpp"
#include <map>
#include <vector>
#include <volk.h>

namespace DigitalTwin
{
    struct ComputePipelineDesc
    {
        Ref<Shader> shader;
    };

    struct GraphicsPipelineDesc
    {
        Ref<Shader> vertexShader;
        Ref<Shader> fragmentShader; // Optional (e.g., depth-only pass)

        // Input Assembly
        VkPrimitiveTopology topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

        // Rasterization
        VkPolygonMode   polygonMode = VK_POLYGON_MODE_FILL;
        VkCullModeFlags cullMode    = VK_CULL_MODE_BACK_BIT;
        VkFrontFace     frontFace   = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        float           lineWidth   = 1.0f;

        // Depth Stencil
        bool        depthTestEnable  = true;
        bool        depthWriteEnable = true;
        VkCompareOp depthCompareOp   = VK_COMPARE_OP_LESS;

        // Blending (Simple one-size-fits-all approach for now)
        bool          blendEnable         = false;
        VkBlendFactor srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
        VkBlendFactor dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        VkBlendOp     colorBlendOp        = VK_BLEND_OP_ADD;
        VkBlendFactor srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        VkBlendFactor dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
        VkBlendOp     alphaBlendOp        = VK_BLEND_OP_ADD;

        // Dynamic Rendering Formats (Vulkan 1.3)
        // Pipeline must know what it is rendering into
        std::vector<VkFormat> colorAttachmentFormats = { VK_FORMAT_R8G8B8A8_UNORM };
        VkFormat              depthAttachmentFormat  = VK_FORMAT_D32_SFLOAT;
    };

    namespace PipelineUtils
    {
        struct PipelineLayoutResult
        {
            VkPipelineLayout                          pipelineLayout = VK_NULL_HANDLE;
            std::map<uint32_t, VkDescriptorSetLayout> descriptorSetLayouts;
        };

        // Analyzes shaders via reflection, merges resources, and creates Vulkan layouts
        PipelineLayoutResult CreatePipelineLayout( VkDevice device, const VolkDeviceTable* api, const std::vector<Ref<Shader>>& shaders );

        // Destroys pipeline layout and all descriptor set layouts
        void DestroyPipelineLayout( VkDevice device, const VolkDeviceTable* api, const PipelineLayoutResult& resources );

        // Helper to merge reflection data from multiple shaders into one map
        ShaderReflectionData MergeReflectionData( const std::vector<Ref<Shader>>& shaders );
    } // namespace PipelineUtils

    class ComputePipeline
    {
    public:
        ComputePipeline( VkDevice device, const VolkDeviceTable* api, const ComputePipelineDesc& desc, VkPipelineCache cache = VK_NULL_HANDLE );
        ~ComputePipeline();

        ComputePipeline( const ComputePipeline& )            = delete;
        ComputePipeline& operator=( const ComputePipeline& ) = delete;

        VkPipeline                  GetHandle() const { return m_pipeline; }
        VkPipelineLayout            GetLayout() const { return m_resources.pipelineLayout; }
        const ShaderReflectionData& GetReflectionData() const { return m_reflectionData; }
        VkDescriptorSetLayout       GetDescriptorSetLayout( uint32_t set ) const;

    private:
        VkDevice                            m_device;
        const VolkDeviceTable*              m_api;
        VkPipeline                          m_pipeline = VK_NULL_HANDLE;
        ShaderReflectionData                m_reflectionData;
        PipelineUtils::PipelineLayoutResult m_resources;
    };

    class GraphicsPipeline
    {
    public:
        GraphicsPipeline( VkDevice device, const VolkDeviceTable* api, const GraphicsPipelineDesc& desc, VkPipelineCache cache = VK_NULL_HANDLE );
        ~GraphicsPipeline();

        GraphicsPipeline( const GraphicsPipeline& )            = delete;
        GraphicsPipeline& operator=( const GraphicsPipeline& ) = delete;

        VkPipeline                  GetHandle() const { return m_pipeline; }
        VkPipelineLayout            GetLayout() const { return m_resources.pipelineLayout; }
        const ShaderReflectionData& GetReflectionData() const { return m_reflectionData; }
        VkDescriptorSetLayout       GetDescriptorSetLayout( uint32_t set ) const;

    private:
        VkDevice                            m_device;
        const VolkDeviceTable*              m_api;
        VkPipeline                          m_pipeline = VK_NULL_HANDLE;
        ShaderReflectionData                m_reflectionData;
        PipelineUtils::PipelineLayoutResult m_resources;
    };
} // namespace DigitalTwin