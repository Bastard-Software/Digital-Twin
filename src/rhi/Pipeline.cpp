#include "rhi/Pipeline.hpp"

#include <algorithm>
#include <map>
#include <set>

namespace DigitalTwin
{

    static VkDescriptorType MapResourceTypeToVulkan( ShaderResourceType type )
    {
        switch( type )
        {
            case ShaderResourceType::UNIFORM_BUFFER:
                return VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            case ShaderResourceType::STORAGE_BUFFER:
                return VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            case ShaderResourceType::SAMPLED_IMAGE:
                return VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            case ShaderResourceType::STORAGE_IMAGE:
                return VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            default:
                return VK_DESCRIPTOR_TYPE_MAX_ENUM;
        }
    }

    namespace PipelineUtils
    {
        PipelineLayoutResult CreatePipelineLayout( VkDevice device, const VolkDeviceTable* api, const std::vector<Ref<Shader>>& shaders )
        {
            PipelineLayoutResult result;

            // 1. Merge resources from all shaders
            // Map: Set -> Binding -> ResourceInfo
            std::map<uint32_t, std::map<uint32_t, ShaderResource>> mergedResources;
            std::vector<VkPushConstantRange>                       pushConstants;

            for( const auto& shader: shaders )
            {
                if( !shader )
                    continue;

                // Merge Descriptor Sets
                for( const auto& [ name, res ]: shader->GetReflectionData() )
                {
                    if( res.type == ShaderResourceType::PUSH_CONSTANT || res.type == ShaderResourceType::UNKNOWN )
                        continue;

                    // Check collision (simplified logic: last one wins, assumes shaders are consistent)
                    mergedResources[ res.set ][ res.binding ] = res;
                }

                // Merge Push Constants
                const auto& pcs = shader->GetPushConstantRanges();
                pushConstants.insert( pushConstants.end(), pcs.begin(), pcs.end() );
            }

            // 2a. Create Descriptor Set Layouts for active resources
            for( const auto& [ setIndex, bindingsMap ]: mergedResources )
            {
                std::vector<VkDescriptorSetLayoutBinding> vkBindings;
                for( const auto& [ bindingIndex, res ]: bindingsMap )
                {
                    VkDescriptorSetLayoutBinding b{};
                    b.binding            = bindingIndex;
                    b.descriptorType     = MapResourceTypeToVulkan( res.type );
                    b.descriptorCount    = res.arraySize;
                    b.stageFlags         = VK_SHADER_STAGE_ALL; // Simplify visibility for now
                    b.pImmutableSamplers = nullptr;
                    vkBindings.push_back( b );
                }

                VkDescriptorSetLayoutCreateInfo layoutInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
                layoutInfo.bindingCount                    = static_cast<uint32_t>( vkBindings.size() );
                layoutInfo.pBindings                       = vkBindings.data();

                VkDescriptorSetLayout layout = VK_NULL_HANDLE;
                if( api->vkCreateDescriptorSetLayout( device, &layoutInfo, nullptr, &layout ) != VK_SUCCESS )
                {
                    DT_CORE_CRITICAL( "Failed to create descriptor set layout for set {}", setIndex );
                    continue;
                }
                result.descriptorSetLayouts[ setIndex ] = layout;
            }

            // 2b. FILL GAPS: Ensure no null handles in the contiguous range [0, maxSet]
            // If Set 0 is optimized out by shader compiler, but Set 1 exists, we must provide an empty layout for Set 0.
            if( !result.descriptorSetLayouts.empty() )
            {
                uint32_t maxSet = result.descriptorSetLayouts.rbegin()->first;
                for( uint32_t i = 0; i < maxSet; ++i )
                {
                    if( result.descriptorSetLayouts.find( i ) == result.descriptorSetLayouts.end() )
                    {
                        // Gap detected at index 'i'.
                        VkDescriptorSetLayoutCreateInfo           layoutInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
                        std::vector<VkDescriptorSetLayoutBinding> forcedBindings;

                        // --- CRITICAL FIX: Enforce GlobalData Layout for Set 0 ---
                        // If Set 0 is missing (optimized out by shader), we must recreate it
                        // so that vkUpdateDescriptorSets in the engine doesn't crash when binding the Global UBO.
                        if( i == 0 )
                        {
                            DT_CORE_WARN( "PipelineUtils: Shader optimized out Set 0 (GlobalData). Enforcing layout compatibility." );

                            VkDescriptorSetLayoutBinding globalBinding{};
                            globalBinding.binding            = 0; // GlobalData is always at Binding 0
                            globalBinding.descriptorType     = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
                            globalBinding.descriptorCount    = 1;
                            globalBinding.stageFlags         = VK_SHADER_STAGE_ALL;
                            globalBinding.pImmutableSamplers = nullptr;

                            forcedBindings.push_back( globalBinding );

                            layoutInfo.bindingCount = static_cast<uint32_t>( forcedBindings.size() );
                            layoutInfo.pBindings    = forcedBindings.data();
                        }
                        else
                        {
                            // For other gaps (e.g. Set 1 missing but Set 2 present), create an empty dummy layout.
                            // This allows Pipeline Layout creation but prevents writing descriptors to this set.
                            layoutInfo.bindingCount = 0;
                            layoutInfo.pBindings    = nullptr;
                        }

                        VkDescriptorSetLayout layout = VK_NULL_HANDLE;
                        if( api->vkCreateDescriptorSetLayout( device, &layoutInfo, nullptr, &layout ) == VK_SUCCESS )
                        {
                            result.descriptorSetLayouts[ i ] = layout;
                        }
                        else
                        {
                            DT_CORE_CRITICAL( "PipelineUtils: Failed to create gap-filling layout for Set {}", i );
                        }
                    }
                }
            }

            // 3. Create Pipeline Layout
            std::vector<VkDescriptorSetLayout> contiguousLayouts;
            if( !result.descriptorSetLayouts.empty() )
            {
                uint32_t maxSet = result.descriptorSetLayouts.rbegin()->first;
                contiguousLayouts.resize( maxSet + 1, VK_NULL_HANDLE );
                for( const auto& [ set, layout ]: result.descriptorSetLayouts )
                {
                    contiguousLayouts[ set ] = layout;
                }
            }

            VkPipelineLayoutCreateInfo pipelineLayoutInfo = { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
            pipelineLayoutInfo.setLayoutCount             = static_cast<uint32_t>( contiguousLayouts.size() );
            pipelineLayoutInfo.pSetLayouts                = contiguousLayouts.data();
            pipelineLayoutInfo.pushConstantRangeCount     = static_cast<uint32_t>( pushConstants.size() );
            pipelineLayoutInfo.pPushConstantRanges        = pushConstants.data();

            if( api->vkCreatePipelineLayout( device, &pipelineLayoutInfo, nullptr, &result.pipelineLayout ) != VK_SUCCESS )
            {
                DT_CORE_CRITICAL( "Failed to create pipeline layout!" );
            }

            return result;
        }

        void DestroyPipelineLayout( VkDevice device, const VolkDeviceTable* api, const PipelineLayoutResult& resources )
        {
            if( resources.pipelineLayout != VK_NULL_HANDLE )
            {
                api->vkDestroyPipelineLayout( device, resources.pipelineLayout, nullptr );
            }
            for( auto& [ set, layout ]: resources.descriptorSetLayouts )
            {
                api->vkDestroyDescriptorSetLayout( device, layout, nullptr );
            }
        }

        ShaderReflectionData MergeReflectionData( const std::vector<Ref<Shader>>& shaders )
        {
            ShaderReflectionData merged;

            for( const auto& shader: shaders )
            {
                if( !shader )
                    continue;

                const auto& stageResources = shader->GetReflectionData();
                for( const auto& [ name, resource ]: stageResources )
                {
                    // Insert if not exists.
                    // If the same resource name exists (e.g., "GlobalUniforms" in Vert and Frag),
                    // we assume it maps to the same binding/set configuration.
                    if( merged.find( name ) == merged.end() )
                    {
                        merged[ name ] = resource;
                    }
                }
            }
            return merged;
        }
    } // namespace PipelineUtils

    ComputePipeline::ComputePipeline( VkDevice device, const VolkDeviceTable* api, const ComputePipelineDesc& desc, VkPipelineCache cache )
        : m_device( device )
        , m_api( api )
    {
        DT_CORE_ASSERT( desc.shader, "ComputePipeline requires a shader!" );

        // 1. Create Layouts (Layouts are owned by this pipeline instance)
        m_resources      = PipelineUtils::CreatePipelineLayout( m_device, m_api, { desc.shader } );
        m_reflectionData = desc.shader->GetReflectionData();

        // 2. Create Pipeline
        VkComputePipelineCreateInfo pipelineInfo = { VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
        pipelineInfo.layout                      = m_resources.pipelineLayout;
        pipelineInfo.stage.sType                 = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        pipelineInfo.stage.stage                 = VK_SHADER_STAGE_COMPUTE_BIT;
        pipelineInfo.stage.module                = desc.shader->GetModule();
        pipelineInfo.stage.pName                 = "main";

        if( m_api->vkCreateComputePipelines( m_device, cache, 1, &pipelineInfo, nullptr, &m_pipeline ) != VK_SUCCESS )
        {
            DT_CORE_CRITICAL( "Failed to create compute pipeline!" );
        }
    }

    ComputePipeline::~ComputePipeline()
    {
        if( m_pipeline )
            m_api->vkDestroyPipeline( m_device, m_pipeline, nullptr );
        PipelineUtils::DestroyPipelineLayout( m_device, m_api, m_resources );
    }

    VkDescriptorSetLayout ComputePipeline::GetDescriptorSetLayout( uint32_t set ) const
    {
        auto it = m_resources.descriptorSetLayouts.find( set );
        return ( it != m_resources.descriptorSetLayouts.end() ) ? it->second : VK_NULL_HANDLE;
    }

    GraphicsPipeline::GraphicsPipeline( VkDevice device, const VolkDeviceTable* api, const GraphicsPipelineDesc& desc, VkPipelineCache cache )
        : m_device( device )
        , m_api( api )
    {
        DT_CORE_ASSERT( desc.vertexShader, "GraphicsPipeline requires a Vertex Shader!" );

        // 1. Create Layouts (Merge VS + FS resources)
        std::vector<Ref<Shader>> shaders = { desc.vertexShader };
        if( desc.fragmentShader )
            shaders.push_back( desc.fragmentShader );

        m_resources      = PipelineUtils::CreatePipelineLayout( m_device, m_api, shaders );
        m_reflectionData = PipelineUtils::MergeReflectionData( shaders );

        // 2. Shader Stages
        std::vector<VkPipelineShaderStageCreateInfo> shaderStages;
        VkPipelineShaderStageCreateInfo              vertStage = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO };
        vertStage.stage                                        = VK_SHADER_STAGE_VERTEX_BIT;
        vertStage.module                                       = desc.vertexShader->GetModule();
        vertStage.pName                                        = "main";
        shaderStages.push_back( vertStage );

        if( desc.fragmentShader )
        {
            VkPipelineShaderStageCreateInfo fragStage = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO };
            fragStage.stage                           = VK_SHADER_STAGE_FRAGMENT_BIT;
            fragStage.module                          = desc.fragmentShader->GetModule();
            fragStage.pName                           = "main";
            shaderStages.push_back( fragStage );
        }

        // 3. Vertex Input (EMPTY - Vertex Pulling)
        VkPipelineVertexInputStateCreateInfo vertexInputInfo = { VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO };

        // 4. Input Assembly
        VkPipelineInputAssemblyStateCreateInfo inputAssembly = { VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO };
        inputAssembly.topology                               = desc.topology;
        inputAssembly.primitiveRestartEnable                 = VK_FALSE;

        // 5. Viewport & Scissor (Dynamic State)
        VkPipelineViewportStateCreateInfo viewportState = { VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO };
        viewportState.viewportCount                     = 1;
        viewportState.scissorCount                      = 1;

        // 6. Rasterizer
        VkPipelineRasterizationStateCreateInfo rasterizer = { VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO };
        rasterizer.depthClampEnable                       = VK_FALSE;
        rasterizer.rasterizerDiscardEnable                = VK_FALSE;
        rasterizer.polygonMode                            = desc.polygonMode;
        rasterizer.lineWidth                              = desc.lineWidth;
        rasterizer.cullMode                               = desc.cullMode;
        rasterizer.frontFace                              = desc.frontFace;
        rasterizer.depthBiasEnable                        = VK_FALSE;

        // 7. Multisampling (Default 1 sample)
        VkPipelineMultisampleStateCreateInfo multisampling = { VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO };
        multisampling.sampleShadingEnable                  = VK_FALSE;
        multisampling.rasterizationSamples                 = VK_SAMPLE_COUNT_1_BIT;

        // 8. Depth Stencil
        VkPipelineDepthStencilStateCreateInfo depthStencil = { VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO };
        depthStencil.depthTestEnable                       = desc.depthTestEnable;
        depthStencil.depthWriteEnable                      = desc.depthWriteEnable;
        depthStencil.depthCompareOp                        = desc.depthCompareOp;
        depthStencil.depthBoundsTestEnable                 = VK_FALSE;
        depthStencil.stencilTestEnable                     = VK_FALSE;

        // 9. Color Blending
        // One blend attachment per color attachment format
        std::vector<VkPipelineColorBlendAttachmentState> colorBlendAttachments;
        for( size_t i = 0; i < desc.colorAttachmentFormats.size(); ++i )
        {
            VkPipelineColorBlendAttachmentState colorBlendAttachment{};
            colorBlendAttachment.colorWriteMask =
                VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
            colorBlendAttachment.blendEnable         = desc.blendEnable;
            colorBlendAttachment.srcColorBlendFactor = desc.srcColorBlendFactor;
            colorBlendAttachment.dstColorBlendFactor = desc.dstColorBlendFactor;
            colorBlendAttachment.colorBlendOp        = desc.colorBlendOp;
            colorBlendAttachment.srcAlphaBlendFactor = desc.srcAlphaBlendFactor;
            colorBlendAttachment.dstAlphaBlendFactor = desc.dstAlphaBlendFactor;
            colorBlendAttachment.alphaBlendOp        = desc.alphaBlendOp;
            colorBlendAttachments.push_back( colorBlendAttachment );
        }

        VkPipelineColorBlendStateCreateInfo colorBlending = { VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO };
        colorBlending.logicOpEnable                       = VK_FALSE;
        colorBlending.attachmentCount                     = static_cast<uint32_t>( colorBlendAttachments.size() );
        colorBlending.pAttachments                        = colorBlendAttachments.data();

        // 10. Dynamic States
        std::vector<VkDynamicState>      dynamicStates = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
        VkPipelineDynamicStateCreateInfo dynamicState  = { VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO };
        dynamicState.dynamicStateCount                 = static_cast<uint32_t>( dynamicStates.size() );
        dynamicState.pDynamicStates                    = dynamicStates.data();

        // 11. Dynamic Rendering Info (Vulkan 1.3)
        VkPipelineRenderingCreateInfo renderingInfo = { VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO };
        renderingInfo.colorAttachmentCount          = static_cast<uint32_t>( desc.colorAttachmentFormats.size() );
        renderingInfo.pColorAttachmentFormats       = desc.colorAttachmentFormats.data();
        renderingInfo.depthAttachmentFormat         = desc.depthAttachmentFormat;
        // renderingInfo.stencilAttachmentFormat = ... (omitted for now)

        // 12. Create Pipeline
        VkGraphicsPipelineCreateInfo pipelineInfo = { VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO };
        pipelineInfo.pNext                        = &renderingInfo; // Chain for Dynamic Rendering
        pipelineInfo.stageCount                   = static_cast<uint32_t>( shaderStages.size() );
        pipelineInfo.pStages                      = shaderStages.data();
        pipelineInfo.pVertexInputState            = &vertexInputInfo;
        pipelineInfo.pInputAssemblyState          = &inputAssembly;
        pipelineInfo.pViewportState               = &viewportState;
        pipelineInfo.pRasterizationState          = &rasterizer;
        pipelineInfo.pMultisampleState            = &multisampling;
        pipelineInfo.pDepthStencilState           = &depthStencil;
        pipelineInfo.pColorBlendState             = &colorBlending;
        pipelineInfo.pDynamicState                = &dynamicState;
        pipelineInfo.layout                       = m_resources.pipelineLayout;
        pipelineInfo.renderPass                   = VK_NULL_HANDLE; // Null because of Dynamic Rendering
        pipelineInfo.subpass                      = 0;

        if( m_api->vkCreateGraphicsPipelines( m_device, cache, 1, &pipelineInfo, nullptr, &m_pipeline ) != VK_SUCCESS )
        {
            DT_CORE_CRITICAL( "Failed to create graphics pipeline!" );
        }
    }

    GraphicsPipeline::~GraphicsPipeline()
    {
        if( m_pipeline )
            m_api->vkDestroyPipeline( m_device, m_pipeline, nullptr );
        PipelineUtils::DestroyPipelineLayout( m_device, m_api, m_resources );
    }

    VkDescriptorSetLayout GraphicsPipeline::GetDescriptorSetLayout( uint32_t set ) const
    {
        auto it = m_resources.descriptorSetLayouts.find( set );
        return ( it != m_resources.descriptorSetLayouts.end() ) ? it->second : VK_NULL_HANDLE;
    }

} // namespace DigitalTwin