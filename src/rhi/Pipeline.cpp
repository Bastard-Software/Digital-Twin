#include "rhi/Pipeline.h"

#include "core/Log.h"
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
        PipelineLayoutResult CreatePipelineLayout( VkDevice device, const VolkDeviceTable* api, const std::vector<Shader*>& shaders )
        {
            PipelineLayoutResult result;

            // 1. Merge Resources & Push Constants
            for( const auto& shader: shaders )
            {
                if( !shader )
                    continue;
                const auto& shaderData = shader->GetReflectionData();

                // Merge Resources
                for( const auto& [ setIdx, bindings ]: shaderData.resources )
                {
                    for( const auto& [ bindingIdx, res ]: bindings )
                    {
                        // Check if exists
                        if( result.reflectionData.Find( setIdx, bindingIdx ) )
                        {
                            // Already exists: Merge stage flags
                            result.reflectionData.resources[ setIdx ][ bindingIdx ].stageFlags |= res.stageFlags;
                        }
                        else
                        {
                            // New: Insert
                            result.reflectionData.resources[ setIdx ][ bindingIdx ] = res;
                        }
                    }
                }

                // Merge Push Constants (Simplified: Just accumulate all ranges)
                // Note: Proper merging would check for overlapping ranges.
                for( const auto& pc: shaderData.pushConstants )
                {
                    result.reflectionData.pushConstants.push_back( pc );
                }
            }

            // 2. Create Descriptor Set Layouts (using merged data)
            std::vector<VkDescriptorSetLayout> setLayoutsVector;

            // Need to iterate in order of set index to create pipeline layout correctly (though gaps are allowed in vulkan, usually contiguous)
            // Finding max set index
            uint32_t maxSet = 0;
            for( const auto& [ set, _ ]: result.reflectionData.resources )
                maxSet = std::max( maxSet, set );

            for( uint32_t set = 0; set <= maxSet; ++set )
            {
                // If set exists in resources
                if( result.reflectionData.resources.count( set ) )
                {
                    std::vector<VkDescriptorSetLayoutBinding> bindings;
                    const auto&                               setBindings = result.reflectionData.resources[ set ];

                    for( const auto& [ bindingIdx, res ]: setBindings )
                    {
                        VkDescriptorSetLayoutBinding b{};
                        b.binding            = bindingIdx;
                        b.descriptorType     = MapResourceTypeToVulkan( res.type );
                        b.descriptorCount    = res.arraySize;
                        b.stageFlags         = res.stageFlags;
                        b.pImmutableSamplers = nullptr;
                        bindings.push_back( b );
                    }

                    VkDescriptorSetLayoutCreateInfo layoutInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
                    layoutInfo.bindingCount                    = ( uint32_t )bindings.size();
                    layoutInfo.pBindings                       = bindings.data();

                    VkDescriptorSetLayout layout;
                    if( api->vkCreateDescriptorSetLayout( device, &layoutInfo, nullptr, &layout ) != VK_SUCCESS )
                    {
                        DT_ASSERT( false, "Failed to create descriptor set layout for set {}", set );
                        DT_ERROR( "Failed to create descriptor set layout for set {}", set );
                    }

                    result.descriptorSetLayouts[ set ] = layout;
                    setLayoutsVector.push_back( layout );
                }
                else
                {
                    VkDescriptorSetLayoutCreateInfo layoutInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
                    layoutInfo.bindingCount                    = 0;
                    layoutInfo.pBindings                       = nullptr;

                    VkDescriptorSetLayout layout;
                    if( api->vkCreateDescriptorSetLayout( device, &layoutInfo, nullptr, &layout ) != VK_SUCCESS )
                    {
                        DT_ASSERT( false, "Failed to create empty descriptor set layout for gap at set {}", set );
                        DT_ERROR( "Failed to create empty descriptor set layout for gap at set {}", set );
                    }
                    else
                    {
                        // We store it so we can destroy it later, but we don't add it to reflection resources
                        // since there are no resources to bind there.
                        result.descriptorSetLayouts[ set ] = layout;
                        setLayoutsVector.push_back( layout );
                    }
                }
            }

            // 3. Create Pipeline Layout
            VkPipelineLayoutCreateInfo pipelineInfo = { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
            pipelineInfo.setLayoutCount             = ( uint32_t )setLayoutsVector.size();
            pipelineInfo.pSetLayouts                = setLayoutsVector.data();

            // Push Constants
            pipelineInfo.pushConstantRangeCount = ( uint32_t )result.reflectionData.pushConstants.size();
            pipelineInfo.pPushConstantRanges    = result.reflectionData.pushConstants.data();

            if( api->vkCreatePipelineLayout( device, &pipelineInfo, nullptr, &result.pipelineLayout ) != VK_SUCCESS )
            {
                DT_ASSERT( false, "Failed to create pipeline layout!" );
                DT_ERROR( "Failed to create pipeline layout!" );
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
    } // namespace PipelineUtils

    ComputePipeline::ComputePipeline( VkDevice device, const VolkDeviceTable* api )
        : m_device( device )
        , m_api( api )
    {
    }

    ComputePipeline::~ComputePipeline()
    {
    }

    Result ComputePipeline::Create( const ComputePipelineNativeDesc& desc )
    {
        DT_CORE_ASSERT( desc.shader, "ComputePipeline requires a shader!" );

        // 1. Create Layouts (Layouts are owned by this pipeline instance)
        m_resources = PipelineUtils::CreatePipelineLayout( m_device, m_api, { desc.shader } );

        // 2. Create Pipeline
        VkComputePipelineCreateInfo pipelineInfo = { VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
        pipelineInfo.layout                      = m_resources.pipelineLayout;
        pipelineInfo.stage.sType                 = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        pipelineInfo.stage.stage                 = VK_SHADER_STAGE_COMPUTE_BIT;
        pipelineInfo.stage.module                = desc.shader->GetModule();
        pipelineInfo.stage.pName                 = "main";

        if( m_api->vkCreateComputePipelines( m_device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &m_pipeline ) != VK_SUCCESS )
        {
            DT_CRITICAL( "Failed to create compute pipeline!" );
            return Result::FAIL;
        }

        return Result::SUCCESS;
    }

    void ComputePipeline::Destroy()
    {
        if( m_pipeline && m_api )
            m_api->vkDestroyPipeline( m_device, m_pipeline, nullptr );
        PipelineUtils::DestroyPipelineLayout( m_device, m_api, m_resources );
    }

    VkDescriptorSetLayout ComputePipeline::GetDescriptorSetLayout( uint32_t set ) const
    {
        auto it = m_resources.descriptorSetLayouts.find( set );
        return ( it != m_resources.descriptorSetLayouts.end() ) ? it->second : VK_NULL_HANDLE;
    }

    ComputePipeline::ComputePipeline( ComputePipeline&& other ) noexcept
        : m_device( other.m_device )
        , m_api( other.m_api )
        , m_pipeline( other.m_pipeline )
        , m_resources( std::move( other.m_resources ) )
    {
        other.m_pipeline                 = VK_NULL_HANDLE;
        other.m_resources.pipelineLayout = VK_NULL_HANDLE;
        other.m_resources.descriptorSetLayouts.clear();
    }

    ComputePipeline& ComputePipeline::operator=( ComputePipeline&& other ) noexcept
    {
        if( this != &other )
        {
            Destroy();
            m_device    = other.m_device;
            m_api       = other.m_api;
            m_pipeline  = other.m_pipeline;
            m_resources = std::move( other.m_resources );

            other.m_pipeline                 = VK_NULL_HANDLE;
            other.m_resources.pipelineLayout = VK_NULL_HANDLE;
            other.m_resources.descriptorSetLayouts.clear();
        }
        return *this;
    }

    GraphicsPipeline::GraphicsPipeline( VkDevice device, const VolkDeviceTable* api )
        : m_device( device )
        , m_api( api )
    {
    }

    GraphicsPipeline::~GraphicsPipeline()
    {
    }

    Result GraphicsPipeline::Create( const GraphicsPipelineNativeDesc& desc )
    {
        DT_CORE_ASSERT( desc.vertexShader, "GraphicsPipeline requires a Vertex Shader!" );

        // 1. Create Layouts (Merge VS + FS resources)
        std::vector<Shader*> shaders = { desc.vertexShader };
        if( desc.fragmentShader )
            shaders.push_back( desc.fragmentShader );

        m_resources = PipelineUtils::CreatePipelineLayout( m_device, m_api, shaders );

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

        if( m_api->vkCreateGraphicsPipelines( m_device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &m_pipeline ) != VK_SUCCESS )
        {
            DT_CRITICAL( "Failed to create graphics pipeline!" );
            return Result::FAIL;
        }

        return Result::SUCCESS;
    }

    void GraphicsPipeline::Destroy()
    {
        if( m_pipeline && m_api )
            m_api->vkDestroyPipeline( m_device, m_pipeline, nullptr );
        PipelineUtils::DestroyPipelineLayout( m_device, m_api, m_resources );
    }

    VkDescriptorSetLayout GraphicsPipeline::GetDescriptorSetLayout( uint32_t set ) const
    {
        auto it = m_resources.descriptorSetLayouts.find( set );
        return ( it != m_resources.descriptorSetLayouts.end() ) ? it->second : VK_NULL_HANDLE;
    }

    GraphicsPipeline::GraphicsPipeline( GraphicsPipeline&& other ) noexcept
        : m_device( other.m_device )
        , m_api( other.m_api )
        , m_pipeline( other.m_pipeline )
        , m_resources( std::move( other.m_resources ) )
    {
        other.m_pipeline                 = VK_NULL_HANDLE;
        other.m_resources.pipelineLayout = VK_NULL_HANDLE;
        other.m_resources.descriptorSetLayouts.clear();
    }

    GraphicsPipeline& GraphicsPipeline::operator=( GraphicsPipeline&& other ) noexcept
    {
        if( this != &other )
        {
            Destroy();

            m_device    = other.m_device;
            m_api       = other.m_api;
            m_pipeline  = other.m_pipeline;
            m_resources = std::move( other.m_resources );

            other.m_pipeline                 = VK_NULL_HANDLE;
            other.m_resources.pipelineLayout = VK_NULL_HANDLE;
            other.m_resources.descriptorSetLayouts.clear();
        }
        return *this;
    }

} // namespace DigitalTwin