#include "renderer/passes/GridVisualizationPass.h"

#include "DigitalTwinTypes.h"

#include "resources/ResourceManager.h"
#include "rhi/BindingGroup.h"
#include "rhi/CommandBuffer.h"
#include "rhi/Pipeline.h"
#include "simulation/SimulationState.h"

namespace DigitalTwin
{
    GridVisualizationPass::GridVisualizationPass( Device* device, ResourceManager* rm )
        : m_device( device )
        , m_resourceManager( rm )
    {
    }

    GridVisualizationPass::~GridVisualizationPass()
    {
    }

    Result GridVisualizationPass::Initialize( VkFormat colorFormat, VkFormat depthFormat, VkSampleCountFlagBits sampleCount )
    {
        m_vertShader = m_resourceManager->CreateShader( "shaders/graphics/grid_overlay.vert" );
        m_fragShader = m_resourceManager->CreateShader( "shaders/graphics/grid_overlay.frag" );

        GraphicsPipelineDesc desc{};
        desc.vertexShader           = m_vertShader;
        desc.fragmentShader         = m_fragShader;
        desc.colorAttachmentFormats = { colorFormat };
        desc.depthAttachmentFormat  = depthFormat;
        desc.sampleCount            = sampleCount;
        desc.depthTestEnable        = false;             // We just overlay based on math, no depth test needed yet
        desc.depthWriteEnable       = false;             // Do not write transparent cloud to depth buffer
        desc.cullMode               = VK_CULL_MODE_NONE; // Fullscreen quad
        desc.topology               = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

        // ENABLE BLENDING for transparency
        desc.blendEnable         = true;
        desc.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
        desc.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        desc.colorBlendOp        = VK_BLEND_OP_ADD;
        desc.srcAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
        desc.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        desc.alphaBlendOp        = VK_BLEND_OP_ADD;

        m_pipeline = m_resourceManager->CreatePipeline( desc );
        if( !m_pipeline.IsValid() )
            return Result::FAIL;

        SamplerDesc samplerDesc{};
        samplerDesc.minFilter    = VK_FILTER_LINEAR;
        samplerDesc.magFilter    = VK_FILTER_LINEAR;
        samplerDesc.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerDesc.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerDesc.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;

        m_linearSampler = m_resourceManager->CreateSampler( samplerDesc );
        if( !m_linearSampler.IsValid() )
            return Result::FAIL;

        for( uint32_t i = 0; i < FRAMES_IN_FLIGHT; ++i )
        {
            m_bindingGroups[ i ] = m_resourceManager->CreateBindingGroup( m_pipeline, 0 );
        }
        return Result::SUCCESS;
    }

    void GridVisualizationPass::Shutdown()
    {
        if( !m_linearSampler.IsValid() )
            m_resourceManager->DestroySampler( m_linearSampler );
        if( m_pipeline.IsValid() )
            m_resourceManager->DestroyPipeline( m_pipeline );
        if( m_vertShader.IsValid() )
            m_resourceManager->DestroyShader( m_vertShader );
        if( m_fragShader.IsValid() )
            m_resourceManager->DestroyShader( m_fragShader );

        for( uint32_t i = 0; i < FRAMES_IN_FLIGHT; ++i )
        {
            if( m_bindingGroups[ i ].IsValid() )
                m_resourceManager->DestroyBindingGroup( m_bindingGroups[ i ] );
        }
    }

    void GridVisualizationPass::Execute( CommandBuffer* cmd, BufferHandle cameraUBO, const GridFieldState* gridState,
                                         const GridVisualizationSettings& settings, const glm::vec3& domainSize, uint32_t flightIndex )
    {
        BindingGroup* bg = m_resourceManager->GetBindingGroup( m_bindingGroups[ flightIndex ] );

        // Bind Camera UBO and the currently READABLE texture from the PDE ping-pong
        bg->Bind( 0, m_resourceManager->GetBuffer( cameraUBO ) );
        bg->Bind( 1, m_resourceManager->GetTexture( gridState->textures[ gridState->currentReadIndex ] ),
                  m_resourceManager->GetSampler( m_linearSampler ), VK_IMAGE_LAYOUT_GENERAL );
        bg->Build();

        GraphicsPipeline* pipeline = m_resourceManager->GetPipeline( m_pipeline );

        // TODO: set layout for read only and transit from currentLayout -> read only

        cmd->SetPipeline( pipeline );
        cmd->SetBindingGroup( bg, pipeline->GetLayout(), VK_PIPELINE_BIND_POINT_GRAPHICS );

        // Setup Push Constants — layout must match PC block in grid_overlay.frag exactly.
        // vec4 is used for custom color stops to satisfy std430 16-byte alignment for vec4.
        struct VisPushConstants
        {
            int       mode;          // 0
            float     sliceZ;        // 4
            float     opacitySlice;  // 8
            float     opacityCloud;  // 12
            glm::vec3 domainSize;    // 16  (lands at 16 because 4 floats precede it)
            float     minValue;      // 28
            float     maxValue;      // 32
            float     alphaCutoff;   // 36
            int       colormap;      // 40
            float     gamma;         // 44
            glm::vec4 customLow;     // 48  (vec4 align=16; 48 is multiple of 16)
            glm::vec4 customMid;     // 64
            glm::vec4 customHigh;    // 80
        } pc;                        // total: 96 bytes

        const auto& fv   = settings.fieldVis;
        pc.mode          = static_cast<int>( settings.mode );
        pc.sliceZ        = settings.sliceZ;
        pc.opacitySlice  = settings.opacitySlice;
        pc.opacityCloud  = settings.opacityCloud;
        pc.domainSize    = domainSize;
        pc.minValue      = fv.minValue;
        pc.maxValue      = fv.maxValue;
        pc.alphaCutoff   = fv.alphaCutoff;
        pc.colormap      = static_cast<int>( fv.colormap );
        pc.gamma         = fv.gamma;
        pc.customLow     = glm::vec4( fv.customLow,  0.0f );
        pc.customMid     = glm::vec4( fv.customMid,  0.0f );
        pc.customHigh    = glm::vec4( fv.customHigh, 0.0f );

        cmd->PushConstants( pipeline->GetLayout(), VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof( VisPushConstants ), &pc );

        // Draw Fullscreen Triangle (3 Vertices generated in shader)
        cmd->Draw( 3, 1, 0, 0 );
    }
} // namespace DigitalTwin