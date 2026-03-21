#include "renderer/passes/VesselVisualizationPass.h"

#include "DigitalTwinTypes.h"

#include "resources/ResourceManager.h"
#include "rhi/BindingGroup.h"
#include "rhi/CommandBuffer.h"
#include "rhi/Pipeline.h"
#include "simulation/SimulationState.h"

namespace DigitalTwin
{
    VesselVisualizationPass::VesselVisualizationPass( Device* device, ResourceManager* rm )
        : m_device( device )
        , m_resourceManager( rm )
    {
    }

    VesselVisualizationPass::~VesselVisualizationPass()
    {
    }

    Result VesselVisualizationPass::Initialize( VkFormat colorFormat, VkFormat depthFormat )
    {
        m_vertShader = m_resourceManager->CreateShader( "shaders/graphics/vessel_lines.vert" );
        m_fragShader = m_resourceManager->CreateShader( "shaders/graphics/vessel_lines.frag" );

        GraphicsPipelineDesc desc{};
        desc.vertexShader           = m_vertShader;
        desc.fragmentShader         = m_fragShader;
        desc.colorAttachmentFormats = { colorFormat };
        desc.depthAttachmentFormat  = depthFormat;
        desc.topology               = VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
        desc.depthTestEnable        = true;
        desc.depthWriteEnable       = false;
        desc.cullMode               = VK_CULL_MODE_NONE;
        desc.lineWidth              = 1.0f;

        m_pipeline = m_resourceManager->CreatePipeline( desc );
        if( !m_pipeline.IsValid() )
            return Result::FAIL;

        for( uint32_t i = 0; i < FRAMES_IN_FLIGHT; ++i )
        {
            m_bindingGroups[ i ] = m_resourceManager->CreateBindingGroup( m_pipeline, 0 );
        }
        return Result::SUCCESS;
    }

    void VesselVisualizationPass::Shutdown()
    {
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

    void VesselVisualizationPass::Execute( CommandBuffer* cmd, BufferHandle cameraUBO, const SimulationState* state,
                                           const VesselVisualizationSettings& settings, uint32_t flightIndex )
    {
        if( !state || !state->vesselEdgeBuffer.IsValid() || !state->vesselEdgeCountBuffer.IsValid() )
            return;

        // Determine which agent buffer holds the latest positions
        BufferHandle agentBuf = state->agentBuffers[ state->latestAgentBuffer ];
        if( !agentBuf.IsValid() )
            return;

        BindingGroup* bg = m_resourceManager->GetBindingGroup( m_bindingGroups[ flightIndex ] );

        bg->Bind( 0, m_resourceManager->GetBuffer( cameraUBO ) );
        bg->Bind( 1, m_resourceManager->GetBuffer( agentBuf ) );
        bg->Bind( 2, m_resourceManager->GetBuffer( state->vesselEdgeBuffer ) );
        bg->Bind( 3, m_resourceManager->GetBuffer( state->vesselEdgeCountBuffer ) );
        bg->Build();

        GraphicsPipeline* pipeline = m_resourceManager->GetPipeline( m_pipeline );

        cmd->SetPipeline( pipeline );
        cmd->SetBindingGroup( bg, pipeline->GetLayout(), VK_PIPELINE_BIND_POINT_GRAPHICS );

        cmd->PushConstants( pipeline->GetLayout(), VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof( glm::vec4 ), &settings.lineColor );

        // Draw: 2 vertices per edge, up to totalPaddedAgents edges max capacity
        // The vertex shader clips out-of-range edges via edgeCount guard
        uint32_t maxEdges = state->totalPaddedAgents;
        cmd->Draw( maxEdges * 2, 1, 0, 0 );
    }
} // namespace DigitalTwin
