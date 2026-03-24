#include "renderer/passes/GeometryPass.h"

#include "core/Log.h"
#include "renderer/Scene.h"
#include "resources/ResourceManager.h"
#include "rhi/BindingGroup.h"
#include "rhi/Buffer.h"
#include "rhi/CommandBuffer.h"
#include "rhi/Device.h"
#include "rhi/Pipeline.h"

namespace DigitalTwin
{
    GeometryPass::GeometryPass( Device* device, ResourceManager* rm )
        : m_device( device )
        , m_resourceManager( rm )
    {
    }

    GeometryPass::~GeometryPass()
    {
    }

    Result GeometryPass::Initialize()
    {
        // Shaders
        m_vertShader = m_resourceManager->CreateShader( "shaders/graphics/geometry.vert" );
        m_fragShader = m_resourceManager->CreateShader( "shaders/graphics/geometry.frag" );

        // Pipeline
        GraphicsPipelineDesc desc{};
        desc.vertexShader           = m_vertShader;
        desc.fragmentShader         = m_fragShader;
        desc.colorAttachmentFormats = { VK_FORMAT_R8G8B8A8_UNORM };
        desc.depthAttachmentFormat  = VK_FORMAT_D32_SFLOAT;
        desc.depthTestEnable        = true;
        desc.depthWriteEnable       = true;
        desc.cullMode               = VK_CULL_MODE_BACK_BIT;

        m_pipeline = m_resourceManager->CreatePipeline( desc );
        if( !m_pipeline.IsValid() )
            return Result::FAIL;

        // Create Binding Group for Set 0
        for( uint32_t i = 0; i < FRAMES_IN_FLIGHT; ++i )
        {
            m_bindingGroups[ i ] = m_resourceManager->CreateBindingGroup( m_pipeline, 0 );
        }

        return Result::SUCCESS;
    }

    void GeometryPass::Shutdown()
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

    void GeometryPass::Execute( CommandBuffer* cmd, BufferHandle cameraUBO, Scene* scene, uint32_t flightIndex )
    {
        if( !scene->vertexBuffer.IsValid() || !scene->indexBuffer.IsValid() || !scene->indirectCmdBuffer.IsValid() )
        {
            return;
        }

        BindingGroup* bg = m_resourceManager->GetBindingGroup( m_bindingGroups[ flightIndex ] );

        // Update bindings dynamically for the current frame
        bg->Bind( 0, m_resourceManager->GetBuffer( cameraUBO ) );
        bg->Bind( 1, m_resourceManager->GetBuffer( scene->vertexBuffer ) );
        bg->Bind( 2, m_resourceManager->GetBuffer( scene->GetAgentReadBuffer() ) );
        bg->Bind( 3, m_resourceManager->GetBuffer( scene->groupDataBuffer ) );
        // Binding 4 must always be updated (Vulkan spec: statically used descriptors must be valid).
        // Fall back to the agent buffer when no phenotype data exists (e.g. no CellCycle behaviour).
        if( scene->phenotypeBuffer.IsValid() )
            bg->Bind( 4, m_resourceManager->GetBuffer( scene->phenotypeBuffer ) );
        else
            bg->Bind( 4, m_resourceManager->GetBuffer( scene->GetAgentReadBuffer() ) );
        // Binding 5: reorder buffer — maps draw-command instance indices to global agent indices
        bg->Bind( 5, m_resourceManager->GetBuffer( scene->agentReorderBuffer ) );
        bg->Build();

        GraphicsPipeline* pipeline = m_resourceManager->GetPipeline( m_pipeline );

        cmd->SetPipeline( pipeline );
        cmd->SetBindingGroup( bg, pipeline->GetLayout(), VK_PIPELINE_BIND_POINT_GRAPHICS );
        cmd->SetIndexBuffer( m_resourceManager->GetBuffer( scene->indexBuffer ), 0, VK_INDEX_TYPE_UINT32 );

        // Rysowanie z bufora Indirect
        Buffer* indirect = m_resourceManager->GetBuffer( scene->indirectCmdBuffer );
        cmd->DrawIndexedIndirect( indirect, 0, scene->drawCount, sizeof( VkDrawIndexedIndirectCommand ) );
    }
} // namespace DigitalTwin