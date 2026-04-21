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

    Result GeometryPass::Initialize( VkSampleCountFlagBits sampleCount )
    {
        // --- Static mesh pipeline (existing path) ---
        m_vertShader = m_resourceManager->CreateShader( "shaders/graphics/geometry.vert" );
        m_fragShader = m_resourceManager->CreateShader( "shaders/graphics/geometry.frag" );

        GraphicsPipelineDesc desc{};
        desc.vertexShader           = m_vertShader;
        desc.fragmentShader         = m_fragShader;
        desc.colorAttachmentFormats = { VK_FORMAT_R8G8B8A8_UNORM };
        desc.depthAttachmentFormat  = VK_FORMAT_D32_SFLOAT;
        desc.depthTestEnable        = true;
        desc.depthWriteEnable       = true;
        desc.cullMode               = VK_CULL_MODE_BACK_BIT;
        desc.sampleCount            = sampleCount;

        m_pipeline = m_resourceManager->CreatePipeline( desc );
        if( !m_pipeline.IsValid() )
            return Result::FAIL;

        for( uint32_t i = 0; i < FRAMES_IN_FLIGHT; ++i )
        {
            m_bindingGroups[ i ] = m_resourceManager->CreateBindingGroup( m_pipeline, 0 );
        }

        // --- Phase 2.6.5.c dynamic-topology pipeline (voronoi_fan.vert + reused frag) ---
        // Different VS → different descriptor set layout (binding 7 = PolygonBuffer,
        // no binding 1 = vertex buffer). Separate BindingGroups are required.
        // Cull mode FRONT: polygon vertices are ordered counterclockwise around the
        // cell centre when viewed from OUTSIDE the vessel (radial-outward normal
        // convention from VesselTreeGenerator), so the fan triangulation produces
        // clockwise winding in screen space → flipped cull direction vs the static
        // mesh pipeline. Disabling cull entirely would double-shade back faces.
        m_voronoiVertShader = m_resourceManager->CreateShader( "shaders/graphics/voronoi_fan.vert" );

        GraphicsPipelineDesc voronoiDesc = desc;
        voronoiDesc.vertexShader   = m_voronoiVertShader;
        voronoiDesc.fragmentShader = m_fragShader; // reuse
        voronoiDesc.cullMode       = VK_CULL_MODE_NONE; // polygon winding depends on neighbour order — disable cull for robustness

        m_voronoiPipeline = m_resourceManager->CreatePipeline( voronoiDesc );
        if( !m_voronoiPipeline.IsValid() )
            return Result::FAIL;

        for( uint32_t i = 0; i < FRAMES_IN_FLIGHT; ++i )
        {
            m_voronoiBindingGroups[ i ] = m_resourceManager->CreateBindingGroup( m_voronoiPipeline, 0 );
        }

        return Result::SUCCESS;
    }

    void GeometryPass::Shutdown()
    {
        if( m_pipeline.IsValid() )
        {
            m_resourceManager->DestroyPipeline( m_pipeline );
            m_pipeline = GraphicsPipelineHandle::Invalid;
        }
        if( m_voronoiPipeline.IsValid() )
        {
            m_resourceManager->DestroyPipeline( m_voronoiPipeline );
            m_voronoiPipeline = GraphicsPipelineHandle::Invalid;
        }
        if( m_vertShader.IsValid() )
        {
            m_resourceManager->DestroyShader( m_vertShader );
            m_vertShader = ShaderHandle::Invalid;
        }
        if( m_voronoiVertShader.IsValid() )
        {
            m_resourceManager->DestroyShader( m_voronoiVertShader );
            m_voronoiVertShader = ShaderHandle::Invalid;
        }
        if( m_fragShader.IsValid() )
        {
            m_resourceManager->DestroyShader( m_fragShader );
            m_fragShader = ShaderHandle::Invalid;
        }

        for( uint32_t i = 0; i < FRAMES_IN_FLIGHT; ++i )
        {
            if( m_bindingGroups[ i ].IsValid() )
            {
                m_resourceManager->DestroyBindingGroup( m_bindingGroups[ i ] );
                m_bindingGroups[ i ] = BindingGroupHandle::Invalid;
            }
            if( m_voronoiBindingGroups[ i ].IsValid() )
            {
                m_resourceManager->DestroyBindingGroup( m_voronoiBindingGroups[ i ] );
                m_voronoiBindingGroups[ i ] = BindingGroupHandle::Invalid;
            }
        }
    }

    void GeometryPass::Execute( CommandBuffer* cmd, BufferHandle cameraUBO, Scene* scene, uint32_t flightIndex )
    {
        // Phase 2.6.5.c: each path has independent buffer requirements.
        // Static path needs a vertex buffer; dynamic-topology path doesn't. A
        // vessel demo where every group opts into dynamic topology has no
        // static meshes uploaded → vertexBuffer stays invalid but we still
        // need to run the dynamic path. Guard each pass separately.
        if( !scene->indexBuffer.IsValid() || !scene->indirectCmdBuffer.IsValid() )
        {
            return;
        }

        Buffer*        indirect        = m_resourceManager->GetBuffer( scene->indirectCmdBuffer );
        const uint32_t staticDrawCount = scene->StaticDrawCount();

        cmd->SetIndexBuffer( m_resourceManager->GetBuffer( scene->indexBuffer ), 0, VK_INDEX_TYPE_UINT32 );

        // --- Pass A: static-mesh draws (DrawMetas [0, staticDrawCount)) ---
        if( staticDrawCount > 0 && scene->vertexBuffer.IsValid() )
        {
            BindingGroup* bg = m_resourceManager->GetBindingGroup( m_bindingGroups[ flightIndex ] );
            bg->Bind( 0, m_resourceManager->GetBuffer( cameraUBO ) );
            bg->Bind( 1, m_resourceManager->GetBuffer( scene->vertexBuffer ) );
            bg->Bind( 2, m_resourceManager->GetBuffer( scene->GetAgentReadBuffer() ) );
            bg->Bind( 3, m_resourceManager->GetBuffer( scene->groupDataBuffer ) );
            if( scene->phenotypeBuffer.IsValid() )
                bg->Bind( 4, m_resourceManager->GetBuffer( scene->phenotypeBuffer ) );
            else
                bg->Bind( 4, m_resourceManager->GetBuffer( scene->GetAgentReadBuffer() ) );
            bg->Bind( 5, m_resourceManager->GetBuffer( scene->agentReorderBuffer ) );
            if( scene->orientationBuffer.IsValid() )
                bg->Bind( 6, m_resourceManager->GetBuffer( scene->orientationBuffer ) );
            else
                bg->Bind( 6, m_resourceManager->GetBuffer( scene->GetAgentReadBuffer() ) );
            bg->Build();

            GraphicsPipeline* pipeline = m_resourceManager->GetPipeline( m_pipeline );
            cmd->SetPipeline( pipeline );
            cmd->SetBindingGroup( bg, pipeline->GetLayout(), VK_PIPELINE_BIND_POINT_GRAPHICS );
            cmd->DrawIndexedIndirect( indirect, 0, staticDrawCount, sizeof( VkDrawIndexedIndirectCommand ) );
        }

        // --- Pass B: Phase 2.6.5.c dynamic-topology draws ---
        //   DrawMetas for opted-in AgentGroups sit at the END of the indirect
        //   buffer ([staticDrawCount, drawCount)). A second DrawIndexedIndirect
        //   with a byte offset into the same buffer dispatches them; the
        //   per-pipeline gl_DrawIDARB starts at 0 so voronoi_fan.vert applies
        //   `drawIdOffset = staticDrawCount` to the color lookup.
        if( scene->dynamicDrawCount > 0 && scene->polygonBuffer.IsValid() )
        {
            GraphicsPipeline* vPipe = m_resourceManager->GetPipeline( m_voronoiPipeline );
            BindingGroup*     vBg   = m_resourceManager->GetBindingGroup( m_voronoiBindingGroups[ flightIndex ] );

            // Voronoi VS bindings: no vertex buffer (binding 1 absent from VS),
            // polygon buffer at binding 7, everything else shared with static path.
            vBg->Bind( 0, m_resourceManager->GetBuffer( cameraUBO ) );
            vBg->Bind( 2, m_resourceManager->GetBuffer( scene->GetAgentReadBuffer() ) );
            vBg->Bind( 3, m_resourceManager->GetBuffer( scene->groupDataBuffer ) );
            if( scene->phenotypeBuffer.IsValid() )
                vBg->Bind( 4, m_resourceManager->GetBuffer( scene->phenotypeBuffer ) );
            else
                vBg->Bind( 4, m_resourceManager->GetBuffer( scene->GetAgentReadBuffer() ) );
            vBg->Bind( 5, m_resourceManager->GetBuffer( scene->agentReorderBuffer ) );
            if( scene->orientationBuffer.IsValid() )
                vBg->Bind( 6, m_resourceManager->GetBuffer( scene->orientationBuffer ) );
            else
                vBg->Bind( 6, m_resourceManager->GetBuffer( scene->GetAgentReadBuffer() ) );
            vBg->Bind( 7, m_resourceManager->GetBuffer( scene->polygonBuffer ) );
            vBg->Build();

            cmd->SetPipeline( vPipe );
            cmd->SetBindingGroup( vBg, vPipe->GetLayout(), VK_PIPELINE_BIND_POINT_GRAPHICS );

            uint32_t pushConst = staticDrawCount;
            cmd->PushConstants( vPipe->GetLayout(), VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof( uint32_t ), &pushConst );

            size_t byteOffset = static_cast<size_t>( staticDrawCount ) * sizeof( VkDrawIndexedIndirectCommand );
            cmd->DrawIndexedIndirect( indirect, byteOffset, scene->dynamicDrawCount, sizeof( VkDrawIndexedIndirectCommand ) );
        }
    }
} // namespace DigitalTwin