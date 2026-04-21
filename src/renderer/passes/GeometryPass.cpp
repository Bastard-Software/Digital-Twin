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
        // Phase 2.6.5.c.2 Step D — dedicated frag shader for the Voronoi pipeline.
        // Same two-light lighting as `geometry.frag`, plus wireframe + vertex-
        // count-tint debug paths gated by push-constant `debugFlags`.
        m_voronoiFragShader = m_resourceManager->CreateShader( "shaders/graphics/voronoi_fan.frag" );

        GraphicsPipelineDesc voronoiDesc = desc;
        voronoiDesc.vertexShader   = m_voronoiVertShader;
        voronoiDesc.fragmentShader = m_voronoiFragShader;
        voronoiDesc.cullMode       = VK_CULL_MODE_NONE; // polygon winding depends on neighbour order — disable cull for robustness

        m_voronoiPipeline = m_resourceManager->CreatePipeline( voronoiDesc );
        if( !m_voronoiPipeline.IsValid() )
            return Result::FAIL;

        for( uint32_t i = 0; i < FRAMES_IN_FLIGHT; ++i )
        {
            m_voronoiBindingGroups[ i ] = m_resourceManager->CreateBindingGroup( m_voronoiPipeline, 0 );
        }

        // --- Phase 2.6.5.c.2 Step D.2 — debug markers pipeline ---
        // Line list topology: 16 verts per instance (8 hull rays × 2 endpoints).
        // Depth write DISABLED so markers don't pollute depth for the main
        // passes; depth test ENABLED so markers correctly occlude behind
        // the tube surface where appropriate. Cull NONE — lines have no face.
        m_debugMarkersVertShader = m_resourceManager->CreateShader( "shaders/graphics/debug_markers.vert" );
        m_debugMarkersFragShader = m_resourceManager->CreateShader( "shaders/graphics/debug_markers.frag" );

        GraphicsPipelineDesc debugDesc = desc;
        debugDesc.vertexShader   = m_debugMarkersVertShader;
        debugDesc.fragmentShader = m_debugMarkersFragShader;
        debugDesc.cullMode       = VK_CULL_MODE_NONE;
        debugDesc.topology       = VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
        debugDesc.depthWriteEnable = false;

        m_debugMarkersPipeline = m_resourceManager->CreatePipeline( debugDesc );
        if( !m_debugMarkersPipeline.IsValid() )
            return Result::FAIL;

        for( uint32_t i = 0; i < FRAMES_IN_FLIGHT; ++i )
        {
            m_debugMarkersBindingGroups[ i ] = m_resourceManager->CreateBindingGroup( m_debugMarkersPipeline, 0 );
        }

        // --- Phase 2.6.5.c.2 Step D.3 — debug vectors pipeline (polarity / drift) ---
        m_debugVectorsVertShader = m_resourceManager->CreateShader( "shaders/graphics/debug_vectors.vert" );
        m_debugVectorsFragShader = m_resourceManager->CreateShader( "shaders/graphics/debug_vectors.frag" );

        GraphicsPipelineDesc vecDesc = desc;
        vecDesc.vertexShader     = m_debugVectorsVertShader;
        vecDesc.fragmentShader   = m_debugVectorsFragShader;
        vecDesc.cullMode         = VK_CULL_MODE_NONE;
        vecDesc.topology         = VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
        vecDesc.depthWriteEnable = false;

        m_debugVectorsPipeline = m_resourceManager->CreatePipeline( vecDesc );
        if( !m_debugVectorsPipeline.IsValid() )
            return Result::FAIL;

        for( uint32_t i = 0; i < FRAMES_IN_FLIGHT; ++i )
        {
            m_debugVectorsBindingGroups[ i ] = m_resourceManager->CreateBindingGroup( m_debugVectorsPipeline, 0 );
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
        if( m_voronoiFragShader.IsValid() )
        {
            m_resourceManager->DestroyShader( m_voronoiFragShader );
            m_voronoiFragShader = ShaderHandle::Invalid;
        }
        if( m_debugMarkersPipeline.IsValid() )
        {
            m_resourceManager->DestroyPipeline( m_debugMarkersPipeline );
            m_debugMarkersPipeline = GraphicsPipelineHandle::Invalid;
        }
        if( m_debugMarkersVertShader.IsValid() )
        {
            m_resourceManager->DestroyShader( m_debugMarkersVertShader );
            m_debugMarkersVertShader = ShaderHandle::Invalid;
        }
        if( m_debugMarkersFragShader.IsValid() )
        {
            m_resourceManager->DestroyShader( m_debugMarkersFragShader );
            m_debugMarkersFragShader = ShaderHandle::Invalid;
        }
        if( m_debugVectorsPipeline.IsValid() )
        {
            m_resourceManager->DestroyPipeline( m_debugVectorsPipeline );
            m_debugVectorsPipeline = GraphicsPipelineHandle::Invalid;
        }
        if( m_debugVectorsVertShader.IsValid() )
        {
            m_resourceManager->DestroyShader( m_debugVectorsVertShader );
            m_debugVectorsVertShader = ShaderHandle::Invalid;
        }
        if( m_debugVectorsFragShader.IsValid() )
        {
            m_resourceManager->DestroyShader( m_debugVectorsFragShader );
            m_debugVectorsFragShader = ShaderHandle::Invalid;
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
            if( m_debugMarkersBindingGroups[ i ].IsValid() )
            {
                m_resourceManager->DestroyBindingGroup( m_debugMarkersBindingGroups[ i ] );
                m_debugMarkersBindingGroups[ i ] = BindingGroupHandle::Invalid;
            }
            if( m_debugVectorsBindingGroups[ i ].IsValid() )
            {
                m_resourceManager->DestroyBindingGroup( m_debugVectorsBindingGroups[ i ] );
                m_debugVectorsBindingGroups[ i ] = BindingGroupHandle::Invalid;
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
            // Phase 2.6.5.c.2 Step 1 — surface info (R per cell) for per-vertex
            // cylinder-normal computation. Fall back to the agent buffer when
            // no vessel group is present (R = 0 path in the VS, bit-identical
            // to pre-Step-1 shading).
            if( scene->surfaceInfoBuffer.IsValid() )
                vBg->Bind( 8, m_resourceManager->GetBuffer( scene->surfaceInfoBuffer ) );
            else
                vBg->Bind( 8, m_resourceManager->GetBuffer( scene->GetAgentReadBuffer() ) );
            vBg->Build();

            cmd->SetPipeline( vPipe );
            cmd->SetBindingGroup( vBg, vPipe->GetLayout(), VK_PIPELINE_BIND_POINT_GRAPHICS );

            // Phase 2.6.5.c.2 Step D — push `{ drawIdOffset, debugFlags }`.
            // VS forwards `debugFlags` to the frag shader via a flat varying;
            // 0 means "no debug overlay, render normally" and is bit-identical
            // to the pre-Step-D output.
            struct VoronoiPushConstants { uint32_t drawIdOffset; uint32_t debugFlags; };
            VoronoiPushConstants pushConst{ staticDrawCount, scene->debugFlags };
            cmd->PushConstants( vPipe->GetLayout(), VK_SHADER_STAGE_VERTEX_BIT, 0,
                                sizeof( VoronoiPushConstants ), &pushConst );

            size_t byteOffset = static_cast<size_t>( staticDrawCount ) * sizeof( VkDrawIndexedIndirectCommand );
            cmd->DrawIndexedIndirect( indirect, byteOffset, scene->dynamicDrawCount, sizeof( VkDrawIndexedIndirectCommand ) );
        }

        // --- Pass C: Phase 2.6.5.c.2 Step D.2 debug markers (hull + centers) ---
        // Drawn ONLY when the DYNAMIC_TOPOLOGY_DEBUG_HULL_MARKERS flag is set
        // (bit 2 of scene->debugFlags). Renders 8 radial lines per agent from
        // the cell center (yellow endpoint) to each contact-hull point (cyan
        // endpoint). Requires the contact-hull buffer to be valid (any
        // AgentGroup with Biomechanics allocates it).
        constexpr uint32_t DEBUG_HULL_MARKERS_BIT = 1u << 2;
        if( ( scene->debugFlags & DEBUG_HULL_MARKERS_BIT ) != 0u
            && scene->contactHullBuffer.IsValid()
            && scene->orientationBuffer.IsValid() )
        {
            GraphicsPipeline* dPipe = m_resourceManager->GetPipeline( m_debugMarkersPipeline );
            BindingGroup*     dBg   = m_resourceManager->GetBindingGroup( m_debugMarkersBindingGroups[ flightIndex ] );

            dBg->Bind( 0, m_resourceManager->GetBuffer( cameraUBO ) );
            dBg->Bind( 1, m_resourceManager->GetBuffer( scene->GetAgentReadBuffer() ) );
            dBg->Bind( 2, m_resourceManager->GetBuffer( scene->orientationBuffer ) );
            dBg->Bind( 3, m_resourceManager->GetBuffer( scene->contactHullBuffer ) );
            dBg->Build();

            cmd->SetPipeline( dPipe );
            cmd->SetBindingGroup( dBg, dPipe->GetLayout(), VK_PIPELINE_BIND_POINT_GRAPHICS );

            // 16 verts per instance × agent count. Use totalPaddedAgents as the
            // upper bound — the VS skips dead agents (w == 0) by collapsing to
            // the origin so the extra vertices are invisible.
            if( scene->totalPaddedAgents > 0 )
                cmd->Draw( 16, scene->totalPaddedAgents, 0, 0 );
        }

        // --- Pass D: Phase 2.6.5.c.2 Step D.3 debug vectors ---
        // Polarity arrows (red, mode=0) and drift lines (yellow, mode=1).
        // Two dispatches of the same pipeline; push constant selects mode.
        constexpr uint32_t DEBUG_POLARITY_BIT = 1u << 3;
        constexpr uint32_t DEBUG_DRIFT_BIT    = 1u << 4;
        const bool wantPolarity = ( scene->debugFlags & DEBUG_POLARITY_BIT ) != 0u;
        const bool wantDrift    = ( scene->debugFlags & DEBUG_DRIFT_BIT    ) != 0u;
        if( ( wantPolarity || wantDrift ) && scene->totalPaddedAgents > 0 )
        {
            GraphicsPipeline* vPipe = m_resourceManager->GetPipeline( m_debugVectorsPipeline );
            BindingGroup*     vBg   = m_resourceManager->GetBindingGroup( m_debugVectorsBindingGroups[ flightIndex ] );

            vBg->Bind( 0, m_resourceManager->GetBuffer( cameraUBO ) );
            vBg->Bind( 1, m_resourceManager->GetBuffer( scene->GetAgentReadBuffer() ) );
            // Polarity and initial-positions — fall back to agents if missing
            // so the descriptor still validates; the VS just reads garbage for
            // the inactive mode, and we only draw active modes below.
            if( scene->polarityBuffer.IsValid() )
                vBg->Bind( 2, m_resourceManager->GetBuffer( scene->polarityBuffer ) );
            else
                vBg->Bind( 2, m_resourceManager->GetBuffer( scene->GetAgentReadBuffer() ) );
            if( scene->initialPositionsBuffer.IsValid() )
                vBg->Bind( 3, m_resourceManager->GetBuffer( scene->initialPositionsBuffer ) );
            else
                vBg->Bind( 3, m_resourceManager->GetBuffer( scene->GetAgentReadBuffer() ) );
            vBg->Build();

            cmd->SetPipeline( vPipe );
            cmd->SetBindingGroup( vBg, vPipe->GetLayout(), VK_PIPELINE_BIND_POINT_GRAPHICS );

            struct VectorPushConstants { uint32_t mode; float polarityScale; };

            if( wantPolarity && scene->polarityBuffer.IsValid() )
            {
                VectorPushConstants pc{ 0u, 1.0f }; // 1 world unit per magnitude=1
                cmd->PushConstants( vPipe->GetLayout(), VK_SHADER_STAGE_VERTEX_BIT, 0,
                                    sizeof( VectorPushConstants ), &pc );
                cmd->Draw( 2, scene->totalPaddedAgents, 0, 0 );
            }
            if( wantDrift && scene->initialPositionsBuffer.IsValid() )
            {
                VectorPushConstants pc{ 1u, 1.0f };
                cmd->PushConstants( vPipe->GetLayout(), VK_SHADER_STAGE_VERTEX_BIT, 0,
                                    sizeof( VectorPushConstants ), &pc );
                cmd->Draw( 2, scene->totalPaddedAgents, 0, 0 );
            }
        }
    }
} // namespace DigitalTwin