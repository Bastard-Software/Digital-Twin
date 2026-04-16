#include "renderer/passes/BuildIndirectPass.h"

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
    BuildIndirectPass::BuildIndirectPass( Device* device, ResourceManager* rm )
        : m_device( device )
        , m_resourceManager( rm )
    {
    }

    BuildIndirectPass::~BuildIndirectPass()
    {
    }

    Result BuildIndirectPass::Initialize()
    {
        m_compShader = m_resourceManager->CreateShader( "shaders/graphics/build_indirect.comp" );

        ComputePipelineDesc desc{};
        desc.shader    = m_compShader;
        desc.debugName = "BuildIndirectPipeline";
        m_pipeline     = m_resourceManager->CreatePipeline( desc );

        for( uint32_t i = 0; i < FRAMES_IN_FLIGHT; ++i )
        {
            m_bindingGroups[ i ] = m_resourceManager->CreateBindingGroup( m_pipeline, 0 );
        }

        return Result::SUCCESS;
    }

    void BuildIndirectPass::Shutdown()
    {
        if( m_pipeline.IsValid() )
        {
            m_resourceManager->DestroyPipeline( m_pipeline );
            m_pipeline = ComputePipelineHandle::Invalid;
        }
        if( m_compShader.IsValid() )
        {
            m_resourceManager->DestroyShader( m_compShader );
            m_compShader = ShaderHandle::Invalid;
        }

        for( uint32_t i = 0; i < FRAMES_IN_FLIGHT; ++i )
        {
            if( m_bindingGroups[ i ].IsValid() )
            {
                m_resourceManager->DestroyBindingGroup( m_bindingGroups[ i ] );
                m_bindingGroups[ i ] = BindingGroupHandle::Invalid;
            }
        }
    }

    void BuildIndirectPass::Execute( CommandBuffer* cmd, Scene* scene, uint32_t flightIndex )
    {
        if( !scene->indirectCmdBuffer.IsValid() || !scene->agentCountBuffer.IsValid() || scene->drawCount == 0 )
            return;

        BindingGroup* bg = m_resourceManager->GetBindingGroup( m_bindingGroups[ flightIndex ] );

        // Bind all 6 buffers for the multi-mesh build_indirect shader
        bg->Bind( 0, m_resourceManager->GetBuffer( scene->agentCountBuffer ) );
        bg->Bind( 1, m_resourceManager->GetBuffer( scene->indirectCmdBuffer ) );

        // Phenotype buffer (binding 2) — fall back to agent buffer if no phenotypes
        if( scene->phenotypeBuffer.IsValid() )
            bg->Bind( 2, m_resourceManager->GetBuffer( scene->phenotypeBuffer ) );
        else
            bg->Bind( 2, m_resourceManager->GetBuffer( scene->GetAgentReadBuffer() ) );

        // Reorder buffer (binding 3)
        bg->Bind( 3, m_resourceManager->GetBuffer( scene->agentReorderBuffer ) );

        // Draw meta buffer (binding 4)
        bg->Bind( 4, m_resourceManager->GetBuffer( scene->drawMetaBuffer ) );

        // Agent positions (binding 5) — for alive check
        bg->Bind( 5, m_resourceManager->GetBuffer( scene->GetAgentReadBuffer() ) );

        // Visibility flags (binding 6) — per-group, CPU-writable UPLOAD buffer
        bg->Bind( 6, m_resourceManager->GetBuffer( scene->visibilityBuffer ) );

        bg->Build();

        ComputePipeline* pipeline = m_resourceManager->GetPipeline( m_pipeline );

        cmd->SetPipeline( pipeline );
        cmd->SetBindingGroup( bg, pipeline->GetLayout(), VK_PIPELINE_BIND_POINT_COMPUTE );

        // Standard 80-byte push constant block
        struct UpdateIndirectPC
        {
            float    dt, totalTime, fParam0, fParam1, fParam2, fParam3, fParam4, fParam5;
            uint32_t offset, maxCapacity, uParam0, grpNdx;
            float    domainSize[ 4 ];
            uint32_t gridSize[ 4 ];
        };

        // --- Dispatch 1: RESET — zero all instanceCounts ---
        UpdateIndirectPC resetPC{};
        resetPC.grpNdx = scene->drawCount;
        resetPC.uParam0 = 0; // reset mode

        cmd->PushConstants( pipeline->GetLayout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof( UpdateIndirectPC ), &resetPC );
        cmd->Dispatch( ( scene->drawCount + 255 ) / 256, 1, 1 );

        // Barrier: reset writes must complete before classify reads
        VkMemoryBarrier2 computeBarrier  = { VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
        computeBarrier.srcStageMask      = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        computeBarrier.srcAccessMask     = VK_ACCESS_2_SHADER_WRITE_BIT;
        computeBarrier.dstStageMask      = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        computeBarrier.dstAccessMask     = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT;

        VkDependencyInfo midDep         = { VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        midDep.memoryBarrierCount       = 1;
        midDep.pMemoryBarriers          = &computeBarrier;
        cmd->PipelineBarrier( &midDep );

        // --- Dispatch 2: CLASSIFY — scatter agents into reorder buffer ---
        UpdateIndirectPC classifyPC{};
        classifyPC.grpNdx      = scene->drawCount;
        classifyPC.maxCapacity = scene->totalPaddedAgents;
        classifyPC.uParam0     = 1; // classify mode

        cmd->PushConstants( pipeline->GetLayout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof( UpdateIndirectPC ), &classifyPC );
        cmd->Dispatch( ( scene->totalPaddedAgents + 255 ) / 256, 1, 1 );

        // Final barrier: indirect buffer + reorder buffer must be visible to draw indirect + vertex shader
        VkBufferMemoryBarrier2 indirectBarrier = { VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2 };
        indirectBarrier.srcStageMask           = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        indirectBarrier.srcAccessMask          = VK_ACCESS_2_SHADER_WRITE_BIT;
        indirectBarrier.dstStageMask           = VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT;
        indirectBarrier.dstAccessMask          = VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_2_SHADER_READ_BIT;
        indirectBarrier.srcQueueFamilyIndex    = VK_QUEUE_FAMILY_IGNORED;
        indirectBarrier.dstQueueFamilyIndex    = VK_QUEUE_FAMILY_IGNORED;
        indirectBarrier.buffer                 = m_resourceManager->GetBuffer( scene->indirectCmdBuffer )->GetHandle();
        indirectBarrier.offset                 = 0;
        indirectBarrier.size                   = VK_WHOLE_SIZE;

        VkBufferMemoryBarrier2 reorderBarrier = { VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2 };
        reorderBarrier.srcStageMask           = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        reorderBarrier.srcAccessMask          = VK_ACCESS_2_SHADER_WRITE_BIT;
        reorderBarrier.dstStageMask           = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT;
        reorderBarrier.dstAccessMask          = VK_ACCESS_2_SHADER_READ_BIT;
        reorderBarrier.srcQueueFamilyIndex    = VK_QUEUE_FAMILY_IGNORED;
        reorderBarrier.dstQueueFamilyIndex    = VK_QUEUE_FAMILY_IGNORED;
        reorderBarrier.buffer                 = m_resourceManager->GetBuffer( scene->agentReorderBuffer )->GetHandle();
        reorderBarrier.offset                 = 0;
        reorderBarrier.size                   = VK_WHOLE_SIZE;

        VkBufferMemoryBarrier2 barriers[] = { indirectBarrier, reorderBarrier };

        VkDependencyInfo depInfo         = { VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        depInfo.bufferMemoryBarrierCount = 2;
        depInfo.pBufferMemoryBarriers    = barriers;

        cmd->PipelineBarrier( &depInfo );
    }
} // namespace DigitalTwin
