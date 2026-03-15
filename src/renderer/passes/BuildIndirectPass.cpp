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
            m_resourceManager->DestroyPipeline( m_pipeline );

        for( uint32_t i = 0; i < FRAMES_IN_FLIGHT; ++i )
        {
            if( m_bindingGroups[ i ].IsValid() )
                m_resourceManager->DestroyBindingGroup( m_bindingGroups[ i ] );
        }
    }

    void BuildIndirectPass::Execute( CommandBuffer* cmd, Scene* scene, uint32_t flightIndex )
    {
        if( !scene->indirectCmdBuffer.IsValid() || !scene->agentCountBuffer.IsValid() || scene->drawCount == 0 )
            return;

        BindingGroup* bg = m_resourceManager->GetBindingGroup( m_bindingGroups[ flightIndex ] );

        // 1. Bind the simulation's CountBuffer (ReadOnly) and renderer's IndirectBuffer (Write)
        bg->Bind( 0, m_resourceManager->GetBuffer( scene->agentCountBuffer ) );
        bg->Bind( 1, m_resourceManager->GetBuffer( scene->indirectCmdBuffer ) );
        bg->Build();

        ComputePipeline* pipeline = m_resourceManager->GetPipeline( m_pipeline );

        cmd->SetPipeline( pipeline );
        cmd->SetBindingGroup( bg, pipeline->GetLayout(), VK_PIPELINE_BIND_POINT_COMPUTE );

        // 2. Local Push Constants to match the standardized 80-byte block
        struct UpdateIndirectPC
        {
            float    dt, totalTime, fParam0, fParam1, fParam2, fParam3, fParam4, fParam5;
            uint32_t offset, maxCapacity, uParam0, grpNdx;
            float    domainSize[ 4 ];
            uint32_t gridSize[ 4 ];
        } pc{};

        pc.grpNdx = scene->drawCount; // Number of groups (Draw Commands) to update

        cmd->PushConstants( pipeline->GetLayout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof( UpdateIndirectPC ), &pc );

        // 3. Dispatch the compute shader
        uint32_t groupX = ( scene->drawCount + 63 ) / 64;
        cmd->Dispatch( groupX, 1, 1 );

        // 4. Crucial Memory Barrier:
        // Ensure that the compute shader has finished writing to the IndirectCmdBuffer
        // BEFORE the upcoming GeometryPass tries to read from it for the draw call!
        VkBufferMemoryBarrier2 indirectBarrier = { VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2 };
        indirectBarrier.srcStageMask           = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        indirectBarrier.srcAccessMask          = VK_ACCESS_2_SHADER_WRITE_BIT;
        indirectBarrier.dstStageMask           = VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT;
        indirectBarrier.dstAccessMask          = VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT;
        indirectBarrier.srcQueueFamilyIndex    = VK_QUEUE_FAMILY_IGNORED;
        indirectBarrier.dstQueueFamilyIndex    = VK_QUEUE_FAMILY_IGNORED;
        indirectBarrier.buffer                 = m_resourceManager->GetBuffer( scene->indirectCmdBuffer )->GetHandle();
        indirectBarrier.offset                 = 0;
        indirectBarrier.size                   = VK_WHOLE_SIZE;

        VkDependencyInfo depInfo         = { VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        depInfo.bufferMemoryBarrierCount = 1;
        depInfo.pBufferMemoryBarriers    = &indirectBarrier;

        cmd->PipelineBarrier( &depInfo );
    }
} // namespace DigitalTwin