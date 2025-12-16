#include "renderer/SimulationPass.hpp"

#include "core/FileSystem.hpp"
#include "core/Log.hpp"
#include "rhi/BindingGroup.hpp"
#include "rhi/Device.hpp"

namespace DigitalTwin
{

    SimulationPass::SimulationPass( Ref<Device> device )
        : m_device( device )
    {
    }
    SimulationPass::~SimulationPass()
    {
    }

    void SimulationPass::Init( VkFormat colorFormat, VkFormat depthFormat )
    {
        // Load shaders directly from assets
        auto vert = m_device->CreateShader( FileSystem::GetPath( "shaders/graphics/cell.vert" ).string() );
        auto frag = m_device->CreateShader( FileSystem::GetPath( "shaders/graphics/cell.frag" ).string() );

        if( !vert || !frag )
        {
            DT_CORE_CRITICAL( "Failed to load SimulationPass shaders!" );
            return;
        }

        GraphicsPipelineDesc desc;
        desc.vertexShader           = vert;
        desc.fragmentShader         = frag;
        desc.colorAttachmentFormats = { colorFormat };
        desc.depthAttachmentFormat  = depthFormat; // Disabled depth for now
        desc.depthTestEnable        = true;
        desc.depthWriteEnable       = true;
        desc.blendEnable            = false; // Nice for scientific viz
        desc.topology               = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        desc.cullMode               = VK_CULL_MODE_NONE;

        m_pipeline = m_device->CreateGraphicsPipeline( desc );
    }

    struct PushConst
    {
        glm::mat4 viewProj;
    };

    void SimulationPass::Draw( CommandBuffer* cmd, const Scene& scene )
    {
        if( !scene.instanceBuffer || scene.instanceCount == 0 )
            return;

        cmd->BindGraphicsPipeline( m_pipeline );

        // Transient Binding Group creation (per frame)
        // SET 0
        VkDescriptorSet set0 = VK_NULL_HANDLE;
        if( m_device->AllocateDescriptor( m_pipeline->GetDescriptorSetLayout( 0 ), set0 ) == Result::SUCCESS )
        {
            auto bindings = CreateRef<BindingGroup>( m_device, set0, m_pipeline->GetReflectionData() );
            bindings->Set( "population", scene.instanceBuffer );
            bindings->Build();
            cmd->BindDescriptorSets( VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipeline->GetLayout(), 0, { set0 } );
        }
        // SET 1
        VkDescriptorSet set1 = VK_NULL_HANDLE;
        if( m_device->AllocateDescriptor( m_pipeline->GetDescriptorSetLayout( 1 ), set1 ) == Result::SUCCESS )
        {
            auto bindings = CreateRef<BindingGroup>( m_device, set1, m_pipeline->GetReflectionData() );
            bindings->Set( "mesh", scene.mesh->GetBuffer() );
            bindings->Build();
            cmd->BindDescriptorSets( VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipeline->GetLayout(), 1, { set1 } );
        }

        PushConst pc;
        pc.viewProj = scene.camera ? scene.camera->GetViewProjection() : glm::mat4( 1.0f );
        cmd->PushConstants( m_pipeline->GetLayout(), VK_SHADER_STAGE_VERTEX_BIT, 0, pc );

        cmd->BindIndexBuffer( scene.mesh->GetBuffer(), scene.mesh->GetIndexOffset(), VK_INDEX_TYPE_UINT32 );

        // Instanced Draw: IndexCount * N instances
        cmd->DrawIndexed( scene.mesh->GetIndexCount(), scene.instanceCount, 0, 0, 0 );
    }
} // namespace DigitalTwin