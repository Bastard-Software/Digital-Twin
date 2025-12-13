#include "renderer/SimulationPass.hpp"

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
        auto vert = m_device->CreateShader( "assets/shaders/graphics/cell.vert" );
        auto frag = m_device->CreateShader( "assets/shaders/graphics/cell.frag" );

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
        desc.depthTestEnable        = false;
        desc.depthWriteEnable       = false;
        desc.blendEnable            = true; // Nice for scientific viz
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
        VkDescriptorSetLayout layout = m_pipeline->GetDescriptorSetLayout( 0 );
        VkDescriptorSet       set    = VK_NULL_HANDLE;

        if( m_device->AllocateDescriptor( layout, set ) == Result::SUCCESS )
        {
            auto bindings = CreateRef<BindingGroup>( m_device, set, m_pipeline->GetReflectionData() );
            bindings->Set( "population", scene.instanceBuffer );
            bindings->Build();

            cmd->BindDescriptorSets( VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipeline->GetLayout(), 0, { set } );
        }

        PushConst pc;
        pc.viewProj = scene.camera ? scene.camera->GetViewProjection() : glm::mat4( 1.0f );

        cmd->PushConstants( m_pipeline->GetLayout(), VK_SHADER_STAGE_VERTEX_BIT, 0, pc );

        // Instanced Draw: 36 vertices (Cube) * N instances
        cmd->Draw( 36, scene.instanceCount, 0, 0 );
    }
} // namespace DigitalTwin