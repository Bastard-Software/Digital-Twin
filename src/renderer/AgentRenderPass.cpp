#include "renderer/AgentRenderPass.hpp"

#include "core/FileSystem.hpp"
#include "core/Log.hpp"
#include "resources/ResourceManager.hpp"
#include "rhi/BindingGroup.hpp"
#include "rhi/Device.hpp"

namespace DigitalTwin
{

    AgentRenderPass::AgentRenderPass( Ref<Device> device, Ref<ResourceManager> resManager )
        : m_device( device )
        , m_resManager( resManager )
    {
    }

    AgentRenderPass::~AgentRenderPass()
    {
    }

    void AgentRenderPass::Init( VkFormat colorFormat, VkFormat depthFormat )
    {
        // Load shaders directly from assets
        auto vert = m_device->CreateShader( FileSystem::GetPath( "shaders/graphics/cell.vert" ).string() );
        auto frag = m_device->CreateShader( FileSystem::GetPath( "shaders/graphics/cell.frag" ).string() );

        if( !vert || !frag )
        {
            DT_CORE_CRITICAL( "Failed to load AgentRenderPass shaders!" );
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
        desc.cullMode               = VK_CULL_MODE_BACK_BIT;

        m_pipeline = m_device->CreateGraphicsPipeline( desc );
    }

    struct PushConst
    {
        glm::mat4 viewProj;
        uint32_t  targetMeshID;
    };

    void AgentRenderPass::Draw( CommandBuffer* cmd, const Scene& scene )
    {
        DT_CORE_ASSERT( cmd, "CommandBuffer is null!" );

        // Ensure we have data to draw
        if( !scene.instanceBuffer || scene.activeMeshIDs.empty() )
            return;

        cmd->BindGraphicsPipeline( m_pipeline );

        // --- 1. BIND SET 0: POPULATION DATA ---
        // This set is static for the duration of the render pass (all meshes use the same population buffer).
        VkDescriptorSet set0 = VK_NULL_HANDLE;

        // Allocate Set 0 based on the pipeline layout
        if( m_device->AllocateDescriptor( m_pipeline->GetDescriptorSetLayout( 0 ), set0 ) == Result::SUCCESS )
        {
            // Create binding group helper
            auto bindings = CreateRef<BindingGroup>( m_device, set0, m_pipeline->GetReflectionData() );

            // Bind the simulation buffer to the "population" resource in the shader (Set 0, Binding 0)
            bindings->Set( "population", scene.instanceBuffer );
            bindings->Build();

            // Bind Descriptor Set 0
            cmd->BindDescriptorSets( VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipeline->GetLayout(), 0, { set0 } );
        }
        else
        {
            DT_CORE_ERROR( "[AgentRenderPass] Failed to allocate Descriptor Set 0 (Population)!" );
            return;
        }

        // --- 2. DRAW LOOP (Iterate over active mesh types) ---
        for( AssetID meshID: scene.activeMeshIDs )
        {
            // Retrieve mesh from Resource Manager
            auto gpuMesh = m_resManager->GetMesh( meshID );
            if( !gpuMesh )
                continue;

            // --- BIND SET 1: GEOMETRY DATA ---
            VkDescriptorSet set1 = VK_NULL_HANDLE;
            if( m_device->AllocateDescriptor( m_pipeline->GetDescriptorSetLayout( 1 ), set1 ) == Result::SUCCESS )
            {
                auto bindings = CreateRef<BindingGroup>( m_device, set1, m_pipeline->GetReflectionData() );

                // Bind the unified vertex/index buffer to "mesh" (Set 1, Binding 0)
                bindings->Set( "mesh", gpuMesh->GetBuffer() );
                bindings->Build();

                // Bind Descriptor Set 1
                cmd->BindDescriptorSets( VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipeline->GetLayout(), 1, { set1 } );
            }
            else
            {
                DT_CORE_WARN( "[AgentRenderPass] Failed to allocate Set 1 for Mesh {}", meshID );
                continue;
            }

            // --- PUSH CONSTANTS ---
            // Update camera matrix and target mesh ID for filtering
            PushConst pc;
            pc.viewProj     = scene.camera ? scene.camera->GetViewProjection() : glm::mat4( 1.0f );
            pc.targetMeshID = meshID;

            cmd->PushConstants( m_pipeline->GetLayout(), VK_SHADER_STAGE_VERTEX_BIT, 0, pc );

            // --- DRAW CALL ---
            // Use instanced rendering: draw 'instanceCount' instances of this mesh.
            cmd->BindIndexBuffer( gpuMesh->GetBuffer(), gpuMesh->GetIndexOffset(), VK_INDEX_TYPE_UINT32 );
            cmd->DrawIndexed( gpuMesh->GetIndexCount(), scene.instanceCount, 0, 0, 0 );
        }
    }
} // namespace DigitalTwin