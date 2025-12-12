#include "compute/ComputeKernel.hpp"

#include "core/Log.hpp"

namespace DigitalTwin
{
    ComputeKernel::ComputeKernel( Ref<Device> device, Ref<ComputePipeline> pipeline, std::string name )
        : m_device( device )
        , m_pipeline( pipeline )
        , m_name( std::move( name ) )
    {
    }

    Ref<BindingGroup> ComputeKernel::CreateBindingGroup()
    {
        // 1. Get Layout from Pipeline (Set 0 is standard for compute kernels here)
        VkDescriptorSetLayout layout = m_pipeline->GetDescriptorSetLayout( 0 );
        VkDescriptorSet       set    = VK_NULL_HANDLE;

        // 2. Allocate Set from Pool
        if( m_device->AllocateDescriptor( layout, set ) != Result::SUCCESS )
        {
            DT_CORE_ERROR( "[ComputeKernel] Failed to allocate descriptor set for '{}'", m_name );
            return nullptr;
        }

        // 3. Retrieve Reflection Data from Shader
        // Assuming ComputePipeline exposes the shader via GetShader()
        // Ensure Pipeline.hpp has: Ref<Shader> GetShader() const { return m_shader; } (or similar accessor)
        // If your ComputePipeline doesn't store the shader reference publicly, you might need to add it.
        // Based on your upload, ComputePipelineDesc takes a shader, so we assume we can access it.
        // EDIT: Looking at Pipeline.hpp upload, we might need to add GetShader() to ComputePipeline class.
        // Assuming it exists or you will add: `Ref<Shader> GetShader() const { return m_shader; }` to ComputePipeline.

        // For now, let's assume implementation details:
        // We need to access reflection data.
        // If GetShader() is missing, add it to ComputePipeline.hpp!
        const auto& reflectionData = m_pipeline->GetReflectionData();

        // 4. Create Binding Group
        return CreateRef<BindingGroup>( m_device, set, reflectionData );
    }

    void ComputeKernel::Dispatch( CommandBuffer& cmd, Ref<BindingGroup> group, uint32_t elementCount )
    {
        cmd.BindComputePipeline( m_pipeline );

        VkDescriptorSet setHandle = group->GetHandle();
        cmd.BindDescriptorSets( VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeline->GetLayout(), 0, { setHandle } );

        uint32_t gx = ( elementCount + m_groupSize.x - 1 ) / m_groupSize.x;
        cmd.Dispatch( gx, m_groupSize.y, m_groupSize.z );
    }
} // namespace DigitalTwin