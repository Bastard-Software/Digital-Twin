#pragma once
#include "core/Base.hpp"
#include "rhi/Buffer.hpp"
#include "rhi/Shader.hpp" // Use existing ShaderReflectionData
#include <vector>
#include <volk.h>

namespace DigitalTwin
{
    class Device; // Forward declaration

    /**
     * @brief Represents a specific instance of resource bindings for a pipeline.
     * Acts as a high-level wrapper around VkDescriptorSet.
     * Follows the "Retained Mode" pattern: Configure once -> Build -> Use many times.
     */
    class BindingGroup
    {
    public:
        BindingGroup( Ref<Device> device, VkDescriptorSet set, const ShaderReflectionData& layoutMap );
        ~BindingGroup() = default;

        // --- Data Setting API ---

        // Binds a buffer to a named resource slot
        void Set( const std::string& name, Ref<Buffer> buffer );

        // Future: void Set(const std::string& name, Ref<Texture> texture);

        // --- Operations ---

        /**
         * @brief Commits all pending resource bindings to the GPU.
         * Calls vkUpdateDescriptorSets internally. MUST be called before using the group.
         */
        void Build();

        // Returns the native Vulkan handle for binding
        VkDescriptorSet GetHandle() const { return m_set; }

    private:
        Ref<Device>     m_device;
        VkDescriptorSet m_set;

        // Reference to metadata from Shader (Name -> ShaderResource)
        const ShaderReflectionData& m_layoutMap;

        // Pending writes to be flushed on Build()
        std::vector<VkWriteDescriptorSet>   m_pendingWrites;
        std::vector<VkDescriptorBufferInfo> m_bufferInfos; // Storage to keep pointers valid
    };
} // namespace DigitalTwin