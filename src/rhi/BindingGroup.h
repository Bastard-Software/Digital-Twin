#pragma once
#include "rhi/RHITypes.h"

#include "core/Core.h"
#include <map>
#include <vector>

namespace DigitalTwin
{

    /**
     * @brief Represents a group of resources (buffers, textures) bound to a specific Descriptor Set.
     * Corresponds to a specific 'set = N' in the shader.
     * * Architecture Note:
     * This class is decoupled from a specific Pipeline instance. It only holds a reference
     * to a VkDescriptorSetLayout. It can be reused with ANY pipeline that has a compatible layout
     * at the specified set index.
     */
    class BindingGroup
    {
    public:
        /**
         * @brief Creates a binding group.
         * @param device Pointer to the device.
         * @param allocator Pointer to the descriptor allocator (usually the global one from ResourceManager).
         * @param layout The Vulkan descriptor set layout this group must conform to.
         * @param setIndex The index of this set in the shader (e.g., set = 0).
         * @param reflection Reflection data for name lookups and validation.
         */
        BindingGroup( Device* device, DescriptorAllocator* allocator, VkDescriptorSetLayout layout, uint32_t setIndex,
                      const ShaderReflectionData* reflection );
        ~BindingGroup();

        // --- Binding API ---

        void Bind( const std::string& name, Buffer* buffer, VkDeviceSize offset = 0, VkDeviceSize range = VK_WHOLE_SIZE );
        void Bind( const std::string& name, Texture* texture, VkImageLayout layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL );
        void Bind( const std::string& name, Texture* texture, Sampler* sampler, VkImageLayout layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL );

        /**
         * @brief Binds a buffer to a specific binding point.
         * @param binding The binding index in the shader (layout(binding = N)).
         * @param buffer The buffer to bind.
         * @param offset Offset into the buffer.
         * @param range Size of the range to bind (or VK_WHOLE_SIZE).
         */
        void Bind( uint32_t binding, Buffer* buffer, VkDeviceSize offset = 0, VkDeviceSize range = VK_WHOLE_SIZE );

        /**
         * @brief Binds a texture (Sampled or Storage) to a specific binding point.
         * @param layout The layout the image will be in when accessed by the shader.
         */
        void Bind( uint32_t binding, Texture* texture, VkImageLayout layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL );

        /**
         * @brief Binds a texture with a sampler (Combined Image Sampler).
         */
        void Bind( uint32_t binding, Texture* texture, Sampler* sampler, VkImageLayout layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL );

        // --- Management ---

        /**
         * @brief Commits pending bindings to the GPU.
         * Allocates the Vulkan DescriptorSet if not already allocated.
         * Calls vkUpdateDescriptorSets.
         * @return Result::SUCCESS on success.
         */
        Result Build();

        // --- Getters ---
        VkDescriptorSet       GetHandle() const { return m_set; }
        VkDescriptorSetLayout GetLayout() const { return m_layout; }
        uint32_t              GetSetIndex() const { return m_setIndex; }

    private:
        bool  GetBindingIndex( const std::string& name, uint32_t& outBinding ) const;
        bool ValidateBinding( uint32_t binding, VkDescriptorType type ) const;

        struct BindingInfo
        {
            uint32_t         binding;
            VkDescriptorType type;

            // Union-like storage for descriptor info
            VkDescriptorBufferInfo bufferInfo{};
            VkDescriptorImageInfo  imageInfo{};
        };

    private:
        Device*              m_device;
        DescriptorAllocator* m_allocator;

        VkDescriptorSetLayout m_layout;
        VkDescriptorSet       m_set = VK_NULL_HANDLE;
        uint32_t              m_setIndex;

        const ShaderReflectionData* m_reflection;

        // Pending writes to be flushed on Build()
        // Map ensures bindings are unique and sorted
        std::map<uint32_t, BindingInfo> m_pendingBindings;
        bool                            m_dirty = true;
    };

} // namespace DigitalTwin