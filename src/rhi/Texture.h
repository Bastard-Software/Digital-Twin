#pragma once
#include "rhi/RHITypes.h"

#include "core/Core.h"
#include <vk_mem_alloc.h>
#include <volk.h>

namespace DigitalTwin
{

    class Texture
    {
    public:
        Texture( VmaAllocator allocator, VkDevice device, const VolkDeviceTable* api );
        ~Texture();

        Result Create( const TextureDesc& desc );
        void   Destroy();

        VkDescriptorImageInfo GetDescriptorInfo( VkSampler sampler = VK_NULL_HANDLE, VkImageLayout layout = VK_IMAGE_LAYOUT_GENERAL ) const;

        VkImage       GetHandle() const { return m_image; }
        VkImageView   GetView() const { return m_view; }
        VkExtent3D    GetExtent() const { return m_extent; }
        VkFormat      GetFormat() const { return m_format; }
        VkImageLayout GetCurrentLayout() const { return m_currentLayout; }
        TextureType   GetType() const { return m_type; }

    public:
        // Disable copying (RAII), allow moving
        Texture( const Texture& )            = delete;
        Texture& operator=( const Texture& ) = delete;
        Texture( Texture&& other ) noexcept;
        Texture& operator=( Texture&& other ) noexcept;

    private:
        VmaAllocator           m_allocator;
        VkDevice               m_device;
        const VolkDeviceTable* m_api; // Local function table

        VkImage       m_image      = VK_NULL_HANDLE;
        VkImageView   m_view       = VK_NULL_HANDLE;
        VmaAllocation m_allocation = VK_NULL_HANDLE;

        VkExtent3D    m_extent        = { 0, 0, 0 };
        VkFormat      m_format        = VK_FORMAT_UNDEFINED;
        VkImageLayout m_currentLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        TextureType   m_type          = TextureType::Texture2D;
    };

} // namespace DigitalTwin