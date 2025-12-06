#pragma once
#include "core/Base.hpp"
#include <vk_mem_alloc.h>
#include <volk.h>

namespace DigitalTwin
{
    class CommandBuffer;

    enum class TextureUsage : uint32_t
    {
        NONE                 = 0,
        SAMPLED              = 1 << 0, // Read by shader (sampler/texture)
        STORAGE              = 1 << 1, // Written by Compute Shader (imageStore)
        RENDER_TARGET        = 1 << 2, // Color Attachment
        DEPTH_STENCIL_TARGET = 1 << 3, // Depth/Stencil Attachment
        TRANSFER_SRC         = 1 << 4, // Can be copied from
        TRANSFER_DST         = 1 << 5  // Can be copied to
    };

    inline TextureUsage operator|( TextureUsage a, TextureUsage b )
    {
        return static_cast<TextureUsage>( static_cast<uint32_t>( a ) | static_cast<uint32_t>( b ) );
    }
    inline TextureUsage operator&( TextureUsage a, TextureUsage b )
    {
        return static_cast<TextureUsage>( static_cast<uint32_t>( a ) & static_cast<uint32_t>( b ) );
    }
    inline bool HasFlag( TextureUsage value, TextureUsage flag )
    {
        return ( static_cast<uint32_t>( value ) & static_cast<uint32_t>( flag ) ) == static_cast<uint32_t>( flag );
    }

    enum class TextureType
    {
        Texture1D,
        Texture2D,
        Texture3D,
        TextureCube, // TODO: Not implemented yet
    };

    struct TextureDesc
    {
        uint32_t width  = 1;
        uint32_t height = 1;
        uint32_t depth  = 1;

        TextureType type   = TextureType::Texture2D;
        VkFormat    format = VK_FORMAT_R8G8B8A8_UNORM;

        TextureUsage usage = TextureUsage::SAMPLED | TextureUsage::STORAGE | TextureUsage::TRANSFER_SRC | TextureUsage::TRANSFER_DST;
    };

    class Texture
    {
    public:
        Texture( VmaAllocator allocator, VkDevice device, const VolkDeviceTable* api );
        ~Texture();

        Result Create( const TextureDesc& desc );
        Result Create1D( uint32_t width, VkFormat format, TextureUsage usage );
        Result Create2D( uint32_t width, uint32_t height, VkFormat format, TextureUsage usage );
        Result Create3D( uint32_t width, uint32_t height, uint32_t depth, VkFormat format, TextureUsage usage );

        void Destroy();

        void TransitionLayout( Ref<CommandBuffer> cmd, VkImageLayout newLayout );

        VkDescriptorImageInfo GetDescriptorInfo( VkSampler sampler = VK_NULL_HANDLE, VkImageLayout layout = VK_IMAGE_LAYOUT_GENERAL ) const;

        VkImage       GetImage() const { return m_image; }
        VkImageView   GetView() const { return m_view; }
        VkExtent3D    GetExtent() const { return m_extent; }
        VkFormat      GetFormat() const { return m_format; }
        VkImageLayout GetCurrentLayout() const { return m_currentLayout; }
        TextureType   GetType() const { return m_type; }

    public:
        // Disable copying, allow moving
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