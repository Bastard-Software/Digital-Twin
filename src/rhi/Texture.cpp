#include "rhi/Texture.h"

#include "core/Log.h"

namespace DigitalTwin
{
    // Helper to check if format is a depth format
    static bool IsDepthFormat( VkFormat format )
    {
        return format == VK_FORMAT_D16_UNORM || format == VK_FORMAT_X8_D24_UNORM_PACK32 || format == VK_FORMAT_D32_SFLOAT ||
               format == VK_FORMAT_D16_UNORM_S8_UINT || format == VK_FORMAT_D24_UNORM_S8_UINT || format == VK_FORMAT_D32_SFLOAT_S8_UINT;
    }

    // Helper to get Aspect Flags based on format
    static VkImageAspectFlags GetAspectFlags( VkFormat format )
    {
        if( IsDepthFormat( format ) )
        {
            bool hasStencil =
                ( format == VK_FORMAT_D16_UNORM_S8_UINT || format == VK_FORMAT_D24_UNORM_S8_UINT || format == VK_FORMAT_D32_SFLOAT_S8_UINT );

            return VK_IMAGE_ASPECT_DEPTH_BIT | ( hasStencil ? VK_IMAGE_ASPECT_STENCIL_BIT : 0 );
        }
        return VK_IMAGE_ASPECT_COLOR_BIT;
    }

    Texture::Texture( VmaAllocator allocator, VkDevice device, const VolkDeviceTable* api )
        : m_allocator( allocator )
        , m_device( device )
        , m_api( api )
    {
    }

    Texture::Texture( Texture&& other ) noexcept
        : m_allocator( other.m_allocator )
        , m_device( other.m_device )
        , m_api( other.m_api )
        , m_image( other.m_image )
        , m_view( other.m_view )
        , m_allocation( other.m_allocation )
        , m_extent( other.m_extent )
        , m_format( other.m_format )
        , m_currentLayout( other.m_currentLayout )
        , m_type( other.m_type )
    {
        other.m_image      = VK_NULL_HANDLE;
        other.m_view       = VK_NULL_HANDLE;
        other.m_allocation = VK_NULL_HANDLE;
    }

    Texture& Texture::operator=( Texture&& other ) noexcept
    {
        if( this != &other )
        {
            Destroy();
            m_allocator     = other.m_allocator;
            m_device        = other.m_device;
            m_api           = other.m_api;
            m_image         = other.m_image;
            m_view          = other.m_view;
            m_allocation    = other.m_allocation;
            m_extent        = other.m_extent;
            m_format        = other.m_format;
            m_currentLayout = other.m_currentLayout;
            m_type          = other.m_type;

            other.m_image      = VK_NULL_HANDLE;
            other.m_view       = VK_NULL_HANDLE;
            other.m_allocation = VK_NULL_HANDLE;
        }
        return *this;
    }

    Texture::~Texture()
    {
    }

    void Texture::Destroy()
    {
        if( m_view != VK_NULL_HANDLE )
        {
            m_api->vkDestroyImageView( m_device, m_view, nullptr );
            m_view = VK_NULL_HANDLE;
        }
        if( m_image != VK_NULL_HANDLE )
        {
            vmaDestroyImage( m_allocator, m_image, m_allocation );
            m_image      = VK_NULL_HANDLE;
            m_allocation = VK_NULL_HANDLE;
        }
    }

    Result Texture::Create( const TextureDesc& desc )
    {
        m_extent        = { desc.width, desc.height, desc.depth };
        m_format        = desc.format;
        m_type          = desc.type;
        m_currentLayout = VK_IMAGE_LAYOUT_UNDEFINED;

        VkImageType     imageType = VK_IMAGE_TYPE_2D;
        VkImageViewType viewType  = VK_IMAGE_VIEW_TYPE_2D;

        // Configure types based on the requested texture type
        switch( desc.type )
        {
            case TextureType::Texture1D:
                imageType = VK_IMAGE_TYPE_1D;
                viewType  = VK_IMAGE_VIEW_TYPE_1D;
                break;
            case TextureType::Texture3D:
                imageType = VK_IMAGE_TYPE_3D;
                viewType  = VK_IMAGE_VIEW_TYPE_3D;
                break;
            case TextureType::Texture2D:
            default:
                imageType = VK_IMAGE_TYPE_2D;
                viewType  = VK_IMAGE_VIEW_TYPE_2D;
                break;
        }

        // Map abstract TextureUsage flags to actual Vulkan Image Usage bits
        VkImageUsageFlags usageFlags = 0;
        if( HasFlag( desc.usage, TextureUsage::SAMPLED ) )
            usageFlags |= VK_IMAGE_USAGE_SAMPLED_BIT;
        if( HasFlag( desc.usage, TextureUsage::STORAGE ) )
            usageFlags |= VK_IMAGE_USAGE_STORAGE_BIT;
        if( HasFlag( desc.usage, TextureUsage::RENDER_TARGET ) )
            usageFlags |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
        if( HasFlag( desc.usage, TextureUsage::DEPTH_STENCIL_TARGET ) )
            usageFlags |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
        if( HasFlag( desc.usage, TextureUsage::TRANSFER_SRC ) )
            usageFlags |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        if( HasFlag( desc.usage, TextureUsage::TRANSFER_DST ) )
            usageFlags |= VK_IMAGE_USAGE_TRANSFER_DST_BIT;

        VkImageCreateInfo imageInfo = { VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
        imageInfo.imageType         = imageType;
        imageInfo.extent            = m_extent;
        imageInfo.mipLevels         = 1;
        imageInfo.arrayLayers       = 1;
        imageInfo.format            = m_format;
        imageInfo.tiling            = VK_IMAGE_TILING_OPTIMAL;
        imageInfo.initialLayout     = VK_IMAGE_LAYOUT_UNDEFINED;
        imageInfo.usage             = usageFlags;
        imageInfo.sharingMode       = VK_SHARING_MODE_EXCLUSIVE;
        imageInfo.samples           = VK_SAMPLE_COUNT_1_BIT;

        VmaAllocationCreateInfo allocInfo = {};
        allocInfo.usage                   = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

        // VMA allocation doesn't use our table directly, but it was initialized with it in Device.cpp
        if( vmaCreateImage( m_allocator, &imageInfo, &allocInfo, &m_image, &m_allocation, nullptr ) != VK_SUCCESS )
        {
            DT_CRITICAL( "Failed to create texture image!" );
            return Result::FAIL;
        }

        // Create Image View
        VkImageViewCreateInfo viewInfo = { VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
        viewInfo.image                 = m_image;
        viewInfo.viewType              = viewType;
        viewInfo.format                = m_format;

        viewInfo.subresourceRange.aspectMask     = GetAspectFlags( m_format );
        viewInfo.subresourceRange.baseMipLevel   = 0;
        viewInfo.subresourceRange.levelCount     = 1;
        viewInfo.subresourceRange.baseArrayLayer = 0;
        viewInfo.subresourceRange.layerCount     = 1;

        // Use the function table for creation
        if( m_api->vkCreateImageView( m_device, &viewInfo, nullptr, &m_view ) != VK_SUCCESS )
        {
            DT_CRITICAL( "Failed to create texture image view!" );
            return Result::FAIL;
        }

        return Result::SUCCESS;
    }

    VkDescriptorImageInfo Texture::GetDescriptorInfo( VkSampler sampler, VkImageLayout layout ) const
    {
        VkDescriptorImageInfo info{};
        info.imageLayout = layout;
        info.imageView   = m_view;
        info.sampler     = sampler;
        return info;
    }
} // namespace DigitalTwin