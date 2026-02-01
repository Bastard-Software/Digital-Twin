#include "rhi/Swapchain.h"

#include "core/Log.h"
#include "platform/Window.h"
#include "rhi/Device.h"
#include <algorithm>

namespace DigitalTwin
{
    Swapchain::Swapchain( Device* device )
        : m_device( device )
    {
    }

    Swapchain::~Swapchain()
    {
        Destroy();
    }

    Result Swapchain::Create( Window* window, bool vsync )
    {
        if( !window )
            return Result::INVALID_ARGS;

        m_window = window;
        m_vsync  = vsync;

        // 1. Create Surface (Once)
        m_surface = m_window->CreateSurface( m_device->GetInstance() );
        if( m_surface == VK_NULL_HANDLE )
        {
            return Result::FAIL;
        }

        // 2. Create Swapchain and Images
        return CreateInternal( m_window->GetWidth(), m_window->GetHeight(), m_vsync );
    }

    void Swapchain::Destroy()
    {
        CleanupInternal();

        // Destroy Semaphores
        for( auto sem: m_imageAvailableSemaphores )
        {
            m_device->GetAPI().vkDestroySemaphore( m_device->GetHandle(), sem, nullptr );
        }
        m_imageAvailableSemaphores.clear();

        // Destroy Surface
        if( m_surface != VK_NULL_HANDLE )
        {
            vkDestroySurfaceKHR( m_device->GetInstance(), m_surface, nullptr );
            m_surface = VK_NULL_HANDLE;
        }

        m_window = nullptr;
    }

    Result Swapchain::Recreate()
    {
        if( !m_window )
            return Result::FAIL;

        uint32_t width = 0, height = 0;
        m_window->GetFramebufferSize( width, height );

        if( width == 0 || height == 0 )
        {
            return Result::SUCCESS; // Technically not a failure, just nothing to do
        }

        return CreateInternal( width, height, m_vsync );
    }

    void Swapchain::CleanupInternal()
    {
        m_device->WaitIdle(); // Safety wait

        for( auto& tex: m_textures )
            tex.Destroy();
        m_textures.clear(); // Clears texture wrappers (views destroyed)

        if( m_swapchain != VK_NULL_HANDLE )
        {
            m_device->GetAPI().vkDestroySwapchainKHR( m_device->GetHandle(), m_swapchain, nullptr );
            m_swapchain = VK_NULL_HANDLE;
        }
    }

    Result Swapchain::CreateInternal( uint32_t width, uint32_t height, bool vsync )
    {
        CleanupInternal(); // Cleanup old swapchain if exists (for resize)

        const auto&      api    = m_device->GetAPI();
        VkPhysicalDevice pd     = m_device->GetPhysicalDevice();
        VkDevice         device = m_device->GetHandle();

        // --- Surface Capabilities ---
        VkSurfaceCapabilitiesKHR caps;
        vkGetPhysicalDeviceSurfaceCapabilitiesKHR( pd, m_surface, &caps );

        // --- Format ---
        uint32_t formatCount;
        vkGetPhysicalDeviceSurfaceFormatsKHR( pd, m_surface, &formatCount, nullptr );
        std::vector<VkSurfaceFormatKHR> formats( formatCount );
        vkGetPhysicalDeviceSurfaceFormatsKHR( pd, m_surface, &formatCount, formats.data() );

        m_format = formats[ 0 ];
        for( const auto& f: formats )
        {
            if( f.format == VK_FORMAT_B8G8R8A8_SRGB && f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR )
            {
                m_format = f;
                break;
            }
        }

        // --- Present Mode ---
        uint32_t presentModeCount;
        vkGetPhysicalDeviceSurfacePresentModesKHR( pd, m_surface, &presentModeCount, nullptr );
        std::vector<VkPresentModeKHR> presentModes( presentModeCount );
        vkGetPhysicalDeviceSurfacePresentModesKHR( pd, m_surface, &presentModeCount, presentModes.data() );

        m_presentMode = VK_PRESENT_MODE_FIFO_KHR; // VSync guaranteed
        if( !vsync )
        {
            for( const auto& mode: presentModes )
            {
                if( mode == VK_PRESENT_MODE_MAILBOX_KHR )
                {
                    m_presentMode = mode;
                    break;
                }
                if( mode == VK_PRESENT_MODE_IMMEDIATE_KHR )
                {
                    m_presentMode = mode;
                }
            }
        }

        // --- Extent ---
        if( caps.currentExtent.width != UINT32_MAX )
        {
            m_extent = caps.currentExtent;
        }
        else
        {
            m_extent        = { width, height };
            m_extent.width  = std::clamp( m_extent.width, caps.minImageExtent.width, caps.maxImageExtent.width );
            m_extent.height = std::clamp( m_extent.height, caps.minImageExtent.height, caps.maxImageExtent.height );
        }

        // --- Image Count ---
        uint32_t imageCount = caps.minImageCount + 1;
        if( caps.maxImageCount > 0 && imageCount > caps.maxImageCount )
            imageCount = caps.maxImageCount;

        // --- Creation ---
        VkSwapchainCreateInfoKHR createInfo = { VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR };
        createInfo.surface                  = m_surface;
        createInfo.minImageCount            = imageCount;
        createInfo.imageFormat              = m_format.format;
        createInfo.imageColorSpace          = m_format.colorSpace;
        createInfo.imageExtent              = m_extent;
        createInfo.imageArrayLayers         = 1;
        createInfo.imageUsage               = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        createInfo.imageSharingMode         = VK_SHARING_MODE_EXCLUSIVE;
        createInfo.preTransform             = caps.currentTransform;
        createInfo.compositeAlpha           = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        createInfo.presentMode              = m_presentMode;
        createInfo.clipped                  = VK_TRUE;

        if( api.vkCreateSwapchainKHR( device, &createInfo, nullptr, &m_swapchain ) != VK_SUCCESS )
        {
            DT_ERROR( "Failed to create Swapchain!" );
            return Result::FAIL;
        }

        // --- Retrieve Images ---
        api.vkGetSwapchainImagesKHR( device, m_swapchain, &imageCount, nullptr );
        std::vector<VkImage> images( imageCount );
        api.vkGetSwapchainImagesKHR( device, m_swapchain, &imageCount, images.data() );

        m_textures.reserve( imageCount );
        VkExtent3D ext3D = { m_extent.width, m_extent.height, 1 };
        for( auto img: images )
        {
            m_textures.emplace_back( device, &api, img, m_format.format, ext3D );
        }

        // --- Sync Objects (Lazy Init) ---
        if( m_imageAvailableSemaphores.empty() )
        {
            m_imageAvailableSemaphores.resize( MAX_FRAMES_IN_FLIGHT );
            VkSemaphoreCreateInfo semInfo = { VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
            for( int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i )
                api.vkCreateSemaphore( device, &semInfo, nullptr, &m_imageAvailableSemaphores[ i ] );
        }

        DT_INFO( "Swapchain (Re)Created. Size: {}x{}", m_extent.width, m_extent.height );
        return Result::SUCCESS;
    }

    Result Swapchain::AcquireNextImage( uint32_t* outImageIndex, VkSemaphore* outSignalSemaphore )
    {
        VkSemaphore sem = m_imageAvailableSemaphores[ m_currentFrame ];
        VkResult res = m_device->GetAPI().vkAcquireNextImageKHR( m_device->GetHandle(), m_swapchain, 2000000000, sem, VK_NULL_HANDLE, outImageIndex );

        if( res == VK_ERROR_OUT_OF_DATE_KHR )
            return Result::RECREATE_SWAPCHAIN;
        if( res != VK_SUCCESS && res != VK_SUBOPTIMAL_KHR )
            return Result::FAIL;

        *outSignalSemaphore = sem;
        m_currentFrame      = ( m_currentFrame + 1 ) % MAX_FRAMES_IN_FLIGHT;
        return Result::SUCCESS;
    }

    Result Swapchain::Present( uint32_t imageIndex, VkSemaphore waitSemaphore )
    {
        VkPresentInfoKHR info   = { VK_STRUCTURE_TYPE_PRESENT_INFO_KHR };
        info.waitSemaphoreCount = 1;
        info.pWaitSemaphores    = &waitSemaphore;
        info.swapchainCount     = 1;
        info.pSwapchains        = &m_swapchain;
        info.pImageIndices      = &imageIndex;

        VkResult res = m_device->GetAPI().vkQueuePresentKHR( m_device->GetGraphicsQueue()->GetHandle(), &info );
        if( res == VK_ERROR_OUT_OF_DATE_KHR || res == VK_SUBOPTIMAL_KHR )
            return Result::RECREATE_SWAPCHAIN;
        if( res != VK_SUCCESS )
            return Result::FAIL;

        return Result::SUCCESS;
    }
} // namespace DigitalTwin