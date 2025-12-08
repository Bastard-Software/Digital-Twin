#include "rhi/Swapchain.hpp"

#include <GLFW/glfw3.h>
#include <algorithm>
#include <limits>

namespace DigitalTwin
{
    Swapchain::Swapchain( VkDevice device, VkPhysicalDevice physicalDevice, VkInstance instance, VkQueue presentQueue,
                          uint32_t presentQueueFamilyIndex, const VolkDeviceTable* api, const SwapchainDesc& desc )
        : m_deviceHandle( device )
        , m_physicalDevice( physicalDevice )
        , m_instance( instance )
        , m_presentQueue( presentQueue )
        , m_presentQueueFamilyIndex( presentQueueFamilyIndex )
        , m_api( api )
        , m_desc( desc )
    {
        DT_CORE_ASSERT( m_api, "API table is null!" );
        CreateSurface();
        CreateSwapchain();
        CreateImageViews();
        CreateSyncObjects();
    }

    Swapchain::~Swapchain()
    {
        // Wait for device idle before destruction
        if( m_deviceHandle )
        {
            vkDeviceWaitIdle( m_deviceHandle );
        }

        CleanupSwapchain();

        for( auto sem: m_imageAvailableSemaphores )
        {
            if( sem )
                m_api->vkDestroySemaphore( m_deviceHandle, sem, nullptr );
        }

        if( m_surface != VK_NULL_HANDLE )
        {
            vkDestroySurfaceKHR( m_instance, m_surface, nullptr );
        }
    }

    void Swapchain::CreateSurface()
    {
        // GLFW requires the instance to create a surface
        if( glfwCreateWindowSurface( m_instance, ( GLFWwindow* )m_desc.windowHandle, nullptr, &m_surface ) != VK_SUCCESS )
        {
            DT_CORE_CRITICAL( "Failed to create window surface!" );
        }
    }

    void Swapchain::CleanupSwapchain()
    {
        for( auto view: m_imageViews )
        {
            m_api->vkDestroyImageView( m_deviceHandle, view, nullptr );
        }
        m_imageViews.clear();

        if( m_swapchain != VK_NULL_HANDLE )
        {
            m_api->vkDestroySwapchainKHR( m_deviceHandle, m_swapchain, nullptr );
            m_swapchain = VK_NULL_HANDLE;
        }
    }

    void Swapchain::Resize( uint32_t width, uint32_t height )
    {
        m_desc.width  = width;
        m_desc.height = height;

        if( width == 0 || height == 0 )
            return;

        vkDeviceWaitIdle( m_deviceHandle );
        CleanupSwapchain();
        CreateSwapchain();
        CreateImageViews();
    }

    void Swapchain::CreateSwapchain()
    {
        // 1. Capabilities
        VkSurfaceCapabilitiesKHR capabilities;
        vkGetPhysicalDeviceSurfaceCapabilitiesKHR( m_physicalDevice, m_surface, &capabilities );

        // 2. Formats
        uint32_t formatCount;
        vkGetPhysicalDeviceSurfaceFormatsKHR( m_physicalDevice, m_surface, &formatCount, nullptr );
        std::vector<VkSurfaceFormatKHR> formats( formatCount );
        vkGetPhysicalDeviceSurfaceFormatsKHR( m_physicalDevice, m_surface, &formatCount, formats.data() );

        // 3. Present Modes
        uint32_t presentModeCount;
        vkGetPhysicalDeviceSurfacePresentModesKHR( m_physicalDevice, m_surface, &presentModeCount, nullptr );
        std::vector<VkPresentModeKHR> presentModes( presentModeCount );
        vkGetPhysicalDeviceSurfacePresentModesKHR( m_physicalDevice, m_surface, &presentModeCount, presentModes.data() );

        // Select Format (Prefer SRGB/UNORM)
        m_surfaceFormat = formats[ 0 ];
        for( const auto& fmt: formats )
        {
            if( fmt.format == VK_FORMAT_B8G8R8A8_UNORM && fmt.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR )
            {
                m_surfaceFormat = fmt;
                break;
            }
        }

        // Select Present Mode (Mailbox if available and no vsync, else FIFO)
        m_presentMode = VK_PRESENT_MODE_FIFO_KHR;
        if( !m_desc.vsync )
        {
            for( const auto& mode: presentModes )
            {
                if( mode == VK_PRESENT_MODE_MAILBOX_KHR )
                {
                    m_presentMode = mode;
                    break;
                }
            }
        }

        // Select Extent
        if( capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max() )
        {
            m_extent = capabilities.currentExtent;
        }
        else
        {
            VkExtent2D actual = { m_desc.width, m_desc.height };
            actual.width      = std::clamp( actual.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width );
            actual.height     = std::clamp( actual.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height );
            m_extent          = actual;
        }

        uint32_t imageCount = capabilities.minImageCount + 1;
        if( capabilities.maxImageCount > 0 && imageCount > capabilities.maxImageCount )
        {
            imageCount = capabilities.maxImageCount;
        }

        VkSwapchainCreateInfoKHR createInfo = { VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR };
        createInfo.surface                  = m_surface;
        createInfo.minImageCount            = imageCount;
        createInfo.imageFormat              = m_surfaceFormat.format;
        createInfo.imageColorSpace          = m_surfaceFormat.colorSpace;
        createInfo.imageExtent              = m_extent;
        createInfo.imageArrayLayers         = 1;
        createInfo.imageUsage               = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;

        // Concurrent sharing is complex, sticking to Exclusive for now (assumes present queue = graphics queue)
        createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        createInfo.preTransform     = capabilities.currentTransform;
        createInfo.compositeAlpha   = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        createInfo.presentMode      = m_presentMode;
        createInfo.clipped          = VK_TRUE;
        createInfo.oldSwapchain     = VK_NULL_HANDLE;

        if( m_api->vkCreateSwapchainKHR( m_deviceHandle, &createInfo, nullptr, &m_swapchain ) != VK_SUCCESS )
        {
            DT_CORE_CRITICAL( "Failed to create swapchain!" );
        }

        // Retrieve images
        vkGetSwapchainImagesKHR( m_deviceHandle, m_swapchain, &imageCount, nullptr );
        m_images.resize( imageCount );
        vkGetSwapchainImagesKHR( m_deviceHandle, m_swapchain, &imageCount, m_images.data() );
    }

    void Swapchain::CreateImageViews()
    {
        m_imageViews.resize( m_images.size() );
        for( size_t i = 0; i < m_images.size(); i++ )
        {
            VkImageViewCreateInfo viewInfo           = { VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
            viewInfo.image                           = m_images[ i ];
            viewInfo.viewType                        = VK_IMAGE_VIEW_TYPE_2D;
            viewInfo.format                          = m_surfaceFormat.format;
            viewInfo.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
            viewInfo.subresourceRange.baseMipLevel   = 0;
            viewInfo.subresourceRange.levelCount     = 1;
            viewInfo.subresourceRange.baseArrayLayer = 0;
            viewInfo.subresourceRange.layerCount     = 1;

            if( m_api->vkCreateImageView( m_deviceHandle, &viewInfo, nullptr, &m_imageViews[ i ] ) != VK_SUCCESS )
            {
                DT_CORE_CRITICAL( "Failed to create swapchain image view!" );
            }
        }
    }

    void Swapchain::CreateSyncObjects()
    {
        m_imageAvailableSemaphores.resize( MAX_FRAMES_IN_FLIGHT );
        VkSemaphoreCreateInfo semaphoreInfo = { VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };

        for( size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++ )
        {
            if( m_api->vkCreateSemaphore( m_deviceHandle, &semaphoreInfo, nullptr, &m_imageAvailableSemaphores[ i ] ) != VK_SUCCESS )
            {
                DT_CORE_CRITICAL( "Failed to create swapchain semaphore!" );
            }
        }
    }

    VkSemaphore Swapchain::AcquireNextImage( uint32_t& outImageIndex )
    {
        VkResult result = m_api->vkAcquireNextImageKHR( m_deviceHandle, m_swapchain, UINT64_MAX, m_imageAvailableSemaphores[ m_currentFrame ],
                                                        VK_NULL_HANDLE, &outImageIndex );

        if( result == VK_ERROR_OUT_OF_DATE_KHR )
        {
            Resize( m_desc.width, m_desc.height );
            return VK_NULL_HANDLE;
        }
        else if( result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR )
        {
            DT_CORE_ERROR( "Failed to acquire next image!" );
            return VK_NULL_HANDLE;
        }

        m_currentImageIndex = outImageIndex;
        return m_imageAvailableSemaphores[ m_currentFrame ];
    }

    void Swapchain::Present( VkSemaphore waitSemaphore )
    {
        VkPresentInfoKHR presentInfo   = { VK_STRUCTURE_TYPE_PRESENT_INFO_KHR };
        presentInfo.waitSemaphoreCount = 1;
        presentInfo.pWaitSemaphores    = &waitSemaphore;

        VkSwapchainKHR swapchains[] = { m_swapchain };
        presentInfo.swapchainCount  = 1;
        presentInfo.pSwapchains     = swapchains;
        presentInfo.pImageIndices   = &m_currentImageIndex;

        VkResult result = m_api->vkQueuePresentKHR( m_presentQueue, &presentInfo );

        if( result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR )
        {
            Resize( m_desc.width, m_desc.height );
        }
        else if( result != VK_SUCCESS )
        {
            DT_CORE_ERROR( "Failed to present image!" );
        }

        m_currentFrame = ( m_currentFrame + 1 ) % MAX_FRAMES_IN_FLIGHT;
    }
} // namespace DigitalTwin