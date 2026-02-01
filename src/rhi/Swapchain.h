#pragma once
#include "rhi/RHITypes.h"

#include "rhi/Texture.h"
#include <vector>
#include <volk.h>

namespace DigitalTwin
{
    class Device;
    class Window;

    class Swapchain
    {
    public:
        Swapchain( Device* device );
        ~Swapchain();

        /**
         * @brief Creates the swapchain and the surface associated with the window.
         * @param window Pointer to the OS window.
         * @param vsync Enable vertical sync.
         */
        Result Create( Window* window, bool vsync = true );

        /**
         * @brief Destroys the swapchain, images views, sync objects, and the surface.
         */
        void Destroy();

        /**
         * @brief Recreates the swapchain (e.g. on resize).
         * Reuses the existing Surface and Window pointer stored internally.
         */
        Result Recreate();

        Result AcquireNextImage( uint32_t* outImageIndex, VkSemaphore* outSignalSemaphore );
        Result Present( uint32_t imageIndex, VkSemaphore waitSemaphore );

        Texture*   GetTexture( uint32_t index ) { return &m_textures[ index ]; }
        uint32_t   GetImageCount() const { return ( uint32_t )m_textures.size(); }
        VkFormat   GetFormat() const { return m_format.format; }
        VkExtent2D GetExtent() const { return m_extent; }

    private:
        Result CreateInternal( uint32_t width, uint32_t height, bool vsync );
        void   CleanupInternal(); // Cleans up swapchain resources but keeps Surface

    private:
        Device* m_device;
        Window* m_window = nullptr; // Kept for recreation

        VkSurfaceKHR   m_surface   = VK_NULL_HANDLE;
        VkSwapchainKHR m_swapchain = VK_NULL_HANDLE;

        VkSurfaceFormatKHR m_format;
        VkPresentModeKHR   m_presentMode;
        VkExtent2D         m_extent;
        bool               m_vsync = true;

        std::vector<Texture>     m_textures;
        std::vector<VkSemaphore> m_imageAvailableSemaphores;

        uint32_t  m_currentFrame       = 0;
        const int MAX_FRAMES_IN_FLIGHT = 2;
    };
} // namespace DigitalTwin