#pragma once
#include "core/Base.hpp"
#include <vector>
#include <volk.h>

namespace DigitalTwin
{
    struct SwapchainDesc
    {
        void*    windowHandle = nullptr; // Raw GLFWwindow pointer
        uint32_t width        = 0;
        uint32_t height       = 0;
        bool     vsync        = true;
    };

    class Swapchain
    {
    public:
        /**
         * @brief Constructor. Creates surface and swapchain.
         * @param device Logical device handle.
         * @param physicalDevice Physical device handle (for capabilities query).
         * @param instance Vulkan instance handle (for surface creation).
         * @param presentQueue Queue handle used for presentation.
         * @param presentQueueFamilyIndex Family index of the present queue.
         * @param api Device function table.
         * @param desc Configuration descriptor.
         */
        Swapchain( VkDevice device, VkPhysicalDevice physicalDevice, VkInstance instance, VkQueue presentQueue, uint32_t presentQueueFamilyIndex,
                   const VolkDeviceTable* api, const SwapchainDesc& desc );

        ~Swapchain();

        /**
         * @brief Recreates the swapchain (e.g. on window resize).
         * Should be called when AcquireNextImage returns VK_ERROR_OUT_OF_DATE_KHR.
         * @param width New width.
         * @param height New height.
         */
        void Resize( uint32_t width, uint32_t height );

        /**
         * @brief Requests the next available image index from the presentation engine.
         * @param outImageIndex [Out] Index of the swapchain image to render into.
         * @return Semaphore that will be signaled when the image is available.
         * Returns VK_NULL_HANDLE if swapchain needs resizing.
         */
        VkSemaphore AcquireNextImage( uint32_t& outImageIndex );

        /**
         * @brief Presents the rendered image to the screen.
         * @param waitSemaphore The semaphore signaled by the render queue when drawing is finished.
         */
        void Present( VkSemaphore waitSemaphore );

        // --- Getters ---
        VkFormat    GetFormat() const { return m_surfaceFormat.format; }
        VkExtent2D  GetExtent() const { return m_extent; }
        VkImageView GetImageView( uint32_t index ) const { return m_imageViews[ index ]; }
        VkImage     GetImage( uint32_t index ) const { return m_images[ index ]; }
        size_t      GetImageCount() const { return m_images.size(); }

    public:
        // Disable copying
        Swapchain( const Swapchain& )            = delete;
        Swapchain& operator=( const Swapchain& ) = delete;

    private:
        void CreateSurface();
        void CreateSwapchain();
        void CreateImageViews();
        void CreateSyncObjects();
        void CleanupSwapchain(); // Destroys images/views/swapchain but keeps surface

    private:
        VkDevice               m_deviceHandle;
        VkPhysicalDevice       m_physicalDevice;
        VkInstance             m_instance;
        VkQueue                m_presentQueue;
        uint32_t               m_presentQueueFamilyIndex;
        const VolkDeviceTable* m_api;

        SwapchainDesc m_desc;

        VkSurfaceKHR   m_surface   = VK_NULL_HANDLE;
        VkSwapchainKHR m_swapchain = VK_NULL_HANDLE;

        VkSurfaceFormatKHR m_surfaceFormat = {};
        VkPresentModeKHR   m_presentMode   = VK_PRESENT_MODE_FIFO_KHR;
        VkExtent2D         m_extent        = { 0, 0 };

        std::vector<VkImage>     m_images;
        std::vector<VkImageView> m_imageViews;

        // Synchronization
        static constexpr int     MAX_FRAMES_IN_FLIGHT = 2;
        std::vector<VkSemaphore> m_imageAvailableSemaphores;

        uint32_t m_currentFrame      = 0;
        uint32_t m_currentImageIndex = 0;
    };
} // namespace DigitalTwin