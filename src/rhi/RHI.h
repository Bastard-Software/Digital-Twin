#pragma once

#include "core/Core.h"
#include "rhi/RHITypes.h"
#include <string>
#include <vector>

namespace DigitalTwin
{
    /**
     * @brief Render Hardware Interface (Vulkan Instance Wrapper).
     * Manages the Vulkan Instance, physical device enumeration, and Logical Device creation.
     * Owned uniquely by the Engine instance.
     */
    class DT_API RHI
    {
    public:
        RHI();
        ~RHI();

        /**
         * @brief Initializes the Vulkan instance and enumerates physical devices.
         * @param config RHI Configuration (validation layers, etc.)
         * @param requiredExtensions List of instance extensions required by the platform (e.g. Surface extensions).
         * @return Result::SUCCESS on success.
         */
        Result Initialize( const RHIConfig& config, const std::vector<const char*>& requiredExtensions = {} );

        /**
         * @brief Destroys the Vulkan instance.
         */
        void Shutdown();

/**
         * @brief Creates a Logical Device on the specified adapter.
         * @param adapterIndex Index into the adapter list (GetAdapters).
         * @param outDevice [Out] Unique pointer to the created Device.
         * @return Result::SUCCESS if creation and initialization were successful.
         */
        Result CreateDevice( uint32_t adapterIndex, Scope<Device>& outDevice );

        // --- Getters ---

        bool_t     IsInitialized() const { return m_initialized; }
        VkInstance GetInstance() const { return m_instance; }

        /**
         * @brief Returns a list of all available GPUs on the system.
         */
        const std::vector<AdapterInfo>& GetAdapters() const { return m_adapters; }

    private:
        Result CreateInstance( bool_t enableValidation, const std::vector<const char*>& requiredExtensions );
        void   SetupDebugMessenger();
        void   EnumeratePhysicalDevices();

    private:
        VkInstance               m_instance       = VK_NULL_HANDLE;
        VkDebugUtilsMessengerEXT m_debugMessenger = VK_NULL_HANDLE;

        // List of enumerated physical devices with metadata
        std::vector<AdapterInfo> m_adapters;

        bool_t    m_initialized = false;
        RHIConfig m_config      = {};
    };
} // namespace DigitalTwin