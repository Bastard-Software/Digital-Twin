#pragma once

#include "core/Base.hpp"
#include "rhi/Device.hpp"
#include <volk.h>

namespace DigitalTwin
{
    struct RHIConfig
    {
        bool_t enableValidation = false;
        bool_t headless         = false;
    };

    class RHI
    {
    public:
        static Result Init( RHIConfig config );
        static void   Shutdown();

        static Ref<Device> CreateDevice( uint32_t adapterIndex );
        static void        DestroyDevice( Ref<Device> device );

        static bool_t IsInitialized() { return s_initialized; }
        static VkInstance GetInstance() { return s_instance; }
        static uint32_t   GetAdapterCount() { return static_cast<uint32_t>( s_physicalDevices.size() ); }

    private:
        static Result CreateInstance( bool_t enableValidation );
        static void   SetupDebugMessenger();
        static void   EnumeratePhysicalDevices();

    private:
        static VkInstance                    s_instance;
        static VkDebugUtilsMessengerEXT      s_debugMessenger;
        static std::vector<VkPhysicalDevice> s_physicalDevices;
        static bool_t                        s_initialized;
        static RHIConfig                     s_config;
    };
} // namespace DigitalTwin