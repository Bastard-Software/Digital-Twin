#include "rhi/RHI.hpp"

#include <algorithm>

namespace DigitalTwin
{

    VkInstance                    RHI::s_instance        = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT      RHI::s_debugMessenger  = VK_NULL_HANDLE;
    std::vector<VkPhysicalDevice> RHI::s_physicalDevices = {};
    bool_t                        RHI::s_initialized     = false;
    RHIConfig                     RHI::s_config          = {};

    VKAPI_ATTR VkBool32 VKAPI_CALL DebugCallback( VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, VkDebugUtilsMessageTypeFlagsEXT messageType,
                                                  const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData, void* pUserData );

    Result RHI::Init( RHIConfig config )
    {
        if( s_initialized )
        {
            DT_CORE_WARN( "RHI already initialized!" );
            return Result::SUCCESS;
        }

        if( volkInitialize() != VK_SUCCESS )
        {
            DT_CORE_CRITICAL( "Failed to initialize volk!" );
            return Result::FAIL;
        }

        s_config   = config;
        Result res = CreateInstance( s_config.enableValidation );
        DT_CHECK( res );
        if( res != Result::SUCCESS )
        {
            DT_CORE_CRITICAL( "Failed to create Vulkan instance!" );
            return Result::FAIL;
        }
        volkLoadInstance( s_instance );

        EnumeratePhysicalDevices();

        DT_CORE_INFO( "Vulkan RHI initialized successfully with {} physical devices found.", s_physicalDevices.size() );
        s_initialized = true;
        return Result::SUCCESS;
    }

    void RHI::Shutdown()
    {
        if( !s_initialized )
        {
            DT_CORE_WARN( "RHI not initialized!" );
            return;
        }

        if( s_debugMessenger != VK_NULL_HANDLE )
        {
            auto func = ( PFN_vkDestroyDebugUtilsMessengerEXT )vkGetInstanceProcAddr( s_instance, "vkDestroyDebugUtilsMessengerEXT" );
            if( func != nullptr )
            {
                func( s_instance, s_debugMessenger, nullptr );
                s_debugMessenger = VK_NULL_HANDLE;
            }
        }
        s_physicalDevices.clear();

        if( s_instance != VK_NULL_HANDLE )
        {
            vkDestroyInstance( s_instance, nullptr );
            s_instance = VK_NULL_HANDLE;
        }
        volkFinalize();
        s_initialized = false;
        DT_CORE_INFO( "Vulkan RHI shutdown complete." );
    }

    Ref<Device> RHI::CreateDevice( uint32_t adapterIndex )
    {
        if( !s_initialized )
        {
            DT_CORE_CRITICAL( "RHI not initialized! Cannot create device." );
            return nullptr;
        }

        if( adapterIndex >= s_physicalDevices.size() )
        {
            DT_CORE_CRITICAL( "Invalid adapter index: {}. Only {} physical devices available.", adapterIndex, s_physicalDevices.size() );
            return nullptr;
        }

        auto device = CreateRef<Device>( s_physicalDevices[ adapterIndex ] );

        DeviceDesc desc = {};
        desc.headless   = s_config.headless;
        if( device->Init( desc ) != Result::SUCCESS )
        {
            DT_CORE_CRITICAL( "Failed to initialize device for adapter index: {}.", adapterIndex );
            return nullptr;
        }

        return device;
    }

    void RHI::DestroyDevice( Ref<Device> device )
    {
        if( device )
        {
            device->Shutdown();
        }
    }

    Result RHI::CreateInstance( bool_t enableValidation )
    {
        VkApplicationInfo appInfo = {};
        appInfo.sType             = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.pApplicationName  = "Digital Twin";
        appInfo.apiVersion        = VK_API_VERSION_1_4;

        std::vector<const char*> extensions;
        std::vector<const char*> layers;
        if( enableValidation )
        {
            extensions.push_back( VK_EXT_DEBUG_UTILS_EXTENSION_NAME );
            layers.push_back( "VK_LAYER_KHRONOS_validation" );
        }

        VkInstanceCreateInfo createInfo    = {};
        createInfo.sType                   = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        createInfo.pApplicationInfo        = &appInfo;
        createInfo.enabledExtensionCount   = static_cast<uint32_t>( extensions.size() );
        createInfo.ppEnabledExtensionNames = extensions.data();
        createInfo.enabledLayerCount       = static_cast<uint32_t>( layers.size() );
        createInfo.ppEnabledLayerNames     = layers.data();

        return vkCreateInstance( &createInfo, nullptr, &s_instance ) == VK_SUCCESS ? Result::SUCCESS : Result::FAIL;
    }

    void RHI::SetupDebugMessenger()
    {
        VkDebugUtilsMessengerCreateInfoEXT createInfo{};
        createInfo.sType           = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
        createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        createInfo.messageType     = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT;
        createInfo.pfnUserCallback = DebugCallback;

        auto vkCreateDebugUtilsMessengerEXT =
            ( PFN_vkCreateDebugUtilsMessengerEXT )vkGetInstanceProcAddr( s_instance, "vkCreateDebugUtilsMessengerEXT" );
        if( vkCreateDebugUtilsMessengerEXT )
        {
            vkCreateDebugUtilsMessengerEXT( s_instance, &createInfo, nullptr, &s_debugMessenger );
        }
    }

    void RHI::EnumeratePhysicalDevices()
    {
        uint32_t count = 0;
        vkEnumeratePhysicalDevices( s_instance, &count, nullptr );
        if( count == 0 )
        {
            DT_CORE_CRITICAL( "No Vulkan GPUs found!" );
            return;
        }

        s_physicalDevices.resize( count );
        vkEnumeratePhysicalDevices( s_instance, &count, s_physicalDevices.data() );

        DT_CORE_INFO( "Found {0} physical device(s):", count );
        for( const auto& device: s_physicalDevices )
        {
            VkPhysicalDeviceProperties props;
            vkGetPhysicalDeviceProperties( device, &props );
            DT_CORE_INFO( "  - {0} (API: {1}.{2})", props.deviceName, VK_API_VERSION_MAJOR( props.apiVersion ),
                          VK_API_VERSION_MINOR( props.apiVersion ) );
        }
    }

    VKAPI_ATTR VkBool32 VKAPI_CALL DebugCallback( VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, VkDebugUtilsMessageTypeFlagsEXT messageType,
                                                  const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData, void* pUserData )
    {
        ( void )messageType;
        ( void )pUserData;
        if( messageSeverity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT )
        {
            DT_CORE_ERROR( "Validation: {0}", pCallbackData->pMessage );
        }
        else if( messageSeverity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT )
        {
            DT_CORE_WARN( "Validation: {0}", pCallbackData->pMessage );
        }
        else
        {
            DT_CORE_INFO( "Validation: {0}", pCallbackData->pMessage );
        }

        return VK_FALSE;
    }

} // namespace DigitalTwin