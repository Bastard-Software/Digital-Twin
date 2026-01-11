#include "rhi/RHI.h"

#include "core/Log.h"
#include "rhi/Device.h"
#include <algorithm>
#include <vector>

namespace DigitalTwin
{
    // Static debug callback required by Vulkan C API
    static VKAPI_ATTR VkBool32 VKAPI_CALL DebugCallback( VkDebugUtilsMessageSeverityFlagBitsEXT      messageSeverity,
                                                         VkDebugUtilsMessageTypeFlagsEXT             messageType,
                                                         const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData, void* pUserData );

    RHI::RHI() = default;

    RHI::~RHI()
    {
        Shutdown();
    }

    Result RHI::Initialize( const RHIConfig& config, const std::vector<const char*>& requiredExtensions )
    {
        if( m_initialized )
        {
            DT_WARN( "RHI already initialized!" );
            return Result::SUCCESS;
        }

        // Initialize Volk to load Vulkan entry points
        if( volkInitialize() != VK_SUCCESS )
        {
            DT_CRITICAL( "Failed to initialize volk! Vulkan drivers may be missing." );
            return Result::FAIL;
        }

        m_config = config;

        // Create Vulkan Instance with dependency injected extensions
        Result res = CreateInstance( m_config.enableValidation, requiredExtensions );
        DT_CHECK( res );
        if( res != Result::SUCCESS )
        {
            DT_CRITICAL( "Failed to create Vulkan instance!" );
            return Result::FAIL;
        }

        // Load instance function pointers
        volkLoadInstance( m_instance );

        // Setup debug messenger if validation is enabled
        SetupDebugMessenger();

        // Enumerate GPUs and populate AdapterInfo vector
        EnumeratePhysicalDevices();

        DT_INFO( "Vulkan RHI initialized. Found {} adapter(s).", m_adapters.size() );
        m_initialized = true;
        return Result::SUCCESS;
    }

    void RHI::Shutdown()
    {
        if( !m_initialized )
            return;

        // Destroy Debug Messenger
        if( m_debugMessenger != VK_NULL_HANDLE )
        {
            auto func = ( PFN_vkDestroyDebugUtilsMessengerEXT )vkGetInstanceProcAddr( m_instance, "vkDestroyDebugUtilsMessengerEXT" );
            if( func )
            {
                func( m_instance, m_debugMessenger, nullptr );
            }
            m_debugMessenger = VK_NULL_HANDLE;
        }

        m_adapters.clear();

        // Destroy Instance
        if( m_instance != VK_NULL_HANDLE )
        {
            vkDestroyInstance( m_instance, nullptr );
            m_instance = VK_NULL_HANDLE;
        }

        m_initialized = false;
        DT_INFO( "Vulkan RHI shutdown complete." );
    }

    Result RHI::CreateDevice( uint32_t adapterIndex, Scope<Device>& outDevice )
    {
        if( !m_initialized )
        {
            DT_CRITICAL( "Cannot create device: RHI not initialized." );
            return Result::FAIL;
        }

        if( adapterIndex >= m_adapters.size() )
        {
            DT_CRITICAL( "Invalid adapter index: {}. Only {} adapters available.", adapterIndex, m_adapters.size() );
            return Result::FAIL;
        }

        // Retrieve the physical device handle from our stored info
        VkPhysicalDevice physicalDevice = m_adapters[ adapterIndex ].handle;

        DT_INFO( "Creating Logical Device on adapter: {}", m_adapters[ adapterIndex ].name );

        // Create the Device object using Scope (unique_ptr)
        outDevice = CreateScope<Device>( physicalDevice, m_instance );

        DeviceDesc desc   = {};
        desc.headless     = m_config.headless;
        desc.adapterIndex = adapterIndex;

        if( outDevice->Initialize( desc ) != Result::SUCCESS )
        {
            DT_ERROR( "Failed to initialize Logical Device." );
            outDevice.reset(); // Cleanup partial object
            return Result::FAIL;
        }

        return Result::SUCCESS;
    }

    Result RHI::CreateInstance( bool_t enableValidation, const std::vector<const char*>& requiredExtensions )
    {
        VkApplicationInfo appInfo = { VK_STRUCTURE_TYPE_APPLICATION_INFO };
        appInfo.pApplicationName  = "Digital-Twin";
        appInfo.apiVersion        = VK_API_VERSION_1_3; // Using 1.3 as safe baseline

        std::vector<const char*> extensions = requiredExtensions;
        std::vector<const char*> layers;

        if( enableValidation )
        {
            extensions.push_back( VK_EXT_DEBUG_UTILS_EXTENSION_NAME );
            layers.push_back( "VK_LAYER_KHRONOS_validation" );
        }

// Portability subset extension is required for MoltenVK (macOS)
#ifdef __APPLE__
        extensions.push_back( VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME );
#endif

        VkInstanceCreateInfo createInfo    = { VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
        createInfo.pApplicationInfo        = &appInfo;
        createInfo.enabledExtensionCount   = static_cast<uint32_t>( extensions.size() );
        createInfo.ppEnabledExtensionNames = extensions.data();
        createInfo.enabledLayerCount       = static_cast<uint32_t>( layers.size() );
        createInfo.ppEnabledLayerNames     = layers.data();

#ifdef __APPLE__
        createInfo.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
#endif

        return vkCreateInstance( &createInfo, nullptr, &m_instance ) == VK_SUCCESS ? Result::SUCCESS : Result::FAIL;
    }

    void RHI::SetupDebugMessenger()
    {
        if( !m_config.enableValidation )
            return;

        VkDebugUtilsMessengerCreateInfoEXT createInfo{ VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT };
        createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        createInfo.messageType     = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT;
        createInfo.pfnUserCallback = DebugCallback;

        auto func = ( PFN_vkCreateDebugUtilsMessengerEXT )vkGetInstanceProcAddr( m_instance, "vkCreateDebugUtilsMessengerEXT" );
        if( func )
        {
            func( m_instance, &createInfo, nullptr, &m_debugMessenger );
        }
    }

    void RHI::EnumeratePhysicalDevices()
    {
        uint32_t count = 0;
        vkEnumeratePhysicalDevices( m_instance, &count, nullptr );
        if( count == 0 )
        {
            DT_CRITICAL( "No Vulkan-capable GPUs found!" );
            return;
        }

        std::vector<VkPhysicalDevice> physicalDevices( count );
        vkEnumeratePhysicalDevices( m_instance, &count, physicalDevices.data() );

        m_adapters.clear();
        m_adapters.reserve( count );

        DT_INFO( "Enumerating Adapters:" );

        for( VkPhysicalDevice pd: physicalDevices )
        {
            VkPhysicalDeviceProperties props;
            vkGetPhysicalDeviceProperties( pd, &props );

            // Get Memory Properties to find VRAM size
            VkPhysicalDeviceMemoryProperties memProps;
            vkGetPhysicalDeviceMemoryProperties( pd, &memProps );

            uint64_t vramBytes = 0;
            for( uint32_t i = 0; i < memProps.memoryHeapCount; ++i )
            {
                if( memProps.memoryHeaps[ i ].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT )
                {
                    vramBytes += memProps.memoryHeaps[ i ].size;
                }
            }

            AdapterInfo info;
            info.handle           = pd;
            info.name             = props.deviceName;
            info.vendorID         = props.vendorID;
            info.deviceID         = props.deviceID;
            info.type             = props.deviceType;
            info.deviceMemorySize = vramBytes;

            m_adapters.push_back( info );

            DT_INFO( "  - [{}] {}: Type: {}, VRAM: {} MB", m_adapters.size() - 1, info.name, ( int )info.type,
                     info.deviceMemorySize / ( 1024 * 1024 ) );
        }
    }

    VKAPI_ATTR VkBool32 VKAPI_CALL DebugCallback( VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, VkDebugUtilsMessageTypeFlagsEXT messageType,
                                                  const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData, void* pUserData )
    {
        // Simple logging of validation messages
        if( messageSeverity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT )
        {
            DT_ERROR( "Validation: {0}", pCallbackData->pMessage );
        }
        else if( messageSeverity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT )
        {
            DT_WARN( "Validation: {0}", pCallbackData->pMessage );
        }
        // Info messages can be noisy, so we often skip them or log as TRACE

        return VK_FALSE;
    }

} // namespace DigitalTwin