#pragma once

#include "DigitalTwin.h"
#include "core/Core.h"
#include <string>
#include <volk.h>

namespace DigitalTwin
{

    class RHI;
    class Device;
    class Queue;

    /**
     * @brief Configuration for RHI initialization.
     */
    struct RHIConfig
    {
        bool_t enableValidation = false;
        bool_t headless         = false;
    };

    /**
     * @brief Detailed information about a physical GPU.
     */
    struct AdapterInfo
    {
        VkPhysicalDevice     handle = VK_NULL_HANDLE;
        std::string          name;
        uint32_t             vendorID         = 0;
        uint32_t             deviceID         = 0;
        uint64_t             deviceMemorySize = 0;
        VkPhysicalDeviceType type             = VK_PHYSICAL_DEVICE_TYPE_OTHER;

        // Helper to check if it's a discrete GPU (usually preferred)
        bool IsDiscrete() const { return type == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU; }
    };

    /**
     * @brief Types of command queues available on the device.
     */
    enum class QueueType
    {
        GRAPHICS,
        COMPUTE,
        TRANSFER
    };

    /**
     * @brief Description for creating a Device.
     */
    struct DeviceDesc
    {
        uint32_t adapterIndex = 0;
        bool_t   headless     = false;
    };

} // namespace DigitalTwin