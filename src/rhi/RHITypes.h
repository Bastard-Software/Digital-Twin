#pragma once

#include "core/Core.h"
#include <string>
#include <unordered_map>
#include <volk.h>
#include "resources/Handle.h"

namespace DigitalTwin
{

    class RHI;
    class Device;
    class Queue;
    class CommandBuffer;
    class Buffer;
    class Texture;
    class Sampler;
    class Shader;
    class ComputePipeline;
    class GraphicsPipeline;
    class CommandBuffer;
    class ThreadContext;

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

    /**
     * @brief Types of buffers used in the RHI.
     */
    enum class BufferType
    {
        UPLOAD,
        READBACK,
        STORAGE,
        UNIFORM,
        MESH,
        INDIRECT,
        ATOMIC_COUNTER,
        _MAX_ENUM
    };

    /**
     * @brief Description for creating a Buffer.
     */
    struct BufferDesc
    {
        VkDeviceSize size = 0;
        BufferType   type = BufferType::STORAGE;
    };

    /**
     * @brief Usage flags for Textures.
     */
    enum class TextureUsage : uint32_t
    {
        NONE                 = 0,
        SAMPLED              = 1 << 0, // Read by shader (sampler/texture)
        STORAGE              = 1 << 1, // Written by Compute Shader (imageStore)
        RENDER_TARGET        = 1 << 2, // Color Attachment
        DEPTH_STENCIL_TARGET = 1 << 3, // Depth/Stencil Attachment
        TRANSFER_SRC         = 1 << 4, // Can be copied from
        TRANSFER_DST         = 1 << 5  // Can be copied to
    };

    /**
     * @brief Bitwise operators for TextureUsage enum.
     */
    inline TextureUsage operator|( TextureUsage a, TextureUsage b )
    {
        return static_cast<TextureUsage>( static_cast<uint32_t>( a ) | static_cast<uint32_t>( b ) );
    }
    inline TextureUsage operator&( TextureUsage a, TextureUsage b )
    {
        return static_cast<TextureUsage>( static_cast<uint32_t>( a ) & static_cast<uint32_t>( b ) );
    }
    inline bool HasFlag( TextureUsage value, TextureUsage flag )
    {
        return ( static_cast<uint32_t>( value ) & static_cast<uint32_t>( flag ) ) == static_cast<uint32_t>( flag );
    }

    /**
     * @brief Types of textures.
     */
    enum class TextureType
    {
        Texture1D,
        Texture2D,
        Texture3D,
        TextureCube, // TODO: Not implemented yet
    };

    /**
     * @brief Description for creating a Texture.
     */
    struct TextureDesc
    {
        uint32_t width  = 1;
        uint32_t height = 1;
        uint32_t depth  = 1;

        TextureType type   = TextureType::Texture2D;
        VkFormat    format = VK_FORMAT_R8G8B8A8_UNORM;

        TextureUsage usage = TextureUsage::SAMPLED | TextureUsage::STORAGE | TextureUsage::TRANSFER_SRC | TextureUsage::TRANSFER_DST;
    };

    /**
     * @brief Description for creating a Sampler.
     */
    struct SamplerDesc
    {
        VkFilter             magFilter    = VK_FILTER_LINEAR;
        VkFilter             minFilter    = VK_FILTER_LINEAR;
        VkSamplerAddressMode addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        VkSamplerAddressMode addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        VkSamplerAddressMode addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        VkSamplerMipmapMode  mipmapMode   = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    };

    /*
     * @brief Types of shader resources.
     */
    enum class ShaderResourceType
    {
        UNIFORM_BUFFER,
        STORAGE_BUFFER,
        SAMPLED_IMAGE,
        STORAGE_IMAGE,
        PUSH_CONSTANT,
        UNKNOWN,
        _MAX_ENUM,
    };

    /**
     * @brief Information about a shader resource (uniform/storage buffer, image, push constant).
     */
    struct ShaderResource
    {
        std::string        name;
        uint32_t           set;
        uint32_t           binding;
        uint32_t           size;      // Size in bytes (for buffers)
        uint32_t           arraySize; // Element count (for descriptor arrays)
        uint32_t           offset;    // Offset (for Push Constants)
        ShaderResourceType type;
    };

    // Map: Shader Variable Name -> Resource Information
    using ShaderReflectionData = std::unordered_map<std::string, ShaderResource>;

    /**
     * @brief Description for creating a Compute Pipeline.
     */
    struct ComputePipelineNativeDesc
    {
        Shader* shader;
    };

    /**
     * @brief Description for creating a Compute Pipeline (handle varsion).
     */
    struct ComputePipelineDesc
    {
        ShaderHandle shader;
    };

    /**
     * @brief Description for creating a Graphics Pipeline.
     * TODO: Refactor to be less verbose
     */
    struct GraphicsPipelineNativeDesc
    {
        Shader* vertexShader;
        Shader* fragmentShader;

        // Input Assembly
        VkPrimitiveTopology topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

        // Rasterization
        VkPolygonMode   polygonMode = VK_POLYGON_MODE_FILL;
        VkCullModeFlags cullMode    = VK_CULL_MODE_BACK_BIT;
        VkFrontFace     frontFace   = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        float           lineWidth   = 1.0f;

        // Depth Stencil
        bool        depthTestEnable  = true;
        bool        depthWriteEnable = true;
        VkCompareOp depthCompareOp   = VK_COMPARE_OP_LESS;

        // Blending (Simple one-size-fits-all approach for now)
        bool          blendEnable         = false;
        VkBlendFactor srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
        VkBlendFactor dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        VkBlendOp     colorBlendOp        = VK_BLEND_OP_ADD;
        VkBlendFactor srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        VkBlendFactor dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
        VkBlendOp     alphaBlendOp        = VK_BLEND_OP_ADD;

        // Dynamic Rendering Formats (Vulkan 1.3)
        // Pipeline must know what it is rendering into
        std::vector<VkFormat> colorAttachmentFormats = { VK_FORMAT_R8G8B8A8_UNORM };
        VkFormat              depthAttachmentFormat  = VK_FORMAT_D32_SFLOAT;
    };

    /**
     * @brief Description for creating a Graphics Pipeline (handle version).
     */
    struct GraphicsPipelineDesc
    {
        ShaderHandle vertexShader;
        ShaderHandle fragmentShader;

        // Input Assembly
        VkPrimitiveTopology topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

        // Rasterization
        VkPolygonMode   polygonMode = VK_POLYGON_MODE_FILL;
        VkCullModeFlags cullMode    = VK_CULL_MODE_BACK_BIT;
        VkFrontFace     frontFace   = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        float           lineWidth   = 1.0f;

        // Depth Stencil
        bool        depthTestEnable  = true;
        bool        depthWriteEnable = true;
        VkCompareOp depthCompareOp   = VK_COMPARE_OP_LESS;

        // Blending (Simple one-size-fits-all approach for now)
        bool          blendEnable         = false;
        VkBlendFactor srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
        VkBlendFactor dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        VkBlendOp     colorBlendOp        = VK_BLEND_OP_ADD;
        VkBlendFactor srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        VkBlendFactor dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
        VkBlendOp     alphaBlendOp        = VK_BLEND_OP_ADD;

        // Dynamic Rendering Formats (Vulkan 1.3)
        // Pipeline must know what it is rendering into
        std::vector<VkFormat> colorAttachmentFormats = { VK_FORMAT_R8G8B8A8_UNORM };
        VkFormat              depthAttachmentFormat  = VK_FORMAT_D32_SFLOAT;
    };

} // namespace DigitalTwin