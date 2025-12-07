#pragma once
#include "core/Base.hpp"
#include <string>
#include <unordered_map>
#include <vector>
#include <volk.h>

namespace DigitalTwin
{
    // Enum representing the type of resource extracted via SPIR-V reflection
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

    // Structure describing a single shader resource (binding or constant)
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

    class Shader
    {
    public:
        /**
         * @brief Constructor. Compiles GLSL to SPIR-V and reflects resources.
         * @param device Vulkan logical device handle.
         * @param api Pointer to the device-specific function table (volk).
         * @param filepath Path to the source GLSL file (.vert, .frag, .comp).
         */
        Shader( VkDevice device, const VolkDeviceTable* api, const std::string& filepath );

        ~Shader();

        // Disable copying to prevent double-free of VkShaderModule
        Shader( const Shader& )            = delete;
        Shader& operator=( const Shader& ) = delete;

        // Allow moving
        Shader( Shader&& other ) noexcept;
        Shader& operator=( Shader&& other ) noexcept;

        // --- Getters ---
        VkShaderModule                          GetModule() const { return m_module; }
        VkShaderStageFlagBits                   GetStage() const { return m_stage; }
        const ShaderReflectionData&             GetReflectionData() const { return m_reflectionData; }
        const std::vector<VkPushConstantRange>& GetPushConstantRanges() const { return m_pushConstantRanges; }

        // Debug helper - logs detected resources to the console
        void LogResources() const;

    private:
        // Reads file, compiles GLSL to SPIR-V using shaderc, handles caching
        // Returns the SPIR-V binary code
        std::vector<uint32_t> CompileOrGetCache( const std::string& source, const std::string& filename );

        // Analyzes SPIR-V binary using SPIRV-Reflect to extract bindings
        void Reflect( const std::vector<uint32_t>& spirv );

        // Helper to read text file content from disk
        std::string ReadFile( const std::string& filepath );

        // Helper to deduce shader stage from file extension (e.g., .vert -> VERTEX_BIT)
        VkShaderStageFlagBits InferStageFromPath( const std::string& filepath );

    private:
        VkDevice               m_deviceHandle = VK_NULL_HANDLE;
        const VolkDeviceTable* m_api          = nullptr; // Function pointers for this device

        VkShaderModule        m_module = VK_NULL_HANDLE;
        VkShaderStageFlagBits m_stage  = VK_SHADER_STAGE_FLAG_BITS_MAX_ENUM;

        // Reflected metadata
        ShaderReflectionData             m_reflectionData;
        std::vector<VkPushConstantRange> m_pushConstantRanges;
    };
} // namespace DigitalTwin