#pragma once
#include "rhi/RHITypes.h"

#include "core/Core.h"
#include <map>
#include <string>
#include <unordered_map>
#include <vector>
#include <volk.h>

namespace DigitalTwin
{

    /**
     * @brief Consolidated reflection data for a Shader or a Pipeline.
     * Structured for efficient lookup during binding.
     */
    struct ShaderReflectionData
    {
        // Organized as: Set Index -> Binding Index -> Resource Data
        std::map<uint32_t, std::map<uint32_t, ShaderResource>> resources;

        // Push Constants ranges found in the shader(s)
        std::vector<VkPushConstantRange> pushConstants;

        /**
         * @brief Finds a resource by set and binding index.
         * @return Pointer to the resource or nullptr if not found.
         */
        const ShaderResource* Find( uint32_t set, uint32_t binding ) const
        {
            auto setIt = resources.find( set );
            if( setIt != resources.end() )
            {
                auto bindIt = setIt->second.find( binding );
                if( bindIt != setIt->second.end() )
                {
                    return &bindIt->second;
                }
            }
            return nullptr;
        }

        /**
         * @brief Finds a resource by name within a specific set.
         * Note: This performs a linear search within the set.
         */
        const ShaderResource* Find( uint32_t set, const std::string& name ) const
        {
            auto setIt = resources.find( set );
            if( setIt != resources.end() )
            {
                for( const auto& [ binding, res ]: setIt->second )
                {
                    if( res.name == name )
                        return &res;
                }
            }
            return nullptr;
        }
    };

    class Shader
    {
    public:
        /**
         * @brief Constructor. Compiles GLSL to SPIR-V and reflects resources.
         * @param device Vulkan logical device handle.
         * @param api Pointer to the device-specific function table (volk).
         * @param filepath Path to the source GLSL file (.vert, .frag, .comp).
         */
        Shader( VkDevice device, const VolkDeviceTable* api );
        ~Shader();

        Result Create( const std::string& filepath );
        void   Destroy();

        // --- Getters ---
        VkShaderModule              GetModule() const { return m_module; }
        VkShaderStageFlagBits       GetStage() const { return m_stage; }
        const ShaderReflectionData& GetReflectionData() const { return m_reflectionData; }

        // Debug helper - logs detected resources to the console
        void LogResources() const;

    public:
        // Disable copying (RAII), allow moving
        Shader( const Shader& )            = delete;
        Shader& operator=( const Shader& ) = delete;
        Shader( Shader&& other ) noexcept;
        Shader& operator=( Shader&& other ) noexcept;

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
        ShaderReflectionData m_reflectionData;
    };
} // namespace DigitalTwin