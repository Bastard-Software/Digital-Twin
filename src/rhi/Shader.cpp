#include "rhi/Shader.h"

#include "core/Log.h"
#include <filesystem>
#include <fstream>
#include <shaderc/shaderc.hpp>
#include <spirv_reflect.h>

namespace DigitalTwin
{

    static ShaderResourceType ConvertSpirvType( SpvReflectDescriptorType type )
    {
        switch( type )
        {
            case SPV_REFLECT_DESCRIPTOR_TYPE_SAMPLER:
                return ShaderResourceType::SAMPLER;
            case SPV_REFLECT_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
                return ShaderResourceType::COMBINED_IMAGE_SAMPLER;
            case SPV_REFLECT_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
                return ShaderResourceType::SAMPLED_IMAGE;
            case SPV_REFLECT_DESCRIPTOR_TYPE_STORAGE_IMAGE:
                return ShaderResourceType::STORAGE_IMAGE;
            case SPV_REFLECT_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER:
                return ShaderResourceType::UNIFORM_TEXEL_BUFFER;
            case SPV_REFLECT_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER:
                return ShaderResourceType::STORAGE_TEXEL_BUFFER;
            case SPV_REFLECT_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
                return ShaderResourceType::UNIFORM_BUFFER;
            case SPV_REFLECT_DESCRIPTOR_TYPE_STORAGE_BUFFER:
                return ShaderResourceType::STORAGE_BUFFER;
            case SPV_REFLECT_DESCRIPTOR_TYPE_INPUT_ATTACHMENT:
                return ShaderResourceType::INPUT_ATTACHMENT;
            case SPV_REFLECT_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR:
                return ShaderResourceType::ACCELERATION_STRUCTURE;
            default:
                return ShaderResourceType::UNKNOWN;
        }
    }

    Shader::Shader( VkDevice device, const VolkDeviceTable* api )
        : m_deviceHandle( device )
        , m_api( api )
    {
        DT_CORE_ASSERT( m_api, "VolkDeviceTable is null!" );
        DT_CORE_ASSERT( m_deviceHandle != VK_NULL_HANDLE, "Invalid device handle!" );
    }

    Shader::~Shader()
    {
    }

    Result Shader::Create( const std::string& filepath )
    {
        std::string source = ReadFile( filepath );
        m_stage            = InferStageFromPath( filepath );

        // 1. Compile or load from cache
        std::vector<uint32_t> spirv = CompileOrGetCache( source, filepath );

        if( spirv.empty() )
        {
            DT_ERROR( "Failed to compile or load shader: {}", filepath );
            return Result::FAIL;
        }

        // 2. Create Vulkan Shader Module using provided API table
        VkShaderModuleCreateInfo createInfo = { VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
        createInfo.codeSize                 = spirv.size() * sizeof( uint32_t );
        createInfo.pCode                    = spirv.data();

        if( m_api->vkCreateShaderModule( m_deviceHandle, &createInfo, nullptr, &m_module ) != VK_SUCCESS )
        {
            DT_CRITICAL( "Failed to create shader module for: {}", filepath );
            return Result::FAIL;
        }

        // 3. Reflect Resources (Extract Layout)
        Reflect( spirv );

        return Result::SUCCESS;
    }

    void Shader::Destroy()
    {
        if( m_module != VK_NULL_HANDLE && m_api )
        {
            m_api->vkDestroyShaderModule( m_deviceHandle, m_module, nullptr );
        }
    }

    Shader::Shader( Shader&& other ) noexcept
        : m_deviceHandle( other.m_deviceHandle )
        , m_api( other.m_api )
        , m_module( other.m_module )
        , m_stage( other.m_stage )
        , m_reflectionData( std::move( other.m_reflectionData ) )
    {
        other.m_module = VK_NULL_HANDLE;
    }

    Shader& Shader::operator=( Shader&& other ) noexcept
    {
        if( this != &other )
        {
            // Destroy current resources
            if( m_module != VK_NULL_HANDLE && m_api )
            {
                m_api->vkDestroyShaderModule( m_deviceHandle, m_module, nullptr );
            }

            // Move data
            m_deviceHandle   = other.m_deviceHandle;
            m_api            = other.m_api;
            m_module         = other.m_module;
            m_stage          = other.m_stage;
            m_reflectionData = std::move( other.m_reflectionData );

            other.m_module = VK_NULL_HANDLE;
        }
        return *this;
    }

    std::vector<uint32_t> Shader::CompileOrGetCache( const std::string& source, const std::string& filepath )
    {
        // 1. Convert strings to filesystem paths for easier manipulation
        std::filesystem::path sourcePath( filepath );
        std::filesystem::path cachePath = sourcePath;
        cachePath += ".spv"; // Append .spv extension

        // 2. Convert back to string using generic_string()
        // This forces forward slashes ('/') on Windows, fixing the mixed slash visual issue in logs.
        std::string sourceStr = sourcePath.generic_string();
        std::string cacheStr  = cachePath.generic_string();

        bool shouldCompile = false;

        // --- SMART CACHE CHECK ---

        // A. Check if the binary file exists
        if( !std::filesystem::exists( cachePath ) )
        {
            shouldCompile = true;
            DT_WARN( "Shader cache missing for: {}. Compiling...", sourceStr );
        }
        else
        {
            // B. Check Timestamps (Source vs. Binary)
            // We compare the last modification time of the source file against the cache file.
            auto sourceTime = std::filesystem::last_write_time( sourcePath );
            auto cacheTime  = std::filesystem::last_write_time( cachePath );

            // If the source file is newer than the cache, the code has changed.
            if( sourceTime > cacheTime )
            {
                shouldCompile = true;
                DT_WARN( "Shader source detected as newer than cache: {}. Recompiling...", sourceStr );
            }
        }

        // --- PATH A: LOAD FROM CACHE ---
        // If the cache is valid, simply load the binary data from disk.
        if( !shouldCompile )
        {
            DT_INFO( "Loading shader from cache: {}", cacheStr );

            std::ifstream in( cachePath, std::ios::in | std::ios::binary );
            if( in )
            {
                in.seekg( 0, std::ios::end );
                size_t fileSize = in.tellg();
                in.seekg( 0, std::ios::beg );

                std::vector<uint32_t> spirv( fileSize / sizeof( uint32_t ) );
                in.read( reinterpret_cast<char*>( spirv.data() ), fileSize );
                return spirv;
            }
            else
            {
                // Fallback: If opening the file failed (e.g., lock issue), force recompilation.
                DT_ERROR( "Failed to open cache file despite it existing. Forcing recompilation." );
                shouldCompile = true;
            }
        }

        // --- PATH B: COMPILE FROM SOURCE ---
        // Compile GLSL to SPIR-V using shaderc.
        DT_INFO( "Compiling shader: {}", sourceStr );

        shaderc::Compiler       compiler;
        shaderc::CompileOptions options;

        options.SetTargetEnvironment( shaderc_target_env_vulkan, shaderc_env_version_vulkan_1_3 );

        // We use performance optimization, but keep DebugInfo enabled.
        // DebugInfo is required for SPIRV-Reflect to correctly read variable names.
        options.SetOptimizationLevel( shaderc_optimization_level_performance );
        options.SetGenerateDebugInfo();

        shaderc_shader_kind kind;
        switch( m_stage )
        {
            case VK_SHADER_STAGE_VERTEX_BIT:
                kind = shaderc_glsl_vertex_shader;
                break;
            case VK_SHADER_STAGE_FRAGMENT_BIT:
                kind = shaderc_glsl_fragment_shader;
                break;
            case VK_SHADER_STAGE_COMPUTE_BIT:
                kind = shaderc_glsl_compute_shader;
                break;
            default:
                kind = shaderc_glsl_infer_from_source;
                break;
        }

        // Note: We pass 'sourceStr.c_str()' as the filename so compiler errors also use correct slashes.
        shaderc::SpvCompilationResult module = compiler.CompileGlslToSpv( source, kind, sourceStr.c_str(), options );

        if( module.GetCompilationStatus() != shaderc_compilation_status_success )
        {
            DT_CRITICAL( "Shader Compilation Failed ({0}):\n{1}", sourceStr, module.GetErrorMessage() );
            return {};
        }

        std::vector<uint32_t> spirv( module.cbegin(), module.cend() );

        // --- SAVE TO CACHE ---
        // Write the compiled SPIR-V back to disk for next time.
        std::ofstream out( cachePath, std::ios::out | std::ios::binary );
        if( out )
        {
            out.write( reinterpret_cast<const char*>( spirv.data() ), spirv.size() * sizeof( uint32_t ) );
            DT_INFO( "Shader cache saved to: {}", cacheStr );
        }
        else
        {
            DT_ERROR( "Failed to write shader cache to: {}", cacheStr );
        }

        return spirv;
    }

    void Shader::Reflect( const std::vector<uint32_t>& spirv )
    {
        SpvReflectShaderModule module;
        // Parse SPIR-V binary using SPIRV-Reflect
        SpvReflectResult result = spvReflectCreateShaderModule( spirv.size() * sizeof( uint32_t ), spirv.data(), &module );
        DT_ASSERT( result == SPV_REFLECT_RESULT_SUCCESS, "SPIRV-Reflect failed!" );

        // 1. Enumerate Descriptor Sets
        uint32_t count = 0;
        spvReflectEnumerateDescriptorSets( &module, &count, nullptr );
        std::vector<SpvReflectDescriptorSet*> sets( count );
        spvReflectEnumerateDescriptorSets( &module, &count, sets.data() );

        for( auto* set: sets )
        {
            for( uint32_t i = 0; i < set->binding_count; ++i )
            {
                const SpvReflectDescriptorBinding* binding = set->bindings[ i ];

                ShaderResource res{};
                res.name       = binding->name ? binding->name : "";
                res.set        = binding->set;
                res.binding    = binding->binding;
                res.type       = ConvertSpirvType( binding->descriptor_type );
                res.arraySize  = binding->count;
                res.stageFlags = ( VkShaderStageFlags )m_stage; // Only current stage known here

                // For UBO/SSBO, calculate size
                res.size = 0;
                if( binding->descriptor_type == SPV_REFLECT_DESCRIPTOR_TYPE_UNIFORM_BUFFER ||
                    binding->descriptor_type == SPV_REFLECT_DESCRIPTOR_TYPE_STORAGE_BUFFER )
                {
                    res.size = binding->block.size;
                }

                // Insert into map
                m_reflectionData.resources[ res.set ][ res.binding ] = res;
            }
        }

        // 2. Enumerate Push Constants
        count = 0;
        spvReflectEnumeratePushConstantBlocks( &module, &count, nullptr );
        std::vector<SpvReflectBlockVariable*> pushConstants( count );
        spvReflectEnumeratePushConstantBlocks( &module, &count, pushConstants.data() );

        m_reflectionData.pushConstants.clear();
        for( const auto* pc: pushConstants )
        {
            VkPushConstantRange range{};
            range.offset     = pc->offset;
            range.size       = pc->size;
            range.stageFlags = ( VkShaderStageFlags )m_stage;
            m_reflectionData.pushConstants.push_back( range );
        }

        spvReflectDestroyShaderModule( &module );
    }

    std::string Shader::ReadFile( const std::string& filepath )
    {
        std::ifstream in( filepath, std::ios::in | std::ios::binary );
        if( in )
        {
            std::string contents;
            in.seekg( 0, std::ios::end );
            contents.resize( in.tellg() );
            in.seekg( 0, std::ios::beg );
            in.read( &contents[ 0 ], contents.size() );
            in.close();
            return contents;
        }
        DT_CRITICAL( "Could not open shader file: {}", filepath );
        throw std::runtime_error( "Shader file not found" );
    }

    VkShaderStageFlagBits Shader::InferStageFromPath( const std::string& filepath )
    {
        if( filepath.find( ".vert" ) != std::string::npos )
            return VK_SHADER_STAGE_VERTEX_BIT;
        if( filepath.find( ".frag" ) != std::string::npos )
            return VK_SHADER_STAGE_FRAGMENT_BIT;
        if( filepath.find( ".comp" ) != std::string::npos )
            return VK_SHADER_STAGE_COMPUTE_BIT;

        DT_WARN( "Could not infer shader stage from file extension: {}", filepath );
        return VK_SHADER_STAGE_ALL;
    }

    void Shader::LogResources() const
    {
        DT_INFO( "--- Shader Reflection ({}) ---", ( int )m_stage );
        for( const auto& [ setIdx, bindings ]: m_reflectionData.resources )
        {
            for( const auto& [ bindingIdx, res ]: bindings )
            {
                DT_INFO( "  Set {} Binding {} : '{}' (Type: {}) Size: {}", setIdx, bindingIdx, res.name, ( int )res.type, res.size );
            }
        }
    }
} // namespace DigitalTwin