#pragma once
#include "rhi/RHITypes.h"

#include "core/Memory/MemorySystem.h"
#include "resources/ResourcePool.h"
#include <deque>
#include <functional>

namespace DigitalTwin
{
    /**
     * @brief Central manager for all GPU resources.
     * Owns all resources and manages their lifetime using a Handle system.
     * Implements lazy deletion synchronized with GPU fences.
     */
    class ResourceManager
    {
    public:
        /**
         * @brief Constructor.
         * @param device Pointer to the RHI Device (Resource Manager does NOT own the device).
         */
        ResourceManager( Device* device, MemorySystem* memSystem );
        ~ResourceManager();

        // --- Frame Lifecycle ---

        /**
         * @brief Must be called at the start of every frame.
         * Checks GPU fences and permanently deletes resources that are no longer in use.
         */
        void BeginFrame();

        Result Initialize();
        /**
         * @brief Forcefully destroys all resources. Called on application exit.
         */
        void Shutdown();

        // --- Buffer API ---
        BufferHandle CreateBuffer( const BufferDesc& desc );
        Buffer*      GetBuffer( BufferHandle handle );
        void         DestroyBuffer( BufferHandle handle );

        // --- Texture API ---
        TextureHandle CreateTexture( const TextureDesc& desc );
        Texture*      GetTexture( TextureHandle handle );
        void          DestroyTexture( TextureHandle handle );

        // --- Sampler API ---
        SamplerHandle CreateSampler( const SamplerDesc& desc );
        Sampler*      GetSampler( SamplerHandle handle );
        void          DestroySampler( SamplerHandle handle );

        // --- Shader API ---
        ShaderHandle CreateShader( const std::string& filepath );
        Shader*      GetShader( ShaderHandle handle );
        void         DestroyShader( ShaderHandle handle );

        // --- Pipeline API ---
        ComputePipelineHandle  CreatePipeline( const ComputePipelineDesc& desc );
        GraphicsPipelineHandle CreatePipeline( const GraphicsPipelineDesc& desc );
        ComputePipeline*       GetPipeline( ComputePipelineHandle handle );
        GraphicsPipeline*      GetPipeline( GraphicsPipelineHandle handle );
        void                   DestroyPipeline( ComputePipelineHandle handle );
        void                   DestroyPipeline( GraphicsPipelineHandle handle );

    private:
        struct ZombieResource
        {
            // Type-erased deleter that holds the resource until destruction.
            std::function<void()> deleter;

            // We capture fence values for ALL queues at the moment of destruction.
            // The resource is only safe to delete when ALL queues have passed these values.
            uint64_t graphicsFenceValue = 0;
            uint64_t computeFenceValue  = 0;
            uint64_t transferFenceValue = 0;
        };

        // Helper to move a resource to the deletion queue
        template<typename T, typename Deleter>
        void EnqueueDeletion( std::unique_ptr<T, Deleter> resource );

    private:
        Device*       m_device;
        MemorySystem* m_memorySystem;

        //  Pools
        using BufferDeleter = std::function<void( Buffer* )>;
        ResourcePool<Buffer, BufferHandle, BufferDeleter> m_buffers;
        using TextureDeleter = std::function<void( Texture* )>;
        ResourcePool<Texture, TextureHandle, TextureDeleter> m_textures;
        using SamplerDeleter = std::function<void( Sampler* )>;
        ResourcePool<Sampler, SamplerHandle, SamplerDeleter> m_samplers;
        using ShaderDeleter = std::function<void( Shader* )>;
        ResourcePool<Shader, ShaderHandle, ShaderDeleter> m_shaders;
        using ComputePipelineDeleter = std::function<void( ComputePipeline* )>;
        ResourcePool<ComputePipeline, ComputePipelineHandle, ComputePipelineDeleter> m_computePipelines;
        using GraphicsPipelineDeleter = std::function<void( GraphicsPipeline* )>;
        ResourcePool<GraphicsPipeline, GraphicsPipelineHandle, GraphicsPipelineDeleter> m_graphicsPipelines;

        // Lazy deletion queue
        std::deque<ZombieResource> m_zombies;
    };
} // namespace DigitalTwin