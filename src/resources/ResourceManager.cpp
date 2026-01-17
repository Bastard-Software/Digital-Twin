#include "resources/ResourceManager.h"

// RHI Headers - needed for full type definitions (destructors, create methods)
#include "rhi/Buffer.h"
#include "rhi/Device.h"
#include "rhi/Pipeline.h"
#include "rhi/Queue.h"
#include "rhi/Sampler.h"
#include "rhi/Shader.h"
#include "rhi/Texture.h"

namespace DigitalTwin
{
    ResourceManager::ResourceManager( Device* device, MemorySystem* memSystem )
        : m_device( device )
        , m_memorySystem( memSystem )
    {
        DT_CORE_ASSERT( m_device, "ResourceManager received null Device!" );
        DT_CORE_ASSERT( m_memorySystem, "ResourceManager received null MemorySystem!" );
    }

    ResourceManager::~ResourceManager()
    {
    }

    void ResourceManager::Shutdown()
    {
        DT_INFO( "[ResourceManager] Shutting down..." );

        // 1. Wait for GPU to finish everything
        m_device->WaitIdle();

        // 2. Clear all pools.
        // The unique_ptrs will be destroyed, triggering VMA/API cleanup immediately
        // because we are shutting down and GPU is idle.
        m_buffers.Clear();
        m_textures.Clear();
        /*
        m_shaders.Clear();
        m_samplers.Clear();
        m_computePipelines.Clear();
        m_graphicsPipelines.Clear();
        */

        // 3. Clear pending zombies
        m_zombies.clear();

        DT_INFO( "[ResourceManager] All resources released." );
    }

    void ResourceManager::BeginFrame()
    {
        // Garbage Collection: Process the zombie queue
        if( m_zombies.empty() )
            return;

        // Process deletion queue
        while( !m_zombies.empty() )
        {
            ZombieResource& zombie = m_zombies.front();

            // Check if ALL queues have finished using the resource since it was marked for deletion
            bool gfxDone   = m_device->GetGraphicsQueue()->IsValueCompleted( zombie.graphicsFenceValue );
            bool compDone  = m_device->GetComputeQueue()->IsValueCompleted( zombie.computeFenceValue );
            bool transDone = m_device->GetTransferQueue()->IsValueCompleted( zombie.transferFenceValue );

            if( gfxDone && compDone && transDone )
            {
                // Safe to delete
                m_zombies.pop_front();
            }
            else
            {
                // Oldest resource still in use, stop checking
                break;
            }
        }
    }

    Result ResourceManager::Initialize()
    {
        // No special initialization needed yet
        DT_INFO( "[ResourceManager] Initialized." );

        return Result::SUCCESS;
    }

    template<typename T, typename Deleter>
    void ResourceManager::EnqueueDeletion( std::unique_ptr<T, Deleter> resource )
    {
        if( !resource )
            return;

        ZombieResource zombie;

        // Capture the LAST submitted fence value for every queue.
        // Even if the resource was only used on Graphics, waiting for Compute/Transfer
        // to catch up to their current point is safe (though slightly conservative).
        // This guarantees we never delete a resource used by an async queue.
        zombie.graphicsFenceValue = m_device->GetGraphicsQueue()->GetLastSubmittedValue();
        zombie.computeFenceValue  = m_device->GetComputeQueue()->GetLastSubmittedValue();
        zombie.transferFenceValue = m_device->GetTransferQueue()->GetLastSubmittedValue();

        // Capture ownership in the lambda
        std::shared_ptr<T> keptResource = std::move( resource );
        Device*            devicePtr    = m_device;
        zombie.deleter                  = [ keptResource, devicePtr ]() {
            if constexpr( std::is_same_v<T, Buffer> )
            {
                devicePtr->DestroyBuffer( keptResource.get() );
            }
            else if constexpr( std::is_same_v<T, Texture> )
            {
                devicePtr->DestroyTexture( keptResource.get() );
            }
            else if constexpr( std::is_same_v<T, Sampler> )
            {
                devicePtr->DestroySampler( keptResource.get() );
            }
            else if constexpr( std::is_same_v<T, Shader> )
            {
                devicePtr->DestroyShader( keptResource.get() );
            }
            else if constexpr( std::is_same_v<T, ComputePipeline> )
            {
                devicePtr->DestroyComputePipeline( keptResource.get() );
            }
            else if constexpr( std::is_same_v<T, GraphicsPipeline> )
            {
                devicePtr->DestroyGraphicsPipeline( keptResource.get() );
            }
        };

        m_zombies.push_back( std::move( zombie ) );
    }

    // --- Buffer ---

    BufferHandle ResourceManager::CreateBuffer( const BufferDesc& desc )
    {
        // 1. Allocate Raw Memory (Tracked)
        IAllocator* allocator = m_memorySystem->GetSystemAllocator();
        void*       mem       = allocator->Allocate( sizeof( Buffer ), __FILE__, __LINE__ );

        if( !mem )
        {
            DT_ERROR( "Failed to allocate memory for Buffer!" );
            return BufferHandle::Invalid;
        }

        // 2. Construct Object (Placement New)
        // Pass device dependencies needed for internal logic
        Buffer* rawBuffer = new( mem ) Buffer( m_device->GetAllocator(), m_device->GetHandle(), &m_device->GetAPI() );

        // 3. Define Deleter
        // This lambda encapsulates how to destroy the C++ object (destructor + free)
        auto deleter = [ allocator ]( Buffer* ptr ) {
            if( ptr )
            {
                ptr->Destroy();         // Ensure Vulkan resources are released
                ptr->~Buffer();         // Call destructor manually
                allocator->Free( ptr ); // Free memory
            }
        };

        // 4. Initialize Vulkan Resource via Device
        if( m_device->CreateBuffer( desc, rawBuffer ) != Result::SUCCESS )
        {
            DT_ERROR( "Failed to initialize Vulkan Buffer!" );
            deleter( rawBuffer ); // Cleanup if init fails
            return BufferHandle::Invalid;
        }

        // 5. Wrap in unique_ptr and Insert into Pool
        std::unique_ptr<Buffer, BufferDeleter> managedBuffer( rawBuffer, deleter );

        return m_buffers.Insert( std::move( managedBuffer ) );
    }

    Buffer* ResourceManager::GetBuffer( BufferHandle handle )
    {
        return m_buffers.Get( handle );
    }

    void ResourceManager::DestroyBuffer( BufferHandle handle )
    {
        auto bufPtr = m_buffers.Remove( handle );
        EnqueueDeletion( std::move( bufPtr ) );
    }

    // --- Texture ---

    TextureHandle ResourceManager::CreateTexture( const TextureDesc& desc )
    {
        // 1. Allocate Raw Memory (Tracked)
        IAllocator* allocator = m_memorySystem->GetSystemAllocator();
        void*       mem       = allocator->Allocate( sizeof( Texture ), __FILE__, __LINE__ );

        if( !mem )
        {
            DT_ERROR( "Failed to allocate memory for Buffer!" );
            return TextureHandle::Invalid;
        }

        // 2. Construct Object (Placement New)
        // Pass device dependencies needed for internal logic
        Texture* rawTexture = new( mem ) Texture( m_device->GetAllocator(), m_device->GetHandle(), &m_device->GetAPI() );

        // 3. Define Deleter
        // This lambda encapsulates how to destroy the C++ object (destructor + free)
        auto deleter = [ allocator ]( Texture* ptr ) {
            if( ptr )
            {
                ptr->Destroy();         // Ensure Vulkan resources are released
                ptr->~Texture();        // Call destructor manually
                allocator->Free( ptr ); // Free memory
            }
        };

        // 4. Initialize Vulkan Resource via Device
        if( m_device->CreateTexture( desc, rawTexture ) != Result::SUCCESS )
        {
            DT_ERROR( "Failed to initialize Vulkan Texture!" );
            deleter( rawTexture ); // Cleanup if init fails
            return TextureHandle::Invalid;
        }

        // 5. Wrap in unique_ptr and Insert into Pool
        std::unique_ptr<Texture, TextureDeleter> managedTexture( rawTexture, deleter );

        return m_textures.Insert( std::move( managedTexture ) );
    }

    Texture* ResourceManager::GetTexture( TextureHandle handle )
    {
        return m_textures.Get( handle );
    }

    void ResourceManager::DestroyTexture( TextureHandle handle )
    {
        auto texPtr = m_textures.Remove( handle );
        EnqueueDeletion( std::move( texPtr ) );
    }

    SamplerHandle ResourceManager::CreateSampler( const SamplerDesc& desc )
    {
        // 1. Allocate Raw Memory (Tracked)
        IAllocator* allocator = m_memorySystem->GetSystemAllocator();
        void*       mem       = allocator->Allocate( sizeof( Sampler ), __FILE__, __LINE__ );

        if( !mem )
        {
            DT_ERROR( "Failed to allocate memory for Sampler!" );
            return SamplerHandle::Invalid;
        }

        // 2. Construct Object (Placement New)
        // Pass device dependencies needed for internal logic
        Sampler* rawSampler = new( mem ) Sampler( m_device->GetHandle(), &m_device->GetAPI() );

        // 3. Define Deleter
        // This lambda encapsulates how to destroy the C++ object (destructor + free)
        auto deleter = [ allocator ]( Sampler* ptr ) {
            if( ptr )
            {
                ptr->Destroy();         // Ensure Vulkan resources are released
                ptr->~Sampler();        // Call destructor manually
                allocator->Free( ptr ); // Free memory
            }
        };

        // 4. Initialize Vulkan Resource via Device
        if( m_device->CreateSampler( desc, rawSampler ) != Result::SUCCESS )
        {
            DT_ERROR( "Failed to initialize Vulkan Shader!" );
            deleter( rawSampler ); // Cleanup if init fails
            return SamplerHandle::Invalid;
        }

        // 5. Wrap in unique_ptr and Insert into Pool
        std::unique_ptr<Sampler, SamplerDeleter> managedSampler( rawSampler, deleter );

        return m_samplers.Insert( std::move( managedSampler ) );
    }

    Sampler* ResourceManager::GetSampler( SamplerHandle handle )
    {
        return m_samplers.Get( handle );
    }

    void ResourceManager::DestroySampler( SamplerHandle handle )
    {
        auto samplerPtr = m_samplers.Remove( handle );
        EnqueueDeletion( std::move( samplerPtr ) );
    }

    ShaderHandle ResourceManager::CreateShader( const std::string& filepath )
    {
        // 1. Allocate Raw Memory (Tracked)
        IAllocator* allocator = m_memorySystem->GetSystemAllocator();
        void*       mem       = allocator->Allocate( sizeof( Shader ), __FILE__, __LINE__ );

        if( !mem )
        {
            DT_ERROR( "Failed to allocate memory for Shader!" );
            return ShaderHandle::Invalid;
        }

        // 2. Construct Object (Placement New)
        // Pass device dependencies needed for internal logic
        Shader* rawShader = new( mem ) Shader( m_device->GetHandle(), &m_device->GetAPI() );

        // 3. Define Deleter
        // This lambda encapsulates how to destroy the C++ object (destructor + free)
        auto deleter = [ allocator ]( Shader* ptr ) {
            if( ptr )
            {
                ptr->Destroy();         // Ensure Vulkan resources are released
                ptr->~Shader();         // Call destructor manually
                allocator->Free( ptr ); // Free memory
            }
        };

        // 4. Initialize Vulkan Resource via Device
        if( m_device->CreateShader( filepath, rawShader ) != Result::SUCCESS )
        {
            DT_ERROR( "Failed to initialize Vulkan Shader!" );
            deleter( rawShader ); // Cleanup if init fails
            return ShaderHandle::Invalid;
        }

        // 5. Wrap in unique_ptr and Insert into Pool
        std::unique_ptr<Shader, ShaderDeleter> managedShader( rawShader, deleter );

        return m_shaders.Insert( std::move( managedShader ) );
    }

    Shader* ResourceManager::GetShader( ShaderHandle handle )
    {
        return m_shaders.Get( handle );
    }

    void ResourceManager::DestroyShader( ShaderHandle handle )
    {
        auto shaderPtr = m_shaders.Remove( handle );
        EnqueueDeletion( std::move( shaderPtr ) );
    }

    ComputePipelineHandle ResourceManager::CreatePipeline( const ComputePipelineDesc& desc )
    {
        // Resolve ShaderHandle to raw pointer for creation
        Shader* shader = GetShader( desc.shader );
        if( !shader )
        {
            DT_ERROR( "Invalid shader handle in CreateComputePipeline" );
            return ComputePipelineHandle::Invalid;
        }

        ComputePipelineNativeDesc nativeDesc;
        nativeDesc.shader = shader;

        // 1. Allocate Raw Memory (Tracked)
        IAllocator* allocator = m_memorySystem->GetSystemAllocator();
        void*       mem       = allocator->Allocate( sizeof( ComputePipeline ), __FILE__, __LINE__ );

        if( !mem )
        {
            DT_ERROR( "Failed to allocate memory for Shader!" );
            return ComputePipelineHandle::Invalid;
        }

        // 2. Construct Object (Placement New)
        // Pass device dependencies needed for internal logic
        ComputePipeline* rawPipeline = new( mem ) ComputePipeline( m_device->GetHandle(), &m_device->GetAPI() );

        // 3. Define Deleter
        // This lambda encapsulates how to destroy the C++ object (destructor + free)
        auto deleter = [ allocator ]( ComputePipeline* ptr ) {
            if( ptr )
            {
                ptr->Destroy();          // Ensure Vulkan resources are released
                ptr->~ComputePipeline(); // Call destructor manually
                allocator->Free( ptr );  // Free memory
            }
        };

        // 4. Initialize Vulkan Resource via Device
        if( m_device->CreateComputePipeline( nativeDesc, rawPipeline ) != Result::SUCCESS )
        {
            DT_ERROR( "Failed to initialize Vulkan Compute pipeline!" );
            deleter( rawPipeline ); // Cleanup if init fails
            return ComputePipelineHandle::Invalid;
        }

        // 5. Wrap in unique_ptr and Insert into Pool
        std::unique_ptr<ComputePipeline, ComputePipelineDeleter> managedPipeline( rawPipeline, deleter );

        return m_computePipelines.Insert( std::move( managedPipeline ) );
    }

    GraphicsPipelineHandle ResourceManager::CreatePipeline( const GraphicsPipelineDesc& desc )
    {
        Shader* vs = GetShader( desc.vertexShader );
        Shader* fs = GetShader( desc.fragmentShader );

        if( !vs || !fs )
            return GraphicsPipelineHandle::Invalid;

        GraphicsPipelineNativeDesc nativeDesc;
        nativeDesc.vertexShader           = vs;
        nativeDesc.fragmentShader         = fs;
        nativeDesc.topology               = desc.topology;
        nativeDesc.polygonMode            = desc.polygonMode;
        nativeDesc.cullMode               = desc.cullMode;
        nativeDesc.frontFace              = desc.frontFace;
        nativeDesc.lineWidth              = desc.lineWidth;
        nativeDesc.depthTestEnable        = desc.depthTestEnable;
        nativeDesc.depthWriteEnable       = desc.depthWriteEnable;
        nativeDesc.depthCompareOp         = desc.depthCompareOp;
        nativeDesc.blendEnable            = desc.blendEnable;
        nativeDesc.srcColorBlendFactor    = desc.srcColorBlendFactor;
        nativeDesc.dstColorBlendFactor    = desc.dstColorBlendFactor;
        nativeDesc.colorBlendOp           = desc.colorBlendOp;
        nativeDesc.srcAlphaBlendFactor    = desc.srcAlphaBlendFactor;
        nativeDesc.dstAlphaBlendFactor    = desc.dstAlphaBlendFactor;
        nativeDesc.alphaBlendOp           = desc.alphaBlendOp;
        nativeDesc.colorAttachmentFormats = desc.colorAttachmentFormats;
        nativeDesc.depthAttachmentFormat  = desc.depthAttachmentFormat;

        // 1. Allocate Raw Memory (Tracked)
        IAllocator* allocator = m_memorySystem->GetSystemAllocator();
        void*       mem       = allocator->Allocate( sizeof( GraphicsPipeline ), __FILE__, __LINE__ );

        if( !mem )
        {
            DT_ERROR( "Failed to allocate memory for Shader!" );
            return GraphicsPipelineHandle::Invalid;
        }

        // 2. Construct Object (Placement New)
        // Pass device dependencies needed for internal logic
        GraphicsPipeline* rawPipeline = new( mem ) GraphicsPipeline( m_device->GetHandle(), &m_device->GetAPI() );

        // 3. Define Deleter
        // This lambda encapsulates how to destroy the C++ object (destructor + free)
        auto deleter = [ allocator ]( GraphicsPipeline* ptr ) {
            if( ptr )
            {
                ptr->Destroy();           // Ensure Vulkan resources are released
                ptr->~GraphicsPipeline(); // Call destructor manually
                allocator->Free( ptr );   // Free memory
            }
        };

        // 4. Initialize Vulkan Resource via Device
        if( m_device->CreateGraphicsPipeline( nativeDesc, rawPipeline ) != Result::SUCCESS )
        {
            DT_ERROR( "Failed to initialize Vulkan Graphics pipeline!" );
            deleter( rawPipeline ); // Cleanup if init fails
            return GraphicsPipelineHandle::Invalid;
        }

        // 5. Wrap in unique_ptr and Insert into Pool
        std::unique_ptr<GraphicsPipeline, GraphicsPipelineDeleter> managedPipeline( rawPipeline, deleter );

        return m_graphicsPipelines.Insert( std::move( managedPipeline ) );
    }

    ComputePipeline* ResourceManager::GetPipeline( ComputePipelineHandle handle )
    {
        return m_computePipelines.Get( handle );
    }

    GraphicsPipeline* ResourceManager::GetPipeline( GraphicsPipelineHandle handle )
    {
        return m_graphicsPipelines.Get( handle );
    }

    void ResourceManager::DestroyPipeline( ComputePipelineHandle handle )
    {
        auto pipelinePtr = m_computePipelines.Remove( handle );
        EnqueueDeletion( std::move( pipelinePtr ) );
    }

    void ResourceManager::DestroyPipeline( GraphicsPipelineHandle handle )
    {
        auto pipelinePtr = m_graphicsPipelines.Remove( handle );
        EnqueueDeletion( std::move( pipelinePtr ) );
    }

} // namespace DigitalTwin