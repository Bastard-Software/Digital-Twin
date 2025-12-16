#include "resources/ResourceManager.hpp"

#include "core/Log.hpp"
#include "resources/ShapeGenerator.hpp"

namespace DigitalTwin
{
    ResourceManager::ResourceManager( Ref<Device> device, Ref<StreamingManager> streamer )
        : m_device( device )
        , m_streamer( streamer )
    {
    }

    ResourceManager::~ResourceManager()
    {
        Shutdown();
    }

    void ResourceManager::Shutdown()
    {
        m_meshCache.clear();
        while( !m_uploadQueue.empty() )
            m_uploadQueue.pop();
    }

    Ref<GPUMesh> ResourceManager::GetMesh( const std::string& name )
    {
        // 1. Cache Check
        auto it = m_meshCache.find( name );
        if( it != m_meshCache.end() )
        {
            return it->second;
        }

        // 2. Generate CPU Data
        DT_CORE_INFO( "[Resources] Generating mesh: '{}'", name );
        Mesh data;

        if( name == "Cube" )
            data = ShapeGenerator::CreateCube();
        else if( name == "Sphere" )
            data = ShapeGenerator::CreateSphere( 0.5f, 32, 32 );
        else
        {
            DT_CORE_WARN( "Unknown mesh '{}', defaulting to Cube.", name );
            data = ShapeGenerator::CreateCube();
        }

        // 3. Create Unified GPU Mesh
        auto mesh           = CreateGPUMesh( data );
        m_meshCache[ name ] = mesh;

        return mesh;
    }

    Ref<GPUMesh> ResourceManager::CreateGPUMesh( const Mesh& data )
    {
        // Calculate sizes
        VkDeviceSize vSize = data.vertices.size() * sizeof( Vertex );
        VkDeviceSize iSize = data.indices.size() * sizeof( uint32_t );

        // Vertices are vec4-based (16-byte aligned), so vSize is always a multiple of 16.
        // This is safe for uint32_t indices which require 4-byte alignment.
        // If we had custom vertex formats, we might need padding here.
        VkDeviceSize totalSize = vSize + iSize;

        // 1. Allocate ONE buffer for everything
        BufferDesc desc{};
        desc.size = totalSize;
        desc.type = BufferType::MESH; // Uses STORAGE | INDEX | TRANSFER_DST

        auto mergedBuffer = m_device->CreateBuffer( desc );
        if( !mergedBuffer )
        {
            DT_CORE_CRITICAL( "Failed to allocate mesh buffer!" );
            return nullptr;
        }

        // 2. Defer Upload
        // We capture data by value.
        m_uploadQueue.push( [ =, vertices = data.vertices, indices = data.indices ]() {
            if( mergedBuffer )
            {
                // A. Upload Vertices at Offset 0
                m_streamer->UploadToBuffer( mergedBuffer, vertices.data(), vSize, 0 );

                // B. Upload Indices at Offset vSize
                if( iSize > 0 )
                {
                    m_streamer->UploadToBuffer( mergedBuffer, indices.data(), iSize, vSize );
                }
            }
            DT_CORE_TRACE( "[Resources] Uploaded merged mesh data (Size: {} bytes).", totalSize );
        } );

        // 3. Return wrapper
        // The GPUMesh needs to know where the indices start (vSize) to bind correctly later.
        return CreateRef<GPUMesh>( mergedBuffer, vSize, ( uint32_t )data.indices.size() );
    }

    void ResourceManager::BeginFrame( uint64_t frameNumber )
    {
        if( m_streamer )
        {
            m_streamer->BeginFrame( frameNumber );
            ProcessPendingUploads();
        }
    }

    void ResourceManager::ProcessPendingUploads()
    {
        if( m_uploadQueue.empty() )
            return;

        // Execute all pending transfers
        while( !m_uploadQueue.empty() )
        {
            auto& task = m_uploadQueue.front();
            task();
            m_uploadQueue.pop();
        }
    }

    SyncPoint ResourceManager::EndFrame()
    {
        if( m_streamer )
            return m_streamer->EndFrame();
        return {};
    }

} // namespace DigitalTwin