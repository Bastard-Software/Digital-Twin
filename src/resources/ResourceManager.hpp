#pragma once
#include "core/Base.hpp"
#include "resources/GPUMesh.hpp"
#include "resources/Mesh.hpp"
#include "resources/StreamingManager.hpp"
#include "rhi/Device.hpp"
#include <functional>
#include <queue>
#include <string>
#include <unordered_map>

namespace DigitalTwin
{
    /**
     * @brief Central hub for GPU resources.
     * Manages creation, caching, and async upload of meshes into Unified Buffers.
     */
    class ResourceManager
    {
    public:
        ResourceManager( Ref<Device> device, Ref<StreamingManager> streamer );
        ~ResourceManager();

        void Shutdown();

        /**
         * @brief Gets or creates a mesh and returns its ID.
         */
        AssetID GetMeshID( const std::string& name );

        /**
         * @brief Retrieves the GPU resource by ID. Returns nullptr if invalid.
         */
        Ref<GPUMesh> GetMesh( AssetID id );

        void      BeginFrame( uint64_t frameNumber );
        SyncPoint EndFrame();

    private:
        /**
         * @brief Internal helper.
         * Allocates ONE buffer, calculates offsets, and queues upload tasks.
         */
        Ref<GPUMesh> CreateGPUMesh( const Mesh& data );

        void ProcessPendingUploads();

    private:
        Ref<Device>                       m_device;
        Ref<StreamingManager>             m_streamer;
        std::queue<std::function<void()>> m_uploadQueue;

        // Maps Name -> ID (e.g. "Sphere" -> 1)
        std::unordered_map<std::string, AssetID> m_nameToId;

        // Maps ID -> Resource
        std::unordered_map<AssetID, Ref<GPUMesh>> m_meshes;

        // Counter for generating new IDs
        AssetID m_nextID = 1;
    };
} // namespace DigitalTwin