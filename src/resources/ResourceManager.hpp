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
         * @brief Requests a mesh by name.
         * Returns a cached instance or creates a new one.
         */
        Ref<GPUMesh> GetMesh( const std::string& name );

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
        Ref<Device>           m_device;
        Ref<StreamingManager> m_streamer;

        std::unordered_map<std::string, Ref<GPUMesh>> m_meshCache;
        std::queue<std::function<void()>>             m_uploadQueue;
    };
} // namespace DigitalTwin