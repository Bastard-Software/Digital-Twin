#pragma once
#include "core/Base.hpp"
#include "platform/Window.hpp"
#include "resources/ResourceManager.hpp"
#include "resources/StreamingManager.hpp"
#include "rhi/Device.hpp"
#include <memory>

namespace DigitalTwin
{
    class Simulation;
    class Renderer;

    struct EngineConfig
    {
        uint32_t width;
        uint32_t height;
        bool     headless;
    };

    /**
     * @brief Main Engine Root.
     * Manages Device, Window and Resource Managers.
     */
    class Engine
    {
    public:
        Engine();
        ~Engine();

        Result Init( const EngineConfig& config );
        void   Shutdown();

        // --- Frame Orchestration ---
        void BeginFrame();
        void EndFrame( Simulation& simulation, Renderer& renderer );

        // --- Accessors ---
        Ref<Device>           GetDevice() const { return m_device; }
        Ref<Window>           GetWindow() const { return m_window; }
        Ref<ResourceManager>  GetResourceManager() const { return m_resourceManager; }
        Ref<StreamingManager> GetStreamingManager() const { return m_streamingManager; }

        // --- Helpers ---
        void WaitIdle();
        void PollEvents();
        bool IsInitialized() const { return m_initialized; }
        bool IsHeadless() const { return m_config.headless; }

    private:
        bool         m_initialized = false;
        EngineConfig m_config;

        Ref<Device>           m_device;
        Ref<Window>           m_window;
        Ref<ResourceManager>  m_resourceManager;
        Ref<StreamingManager> m_streamingManager;

        uint64_t m_frameCounter = 0;
    };
} // namespace DigitalTwin