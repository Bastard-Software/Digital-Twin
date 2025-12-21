#pragma once
#include "core/Base.hpp"
#include "platform/Window.hpp"
#include "resources/ResourceManager.hpp"
#include "resources/StreamingManager.hpp"
#include "rhi/Device.hpp"
#include <memory>

namespace DigitalTwin
{
    class Simulation; // Forward declaration

    struct EngineConfig
    {
        bool_t   headless = true; // Default to true (Compute/Headless) as per tests
        uint32_t width    = 1280;
        uint32_t height   = 720;
    };

    /**
     * @brief Main runtime class responsible for system initialization and the game loop.
     * Manages the lifetime of RHI Device, Window, and StreamingManager.
     */
    class Engine
    {
    public:
        Engine();
        ~Engine();

        /**
         * @brief Initializes the engine, RHI, and core subsystems.
         * @param config Configuration parameters (window size, mode).
         * @return Result::SUCCESS if initialization was successful.
         */
        Result Init( const EngineConfig& config = EngineConfig() );

        /**
         * @brief Shuts down all subsystems and releases GPU resources.
         */
        void Shutdown();

        /**
         * @brief Handles start-of-frame housekeeping (frame counter, descriptor resets)
         */
        void BeginFrame();

        /**
         * @brief Blocks CPU execution until the GPU has finished all currently submitted commands.
         * Useful before destroying resources to ensure they are not in use.
         */
        void WaitIdle();

        // --- Accessors ---
        Ref<Device>           GetDevice() const { return m_device; }
        Ref<StreamingManager> GetStreamingManager() const { return m_streamingManager; }
        Ref<ResourceManager>  GetResourceManager() const { return m_resourceManager; }
        Window*               GetWindow() const { return m_window.get(); }
        const EngineConfig&   GetConfig() const { return m_config; }

        // Added: Accessor for the current frame number (needed for StreamingManager synchronization)
        uint64_t GetFrameCount() const { return m_frameCounter; }

        // Status checks
        bool IsHeadless() const { return m_config.headless; }
        bool IsInitialized() const { return m_initialized; }

    private:
        Ref<Device>           m_device;
        Ref<StreamingManager> m_streamingManager;
        Ref<ResourceManager>  m_resourceManager;
        Scope<Window>         m_window;
        EngineConfig          m_config;

        bool     m_initialized  = false;
        uint64_t m_frameCounter = 0;
    };
} // namespace DigitalTwin