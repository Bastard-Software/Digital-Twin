#pragma once
#include "core/Base.hpp"
#include "platform/Window.hpp"
#include "rhi/Device.hpp"
#include "streaming/StreamingManager.hpp"
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
         * @brief Starts the main loop. Blocks until the window is closed or simulation ends.
         * @param simulation The simulation instance to update every frame.
         */
        void Run( Simulation& simulation );

        // --- Accessors ---
        Ref<Device>           GetDevice() const { return m_device; }
        Ref<StreamingManager> GetStreamingManager() const { return m_streamingManager; }
        Window*               GetWindow() const { return m_window.get(); }
        const EngineConfig&   GetConfig() const { return m_config; }

        // Status checks
        bool IsHeadless() const { return m_config.headless; }
        bool IsInitialized() const { return m_initialized; }

    private:
        Ref<Device>           m_device;
        Ref<StreamingManager> m_streamingManager;
        Scope<Window>         m_window;
        EngineConfig          m_config;

        bool     m_initialized  = false;
        uint64_t m_frameCounter = 0;
    };
} // namespace DigitalTwin