#pragma once

#include "compute/ComputeEngine.hpp"
#include "compute/ComputeGraph.hpp"
#include "core/Base.hpp"
#include "renderer/Renderer.hpp"
#include "runtime/Engine.hpp"
#include "simulation/Simulation.hpp"
#include <memory>
#include <string>

namespace DigitalTwin
{
    /**
     * @brief Configuration for the application window and runtime mode.
     */
    struct AppConfig
    {
        std::string windowTitle = "Digital Twin Simulation";
        uint32_t    width       = 1280;
        uint32_t    height      = 720;
        bool        headless    = false;
    };

    /**
     * @brief Base class for all user experiments.
     * Manages the Engine, Simulation, Renderer, and the main loop.
     */
    class Application
    {
    public:
        Application( const AppConfig& config = AppConfig() );
        virtual ~Application();

        /**
         * @brief Starts the main loop.
         * Initializes the engine, calls OnSetup(), and runs until the window is closed.
         */
        void Run();

        /**
         * @brief Stops the application at the end of the current frame.
         */
        void Close();

        // --- Virtuals to be implemented by the User ---

        /**
         * @brief Phase 1: Define the Simulation World.
         * Use this to SpawnCell(), set Gravity, Viscosity etc.
         * Note: GPU buffers are NOT created yet.s
         */
        virtual void OnConfigureWorld() = 0;

        /**
         * @brief Phase 2: Define Compute Logic.
         * Use this to create Shaders, Pipelines, and bind Buffers.
         * Note: GPU buffers ARE created and ready to be bound.
         */
        virtual void OnConfigurePhysics() = 0;

        /**
         * @brief Called every frame to render UI overlay (if GUI layer is active).
         */
        virtual void OnGui() {}

    protected:
        // --- Protected API for User Subclasses ---

        Simulation&      GetSimulation() { return *m_simulation; }
        ResourceManager& GetResourceManager() { return *m_engine->GetResourceManager(); }
        const AppConfig& GetConfig() const { return m_config; }

        // Accessors needed to build the Compute Graph in OnSetup()
        ComputeGraph& GetComputeGraph() { return m_graph; }
        Ref<Device>   GetDevice() { return m_engine->GetDevice(); }

    private:
        void InitCore();

    private:
        AppConfig m_config;
        bool      m_running = true;

        // Core Systems
        Scope<Engine>      m_engine;
        Scope<Simulation>  m_simulation;
        Scope<Renderer>    m_renderer;
        Ref<ComputeEngine> m_computeEngine;

        // The compute graph executing physics
        ComputeGraph m_graph;
    };

    // To be defined in client application
    Application* CreateApplication();

} // namespace DigitalTwin