#pragma once

#include "compute/ComputeEngine.hpp"
#include "core/Base.hpp"
#include "renderer/Renderer.hpp"
#include "runtime/Engine.hpp"
#include "simulation/Simulation.hpp"
#include <memory>
#include <string>

namespace DigitalTwin
{
    struct AppConfig
    {
        std::string windowTitle = "Digital Twin Simulation";
        uint32_t    width       = 1280;
        uint32_t    height      = 720;
        bool        headless    = false;
    };

    /**
     * @brief The Application Host.
     * Responsible for initializing the engine and driving the User's Simulation.
     */
    class Application
    {
    public:
        /**
         * @brief Constructs the Application with a User Simulation instance.
         * @param userSimulation Pointer to the simulation created by the user (takes ownership or manages ref).
         */
        Application( Simulation* userSimulation, const AppConfig& config = AppConfig() );
        ~Application();

        void Run();
        void Close();

    private:
        void InitCore();
        void Render();

    private:
        AppConfig m_config;
        bool      m_running = true;

        Scope<Engine>      m_engine;
        Scope<Renderer>    m_renderer;
        Ref<ComputeEngine> m_computeEngine;

        // Pointer to the user's experiment
        Simulation* m_simulation = nullptr;
    };
} // namespace DigitalTwin