#pragma once
#include "compute/ComputeEngine.hpp"
#include "core/Base.hpp"
#include "renderer/Renderer.hpp"
#include "runtime/Engine.hpp"
#include "simulation/Simulation.hpp"

namespace DigitalTwin
{
    struct AppConfig
    {
        uint32_t width    = 1280;
        uint32_t height   = 720;
        bool     headless = false;
    };

    class Application
    {
    public:
        Application( Simulation* userSimulation, const AppConfig& config );
        ~Application();

        void InitCore();
        void Run();

    private:
        AppConfig m_config;
        bool      m_running = false;

        Scope<Engine>      m_engine;
        Ref<ComputeEngine> m_computeEngine;
        Scope<Renderer>    m_renderer;

        Simulation* m_simulation = nullptr;
    };
} // namespace DigitalTwin