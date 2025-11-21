#pragma once
#include <cstdint>
#include <stdio.h>
#include <entt/entt.hpp>

namespace DigitalTwin
{
    using bool_t    = bool;
    using float32_t = float;
    using float64_t = double;

    struct SimulationConfig
    {
        uint32_t maxSteps = 100;
    };

    class Simulation
    {
    public:
        Simulation();
        ~Simulation();

        void Initialize();
        void Step();

        bool_t   IsComplete() const;
        uint32_t GetCurrentStep() const { return m_currentStep; }

    private:
        uint32_t         m_currentStep;
        SimulationConfig m_config;
        entt::registry   m_registry;
    };
} // namespace DigitalTwin