#pragma once
#include "Types.hpp"
#include "Config.hpp"

namespace DigitalTwin
{
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
        uint32_t m_currentStep;
        SimulationConfig m_config;
    };
} // namespace DigitalTwin
