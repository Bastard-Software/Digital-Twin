#pragma once
#include "compute/ComputeEngine.hpp"
#include "core/Base.hpp"
#include "core/TimeController.hpp"
#include "rhi/Buffer.hpp"
#include "simulation/SimulationContext.hpp"
#include <string>
#include <vector>

namespace DigitalTwin
{
    /**
     * @brief Structure matching the layout of the Global Uniform Buffer (std140).
     * This data is available in all compute shaders at Set 0, Binding 0.
     */
    struct GlobalContextData
    {
        float    time;      // Total simulation time
        float    dt;        // Delta time for the CURRENT execution step
        float    timeScale; // Current time scale multiplier
        uint32_t frame;     // Global simulation frame counter

        float worldSize;    // Boundary size (e.g., 20.0)
        float padding[ 3 ]; // Padding for 16-byte alignment (std140)
    };

    /**
     * @brief Represents a single simulation system (e.g., Physics, Biology).
     */
    struct SimulationPass
    {
        std::string  name;        // Debug name of the system
        ComputeGraph graph;       // The compute workload
        float        interval;    // Fixed time step (e.g., 0.016s for 60Hz)
        float        accumulator; // Internal time accumulator
        bool         enabled = true;
    };

    /**
     * @brief Orchestrates the execution of simulation systems based on time accumulation.
     * Manages the Global Uniform Buffer and dispatches Compute Graphs.
     */
    class SimulationScheduler
    {
    public:
        SimulationScheduler( Ref<ComputeEngine> engine, Ref<SimulationContext> context );

        /**
         * @brief Registers a new system to be executed periodically.
         */
        void AddSystem( const std::string& name, ComputeGraph graph, float interval );

        /**
         * @brief Advances the simulation by the specified real delta time.
         * Executes systems if their time accumulator exceeds their interval.
         */
        void Tick( float realDt );

        Ref<Buffer>     GetGlobalBuffer() { return m_globalUBO; }
        TimeController& GetTimeController() { return m_timeCtrl; }
        uint64_t        GetLastComputeSignal() const { return m_lastComputeSignal; }

    private:
        /**
         * @brief Uploads new time data to the GPU using Buffer::Write.
         */
        void UpdateGlobalUBO( float currentStepDt );

    private:
        Ref<ComputeEngine>          m_computeEngine;
        Ref<SimulationContext>      m_context;
        Ref<Buffer>                 m_globalUBO;
        TimeController              m_timeCtrl;
        std::vector<SimulationPass> m_passes;
        uint64_t                    m_lastComputeSignal = 0;
    };
} // namespace DigitalTwin