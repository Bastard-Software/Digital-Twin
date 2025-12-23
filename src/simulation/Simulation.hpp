#pragma once
#include "compute/ComputeGraph.hpp"
#include "core/Base.hpp"
#include "simulation/Types.hpp"
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace DigitalTwin
{

    class Application;
    class Engine;
    class ComputeEngine;
    class SimulationContext;
    class SimulationScheduler;

    /**
     * @brief Base class for User Experiments.
     * Users inherit from this class to define their simulation logic.
     */
    class Simulation
    {
    public:
        Simulation();
        virtual ~Simulation();

        // --- USER API (Virtuals) ---

        /**
         * @brief Phase 1: Setup world parameters and spawn initial cells.
         * Called once before GPU initialization.
         */
        virtual void OnConfigureWorld() = 0;

        /**
         * @brief Phase 2: Define Compute Logic.
         * Register systems and graphs here. Called after GPU init.
         */
        virtual void OnConfigureSystems() = 0;

        /**
         * @brief Optional: Called every frame (CPU side).
         * Use this for custom stop conditions, logging, or interacting with the scheduler.
         * @param realDt Real elapsed time in seconds.
         */
        virtual void OnUpdate( float realDt ) {}

        /**
         * @brief Optional: Called during UI render pass.
         * Use ImGui calls here to add custom sliders/buttons.
         */
        virtual void OnRenderGui() {}

        // --- USER API (Actions) ---
    protected:
        void SetMicroenvironment( float viscosity, float gravity );
        void SpawnCell( uint32_t meshID, glm::vec4 pos, glm::vec3 vel, glm::vec4 color );

        /**
         * @brief Registers a Compute Graph to be executed at a fixed interval.
         * @param name Debug name of the system.
         * @param graph The compute graph (kernels + bindings).
         * @param interval Simulation time interval (e.g., 0.016 for 60Hz physics).
         */
        void RegisterSystem( const std::string& name, ComputeGraph graph, float interval );

        // Control API
        void  SetTimeScale( float scale );
        float GetTimeScale() const;
        void  Pause();
        void  Resume();

        // Accessors for building graphs
        SimulationContext* GetContext();
        // Helper to get Global UBO for shader binding
        Ref<Buffer> GetGlobalUniformBuffer();

        friend class Application;

        // --- INTERNAL ENGINE API (Hidden from common usage) ---
    protected:
        void                         InitializeRuntime( Engine& engine, Ref<ComputeEngine> computeEngine );
        void                         Tick( float realDt ); // Called by Application
        const std::vector<uint32_t>& GetActiveMeshes() const;
        uint64_t                     GetComputeSignal() const;
        AssetID                      GetMeshID( const std::string& name ) const;

    private:
        // Pimpl dependencies to keep header clean
        Engine*                    m_engineRef = nullptr;
        Ref<ComputeEngine>         m_computeEngine;
        Ref<SimulationContext>     m_context;
        Scope<SimulationScheduler> m_scheduler;

        std::vector<uint32_t> m_activeMeshes;

        // Staging data
        std::vector<Cell> m_initialCells;
        EnvironmentParams m_envParams{};
    };

    // --- FACTORY ---
    // User must implement this in their main.cpp
    extern Simulation* CreateSimulation();
} // namespace DigitalTwin