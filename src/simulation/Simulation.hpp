#pragma once
#include "compute/ComputeGraph.hpp"
#include "core/Base.hpp"
#include "simulation/Types.hpp"
#include <array>
#include <functional>
#include <string>
#include <vector>

namespace DigitalTwin
{
    class Application;
    class Engine;
    class ComputeEngine;
    class SimulationContext;

    /**
     * @brief Recipe for building a graph. Called twice for Double Buffering.
     */
    using GraphBuilder = std::function<ComputeGraph( SimulationContext& ctx )>;

    class Simulation
    {
    public:
        Simulation();
        virtual ~Simulation();

        // --- USER API ---
        virtual void OnConfigureWorld()   = 0;
        virtual void OnConfigureSystems() = 0;
        virtual void OnUpdate( float dt ) {}
        virtual void OnRenderGui() {}

        // --- Control API ---
        void  SetTimeScale( float scale );
        float GetTimeScale() const;
        void  Pause();
        void  Resume();

    protected:
        /**
         * @brief Registers a system using the Builder pattern (Lambda).
         * Handles Double Buffering automatically.
         */
        void RegisterSystem( const std::string& name, GraphBuilder builder, float interval );

        // Helpers
        void SetMicroenvironment( float viscosity, float gravity );
        void SpawnCell( uint32_t meshID, glm::vec4 pos, glm::vec3 vel, glm::vec4 color );

        SimulationContext* GetContext();
        Ref<Buffer>        GetGlobalUniformBuffer();
        Ref<Device>        GetDevice();
        AssetID            GetMeshID( const std::string& name ) const;

        friend class Application;

    protected:
        // Internal Engine API
        void                         InitializeRuntime( Engine& engine, Ref<ComputeEngine> computeEngine );
        void                         ShutdownRuntime();
        void                         Tick( float realDt );
        const std::vector<uint32_t>& GetActiveMeshes() const;
        uint64_t                     GetComputeSignal() const;

    private:
        void UpdateGlobalData( float dt );

    private:
        struct SystemInstance
        {
            std::string name;
            float       interval;
            float       timer = 0.0f;

            // Double Buffering: Index 0 (Frame 0), Index 1 (Frame 1)
            std::array<ComputeGraph, 2> graphs;
        };

        Engine*                m_engineRef = nullptr;
        Ref<ComputeEngine>     m_computeEngine;
        Ref<SimulationContext> m_context;
        Ref<Buffer>            m_globalUBO;

        std::vector<SystemInstance> m_systems;
        std::vector<uint32_t>       m_activeMeshes;
        std::vector<Cell>           m_initialCells;
        EnvironmentParams           m_envParams{};

        uint64_t m_lastComputeSignal = 0;
        bool     m_paused            = false;
        float    m_timeScale         = 1.0f;
        float    m_totalTime         = 0.0f;
    };

    extern Simulation* CreateSimulation();
} // namespace DigitalTwin