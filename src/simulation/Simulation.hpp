#pragma once
#include "simulation/SimulationContext.hpp"
#include "simulation/Types.hpp"
#include "resources/GPUMesh.hpp"
#include <glm/glm.hpp>
#include <memory>
#include <vector>

namespace DigitalTwin
{
    class Engine; // Forward declaration

    /**
     * @brief High-level API for defining and controlling biological simulations.
     * Decoupled from low-level rendering/compute details.
     */
    class Simulation
    {
    public:
        // Simulation needs the Engine to access GPU resources
        Simulation( Engine& engine );
        ~Simulation();

        // --- Biological API ---

        /**
         * @brief Sets global microenvironment parameters.
         */
        void SetMicroenvironment( float viscosity, float gravity );

        /**
         * @brief Spawns a new cell. Added to CPU staging list until InitializeGPU() is called.
         */
        void SpawnCell( glm::vec4 position, glm::vec3 velocity, glm::vec4 phenotypeColor );

        /**
         * @brief Finalizes configuration, allocates GPU memory, and uploads initial state.
         * MUST be called before Engine::Run().
         */
        void InitializeGPU();

        // --- Internal System API (Called by Engine) ---

        void Step( float dt );

        // Expose context for Renderer/Compute systems
        SimulationContext* GetContext() { return m_context.get(); }
        Ref<GPUMesh>       GetCellMesh() const { return m_cellMesh; }

    private:
        Engine&                            m_engine;
        std::unique_ptr<SimulationContext> m_context;
        Ref<GPUMesh>                       m_cellMesh;

        // Staging data on CPU (initial configuration)
        std::vector<Cell> m_initialCells;
        EnvironmentParams m_envParams{};
    };
} // namespace DigitalTwin