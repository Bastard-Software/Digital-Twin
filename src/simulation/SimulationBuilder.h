#pragma once
#include "simulation/SimulationState.h"

namespace DigitalTwin
{
    class SimulationBlueprint;
    class ResourceManager;
    class StreamingManager;

    /**
     * @brief Compiles a CPU-side SimulationBlueprint into a highly optimized,
     * GPU-ready SimulationState (Data-Oriented mega-buffers).
     */
    class SimulationBuilder
    {
    public:
        SimulationBuilder( ResourceManager* resourceManager, StreamingManager* streamingManager );

        /**
         * @brief Allocates and uploads all necessary buffers to the GPU.
         * @param blueprint The declarative recipe of the simulation.
         * @return A valid SimulationState on success, or an invalid state on failure.
         */
        SimulationState Build( const SimulationBlueprint& blueprint );

        /**
         * @brief Updates behaviour push constants on a live simulation without rebuilding GPU buffers.
         * Only call while PLAYING or PAUSED. Structural parameters (offsets, buffer sizes) are unchanged.
         */
        void UpdateParameters( const SimulationBlueprint& blueprint, SimulationState& state );

    private:
        void AllocateAgentBuffers( const SimulationBlueprint& blueprint, SimulationState& outState );
        void CompileSpatialGrid( const SimulationBlueprint& blueprint, SimulationState& outState );
        void CompileGridFields( const SimulationBlueprint& blueprint, SimulationState& outState );
        void CompileBehaviours( const SimulationBlueprint& blueprint, SimulationState& outState );
        // Phase 2.6.5.b — allocates PolygonBuffer and registers the per-cell
        // Voronoi compute pass. Runs after behaviour tasks so vertices reflect
        // settled agent positions.
        void CompileVoronoiPolygon( const SimulationBlueprint& blueprint, SimulationState& outState );

    private:
        ResourceManager*  m_resourceManager;
        StreamingManager* m_streamingManager;
    };

} // namespace DigitalTwin