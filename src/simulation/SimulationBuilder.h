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

    private:
        void CompileBehaviours( const SimulationBlueprint& blueprint, SimulationState& outState );

    private:
        ResourceManager*  m_resourceManager;
        StreamingManager* m_streamingManager;
    };

} // namespace DigitalTwin