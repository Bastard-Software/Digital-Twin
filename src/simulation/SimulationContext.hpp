#pragma once
#include "core/Base.hpp"
#include "resources/StreamingManager.hpp"
#include "rhi/Buffer.hpp"
#include "rhi/Device.hpp"
#include "simulation/Types.hpp"
#include <vector>

namespace DigitalTwin
{
    /**
     * @brief Backend container for simulation data on GPU.
     * Manages SSBOs and atomic counters for the agent population.
     */
    class SimulationContext
    {
    public:
        SimulationContext( Ref<Device> device );
        ~SimulationContext();

        /**
         * @brief Allocates GPU buffers (SSBO + Atomic Counter).
         * @param maxCells Maximum capacity of the simulation (for buffer allocation).
         */
        void Init( uint32_t maxCells );
        void Shutdown();

        /**
         * @brief Uploads the initial state of cells and resets the atomic counter.
         * @param streamer StreamingManager for async transfer.
         * @param cells Initial vector of cells (must be <= maxCells).
         */
        void UploadState( StreamingManager* streamer, const std::vector<Cell>& cells );

        // --- Getters ---

        // Main data buffer (Array of Structs: Cell[])
        Ref<Buffer> GetCellBuffer() const { return m_cellBuffer; }

        // Atomic counter containing the number of active agents (uint32_t)
        // Used by shaders to append new agents or clamp threads.
        Ref<Buffer> GetCounterBuffer() const { return m_atomicCounter; }

        uint32_t GetMaxCellCount() const { return m_maxCellCount; }

    private:
        Ref<Device> m_device;

        Ref<Buffer> m_cellBuffer;
        Ref<Buffer> m_atomicCounter;

        uint32_t m_maxCellCount = 0;
    };
} // namespace DigitalTwin