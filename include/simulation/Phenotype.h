#pragma once
#include <cstdint>

namespace DigitalTwin
{
    /**
     * @brief Represents the biological state machine of a cell.
     * These values are mirrored exactly in the GPU compute shaders.
     */
    enum class PhenotypeState : uint32_t
    {
        Live      = 0, // Actively growing, consuming nutrients, and preparing to divide
        Quiescent = 1, // Cycle arrested (e.g., due to high mechanical pressure / contact inhibition)
        Hypoxic   = 2, // Not growing, asking for help
        Apoptotic = 3, // Programmed cell death (shrinking, safe removal)
        Necrotic  = 4, // Ruptured death due to starvation/hypoxia (spills toxins)

        // Internal state used by the GPU during Stream Compaction to mark cells for deletion
        Dead_PendingRemoval = 5
    };

} // namespace DigitalTwin