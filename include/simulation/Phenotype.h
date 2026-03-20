#pragma once
#include <cstdint>

namespace DigitalTwin
{
    /**
     * @brief Represents the lifecycle state of a cell.
     * These values are mirrored exactly in the GPU compute shaders.
     */
    enum class LifecycleState : uint32_t
    {
        Live      = 0, // Actively growing, consuming nutrients, and preparing to divide
        Quiescent = 1, // Cycle arrested (e.g., due to high mechanical pressure / contact inhibition)
        Hypoxic   = 2, // Not growing, asking for help
        Apoptotic = 3, // Programmed cell death (shrinking, safe removal)
        Necrotic  = 4, // Ruptured death due to starvation/hypoxia (spills toxins)

        // Internal state used by the GPU during Stream Compaction to mark cells for deletion
        Dead_PendingRemoval = 5
    };

    /**
     * @brief Represents the functional cell type, orthogonal to LifecycleState.
     * A TipCell can be Live, Hypoxic, or Necrotic independently.
     */
    enum class CellType : uint32_t
    {
        Default     = 0, // Generic cell (tumor cell, etc.)
        TipCell     = 1, // Endothelial tip cell — migrates toward VEGF
        StalkCell   = 2, // Endothelial stalk cell — proliferates to extend vessel
        PhalanxCell = 3  // Quiescent endothelial cell in mature vessel
    };

    static_assert( static_cast<uint32_t>( LifecycleState::Live )              == 0 );
    static_assert( static_cast<uint32_t>( LifecycleState::Dead_PendingRemoval ) == 5 );
    static_assert( static_cast<uint32_t>( CellType::Default ) == 0 );

} // namespace DigitalTwin
