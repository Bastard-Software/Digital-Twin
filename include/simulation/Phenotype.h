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
        Dead_PendingRemoval = 5,

        Any = 0xFFFFFFFFu // No lifecycle filter (shaders interpret 0xFFFFFFFF as "match all")
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
        PhalanxCell = 3, // Quiescent endothelial cell in mature vessel

        Any = 0xFFFFFFFFu // No cell-type filter (shaders interpret 0xFFFFFFFF as "match all")
    };

    static_assert( static_cast<uint32_t>( LifecycleState::Live )              == 0 );
    static_assert( static_cast<uint32_t>( LifecycleState::Dead_PendingRemoval ) == 5 );
    static_assert( static_cast<uint32_t>( CellType::Default ) == 0 );

    // CPU-side mirror of the std430 SSBO layout used by phenotypeBuffer.
    // Size stays at 16 bytes — cadherin profile lives in a separate buffer (Stage 3).
    // This struct is defined locally in 14 GLSL shaders (no GLSL include mechanism);
    // this header is the single source of truth for C++ code.
    //
    // Bit layout of `cellType` (Item 2 Phase 2.1, 2026-04-19):
    //   bits  0..15 = CellType enum (Tip / Stalk / Phalanx / Default) — biological role
    //   bits 16..31 = morphologyIndex — mesh variant index within the AgentGroup's
    //                 variant list (0 = default / first registered mesh)
    //
    // Biologically: a mature vascular monolayer contains heterogeneous EC shapes
    // within a single lineage-continuous layer (Aird 2007 DOI 10.1161/01.RES.0000255691.76142.4a).
    // Keeping all variants in ONE AgentGroup preserves cross-variant cadherin
    // neighbourship + spatial hashing; a per-cell morphology index selects which
    // mesh to render without fragmenting the group.
    //
    // Shaders that filter on biological role via `reqCT` MUST mask to the lower
    // 16 bits when comparing: `(cellType & 0xFFFFu) != uint(reqCT)`. The
    // `build_indirect.comp` shader performs an exact uint match against
    // `DrawMeta.targetCellType` which is populated by the builder using
    // `PackCellType(biologicalType, morphIdx)` on both sides, so that path
    // needs no mask — the packed 32-bit values match.
    struct PhenotypeData
    {
        uint32_t lifecycleState = 0;
        float    biomass        = 0.5f;
        float    timer          = 0.0f;
        uint32_t cellType       = 0;
    };

    inline uint32_t PackCellType( CellType type, uint32_t morphologyIndex )
    {
        return ( static_cast<uint32_t>( type ) & 0xFFFFu ) | ( ( morphologyIndex & 0xFFFFu ) << 16 );
    }

    inline CellType UnpackCellType( uint32_t packed )
    {
        return static_cast<CellType>( packed & 0xFFFFu );
    }

    inline uint32_t UnpackMorphologyIndex( uint32_t packed )
    {
        return ( packed >> 16 ) & 0xFFFFu;
    }

} // namespace DigitalTwin
