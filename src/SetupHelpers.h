#pragma once
#include "rhi/RHITypes.h"

#include <filesystem>
#include <vector>

namespace DigitalTwin::Helpers
{

    /**
     * @brief Attempts to dynamically find the engine's internal root directory.
     * Useful for locating default shaders and engine assets.
     */
    std::filesystem::path FindEngineRoot();

    /**
     * @brief Selects the best GPU adapter index based on the user preference.
     * @param preference The GPUType preference (Default, Discrete, Integrated).
     * @param adapters The list of available adapters.
     * @return The index of the selected adapter.
     */
    uint32_t SelectGPU( GPUType preference, const std::vector<AdapterInfo>& adapters );

} // namespace DigitalTwin::Helpers