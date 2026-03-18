#pragma once

namespace Gaudi
{
    struct EditorSelection
    {
        int groupIndex     = -1; // >= 0 = agent group selected
        int behaviourIndex = -1; // >= 0 = behaviour within groupIndex selected
        int gridFieldIndex = -1; // >= 0 = grid field selected (groupIndex must be -1)
    };
} // namespace Gaudi
