#pragma once

namespace Gaudi
{
    struct EditorSelection
    {
        bool simRootSelected = false; // true = blueprint-level (domain/voxel/partitioning) selected
        int  groupIndex      = -1;    // >= 0 = agent group selected
        int  behaviourIndex  = -1;    // >= 0 = behaviour within groupIndex selected
        int  gridFieldIndex  = -1;    // >= 0 = grid field selected (groupIndex must be -1)

        void ClearAll()
        {
            simRootSelected = false;
            groupIndex      = -1;
            behaviourIndex  = -1;
            gridFieldIndex  = -1;
        }
    };
} // namespace Gaudi
