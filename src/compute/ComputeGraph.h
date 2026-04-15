#pragma once
#include "compute/ComputeTask.h"
#include <vector>

namespace DigitalTwin
{
    class GPUProfiler;

    // TODO: In future DAG

    class ComputeGraph
    {
    public:
        void         AddTask( const ComputeTask& task ) { m_tasks.push_back( task ); }
        uint32_t     Execute( CommandBuffer* cmd, float dt, float totalTime, uint32_t activeIndex,
                              GPUProfiler* profiler = nullptr, uint32_t flightIndex = 0 );
        bool         IsEmpty() const { return m_tasks.empty(); }
        ComputeTask* FindTask( const std::string& tag );

    private:
        std::vector<ComputeTask> m_tasks;
    };
} // namespace DigitalTwin