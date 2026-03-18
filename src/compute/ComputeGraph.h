#pragma once
#include "compute/ComputeTask.h"
#include <vector>

namespace DigitalTwin
{
    // TODO: In future DAG

    class ComputeGraph
    {
    public:
        void         AddTask( const ComputeTask& task ) { m_tasks.push_back( task ); }
        void         Execute( CommandBuffer* cmd, float dt, float totalTime, uint32_t activeIndex );
        bool         IsEmpty() const { return m_tasks.empty(); }
        ComputeTask* FindTask( const std::string& tag );

    private:
        std::vector<ComputeTask> m_tasks;
    };
} // namespace DigitalTwin