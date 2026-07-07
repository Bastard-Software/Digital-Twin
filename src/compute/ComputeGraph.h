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

        // The field-texture ping-pong index (advances once per frame, independent of the agent
        // buffer index). After Execute this points at the texture holding each field's freshest
        // data — the renderer and readbacks should read textures[GetFieldActive()].
        uint32_t GetFieldActive() const { return m_fieldActive; }

    private:
        std::vector<ComputeTask> m_tasks;
        uint32_t                 m_fieldActive = 0;
    };
} // namespace DigitalTwin