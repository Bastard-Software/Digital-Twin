#pragma once
#include "compute/ComputeKernel.hpp"
#include <vector>

namespace DigitalTwin
{

    struct ComputeTask
    {
        Ref<ComputeKernel> kernel;
        Ref<BindingGroup>  bindings;
    };

    /**
     * @brief Defines a sequence of compute kernels to be executed.
     * Manages barriers and execution order.
     */
    class ComputeGraph
    {
    public:
        // Adds a task to the sequence
        void AddTask( Ref<ComputeKernel> kernel, Ref<BindingGroup> bindings );

        // Records the entire sequence into the command buffer
        void Record( CommandBuffer& cmd, uint32_t agentCount );

    private:
        std::vector<ComputeTask> m_tasks;
    };
} // namespace DigitalTwin