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

        /**
         * @brief Checks if the graph contains any tasks.
         * @return true if there are no kernels to execute.
         */
        bool IsEmpty() const { return m_tasks.empty(); }

        /**
         * @brief Clears all tasks and releases references to kernels/pipelines.
         * useful for clean shutdown.
         */
        void Clear() { m_tasks.clear(); }

    private:
        std::vector<ComputeTask> m_tasks;
    };
} // namespace DigitalTwin