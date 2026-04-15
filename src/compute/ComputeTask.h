#pragma once
#include "rhi/RHITypes.h"

#include <glm/glm.hpp>
#include <string>

namespace DigitalTwin
{

    /**
     * @brief Raw data mapped directly to Vulkan Push Constants in the compute shader.
     */
    /**
     * @brief Standardized 64-byte payload mapped to Vulkan Push Constants.
     * All compute shaders MUST define this exact memory layout.
     */
    struct ComputePushConstants
    {
        float      dt;
        float      totalTime;
        float      fParam0;
        float      fParam1;
        float      fParam2;
        float      fParam3;
        float      fParam4;
        float      fParam5;
        uint32_t   offset;
        uint32_t   maxCapacity;
        uint32_t   uParam0;
        uint32_t   uParam1;
        glm::vec4  domainSize;
        glm::uvec4 gridSize;
    };

    /**
     * @brief Private internal class representing a single dispatch operation on the GPU.
     */
    class ComputeTask
    {
    public:
        ComputeTask( ComputePipeline* pipeline, BindingGroup* bgRead0, BindingGroup* bgRead1, float targetHz, const ComputePushConstants& pc,
                     glm::uvec3 dispatchSize );

        /**
         * @brief Checks the time accumulator against targetHz to see if it should run this frame.
         */
        bool ShouldExecute( float dt );

        /**
         * @brief Binds pipeline, descriptors (depend on active ndx), pushes constants, and dispatches.
         */
        void Record( CommandBuffer* cmd, float totalTime, uint32_t activeIndex );

        void               SetTag( const std::string& tag ) { m_tag = tag; }
        const std::string& GetTag() const { return m_tag; }

        void               SetPhaseName( const std::string& phase ) { m_phaseName = phase; }
        const std::string& GetPhaseName() const { return m_phaseName; }

        // When true, this task writes agent positions. ComputeGraph flips the active
        // buffer index after it runs so the next task in the chain reads the updated positions.
        void SetChainFlip( bool v ) { m_chainFlip = v; }
        bool GetChainFlip() const { return m_chainFlip; }

        void SetDtScale( float scale ) { m_dtScale = scale; }

        const ComputePushConstants& GetPushConstants() const { return m_pc; }

        /**
         * @brief Replaces the stored push constants (behaviour params). dt/totalTime are still overwritten per-frame.
         */
        void UpdatePushConstants( const ComputePushConstants& pc ) { m_pc = pc; }

    private:
        ComputePipeline* m_pipeline;
        BindingGroup*    m_bindings[ 2 ]; // Array holding Ping and Pong binding groups
        float            m_targetHz;
        float            m_timeAccumulator = 0.0f;

        std::string          m_tag;
        std::string          m_phaseName;
        ComputePushConstants m_pc;
        glm::uvec3           m_dispatchSize;
        bool                 m_chainFlip = false;
        float                m_dtScale   = 1.0f;
    };

} // namespace DigitalTwin