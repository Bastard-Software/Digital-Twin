#pragma once
#include "simulation/SimulationTypes.h"

#include "core/Core.h"
#include "simulation/Behaviours.h"
#include <glm/glm.hpp>
#include <string>
#include <variant>
#include <vector>

namespace DigitalTwin
{

    // A variant holding all possible behaviours the engine understands
    using BehaviourVariant = std::variant<Behaviours::BrownianMotion>;

    // Wrapper to attach execution parameters (like frequency) to a behaviour
    struct BehaviourRecord
    {
        BehaviourVariant behaviour;
        float            targetHz = 60.0f; // Default to 60 executions per second

        // Fluent API for frequency
        BehaviourRecord& SetHz( float hz )
        {
            targetHz = hz;
            return *this;
        }
    };

    /**
     * @brief Represents a specific type/population of agents within the simulation.
     * Uses a fluent interface (method chaining) for easy configuration.
     */
    class DT_API AgentGroup
    {
    public:
        explicit AgentGroup( const std::string& name )
            : m_name( name )
        {
        }

        AgentGroup& SetCount( uint32_t count )
        {
            m_count = count;
            return *this;
        }

        AgentGroup& SetMorphology( const MorphologyData& morphology )
        {
            m_morphology = morphology;
            return *this;
        }

        AgentGroup& SetDistribution( const std::vector<glm::vec4>& positions )
        {
            m_positions = positions;
            return *this;
        }

        AgentGroup& SetColor( const glm::vec4& color )
        {
            m_color = color;
            return *this;
        }

        /**
         * @brief Adds a specific computational behaviour to the agents.
         * @return A reference to allow chaining, e.g., .SetHz(30.0f)
         */
        template<typename T>
        BehaviourRecord& AddBehaviour( const T& behaviour )
        {
            BehaviourRecord record;
            record.behaviour = behaviour;
            m_behaviours.push_back( record );
            return m_behaviours.back();
        }

        // Getters
        const std::string&                  GetName() const { return m_name; }
        uint32_t                            GetCount() const { return m_count; }
        const MorphologyData&               GetMorphology() const { return m_morphology; }
        const std::vector<glm::vec4>&       GetPositions() const { return m_positions; }
        const glm::vec4&                    GetColor() const { return m_color; }
        const std::vector<BehaviourRecord>& GetBehaviours() const { return m_behaviours; }

    private:
        std::string                  m_name;
        uint32_t                     m_count = 0;
        MorphologyData               m_morphology;
        std::vector<glm::vec4>       m_positions;
        glm::vec4                    m_color = glm::vec4( 1.0f );
        std::vector<BehaviourRecord> m_behaviours;
    };

    /**
     * @brief A data container representing the initial recipe/plan for a simulation.
     * This exists entirely on the CPU and contains no graphics or Vulkan dependencies.
     */
    class DT_API SimulationBlueprint
    {
    public:
        /**
         * @brief Adds a new group of agents to the blueprint.
         * @param name A descriptive name for the group (e.g., "T-Cells").
         * @return A reference to the newly created group for method chaining.
         */
        AgentGroup& AddAgentGroup( const std::string& name )
        {
            m_groups.emplace_back( name );
            return m_groups.back();
        }

        const std::vector<AgentGroup>& GetGroups() const { return m_groups; }

    private:
        std::vector<AgentGroup> m_groups;
    };

} // namespace DigitalTwin