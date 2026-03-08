#pragma once
#include "core/Core.h"
#include "simulation/AgentGroup.h"
#include "simulation/GridField.h"
#include <glm/glm.hpp>
#include <vector>

namespace DigitalTwin
{
    /**
     * @brief A data container representing the initial recipe/plan for a simulation.
     * This exists entirely on the CPU and contains no graphics or Vulkan dependencies.
     */
    class DT_API SimulationBlueprint
    {
    public:
        // --- Domain & Grid Fields API ---
        void SetDomainSize( glm::vec3 size, float voxelSize )
        {
            m_domainSize = size;
            m_voxelSize  = voxelSize;
        }

        GridField& AddGridField( const std::string& name )
        {
            m_gridFields.emplace_back( name );
            return m_gridFields.back();
        }

        // --- Agents API ---
        AgentGroup& AddAgentGroup( const std::string& name )
        {
            m_groups.emplace_back( name );
            return m_groups.back();
        }

        // --- Getters ---
        const glm::vec3&               GetDomainSize() const { return m_domainSize; }
        float                          GetVoxelSize() const { return m_voxelSize; }
        const std::vector<GridField>&  GetGridFields() const { return m_gridFields; }
        const std::vector<AgentGroup>& GetGroups() const { return m_groups; }

    private:
        glm::vec3               m_domainSize = glm::vec3( 1000.0f );
        float                   m_voxelSize  = 10.0f;
        std::vector<GridField>  m_gridFields;
        std::vector<AgentGroup> m_groups;
    };
} // namespace DigitalTwin