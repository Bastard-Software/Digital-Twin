#pragma once
#include "core/Core.h"
#include "simulation/AgentGroup.h"
#include "simulation/GridField.h"
#include <glm/glm.hpp>
#include <vector>

namespace DigitalTwin
{

    enum class SpatialPartitioningMethod
    {
        HashGrid,
        HierarchicalGrid, // For future use
    };

    struct SpatialPartitioningConfig
    {
        SpatialPartitioningMethod method     = SpatialPartitioningMethod::HashGrid;
        float                     cellSize   = 30.0f;
        uint32_t                  maxDensity = 64;
        float                     computeHz  = 60.0f;

        SpatialPartitioningConfig& SetMethod( SpatialPartitioningMethod m ) // For future use
        {
            method = m;
            return *this;
        }
        SpatialPartitioningConfig& SetCellSize( float size )
        {
            cellSize = size;
            return *this;
        }
        SpatialPartitioningConfig& SetMaxDensity( uint32_t density ) // For future use
        {
            maxDensity = density;
            return *this;
        }
        SpatialPartitioningConfig& SetComputeHz( float hz ) // For future use
        {
            computeHz = hz;
            return *this;
        }
    };

    /**
     * @brief A data container representing the initial recipe/plan for a simulation.
     * This exists entirely on the CPU and contains no graphics or Vulkan dependencies.
     */
    class DT_API SimulationBlueprint
    {
    public:
        // --- Identity ---
        SimulationBlueprint& SetName( const std::string& name )
        {
            m_name = name;
            return *this;
        }
        const std::string& GetName() const { return m_name; }

        // --- Domain & Grid Fields API ---
        SimulationBlueprint& SetDomainSize( glm::vec3 size, float voxelSize )
        {
            m_domainSize = size;
            m_voxelSize  = voxelSize;
            return *this;
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
        const glm::vec3&                 GetDomainSize() const { return m_domainSize; }
        float                            GetVoxelSize() const { return m_voxelSize; }
        const std::vector<GridField>&    GetGridFields() const { return m_gridFields; }
        const std::vector<AgentGroup>&   GetGroups() const { return m_groups; }
        std::vector<AgentGroup>&         GetGroupsMutable() { return m_groups; }
        std::vector<GridField>&          GetGridFieldsMutable() { return m_gridFields; }
        const SpatialPartitioningConfig& GetSpatialPartitioning() const { return m_spatialConfig; }
        SpatialPartitioningConfig&       ConfigureSpatialPartitioning() { return m_spatialConfig; }

    private:
        std::string               m_name       = "Untitled Simulation";
        glm::vec3                 m_domainSize = glm::vec3( 1000.0f );
        float                     m_voxelSize  = 10.0f;
        SpatialPartitioningConfig m_spatialConfig;
        std::vector<GridField>    m_gridFields;
        std::vector<AgentGroup>   m_groups;
    };
} // namespace DigitalTwin