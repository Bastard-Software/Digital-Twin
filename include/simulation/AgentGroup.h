#pragma once
#include "simulation/SimulationTypes.h"

#include "core/Core.h"
#include "simulation/Behaviours.h"
#include <glm/glm.hpp>
#include <string>
#include <vector>

namespace DigitalTwin
{
    // ── Distribution spec ────────────────────────────────────────────────────
    // Editor-visible, round-trippable description of an agent spatial distribution.
    // Kept alongside the compiled position list so the UI can display/edit the choice.

    enum class DistributionType
    {
        Point,             // no auto-generation; uses raw SetDistribution() positions
        UniformInSphere,
        UniformInBox,
        UniformInCylinder,
    };

    struct DistributionSpec
    {
        DistributionType type       = DistributionType::UniformInSphere;
        glm::vec3        center     = { 0.0f, 0.0f, 0.0f };
        float            radius     = 50.0f;                    // Sphere / Cylinder outer radius
        glm::vec3        halfExtents = { 50.0f, 50.0f, 50.0f }; // Box
        float            halfLength = 50.0f;                    // Cylinder
        glm::vec3        axis       = { 0.0f, 1.0f, 0.0f };    // Cylinder
        uint32_t         seed       = 42;
    };

    // ── Morphology preset spec ────────────────────────────────────────────────
    // Editor-visible shorthand for agent shape. More presets (Cylinder, Disc, Ellipsoid)
    // can be added later without changing the UI structure.

    enum class MorphologyPreset
    {
        Sphere,
    };

    struct MorphologyPresetSpec
    {
        MorphologyPreset preset = MorphologyPreset::Sphere;
        float            radius = 0.5f;
    };

    /**
     * @brief Associates a mesh with a specific cell type index for multi-mesh rendering.
     */
    struct MorphologyEntry
    {
        int            cellTypeIndex;
        MorphologyData mesh;
        glm::vec4      color = glm::vec4( -1.0f ); // negative = fall back to group base color
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

        // --- Fluent Setters ---
        AgentGroup& SetName( const std::string& name )
        {
            m_name = name;
            return *this;
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
        AgentGroup& SetDistributionSpec( const DistributionSpec& spec )
        {
            m_distSpec = spec;
            return *this;
        }
        AgentGroup& SetMorphologyPreset( const MorphologyPresetSpec& spec )
        {
            m_morphSpec = spec;
            return *this;
        }
        // Per-cell outward orientation normal (xyz=normal, w=0). When provided, the renderer
        // rotates each cell's mesh so its +Y axis aligns with the stored normal.
        // Used by VesselTreeGenerator to orient disc cells outward from the tube centerline.
        AgentGroup& SetOrientations( const std::vector<glm::vec4>& orientations )
        {
            m_orientations = orientations;
            return *this;
        }
        AgentGroup& SetColor( const glm::vec4& color )
        {
            m_color = color;
            return *this;
        }
        AgentGroup& SetInitialCellType( int cellType )
        {
            m_initialCellType = cellType;
            return *this;
        }
        AgentGroup& SetVisible( bool visible )
        {
            m_visible = visible;
            return *this;
        }

        template<typename T>
        BehaviourRecord& AddBehaviour( T behaviour )
        {
            m_behaviours.push_back( { behaviour, 60.0f } );
            return m_behaviours.back();
        }

        AgentGroup& AddCellTypeMorphology( int cellTypeIndex, MorphologyData mesh )
        {
            m_cellTypeMorphologies.push_back( { cellTypeIndex, std::move( mesh ) } );
            return *this;
        }
        AgentGroup& AddCellTypeMorphology( int cellTypeIndex, MorphologyData mesh, const glm::vec4& color )
        {
            m_cellTypeMorphologies.push_back( { cellTypeIndex, std::move( mesh ), color } );
            return *this;
        }

        // --- Getters ---
        const std::string&                   GetName() const { return m_name; }
        uint32_t                             GetCount() const { return m_count; }
        const MorphologyData&                GetMorphology() const { return m_morphology; }
        const std::vector<glm::vec4>&        GetPositions() const { return m_positions; }
        const std::vector<glm::vec4>&        GetOrientations() const { return m_orientations; }
        const glm::vec4&                     GetColor() const { return m_color; }
        int                                  GetInitialCellType() const { return m_initialCellType; }
        bool                                 IsVisible() const { return m_visible; }
        const std::vector<BehaviourRecord>&  GetBehaviours() const { return m_behaviours; }
        std::vector<BehaviourRecord>&        GetBehavioursMutable() { return m_behaviours; }
        const std::vector<MorphologyEntry>&  GetCellTypeMorphologies() const { return m_cellTypeMorphologies; }
        const DistributionSpec&              GetDistributionSpec() const { return m_distSpec; }
        const MorphologyPresetSpec&          GetMorphologyPresetSpec() const { return m_morphSpec; }

    private:
        std::string                   m_name;
        uint32_t                      m_count = 0;
        MorphologyData                m_morphology;
        std::vector<glm::vec4>        m_positions;
        std::vector<glm::vec4>        m_orientations; // per-cell normals (empty = use default +Y)
        glm::vec4                     m_color = glm::vec4( 1.0f );
        int                           m_initialCellType = 0;
        bool                          m_visible         = true;
        std::vector<BehaviourRecord>  m_behaviours;
        std::vector<MorphologyEntry>  m_cellTypeMorphologies;
        DistributionSpec              m_distSpec;
        MorphologyPresetSpec          m_morphSpec;
    };
} // namespace DigitalTwin