#include "simulation/SimulationValidator.h"

#include "simulation/AgentGroup.h"
#include "simulation/Behaviours.h"
#include "simulation/GridField.h"
#include "simulation/SimulationBlueprint.h"

#include <algorithm>
#include <glm/glm.hpp>
#include <unordered_set>

namespace DigitalTwin
{
    // Valid filter values: LifecycleState::Any (0xFFFFFFFF = no requirement) or 0-4 (Live through Necrotic).
    // Dead_PendingRemoval (5) is a transient GPU-side state — behaviours conditioned on it never fire.
    static constexpr uint32_t k_MaxRequiredState = 4;
    static constexpr uint32_t k_MaxCellType      = 3; // PhalanxCell

    // Format a float to 2 decimal places for error messages.
    static std::string Fmt2f( float v )
    {
        char buf[ 32 ];
        snprintf( buf, sizeof( buf ), "%.2f", v );
        return buf;
    }

    ValidationResult SimulationValidator::Validate( const SimulationBlueprint& blueprint )
    {
        ValidationResult result;

        CheckDomain( blueprint, result );
        CheckDomainBounds( blueprint, result );
        CheckNames( blueprint, result );
        CheckFieldReferences( blueprint, result );
        CheckBehaviourParams( blueprint, result );
        CheckCellCycleThresholds( blueprint, result );
        CheckCadherinAdhesion( blueprint, result );
        CheckCellPolarity( blueprint, result );
        CheckBasementMembrane( blueprint, result );
        CheckPopulations( blueprint, result );

        return result;
    }

    void SimulationValidator::CheckDomain( const SimulationBlueprint& blueprint, ValidationResult& result )
    {
        const glm::vec3& size = blueprint.GetDomainSize();

        if( size.x <= 0.0f )
            result.AddError( "Domain size X axis must be > 0 (got: " + Fmt2f( size.x ) + ")" );
        if( size.y <= 0.0f )
            result.AddError( "Domain size Y axis must be > 0 (got: " + Fmt2f( size.y ) + ")" );
        if( size.z <= 0.0f )
            result.AddError( "Domain size Z axis must be > 0 (got: " + Fmt2f( size.z ) + ")" );

        if( blueprint.GetVoxelSize() <= 0.0f )
            result.AddError( "Voxel size must be > 0 (got: " + Fmt2f( blueprint.GetVoxelSize() ) + ")" );

        if( blueprint.GetSpatialPartitioning().cellSize <= 0.0f )
            result.AddError( "SpatialPartitioning cell size must be > 0 (got: " +
                             Fmt2f( blueprint.GetSpatialPartitioning().cellSize ) + ")" );
    }

    // Domain is centred at origin — a point p is inside iff |p.xyz| <= domainSize / 2.
    // Returns the first axis ('X'/'Y'/'Z') on which an AABB min..max leaks outside ±half.
    // Returns 0 if fully inside. Writes the offending coordinate and the relevant half-extent
    // into out_reached / out_halfExtent for error messages.
    static char FindDomainOverflowAxis( const glm::vec3& aabbMin, const glm::vec3& aabbMax,
                                        const glm::vec3& half, float& out_reached, float& out_halfExtent )
    {
        const float ax[ 3 ] = { aabbMin.x, aabbMin.y, aabbMin.z };
        const float bx[ 3 ] = { aabbMax.x, aabbMax.y, aabbMax.z };
        const float hv[ 3 ] = { half.x, half.y, half.z };
        const char  names[ 3 ] = { 'X', 'Y', 'Z' };
        for( int i = 0; i < 3; ++i )
        {
            if( ax[ i ] < -hv[ i ] )
            {
                out_reached    = ax[ i ];
                out_halfExtent = hv[ i ];
                return names[ i ];
            }
            if( bx[ i ] > hv[ i ] )
            {
                out_reached    = bx[ i ];
                out_halfExtent = hv[ i ];
                return names[ i ];
            }
        }
        return 0;
    }

    void SimulationValidator::CheckDomainBounds( const SimulationBlueprint& blueprint, ValidationResult& result )
    {
        const glm::vec3 size = blueprint.GetDomainSize();
        // If any axis of domain is non-positive, CheckDomain already errored — skip bounds checks
        // to avoid cascading confusing messages against a broken domain.
        if( size.x <= 0.0f || size.y <= 0.0f || size.z <= 0.0f )
            return;

        const glm::vec3 half = size * 0.5f;

        // ── Agent group distributions ───────────────────────────────────────────
        for( const auto& group : blueprint.GetGroups() )
        {
            const DistributionSpec& d    = group.GetDistributionSpec();
            glm::vec3               mn   = glm::vec3( 0.0f );
            glm::vec3               mx   = glm::vec3( 0.0f );
            bool                    hasBox = false;

            // If raw positions are already populated (e.g. via SetDistribution()), they are the
            // ground truth regardless of the spec (spec is only consulted by the compile step
            // when positions are empty). Iterate actual positions.
            if( !group.GetPositions().empty() )
            {
                for( const auto& p : group.GetPositions() )
                {
                    const glm::vec3 v( p.x, p.y, p.z );
                    float           reached = 0.0f, he = 0.0f;
                    char            axis = FindDomainOverflowAxis( v, v, half, reached, he );
                    if( axis )
                    {
                        result.AddError( "AgentGroup '" + group.GetName() + "' has a position outside domain on " +
                                         std::string( 1, axis ) + " (reached " + Fmt2f( reached ) +
                                         ", domain half-extent " + Fmt2f( he ) + ")" );
                        break;
                    }
                }
                continue;
            }

            switch( d.type )
            {
            case DistributionType::Point:
                // Point type with no positions set — nothing to check.
                break;
            case DistributionType::UniformInSphere:
                mn     = d.center - glm::vec3( d.radius );
                mx     = d.center + glm::vec3( d.radius );
                hasBox = true;
                break;
            case DistributionType::UniformInBox:
                mn     = d.center - d.halfExtents;
                mx     = d.center + d.halfExtents;
                hasBox = true;
                break;
            case DistributionType::UniformInCylinder:
            {
                // Axis-agnostic conservative AABB: max(radius, halfLength) in every axis.
                const float r = std::max( d.radius, d.halfLength );
                mn            = d.center - glm::vec3( r );
                mx            = d.center + glm::vec3( r );
                hasBox        = true;
                break;
            }
            }

            if( hasBox )
            {
                float reached = 0.0f, he = 0.0f;
                char  axis = FindDomainOverflowAxis( mn, mx, half, reached, he );
                if( axis )
                    result.AddError( "AgentGroup '" + group.GetName() + "' distribution extends outside domain on " +
                                     std::string( 1, axis ) + " (reaches " + Fmt2f( reached ) +
                                     ", domain half-extent " + Fmt2f( he ) + ")" );
            }
        }

        // ── Grid field initializers ─────────────────────────────────────────────
        for( const auto& field : blueprint.GetGridFields() )
        {
            const InitializerSpec& s = field.GetInitializerSpec();
            glm::vec3              mn = glm::vec3( 0.0f );
            glm::vec3              mx = glm::vec3( 0.0f );
            bool                   hasBox = false;

            switch( s.type )
            {
            case InitializerType::Constant:
                continue; // uniform fill — never extends outside
            case InitializerType::Sphere:
                mn     = s.center - glm::vec3( s.radius );
                mx     = s.center + glm::vec3( s.radius );
                hasBox = true;
                break;
            case InitializerType::BoxWall:
                mn     = s.center - s.halfExtents;
                mx     = s.center + s.halfExtents;
                hasBox = true;
                break;
            case InitializerType::Gaussian:
            {
                // Centre must be in-domain (error); 3σ tail leaking is warning-worthy.
                float reached = 0.0f, he = 0.0f;
                char  axis = FindDomainOverflowAxis( s.center, s.center, half, reached, he );
                if( axis )
                {
                    result.AddError( "GridField '" + field.GetName() + "' Gaussian centre is outside domain on " +
                                     std::string( 1, axis ) + " (reached " + Fmt2f( reached ) +
                                     ", domain half-extent " + Fmt2f( he ) + ")" );
                }
                else
                {
                    const glm::vec3 tmn = s.center - glm::vec3( 3.0f * s.sigma );
                    const glm::vec3 tmx = s.center + glm::vec3( 3.0f * s.sigma );
                    char            tAxis = FindDomainOverflowAxis( tmn, tmx, half, reached, he );
                    if( tAxis )
                        result.AddWarning( "GridField '" + field.GetName() + "' Gaussian 3-sigma tail is truncated at domain edge on " +
                                           std::string( 1, tAxis ) + " (reaches " + Fmt2f( reached ) +
                                           ", domain half-extent " + Fmt2f( he ) + ")" );
                }
                continue;
            }
            case InitializerType::MultiGaussian:
            {
                bool anyCentreError = false;
                for( const auto& c : s.centers )
                {
                    float reached = 0.0f, he = 0.0f;
                    char  axis = FindDomainOverflowAxis( c, c, half, reached, he );
                    if( axis )
                    {
                        result.AddError( "GridField '" + field.GetName() + "' MultiGaussian centre is outside domain on " +
                                         std::string( 1, axis ) + " (reached " + Fmt2f( reached ) +
                                         ", domain half-extent " + Fmt2f( he ) + ")" );
                        anyCentreError = true;
                        break;
                    }
                }
                if( !anyCentreError )
                {
                    // Warn once if any 3σ tail leaks.
                    for( const auto& c : s.centers )
                    {
                        const glm::vec3 tmn = c - glm::vec3( 3.0f * s.sigma );
                        const glm::vec3 tmx = c + glm::vec3( 3.0f * s.sigma );
                        float           reached = 0.0f, he = 0.0f;
                        char            tAxis = FindDomainOverflowAxis( tmn, tmx, half, reached, he );
                        if( tAxis )
                        {
                            result.AddWarning( "GridField '" + field.GetName() + "' MultiGaussian 3-sigma tail is truncated at domain edge on " +
                                               std::string( 1, tAxis ) + " (reaches " + Fmt2f( reached ) +
                                               ", domain half-extent " + Fmt2f( he ) + ")" );
                            break;
                        }
                    }
                }
                continue;
            }
            }

            if( hasBox )
            {
                float reached = 0.0f, he = 0.0f;
                char  axis = FindDomainOverflowAxis( mn, mx, half, reached, he );
                if( axis )
                    result.AddError( "GridField '" + field.GetName() + "' initializer extends outside domain on " +
                                     std::string( 1, axis ) + " (reaches " + Fmt2f( reached ) +
                                     ", domain half-extent " + Fmt2f( he ) + ")" );
            }
        }
    }

    void SimulationValidator::CheckNames( const SimulationBlueprint& blueprint, ValidationResult& result )
    {
        // Check for duplicate AgentGroup names
        std::unordered_set<std::string> groupNames;
        for( const auto& group : blueprint.GetGroups() )
        {
            if( !groupNames.insert( group.GetName() ).second )
                result.AddError( "Duplicate AgentGroup name: '" + group.GetName() + "'" );
        }

        // Check for duplicate GridField names
        std::unordered_set<std::string> fieldNames;
        for( const auto& field : blueprint.GetGridFields() )
        {
            if( !fieldNames.insert( field.GetName() ).second )
                result.AddError( "Duplicate GridField name: '" + field.GetName() + "'" );
        }
    }

    void SimulationValidator::CheckFieldReferences( const SimulationBlueprint& blueprint, ValidationResult& result )
    {
        // Build the set of declared field names for O(1) lookup
        std::unordered_set<std::string> declaredFields;
        for( const auto& field : blueprint.GetGridFields() )
            declaredFields.insert( field.GetName() );

        // Build a helper string listing available fields for error messages
        std::string availableFields = "[";
        for( const auto& name : declaredFields )
            availableFields += name + ", ";
        if( availableFields.size() > 1 )
            availableFields.erase( availableFields.size() - 2 ); // trim trailing ", "
        availableFields += "]";

        for( const auto& group : blueprint.GetGroups() )
        {
            for( const auto& record : group.GetBehaviours() )
            {
                std::visit(
                    [ & ]( const auto& behaviour )
                    {
                        using T = std::decay_t<decltype( behaviour )>;

                        if constexpr( std::is_same_v<T, Behaviours::ConsumeField> ||
                                      std::is_same_v<T, Behaviours::SecreteField> )
                        {
                            if( behaviour.fieldName.empty() )
                            {
                                result.AddError( "AgentGroup '" + group.GetName() +
                                                 "': field name must not be empty in " +
                                                 ( std::is_same_v<T, Behaviours::ConsumeField> ? "ConsumeField"
                                                                                                : "SecreteField" ) );
                            }
                            else if( declaredFields.find( behaviour.fieldName ) == declaredFields.end() )
                            {
                                result.AddError( "AgentGroup '" + group.GetName() + "': " +
                                                 ( std::is_same_v<T, Behaviours::ConsumeField> ? "ConsumeField"
                                                                                                : "SecreteField" ) +
                                                 " references unknown field '" + behaviour.fieldName +
                                                 "'. Available fields: " + availableFields );
                            }
                        }

                        if constexpr( std::is_same_v<T, Behaviours::Chemotaxis> )
                        {
                            if( behaviour.fieldName.empty() )
                                result.AddError( "AgentGroup '" + group.GetName() +
                                                 "': Chemotaxis fieldName must not be empty" );
                            else if( declaredFields.find( behaviour.fieldName ) == declaredFields.end() )
                                result.AddError( "AgentGroup '" + group.GetName() +
                                                 "': Chemotaxis references unknown field '" + behaviour.fieldName +
                                                 "'. Available fields: " + availableFields );
                        }

                        if constexpr( std::is_same_v<T, Behaviours::PhalanxActivation> )
                        {
                            if( behaviour.vegfFieldName.empty() )
                                result.AddError( "AgentGroup '" + group.GetName() +
                                                 "': PhalanxActivation vegfFieldName must not be empty" );
                            else if( declaredFields.find( behaviour.vegfFieldName ) == declaredFields.end() )
                                result.AddError( "AgentGroup '" + group.GetName() +
                                                 "': PhalanxActivation references unknown field '" + behaviour.vegfFieldName +
                                                 "'. Available fields: " + availableFields );
                        }

                        if constexpr( std::is_same_v<T, Behaviours::Perfusion> ||
                                      std::is_same_v<T, Behaviours::Drain> )
                        {
                            const std::string typeName = std::is_same_v<T, Behaviours::Perfusion> ? "Perfusion" : "Drain";
                            if( behaviour.fieldName.empty() )
                                result.AddError( "AgentGroup '" + group.GetName() +
                                                 "': " + typeName + " fieldName must not be empty" );
                            else if( declaredFields.find( behaviour.fieldName ) == declaredFields.end() )
                                result.AddError( "AgentGroup '" + group.GetName() +
                                                 "': " + typeName + " references unknown field '" + behaviour.fieldName +
                                                 "'. Available fields: " + availableFields );
                        }
                    },
                    record.behaviour );
            }
        }
    }

    void SimulationValidator::CheckBehaviourParams( const SimulationBlueprint& blueprint, ValidationResult& result )
    {
        for( const auto& group : blueprint.GetGroups() )
        {
            // Track variant indices to detect duplicate behaviour types within one group
            std::unordered_set<size_t> seenVariantIndices;

            for( const auto& record : group.GetBehaviours() )
            {
                // Frequency check
                if( record.targetHz <= 0.0f )
                    result.AddError( "AgentGroup '" + group.GetName() +
                                     "': BehaviourRecord targetHz must be > 0 (got: " +
                                     std::to_string( record.targetHz ) + ")" );

                // Duplicate behaviour type check
                size_t variantIndex = record.behaviour.index();
                if( !seenVariantIndices.insert( variantIndex ).second )
                    result.AddWarning( "AgentGroup '" + group.GetName() +
                                       "': duplicate behaviour type detected. Only the first instance will be effective." );

                // BehaviourRecord-level lifecycle and cell type filtering
                {
                    uint32_t ls = static_cast<uint32_t>( record.requiredLifecycleState );
                    if( ls != 0xFFFFFFFFu && ls > k_MaxRequiredState )
                        result.AddError( "AgentGroup '" + group.GetName() +
                                         "': BehaviourRecord requiredLifecycleState out of range (got: " +
                                         std::to_string( ls ) + ", valid values: 0-4 or Any)" );
                }
                {
                    uint32_t ct = static_cast<uint32_t>( record.requiredCellType );
                    if( ct != 0xFFFFFFFFu && ct > k_MaxCellType )
                        result.AddError( "AgentGroup '" + group.GetName() +
                                         "': BehaviourRecord requiredCellType out of range (got: " +
                                         std::to_string( ct ) + ", valid values: 0-3 or Any)" );
                }

                // Per-type parameter checks
                std::visit(
                    [ & ]( const auto& behaviour )
                    {
                        using T = std::decay_t<decltype( behaviour )>;

                        if constexpr( std::is_same_v<T, Behaviours::Biomechanics> )
                        {
                            if( behaviour.maxRadius <= 0.0f )
                                result.AddError( "AgentGroup '" + group.GetName() +
                                                 "': Biomechanics maxRadius must be > 0 (got: " +
                                                 std::to_string( behaviour.maxRadius ) + ")" );
                        }

                        if constexpr( std::is_same_v<T, Behaviours::ConsumeField> ||
                                      std::is_same_v<T, Behaviours::SecreteField> )
                        {
                            const std::string typeName =
                                std::is_same_v<T, Behaviours::ConsumeField> ? "ConsumeField" : "SecreteField";

                            {
                                uint32_t ls = static_cast<uint32_t>( behaviour.requiredLifecycleState );
                                if( ls != 0xFFFFFFFFu && ls > k_MaxRequiredState )
                                    result.AddError( "AgentGroup '" + group.GetName() + "': " + typeName +
                                                     " requiredLifecycleState is out of range (got: " +
                                                     std::to_string( ls ) + ", valid values: 0-4 or Any)" );
                            }

                            if( behaviour.rate == 0.0f )
                                result.AddWarning( "AgentGroup '" + group.GetName() + "': " + typeName +
                                                   " has rate == 0. The behaviour is registered but has no effect." );
                        }

                        if constexpr( std::is_same_v<T, Behaviours::BrownianMotion> )
                        {
                            if( behaviour.speed <= 0.0f )
                                result.AddWarning( "AgentGroup '" + group.GetName() +
                                                   "': BrownianMotion speed <= 0. The behaviour will have no effect." );
                        }

                        if constexpr( std::is_same_v<T, Behaviours::Chemotaxis> )
                        {
                            if( behaviour.chemotacticSensitivity <= 0.0f )
                                result.AddError( "AgentGroup '" + group.GetName() +
                                                 "': Chemotaxis chemotacticSensitivity must be > 0" );
                            if( behaviour.receptorSaturation < 0.0f )
                                result.AddError( "AgentGroup '" + group.GetName() +
                                                 "': Chemotaxis receptorSaturation must be >= 0" );
                            if( behaviour.maxVelocity <= 0.0f )
                                result.AddError( "AgentGroup '" + group.GetName() +
                                                 "': Chemotaxis maxVelocity must be > 0" );
                        }

                        if constexpr( std::is_same_v<T, Behaviours::PhalanxActivation> )
                        {
                            if( behaviour.activationThreshold <= 0.0f )
                                result.AddError( "AgentGroup '" + group.GetName() + "': PhalanxActivation activationThreshold must be > 0" );
                            if( behaviour.deactivationThreshold < 0.0f )
                                result.AddError( "AgentGroup '" + group.GetName() + "': PhalanxActivation deactivationThreshold must be >= 0" );
                            if( behaviour.deactivationThreshold >= behaviour.activationThreshold )
                                result.AddError( "AgentGroup '" + group.GetName() + "': PhalanxActivation deactivationThreshold must be < activationThreshold" );
                        }

                        if constexpr( std::is_same_v<T, Behaviours::NotchDll4> )
                        {
                            if( behaviour.dll4ProductionRate <= 0.0f )
                                result.AddError( "AgentGroup '" + group.GetName() + "': NotchDll4 dll4ProductionRate must be > 0" );
                            if( behaviour.dll4DecayRate <= 0.0f )
                                result.AddError( "AgentGroup '" + group.GetName() + "': NotchDll4 dll4DecayRate must be > 0" );
                            if( behaviour.notchInhibitionGain <= 0.0f )
                                result.AddError( "AgentGroup '" + group.GetName() + "': NotchDll4 notchInhibitionGain must be > 0" );
                            if( behaviour.vegfr2BaseExpression <= 0.0f )
                                result.AddError( "AgentGroup '" + group.GetName() + "': NotchDll4 vegfr2BaseExpression must be > 0" );
                            if( behaviour.tipThreshold <= behaviour.stalkThreshold )
                                result.AddError( "AgentGroup '" + group.GetName() + "': NotchDll4 tipThreshold must be > stalkThreshold" );
                            if( behaviour.subSteps < 1 )
                                result.AddError( "AgentGroup '" + group.GetName() + "': NotchDll4 subSteps must be >= 1" );
                        }

                        if constexpr( std::is_same_v<T, Behaviours::Anastomosis> )
                        {
                            if( behaviour.contactDistance <= 0.0f )
                                result.AddError( "AgentGroup '" + group.GetName() + "': Anastomosis contactDistance must be > 0" );
                        }

                        if constexpr( std::is_same_v<T, Behaviours::VesselSpring> )
                        {
                            if( behaviour.springStiffness <= 0.0f )
                                result.AddError( "AgentGroup '" + group.GetName() + "': VesselSpring springStiffness must be > 0" );
                            if( behaviour.restingLength <= 0.0f )
                                result.AddError( "AgentGroup '" + group.GetName() + "': VesselSpring restingLength must be > 0" );
                        }

                        if constexpr( std::is_same_v<T, Behaviours::Perfusion> ||
                                      std::is_same_v<T, Behaviours::Drain> )
                        {
                            const std::string typeName = std::is_same_v<T, Behaviours::Perfusion> ? "Perfusion" : "Drain";
                            if( behaviour.rate <= 0.0f )
                                result.AddError( "AgentGroup '" + group.GetName() + "': " + typeName + " rate must be > 0" );
                        }

                        if constexpr( std::is_same_v<T, Behaviours::VesselSeed> )
                        {
                            // explicitEdges mode bypasses segmentCounts entirely
                            if( behaviour.segmentCounts.empty() && behaviour.explicitEdges.empty() )
                            {
                                result.AddError( "AgentGroup '" + group.GetName() +
                                                 "': VesselSeed requires either segmentCounts or explicitEdges" );
                            }
                            else if( !behaviour.segmentCounts.empty() )
                            {
                                uint32_t sum     = 0;
                                bool     hasZero = false;
                                for( uint32_t sc : behaviour.segmentCounts )
                                {
                                    if( sc == 0 )
                                    {
                                        result.AddError( "AgentGroup '" + group.GetName() +
                                                         "': VesselSeed segmentCounts contains a zero entry" );
                                        hasZero = true;
                                        break;
                                    }
                                    sum += sc;
                                }
                                if( !hasZero && sum > group.GetCount() )
                                    result.AddError( "AgentGroup '" + group.GetName() +
                                                     "': VesselSeed segmentCounts sum (" + std::to_string( sum ) +
                                                     ") exceeds group count (" + std::to_string( group.GetCount() ) + ")" );
                            }
                        }
                    },
                    record.behaviour );
            }
        }
    }

    void SimulationValidator::CheckCellCycleThresholds( const SimulationBlueprint& blueprint, ValidationResult& result )
    {
        for( const auto& group : blueprint.GetGroups() )
        {
            bool hasCellCycle    = false;
            bool hasConsumeField = false;

            for( const auto& record : group.GetBehaviours() )
            {
                std::visit(
                    [ & ]( const auto& behaviour )
                    {
                        using T = std::decay_t<decltype( behaviour )>;

                        if constexpr( std::is_same_v<T, Behaviours::CellCycle> )
                        {
                            hasCellCycle = true;

                            // Threshold ordering: necrosisO2 < hypoxiaO2 < targetO2
                            if( behaviour.necrosisO2 >= behaviour.hypoxiaO2 )
                                result.AddError(
                                    "AgentGroup '" + group.GetName() +
                                    "': CellCycle necrosisO2 (" + std::to_string( behaviour.necrosisO2 ) +
                                    ") must be < hypoxiaO2 (" + std::to_string( behaviour.hypoxiaO2 ) +
                                    "). Required order: necrosisO2 < hypoxiaO2 < targetO2" );

                            if( behaviour.hypoxiaO2 >= behaviour.targetO2 )
                                result.AddError(
                                    "AgentGroup '" + group.GetName() +
                                    "': CellCycle hypoxiaO2 (" + std::to_string( behaviour.hypoxiaO2 ) +
                                    ") must be < targetO2 (" + std::to_string( behaviour.targetO2 ) +
                                    "). Required order: necrosisO2 < hypoxiaO2 < targetO2" );
                        }

                        if constexpr( std::is_same_v<T, Behaviours::ConsumeField> )
                            hasConsumeField = true;
                    },
                    record.behaviour );
            }

            // CellCycle without a ConsumeField means hypoxia/necrosis thresholds are dead code
            if( hasCellCycle && !hasConsumeField )
                result.AddWarning( "AgentGroup '" + group.GetName() +
                                   "': CellCycle is present but no ConsumeField is defined. "
                                   "Cells will never become Hypoxic or Necrotic." );
        }
    }

    void SimulationValidator::CheckCadherinAdhesion( const SimulationBlueprint& blueprint, ValidationResult& result )
    {
        for( const auto& group : blueprint.GetGroups() )
        {
            bool hasCadherin    = false;
            bool hasBiomechanics = false;
            const Behaviours::CadherinAdhesion* cadherin = nullptr;

            for( const auto& record : group.GetBehaviours() )
            {
                if( std::holds_alternative<Behaviours::CadherinAdhesion>( record.behaviour ) )
                {
                    hasCadherin = true;
                    cadherin    = &std::get<Behaviours::CadherinAdhesion>( record.behaviour );
                }
                if( std::holds_alternative<Behaviours::Biomechanics>( record.behaviour ) )
                    hasBiomechanics = true;
            }

            if( !hasCadherin )
                continue;

            if( !hasBiomechanics )
                result.AddError( "AgentGroup '" + group.GetName() +
                                 "': CadherinAdhesion requires Biomechanics to also be present on the same group." );

            if( cadherin->expressionRate < 0.0f )
                result.AddError( "AgentGroup '" + group.GetName() +
                                 "': CadherinAdhesion expressionRate must be >= 0 (got: " +
                                 std::to_string( cadherin->expressionRate ) + ")." );

            if( cadherin->degradationRate < 0.0f )
                result.AddError( "AgentGroup '" + group.GetName() +
                                 "': CadherinAdhesion degradationRate must be >= 0 (got: " +
                                 std::to_string( cadherin->degradationRate ) + ")." );

            if( cadherin->couplingStrength <= 0.0f )
                result.AddError( "AgentGroup '" + group.GetName() +
                                 "': CadherinAdhesion couplingStrength must be > 0 (got: " +
                                 std::to_string( cadherin->couplingStrength ) + ")." );

            const glm::vec4& expr = cadherin->targetExpression;
            if( expr.x < 0.0f || expr.x > 1.0f ||
                expr.y < 0.0f || expr.y > 1.0f ||
                expr.z < 0.0f || expr.z > 1.0f ||
                expr.w < 0.0f || expr.w > 1.0f )
                result.AddError( "AgentGroup '" + group.GetName() +
                                 "': CadherinAdhesion targetExpression components must be in [0, 1]." );
        }

        // Warn if any off-diagonal affinity matrix element is outside [-1, 1]
        const glm::mat4& m = blueprint.GetCadherinAffinityMatrix();
        for( int col = 0; col < 4; ++col )
            for( int row = 0; row < 4; ++row )
                if( row != col && ( m[ col ][ row ] < -1.0f || m[ col ][ row ] > 1.0f ) )
                {
                    result.AddWarning( "CadherinAffinityMatrix: off-diagonal element [" +
                                       std::to_string( row ) + "][" + std::to_string( col ) +
                                       "] = " + std::to_string( m[ col ][ row ] ) +
                                       " is outside [-1, 1]. This may produce unexpected adhesion forces." );
                    return; // one warning is enough
                }
    }

    void SimulationValidator::CheckCellPolarity( const SimulationBlueprint& blueprint, ValidationResult& result )
    {
        for( const auto& group : blueprint.GetGroups() )
        {
            bool hasCellPolarity  = false;
            bool hasBiomechanics  = false;
            const Behaviours::CellPolarity* polarity = nullptr;

            for( const auto& record : group.GetBehaviours() )
            {
                if( std::holds_alternative<Behaviours::CellPolarity>( record.behaviour ) )
                {
                    hasCellPolarity = true;
                    polarity        = &std::get<Behaviours::CellPolarity>( record.behaviour );
                }
                if( std::holds_alternative<Behaviours::Biomechanics>( record.behaviour ) )
                    hasBiomechanics = true;
            }

            if( !hasCellPolarity )
                continue;

            if( !hasBiomechanics )
                result.AddError( "AgentGroup '" + group.GetName() +
                                 "': CellPolarity requires Biomechanics to also be present on the same group." );

            if( polarity->regulationRate < 0.0f )
                result.AddError( "AgentGroup '" + group.GetName() +
                                 "': CellPolarity regulationRate must be >= 0 (got: " +
                                 std::to_string( polarity->regulationRate ) + ")." );

            // apicalRepulsion may be negative: in the Phase-3+ cord-hollowing
            // regime, apical-apical contacts become ACTIVELY repulsive (PODXL
            // electrostatic repulsion, Strilic 2009). Only warn on extreme
            // magnitudes that would destabilise integration.
            if( polarity->apicalRepulsion < -5.0f || polarity->apicalRepulsion > 5.0f )
                result.AddWarning( "AgentGroup '" + group.GetName() +
                                   "': CellPolarity apicalRepulsion = " +
                                   std::to_string( polarity->apicalRepulsion ) +
                                   " is outside the typical [-5, 5] range; negative values "
                                   "are biologically legal (apical electrostatic repulsion)." );

            if( polarity->basalAdhesion < 0.0f )
                result.AddError( "AgentGroup '" + group.GetName() +
                                 "': CellPolarity basalAdhesion must be >= 0 (got: " +
                                 std::to_string( polarity->basalAdhesion ) + ")." );
        }
    }

    void SimulationValidator::CheckBasementMembrane( const SimulationBlueprint& blueprint, ValidationResult& result )
    {
        // Count groups carrying BasementMembrane. One global plate per simulation is
        // supported in Phase 2; warn if multiple groups declare one (the builder
        // will pick the first).
        int plateCount = 0;
        for( const auto& group : blueprint.GetGroups() )
        {
            for( const auto& record : group.GetBehaviours() )
            {
                if( std::holds_alternative<Behaviours::BasementMembrane>( record.behaviour ) )
                {
                    const auto& bm = std::get<Behaviours::BasementMembrane>( record.behaviour );
                    ++plateCount;

                    const float nLen = glm::length( bm.planeNormal );
                    if( nLen < 0.001f )
                        result.AddError( "AgentGroup '" + group.GetName() +
                                         "': BasementMembrane planeNormal must be non-zero." );
                    else if( std::abs( nLen - 1.0f ) > 0.05f )
                        result.AddWarning( "AgentGroup '" + group.GetName() +
                                           "': BasementMembrane planeNormal is not unit length (|n| = " +
                                           Fmt2f( nLen ) + "); it will be normalised by the shader." );

                    if( bm.contactStiffness <= 0.0f )
                        result.AddError( "AgentGroup '" + group.GetName() +
                                         "': BasementMembrane contactStiffness must be > 0 (got: " +
                                         Fmt2f( bm.contactStiffness ) + ")." );

                    if( bm.integrinAdhesion < 0.0f )
                        result.AddError( "AgentGroup '" + group.GetName() +
                                         "': BasementMembrane integrinAdhesion must be >= 0 (got: " +
                                         Fmt2f( bm.integrinAdhesion ) + ")." );

                    if( bm.anchorageDistance <= 0.0f )
                        result.AddError( "AgentGroup '" + group.GetName() +
                                         "': BasementMembrane anchorageDistance must be > 0 (got: " +
                                         Fmt2f( bm.anchorageDistance ) + ")." );

                    if( bm.polarityBias < 0.0f )
                        result.AddError( "AgentGroup '" + group.GetName() +
                                         "': BasementMembrane polarityBias must be >= 0 (got: " +
                                         Fmt2f( bm.polarityBias ) + ")." );
                }
            }
        }

        if( plateCount > 1 )
            result.AddWarning( "BasementMembrane declared on more than one AgentGroup (" +
                               std::to_string( plateCount ) +
                               "); only one global plate is supported — the builder will use the first." );
    }

    void SimulationValidator::CheckPopulations( const SimulationBlueprint& blueprint, ValidationResult& result )
    {
        if( blueprint.GetGroups().empty() )
            result.AddWarning( "No AgentGroups defined. Running a pure diffusion simulation." );

        for( const auto& group : blueprint.GetGroups() )
        {
            if( group.GetCount() == 0 )
                result.AddWarning( "AgentGroup '" + group.GetName() +
                                   "' has count == 0. It will be empty at simulation start." );

            if( group.GetBehaviours().empty() )
                result.AddWarning( "AgentGroup '" + group.GetName() +
                                   "' has no behaviours. Agents will be static." );
        }

        for( const auto& field : blueprint.GetGridFields() )
        {
            if( field.GetDiffusionCoefficient() == 0.0f && field.GetDecayRate() == 0.0f )
                result.AddWarning( "GridField '" + field.GetName() +
                                   "' has diffusionCoefficient == 0 and decayRate == 0. "
                                   "The field will remain static after initialization." );
        }
    }

} // namespace DigitalTwin
