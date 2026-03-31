#include "simulation/SimulationValidator.h"

#include "simulation/AgentGroup.h"
#include "simulation/Behaviours.h"
#include "simulation/GridField.h"
#include "simulation/SimulationBlueprint.h"

#include <unordered_set>

namespace DigitalTwin
{
    // Valid requiredLifecycleState values: -1 (no requirement) or 0-4 (Live through Necrotic).
    // Dead_PendingRemoval (5) is a transient GPU-side state — behaviours conditioned on it never fire.
    static constexpr int k_MinRequiredState = -1;
    static constexpr int k_MaxRequiredState = 4;
    static constexpr int k_MaxCellType      = 3; // PhalanxCell

    ValidationResult SimulationValidator::Validate( const SimulationBlueprint& blueprint )
    {
        ValidationResult result;

        CheckDomain( blueprint, result );
        CheckNames( blueprint, result );
        CheckFieldReferences( blueprint, result );
        CheckBehaviourParams( blueprint, result );
        CheckCellCycleThresholds( blueprint, result );
        CheckPopulations( blueprint, result );

        return result;
    }

    void SimulationValidator::CheckDomain( const SimulationBlueprint& blueprint, ValidationResult& result )
    {
        const glm::vec3& size = blueprint.GetDomainSize();

        if( size.x <= 0.0f )
            result.AddError( "Domain size X axis must be > 0 (got: " + std::to_string( size.x ) + ")" );
        if( size.y <= 0.0f )
            result.AddError( "Domain size Y axis must be > 0 (got: " + std::to_string( size.y ) + ")" );
        if( size.z <= 0.0f )
            result.AddError( "Domain size Z axis must be > 0 (got: " + std::to_string( size.z ) + ")" );

        if( blueprint.GetVoxelSize() <= 0.0f )
            result.AddError( "Voxel size must be > 0 (got: " + std::to_string( blueprint.GetVoxelSize() ) + ")" );

        if( blueprint.GetSpatialPartitioning().cellSize <= 0.0f )
            result.AddError( "SpatialPartitioning cell size must be > 0 (got: " +
                             std::to_string( blueprint.GetSpatialPartitioning().cellSize ) + ")" );
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
                if( record.requiredLifecycleState < -1 || record.requiredLifecycleState > k_MaxRequiredState )
                    result.AddError( "AgentGroup '" + group.GetName() +
                                     "': BehaviourRecord requiredLifecycleState out of range (got: " +
                                     std::to_string( record.requiredLifecycleState ) +
                                     ", valid range: -1 to 4)" );

                if( record.requiredCellType < -1 || record.requiredCellType > k_MaxCellType )
                    result.AddError( "AgentGroup '" + group.GetName() +
                                     "': BehaviourRecord requiredCellType out of range (got: " +
                                     std::to_string( record.requiredCellType ) +
                                     ", valid range: -1 to 3)" );

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

                            if( behaviour.requiredLifecycleState < k_MinRequiredState ||
                                behaviour.requiredLifecycleState > k_MaxRequiredState )
                            {
                                result.AddError( "AgentGroup '" + group.GetName() + "': " + typeName +
                                                 " requiredLifecycleState is out of range (got: " +
                                                 std::to_string( behaviour.requiredLifecycleState ) +
                                                 ", valid range: -1 to 4)" );
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
                            if( behaviour.segmentCounts.empty() )
                            {
                                result.AddError( "AgentGroup '" + group.GetName() +
                                                 "': VesselSeed segmentCounts must not be empty" );
                            }
                            else
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
