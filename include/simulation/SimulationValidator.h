#pragma once
#include "core/Core.h"
#include <string>
#include <vector>

namespace DigitalTwin
{
    class SimulationBlueprint;

    struct ValidationIssue
    {
        enum class Severity
        {
            Error,
            Warning
        };

        Severity    severity;
        std::string message;
    };

    struct DT_API ValidationResult
    {
        std::vector<ValidationIssue> issues;

        bool IsValid() const
        {
            for( const auto& issue : issues )
                if( issue.severity == ValidationIssue::Severity::Error )
                    return false;
            return true;
        }

        void AddError( std::string msg )
        {
            issues.push_back( { ValidationIssue::Severity::Error, std::move( msg ) } );
        }

        void AddWarning( std::string msg )
        {
            issues.push_back( { ValidationIssue::Severity::Warning, std::move( msg ) } );
        }
    };

    /**
     * @brief Validates a SimulationBlueprint before it is compiled into GPU state.
     * Returns a ValidationResult containing all errors and warnings found.
     * Errors block the build; warnings are logged but allow the build to proceed.
     */
    class DT_API SimulationValidator
    {
    public:
        static ValidationResult Validate( const SimulationBlueprint& blueprint );

    private:
        static void CheckDomain( const SimulationBlueprint&, ValidationResult& );
        static void CheckDomainBounds( const SimulationBlueprint&, ValidationResult& );
        static void CheckNames( const SimulationBlueprint&, ValidationResult& );
        static void CheckFieldReferences( const SimulationBlueprint&, ValidationResult& );
        static void CheckBehaviourParams( const SimulationBlueprint&, ValidationResult& );
        static void CheckCellCycleThresholds( const SimulationBlueprint&, ValidationResult& );
        static void CheckCadherinAdhesion( const SimulationBlueprint&, ValidationResult& );
        static void CheckCellPolarity( const SimulationBlueprint&, ValidationResult& );
        static void CheckBasementMembrane( const SimulationBlueprint&, ValidationResult& );
        static void CheckPopulations( const SimulationBlueprint&, ValidationResult& );
    };

} // namespace DigitalTwin
