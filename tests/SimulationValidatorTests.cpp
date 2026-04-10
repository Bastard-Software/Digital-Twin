#include "simulation/SimulationValidator.h"

#include "simulation/AgentGroup.h"
#include "simulation/Behaviours.h"
#include "simulation/SimulationBlueprint.h"

#include <gtest/gtest.h>

using namespace DigitalTwin;

// Builds a fully valid blueprint that passes all checks.
// Use this as a baseline and selectively break individual properties per test.
static SimulationBlueprint MakeValidBlueprint()
{
    SimulationBlueprint bp;
    bp.SetDomainSize( { 100.0f, 100.0f, 100.0f }, 2.0f );

    bp.AddGridField( "Oxygen" )
        .SetDiffusionCoefficient( 0.1f )
        .SetDecayRate( 0.01f );

    AgentGroup& cells = bp.AddAgentGroup( "TumorCells" );
    cells.SetCount( 10 );
    cells.AddBehaviour( Behaviours::ConsumeField{ "Oxygen", 1.0f } );
    cells.AddBehaviour( Behaviours::CellCycle{ 0.001f, 0.8f, 10.0f, 0.1f, 0.3f, 0.0001f } );

    return bp;
}

// 1. Fully valid blueprint produces no issues
TEST( SimulationValidatorTest, ValidBlueprint_ReturnsNoIssues )
{
    ValidationResult result = SimulationValidator::Validate( MakeValidBlueprint() );

    EXPECT_TRUE( result.IsValid() );
    EXPECT_TRUE( result.issues.empty() );
}

// 2. Blueprint with only GridFields and no AgentGroups is valid (warning only)
TEST( SimulationValidatorTest, ValidBlueprint_PureDiffusion_NoAgents )
{
    SimulationBlueprint bp;
    bp.SetDomainSize( { 100.0f, 100.0f, 100.0f }, 2.0f );
    bp.AddGridField( "Oxygen" ).SetDiffusionCoefficient( 0.1f ).SetDecayRate( 0.01f );

    ValidationResult result = SimulationValidator::Validate( bp );

    EXPECT_TRUE( result.IsValid() );
    EXPECT_FALSE( result.issues.empty() );
    EXPECT_EQ( result.issues.front().severity, ValidationIssue::Severity::Warning );
}

// 3. AgentGroup with count == 0 produces a warning, not an error
TEST( SimulationValidatorTest, ValidBlueprint_ZeroCountGroup )
{
    SimulationBlueprint bp = MakeValidBlueprint();
    bp.GetGroups(); // access to confirm setup; count is set below via a fresh blueprint
    SimulationBlueprint bp2;
    bp2.SetDomainSize( { 100.0f, 100.0f, 100.0f }, 2.0f );
    bp2.AddGridField( "Oxygen" ).SetDiffusionCoefficient( 0.1f ).SetDecayRate( 0.01f );
    bp2.AddAgentGroup( "Receivers" ).SetCount( 0 );

    ValidationResult result = SimulationValidator::Validate( bp2 );

    EXPECT_TRUE( result.IsValid() );
    bool hasCountWarning = false;
    for( const auto& issue : result.issues )
        if( issue.severity == ValidationIssue::Severity::Warning &&
            issue.message.find( "count == 0" ) != std::string::npos )
            hasCountWarning = true;
    EXPECT_TRUE( hasCountWarning );
}

// 4. Domain size X == 0 produces an error
TEST( SimulationValidatorTest, DomainSizeZeroX_ReturnsError )
{
    SimulationBlueprint bp = MakeValidBlueprint();
    bp.SetDomainSize( { 0.0f, 100.0f, 100.0f }, 2.0f );

    ValidationResult result = SimulationValidator::Validate( bp );

    EXPECT_FALSE( result.IsValid() );
}

// 5. Domain size Y < 0 produces an error
TEST( SimulationValidatorTest, DomainSizeNegativeY_ReturnsError )
{
    SimulationBlueprint bp = MakeValidBlueprint();
    bp.SetDomainSize( { 100.0f, -1.0f, 100.0f }, 2.0f );

    ValidationResult result = SimulationValidator::Validate( bp );

    EXPECT_FALSE( result.IsValid() );
}

// 6. Voxel size == 0 produces an error
TEST( SimulationValidatorTest, VoxelSizeZero_ReturnsError )
{
    SimulationBlueprint bp = MakeValidBlueprint();
    bp.SetDomainSize( { 100.0f, 100.0f, 100.0f }, 0.0f );

    ValidationResult result = SimulationValidator::Validate( bp );

    EXPECT_FALSE( result.IsValid() );
}

// 7. Voxel size < 0 produces an error
TEST( SimulationValidatorTest, VoxelSizeNegative_ReturnsError )
{
    SimulationBlueprint bp = MakeValidBlueprint();
    bp.SetDomainSize( { 100.0f, 100.0f, 100.0f }, -5.0f );

    ValidationResult result = SimulationValidator::Validate( bp );

    EXPECT_FALSE( result.IsValid() );
}

// 8. SpatialPartitioning cellSize == 0 produces an error
TEST( SimulationValidatorTest, SpatialPartitionCellSizeZero_ReturnsError )
{
    SimulationBlueprint bp = MakeValidBlueprint();
    bp.ConfigureSpatialPartitioning().SetCellSize( 0.0f );

    ValidationResult result = SimulationValidator::Validate( bp );

    EXPECT_FALSE( result.IsValid() );
}

// 9. Two AgentGroups with the same name produce an error
TEST( SimulationValidatorTest, DuplicateAgentGroupNames_ReturnsError )
{
    SimulationBlueprint bp = MakeValidBlueprint();
    bp.AddAgentGroup( "TumorCells" ).SetCount( 5 ); // duplicate

    ValidationResult result = SimulationValidator::Validate( bp );

    EXPECT_FALSE( result.IsValid() );
}

// 10. Two GridFields with the same name produce an error
TEST( SimulationValidatorTest, DuplicateGridFieldNames_ReturnsError )
{
    SimulationBlueprint bp = MakeValidBlueprint();
    bp.AddGridField( "Oxygen" ); // duplicate

    ValidationResult result = SimulationValidator::Validate( bp );

    EXPECT_FALSE( result.IsValid() );
}

// 11. An AgentGroup and a GridField sharing a name is valid (different namespaces)
TEST( SimulationValidatorTest, DuplicateGroupAndField_SameName_NoError )
{
    SimulationBlueprint bp;
    bp.SetDomainSize( { 100.0f, 100.0f, 100.0f }, 2.0f );
    bp.AddGridField( "Oxygen" ).SetDiffusionCoefficient( 0.1f ).SetDecayRate( 0.01f );
    bp.AddAgentGroup( "Oxygen" ).SetCount( 10 ); // same name as field — allowed

    ValidationResult result = SimulationValidator::Validate( bp );

    bool hasDuplicateError = false;
    for( const auto& issue : result.issues )
        if( issue.severity == ValidationIssue::Severity::Error &&
            issue.message.find( "Duplicate" ) != std::string::npos )
            hasDuplicateError = true;
    EXPECT_FALSE( hasDuplicateError );
}

// 12. ConsumeField referencing a non-existent field produces an error
TEST( SimulationValidatorTest, ConsumeField_MissingField_ReturnsError )
{
    SimulationBlueprint bp;
    bp.SetDomainSize( { 100.0f, 100.0f, 100.0f }, 2.0f );
    bp.AddAgentGroup( "Cells" ).SetCount( 10 )
      .AddBehaviour( Behaviours::ConsumeField{ "Glucose", 1.0f } ); // "Glucose" never declared

    ValidationResult result = SimulationValidator::Validate( bp );

    EXPECT_FALSE( result.IsValid() );
}

// 13. SecreteField referencing a non-existent field produces an error
TEST( SimulationValidatorTest, SecreteField_MissingField_ReturnsError )
{
    SimulationBlueprint bp;
    bp.SetDomainSize( { 100.0f, 100.0f, 100.0f }, 2.0f );
    bp.AddAgentGroup( "Cells" ).SetCount( 10 )
      .AddBehaviour( Behaviours::SecreteField{ "VEGF", 1.0f } ); // "VEGF" never declared

    ValidationResult result = SimulationValidator::Validate( bp );

    EXPECT_FALSE( result.IsValid() );
}

// 14. ConsumeField referencing an existing field produces no error
TEST( SimulationValidatorTest, ConsumeField_FieldExists_NoError )
{
    ValidationResult result = SimulationValidator::Validate( MakeValidBlueprint() );

    EXPECT_TRUE( result.IsValid() );
}

// 15. ConsumeField with an empty field name produces an error
TEST( SimulationValidatorTest, ConsumeField_EmptyFieldName_ReturnsError )
{
    SimulationBlueprint bp;
    bp.SetDomainSize( { 100.0f, 100.0f, 100.0f }, 2.0f );
    bp.AddGridField( "Oxygen" ).SetDiffusionCoefficient( 0.1f ).SetDecayRate( 0.01f );
    bp.AddAgentGroup( "Cells" ).SetCount( 10 )
      .AddBehaviour( Behaviours::ConsumeField{ "", 1.0f } );

    ValidationResult result = SimulationValidator::Validate( bp );

    EXPECT_FALSE( result.IsValid() );
}

// 16. BehaviourRecord with targetHz == 0 produces an error
TEST( SimulationValidatorTest, BehaviourHz_Zero_ReturnsError )
{
    SimulationBlueprint bp = MakeValidBlueprint();
    bp.GetGroups(); // blueprint has groups; add a bad one
    SimulationBlueprint bp2;
    bp2.SetDomainSize( { 100.0f, 100.0f, 100.0f }, 2.0f );
    bp2.AddGridField( "Oxygen" ).SetDiffusionCoefficient( 0.1f ).SetDecayRate( 0.01f );
    bp2.AddAgentGroup( "Cells" ).SetCount( 10 )
       .AddBehaviour( Behaviours::ConsumeField{ "Oxygen", 1.0f } ).SetHz( 0.0f );

    ValidationResult result = SimulationValidator::Validate( bp2 );

    EXPECT_FALSE( result.IsValid() );
}

// 17. BehaviourRecord with targetHz < 0 produces an error
TEST( SimulationValidatorTest, BehaviourHz_Negative_ReturnsError )
{
    SimulationBlueprint bp;
    bp.SetDomainSize( { 100.0f, 100.0f, 100.0f }, 2.0f );
    bp.AddGridField( "Oxygen" ).SetDiffusionCoefficient( 0.1f ).SetDecayRate( 0.01f );
    bp.AddAgentGroup( "Cells" ).SetCount( 10 )
      .AddBehaviour( Behaviours::ConsumeField{ "Oxygen", 1.0f } ).SetHz( -10.0f );

    ValidationResult result = SimulationValidator::Validate( bp );

    EXPECT_FALSE( result.IsValid() );
}

// 18. Biomechanics with maxRadius == 0 produces an error
TEST( SimulationValidatorTest, Biomechanics_MaxRadiusZero_ReturnsError )
{
    SimulationBlueprint bp;
    bp.SetDomainSize( { 100.0f, 100.0f, 100.0f }, 2.0f );
    bp.AddAgentGroup( "Cells" ).SetCount( 10 )
      .AddBehaviour( Behaviours::Biomechanics{ 15.0f, 2.0f, 0.0f } );

    ValidationResult result = SimulationValidator::Validate( bp );

    EXPECT_FALSE( result.IsValid() );
}

// 19. Biomechanics with maxRadius < 0 produces an error
TEST( SimulationValidatorTest, Biomechanics_MaxRadiusNegative_ReturnsError )
{
    SimulationBlueprint bp;
    bp.SetDomainSize( { 100.0f, 100.0f, 100.0f }, 2.0f );
    bp.AddAgentGroup( "Cells" ).SetCount( 10 )
      .AddBehaviour( Behaviours::Biomechanics{ 15.0f, 2.0f, -1.0f } );

    ValidationResult result = SimulationValidator::Validate( bp );

    EXPECT_FALSE( result.IsValid() );
}

// 20. ConsumeField with requiredState out of valid range produces an error
TEST( SimulationValidatorTest, ConsumeField_RequiredState_OutOfRange_ReturnsError )
{
    SimulationBlueprint bp;
    bp.SetDomainSize( { 100.0f, 100.0f, 100.0f }, 2.0f );
    bp.AddGridField( "Oxygen" ).SetDiffusionCoefficient( 0.1f ).SetDecayRate( 0.01f );
    bp.AddAgentGroup( "Cells" ).SetCount( 10 )
      .AddBehaviour( Behaviours::ConsumeField{ "Oxygen", 1.0f, static_cast<LifecycleState>( 99u ) } );

    ValidationResult result = SimulationValidator::Validate( bp );

    EXPECT_FALSE( result.IsValid() );
}

// 21. ConsumeField with requiredState == Any (no filter) produces no error
TEST( SimulationValidatorTest, ConsumeField_RequiredState_Any_NoError )
{
    ValidationResult result = SimulationValidator::Validate( MakeValidBlueprint() );

    EXPECT_TRUE( result.IsValid() );
}

// 22. ConsumeField with requiredState == Necrotic (4) produces no error
TEST( SimulationValidatorTest, ConsumeField_RequiredState_Four_NoError )
{
    SimulationBlueprint bp;
    bp.SetDomainSize( { 100.0f, 100.0f, 100.0f }, 2.0f );
    bp.AddGridField( "Oxygen" ).SetDiffusionCoefficient( 0.1f ).SetDecayRate( 0.01f );
    bp.AddAgentGroup( "Cells" ).SetCount( 10 )
      .AddBehaviour( Behaviours::ConsumeField{ "Oxygen", 1.0f, LifecycleState::Necrotic } );

    ValidationResult result = SimulationValidator::Validate( bp );

    EXPECT_TRUE( result.IsValid() );
}

// 23. CellCycle where necrosisO2 >= hypoxiaO2 produces an error
TEST( SimulationValidatorTest, CellCycle_NecrosisAboveHypoxia_ReturnsError )
{
    SimulationBlueprint bp;
    bp.SetDomainSize( { 100.0f, 100.0f, 100.0f }, 2.0f );
    bp.AddGridField( "Oxygen" ).SetDiffusionCoefficient( 0.1f ).SetDecayRate( 0.01f );
    AgentGroup& g = bp.AddAgentGroup( "Cells" );
    g.SetCount( 10 );
    g.AddBehaviour( Behaviours::ConsumeField{ "Oxygen", 1.0f } );
    g.AddBehaviour( Behaviours::CellCycle{ 0.001f, 0.8f, 10.0f, 0.5f, 0.3f, 0.0001f } ); // necrosisO2=0.5 > hypoxiaO2=0.3

    ValidationResult result = SimulationValidator::Validate( bp );

    EXPECT_FALSE( result.IsValid() );
}

// 24. CellCycle where hypoxiaO2 >= targetO2 produces an error
TEST( SimulationValidatorTest, CellCycle_HypoxiaAboveTarget_ReturnsError )
{
    SimulationBlueprint bp;
    bp.SetDomainSize( { 100.0f, 100.0f, 100.0f }, 2.0f );
    bp.AddGridField( "Oxygen" ).SetDiffusionCoefficient( 0.1f ).SetDecayRate( 0.01f );
    AgentGroup& g = bp.AddAgentGroup( "Cells" );
    g.SetCount( 10 );
    g.AddBehaviour( Behaviours::ConsumeField{ "Oxygen", 1.0f } );
    g.AddBehaviour( Behaviours::CellCycle{ 0.001f, 0.3f, 10.0f, 0.1f, 0.5f, 0.0001f } ); // hypoxiaO2=0.5 > targetO2=0.3

    ValidationResult result = SimulationValidator::Validate( bp );

    EXPECT_FALSE( result.IsValid() );
}

// 25. CellCycle where necrosisO2 == hypoxiaO2 produces an error (boundary case)
TEST( SimulationValidatorTest, CellCycle_NecrosisEqualsHypoxia_ReturnsError )
{
    SimulationBlueprint bp;
    bp.SetDomainSize( { 100.0f, 100.0f, 100.0f }, 2.0f );
    bp.AddGridField( "Oxygen" ).SetDiffusionCoefficient( 0.1f ).SetDecayRate( 0.01f );
    AgentGroup& g = bp.AddAgentGroup( "Cells" );
    g.SetCount( 10 );
    g.AddBehaviour( Behaviours::ConsumeField{ "Oxygen", 1.0f } );
    g.AddBehaviour( Behaviours::CellCycle{ 0.001f, 0.8f, 10.0f, 0.3f, 0.3f, 0.0001f } ); // necrosisO2 == hypoxiaO2

    ValidationResult result = SimulationValidator::Validate( bp );

    EXPECT_FALSE( result.IsValid() );
}

// 26. CellCycle with valid threshold ordering produces no error
TEST( SimulationValidatorTest, CellCycle_ValidThresholds_NoError )
{
    ValidationResult result = SimulationValidator::Validate( MakeValidBlueprint() );

    EXPECT_TRUE( result.IsValid() );
}

// 27. AgentGroup with count == 0 produces a warning but IsValid() remains true
TEST( SimulationValidatorTest, AgentGroup_ZeroCount_ReturnsWarning_NotError )
{
    SimulationBlueprint bp;
    bp.SetDomainSize( { 100.0f, 100.0f, 100.0f }, 2.0f );
    bp.AddAgentGroup( "Receivers" ).SetCount( 0 );

    ValidationResult result = SimulationValidator::Validate( bp );

    EXPECT_TRUE( result.IsValid() );
    bool hasWarning = false;
    for( const auto& issue : result.issues )
        if( issue.severity == ValidationIssue::Severity::Warning )
            hasWarning = true;
    EXPECT_TRUE( hasWarning );
}

// 28. Blueprint with no AgentGroups produces a warning but IsValid() remains true
TEST( SimulationValidatorTest, NoAgentGroups_ReturnsWarning_NotError )
{
    SimulationBlueprint bp;
    bp.SetDomainSize( { 100.0f, 100.0f, 100.0f }, 2.0f );
    bp.AddGridField( "Oxygen" ).SetDiffusionCoefficient( 0.1f ).SetDecayRate( 0.01f );

    ValidationResult result = SimulationValidator::Validate( bp );

    EXPECT_TRUE( result.IsValid() );
    bool hasWarning = false;
    for( const auto& issue : result.issues )
        if( issue.severity == ValidationIssue::Severity::Warning )
            hasWarning = true;
    EXPECT_TRUE( hasWarning );
}

// 29. AgentGroup with no behaviours produces a warning
TEST( SimulationValidatorTest, AgentGroup_NoBehaviours_ReturnsWarning )
{
    SimulationBlueprint bp;
    bp.SetDomainSize( { 100.0f, 100.0f, 100.0f }, 2.0f );
    bp.AddAgentGroup( "Obstacles" ).SetCount( 5 ); // no behaviours added

    ValidationResult result = SimulationValidator::Validate( bp );

    EXPECT_TRUE( result.IsValid() );
    bool hasNoBehavioursWarning = false;
    for( const auto& issue : result.issues )
        if( issue.severity == ValidationIssue::Severity::Warning &&
            issue.message.find( "no behaviours" ) != std::string::npos )
            hasNoBehavioursWarning = true;
    EXPECT_TRUE( hasNoBehavioursWarning );
}

// 30. Two BrownianMotion behaviours on the same group produce a warning
TEST( SimulationValidatorTest, DuplicateBehaviourType_InSameGroup_ReturnsWarning )
{
    SimulationBlueprint bp;
    bp.SetDomainSize( { 100.0f, 100.0f, 100.0f }, 2.0f );
    AgentGroup& g = bp.AddAgentGroup( "Cells" );
    g.SetCount( 10 );
    g.AddBehaviour( Behaviours::BrownianMotion{ 1.0f } );
    g.AddBehaviour( Behaviours::BrownianMotion{ 2.0f } ); // duplicate type

    ValidationResult result = SimulationValidator::Validate( bp );

    EXPECT_TRUE( result.IsValid() );
    bool hasDuplicateWarning = false;
    for( const auto& issue : result.issues )
        if( issue.severity == ValidationIssue::Severity::Warning &&
            issue.message.find( "duplicate behaviour type" ) != std::string::npos )
            hasDuplicateWarning = true;
    EXPECT_TRUE( hasDuplicateWarning );
}

// 31. CellCycle present without a ConsumeField in the same group produces a warning
TEST( SimulationValidatorTest, CellCycle_WithoutConsumeField_ReturnsWarning )
{
    SimulationBlueprint bp;
    bp.SetDomainSize( { 100.0f, 100.0f, 100.0f }, 2.0f );
    AgentGroup& g = bp.AddAgentGroup( "Cells" );
    g.SetCount( 10 );
    g.AddBehaviour( Behaviours::CellCycle{ 0.001f, 0.8f, 10.0f, 0.1f, 0.3f, 0.0001f } ); // no ConsumeField

    ValidationResult result = SimulationValidator::Validate( bp );

    EXPECT_TRUE( result.IsValid() );
    bool hasWarning = false;
    for( const auto& issue : result.issues )
        if( issue.severity == ValidationIssue::Severity::Warning &&
            issue.message.find( "ConsumeField" ) != std::string::npos )
            hasWarning = true;
    EXPECT_TRUE( hasWarning );
}

// 32. GridField with diffusionCoefficient == 0 and decayRate == 0 produces a warning
TEST( SimulationValidatorTest, StaticGridField_NoDiffusionNoDecay_ReturnsWarning )
{
    SimulationBlueprint bp;
    bp.SetDomainSize( { 100.0f, 100.0f, 100.0f }, 2.0f );
    bp.AddGridField( "Marker" ); // default: diffusion=0, decay=0

    ValidationResult result = SimulationValidator::Validate( bp );

    EXPECT_TRUE( result.IsValid() );
    bool hasStaticFieldWarning = false;
    for( const auto& issue : result.issues )
        if( issue.severity == ValidationIssue::Severity::Warning &&
            issue.message.find( "static" ) != std::string::npos )
            hasStaticFieldWarning = true;
    EXPECT_TRUE( hasStaticFieldWarning );
}

// 33. Blueprint with multiple errors reports all of them, not just the first
TEST( SimulationValidatorTest, MultipleErrors_AllReported )
{
    SimulationBlueprint bp;
    bp.SetDomainSize( { 0.0f, -1.0f, 100.0f }, -5.0f ); // 3 domain errors

    ValidationResult result = SimulationValidator::Validate( bp );

    EXPECT_FALSE( result.IsValid() );
    size_t errorCount = 0;
    for( const auto& issue : result.issues )
        if( issue.severity == ValidationIssue::Severity::Error )
            errorCount++;
    EXPECT_GE( errorCount, 3u );
}

// 34. Errors and warnings together still make IsValid() return false
TEST( SimulationValidatorTest, MixedErrorsAndWarnings_IsValidFalse )
{
    SimulationBlueprint bp;
    bp.SetDomainSize( { 100.0f, 100.0f, 100.0f }, 2.0f );
    bp.AddAgentGroup( "TumorCells" ).SetCount( 0 ); // warning: count==0
    bp.AddAgentGroup( "TumorCells" ).SetCount( 5 ); // error: duplicate name

    ValidationResult result = SimulationValidator::Validate( bp );

    EXPECT_FALSE( result.IsValid() );
    bool hasWarning = false;
    for( const auto& issue : result.issues )
        if( issue.severity == ValidationIssue::Severity::Warning )
            hasWarning = true;
    EXPECT_TRUE( hasWarning );
}

// 35. Warnings alone leave IsValid() true
TEST( SimulationValidatorTest, WarningsOnly_IsValidTrue )
{
    SimulationBlueprint bp;
    bp.SetDomainSize( { 100.0f, 100.0f, 100.0f }, 2.0f );
    bp.AddGridField( "Marker" ); // warning: static field
    // no AgentGroups: warning

    ValidationResult result = SimulationValidator::Validate( bp );

    EXPECT_TRUE( result.IsValid() );
    EXPECT_FALSE( result.issues.empty() );
}

// --- Chemotaxis Validator Tests ---

TEST( SimulationValidatorTests, Chemotaxis_ValidParams_NoError )
{
    SimulationBlueprint bp;
    bp.SetDomainSize( { 100.f, 100.f, 100.f }, 2.f );
    bp.AddGridField( "VEGF" ).SetDiffusionCoefficient( 2.0f );
    bp.AddAgentGroup( "EC" ).SetCount( 10 )
      .AddBehaviour( Behaviours::Chemotaxis{ "VEGF", 1.0f, 0.01f, 5.0f } );
    EXPECT_TRUE( SimulationValidator::Validate( bp ).IsValid() );
}

TEST( SimulationValidatorTests, Chemotaxis_UnknownField_ReturnsError )
{
    SimulationBlueprint bp;
    bp.SetDomainSize( { 100.f, 100.f, 100.f }, 2.f );
    // No VEGF field declared
    bp.AddAgentGroup( "EC" ).SetCount( 10 )
      .AddBehaviour( Behaviours::Chemotaxis{ "VEGF", 1.0f, 0.01f, 5.0f } );
    EXPECT_FALSE( SimulationValidator::Validate( bp ).IsValid() );
}

TEST( SimulationValidatorTests, Chemotaxis_ZeroSensitivity_ReturnsError )
{
    SimulationBlueprint bp;
    bp.SetDomainSize( { 100.f, 100.f, 100.f }, 2.f );
    bp.AddGridField( "VEGF" ).SetDiffusionCoefficient( 2.0f );
    bp.AddAgentGroup( "EC" ).SetCount( 10 )
      .AddBehaviour( Behaviours::Chemotaxis{ "VEGF", 0.0f, 0.01f, 5.0f } ); // sensitivity=0
    EXPECT_FALSE( SimulationValidator::Validate( bp ).IsValid() );
}

TEST( SimulationValidatorTests, Chemotaxis_ZeroMaxVelocity_ReturnsError )
{
    SimulationBlueprint bp;
    bp.SetDomainSize( { 100.f, 100.f, 100.f }, 2.f );
    bp.AddGridField( "VEGF" ).SetDiffusionCoefficient( 2.0f );
    bp.AddAgentGroup( "EC" ).SetCount( 10 )
      .AddBehaviour( Behaviours::Chemotaxis{ "VEGF", 1.0f, 0.01f, 0.0f } ); // maxVelocity=0
    EXPECT_FALSE( SimulationValidator::Validate( bp ).IsValid() );
}

TEST( SimulationValidatorTests, Chemotaxis_NegativeSaturation_ReturnsError )
{
    SimulationBlueprint bp;
    bp.SetDomainSize( { 100.f, 100.f, 100.f }, 2.f );
    bp.AddGridField( "VEGF" ).SetDiffusionCoefficient( 2.0f );
    bp.AddAgentGroup( "EC" ).SetCount( 10 )
      .AddBehaviour( Behaviours::Chemotaxis{ "VEGF", 1.0f, -0.1f, 5.0f } ); // saturation<0
    EXPECT_FALSE( SimulationValidator::Validate( bp ).IsValid() );
}

TEST( SimulationValidatorTests, Chemotaxis_ZeroSaturation_NoError )
{
    // saturation=0 means linear response — valid (just potentially unstable at high gradients)
    SimulationBlueprint bp;
    bp.SetDomainSize( { 100.f, 100.f, 100.f }, 2.f );
    bp.AddGridField( "VEGF" ).SetDiffusionCoefficient( 2.0f );
    bp.AddAgentGroup( "EC" ).SetCount( 10 )
      .AddBehaviour( Behaviours::Chemotaxis{ "VEGF", 1.0f, 0.0f, 5.0f } );
    EXPECT_TRUE( SimulationValidator::Validate( bp ).IsValid() );
}

// --- VesselSeed Validator Tests ---

// VesselSeed with empty segmentCounts produces an error
TEST( SimulationValidatorTests, VesselSeed_EmptySegmentCounts_ReturnsError )
{
    SimulationBlueprint bp;
    bp.SetDomainSize( { 100.f, 100.f, 100.f }, 2.f );
    Behaviours::VesselSeed seed; // segmentCounts is empty by default
    bp.AddAgentGroup( "Vessel" ).SetCount( 10 ).AddBehaviour( seed );
    EXPECT_FALSE( SimulationValidator::Validate( bp ).IsValid() );
}

// VesselSeed with a zero entry in segmentCounts produces an error
TEST( SimulationValidatorTests, VesselSeed_ZeroSegmentCount_ReturnsError )
{
    SimulationBlueprint bp;
    bp.SetDomainSize( { 100.f, 100.f, 100.f }, 2.f );
    Behaviours::VesselSeed seed;
    seed.segmentCounts = { 5u, 0u, 3u };
    bp.AddAgentGroup( "Vessel" ).SetCount( 10 ).AddBehaviour( seed );
    EXPECT_FALSE( SimulationValidator::Validate( bp ).IsValid() );
}

// VesselSeed where sum of segmentCounts exceeds group count produces an error
TEST( SimulationValidatorTests, VesselSeed_SumExceedsGroupCount_ReturnsError )
{
    SimulationBlueprint bp;
    bp.SetDomainSize( { 100.f, 100.f, 100.f }, 2.f );
    Behaviours::VesselSeed seed;
    seed.segmentCounts = { 6u, 6u }; // sum=12 > group count 10
    bp.AddAgentGroup( "Vessel" ).SetCount( 10 ).AddBehaviour( seed );
    EXPECT_FALSE( SimulationValidator::Validate( bp ).IsValid() );
}

// VesselSeed with valid segmentCounts (sum == group count) produces no error
TEST( SimulationValidatorTests, VesselSeed_ValidConfig_NoError )
{
    SimulationBlueprint bp;
    bp.SetDomainSize( { 100.f, 100.f, 100.f }, 2.f );
    Behaviours::VesselSeed seed;
    seed.segmentCounts = { 5u, 5u }; // sum=10, exactly matching group count
    bp.AddAgentGroup( "Vessel" ).SetCount( 10 ).AddBehaviour( seed );
    EXPECT_TRUE( SimulationValidator::Validate( bp ).IsValid() );
}

// ── CadherinAdhesion validator tests ────────────────────────────────────────

static SimulationBlueprint MakeCadherinBlueprint()
{
    SimulationBlueprint bp;
    bp.SetDomainSize( { 100.0f, 100.0f, 100.0f }, 2.0f );
    AgentGroup& g = bp.AddAgentGroup( "Cells" );
    g.SetCount( 10 );
    g.AddBehaviour( Behaviours::Biomechanics{ 15.0f, 2.0f, 1.5f, 0.0f } );
    g.AddBehaviour( Behaviours::CadherinAdhesion{
        glm::vec4( 0.0f, 0.0f, 1.0f, 0.0f ), 0.01f, 0.001f, 1.0f } );
    return bp;
}

TEST( SimulationValidatorTest, Cadherin_ValidSetup_Passes )
{
    EXPECT_TRUE( SimulationValidator::Validate( MakeCadherinBlueprint() ).IsValid() );
}

TEST( SimulationValidatorTest, Cadherin_RequiresBiomechanics )
{
    SimulationBlueprint bp;
    bp.SetDomainSize( { 100.0f, 100.0f, 100.0f }, 2.0f );
    bp.AddAgentGroup( "Cells" ).SetCount( 10 )
        .AddBehaviour( Behaviours::CadherinAdhesion{} ); // no Biomechanics

    EXPECT_FALSE( SimulationValidator::Validate( bp ).IsValid() );
}

TEST( SimulationValidatorTest, Cadherin_NegativeExpressionRate_Fails )
{
    SimulationBlueprint bp = MakeCadherinBlueprint();
    auto& cadherin = std::get<Behaviours::CadherinAdhesion>(
        bp.GetGroupsMutable()[ 0 ].GetBehavioursMutable()[ 1 ].behaviour );
    cadherin.expressionRate = -0.01f;

    EXPECT_FALSE( SimulationValidator::Validate( bp ).IsValid() );
}

TEST( SimulationValidatorTest, Cadherin_NegativeDegradationRate_Fails )
{
    SimulationBlueprint bp = MakeCadherinBlueprint();
    auto& cadherin = std::get<Behaviours::CadherinAdhesion>(
        bp.GetGroupsMutable()[ 0 ].GetBehavioursMutable()[ 1 ].behaviour );
    cadherin.degradationRate = -0.001f;

    EXPECT_FALSE( SimulationValidator::Validate( bp ).IsValid() );
}

TEST( SimulationValidatorTest, Cadherin_ZeroCouplingStrength_Fails )
{
    SimulationBlueprint bp = MakeCadherinBlueprint();
    auto& cadherin = std::get<Behaviours::CadherinAdhesion>(
        bp.GetGroupsMutable()[ 0 ].GetBehavioursMutable()[ 1 ].behaviour );
    cadherin.couplingStrength = 0.0f;

    EXPECT_FALSE( SimulationValidator::Validate( bp ).IsValid() );
}

TEST( SimulationValidatorTest, Cadherin_ExpressionOutOfRange_Fails )
{
    SimulationBlueprint bp = MakeCadherinBlueprint();
    auto& cadherin = std::get<Behaviours::CadherinAdhesion>(
        bp.GetGroupsMutable()[ 0 ].GetBehavioursMutable()[ 1 ].behaviour );
    cadherin.targetExpression = glm::vec4( 0.0f, 1.5f, 0.0f, 0.0f ); // y > 1

    EXPECT_FALSE( SimulationValidator::Validate( bp ).IsValid() );
}
