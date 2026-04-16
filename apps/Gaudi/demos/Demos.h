#pragma once
#include <simulation/AgentGroup.h>
#include <simulation/SimulationBlueprint.h>

namespace Gaudi::Demos
{
    void SetupEmptyBlueprint       ( DigitalTwin::SimulationBlueprint& blueprint );
    void SetupDiffusionDecayDemo   ( DigitalTwin::SimulationBlueprint& blueprint );
    void SetupBrownianMotionDemo   ( DigitalTwin::SimulationBlueprint& blueprint );
    void SetupJKRPackingDemo       ( DigitalTwin::SimulationBlueprint& blueprint );
    void SetupLifecycleDemo        ( DigitalTwin::SimulationBlueprint& blueprint );
    void SetupSecreteDemo          ( DigitalTwin::SimulationBlueprint& blueprint );
    void SetupConsumeDemo          ( DigitalTwin::SimulationBlueprint& blueprint );
    void SetupChemotaxisDemo       ( DigitalTwin::SimulationBlueprint& blueprint );
    void SetupCellCycleDemo        ( DigitalTwin::SimulationBlueprint& blueprint );
    void SetupSimpleVesselDebugDemo( DigitalTwin::SimulationBlueprint& blueprint );
    void SetupStaticVesselTreeDemo ( DigitalTwin::SimulationBlueprint& blueprint );
    void SetupVesselSproutingDemo  ( DigitalTwin::SimulationBlueprint& blueprint );
    void SetupAngiogenesisDemo     ( DigitalTwin::SimulationBlueprint& blueprint );
    void SetupTissueSortingDemo    ( DigitalTwin::SimulationBlueprint& blueprint );
    void SetupECBlobDemo           ( DigitalTwin::SimulationBlueprint& blueprint );
    void SetupECTubeDemo           ( DigitalTwin::SimulationBlueprint& blueprint );
    void SetupECContactDemo        ( DigitalTwin::SimulationBlueprint& blueprint );
    void SetupCellMechanicsZooDemo ( DigitalTwin::SimulationBlueprint& blueprint );
    void SetupStressTestDemo       ( DigitalTwin::SimulationBlueprint& blueprint );

    // Shared helper — defined in ECBlobDemo.cpp, reused by ECTubeDemo.cpp.
    // Populates the given AgentGroup with an elongated random cloud of ~100
    // endothelial cells (CurvedTile morphology, oriented outward from the
    // cluster centroid). Caller composes the behaviour stack after calling.
    void SeedECCloud( DigitalTwin::AgentGroup& group, uint32_t seed = 42 );
} // namespace Gaudi::Demos
