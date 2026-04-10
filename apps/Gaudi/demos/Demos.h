#pragma once
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
} // namespace Gaudi::Demos
