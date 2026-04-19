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
    void SetupTissueSortingDemo    ( DigitalTwin::SimulationBlueprint& blueprint );
    void SetupECBlobDemo           ( DigitalTwin::SimulationBlueprint& blueprint );
    void SetupEC2DMatrigelDemo     ( DigitalTwin::SimulationBlueprint& blueprint );
    void SetupECTubeDemo           ( DigitalTwin::SimulationBlueprint& blueprint );
    void SetupECContactDemo        ( DigitalTwin::SimulationBlueprint& blueprint );
    void SetupCellMechanicsZooDemo ( DigitalTwin::SimulationBlueprint& blueprint );
    void SetupStressTestDemo       ( DigitalTwin::SimulationBlueprint& blueprint );

    // Vessels (Item 2)
    void SetupTwoShapeDemo           ( DigitalTwin::SimulationBlueprint& blueprint );
    void SetupPuzzlePiecePaletteDemo ( DigitalTwin::SimulationBlueprint& blueprint );
    void SetupStraightTubeDemo       ( DigitalTwin::SimulationBlueprint& blueprint );
    void SetupCurvedTubeDemo         ( DigitalTwin::SimulationBlueprint& blueprint );
    void SetupTaperingTubeDemo       ( DigitalTwin::SimulationBlueprint& blueprint );

    // Shared helper — defined in ECBlobDemo.cpp, reused by EC2DMatrigelDemo.cpp.
    // Populates the given AgentGroup with an elongated random cloud of ~100
    // endothelial cells (CurvedTile morphology, oriented outward from the
    // cluster centroid). Caller composes the behaviour stack after calling.
    // Same initial distribution for both demos — matches real experimental
    // practice where the same cell suspension is pipetted onto either plain
    // medium (ECBlobDemo) or a Matrigel-coated dish (EC2DMatrigelDemo).
    void SeedECCloud( DigitalTwin::AgentGroup& group, uint32_t seed = 42 );
} // namespace Gaudi::Demos
