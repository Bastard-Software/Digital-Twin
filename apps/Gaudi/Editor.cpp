#include "Editor.h"

#include "IconsFontAwesome5.h"
#include <core/FileSystem.h>
#include <core/Log.h>
#include <imgui.h>
#include <random>
#include <simulation/BiologyGenerator.h>
#include <simulation/BiomechanicsGenerator.h>
#include <simulation/MorphologyGenerator.h>
#include <simulation/Phenotype.h>
#include <simulation/SpatialDistribution.h>

namespace Gaudi
{
    void Editor::Init()
    {
        DigitalTwin::DigitalTwinConfig config;
        config.headless        = false;
        config.windowDesc.mode = DigitalTwin::WindowMode::FULLSCREEN_WINDOWED;
        m_engine.Initialize( config );

        ImGui::SetCurrentContext( static_cast<ImGuiContext*>( m_engine.GetImGuiContext() ) );
        ImGuiIO&    io      = ImGui::GetIO();
        auto*       fs      = m_engine.GetFileSystem();
        std::string roboto  = fs->ResolvePath( "fonts/Roboto-Medium.ttf" ).string();
        std::string faSolid = fs->ResolvePath( "fonts/fa-solid-900.ttf" ).string();
        io.Fonts->AddFontFromFileTTF( roboto.c_str(), 16.0f );
        ImFontConfig iconCfg;
        iconCfg.MergeMode                 = true;
        iconCfg.GlyphMinAdvanceX          = 16.0f;
        iconCfg.PixelSnapH                = true;
        static const ImWchar iconRanges[] = { ICON_MIN_FA, ICON_MAX_FA, 0 };
        io.Fonts->AddFontFromFileTTF( faSolid.c_str(), 16.0f, &iconCfg, iconRanges );

        // SetupInitialBlueprint();
        // SetupChemotaxisDemo();
        // SetupCellCycleDemo();
        // SetupAngiogenesisDemo();
        SetupSimpleVesselDebugDemo();
    }

    void Editor::SetupInitialBlueprint()
    {
        // ==========================================================================================
        // 1. Domain & Spatial Partitioning
        // ==========================================================================================
        m_blueprint.SetName( "Angiogenesis" );
        m_blueprint.SetDomainSize( glm::vec3( 30.0f ), 2.0f );

        m_blueprint.ConfigureSpatialPartitioning()
            .SetMethod( DigitalTwin::SpatialPartitioningMethod::HashGrid )
            .SetCellSize( 3.0f )
            .SetMaxDensity( 64 )
            .SetComputeHz( 60.0f );

        // ==========================================================================================
        // 2. Environmental Fields
        // ==========================================================================================

        // Oxygen: starts at 50 — enough for initial tumour growth, depletes as tumour expands.
        // Perfusion from StalkCells compensates near the vessel.
        m_blueprint.AddGridField( "Oxygen" )
            .SetInitializer( DigitalTwin::GridInitializer::Constant( 50.0f ) )
            .SetDiffusionCoefficient( 5.0f )
            .SetDecayRate( 0.0f )
            .SetComputeHz( 60.0f );

        // VEGF: starts at 0 — builds organically when tumour core goes hypoxic.
        m_blueprint.AddGridField( "VEGF" )
            .SetInitializer( DigitalTwin::GridInitializer::Constant( 0.0f ) )
            .SetDiffusionCoefficient( 3.0f )
            .SetDecayRate( 0.02f )
            .SetComputeHz( 60.0f );

        // Lactate: tumour metabolic waste cleared by vessel StalkCells
        m_blueprint.AddGridField( "Lactate" )
            .SetInitializer( DigitalTwin::GridInitializer::Constant( 0.0f ) )
            .SetDiffusionCoefficient( 1.5f )
            .SetDecayRate( 0.005f )
            .SetComputeHz( 60.0f );

        // ==========================================================================================
        // 3. Tumour cells — dense cluster at origin (20 cells, radius 4)
        // ==========================================================================================
        auto tumorPositions = DigitalTwin::SpatialDistribution::UniformInSphere( 10, 3.0f );

        auto& tumorCells = m_blueprint.AddAgentGroup( "TumorCells" )
                               .SetCount( 10 )
                               .SetMorphology( DigitalTwin::MorphologyGenerator::CreateSphere( 1.5f ) )
                               .SetDistribution( tumorPositions )
                               .SetColor( glm::vec4( 0.2f, 0.85f, 0.3f, 1.0f ) ); // Bright green

        // Brownian jitter
        tumorCells.AddBehaviour( DigitalTwin::Behaviours::BrownianMotion{ 0.3f } ).SetHz( 60.0f );

        // O2 consumption — aggressive enough to drive core into hypoxia
        tumorCells.AddBehaviour( DigitalTwin::Behaviours::ConsumeField{ "Oxygen", 25.0f } ).SetHz( 60.0f );

        // Lactate secretion — always on (aerobic glycolysis + hypoxic switch)
        tumorCells.AddBehaviour( DigitalTwin::Behaviours::SecreteField{ "Lactate", 40.0f } ).SetHz( 60.0f );

        // VEGF SOS — only when hypoxic
        tumorCells.AddBehaviour( DigitalTwin::Behaviours::SecreteField{ "VEGF", 120.0f, static_cast<int>( DigitalTwin::LifecycleState::Hypoxic ) } )
            .SetHz( 60.0f );

        // Cell cycle: O2 starts at 50 so cells grow immediately; core goes hypoxic at 25
        // after tumour expands and O2 is consumed.
        tumorCells
            .AddBehaviour( DigitalTwin::BiologyGenerator::StandardCellCycle()
                               .SetBaseDoublingTime( 2.0f / 3600.0f )
                               .SetProliferationOxygenTarget( 50.0f )
                               .SetArrestPressureThreshold( 15.0f )
                               .SetHypoxiaOxygenThreshold( 25.0f )
                               .SetNecrosisOxygenThreshold( 12.0f )
                               .SetApoptosisRate( 0.0f )
                               .Build() )
            .SetHz( 60.0f );

        // JKR biomechanics
        tumorCells
            .AddBehaviour( DigitalTwin::BiomechanicsGenerator::JKR()
                               .SetYoungsModulus( 20.0f )
                               .SetPoissonRatio( 0.4f )
                               .SetAdhesionEnergy( 1.5f )
                               .SetMaxInteractionRadius( 1.5f )
                               .Build() )
            .SetHz( 60.0f );

        // ==========================================================================================
        // 4. Endothelial cells — two vessel lines at y=±15 flanking tumour
        //    18-unit span / 14 gaps ≈ 1.29 spacing < cellSize/2=1.5 → Notch-Dll4 inhibition works
        // ==========================================================================================
        auto vesselTop    = DigitalTwin::SpatialDistribution::VesselLine( 10, glm::vec3( -6, 8, 0 ), glm::vec3( 6, 8, 0 ) );
        auto vesselBottom = DigitalTwin::SpatialDistribution::VesselLine( 10, glm::vec3( -6, -8, 0 ), glm::vec3( 6, -8, 0 ) );
        vesselTop.insert( vesselTop.end(), vesselBottom.begin(), vesselBottom.end() );

        auto& endo = m_blueprint.AddAgentGroup( "EndothelialCells" )
                         .SetCount( 20 )
                         .SetMorphology( DigitalTwin::MorphologyGenerator::CreateSphere( 1.0f ) )
                         .SetDistribution( vesselTop )
                         .SetColor( glm::vec4( 1.0f, 0.3f, 0.3f, 1.0f ) ) // Red
                         .AddCellTypeMorphology( static_cast<int>( DigitalTwin::CellType::TipCell ),   DigitalTwin::MorphologyGenerator::CreateSpikySphere( 1.0f, 1.35f ) )
                         .AddCellTypeMorphology( static_cast<int>( DigitalTwin::CellType::StalkCell ), DigitalTwin::MorphologyGenerator::CreateCylinder( 0.6f, 2.2f ) );

        // 1. Notch-Dll4 lateral inhibition — differentiates Tip/Stalk
        endo.AddBehaviour( DigitalTwin::Behaviours::NotchDll4{
                /* dll4ProductionRate   */ 1.0f,
                /* dll4DecayRate        */ 0.1f,
                /* notchInhibitionGain  */ 20.0f,
                /* vegfr2BaseExpression */ 1.0f,
                /* tipThreshold         */ 0.55f,
                /* stalkThreshold       */ 0.3f } )
            .SetHz( 60.0f );

        // 2. Anastomosis — TipCell-TipCell contact fusion into StalkCells
        endo.AddBehaviour( DigitalTwin::Behaviours::Anastomosis{ /* contactDistance */ 1.0f } ).SetHz( 60.0f );

        // 3. Perfusion — StalkCells inject O2 (shader-guarded: cellType==StalkCell only)
        endo.AddBehaviour( DigitalTwin::Behaviours::Perfusion{ "Oxygen", 4.0f } ).SetHz( 60.0f );

        // 4. Drain — StalkCells clear tumour-secreted Lactate (shader-guarded: cellType==StalkCell only)
        endo.AddBehaviour( DigitalTwin::Behaviours::Drain{ "Lactate", 2.0f } ).SetHz( 60.0f );

        // 5. ConsumeField — light metabolic O2 drain (all endo cells)
        endo.AddBehaviour( DigitalTwin::Behaviours::ConsumeField{ "Oxygen", 3.0f } ).SetHz( 60.0f );

        // 6. Chemotaxis toward VEGF — TipCells only; StalkCells stay anchored
        endo.AddBehaviour( DigitalTwin::Behaviours::Chemotaxis{ "VEGF", 5.0f, 0.002f, 12.0f } )
            .SetHz( 60.0f )
            .SetRequiredCellType( static_cast<int>( DigitalTwin::CellType::TipCell ) );

        // 7. Brownian jitter — TipCells only
        endo.AddBehaviour( DigitalTwin::Behaviours::BrownianMotion{ 0.15f } )
            .SetHz( 60.0f )
            .SetRequiredCellType( static_cast<int>( DigitalTwin::CellType::TipCell ) );

        // 8. JKR biomechanics — fills pressure buffer for next frame's CellCycle
        endo.AddBehaviour( DigitalTwin::BiomechanicsGenerator::JKR()
                               .SetYoungsModulus( 15.0f )
                               .SetPoissonRatio( 0.4f )
                               .SetAdhesionEnergy( 1.0f )
                               .SetMaxInteractionRadius( 1.5f )
                               .Build() )
            .SetHz( 60.0f );

        m_engine.SetBlueprint( m_blueprint );
    }

    void Editor::AddPanel( std::shared_ptr<EditorPanel> panel )
    {
        m_panels.push_back( panel );
        panel->OnAttach();
    }

    void Editor::Run()
    {
        DT_INFO( "Starting Gaudi Editor Loop..." );
        ImGui::SetCurrentContext( static_cast<ImGuiContext*>( m_engine.GetImGuiContext() ) );

        while( !m_engine.IsWindowClosed() )
        {
            m_engine.BeginFrame();

            // Pass a lambda that iterates over our panel list
            m_engine.RenderUI( [ this ]() {
                for( auto& panel: m_panels )
                {
                    panel->OnUIRender();
                }
            } );

            m_engine.EndFrame();
        }
    }

    void Editor::Shutdown()
    {
        for( auto& panel: m_panels )
        {
            panel->OnDetach();
        }
        m_panels.clear();

        DT_INFO( "Gaudi Editor closing..." );
        m_engine.Shutdown();
    }
    void Editor::SetupCellCycleDemo()
    {
        m_blueprint = DigitalTwin::SimulationBlueprint{};

        // Small domain — easy to see everything
        m_blueprint.SetName( "CellCycle Demo" );
        m_blueprint.SetDomainSize( glm::vec3( 20.0f ), 2.0f );

        m_blueprint.ConfigureSpatialPartitioning()
            .SetMethod( DigitalTwin::SpatialPartitioningMethod::HashGrid )
            .SetCellSize( 3.0f )
            .SetMaxDensity( 64 )
            .SetComputeHz( 60.0f );

        // Oxygen: moderate initial level — will be consumed by tumor
        m_blueprint.AddGridField( "Oxygen" )
            .SetInitializer( DigitalTwin::GridInitializer::Constant( 40.0f ) )
            .SetDiffusionCoefficient( 2.0f )
            .SetDecayRate( 0.0f )
            .SetComputeHz( 60.0f );

        // 5 tumor cells in a tight cluster at origin
        auto tumorPositions = DigitalTwin::SpatialDistribution::UniformInSphere( 5, 2.0f );

        auto& tumor = m_blueprint.AddAgentGroup( "TumorCells" )
                           .SetCount( 5 )
                           .SetMorphology( DigitalTwin::MorphologyGenerator::CreateSphere( 1.5f ) )
                           .SetDistribution( tumorPositions )
                           .SetColor( glm::vec4( 0.2f, 0.85f, 0.3f, 1.0f ) ); // Green

        // Brownian jitter — small, just for visual life
        tumor.AddBehaviour( DigitalTwin::Behaviours::BrownianMotion{ 0.2f } ).SetHz( 60.0f );

        // O2 consumption — aggressive, drives hypoxia quickly in small domain
        tumor.AddBehaviour( DigitalTwin::Behaviours::ConsumeField{ "Oxygen", 30.0f } ).SetHz( 60.0f );

        // CellCycle: division → hypoxia → necrosis
        // Low diffusion + small domain means O2 depletes fast → visible lifecycle transitions
        tumor
            .AddBehaviour( DigitalTwin::BiologyGenerator::StandardCellCycle()
                               .SetBaseDoublingTime( 4.0f / 3600.0f )  // Fast division to see it happen
                               .SetProliferationOxygenTarget( 40.0f )
                               .SetArrestPressureThreshold( 15.0f )
                               .SetHypoxiaOxygenThreshold( 20.0f )     // Wide hypoxia window
                               .SetNecrosisOxygenThreshold( 8.0f )     // Well below hypoxia
                               .SetApoptosisRate( 0.0f )
                               .Build() )
            .SetHz( 60.0f );

        // JKR biomechanics — cells push each other apart
        tumor
            .AddBehaviour( DigitalTwin::BiomechanicsGenerator::JKR()
                               .SetYoungsModulus( 20.0f )
                               .SetPoissonRatio( 0.4f )
                               .SetAdhesionEnergy( 1.5f )
                               .SetMaxInteractionRadius( 1.5f )
                               .Build() )
            .SetHz( 60.0f );

        m_engine.SetBlueprint( m_blueprint );
    }

    void Editor::SetupAngiogenesisDemo()
    {
        m_blueprint = DigitalTwin::SimulationBlueprint{};

        m_blueprint.SetName( "Angiogenesis Demo" );
        m_blueprint.SetDomainSize( glm::vec3( 80.0f ), 2.0f );

        m_blueprint.ConfigureSpatialPartitioning()
            .SetMethod( DigitalTwin::SpatialPartitioningMethod::HashGrid )
            .SetCellSize( 4.0f )
            .SetMaxDensity( 64 )
            .SetComputeHz( 60.0f );

        // Oxygen — hypoxic core (O2=10) surrounded by normoxic tissue (O2=50).
        // Diffusion spreads O2 inward; perfusion after anastomosis raises core O2 above hypoxia threshold.
        m_blueprint.AddGridField( "Oxygen" )
            .SetInitializer( DigitalTwin::GridInitializer::Sphere( glm::vec3( 0.0f ), 8.0f, 10.0f, 50.0f ) )
            .SetDiffusionCoefficient( 5.0f )
            .SetDecayRate( 0.0f )
            .SetComputeHz( 60.0f );

        // VEGF — pre-seeded Gaussian so TipCells see a gradient from frame 1
        // sigma=12 (tighter than 15) sharpens the gradient; only inner endothelial cells activate first
        m_blueprint.AddGridField( "VEGF" )
            .SetInitializer( DigitalTwin::GridInitializer::Gaussian( glm::vec3( 0.0f ), 12.0f, 100.0f ) )
            .SetDiffusionCoefficient( 1.5f )
            .SetDecayRate( 0.05f )
            .SetComputeHz( 60.0f );

        // 5 hypoxic source cells spread at origin — continuously emit VEGF
        std::vector<glm::vec4> sourcePositions = {
            glm::vec4( -2.0f,  0.0f,  0.0f, 1.0f ),
            glm::vec4(  2.0f,  0.0f,  0.0f, 1.0f ),
            glm::vec4(  0.0f,  0.0f,  2.0f, 1.0f ),
            glm::vec4(  0.0f,  2.0f,  0.0f, 1.0f ),
            glm::vec4(  0.0f, -2.0f,  0.0f, 1.0f ),
        };

        auto& sources = m_blueprint.AddAgentGroup( "HypoxicCells" )
                             .SetCount( 5 )
                             .SetMorphology( DigitalTwin::MorphologyGenerator::CreateSphere( 1.2f ) )
                             .SetDistribution( sourcePositions )
                             .SetColor( glm::vec4( 0.3f, 0.0f, 0.8f, 1.0f ) ); // Purple

        // Oxygen consumption — cells deplete local O2, which drives the hypoxia state transition
        sources.AddBehaviour( DigitalTwin::Behaviours::ConsumeField{ "Oxygen", 5.0f } ).SetHz( 60.0f );

        // VEGF secretion only while Hypoxic (lifecycleState=2) — stops automatically when O2 recovers
        sources.AddBehaviour( DigitalTwin::Behaviours::SecreteField{ "VEGF", 30.0f, /*requiredLifecycleState=*/ 2 } ).SetHz( 60.0f );
        sources.AddBehaviour( DigitalTwin::BiomechanicsGenerator::JKR()
                                   .SetYoungsModulus( 40.0f )
                                   .SetPoissonRatio( 0.4f )
                                   .SetAdhesionEnergy( 0.0f )
                                   .SetMaxInteractionRadius( 1.5f )
                                   .SetDampingCoefficient( 20.0f )
                                   .Build() )
            .SetHz( 60.0f );
        // CellCycle: core O2=10 < hypoxiaO2=18 → cells in core go Hypoxic immediately.
        // Normoxic tissue (O2=50) stays above 18 → Live. When perfusion raises core O2 above 18 → recover.
        sources.AddBehaviour( DigitalTwin::BiologyGenerator::StandardCellCycle()
                                   .SetBaseDoublingTime( 1.0f )            // won't grow while Hypoxic
                                   .SetProliferationOxygenTarget( 60.0f )
                                   .SetArrestPressureThreshold( 999.0f )   // no pressure arrest
                                   .SetHypoxiaOxygenThreshold( 18.0f )     // below ambient (50) but above core (10)
                                   .SetNecrosisOxygenThreshold( 1.0f )
                                   .SetApoptosisRate( 0.0f )
                                   .Build() )
            .SetHz( 10.0f );

        // Two vessel lines at y=±25, 8 cells each — farther from centre gives a clear VEGF gradient
        // across the vessel; inner-most cell (x≈0) sees highest VEGF and wins the Notch competition
        auto vesselTop    = DigitalTwin::SpatialDistribution::VesselLine( 8, glm::vec3( -7, 25, 0 ), glm::vec3( 7, 25, 0 ) );
        auto vesselBottom = DigitalTwin::SpatialDistribution::VesselLine( 8, glm::vec3( -7, -25, 0 ), glm::vec3( 7, -25, 0 ) );
        vesselTop.insert( vesselTop.end(), vesselBottom.begin(), vesselBottom.end() );

        auto& endo = m_blueprint.AddAgentGroup( "EndothelialCells" )
                         .SetCount( 16 )
                         .SetMorphology( DigitalTwin::MorphologyGenerator::CreateSphere( 1.0f ) )
                         .SetDistribution( vesselTop )
                         .SetColor( glm::vec4( 1.0f, 0.3f, 0.3f, 1.0f ) )
                         .AddCellTypeMorphology( static_cast<int>( DigitalTwin::CellType::PhalanxCell ), DigitalTwin::MorphologyGenerator::CreateCylinder( 0.5f, 1.8f ),    glm::vec4( 0.5f, 0.15f, 0.15f, 1.0f ) )  // dark red — anchored vessel lining
                         .AddCellTypeMorphology( static_cast<int>( DigitalTwin::CellType::TipCell ),     DigitalTwin::MorphologyGenerator::CreateSpikySphere( 1.0f, 1.35f ), glm::vec4( 1.0f, 0.8f,  0.1f, 1.0f ) )  // yellow (same as StalkCell — both build vessels; shape distinguishes)
                         .AddCellTypeMorphology( static_cast<int>( DigitalTwin::CellType::StalkCell ),   DigitalTwin::MorphologyGenerator::CreateCylinder( 0.6f, 2.2f ),    glm::vec4( 1.0f, 0.8f,  0.1f, 1.0f ) );  // yellow — proliferating zone

        // PhalanxActivation removed: at y=±25 VEGF≈11 > any useful threshold → ALL cells activate.
        // PhalanxActivation fought with maturation (infinite StalkCell↔PhalanxCell flip-flop).
        // NotchDll4 handles Default→StalkCell/TipCell; mitosis maturation handles StalkCell→PhalanxCell.

        // NotchDll4 — lateral inhibition picks 1 TipCell per vessel via vessel-edge signaling.
        // tipThreshold=0.90: above symmetric ODE steady state (~0.77) so only the Turing winner crosses.
        // stalkThreshold=0.20: wider hysteresis dead zone prevents Tip↔Stalk flip-flop.
        // subSteps=20: more ODE iterations per frame for faster Turing convergence.
        endo.AddBehaviour( DigitalTwin::Behaviours::NotchDll4{
                /* dll4ProductionRate   */ 1.0f,
                /* dll4DecayRate        */ 0.1f,
                /* notchInhibitionGain  */ 20.0f,
                /* vegfr2BaseExpression */ 1.0f,
                /* tipThreshold         */ 0.90f,
                /* stalkThreshold       */ 0.20f,
                /* vegfFieldName        */ "VEGF",
                /* subSteps             */ 20u } )
            .SetHz( 60.0f );

        // VesselSeed — seed initial vessel edges (8 cells per line → 7 edges per line)
        endo.AddBehaviour( DigitalTwin::Behaviours::VesselSeed{ std::vector<uint32_t>{ 8u, 8u } } );

        // Anastomosis — TipCell-TipCell fusion when the two sprouts meet at the centre
        endo.AddBehaviour( DigitalTwin::Behaviours::Anastomosis{ /* contactDistance */ 3.0f } ).SetHz( 60.0f );

        // Oxygen consumption — all EC cells consume O2 (baseline metabolic demand)
        endo.AddBehaviour( DigitalTwin::Behaviours::ConsumeField{ "Oxygen", 1.0f } ).SetHz( 60.0f );

        // Perfusion — StalkCells inject O2; rate=30 overcomes diffusion dilution across 25-unit distance
        endo.AddBehaviour( DigitalTwin::Behaviours::Perfusion{ "Oxygen", 30.0f } ).SetHz( 60.0f );

        // CellCycle — StalkCells only: directed mitosis extends the vessel along the tube axis
        auto stalkCycle = DigitalTwin::BiologyGenerator::StandardCellCycle()
                              .SetBaseDoublingTime( 6.0f / 3600.0f )  // 6 seconds (was 72 s — OK now: only 1 StalkCell can divide)
                              .SetProliferationOxygenTarget( 40.0f )
                              .SetArrestPressureThreshold( 5.0f )
                              .SetHypoxiaOxygenThreshold( 5.0f )
                              .SetNecrosisOxygenThreshold( 1.0f )
                              .SetApoptosisRate( 0.0f )
                              .Build();
        stalkCycle.directedMitosis = true;
        endo.AddBehaviour( stalkCycle )
            .SetHz( 10.0f )
            .SetRequiredCellType( static_cast<int>( DigitalTwin::CellType::StalkCell ) );

        // Chemotaxis toward VEGF — TipCells only; contact inhibition density=3 stops migration when surrounded
        endo.AddBehaviour( DigitalTwin::Behaviours::Chemotaxis{ "VEGF", 6.0f, 0.002f, 4.0f, 3.0f } )
            .SetHz( 60.0f )
            .SetRequiredCellType( static_cast<int>( DigitalTwin::CellType::TipCell ) );


        // Brownian jitter — TipCells only (reduced to not swamp the directional signal)
        endo.AddBehaviour( DigitalTwin::Behaviours::BrownianMotion{ 0.05f } )
            .SetHz( 60.0f )
            .SetRequiredCellType( static_cast<int>( DigitalTwin::CellType::TipCell ) );

        // VesselSpring — TipCell only: leash back to nearest connected StalkCell.
        // reqCT=TipCell ensures base vessel (StalkCells, PhalanxCells) are never displaced by springs.
        endo.AddBehaviour( DigitalTwin::Behaviours::VesselSpring{ /* springStiffness */ 15.0f, /* restingLength */ 1.5f, /* damping */ 10.0f } )
            .SetHz( 60.0f )
            .SetRequiredCellType( static_cast<int>( DigitalTwin::CellType::TipCell ) );

        // JKR mechanics
        endo.AddBehaviour( DigitalTwin::BiomechanicsGenerator::JKR()
                               .SetYoungsModulus( 40.0f )
                               .SetPoissonRatio( 0.4f )
                               .SetAdhesionEnergy( 0.5f )
                               .SetMaxInteractionRadius( 1.0f )
                               .SetDampingCoefficient( 25.0f )
                               .Build() )
            .SetHz( 60.0f );

        m_engine.SetBlueprint( m_blueprint );
    }

    void Editor::SetupSimpleVesselDebugDemo()
    {
        // Minimal debug blueprint: verify VEGF → TipCell selection before adding chemotaxis/division.
        //
        // 1 source cell at origin: consumes O2 → goes Hypoxic → secretes VEGF.
        // 1 vessel of 15 cells at y=+10: all start as PhalanxCell(3).
        //    PhalanxActivation converts PhalanxCell→StalkCell when local VEGF > threshold.
        //    NotchDll4 lateral inhibition then picks exactly 1 TipCell from StalkCells.
        //
        // No chemotaxis, no division, no VesselSpring.
        // Expected result: ~t=2s, vessel cell nearest origin turns GREEN (TipCell); rest dark red (PhalanxCell).

        m_blueprint = DigitalTwin::SimulationBlueprint{};
        m_blueprint.SetName( "Simple Vessel Debug" );
        m_blueprint.SetDomainSize( glm::vec3( 40.0f ), 2.0f );

        m_blueprint.ConfigureSpatialPartitioning()
            .SetMethod( DigitalTwin::SpatialPartitioningMethod::HashGrid )
            .SetCellSize( 3.0f )
            .SetMaxDensity( 64 )
            .SetComputeHz( 60.0f );

        // Oxygen: Gaussian centered at source cell (origin), peak=50 > hypoxia threshold (30).
        // Diffusion spreads the peak outward → center value drops. Cell transitions orange→purple
        // when local O2 falls below 30. With sigma=8 and D=2, expect transition in ~5-10 seconds.
        m_blueprint.AddGridField( "Oxygen" )
            .SetInitializer( DigitalTwin::GridInitializer::Gaussian( glm::vec3( 0.0f ), 8.0f, 50.0f ) )
            .SetDiffusionCoefficient( 2.0f )
            .SetDecayRate( 0.0f )
            .SetComputeHz( 60.0f );

        // VEGF: starts at zero; source cell secretes it when hypoxic.
        // Low diffusion (0.5) keeps concentration near source so it stays visible;
        // still diffuses to the vessel at y=+10 (5 voxels) within ~2-3s.
        m_blueprint.AddGridField( "VEGF" )
            .SetInitializer( DigitalTwin::GridInitializer::Constant( 0.0f ) )
            .SetDiffusionCoefficient( 0.5f )
            .SetDecayRate( 0.02f )
            .SetComputeHz( 60.0f );

        // --- Source cell at origin ---
        auto& source = m_blueprint.AddAgentGroup( "SourceCell" )
                           .SetCount( 1 )
                           .SetMorphology( DigitalTwin::MorphologyGenerator::CreateSphere( 1.2f ) )
                           .SetDistribution( { glm::vec4( 0.0f, 0.0f, 0.0f, 1.0f ) } )
                           .SetColor( glm::vec4( 1.0f, 0.5f, 0.1f, 1.0f ) ); // orange → turns purple when hypoxic

        source.AddBehaviour( DigitalTwin::Behaviours::ConsumeField{ "Oxygen", 10.0f } ).SetHz( 60.0f );
        source.AddBehaviour( DigitalTwin::BiologyGenerator::StandardCellCycle()
                                 .SetBaseDoublingTime( 99.0f )          // no division
                                 .SetProliferationOxygenTarget( 60.0f )
                                 .SetArrestPressureThreshold( 999.0f )
                                 .SetHypoxiaOxygenThreshold( 30.0f )    // goes hypoxic quickly
                                 .SetNecrosisOxygenThreshold( 1.0f )
                                 .SetApoptosisRate( 0.0f )
                                 .Build() )
            .SetHz( 10.0f );
        // Secrete VEGF only when Hypoxic. Rate=50: with D=0.5 and decay=0.02,
        // steady-state center ~1.6 units. Lower the Normalization slider in the
        // VEGF inspector to ~10 to make it visible.
        source.AddBehaviour( DigitalTwin::Behaviours::SecreteField{
                                 "VEGF", 50.0f,
                                 static_cast<int>( DigitalTwin::LifecycleState::Hypoxic ) } )
            .SetHz( 60.0f );

        // --- Vessel: 15 cells at y=+10 ---
        auto vesselPos = DigitalTwin::SpatialDistribution::VesselLine( 15, glm::vec3( -14, 10, 0 ), glm::vec3( 14, 10, 0 ) );

        auto& vessel = m_blueprint.AddAgentGroup( "VesselCells" )
                           .SetCount( 15 )
                           .SetMorphology( DigitalTwin::MorphologyGenerator::CreateCylinder( 0.5f, 1.8f ) )
                           .SetDistribution( vesselPos )
                           .SetColor( glm::vec4( 0.5f, 0.15f, 0.15f, 1.0f ) )               // dark red base
                           .SetInitialCellType( static_cast<int>( DigitalTwin::CellType::PhalanxCell ) )
                           .AddCellTypeMorphology( static_cast<int>( DigitalTwin::CellType::PhalanxCell ),
                               DigitalTwin::MorphologyGenerator::CreateCylinder( 0.5f, 1.8f ),
                               glm::vec4( 0.5f, 0.15f, 0.15f, 1.0f ) )   // dark red — quiescent
                           .AddCellTypeMorphology( static_cast<int>( DigitalTwin::CellType::StalkCell ),
                               DigitalTwin::MorphologyGenerator::CreateCylinder( 0.6f, 2.2f ),
                               glm::vec4( 1.0f, 0.8f, 0.1f, 1.0f ) )     // yellow — activated
                           .AddCellTypeMorphology( static_cast<int>( DigitalTwin::CellType::TipCell ),
                               DigitalTwin::MorphologyGenerator::CreateSpikySphere( 1.0f, 1.35f ),
                               glm::vec4( 0.0f, 1.0f, 0.3f, 1.0f ) );    // bright green — TipCell

        // VesselSeed: wire up 15 cells into a linear chain (1 group of 15)
        vessel.AddBehaviour( DigitalTwin::Behaviours::VesselSeed{ std::vector<uint32_t>{ 15u } } );

        // PhalanxActivation: converts PhalanxCell(3)→StalkCell(2) when VEGF > threshold.
        // Threshold=2 so even weak VEGF signal activates. deactivation=0.5 adds hysteresis.
        vessel.AddBehaviour( DigitalTwin::Behaviours::PhalanxActivation{ "VEGF", 2.0f, 0.5f } ).SetHz( 60.0f );

        // NotchDll4: lateral inhibition → 1 TipCell per connected vessel chain.
        vessel.AddBehaviour( DigitalTwin::Behaviours::NotchDll4{
                /* dll4ProductionRate   */ 1.0f,
                /* dll4DecayRate        */ 0.1f,
                /* notchInhibitionGain  */ 20.0f,
                /* vegfr2BaseExpression */ 1.0f,
                /* tipThreshold         */ 0.90f,
                /* stalkThreshold       */ 0.20f,
                /* vegfFieldName        */ "VEGF",
                /* subSteps             */ 20u } )
            .SetHz( 60.0f );

        m_engine.SetBlueprint( m_blueprint );
    }

    void Editor::SetupChemotaxisDemo()
    {
        // Fresh blueprint — wipe any previous state
        m_blueprint = DigitalTwin::SimulationBlueprint{};

        // ==========================================================================================
        // 1. Domain
        // ==========================================================================================
        m_blueprint.SetName( "Chemotaxis Demo" );
        m_blueprint.SetDomainSize( glm::vec3( 50.0f ), 2.0f );

        m_blueprint.ConfigureSpatialPartitioning()
            .SetMethod( DigitalTwin::SpatialPartitioningMethod::HashGrid )
            .SetCellSize( 3.0f )
            .SetMaxDensity( 64 )
            .SetComputeHz( 60.0f );

        // ==========================================================================================
        // 2. VEGF field — pre-seeded Gaussian at origin so gradient is visible from frame 1.
        //    Source cells top it up continuously; decay is minimal so the cloud stays stable.
        // ==========================================================================================
        m_blueprint.AddGridField( "VEGF" )
            .SetInitializer( DigitalTwin::GridInitializer::Gaussian( glm::vec3( 0.0f ), 8.0f, 200.0f ) )
            .SetDiffusionCoefficient( 0.8f )
            .SetDecayRate( 0.01f )
            .SetComputeHz( 60.0f );

        // ==========================================================================================
        // 3. HypoxicCores — 3 cells at the centre, always secreting VEGF (no CellCycle needed).
        //    Orange so they stand out as the attractant source.
        // ==========================================================================================
        std::vector<glm::vec4> corePositions = {
            glm::vec4( -1.5f,  0.5f, 0.0f, 1.0f ),
            glm::vec4(  1.5f,  0.5f, 0.0f, 1.0f ),
            glm::vec4(  0.0f, -1.0f, 0.0f, 1.0f ),
        };

        auto& cores = m_blueprint.AddAgentGroup( "HypoxicCores" )
                          .SetCount( 3 )
                          .SetMorphology( DigitalTwin::MorphologyGenerator::CreateSphere( 1.5f ) )
                          .SetDistribution( corePositions )
                          .SetColor( glm::vec4( 1.0f, 0.6f, 0.0f, 1.0f ) ); // Orange

        // Unconditional secretion — no requiredState so VEGF flows from simulation start
        cores.AddBehaviour( DigitalTwin::Behaviours::SecreteField{ "VEGF", 500.0f } ).SetHz( 60.0f );
        cores.AddBehaviour( DigitalTwin::BiomechanicsGenerator::JKR()
                                .SetYoungsModulus( 20.0f )
                                .SetPoissonRatio( 0.4f )
                                .SetAdhesionEnergy( 1.5f )
                                .SetMaxInteractionRadius( 1.5f )
                                .Build() )
            .SetHz( 60.0f );

        // ==========================================================================================
        // 4. EndothelialCells — ring of 8 cells at r ≈ 18 in the XZ plane.
        //    No BrownianMotion so directed migration is unambiguous.
        //    Red so the inward collapse toward the orange source is visually obvious.
        // ==========================================================================================
        const float r = 18.0f;
        std::vector<glm::vec4> endoPositions = {
            glm::vec4(  r,     0.0f,  0.0f,  1.0f ),
            glm::vec4(  r * 0.707f, 0.0f,  r * 0.707f, 1.0f ),
            glm::vec4(  0.0f,  0.0f,  r,     1.0f ),
            glm::vec4( -r * 0.707f, 0.0f,  r * 0.707f, 1.0f ),
            glm::vec4( -r,     0.0f,  0.0f,  1.0f ),
            glm::vec4( -r * 0.707f, 0.0f, -r * 0.707f, 1.0f ),
            glm::vec4(  0.0f,  0.0f, -r,     1.0f ),
            glm::vec4(  r * 0.707f, 0.0f, -r * 0.707f, 1.0f ),
        };

        auto& endo = m_blueprint.AddAgentGroup( "EndothelialCells" )
                         .SetCount( 8 )
                         .SetMorphology( DigitalTwin::MorphologyGenerator::CreateSphere( 1.0f ) )
                         .SetDistribution( endoPositions )
                         .SetColor( glm::vec4( 1.0f, 0.2f, 0.2f, 1.0f ) ); // Red

        // Chemotaxis BEFORE Biomechanics — gradient movement resolved first, then collision
        endo.AddBehaviour( DigitalTwin::Behaviours::Chemotaxis{ "VEGF", 6.0f, 0.001f, 15.0f } ).SetHz( 60.0f );
        endo.AddBehaviour( DigitalTwin::BiomechanicsGenerator::JKR()
                               .SetYoungsModulus( 15.0f )
                               .SetPoissonRatio( 0.4f )
                               .SetAdhesionEnergy( 1.0f )
                               .SetMaxInteractionRadius( 1.5f )
                               .Build() )
            .SetHz( 60.0f );

        m_engine.SetBlueprint( m_blueprint );
    }

} // namespace Gaudi