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
#include <simulation/VesselTreeGenerator.h>

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

        // Clear the default name so the Hierarchy shows an empty placeholder
        // until the user creates a new simulation or loads a demo.
        m_blueprint.SetName( "" );
        m_engine.SetBlueprint( m_blueprint );
    }

    void Editor::SetupEmptyBlueprint()
    {
        m_blueprint = DigitalTwin::SimulationBlueprint{};
        m_blueprint.SetName( "Untitled" );
        m_blueprint.SetDomainSize( glm::vec3( 30.0f ), 2.0f );
        m_blueprint.ConfigureSpatialPartitioning()
            .SetMethod( DigitalTwin::SpatialPartitioningMethod::HashGrid )
            .SetCellSize( 3.0f )
            .SetMaxDensity( 64 )
            .SetComputeHz( 60.0f );
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
                         .AddCellTypeMorphology( static_cast<int>( DigitalTwin::CellType::StalkCell ), DigitalTwin::MorphologyGenerator::CreateCurvedTile( 60.0f, 1.35f, 0.25f, 1.5f ), glm::vec4( 1.0f, 0.8f, 0.1f, 1.0f ) );

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

    void Editor::RenderDemoBrowser()
    {
        if( !m_showDemoBrowser )
            return;

        struct Demo
        {
            const char* name;
            const char* description;
            void ( Editor::*setupFn )();
        };

        static const Demo k_demos[] = {
            { "Secrete",
              "A single cell secretes into an initially empty field.\n"
              "Watch the substance accumulate and diffuse outward.\n\n"
              "Demonstrates: SecreteField, field diffusion.",
              &Editor::SetupSecreteDemo },

            { "Consume",
              "A single cell drains a Gaussian pre-seeded field.\n"
              "Watch the local depletion grow around the cell.\n\n"
              "Demonstrates: ConsumeField, field depletion.",
              &Editor::SetupConsumeDemo },

            { "Chemotaxis",
              "Source cell (green) consumes O2 and goes Hypoxic\n"
              "after a few seconds — then starts secreting VEGF.\n"
              "Responder cell (blue) slowly drifts toward\n"
              "the growing VEGF cloud.\n\n"
              "Demonstrates: CellCycle lifecycle, SecreteField\n"
              "(Hypoxic only), Chemotaxis gradient sensing.",
              &Editor::SetupChemotaxisDemo },

            { "Cell Cycle",
              "A small tumour cluster grows, divides, and arrests\n"
              "under pressure and hypoxia.\n\n"
              "Demonstrates: CellCycle, JKR biomechanics, O2 coupling.",
              &Editor::SetupCellCycleDemo },

            { "Diffusion & Decay",
              "A sphere of substance diffuses outward and decays\n"
              "to equilibrium. No cells — field physics only.\n\n"
              "Demonstrates: GridField diffusion, decay rate,\n"
              "Sphere initializer.",
              &Editor::SetupDiffusionDecayDemo },

            { "Brownian Motion",
              "27 particles in a 3x3x3 grid perform pure thermal\n"
              "random-walk. No fields or forces — each drifts\n"
              "independently.\n\n"
              "Demonstrates: BrownianMotion behaviour.",
              &Editor::SetupBrownianMotionDemo },

            { "JKR Packing",
              "10 cells packed tightly at the origin explode apart\n"
              "under pure Hertz repulsion and settle into a stable\n"
              "packing. Zero adhesion energy.\n\n"
              "Demonstrates: Biomechanics repulsion, damping.",
              &Editor::SetupJKRPackingDemo },

            { "Lifecycle",
              "A single cell consumes its local O2 supply and\n"
              "progresses Live (red) -> Hypoxic (purple) ->\n"
              "Necrotic (dark).\n\n"
              "Demonstrates: CellCycle lifecycle states,\n"
              "ConsumeField, O2 depletion.",
              &Editor::SetupLifecycleDemo },

            { "Static Vessel Tree",
              "A branching 2D tube vessel tree — rings of 6\n"
              "plate-like endothelial cells held together by\n"
              "spring forces. No biology, no VEGF.\n\n"
              "Purpose: verify tree shape, ring topology, and\n"
              "structural stability (VesselSpring + Biomechanics).\n\n"
              "Demonstrates: VesselTreeGenerator, VesselSeed\n"
              "(explicit edges), disc morphology, VesselSpring.",
              &Editor::SetupStaticVesselTreeDemo },

            { "Vessel Sprouting",
              "A 2D tube vessel (6-cell rings) rests 10 units\n"
              "above a hypoxic source cell.\n\n"
              "Sequence:\n"
              "  ~t=3s  Source goes Hypoxic, secretes VEGF\n"
              "  ~t=8s  VEGF activates PhalanxCells to StalkCells\n"
              "  ~t=10s NotchDll4 selects 1 TipCell from the ring\n"
              "  ~t=12s TipCell migrates toward source\n"
              "  ~t=18s Directed mitosis extends 1D sprout chain\n\n"
              "Demonstrates: 2D tube to 1D sprout transition,\n"
              "edge flags, PhalanxActivation, NotchDll4,\n"
              "Chemotaxis, directedMitosis, VesselSpring.",
              &Editor::SetupVesselSproutingDemo },

            { "Vessel Angiogenesis",
              "Two vessel tubes flank a hypoxic tumour source.\n\n"
              "Sequence:\n"
              "  ~t=3s  Tumour goes Hypoxic, secretes VEGF\n"
              "  ~t=8s  VEGF activates nearest rings on both tubes\n"
              "  ~t=10s NotchDll4 selects 1 TipCell per tube\n"
              "  ~t=12s Both TipCells migrate toward the tumour\n"
              "  ~t=20s Sprouts meet → Anastomosis fires\n"
              "  ~t=20s+ Perfusion pumps O2 through the loop\n\n"
              "Demonstrates: dual-sprout angiogenesis, anastomosis,\n"
              "post-anastomosis PhalanxCell perfusion.",
              &Editor::SetupAngiogenesisDemo },
        };

        ImGuiIO& io = ImGui::GetIO();
        ImGui::SetNextWindowSize( ImVec2( 640, 360 ), ImGuiCond_Appearing );
        ImGui::SetNextWindowPos( ImVec2( io.DisplaySize.x * 0.5f, io.DisplaySize.y * 0.5f ),
                                 ImGuiCond_Appearing, ImVec2( 0.5f, 0.5f ) );

        if( !ImGui::Begin( "Demo Library", &m_showDemoBrowser, ImGuiWindowFlags_NoCollapse ) )
        {
            ImGui::End();
            return;
        }

        static int s_selected = 0;

        // Left column — demo list
        ImGui::BeginChild( "##demoList", ImVec2( 190, 0 ), true );
        for( int i = 0; i < static_cast<int>( std::size( k_demos ) ); ++i )
        {
            if( ImGui::Selectable( k_demos[ i ].name, s_selected == i ) )
                s_selected = i;
        }
        ImGui::EndChild();

        ImGui::SameLine();

        // Right column — description + load button
        ImGui::BeginChild( "##demoDetail", ImVec2( 0, 0 ), false );

        ImGui::TextWrapped( "%s", k_demos[ s_selected ].description );

        ImGui::SetCursorPosY( ImGui::GetWindowHeight() - ImGui::GetFrameHeightWithSpacing() - 4 );
        if( ImGui::Button( "Load Demo", ImVec2( -1, 0 ) ) )
        {
            LoadDemo( k_demos[ s_selected ].setupFn );
            m_showDemoBrowser = false;
        }

        ImGui::EndChild();

        ImGui::End();
    }

    // Stops the running simulation (if any), calls setupFn to rebuild the blueprint,
    // then notifies the engine so it picks up the new blueprint on the next Play().
    void Editor::LoadDemo( void ( Editor::*setupFn )() )
    {
        if( m_engine.GetState() != DigitalTwin::EngineState::RESET )
            m_engine.Stop();
        ( this->*setupFn )();
        m_engine.SetBlueprint( m_blueprint );
    }

    void Editor::RenderMainMenuBar()
    {
        static bool s_pendingNewSim          = false;
        static char s_newSimNameBuf[ 128 ]   = "Untitled";

        if( ImGui::BeginMainMenuBar() )
        {
            if( ImGui::BeginMenu( "File" ) )
            {
                if( ImGui::MenuItem( "New Empty Simulation" ) )
                {
                    strncpy_s( s_newSimNameBuf, "Untitled", sizeof( s_newSimNameBuf ) - 1 );
                    s_pendingNewSim = true;
                }

                ImGui::Separator();

                if( ImGui::MenuItem( "Save Blueprint..." ) )
                    DT_INFO( "Blueprint save not yet implemented." ); // TODO: Phase 2

                if( ImGui::MenuItem( "Load Blueprint..." ) )
                    DT_INFO( "Blueprint load not yet implemented." ); // TODO: Phase 2

                ImGui::Separator();

                if( ImGui::MenuItem( "Quit" ) )
                    m_shouldQuit = true;

                ImGui::EndMenu();
            }

            if( ImGui::MenuItem( "Demos" ) )
                m_showDemoBrowser = true;

            ImGui::EndMainMenuBar();
        }

        // ── "New Empty Simulation" name modal ─────────────────────────────────
        // OpenPopup must be called at the same level as BeginPopupModal, outside
        // any menu/popup stack — so we defer with a flag set inside the menu above.
        if( s_pendingNewSim )
        {
            ImGui::OpenPopup( "New Simulation" );
            s_pendingNewSim = false;
        }

        if( ImGui::BeginPopupModal( "New Simulation", nullptr, ImGuiWindowFlags_AlwaysAutoResize ) )
        {
            ImGui::Text( "Simulation name:" );
            ImGui::SetNextItemWidth( 280.0f );
            bool confirm = ImGui::InputText( "##newSimName", s_newSimNameBuf, sizeof( s_newSimNameBuf ),
                                             ImGuiInputTextFlags_EnterReturnsTrue | ImGuiInputTextFlags_AutoSelectAll );
            if( ImGui::IsWindowAppearing() )
                ImGui::SetKeyboardFocusHere( -1 );

            ImGui::Spacing();
            if( confirm || ImGui::Button( "Create", ImVec2( 120, 0 ) ) )
            {
                LoadDemo( &Editor::SetupEmptyBlueprint );
                m_blueprint.SetName( s_newSimNameBuf );
                m_engine.SetBlueprint( m_blueprint );
                ImGui::CloseCurrentPopup();
            }
            ImGui::SameLine();
            if( ImGui::Button( "Cancel", ImVec2( 120, 0 ) ) )
                ImGui::CloseCurrentPopup();

            ImGui::EndPopup();
        }
    }

    void Editor::Run()
    {
        DT_INFO( "Starting Gaudi Editor Loop..." );
        ImGui::SetCurrentContext( static_cast<ImGuiContext*>( m_engine.GetImGuiContext() ) );

        while( !m_engine.IsWindowClosed() && !m_shouldQuit )
        {
            m_engine.BeginFrame();

            m_engine.RenderUI( [ this ]() {
                RenderMainMenuBar();
                RenderDemoBrowser();
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

    void Editor::SetupSimpleVesselDebugDemo()
    {
        // Simple vessel sprouting demo (Guo et al. 2024 biology).
        //
        // 1 source cell at (0,0,0): consumes O2 → Hypoxic → secretes VEGF.
        // 1 vessel of 15 cells at y=+10 (10 units from source).
        //
        // Correct two-step differentiation pathway:
        //   1. PhalanxActivation: VEGF > threshold → PhalanxCell activates to StalkCell
        //      (models VEGF-induced EC activation; only ~3-5 cells nearest source activate)
        //   2. NotchDll4: activated StalkCells compete; winner (highest VEGF) → TipCell
        //   3. Chemotaxis: TipCell migrates toward VEGF source
        //   4. CellCycle (directedMitosis): StalkCell behind TipCell divides → stalk elongates
        //   5. VesselSpring: all non-Phalanx cells tethered — chain spacing maintained
        //
        // Expected sequence: ~t=5s Hypoxic → ~t=10s TipCell selected → migrates → stalk grows.

        m_blueprint = DigitalTwin::SimulationBlueprint{};
        m_blueprint.SetName( "Simple Vessel Debug" );
        m_blueprint.SetDomainSize( glm::vec3( 40.0f ), 2.0f );

        m_blueprint.ConfigureSpatialPartitioning()
            .SetMethod( DigitalTwin::SpatialPartitioningMethod::HashGrid )
            .SetCellSize( 3.0f )
            .SetMaxDensity( 64 )
            .SetComputeHz( 60.0f );

        // Source is at (0,0,0), 10 units below the vessel at y=+10.
        // Effective VEGF characteristic length: λ = h*√(D/k) = 2*√(0.5/0.1) = 4.47 units
        // (factor of h²=4 from discrete diffusion — shader does not divide by dx²).
        // At 10 units: ~10% of peak VEGF reaches vessel center, dropping sharply to the sides.

        // Oxygen: sigma=12 ensures vessel at y=+10 gets O2 ≈ 42 (above ProliferationTarget=40).
        // Source at (0,0,0) starts at peak=60, consumes at 10/s → below 30 → Hypoxic.
        m_blueprint.AddGridField( "Oxygen" )
            .SetInitializer( DigitalTwin::GridInitializer::Gaussian( glm::vec3( 0.0f, 0.0f, 0.0f ), 12.0f, 60.0f ) )
            .SetDiffusionCoefficient( 2.0f )
            .SetDecayRate( 0.0f )
            .SetComputeHz( 60.0f );

        // VEGF: decay=0.1 (λ=4.47 effective units), rate=300 reaches vessel from 10 units.
        m_blueprint.AddGridField( "VEGF" )
            .SetInitializer( DigitalTwin::GridInitializer::Constant( 0.0f ) )
            .SetDiffusionCoefficient( 0.5f )
            .SetDecayRate( 0.1f )
            .SetComputeHz( 60.0f );

        // --- Source cell 10 units below vessel at (0,0,0) ---
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
        source.AddBehaviour( DigitalTwin::Behaviours::SecreteField{
                                 "VEGF", 300.0f,
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

        // 1. VesselSeed: wire up 15 cells into a linear chain
        vessel.AddBehaviour( DigitalTwin::Behaviours::VesselSeed{ std::vector<uint32_t>{ 15u } } );

        // 2. PhalanxActivation: spatial gate — only cells with VEGF > threshold activate.
        //    Threshold=20 targets center 3-5 cells. ConsumeField (below) limits lateral spread.
        //    Tune threshold by checking VEGF inspector after source goes Hypoxic.
        vessel.AddBehaviour( DigitalTwin::Behaviours::PhalanxActivation{
                /* vegfFieldName         */ "VEGF",
                /* activationThreshold   */ 20.0f,
                /* deactivationThreshold */ 5.0f } )
            .SetHz( 60.0f );

        // 3. ConsumeField: VEGFR1 (stalk-enriched decoy receptor) — VEGF sink shapes the gradient.
        //    Rate=2.0 (strong sink) prevents lateral VEGF creep → stops multi-TipCell activation.
        vessel.AddBehaviour( DigitalTwin::Behaviours::ConsumeField{ "VEGF", 2.0f } )
            .SetRequiredCellType( 2 )  // StalkCell only
            .SetHz( 60.0f );

        // 4. NotchDll4: lateral inhibition among activated StalkCells → 1 TipCell.
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

        // 5. CellCycle: StalkCell in proliferation zone (adjacent to TipCell) divides → stalk elongates.
        //    TipCell guard in mitosis_vessel_append.comp prevents TipCells from dividing.
        auto stalkCycle = DigitalTwin::BiologyGenerator::StandardCellCycle()
                              .SetBaseDoublingTime( 6.0f / 3600.0f )  // 6 seconds per doubling
                              .SetProliferationOxygenTarget( 40.0f )
                              .SetArrestPressureThreshold( 5.0f )
                              .SetHypoxiaOxygenThreshold( 5.0f )
                              .SetNecrosisOxygenThreshold( 1.0f )
                              .SetApoptosisRate( 0.0f )
                              .Build();
        stalkCycle.directedMitosis = true;
        vessel.AddBehaviour( stalkCycle ).SetHz( 10.0f );

        // 6. Chemotaxis: TipCell follows VEGF gradient toward source.
        vessel.AddBehaviour( DigitalTwin::Behaviours::Chemotaxis{ "VEGF", 6.0f, 0.002f, 4.0f, 3.0f } )
            .SetHz( 60.0f )
            .SetRequiredCellType( static_cast<int>( DigitalTwin::CellType::TipCell ) );

        // 7. VesselSpring: all non-Phalanx cells get spring forces — chain spacing maintained.
        //    StalkCell daughters from directed mitosis need springs to prevent blobbing.
        //    PhalanxCells are anchored by vessel_mechanics.comp hardcode (line 48), not by this filter.
        vessel.AddBehaviour( DigitalTwin::Behaviours::VesselSpring{ 15.0f, 1.5f, 10.0f } )
            .SetHz( 60.0f );

        m_engine.SetBlueprint( m_blueprint );
    }

    void Editor::SetupSecreteDemo()
    {
        // One cell pumps substance into an empty field.
        // Goal: clearly see field growing outward from the cell.
        m_blueprint = DigitalTwin::SimulationBlueprint{};
        m_blueprint.SetName( "Secrete Demo" );
        m_blueprint.SetDomainSize( glm::vec3( 20.0f ), 1.0f );

        m_blueprint.ConfigureSpatialPartitioning()
            .SetMethod( DigitalTwin::SpatialPartitioningMethod::HashGrid )
            .SetCellSize( 3.0f )
            .SetMaxDensity( 64 )
            .SetComputeHz( 60.0f );

        // Field starts empty; no decay so substance accumulates and spreads visibly.
        m_blueprint.AddGridField( "Substance" )
            .SetInitializer( DigitalTwin::GridInitializer::Constant( 0.0f ) )
            .SetDiffusionCoefficient( 2.5f )
            .SetDecayRate( 0.0f )
            .SetComputeHz( 60.0f );

        auto& cells = m_blueprint.AddAgentGroup( "SecretingCell" )
                          .SetCount( 1 )
                          .SetMorphology( DigitalTwin::MorphologyGenerator::CreateSphere( 1.0f ) )
                          .SetDistribution( { glm::vec4( 0.0f, 0.0f, 0.0f, 1.0f ) } )
                          .SetColor( glm::vec4( 0.2f, 0.6f, 1.0f, 1.0f ) ); // blue

        // Rate=500 fills the peak voxel quickly — clearly visible against normalization.
        cells.AddBehaviour( DigitalTwin::Behaviours::SecreteField{ "Substance", 1000.0f } )
            .SetHz( 60.0f );
    }

    void Editor::SetupConsumeDemo()
    {
        // One cell drains a Gaussian pre-seeded field.
        // Goal: clearly see the local depletion spreading outward from the cell.
        m_blueprint = DigitalTwin::SimulationBlueprint{};
        m_blueprint.SetName( "Consume Demo" );
        m_blueprint.SetDomainSize( glm::vec3( 20.0f ), 1.0f );

        m_blueprint.ConfigureSpatialPartitioning()
            .SetMethod( DigitalTwin::SpatialPartitioningMethod::HashGrid )
            .SetCellSize( 3.0f )
            .SetMaxDensity( 64 )
            .SetComputeHz( 60.0f );

        // Field starts full; diffusion replenishes from edges as cell depletes the local peak.
        m_blueprint.AddGridField( "Substance" )
            .SetInitializer( DigitalTwin::GridInitializer::Gaussian( glm::vec3( 0.0f ), 6.0f, 100.0f ) )
            .SetDiffusionCoefficient( 2.5f )
            .SetDecayRate( 0.0f )
            .SetComputeHz( 60.0f );

        auto& cells = m_blueprint.AddAgentGroup( "ConsumingCell" )
                          .SetCount( 1 )
                          .SetMorphology( DigitalTwin::MorphologyGenerator::CreateSphere( 1.0f ) )
                          .SetDistribution( { glm::vec4( 0.0f, 0.0f, 0.0f, 1.0f ) } )
                          .SetColor( glm::vec4( 1.0f, 0.35f, 0.1f, 1.0f ) ); // orange

        // Rate=100 creates a fast, clearly visible depletion hole at the cell position.
        cells.AddBehaviour( DigitalTwin::Behaviours::ConsumeField{ "Substance", 1000.0f } )
            .SetHz( 60.0f );
    }

    void Editor::SetupChemotaxisDemo()
    {
        // Two-cell demo:
        //   Source (green) : consumes O2 → goes Hypoxic after ~5s → starts secreting VEGF.
        //   Responder (blue): slowly chemotaxes toward the growing VEGF cloud.
        //
        // Sequence: green stays green → turns purple (Hypoxic) → VEGF cloud grows from it
        //           → blue cell drifts toward green over 20-30 seconds.

        m_blueprint = DigitalTwin::SimulationBlueprint{};
        m_blueprint.SetName( "Chemotaxis Demo" );
        m_blueprint.SetDomainSize( glm::vec3( 30.0f ), 1.5f );

        m_blueprint.ConfigureSpatialPartitioning()
            .SetMethod( DigitalTwin::SpatialPartitioningMethod::HashGrid )
            .SetCellSize( 3.0f )
            .SetMaxDensity( 64 )
            .SetComputeHz( 60.0f );

        // O2: tight Gaussian so source depletes its local supply in a few seconds.
        // No decay → limited reservoir; diffusion replenishes slowly from surroundings.
        m_blueprint.AddGridField( "Oxygen" )
            .SetInitializer( DigitalTwin::GridInitializer::Gaussian( glm::vec3( 0.0f ), 5.0f, 60.0f ) )
            .SetDiffusionCoefficient( 0.5f )
            .SetDecayRate( 0.0f )
            .SetComputeHz( 60.0f );

        // VEGF: starts empty; source secretes once Hypoxic; no decay so cloud grows steadily.
        m_blueprint.AddGridField( "VEGF" )
            .SetInitializer( DigitalTwin::GridInitializer::Constant( 0.0f ) )
            .SetDiffusionCoefficient( 0.5f )
            .SetDecayRate( 0.0f )
            .SetComputeHz( 60.0f );

        // ── Source cell (green → purple when Hypoxic) ────────────────────────
        auto& source = m_blueprint.AddAgentGroup( "Source" )
                           .SetCount( 1 )
                           .SetMorphology( DigitalTwin::MorphologyGenerator::CreateSphere( 1.0f ) )
                           .SetDistribution( { glm::vec4( 0.0f, 0.0f, 0.0f, 1.0f ) } )
                           .SetColor( glm::vec4( 0.15f, 0.85f, 0.25f, 1.0f ) ); // green

        // Consume O2 until local concentration drops below hypoxia threshold.
        source.AddBehaviour( DigitalTwin::Behaviours::ConsumeField{ "Oxygen", 15.0f } ).SetHz( 60.0f );

        // CellCycle tracks lifecycle only (no division — doubling time = 999 hours).
        source.AddBehaviour( DigitalTwin::BiologyGenerator::StandardCellCycle()
                                 .SetBaseDoublingTime( 999.0f )
                                 .SetProliferationOxygenTarget( 60.0f )
                                 .SetArrestPressureThreshold( 999.0f )
                                 .SetHypoxiaOxygenThreshold( 30.0f )
                                 .SetNecrosisOxygenThreshold( 1.0f )
                                 .SetApoptosisRate( 0.0f )
                                 .Build() )
            .SetHz( 10.0f );

        // Secrete VEGF only while Hypoxic — starts automatically once lifecycle flips.
        source.AddBehaviour( DigitalTwin::Behaviours::SecreteField{
                                 "VEGF", 200.0f,
                                 static_cast<int>( DigitalTwin::LifecycleState::Hypoxic ) } )
            .SetHz( 60.0f );

        // ── Responder cell (blue) ─────────────────────────────────────────────
        auto& responder = m_blueprint.AddAgentGroup( "Responder" )
                              .SetCount( 1 )
                              .SetMorphology( DigitalTwin::MorphologyGenerator::CreateSphere( 1.0f ) )
                              .SetDistribution( { glm::vec4( 10.0f, 0.0f, 0.0f, 1.0f ) } )
                              .SetColor( glm::vec4( 0.2f, 0.45f, 1.0f, 1.0f ) ); // blue

        // Low sensitivity and max-velocity so the drift is clearly visible but slow.
        responder.AddBehaviour( DigitalTwin::Behaviours::Chemotaxis{ "VEGF", 2.0f, 0.001f, 1.5f } )
            .SetHz( 60.0f );

        m_engine.SetBlueprint( m_blueprint );
    }

    void Editor::SetupDiffusionDecayDemo()
    {
        // Field-only demo: a sphere of high concentration diffuses and decays to equilibrium.
        // No cells — watch the substance spread outward and fade.
        m_blueprint = DigitalTwin::SimulationBlueprint{};
        m_blueprint.SetName( "Diffusion & Decay" );
        m_blueprint.SetDomainSize( glm::vec3( 20.0f ), 1.0f );

        m_blueprint.ConfigureSpatialPartitioning()
            .SetMethod( DigitalTwin::SpatialPartitioningMethod::HashGrid )
            .SetCellSize( 3.0f )
            .SetMaxDensity( 64 )
            .SetComputeHz( 60.0f );

        // Gaussian peak at origin: smooth gradient from 100 at center to 0 at edges.
        // High diffusion spreads it visibly; strong decay pulls it back to zero.
        m_blueprint.AddGridField( "Substance" )
            .SetInitializer( DigitalTwin::GridInitializer::Gaussian( glm::vec3( 0.0f ), 3.0f, 100.0f ) )
            .SetDiffusionCoefficient( 3.0f )
            .SetDecayRate( 0.5f )
            .SetComputeHz( 60.0f );
    }

    void Editor::SetupBrownianMotionDemo()
    {
        // 27 cells arranged in a 3x3x3 grid perform pure thermal random-walk.
        // No fields, no forces — each cell drifts independently.
        m_blueprint = DigitalTwin::SimulationBlueprint{};
        m_blueprint.SetName( "Brownian Motion" );
        m_blueprint.SetDomainSize( glm::vec3( 20.0f ), 1.0f );

        m_blueprint.ConfigureSpatialPartitioning()
            .SetMethod( DigitalTwin::SpatialPartitioningMethod::HashGrid )
            .SetCellSize( 3.0f )
            .SetMaxDensity( 64 )
            .SetComputeHz( 60.0f );

        // 3x3x3 grid of cells, 5-unit spacing → fits within 20-unit domain
        std::vector<glm::vec4> positions;
        for( int x = -1; x <= 1; ++x )
            for( int y = -1; y <= 1; ++y )
                for( int z = -1; z <= 1; ++z )
                    positions.push_back( glm::vec4( x * 5.0f, y * 5.0f, z * 5.0f, 1.0f ) );

        auto& cells = m_blueprint.AddAgentGroup( "Particles" )
                          .SetCount( static_cast<uint32_t>( positions.size() ) )
                          .SetMorphology( DigitalTwin::MorphologyGenerator::CreateSphere( 0.5f ) )
                          .SetDistribution( positions )
                          .SetColor( glm::vec4( 0.9f, 0.6f, 0.1f, 1.0f ) ); // amber

        cells.AddBehaviour( DigitalTwin::Behaviours::BrownianMotion{ 2.0f } ).SetHz( 60.0f );
    }

    void Editor::SetupJKRPackingDemo()
    {
        // 40 cells packed into a tight sphere at origin — pure Hertz repulsion pushes them
        // outward into a stable close-packing arrangement. High damping slows the expansion
        // so you can watch the cluster bloom over ~10 seconds.
        m_blueprint = DigitalTwin::SimulationBlueprint{};
        m_blueprint.SetName( "JKR Packing" );
        m_blueprint.SetDomainSize( glm::vec3( 20.0f ), 1.0f );

        m_blueprint.ConfigureSpatialPartitioning()
            .SetMethod( DigitalTwin::SpatialPartitioningMethod::HashGrid )
            .SetCellSize( 3.0f )
            .SetMaxDensity( 64 )
            .SetComputeHz( 60.0f );

        auto positions = DigitalTwin::SpatialDistribution::UniformInSphere( 40, 0.5f );

        auto& cells = m_blueprint.AddAgentGroup( "Cells" )
                          .SetCount( 40 )
                          .SetMorphology( DigitalTwin::MorphologyGenerator::CreateSphere( 0.8f ) )
                          .SetDistribution( positions )
                          .SetColor( glm::vec4( 0.3f, 0.7f, 0.95f, 1.0f ) ); // cyan-blue

        // Damping formula in shader: displacement /= (1 + damping * dt). With dt=1/60:
        //   damping=10000 → divide by ~168 → expansion at ~1 unit/s, settling over ~10s.
        cells.AddBehaviour( DigitalTwin::BiomechanicsGenerator::JKR()
                                .SetYoungsModulus( 5.0f )
                                .SetPoissonRatio( 0.4f )
                                .SetAdhesionEnergy( 0.0f )
                                .SetMaxInteractionRadius( 1.5f )
                                .SetDampingCoefficient( 250.0f )
                                .Build() )
            .SetHz( 60.0f );
    }

    void Editor::SetupLifecycleDemo()
    {
        // Single cell consumes its local O2 supply → turns Hypoxic (purple) → Necrotic (dark).
        // Demonstrates the Live→Hypoxic→Necrotic colour transitions driven by CellCycle.
        m_blueprint = DigitalTwin::SimulationBlueprint{};
        m_blueprint.SetName( "Lifecycle" );
        m_blueprint.SetDomainSize( glm::vec3( 15.0f ), 1.0f );

        m_blueprint.ConfigureSpatialPartitioning()
            .SetMethod( DigitalTwin::SpatialPartitioningMethod::HashGrid )
            .SetCellSize( 3.0f )
            .SetMaxDensity( 64 )
            .SetComputeHz( 60.0f );

        // Small O2 reservoir: tight Gaussian so the cell drains it in ~10s.
        m_blueprint.AddGridField( "Oxygen" )
            .SetInitializer( DigitalTwin::GridInitializer::Gaussian( glm::vec3( 0.0f ), 3.0f, 50.0f ) )
            .SetDiffusionCoefficient( 0.5f )
            .SetDecayRate( 0.0f )
            .SetComputeHz( 60.0f );

        auto& cell = m_blueprint.AddAgentGroup( "Cell" )
                         .SetCount( 1 )
                         .SetMorphology( DigitalTwin::MorphologyGenerator::CreateSphere( 1.0f ) )
                         .SetDistribution( { glm::vec4( 0.0f, 0.0f, 0.0f, 1.0f ) } )
                         .SetColor( glm::vec4( 0.9f, 0.3f, 0.3f, 1.0f ) ); // red

        // Consumes O2 aggressively — triggers Hypoxic in ~3s, Necrotic in ~6s.
        cell.AddBehaviour( DigitalTwin::Behaviours::ConsumeField{ "Oxygen", 30.0f } ).SetHz( 60.0f );

        cell.AddBehaviour( DigitalTwin::BiologyGenerator::StandardCellCycle()
                               .SetBaseDoublingTime( 999.0f )
                               .SetProliferationOxygenTarget( 60.0f )
                               .SetArrestPressureThreshold( 999.0f )
                               .SetHypoxiaOxygenThreshold( 30.0f )
                               .SetNecrosisOxygenThreshold( 5.0f )
                               .SetApoptosisRate( 0.0f )
                               .Build() )
            .SetHz( 10.0f );
    }

    void Editor::SetupStaticVesselTreeDemo()
    {
        // Static branching vessel tree: 2D ring topology, no biology, no VEGF.
        // Purpose: visually verify tree shape, ring connectivity, and structural stability.
        //
        // VesselTreeGenerator builds a trunk + 2-level branching tree. VesselSeed uploads
        // the full edge list (circumferential + axial + junction). VesselSpring keeps the
        // rings coherent. Biomechanics ensures vessel cells collide with each other.

        m_blueprint = DigitalTwin::SimulationBlueprint{};
        m_blueprint.SetName( "Static Vessel Tree" );
        m_blueprint.SetDomainSize( glm::vec3( 80.0f ), 2.0f );

        m_blueprint.ConfigureSpatialPartitioning()
            .SetMethod( DigitalTwin::SpatialPartitioningMethod::HashGrid )
            .SetCellSize( 4.0f )
            .SetMaxDensity( 64 )
            .SetComputeHz( 60.0f );

        // Trunk: ring=6, radius=1.5, spacing=1.5 → cell width ≈ 1.57 (auto-derived).
        // Child branches adapt ring size: depth-1 ≈ ring=4, depth-2 ≈ ring=3.
        // All cells are uniform scale=1.0; thickness comes from fewer cells per ring.
        // Per-edge rest lengths seeded from initial geometry → no global mismatch.
        auto tree = DigitalTwin::VesselTreeGenerator::BranchingTree()
            .SetOrigin( glm::vec3( -25.0f, 0.0f, 0.0f ) )
            .SetDirection( glm::vec3( 1.0f, 0.0f, 0.0f ) )
            .SetLength( 20.0f )
            .SetCellSpacing( 1.5f )
            .SetRingSize( 6 )
            .SetTubeRadius( 1.5f )
            .SetBranchingAngle( 35.0f )
            .SetBranchingDepth( 2 )
            .SetLengthFalloff( 0.65f )
            .SetAngleJitter( 8.0f )
            .SetTubeRadiusFalloff( 0.65f )
            .SetCurvature( 0.15f )
            .SetBranchTwist( 3.0f )
            .SetSeed( 42 )
            .Build();

        DigitalTwin::Behaviours::VesselSeed seed;
        seed.segmentCounts = tree.segmentCounts;
        seed.explicitEdges = tree.edges;

        auto& vessel = m_blueprint.AddAgentGroup( "VesselTree" )
            .SetCount( tree.totalCells )
            .SetMorphology( DigitalTwin::MorphologyGenerator::CreateCurvedTile( 60.0f, 1.35f, 0.25f, 1.5f ) )
            .SetDistribution( tree.positions )
            .SetOrientations( tree.normals )
            .SetColor( glm::vec4( 0.6f, 0.1f, 0.1f, 1.0f ) )
            .SetInitialCellType( static_cast<int>( DigitalTwin::CellType::PhalanxCell ) );

        vessel.AddBehaviour( seed );
        // anchorPhalanxCells=true: PhalanxCells are frozen. Per-edge rest lengths seeded
        // from geometry handle the mixed ring sizes. No JKR: re-added in Phase 3.
        DigitalTwin::Behaviours::VesselSpring spring{};
        spring.springStiffness    = 15.0f;
        spring.restingLength      = 1.5f; // fallback for runtime-appended edges
        spring.dampingCoefficient = 10.0f;
        spring.anchorPhalanxCells = true;
        vessel.AddBehaviour( spring ).SetHz( 60.0f );

        m_engine.SetBlueprint( m_blueprint );
    }

    void Editor::SetupVesselSproutingDemo()
    {
        // Single vessel sprout demo — Phase 3.
        //
        // A straight 2D tube vessel (6-cell rings, no branches) sits 10 units above
        // a hypoxic source cell. The biology pipeline drives sprouting from the tube:
        //
        //   1. Source consumes O2 → Hypoxic → secretes VEGF (~t=5s)
        //   2. PhalanxActivation: VEGF > 20 wakes cells closest to source (~t=10s)
        //   3. NotchDll4: lateral inhibition selects exactly 1 TipCell from the ring (~t=12s)
        //   4. Chemotaxis: TipCell migrates toward source down the VEGF gradient
        //   5. Directed mitosis: StalkCell behind TipCell divides → 1D sprout chain extends
        //   6. Ring cells with only ring edges to TipCell mature back to PhalanxCell
        //      (edge-flag filtering in mitosis_vessel_append.comp)
        //
        // Tube geometry: ringSize=6, length=24, spacing=1.5 → ~17 rings × 6 = ~102 cells.
        // Origin at (-12, 10, 0), direction +x → tube center at (0, 10, 0), 10 units above source.

        m_blueprint = DigitalTwin::SimulationBlueprint{};
        m_blueprint.SetName( "Vessel Sprouting" );
        m_blueprint.SetDomainSize( glm::vec3( 60.0f ), 2.0f );

        m_blueprint.ConfigureSpatialPartitioning()
            .SetMethod( DigitalTwin::SpatialPartitioningMethod::HashGrid )
            .SetCellSize( 3.0f )
            .SetMaxDensity( 64 )
            .SetComputeHz( 60.0f );

        // Oxygen: Gaussian centred at source; sigma=15 ensures vessel at y=+10 gets O2≈52.
        // Source consumes at 40/s → drops below hypoxiaO2=30 in ~0.5s.
        m_blueprint.AddGridField( "Oxygen" )
            .SetInitializer( DigitalTwin::GridInitializer::Gaussian( glm::vec3( 0.0f ), 15.0f, 60.0f ) )
            .SetDiffusionCoefficient( 2.0f )
            .SetDecayRate( 0.0f )
            .SetComputeHz( 60.0f );

        // VEGF: D=0.5, decay=0.12 → λ≈2 voxels. VEGF reaches the vessel (5 voxels away)
        // with a meaningful gradient so chemotaxis can drive TipCell migration.
        // High StalkCell/TipCell consumption (60/40 per cell) depletes VEGF below the PhalanxActivation
        // threshold after the ring matures, preventing PhalanxActivation from fighting maturation.
        m_blueprint.AddGridField( "VEGF" )
            .SetInitializer( DigitalTwin::GridInitializer::Constant( 0.0f ) )
            .SetDiffusionCoefficient( 0.5f )
            .SetDecayRate( 0.12f )
            .SetComputeHz( 60.0f );

        // --- Source cell at (0,0,0) ---
        auto& source = m_blueprint.AddAgentGroup( "SourceCell" )
            .SetCount( 1 )
            .SetMorphology( DigitalTwin::MorphologyGenerator::CreateSphere( 1.2f ) )
            .SetDistribution( { glm::vec4( 0.0f, 0.0f, 0.0f, 1.0f ) } )
            .SetColor( glm::vec4( 1.0f, 0.5f, 0.1f, 1.0f ) ); // orange → purple when hypoxic

        source.AddBehaviour( DigitalTwin::Behaviours::ConsumeField{ "Oxygen", 40.0f } ).SetHz( 60.0f );
        source.AddBehaviour( DigitalTwin::BiologyGenerator::StandardCellCycle()
                                 .SetBaseDoublingTime( 99.0f )
                                 .SetProliferationOxygenTarget( 60.0f )
                                 .SetArrestPressureThreshold( 999.0f )
                                 .SetHypoxiaOxygenThreshold( 30.0f )
                                 .SetNecrosisOxygenThreshold( 1.0f )
                                 .SetApoptosisRate( 0.0f )
                                 .Build() )
            .SetHz( 10.0f );
        source.AddBehaviour( DigitalTwin::Behaviours::SecreteField{
                                 "VEGF", 800.0f,
                                 static_cast<int>( DigitalTwin::LifecycleState::Hypoxic ) } )
            .SetHz( 60.0f );

        // --- Vessel tube: straight trunk, no branches ---
        auto tree = DigitalTwin::VesselTreeGenerator::BranchingTree()
            .SetOrigin( glm::vec3( -12.0f, 10.0f, 0.0f ) )
            .SetDirection( glm::vec3( 1.0f, 0.0f, 0.0f ) )
            .SetLength( 24.0f )
            .SetCellSpacing( 1.5f )
            .SetRingSize( 6 )
            .SetTubeRadius( 1.5f )
            .SetBranchingDepth( 0 )   // straight tube — no branches
            .SetCurvature( 0.0f )     // perfectly straight for predictable sprouting
            .SetSeed( 42 )
            .Build();

        DigitalTwin::Behaviours::VesselSeed seed;
        seed.segmentCounts = tree.segmentCounts;
        seed.explicitEdges = tree.edges;
        seed.edgeFlags     = tree.edgeFlags; // RING=0x1, AXIAL=0x2 — used by mitosis shader

        auto& vessel = m_blueprint.AddAgentGroup( "VesselTube" )
            .SetCount( tree.totalCells )
            .SetMorphology( DigitalTwin::MorphologyGenerator::CreateCurvedTile( 60.0f, 1.35f, 0.25f, 1.5f ) )
            .SetDistribution( tree.positions )
            .SetOrientations( tree.normals )
            .SetColor( glm::vec4( 0.5f, 0.15f, 0.15f, 1.0f ) )
            .SetInitialCellType( static_cast<int>( DigitalTwin::CellType::PhalanxCell ) )
            .AddCellTypeMorphology( static_cast<int>( DigitalTwin::CellType::PhalanxCell ),
                DigitalTwin::MorphologyGenerator::CreateCurvedTile( 60.0f, 1.35f, 0.25f, 1.5f ),
                glm::vec4( 0.5f, 0.15f, 0.15f, 1.0f ) )   // dark red — quiescent
            .AddCellTypeMorphology( static_cast<int>( DigitalTwin::CellType::StalkCell ),
                DigitalTwin::MorphologyGenerator::CreateCurvedTile( 60.0f, 1.35f, 0.25f, 1.5f ),
                glm::vec4( 1.0f, 0.8f, 0.1f, 1.0f ) )     // yellow — activated
            .AddCellTypeMorphology( static_cast<int>( DigitalTwin::CellType::TipCell ),
                DigitalTwin::MorphologyGenerator::CreateSpikySphere( 1.0f, 1.35f ),
                glm::vec4( 0.0f, 1.0f, 0.3f, 1.0f ) );    // bright green — migrating tip

        // 1. VesselSeed: upload ring + axial edges with flags for biology shader filtering
        vessel.AddBehaviour( seed );

        // 2. PhalanxActivation: threshold=35 targets only the 1-2 rings directly facing source.
        //    Deactivation=7 gives hysteresis without cycling.
        vessel.AddBehaviour( DigitalTwin::Behaviours::PhalanxActivation{
                "VEGF", 35.0f, 7.0f } )
            .SetHz( 60.0f );

        // 3a. VEGFR1 sink: StalkCells consume VEGF heavily so the activation zone stays local.
        //     High rate means the few rings that activate eat up VEGF before it can spread axially.
        //     Also depletes VEGF below PhalanxActivation threshold so ring cells can mature properly.
        vessel.AddBehaviour( DigitalTwin::Behaviours::ConsumeField{ "VEGF", 120.0f } )
            .SetRequiredCellType( static_cast<int>( DigitalTwin::CellType::StalkCell ) )
            .SetHz( 60.0f );

        // 3b. VEGFR1 passive sink: PhalanxCells express VEGFR1 (decoy receptor) at low levels.
        //     This attenuates VEGF along the whole tube length, preventing far rings from activating.
        vessel.AddBehaviour( DigitalTwin::Behaviours::ConsumeField{ "VEGF", 4.0f } )
            .SetRequiredCellType( static_cast<int>( DigitalTwin::CellType::PhalanxCell ) )
            .SetHz( 60.0f );

        // 3c. VEGFR2 sink: TipCell consumes VEGF (high VEGFR2 expression in vivo).
        vessel.AddBehaviour( DigitalTwin::Behaviours::ConsumeField{ "VEGF", 60.0f } )
            .SetRequiredCellType( static_cast<int>( DigitalTwin::CellType::TipCell ) )
            .SetHz( 60.0f );

        // 4. NotchDll4: lateral inhibition selects exactly 1 TipCell from activated StalkCells.
        //    subSteps=40 + inhibitionGain=40 ensures the inhibition wave propagates around the
        //    full ring within the 3s grace period, preventing multi-TipCell selection.
        vessel.AddBehaviour( DigitalTwin::Behaviours::NotchDll4{
                1.0f, 0.1f, 40.0f, 1.0f, 0.90f, 0.20f, "VEGF", 40u } )
            .SetHz( 60.0f );

        // 5. CellCycle: directedMitosis=true → StalkCell axially adjacent to TipCell divides
        //    along the vessel axis, extending the sprout. Non-proliferating cells mature to PhalanxCell.
        //    5s doubling time (2x faster than before) so the sprout chain extends more noticeably.
        auto stalkCycle = DigitalTwin::BiologyGenerator::StandardCellCycle()
                              .SetBaseDoublingTime( 2.0f / 3600.0f )
                              .SetProliferationOxygenTarget( 40.0f )
                              .SetArrestPressureThreshold( 5.0f )
                              .SetHypoxiaOxygenThreshold( 5.0f )
                              .SetNecrosisOxygenThreshold( 1.0f )
                              .SetApoptosisRate( 0.0f )
                              .Build();
        stalkCycle.directedMitosis = true;
        vessel.AddBehaviour( stalkCycle ).SetHz( 10.0f );

        // 6. Chemotaxis: TipCell follows the VEGF gradient toward the source.
        //    sensitivity=2 so StalkCell division can keep pace.
        //    contactInhibitionDensity=0: disabled. Once the spatial hash covers all agent groups
        //    (fixed via globalHashCapacity scaling in SimulationBuilder), a TipCell inside its origin
        //    ring finds ~15-20 vessel neighbors within hashCellSize*2 = 6 units, saturating any
        //    sensible density threshold and freezing the tip. Phase 5 VE-Cadherin adhesion will
        //    provide proper tip-tube separation mechanics.
        vessel.AddBehaviour( DigitalTwin::Behaviours::Chemotaxis{ "VEGF", 2.0f, 0.002f, 0.8f, 0.0f } )
            .SetHz( 60.0f )
            .SetRequiredCellType( static_cast<int>( DigitalTwin::CellType::TipCell ) );

        // 7. VesselSpring: maintains ring shape and sprout chain spacing.
        //    Per-edge rest lengths from geometry handle mixed ring/sprout lengths.
        //    anchorPhalanxCells=true freezes the tube; only TipCell and sprout chain move.
        DigitalTwin::Behaviours::VesselSpring spring{};
        spring.springStiffness    = 15.0f;
        spring.restingLength      = 1.5f;
        spring.dampingCoefficient = 10.0f;
        spring.anchorPhalanxCells = true;
        vessel.AddBehaviour( spring ).SetHz( 60.0f );

        // Biomechanics (JKR) intentionally omitted for the vessel group.
        // VesselSpring + PhalanxCell anchoring maintain tube integrity.
        // JKR Hertzian repulsion on closely-packed ring geometry (spacing=1.5, maxRadius=1.5)
        // produces explosive forces — overlap=1.5 → force≈27 per pair.
        // Phase 5 VE-Cadherin adhesion will provide proper vessel cell-cell contact mechanics.

        m_engine.SetBlueprint( m_blueprint );
    }

    void Editor::SetupAngiogenesisDemo()
    {
        // Full angiogenesis demo — Phase 4.
        //
        // A Y-shaped bifurcated vessel: a trunk from the left that forks into two branches.
        // A hypoxic tumour sits ahead of the fork (~10 units from each branch tip), in the
        // notch of the Y. Both branches independently sprout toward the tumour:
        //
        //   1. Tumour consumes O2 → Hypoxic → secretes VEGF (~t=2s)
        //   2. PhalanxActivation wakes the rings nearest each branch tip (~t=5s)
        //   3. NotchDll4 selects 1 TipCell per branch (~t=7s)
        //   4. Both TipCells migrate toward the tumour via chemotaxis
        //   5. Directed mitosis extends a sprout chain behind each TipCell
        //   6. Sprouts meet near the tumour → Anastomosis fires (~t=18s)
        //   7. Matured vessel loop perfuses O2 back into the domain
        //
        // Vessel geometry:
        //   Trunk: origin (-15, 0, 0), direction +X, length=10 → junction at (-5, 0, 0)
        //   BranchingAngle=40°, LengthFalloff=0.8 → each branch ~8 units long
        //   Branch tips ≈ (1.1, ±5.1, 0) — ~10 units from the tumour at (10, 0, 0)

        m_blueprint = DigitalTwin::SimulationBlueprint{};
        m_blueprint.SetName( "Vessel Angiogenesis" );
        // Domain=82 → 41×41×41 voxels at voxelSize=2.  With an ODD voxel count, (0,0,0) is a
        // voxel centre, so the tumour at (10,0,0) hits EXACTLY the centre of its voxel and
        // VEGF is released symmetrically in ±Y — critical for equal activation of both branches.
        // (Domain=80 gives 40 voxels per axis; voxel centres land at odd positions ±1,±3,…
        //  so the tumour at y=0 mapped to voxel-centre y=1, making upper/lower branches unequal.)
        m_blueprint.SetDomainSize( glm::vec3( 82.0f ), 2.0f );

        m_blueprint.ConfigureSpatialPartitioning()
            .SetMethod( DigitalTwin::SpatialPartitioningMethod::HashGrid )
            .SetCellSize( 3.0f )
            .SetMaxDensity( 64 )
            .SetComputeHz( 60.0f );

        // O2: Gaussian centred at the vessel junction; sigma=20 keeps vessel rings well-oxygenated at t=0.
        // Decay=0.05 → diffusion length λ = sqrt(D/decay) = sqrt(2/0.05) ≈ 6.3 units.
        // Cells >6 units from the nearest vessel segment become hypoxic naturally.
        m_blueprint.AddGridField( "Oxygen" )
            .SetInitializer( DigitalTwin::GridInitializer::Gaussian( glm::vec3( -5.0f, 0.0f, 0.0f ), 20.0f, 80.0f ) )
            .SetDiffusionCoefficient( 2.0f )
            .SetDecayRate( 0.05f )
            .SetComputeHz( 60.0f );

        // VEGF: D=0.5, decay=0.12 → λ≈2 units (same as sprouting demo).
        // Steeper gradient ensures only the tip ring of each branch exceeds the activation threshold.
        m_blueprint.AddGridField( "VEGF" )
            .SetInitializer( DigitalTwin::GridInitializer::Constant( 0.0f ) )
            .SetDiffusionCoefficient( 0.5f )
            .SetDecayRate( 0.12f )
            .SetComputeHz( 60.0f );

        // --- Tumour source — ahead of the fork, in the notch of the Y ---
        auto& tumour = m_blueprint.AddAgentGroup( "Tumour" )
            .SetCount( 1 )
            .SetMorphology( DigitalTwin::MorphologyGenerator::CreateSphere( 1.2f ) )
            .SetDistribution( { glm::vec4( 10.0f, 0.0f, 0.0f, 1.0f ) } )
            .SetColor( glm::vec4( 1.0f, 0.5f, 0.1f, 1.0f ) );

        tumour.AddBehaviour( DigitalTwin::Behaviours::ConsumeField{ "Oxygen", 60.0f } ).SetHz( 60.0f );
        tumour.AddBehaviour( DigitalTwin::BiologyGenerator::StandardCellCycle()
                                 .SetBaseDoublingTime( 99.0f )
                                 .SetProliferationOxygenTarget( 60.0f )
                                 .SetArrestPressureThreshold( 999.0f )
                                 .SetHypoxiaOxygenThreshold( 25.0f )
                                 .SetNecrosisOxygenThreshold( 0.0f )  // never necrotic — stays hypoxic and secretes VEGF throughout
                                 .SetApoptosisRate( 0.0f )
                                 .Build() )
            .SetHz( 10.0f );
        tumour.AddBehaviour( DigitalTwin::Behaviours::SecreteField{
                                 "VEGF", 800.0f,
                                 static_cast<int>( DigitalTwin::LifecycleState::Hypoxic ) } )
            .SetHz( 60.0f );

        // --- Bifurcated vessel tree ---
        // Trunk from (-20,0,0) along +X; junction at (-5,0,0); two branches at ±40°.
        // Trunk=15 units, LengthFalloff=0.9 → branches 13.5 units each.
        // Branch tips at approximately (5.3, ±8.7, 0) — ~10 units from tumour at (10, 0, 0).
        auto tree = DigitalTwin::VesselTreeGenerator::BranchingTree()
            .SetOrigin( glm::vec3( -20.0f, 0.0f, 0.0f ) )
            .SetDirection( glm::vec3( 1.0f, 0.0f, 0.0f ) )
            .SetLength( 15.0f )
            .SetCellSpacing( 1.5f )
            .SetRingSize( 6 )
            .SetTubeRadius( 1.5f )
            .SetBranchingAngle( 40.0f )
            .SetBranchingDepth( 1 )
            .SetLengthFalloff( 0.9f )
            .SetAngleJitter( 0.0f )    // symmetric branches — no random jitter
            .SetBranchProbability( 1.0f )
            .SetTubeRadiusFalloff( 0.79f )
            .SetCurvature( 0.0f )
            .SetSeed( 42 )
            .Build();

        DigitalTwin::Behaviours::VesselSeed seed;
        seed.segmentCounts = tree.segmentCounts;
        seed.explicitEdges = tree.edges;
        seed.edgeFlags     = tree.edgeFlags;

        auto& vessel = m_blueprint.AddAgentGroup( "VesselTree" )
            .SetCount( tree.totalCells )
            .SetMorphology( DigitalTwin::MorphologyGenerator::CreateCurvedTile( 60.0f, 1.35f, 0.25f, 1.5f ) )
            .SetDistribution( tree.positions )
            .SetOrientations( tree.normals )
            .SetColor( glm::vec4( 0.5f, 0.15f, 0.15f, 1.0f ) )
            .SetInitialCellType( static_cast<int>( DigitalTwin::CellType::PhalanxCell ) )
            .AddCellTypeMorphology( static_cast<int>( DigitalTwin::CellType::PhalanxCell ),
                DigitalTwin::MorphologyGenerator::CreateCurvedTile( 60.0f, 1.35f, 0.25f, 1.5f ),
                glm::vec4( 0.5f, 0.15f, 0.15f, 1.0f ) )
            .AddCellTypeMorphology( static_cast<int>( DigitalTwin::CellType::StalkCell ),
                DigitalTwin::MorphologyGenerator::CreateCurvedTile( 60.0f, 1.35f, 0.25f, 1.5f ),
                glm::vec4( 1.0f, 0.8f, 0.1f, 1.0f ) )
            .AddCellTypeMorphology( static_cast<int>( DigitalTwin::CellType::TipCell ),
                DigitalTwin::MorphologyGenerator::CreateSpikySphere( 1.0f, 1.35f ),
                glm::vec4( 0.0f, 1.0f, 0.3f, 1.0f ) );

        vessel.AddBehaviour( seed );

        // Anastomosis runs FIRST — before PhalanxActivation — so it can check the TipCell
        // pair while both cells are still cellType=1.  If PhalanxActivation ran first, a
        // transient VEGF dip (StalkCell ring consumption ≈720/s vs tumor secretion 800/s)
        // could revert a TipCell to PhalanxCell in the same frame, permanently missing the
        // connection window.  The spatial hash used here is built at frame start so positions
        // are consistent regardless of where in the frame Anastomosis dispatches.
        vessel.AddBehaviour( DigitalTwin::Behaviours::Anastomosis{ 5.0f, true } ).SetHz( 60.0f );

        // PhalanxActivation: activationThreshold=45 (not the sprouting demo's 35).
        // The Y-branches are angled 40° toward the tumor, so they curve *through* their
        // closest-approach point. Distance to tumor along the final ~4 rings of each branch
        // spans only 9.64–9.85 units — a 0.21-unit window — so the VEGF field is nearly
        // flat there. Using the sprouting threshold (35) activates ~24 cells per branch,
        // ~2.5x the sprouting demo's clean 6–12 cells. Raising to 45 shrinks the band to
        // the 1–2 rings at the true VEGF peak, matching the sprouting demo's look.
        // Deactivation=7: same as sprouting — StalkCells behind the tip quickly mature once
        // local VEGF drops below 7 (120/s consumption depletes the local voxel fast).
        vessel.AddBehaviour( DigitalTwin::Behaviours::PhalanxActivation{ "VEGF", 45.0f, 7.0f } )
            .SetHz( 60.0f );

        vessel.AddBehaviour( DigitalTwin::Behaviours::ConsumeField{ "VEGF", 120.0f } )
            .SetRequiredCellType( static_cast<int>( DigitalTwin::CellType::StalkCell ) )
            .SetHz( 60.0f );
        vessel.AddBehaviour( DigitalTwin::Behaviours::ConsumeField{ "VEGF", 4.0f } )
            .SetRequiredCellType( static_cast<int>( DigitalTwin::CellType::PhalanxCell ) )
            .SetHz( 60.0f );
        vessel.AddBehaviour( DigitalTwin::Behaviours::ConsumeField{ "VEGF", 60.0f } )
            .SetRequiredCellType( static_cast<int>( DigitalTwin::CellType::TipCell ) )
            .SetHz( 60.0f );

        vessel.AddBehaviour( DigitalTwin::Behaviours::NotchDll4{
                1.0f, 0.1f, 40.0f, 1.0f, 0.90f, 0.20f, "VEGF", 40u } )
            .SetHz( 60.0f );

        // Matched to sprouting demo: 2s doubling, sensitivity=2.0, gain=0.8.
        // Faster division + faster tip keeps the sprout chain compact. StalkCells behind the
        // tip mature quickly (deactivation=7) so the chain body is PhalanxCell (dark red),
        // matching the clean single-yellow-StalkCell look from the sprouting demo.
        auto stalkCycle = DigitalTwin::BiologyGenerator::StandardCellCycle()
                              .SetBaseDoublingTime( 2.0f / 3600.0f )
                              .SetProliferationOxygenTarget( 40.0f )
                              .SetArrestPressureThreshold( 5.0f )
                              .SetHypoxiaOxygenThreshold( 5.0f )
                              .SetNecrosisOxygenThreshold( 1.0f )
                              .SetApoptosisRate( 0.0f )
                              .Build();
        stalkCycle.directedMitosis = true;
        // Restrict to StalkCells only — TipCells migrate into the hypoxic tumour zone
        // intentionally; running the lifecycle on them turns them blue (state=Hypoxic) when O2 < 5.
        vessel.AddBehaviour( stalkCycle )
            .SetHz( 10.0f )
            .SetRequiredCellType( static_cast<int>( DigitalTwin::CellType::StalkCell ) );

        // Matched to sprouting demo: sensitivity=2.0, gain=0.8.
        // contactInhibitionDensity=0: disabled (same reason as sprouting demo — vessel ring
        // neighbors saturate any threshold once the hash covers all groups).
        vessel.AddBehaviour( DigitalTwin::Behaviours::Chemotaxis{ "VEGF", 2.0f, 0.002f, 0.8f, 0.0f } )
            .SetHz( 60.0f )
            .SetRequiredCellType( static_cast<int>( DigitalTwin::CellType::TipCell ) );

        DigitalTwin::Behaviours::VesselSpring spring{};
        spring.springStiffness    = 15.0f;
        spring.restingLength      = 1.5f;
        spring.dampingCoefficient = 10.0f;
        spring.anchorPhalanxCells = true;
        vessel.AddBehaviour( spring ).SetHz( 60.0f );

        // PhalanxCell-only perfusion: unconnected sprout StalkCells have no lumen and must not
        // inject O2 — that would re-oxygenate the tumour prematurely and kill the VEGF signal.
        // After anastomosis, PhalanxActivation matures the new StalkCells to PhalanxCells
        // (VEGF drops once the vessel loop delivers O2), and they then begin perfusing.
        vessel.AddBehaviour( DigitalTwin::Behaviours::Perfusion{ "Oxygen", 4.0f } )
            .SetHz( 60.0f )
            .SetRequiredCellType( static_cast<int>( DigitalTwin::CellType::PhalanxCell ) );

        // JKR intentionally omitted for the vessel group — see sprouting demo for rationale.
        // Phase 5 VE-Cadherin adhesion will provide proper vessel cell-cell contact mechanics.

        m_engine.SetBlueprint( m_blueprint );
    }

} // namespace Gaudi