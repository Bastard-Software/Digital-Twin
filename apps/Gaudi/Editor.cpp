#include "Editor.h"

#include "IconsFontAwesome5.h"
#include "demos/Demos.h"
#include <core/FileSystem.h>
#include <core/Log.h>
#include <filesystem>
#include <imgui.h>
#include <spdlog/sinks/ringbuffer_sink.h>
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

        // ── ImGui ini: override auto-persistence with an explicit path pair.
        // User copy  (writable, adjacent to exe):  imgui.ini
        // Shipped default (read-only, in assets/): default_imgui.ini
        // New clone: no user copy → load shipped default.
        // After any drag: user copy is written on shutdown; shipped default untouched.
        io.IniFilename = nullptr; // disable ImGui's own auto-save
        {
            const std::filesystem::path userIni =
                std::filesystem::current_path() / "imgui.ini";
            if( std::filesystem::exists( userIni ) )
            {
                ImGui::LoadIniSettingsFromDisk( userIni.string().c_str() );
            }
            else
            {
                std::filesystem::path defIni = fs->ResolvePath( "default_imgui.ini" );
                if( std::filesystem::exists( defIni ) )
                    ImGui::LoadIniSettingsFromDisk( defIni.string().c_str() );
            }
            m_userIniPath = userIni.string();
        }

        std::string roboto  = fs->ResolvePath( "fonts/Roboto-Medium.ttf" ).string();
        std::string faSolid = fs->ResolvePath( "fonts/fa-solid-900.ttf" ).string();
        io.Fonts->AddFontFromFileTTF( roboto.c_str(), 16.0f );
        ImFontConfig iconCfg;
        iconCfg.MergeMode                 = true;
        iconCfg.GlyphMinAdvanceX          = 16.0f;
        iconCfg.PixelSnapH                = true;
        static const ImWchar iconRanges[] = { ICON_MIN_FA, ICON_MAX_FA, 0 };
        io.Fonts->AddFontFromFileTTF( faSolid.c_str(), 16.0f, &iconCfg, iconRanges );

        // Attach a ringbuffer sink to both spdlog loggers so the Console panel can display
        // engine and app log output without needing stdout.
        m_logSink = std::make_shared<spdlog::sinks::ringbuffer_sink_mt>( 2048 );
        m_logSink->set_pattern( "[%T] [%l] %n: %v" );
        DigitalTwin::Log::GetCoreLogger()->sinks().push_back( m_logSink );
        DigitalTwin::Log::GetClientLogger()->sinks().push_back( m_logSink );

        // Clear the default name so the Hierarchy shows an empty placeholder
        // until the user creates a new simulation or loads a demo.
        m_blueprint.SetName( "" );
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
            DemoSetupFn setupFn;
        };

        static const Demo k_demos[] = {
            { "Secrete",
              "A single cell secretes into an initially empty field.\n"
              "Watch the substance accumulate and diffuse outward.\n\n"
              "Demonstrates: SecreteField, field diffusion.",
              &Demos::SetupSecreteDemo },

            { "Consume",
              "A single cell drains a Gaussian pre-seeded field.\n"
              "Watch the local depletion grow around the cell.\n\n"
              "Demonstrates: ConsumeField, field depletion.",
              &Demos::SetupConsumeDemo },

            { "Chemotaxis",
              "Source cell (green) consumes O2 and goes Hypoxic\n"
              "after a few seconds — then starts secreting VEGF.\n"
              "Responder cell (blue) slowly drifts toward\n"
              "the growing VEGF cloud.\n\n"
              "Demonstrates: CellCycle lifecycle, SecreteField\n"
              "(Hypoxic only), Chemotaxis gradient sensing.",
              &Demos::SetupChemotaxisDemo },

            { "Cell Cycle",
              "A small tumour cluster grows, divides, and arrests\n"
              "under pressure and hypoxia.\n\n"
              "Demonstrates: CellCycle, JKR biomechanics, O2 coupling.",
              &Demos::SetupCellCycleDemo },

            { "Diffusion & Decay",
              "A sphere of substance diffuses outward and decays\n"
              "to equilibrium. No cells — field physics only.\n\n"
              "Demonstrates: GridField diffusion, decay rate,\n"
              "Sphere initializer.",
              &Demos::SetupDiffusionDecayDemo },

            { "Brownian Motion",
              "27 particles in a 3x3x3 grid perform pure thermal\n"
              "random-walk. No fields or forces — each drifts\n"
              "independently.\n\n"
              "Demonstrates: BrownianMotion behaviour.",
              &Demos::SetupBrownianMotionDemo },

            { "JKR Packing",
              "10 cells packed tightly at the origin explode apart\n"
              "under pure Hertz repulsion and settle into a stable\n"
              "packing. Zero adhesion energy.\n\n"
              "Demonstrates: Biomechanics repulsion, damping.",
              &Demos::SetupJKRPackingDemo },

            { "Lifecycle",
              "A single cell consumes its local O2 supply and\n"
              "progresses Live (red) -> Hypoxic (purple) ->\n"
              "Necrotic (dark).\n\n"
              "Demonstrates: CellCycle lifecycle states,\n"
              "ConsumeField, O2 depletion.",
              &Demos::SetupLifecycleDemo },

            { "Static Vessel Tree",
              "A branching 2D tube vessel tree — rings of 6\n"
              "plate-like endothelial cells held together by\n"
              "spring forces. No biology, no VEGF.\n\n"
              "Purpose: verify tree shape, ring topology, and\n"
              "structural stability (VesselSpring + Biomechanics).\n\n"
              "Demonstrates: VesselTreeGenerator, VesselSeed\n"
              "(explicit edges), disc morphology, VesselSpring.",
              &Demos::SetupStaticVesselTreeDemo },

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
              &Demos::SetupVesselSproutingDemo },

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
              &Demos::SetupAngiogenesisDemo },

            { "Tissue Sorting",
              "Two cell types (red=Epithelial, blue=Mesenchymal)\n"
              "with orthogonal cadherin profiles start randomly\n"
              "mixed in a sphere.\n\n"
              "Homophilic adhesion drives spontaneous sorting:\n"
              "same-type cells cluster together, cross-type\n"
              "adhesion is zero.\n\n"
              "Demonstrates: CadherinAdhesion, differential\n"
              "adhesion hypothesis (Steinberg 1963).",
              &Demos::SetupTissueSortingDemo },

            { "EC Contact",
              "5 flat endothelial tiles in a 2D cross pattern.\n\n"
              "Outer tiles start at different Y rotations (+20, +15,\n"
              "-12, -8 deg). VE-cadherin ramps up; distributed hull\n"
              "contacts (4 corners + 4 edge midpoints) generate\n"
              "adhesion-only torques that drive all tiles to parallel\n"
              "edge-to-edge contact within ~10 s regardless of\n"
              "initial rotation direction or magnitude.\n\n"
              "Demonstrates: flat tile JKR, CadherinAdhesion,\n"
              "mechanistic edge alignment morphogenesis.",
              &Demos::SetupECContactDemo },

            { "Cell Mechanics Zoo",
              "5 cell types with distinct morphologies and cadherin\n"
              "profiles in a single tumour microenvironment scene.\n\n"
              "Zones:\n"
              "  Centre: Tumour (SpikySphere) - packed tight, zero\n"
              "    adhesion -> explosion + tumbling (hull torque)\n"
              "  +X: Epithelial (Cube) - E-cadherin lattice ->\n"
              "    stable aggregate with face-alignment torque\n"
              "  -X: Endothelial (Tile) - VE-cadherin, varied\n"
              "    angles -> edge-alignment sheet formation\n"
              "  +Z: Fibroblasts (Ellipsoid) - N-cadherin -> elongated\n"
              "    packing with axis-alignment torque\n"
              "  -Z: Immune (Sphere) - chemotaxis toward tumour\n"
              "    chemokine, bounces off all other groups\n\n"
              "Demonstrates: SpikySphere + Ellipsoid hull torque,\n"
              "differential adhesion, steric repulsion, chemotaxis.",
              &Demos::SetupCellMechanicsZooDemo },

            { "Endothelial Tube",
              "200 VE-cadherin endothelial cells start randomly\n"
              "scattered inside a solid cylinder.\n\n"
              "Apical-basal polarity develops from neighbor\n"
              "geometry: surface cells polarise outward (w->1),\n"
              "interior cells stay symmetric (w->0).\n\n"
              "Polarity-modulated JKR weakens interior contacts\n"
              "-> cavity opens -> lumenised tube self-assembles.\n\n"
              "Demonstrates: CellPolarity + CadherinAdhesion,\n"
              "lumen morphogenesis (Nakamura 2026).",
              &Demos::SetupEndothelialTubeDemo },

            { "Stress Test",
              "100,000 agents across 3 groups in a 250-unit domain.\n"
              "Designed to drive GPU compute into the ms range\n"
              "for meaningful profiler data.\n\n"
              "Groups:\n"
              "  Tumour  (50,000) — JKR + Cadherin + Brownian\n"
              "    + VEGF secretion + O2 consumption\n"
              "  Stromal (45,000) — JKR + Brownian + O2 consumption\n"
              "  Immune   (5,000) — JKR + slow Chemotaxis (VEGF)\n"
              "    + Brownian + O2 consumption\n\n"
              "Target: ~30 FPS. Use this scene for TraceCapture\n"
              "and performance profiling.",
              &Demos::SetupStressTestDemo },
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

    // Stops the running simulation (if any), resets the blueprint, calls fn to populate it,
    // then notifies the engine so it picks up the new blueprint on the next Play().
    void Editor::LoadDemo( DemoSetupFn fn )
    {
        if( m_engine.GetState() != DigitalTwin::EngineState::RESET )
            m_engine.Stop();
        m_blueprint = DigitalTwin::SimulationBlueprint{};
        fn( m_blueprint );
        m_engine.SetBlueprint( m_blueprint );

        if( m_consolePanel )
            m_consolePanel->Clear();
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

            if( ImGui::BeginMenu( "View" ) )
            {
                bool showOverlay = m_engine.IsShowingStatsOverlay();
                if( ImGui::MenuItem( "Stats Overlay", nullptr, &showOverlay ) )
                    m_engine.SetShowStatsOverlay( showOverlay );

                if( m_consolePanel )
                {
                    bool showConsole = m_consolePanel->IsVisible();
                    if( ImGui::MenuItem( "Console", nullptr, &showConsole ) )
                        m_consolePanel->SetVisible( showConsole );
                }

                ImGui::EndMenu();
            }

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
                LoadDemo( &Demos::SetupEmptyBlueprint );
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

        // Persist the current docking layout so it survives restart.
        if( !m_userIniPath.empty() )
            ImGui::SaveIniSettingsToDisk( m_userIniPath.c_str() );

        DT_INFO( "Gaudi Editor closing..." );
        m_engine.Shutdown();
    }

} // namespace Gaudi
