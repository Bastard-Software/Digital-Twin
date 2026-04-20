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
            // Editor-side viz preset so blueprints can stay biology-pure.
            std::optional<DigitalTwin::GridVisualizationSettings> visualization;
        };

        struct DemoCategory
        {
            const char*             name;
            std::vector<Demo>       demos;
            bool                    defaultOpen;
        };

        // Lambdas for visualization presets (readability)
        auto gridSliceViz = []() {
            DigitalTwin::GridVisualizationSettings v;
            v.active     = true;
            v.fieldIndex = 0;
            v.mode       = DigitalTwin::GridVisualizationMode::SLICE_2D;
            return v;
        };
        auto gridSliceGamma = []() {
            DigitalTwin::GridVisualizationSettings v;
            v.active             = true;
            v.fieldIndex         = 0;
            v.mode               = DigitalTwin::GridVisualizationMode::SLICE_2D;
            v.fieldVis.gamma     = 0.6f;
            v.fieldVis.alphaCutoff = 0.0f;
            return v;
        };

        // Grouped demo catalogue. Order within categories reflects "simple → complex".
        // Vessels category is placeholder for Item 2 phase deliverables (StraightTubeDemo,
        // TaperingTubeDemo, BranchingTreeDemo, DesignedVesselDemo, ...) — empty post-demolition.
        static const std::vector<DemoCategory> k_categories = {
            { "Basics", {
                { "Empty Blueprint",
                  "A blank canvas: no agents, no fields, no behaviours.\n"
                  "Useful to start a scene from scratch.\n\n"
                  "Demonstrates: fresh SimulationBlueprint.",
                  &Demos::SetupEmptyBlueprint },
                { "Brownian Motion",
                  "27 particles in a 3x3x3 grid perform pure thermal\n"
                  "random-walk. No fields or forces — each drifts\n"
                  "independently.\n\n"
                  "Demonstrates: BrownianMotion behaviour.",
                  &Demos::SetupBrownianMotionDemo },
                { "Lifecycle",
                  "A single cell consumes its local O2 supply and\n"
                  "progresses Live (red) -> Hypoxic (purple) ->\n"
                  "Necrotic (dark).\n\n"
                  "Demonstrates: CellCycle lifecycle states,\n"
                  "ConsumeField, O2 depletion.",
                  &Demos::SetupLifecycleDemo },
                { "Cell Cycle",
                  "A small tumour cluster grows, divides, and arrests\n"
                  "under pressure and hypoxia.\n\n"
                  "Demonstrates: CellCycle, JKR biomechanics, O2 coupling.",
                  &Demos::SetupCellCycleDemo },
            }, true },

            { "Fields & Diffusion", {
                { "Diffusion & Decay",
                  "A sphere of substance diffuses outward and decays\n"
                  "to equilibrium. No cells — field physics only.\n\n"
                  "Demonstrates: GridField diffusion, decay rate,\n"
                  "Sphere initializer.",
                  &Demos::SetupDiffusionDecayDemo, gridSliceGamma() },
                { "Secrete",
                  "A single cell secretes into an initially empty field.\n"
                  "Watch the substance accumulate and diffuse outward.\n\n"
                  "Demonstrates: SecreteField, field diffusion.",
                  &Demos::SetupSecreteDemo, gridSliceViz() },
                { "Consume",
                  "A single cell drains a Gaussian pre-seeded field.\n"
                  "Watch the local depletion grow around the cell.\n\n"
                  "Demonstrates: ConsumeField, field depletion.",
                  &Demos::SetupConsumeDemo, gridSliceViz() },
                { "Chemotaxis",
                  "Source cell (green) consumes O2 and goes Hypoxic\n"
                  "after a few seconds — then starts secreting VEGF.\n"
                  "Responder cell (blue) slowly drifts toward\n"
                  "the growing VEGF cloud.\n\n"
                  "Demonstrates: CellCycle lifecycle, SecreteField\n"
                  "(Hypoxic only), Chemotaxis gradient sensing.",
                  &Demos::SetupChemotaxisDemo },
            }, false },

            { "Cell Mechanics", {
                { "JKR Packing",
                  "10 cells packed tightly at the origin explode apart\n"
                  "under pure Hertz repulsion and settle into a stable\n"
                  "packing. Zero adhesion energy.\n\n"
                  "Demonstrates: Biomechanics repulsion, damping.",
                  &Demos::SetupJKRPackingDemo },
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
                { "Stress Test",
                  "100,000 agents across 3 groups in a 250-unit domain.\n"
                  "Designed to drive GPU compute into the ms range\n"
                  "for meaningful profiler data.\n\n"
                  "Groups:\n"
                  "  Tumour  (50,000) - JKR + Cadherin + Brownian\n"
                  "    + VEGF secretion + O2 consumption\n"
                  "  Stromal (45,000) - JKR + Brownian + O2 consumption\n"
                  "  Immune   (5,000) - JKR + slow Chemotaxis (VEGF)\n"
                  "    + Brownian + O2 consumption\n\n"
                  "Target: ~30 FPS. Use this scene for TraceCapture\n"
                  "and performance profiling.",
                  &Demos::SetupStressTestDemo },
            }, false },

            { "Endothelial", {
                { "EC Blob (suspension / hanging drop)",
                  "~100 VE-cadherin endothelial cells in a 3D cloud with\n"
                  "NO substrate. Biology: hanging drop / ULA culture.\n\n"
                  "Expected: solid spheroid(s) via cadherin-belt junctions.\n"
                  "NO apical-basal polarity (no BM seed). NO lumen.\n\n"
                  "Paired negative control for EC 2D Matrigel.",
                  &Demos::SetupECBlobDemo },
                { "EC 2D Matrigel (monolayer / cord)",
                  "Same ~100-cell drop as EC Blob, pipetted onto a 2D\n"
                  "Matrigel-like basement-membrane plate at y=0.\n\n"
                  "Biology: 2D Matrigel tube-formation assay (Kubota\n"
                  "1988, Arnaoutova 2009). ECs anchor to the BM via\n"
                  "integrin, flatten into a MONOLAYER, and form cord-\n"
                  "like networks over 4-24 h. NO hollow tube with lumen\n"
                  "- that phenotype requires 3D collagen gel (future\n"
                  "ECTubeDemo with volumetric ECM, roadmap item 5).\n\n"
                  "Positive control for EC Blob: the ONLY difference\n"
                  "between the two demos is the plate.",
                  &Demos::SetupEC2DMatrigelDemo },
                { "EC Tube (3D ECM placeholder)",
                  "Same ~100-cell drop as EC Blob, now inside a four-\n"
                  "plate channel along +X. Floor + ceiling + two Z\n"
                  "walls frame the cluster, providing BM contact on\n"
                  "2-4 sides per cell.\n\n"
                  "Biology: 3D-ECM placeholder. Approximates the\n"
                  "collagen-gel environment where endothelial cords\n"
                  "can undergo hollowing (Strilic 2009). The flat-plate\n"
                  "channel is NOT a true 3D ECM - a volumetric ECM\n"
                  "primitive lands with roadmap item 5.\n\n"
                  "All three EC demos use IDENTICAL cell parameters;\n"
                  "phenotypic divergence comes purely from the\n"
                  "environment (no plate / one plate / four-plate\n"
                  "channel). Same cells, different substrates - the\n"
                  "classical positive/negative control experimental\n"
                  "design.",
                  &Demos::SetupECTubeDemo },
            }, true },

            { "Vessels", {
                // Populated phase-by-phase: Phase 2.1 adds TwoShape; Phase 2.2 adds
                // PuzzlePiecePalette; Phases 2.3+ will append StraightTube, TaperingTube,
                // BranchingTree, DesignedVessel.
                { "Two Shape",
                  "Phase 2.1 engine-plumbing verification. A single AgentGroup\n"
                  "contains 50 cells split 50/50 across two mesh variants:\n"
                  "  Row 1 (z = -2): variant 0, flat disc\n"
                  "  Row 2 (z = +2): variant 1, curved tile\n\n"
                  "Proves the bit-packed morphology-index dispatch pipeline\n"
                  "end-to-end (PhenotypeData.cellType upper-16-bit packing ->\n"
                  "SimulationBuilder multi-DrawMeta emission -> build_indirect.comp\n"
                  "variant match). No biology, no physics - just the plumbing\n"
                  "needed for Phase 2.2's real puzzle-piece primitives.",
                  &Demos::SetupTwoShapeDemo },

                { "Puzzle Piece Palette",
                  "Phase 2.2 puzzle-piece primitive palette. Four cells in a row,\n"
                  "each rendering a distinct morphology variant inside a single\n"
                  "AgentGroup (Phase 2.1 per-cell dispatch):\n"
                  "  x = -4.5   CurvedTile      (Item 1 reference tile)\n"
                  "  x = -1.5   ElongatedQuad   (Davies 2009 flow-aligned EC)\n"
                  "  x = +1.5   PentagonDefect  (+pi/3 Gaussian curvature)\n"
                  "  x = +4.5   HeptagonDefect  (-pi/3 Gaussian curvature)\n\n"
                  "Static render verification only. Phases 2.3+ compose these\n"
                  "variants onto vessel surfaces with adaptive ring counts\n"
                  "(Aird 2007) and Stone-Wales 5/7 defect insertion at diameter\n"
                  "transitions + bifurcation carinas (Stone & Wales 1986;\n"
                  "Chiu & Chien 2011).",
                  &Demos::SetupPuzzlePiecePaletteDemo },

                { "Straight Tube",
                  "Phase 2.3 refactored VesselTreeGenerator — pure cell placer.\n"
                  "A single straight tube (radius 2.5, length 15) with adaptive\n"
                  "ring count (2*pi*r / ECWidth; Aird 2007) in a staggered brick\n"
                  "pattern (Davies 2009). Each cell is emitted with position +\n"
                  "quaternion orientation (local +Y -> radial outward) + polarity\n"
                  "seed (radial outward, magnitude 1.0).\n\n"
                  "No BM plate: pre-seeded polarity self-sustains via Phase-4.5\n"
                  "junctional propagation (Bryant 2010; St Johnston & Ahringer\n"
                  "2010). Same Item-1 behaviour stack as ECBlob / EC2DMatrigel /\n"
                  "ECTube - proving the cell-based physics can hold a designed\n"
                  "tube without the legacy vessel-edge graph (Phase 2.0 demolition).\n\n"
                  "Expected: a tube of elongated rhomboids aligned along +X,\n"
                  "alternating rings offset by half a cell-width. Should hold\n"
                  "shape throughout the sim with polarity sustained everywhere.",
                  &Demos::SetupStraightTubeDemo },

                { "Curved Tube",
                  "Phase 2.3 addendum — same tube as Straight Tube, but with\n"
                  "a non-zero SetCurvature(0.2) that deflects the centreline\n"
                  "via a quadratic Bezier. Exercises the parallel-transported\n"
                  "orientation frames on a curved centreline — the geometry\n"
                  "Phase 2.6 DesignedVesselDemo will need on every branch.\n\n"
                  "Radius 2.5, length 25, seed 7. Provisional demo: will be\n"
                  "removed once DesignedVesselDemo ships, or evolved into a\n"
                  "branching-tree demo during Phase 2.5.",
                  &Demos::SetupCurvedTubeDemo },

                { "Tapering Tube",
                  "Phase 2.4 tapering tube with Stone-Wales 5/7 defects.\n"
                  "Radius tapers linearly from 3.0 (ring ~19) at origin to 1.0\n"
                  "(ring ~6) at the far end. Every ring-count transition inserts\n"
                  "pentagon + heptagon defect pairs (Stone & Wales 1986;\n"
                  "Iijima 1991) symmetrically around the ring:\n"
                  "   heptagons (wider / parent side)   - morph index 2\n"
                  "   pentagons (narrower / child side) - morph index 1\n"
                  "   rhomboids (constant-radius stretches) - morph index 0\n\n"
                  "Pair count is net-zero in Gaussian curvature, so the manifold\n"
                  "stays locally flat outside the defect zone.  Physics stack is\n"
                  "Phase 2.3 Item-1 quiescent regime (damping 500, Brownian 0.02).\n\n"
                  "Look for: visible pentagon and heptagon tiles clustered at\n"
                  "the axial positions where ring count changes; smooth taper;\n"
                  "tube holding shape without cracks at the transitions.",
                  &Demos::SetupTaperingTubeDemo },

                { "Branching Tree",
                  "Phase 2.5 three-level Y-branching tree. Trunk (r=3.0 -> 2.5)\n"
                  "splits into two L1 branches (r~1.98 -> 1.65), each of which\n"
                  "splits into two L2 capillary-scale branches (r~1.3 -> 1.08).\n"
                  "Murray's law (r_parent^3 = sum of r_child^3; factor 0.79 per\n"
                  "Murray 1926) applies at each bifurcation; within each branch\n"
                  "the parent's proportional taper shape is preserved.\n\n"
                  "All cells are rhombus tiles (fish-scale tessellation, Davies\n"
                  "2009) in a single AgentGroup. Phase 2.5 identifies carina\n"
                  "cells at each Y-junction but does not render them differently:\n"
                  "   - Parent-last-ring: 2 cells on the bisection plane\n"
                  "   - Child-first-ring: 2 cells facing the sibling branch\n"
                  "These are the cobblestone endothelium at real flow dividers\n"
                  "(Chiu & Chien 2011; van der Heiden 2013). Phase 2.6.5 dynamic\n"
                  "topology will render them as 6-to-8-sided Voronoi polygons.\n\n"
                  "Look for: artery-to-capillary radius progression across three\n"
                  "levels; clean rhombus tiling on each branch between junctions;\n"
                  "visible gap / bumpiness at each Y-junction apex (the known\n"
                  "static-primitive limit; Phase 2.6.5 closes it). Tree holds\n"
                  "shape under Item-1 physics alone - no vessel-edge springs.",
                  &Demos::SetupBranchingTreeDemo },

                { "Designed Vessel",
                  "Phase 2.6 full artery -> arteriole -> capillary hierarchy.\n"
                  "Depth-3 symmetric Y-branching tree = 15 branches (1 trunk +\n"
                  "2 L1 + 4 L2 + 8 L3 capillary leaves), 7 bifurcations, ~1000\n"
                  "cells. Ring cascade: 19 -> 9 -> 5 -> 2 (dual-seam capillary,\n"
                  "Bar 1984 floor).\n\n"
                  "Anatomical simplification: uses Murray factor 0.5 instead of\n"
                  "the biologically symmetric 2-way 0.79. Each demo-level then\n"
                  "represents ~3 real-physiology Murray bifurcations (0.79^3 =\n"
                  "0.5) so the full arteriole -> capillary scale transition fits\n"
                  "in 3 demo levels rather than the 5-8 it takes in vivo. Pries\n"
                  "& Secomb 2005 'heart -> metarteriole -> capillary' aggregate.\n\n"
                  "Pure composition demo on the Phase 2.5 bifurcation substrate\n"
                  "and Item 1 physics stack (damping 500, catch-bond 2.0 peak\n"
                  "0.3, polarity propagation 1.0). No new generator or shader\n"
                  "work.\n\n"
                  "Look for: artery-scale ring 19 trunk tapering into 8 dual-\n"
                  "seam capillary tips; Murray radius progression visible at\n"
                  "each of the 7 bifurcation apices; clean rhombus tiling on\n"
                  "each branch between junctions; known gap / bumpiness at each\n"
                  "Y-junction carina (static-primitive limit, closes in Phase\n"
                  "2.6.5 dynamic topology). Whole tree holds shape under Item-1\n"
                  "physics alone - no vessel-edge springs, no pre-registered\n"
                  "carina meshes.",
                  &Demos::SetupDesignedVesselDemo },
            }, false },
        };

        ImGuiIO& io = ImGui::GetIO();
        ImGui::SetNextWindowSize( ImVec2( 700, 420 ), ImGuiCond_Appearing );
        ImGui::SetNextWindowPos( ImVec2( io.DisplaySize.x * 0.5f, io.DisplaySize.y * 0.5f ),
                                 ImGuiCond_Appearing, ImVec2( 0.5f, 0.5f ) );

        if( !ImGui::Begin( "Demo Library", &m_showDemoBrowser, ImGuiWindowFlags_NoCollapse ) )
        {
            ImGui::End();
            return;
        }

        // Selection state: (categoryIndex, demoIndex). -1 = no selection.
        static int s_selectedCategory = 0;
        static int s_selectedDemo     = 0;

        // Left column — categorised demo tree
        ImGui::BeginChild( "##demoList", ImVec2( 220, 0 ), true );
        for( int c = 0; c < static_cast<int>( k_categories.size() ); ++c )
        {
            const auto& cat = k_categories[ c ];
            if( cat.defaultOpen )
                ImGui::SetNextItemOpen( true, ImGuiCond_Once );
            if( ImGui::CollapsingHeader( cat.name ) )
            {
                if( cat.demos.empty() )
                {
                    ImGui::Indent();
                    ImGui::TextDisabled( "(empty)" );
                    ImGui::Unindent();
                }
                for( int d = 0; d < static_cast<int>( cat.demos.size() ); ++d )
                {
                    const bool isSelected = ( s_selectedCategory == c && s_selectedDemo == d );
                    ImGui::Indent();
                    if( ImGui::Selectable( cat.demos[ d ].name, isSelected ) )
                    {
                        s_selectedCategory = c;
                        s_selectedDemo     = d;
                    }
                    ImGui::Unindent();
                }
            }
        }
        ImGui::EndChild();

        ImGui::SameLine();

        // Right column — description + load button for the current selection
        ImGui::BeginChild( "##demoDetail", ImVec2( 0, 0 ), false );

        const Demo* selected = nullptr;
        if( s_selectedCategory >= 0 && s_selectedCategory < static_cast<int>( k_categories.size() ) )
        {
            const auto& cat = k_categories[ s_selectedCategory ];
            if( s_selectedDemo >= 0 && s_selectedDemo < static_cast<int>( cat.demos.size() ) )
                selected = &cat.demos[ s_selectedDemo ];
        }

        if( selected )
            ImGui::TextWrapped( "%s", selected->description );
        else
            ImGui::TextDisabled( "Select a demo from the list on the left." );

        ImGui::SetCursorPosY( ImGui::GetWindowHeight() - ImGui::GetFrameHeightWithSpacing() - 4 );
        ImGui::BeginDisabled( selected == nullptr );
        if( ImGui::Button( "Load Demo", ImVec2( -1, 0 ) ) && selected )
        {
            LoadDemo( selected->setupFn, selected->visualization );
            m_showDemoBrowser = false;
        }
        ImGui::EndDisabled();

        ImGui::EndChild();

        ImGui::End();
    }

    // Stops the running simulation (if any), resets the blueprint, calls fn to populate it,
    // then notifies the engine so it picks up the new blueprint on the next Play().
    void Editor::LoadDemo( DemoSetupFn fn,
                           const std::optional<DigitalTwin::GridVisualizationSettings>& vizPreset )
    {
        if( m_engine.GetState() != DigitalTwin::EngineState::RESET )
            m_engine.Stop();
        m_blueprint = DigitalTwin::SimulationBlueprint{};
        fn( m_blueprint );
        m_engine.SetBlueprint( m_blueprint );

        if( vizPreset.has_value() )
            m_engine.SetGridVisualization( *vizPreset );

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

                if( m_renderSettingsPanel )
                {
                    bool show = m_renderSettingsPanel->IsVisible();
                    if( ImGui::MenuItem( "Render Settings", nullptr, &show ) )
                        m_renderSettingsPanel->SetVisible( show );
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
