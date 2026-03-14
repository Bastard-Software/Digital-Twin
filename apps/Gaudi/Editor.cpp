#include "Editor.h"

#include <core/Log.h>
#include <imgui.h>
#include <random>
#include <simulation/BiologyGenerator.h>
#include <simulation/BiomechanicsGenerator.h>
#include <simulation/MorphologyGenerator.h>
#include <simulation/SpatialDistribution.h>

namespace Gaudi
{
    void Editor::Init()
    {
        DigitalTwin::DigitalTwinConfig config;
        config.headless        = false;
        config.windowDesc.mode = DigitalTwin::WindowMode::FULLSCREEN_WINDOWED;
        m_engine.Initialize( config );

        SetupInitialBlueprint();
    }

    void Editor::SetupInitialBlueprint()
    {
        // ==========================================================================================
        // 1. Domain & Spatial Partitioning Setup
        // ==========================================================================================
        m_blueprint.SetDomainSize( glm::vec3( 50.0f ), 2.0f );

        m_blueprint.ConfigureSpatialPartitioning()
            .SetMethod( DigitalTwin::SpatialPartitioningMethod::HashGrid )
            .SetCellSize( 3.0f )
            .SetMaxDensity( 64 )
            .SetComputeHz( 60.0f );

        // ==========================================================================================
        // 2. Environmental Fields (PDEs)
        // ==========================================================================================
        // "In Vitro" approach: Infinite, uniform oxygen supply everywhere in the domain.
        // This guarantees cells will not die of hypoxia for a very long time, allowing us
        // to clearly observe multiple generations of cell division.
        m_blueprint.AddGridField( "Oxygen" )
            .SetInitializer( DigitalTwin::GridInitializer::Gaussian( glm::vec3( 0.0f, 0.0f, 0.0f ), 10.0f, 80.0f ) )
            .SetDiffusionCoefficient( 5.0f )
            .SetDecayRate( 0.0f ) // No natural decay, only consumed by cells
            .SetComputeHz( 60.0f );

        // ==========================================================================================
        // 3. Agent Groups (Patient Zero)
        // ==========================================================================================
        // Start with a SINGLE cell exactly at the origin (0,0,0).
        std::vector<glm::vec4> patientZero = { glm::vec4( 0.0f, 0.0f, 0.0f, 1.0f ) };

        auto& tumorCells = m_blueprint.AddAgentGroup( "TumorCells" )
                               .SetCount( 1 )
                               .SetMorphology( DigitalTwin::MorphologyGenerator::CreateSphere( 1.5f ) )
                               .SetDistribution( patientZero )
                               .SetColor( glm::vec4( 0.2f, 0.8f, 0.3f, 1.0f ) ); // Let's make them bright green this time!

        // ==========================================================================================
        // 4. Behaviours (Physics & Biology)
        // ==========================================================================================

        // A. Brown motion for nice movement
        tumorCells.AddBehaviour( DigitalTwin::Behaviours::BrownianMotion{ 0.5f } ).SetHz( 60.0f );

        // B. Field Interaction: Slow oxygen consumption so the colony can grow large
        tumorCells.AddBehaviour( DigitalTwin::Behaviours::ConsumeField{ "Oxygen", 2.0f } ).SetHz( 60.0f );

        // B. Biomechanics (JKR Model): Softer cells with less adhesion
        // This allows daughter cells to visually "pop" and separate nicely after mitosis
        tumorCells
            .AddBehaviour( DigitalTwin::BiomechanicsGenerator::JKR()
                               .SetYoungsModulus( 20.0f ) // Softer tissue
                               .SetPoissonRatio( 0.4f )
                               .SetAdhesionEnergy( 1.5f ) // Reduced stickiness (easier separation)
                               .SetMaxInteractionRadius( 1.5f )
                               .Build() )
            .SetHz( 60.0f );

        // C. Biology (Cell Cycle): Slow, observable proliferation
        tumorCells
            .AddBehaviour( DigitalTwin::BiologyGenerator::StandardCellCycle()
                               .SetBaseDoublingTime( 5.0f/ 3600.0f ) // One division every 5 seconds!
                               .SetProliferationOxygenTarget( 50.0f )
                               .SetArrestPressureThreshold( 15.0f ) // High threshold: let them pack tightly before sleeping
                               .SetNecrosisOxygenThreshold( 5.0f )
                               .SetApoptosisRate( 0.0f )
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
} // namespace Gaudi