#include "Editor.h"

#include <core/Log.h>
#include <imgui.h>
#include <random>
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
        // 1. Define physical simulation domain( 200x200x200 micrometers )
            // With a voxel size of 2.0, this creates a 100x100x100 grid (1 million voxels)
            m_blueprint.SetDomainSize( glm::vec3( 200.0f ), 2.0f );

        // 2. Add Oxygen Field - Simulating an organic network of blood vessels
        std::vector<glm::vec3> bloodVessels;
        std::mt19937           rng( 42 ); // Fixed seed for reproducible organic shapes

        // Scatter 8 oxygen supply points around the edges of the simulation domain
        std::uniform_real_distribution<float> dist( -60.0f, 60.0f );
        for( int i = 0; i < 8; ++i )
        {
            bloodVessels.push_back( glm::vec3( dist( rng ), dist( rng ), dist( rng ) ) );
        }

        m_blueprint.AddGridField( "Oxygen" )
            .SetInitializer( DigitalTwin::GridInitializer::MultiGaussian( bloodVessels, 25.0f, 100.0f ) )
            .SetDiffusionCoefficient( 0.4f ) // Moderate diffusion allows steep gradients to form
            .SetDecayRate( 0.005f )          // Natural biological decay of oxygen in tissue
            .SetComputeHz( 120.0f );         // High frequency ensures stable PDE integration

        // 3. Add Tumor Cells - A massive, dense, and hungry tumor core
        // We use 8000 cells to create a highly visible and organic-looking mass
        auto& tumorCells = m_blueprint.AddAgentGroup( "TumorCells" )
                               .SetCount( 8000 )
                               .SetMorphology( DigitalTwin::MorphologyGenerator::CreateSphere( 1.5f ) )
                               .SetDistribution( DigitalTwin::SpatialDistribution::UniformInSphere( 8000, 25.0f ) )
                               .SetColor( glm::vec4( 0.1f, 0.8f, 0.2f, 1.0f ) ); // Distinct green color for contrast

        // 4. Attach Biological Behaviours

        // Brownian Motion: Cells jiggle, simulating random thermal motion and cell crawling
        tumorCells.AddBehaviour( DigitalTwin::Behaviours::BrownianMotion{ 0.3f } ).SetHz( 60.0f );

        // Consumption: The tumor aggressively eats oxygen.
        // With 8000 cells, this will quickly drain the center, creating a massive "hypoxic core".
        tumorCells.AddBehaviour( DigitalTwin::Behaviours::ConsumeField{ "Oxygen", 8.0f } ).SetHz( 60.0f );

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