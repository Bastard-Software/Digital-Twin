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
        // 1. Define physical simulation domain (100x100x100 voxels)
        m_blueprint.SetDomainSize( glm::vec3( 200.0f ), 2.0f );

        // 2. Add Oxygen Field - random points representing "blood vessels" or oxygen sources
        std::vector<glm::vec3>                bloodVessels;
        std::mt19937                          rng( 49 );             // Fixed seed for reproducible results during MVP testing
        std::uniform_real_distribution<float> dist( -40.0f, 40.0f ); // Scatter around the center
        for( int i = 0; i < 6; ++i )                                 // Generate 6 sources
        {
            bloodVessels.push_back( glm::vec3( dist( rng ), dist( rng ), dist( rng ) ) );
        }

        m_blueprint.AddGridField( "Oxygen" )
            .SetInitializer( DigitalTwin::GridInitializer::MultiGaussian( bloodVessels, 10.0f, 100.0f ) ) // Multigaussian
            .SetDiffusionCoefficient( 2.0f )                                                              // Moderate diffusion
            .SetDecayRate( 0.001f )                                                                       // Natural background consumption
            .SetComputeHz( 120.0f );                                                                      // High frequency for PDE stability

        // 3. Add Tumor Cells
        m_blueprint.AddAgentGroup( "TumorCells" )
            .SetCount( 50 )
            .SetMorphology( DigitalTwin::MorphologyGenerator::CreateSphere( 1.0f ) )
            .SetDistribution( DigitalTwin::SpatialDistribution::UniformInSphere( 50, 10.0f ) )
            .SetColor( glm::vec4( 0.1f, 0.8f, 0.2f, 1.0f ) )
            .AddBehaviour( DigitalTwin::Behaviours::BrownianMotion{ 0.1f } )
            .SetHz( 30.0f );

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