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
        // 1. Define physical simulation domain (200x200x200 micrometers)
        m_blueprint.SetDomainSize( glm::vec3( 200.0f ), 2.0f );

        m_blueprint.ConfigureSpatialPartitioning()
            .SetMethod( DigitalTwin::SpatialPartitioningMethod::HashGrid )
            .SetCellSize( 3.0f )    // Matches 2x max interaction radius (1.5 * 2)
            .SetMaxDensity( 64 )    // Prevents buffer overflows in dense regions (For future use)
            .SetComputeHz( 30.0f ); // Spatial grid building runs at 120Hz for stability

        // 2A. Add Oxygen Field (O2) - Blood vessels network
        std::vector<glm::vec3> bloodVessels;
        std::mt19937           rng( 42 );

        // Tumor starts with a radius of 10.0f, so vessels spanning [-20.0, 20.0]
        // will intimately wrap around and penetrate the initial cell mass.
        std::uniform_real_distribution<float> dist( -20.0f, 20.0f );
        for( int i = 0; i < 8; ++i )
        {
            bloodVessels.push_back( glm::vec3( dist( rng ), dist( rng ), dist( rng ) ) );
        }

        // Decreased Gaussian sigma (from 15.0f to 10.0f) to make the initial oxygen clouds sharper
        m_blueprint.AddGridField( "Oxygen" )
            .SetInitializer( DigitalTwin::GridInitializer::MultiGaussian( bloodVessels, 10.0f, 100.0f ) )
            .SetDiffusionCoefficient( 2.5f )
            .SetDecayRate( 0.01f )
            .SetComputeHz( 60.0f );

        // 2B. Add Lactate Field (Waste product)
        m_blueprint.AddGridField( "Lactate" )
            .SetInitializer( DigitalTwin::GridInitializer::Constant( 0.0f ) )
            .SetDiffusionCoefficient( 1.2f )
            .SetDecayRate( 0.05f ) // Cleared by vasculature over time
            .SetComputeHz( 60.0f );

        // 3. Create initial tumor mass (Dense sphere of agents)
        auto& tumorCells = m_blueprint.AddAgentGroup( "TumorCells" )
                               .SetCount( 8000 )
                               .SetMorphology( DigitalTwin::MorphologyGenerator::CreateSphere( 1.5f ) )
                               .SetDistribution( DigitalTwin::SpatialDistribution::UniformInSphere( 8000, 10.0f ) )
                               .SetColor( glm::vec4( 0.1f, 0.8f, 0.2f, 1.0f ) );

        // 4. Attach Behaviours

        // Consume Oxygen from the "Oxygen" grid
        tumorCells.AddBehaviour( DigitalTwin::Behaviours::ConsumeField{ "Oxygen", 8.0f } ).SetHz( 60.0f );

        // Secrete Acid into the "Lactate" grid
        tumorCells.AddBehaviour( DigitalTwin::Behaviours::SecreteField{ "Lactate", 15.0f } ).SetHz( 60.0f );

        // Simulate mechanical forces using the scientific JKR Builder
        tumorCells
            .AddBehaviour( DigitalTwin::BiomechanicsGenerator::JKR()
                               .SetYoungsModulus( 25.0f )       // Stiffness in kPa
                               .SetPoissonRatio( 0.4f )         // Compressibility
                               .SetAdhesionEnergy( 2.0f )       // Surface adhesion
                               .SetMaxInteractionRadius( 1.5f ) // Maximum cell radius
                               .Build() )
            .SetHz( 120.0f );

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