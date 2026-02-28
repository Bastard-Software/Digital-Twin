#include <DigitalTwin.h>
#include <core/Log.h>
#include <imgui.h>
#include <iostream>
#include <platform/Input.h>
#include <simulation/MorphologyGenerator.h>
#include <simulation/SpatialDistribution.h>

int main()
{
    DigitalTwin::DigitalTwin       engine;
    DigitalTwin::DigitalTwinConfig config;
    config.headless        = false;
    config.windowDesc.mode = DigitalTwin::WindowMode::FULLSCREEN_WINDOWED;
    engine.Initialize( config );

    // Setting up simulation
    DigitalTwin::SimulationBlueprint blueprint;
    blueprint.AddAgentGroup( "CancerCells" )
        .SetCount( 500 )
        .SetMorphology( DigitalTwin::MorphologyGenerator::CreateCube( 4.0f ) )
        .SetDistribution( DigitalTwin::SpatialDistribution::UniformInBox( 50, glm::vec3( 20.0f ) ) )
        .SetColor( glm::vec4( 0.9f, 0.1f, 0.1f, 1.0f ) ); // RED
    blueprint.AddAgentGroup( "T-Cells" )
        .SetCount( 5000 )
        .SetMorphology( DigitalTwin::MorphologyGenerator::CreateCube( 1.0f ) )
        .SetDistribution( DigitalTwin::SpatialDistribution::UniformInSphere( 500, 150.0f ) )
        .SetColor( glm::vec4( 0.2f, 0.8f, 0.3f, 1.0f ) ); // GREEN
    DT_INFO( "Building simulation from blueprint..." );
    engine.LoadSimulation( blueprint );

    // Main Engine Loop
    DT_INFO( "Starting Editor..." );
    ImGui::SetCurrentContext( ( ImGuiContext* )engine.GetImGuiContext() );
    while( !engine.IsWindowClosed() )
    {
        // 1. Poll events (input, window resize, etc.)
        const auto& ctx = engine.BeginFrame();

        // 2. Editor Logic Here
        engine.RenderUI( [ & ]() {
            ImGui::Begin( "Simulation Control" );
            ImGui::Text( "Total Agent Groups: %zu", blueprint.GetGroups().size() );
            ImGui::End();
        } );

        // 3. End Frame
        engine.EndFrame();
    }

    DT_INFO( "Editor closing..." );
    engine.Shutdown();

    return 0;
}