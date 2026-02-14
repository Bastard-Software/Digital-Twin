#include <DigitalTwin.h>
#include <core/Log.h>
#include <imgui.h>
#include <iostream>
#include <platform/Input.h>

int main()
{

    DigitalTwin::DigitalTwin       engine;
    DigitalTwin::DigitalTwinConfig config;
    config.headless = false;
    engine.Initialize( config );
    DT_INFO( "Starting Editor..." );
    ImGui::SetCurrentContext( ( ImGuiContext* )engine.GetImGuiContext() );

    // Main Engine Loop
    while( !engine.IsWindowClosed() )
    {
        // 1. Poll events (input, window resize, etc.)
        const auto& ctx = engine.BeginFrame();

        // 2. Editor Logic Here
        engine.RenderUI( [ & ]() {
            ImGui::Begin( "Hello, Digital Twin!" );
            ImGui::Text( "This is a sample editor window." );
            ImGui::End();
        } );

        // 3. End Frame
        engine.EndFrame();
    }

    DT_INFO( "Editor closing..." );
    engine.Shutdown();

    return 0;
}