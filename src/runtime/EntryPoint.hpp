#pragma once
#include "core/Base.hpp"
#include "runtime/Application.hpp"

// External declaration: The user must define this function in their client app (Sandbox)
extern DigitalTwin::Application* DigitalTwin::CreateApplication();

int main( int argc, char** argv )
{
    auto app = DigitalTwin::CreateApplication();
    if( app )
    {
        try
        {
            app->Run();
        }
        catch( const std::exception& e )
        {
            DT_CORE_CRITICAL( "Application crashed: {}", e.what() );
            delete app;
            return -1;
        }

        delete app;
    }

    return 0;
}