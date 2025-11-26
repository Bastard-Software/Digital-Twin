#include "runtime/Engine.hpp"

namespace DigitalTwin
{

    bool_t Engine::s_initialized = false;
    bool_t Engine::s_headless    = true;

    void   Engine::Init( const EngineConfig& config )
    {
        DigitalTwin::Log::Init();

        if( s_initialized )
        {
            DT_CORE_WARN( "Engine already initialized!" );
            return;
        }
        s_headless    = config.headless;

        if( s_headless )
        {
            DT_CORE_INFO( "Mode: HEADLESS (Compute Only)" );
        }
        else
        {
            DT_CORE_INFO( "Mode: GRAPHICS" );
        }

        s_initialized = true;
    }

    void Engine::Shutdown()
    {
        if( !s_initialized )
        {
            DT_CORE_WARN( "Engine not initialized!" );
            return;
        }
        s_initialized = false;
        DT_CORE_INFO( "Engine shutdown." );
    }

}