#include "Core/Base.hpp"
#include <spdlog/sinks/stdout_color_sinks.h>

namespace DigitalTwin
{

    Ref<spdlog::logger> Log::s_coreLogger   = nullptr;
    Ref<spdlog::logger> Log::s_clientLogger = nullptr;

    void Log::Init()
    {
        if( s_coreLogger != nullptr )
        {
            return;
        }

        spdlog::set_pattern( "%^[%T] %n: %v%$" );

        s_coreLogger   = spdlog::stdout_color_mt( "CORE" );
        s_clientLogger = spdlog::stdout_color_mt( "CLIENT" );

        s_coreLogger->set_level( spdlog::level::trace );
        s_clientLogger->set_level( spdlog::level::trace );

        DT_CORE_INFO( "Logging system initialized." );
    }

} // namespace DigitalTwin