#pragma once

#include "core/Core.h"
#include <memory>
#include <spdlog/fmt/ostr.h>
#include <spdlog/spdlog.h>

namespace DigitalTwin
{

    class DT_API Log
    {
    public:
        static void Init();

        inline static std::shared_ptr<spdlog::logger>& GetCoreLogger() { return s_CoreLogger; }
        inline static std::shared_ptr<spdlog::logger>& GetClientLogger() { return s_ClientLogger; }

    private:
        static std::shared_ptr<spdlog::logger> s_CoreLogger;
        static std::shared_ptr<spdlog::logger> s_ClientLogger;
    };

} // namespace DigitalTwin

// Automatically detect if we are inside the Engine (DLL) or Client (App)
#ifdef DT_BUILD_DLL
  // Core Logging
#    define DT_TRACE( ... )    ::DigitalTwin::Log::GetCoreLogger()->trace( __VA_ARGS__ )
#    define DT_INFO( ... )     ::DigitalTwin::Log::GetCoreLogger()->info( __VA_ARGS__ )
#    define DT_WARN( ... )     ::DigitalTwin::Log::GetCoreLogger()->warn( __VA_ARGS__ )
#    define DT_ERROR( ... )    ::DigitalTwin::Log::GetCoreLogger()->error( __VA_ARGS__ )
#    define DT_CRITICAL( ... ) ::DigitalTwin::Log::GetCoreLogger()->critical( __VA_ARGS__ )
#else
  // Client Logging
#    define DT_TRACE( ... )    ::DigitalTwin::Log::GetClientLogger()->trace( __VA_ARGS__ )
#    define DT_INFO( ... )     ::DigitalTwin::Log::GetClientLogger()->info( __VA_ARGS__ )
#    define DT_WARN( ... )     ::DigitalTwin::Log::GetClientLogger()->warn( __VA_ARGS__ )
#    define DT_ERROR( ... )    ::DigitalTwin::Log::GetClientLogger()->error( __VA_ARGS__ )
#    define DT_CRITICAL( ... ) ::DigitalTwin::Log::GetClientLogger()->critical( __VA_ARGS__ )
#endif