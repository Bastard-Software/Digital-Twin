#pragma once

#include <memory>
#include <spdlog/fmt/ostr.h>
#include <spdlog/spdlog.h>

namespace DigitalTwin
{
    class Log
    {
    public:
        static void Init();

        inline static std::shared_ptr<spdlog::logger>& GetCoreLogger() { return s_coreLogger; }
        inline static std::shared_ptr<spdlog::logger>& GetClientLogger() { return s_clientLogger; }

    private:
        static std::shared_ptr<spdlog::logger> s_coreLogger;
        static std::shared_ptr<spdlog::logger> s_clientLogger;
    };
} // namespace DigitalTwin

#define DT_CORE_TRACE( ... )    ::DigitalTwin::Log::GetCoreLogger()->trace( __VA_ARGS__ )
#define DT_CORE_INFO( ... )     ::DigitalTwin::Log::GetCoreLogger()->info( __VA_ARGS__ )
#define DT_CORE_WARN( ... )     ::DigitalTwin::Log::GetCoreLogger()->warn( __VA_ARGS__ )
#define DT_CORE_ERROR( ... )    ::DigitalTwin::Log::GetCoreLogger()->error( __VA_ARGS__ )
#define DT_CORE_CRITICAL( ... ) ::DigitalTwin::Log::GetCoreLogger()->critical( __VA_ARGS__ )

#define DT_TRACE( ... )    ::DigitalTwin::Log::GetClientLogger()->trace( __VA_ARGS__ )
#define DT_INFO( ... )     ::DigitalTwin::Log::GetClientLogger()->info( __VA_ARGS__ )
#define DT_WARN( ... )     ::DigitalTwin::Log::GetClientLogger()->warn( __VA_ARGS__ )
#define DT_ERROR( ... )    ::DigitalTwin::Log::GetClientLogger()->error( __VA_ARGS__ )
#define DT_CRITICAL( ... ) ::DigitalTwin::Log::GetClientLogger()->critical( __VA_ARGS__ )