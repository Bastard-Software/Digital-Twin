#pragma once

#include "core/Base.hpp"

namespace DigitalTwin
{

    struct EngineConfig
    {
        bool_t headless = true;
    };

    class Engine
    {
    public:
        static void Init( const EngineConfig& config = EngineConfig() );
        static void Shutdown();

        static bool_t IsInitialized() { return s_initialized; }
        static bool_t IsHeadless() { return s_headless; }

    private:
        static bool_t s_initialized;
        static bool_t s_headless;
    };

} // namespace DigitalTwin