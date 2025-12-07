#pragma once
#include "core/Base.hpp"
#include "platform/Window.hpp"

namespace DigitalTwin
{
    struct EngineConfig
    {
        bool_t   headless = true;
        uint32_t width    = 1280;
        uint32_t height   = 720;
    };

    class Engine
    {
    public:
        static void Init( const EngineConfig& config = EngineConfig() );
        static void Shutdown();

        // Main Loop Step
        static void Update();

        static bool_t  IsInitialized() { return s_initialized; }
        static bool_t  IsHeadless() { return s_headless; }
        static Window& GetWindow() { return *s_window; }

    private:
        static bool_t        s_initialized;
        static bool_t        s_headless;
        static Scope<Window> s_window;
    };
} // namespace DigitalTwin