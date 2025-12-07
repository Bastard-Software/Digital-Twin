#include "runtime/Engine.hpp"

#include "rhi/RHI.hpp"

namespace DigitalTwin
{
    bool_t        Engine::s_initialized = false;
    bool_t        Engine::s_headless    = true;
    Scope<Window> Engine::s_window      = nullptr;

    void Engine::Init( const EngineConfig& config )
    {
        DigitalTwin::Log::Init();

        if( s_initialized )
        {
            DT_CORE_WARN( "Engine already initialized!" );
            return;
        }
        s_headless = config.headless;

        // 1. Create Window (if not headless)
        if( !s_headless )
        {
            DT_CORE_INFO( "Mode: GRAPHICS (Windowed)" );
            WindowConfig winConfig;
            winConfig.width  = config.width;
            winConfig.height = config.height;
            s_window         = CreateScope<Window>( winConfig );
        }
        else
        {
            DT_CORE_INFO( "Mode: HEADLESS (Compute Only)" );
        }

        // 2. Initialize RHI
        RHIConfig rhiConfig;
        rhiConfig.headless         = s_headless;
        rhiConfig.enableValidation = true; // Debug default
        RHI::Init( rhiConfig );

        // 3. Create Default Device (Adapter 0)
        // Note: In the future, we might want to select the best adapter based on capabilities
        if( RHI::GetAdapterCount() > 0 )
        {
            // RHI::CreateDevice(0); // Optional: Engine usually manages the device lifetime
        }

        s_initialized = true;
    }

    void Engine::Shutdown()
    {
        if( !s_initialized )
            return;

        // RHI must be shutdown BEFORE the window, because Surface depends on Window
        RHI::Shutdown();

        // Destroy Window
        s_window.reset();

        s_initialized = false;
        DT_CORE_INFO( "Engine shutdown." );
    }

    void Engine::Update()
    {
        if( s_window )
        {
            s_window->OnUpdate();
        }
    }
} // namespace DigitalTwin