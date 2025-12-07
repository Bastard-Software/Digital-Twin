#pragma once
#include "core/Base.hpp"
#include <utility>

namespace DigitalTwin
{
    class Input
    {
    public:
        // Use GLFW key codes (e.g. GLFW_KEY_SPACE)
        static bool IsKeyPressed( int keycode );

        // Use GLFW mouse codes (e.g. GLFW_MOUSE_BUTTON_LEFT)
        static bool IsMouseButtonPressed( int button );

        static std::pair<float, float> GetMousePosition();
        static float                   GetMouseX();
        static float                   GetMouseY();

    private:
        // Input needs access to the active window handle
        static void SetContext( void* windowHandle );

        friend class Window;
    };
} // namespace DigitalTwin