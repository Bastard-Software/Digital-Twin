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

        static float GetScrollY();
        static void  ResetScroll(); // Call this at start of every frame!

        // Internal setter for GLFW callback
        static void SetScrollY( float yOffset );

    private:
        // Input needs access to the active window handle
        static void SetContext( void* windowHandle );

        friend class Window;
        static float s_ScrollY; // Accumulator
    };
} // namespace DigitalTwin