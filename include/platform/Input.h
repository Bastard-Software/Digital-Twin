#pragma once
#include "core/Core.h"
#include "platform/KeyCodes.h"
#include "platform/MouseCodes.h"
#include <utility>

namespace DigitalTwin
{
    class DT_API Input
    {
    public:
        // Checks if a key is currently pressed (polling)
        bool IsKeyPressed( Key keycode ) const;

        // Checks if a mouse button is currently pressed
        bool IsMouseButtonPressed( Mouse button ) const;

        // Returns the mouse cursor position relative to the client area
        std::pair<float, float> GetMousePosition() const;
        float                   GetMouseX() const;
        float                   GetMouseY() const;

        // Returns accumulated scroll offset for the current frame
        float GetScrollY() const;

        // --- Internal State Management (Called by Platform/Window) ---

        // Sets the accumulation variable for scroll (called by GLFW callback)
        void SetScrollY( float yOffset );

        // Resets frame-based inputs (like scroll delta). Call this start of frame.
        void ResetScroll();

        // Sets the GLFW window context for polling input
        void SetEventContext( void* windowHandle );

    private:
        void* m_activeWindow = nullptr; // Raw pointer to GLFWwindow
        float m_scrollY      = 0.0f;
    };
} // namespace DigitalTwin