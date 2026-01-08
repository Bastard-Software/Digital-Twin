#include "platform/Input.h"

#include <GLFW/glfw3.h>

namespace DigitalTwin
{
    void Input::SetEventContext( void* windowHandle )
    {
        m_activeWindow = windowHandle;
    }

    bool Input::IsKeyPressed( Key keycode ) const
    {
        if( !m_activeWindow )
            return false;

        auto window = static_cast<GLFWwindow*>( m_activeWindow );
        auto state  = glfwGetKey( window, static_cast<int>( keycode ) );
        return state == GLFW_PRESS || state == GLFW_REPEAT;
    }

    bool Input::IsMouseButtonPressed( Mouse button ) const
    {
        if( !m_activeWindow )
            return false;

        auto window = static_cast<GLFWwindow*>( m_activeWindow );
        auto state  = glfwGetMouseButton( window, static_cast<int>( button ) );
        return state == GLFW_PRESS;
    }

    std::pair<float, float> Input::GetMousePosition() const
    {
        if( !m_activeWindow )
            return { 0.0f, 0.0f };

        auto   window = static_cast<GLFWwindow*>( m_activeWindow );
        double x, y;
        glfwGetCursorPos( window, &x, &y );
        return { ( float )x, ( float )y };
    }

    float Input::GetMouseX() const
    {
        return GetMousePosition().first;
    }

    float Input::GetMouseY() const
    {
        return GetMousePosition().second;
    }

    float Input::GetScrollY() const
    {
        return m_scrollY;
    }

    void Input::SetScrollY( float yOffset )
    {
        m_scrollY = yOffset;
    }

    void Input::ResetScroll()
    {
        m_scrollY = 0.0f;
    }
} // namespace DigitalTwin