#include "platform/Input.hpp"

#include <GLFW/glfw3.h>

namespace DigitalTwin
{
    static GLFWwindow* s_ActiveWindow = nullptr;

    void Input::SetContext( void* windowHandle )
    {
        s_ActiveWindow = static_cast<GLFWwindow*>( windowHandle );
    }

    bool Input::IsKeyPressed( int keycode )
    {
        if( !s_ActiveWindow )
            return false;
        auto state = glfwGetKey( s_ActiveWindow, keycode );
        return state == GLFW_PRESS || state == GLFW_REPEAT;
    }

    bool Input::IsMouseButtonPressed( int button )
    {
        if( !s_ActiveWindow )
            return false;
        auto state = glfwGetMouseButton( s_ActiveWindow, button );
        return state == GLFW_PRESS;
    }

    std::pair<float, float> Input::GetMousePosition()
    {
        if( !s_ActiveWindow )
            return { 0.0f, 0.0f };
        double x, y;
        glfwGetCursorPos( s_ActiveWindow, &x, &y );
        return { ( float )x, ( float )y };
    }

    float Input::GetMouseX()
    {
        return GetMousePosition().first;
    }

    float Input::GetMouseY()
    {
        return GetMousePosition().second;
    }
} // namespace DigitalTwin