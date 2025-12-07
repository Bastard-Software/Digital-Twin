#include "platform/Window.hpp"

#include "platform/Input.hpp"
#include <GLFW/glfw3.h>

namespace DigitalTwin
{
    static bool s_GLFWInitialized = false;

    static void GLFWErrorCallback( int error, const char* description )
    {
        DT_CORE_ERROR( "GLFW Error ({0}): {1}", error, description );
    }

    Window::Window( const WindowConfig& config )
    {
        Init( config );
    }

    Window::~Window()
    {
        Shutdown();
    }

    void Window::Init( const WindowConfig& config )
    {
        m_data.title  = config.title;
        m_data.width  = config.width;
        m_data.height = config.height;
        m_data.vsync  = config.vsync;

        DT_CORE_INFO( "Creating window {0} ({1}x{2})", config.title, config.width, config.height );

        if( !s_GLFWInitialized )
        {
            int success = glfwInit();
            DT_CORE_ASSERT( success, "Could not initialize GLFW!" );
            glfwSetErrorCallback( GLFWErrorCallback );
            s_GLFWInitialized = true;
        }

        // Tell GLFW we are using Vulkan (no OpenGL context)
        glfwWindowHint( GLFW_CLIENT_API, GLFW_NO_API );
        glfwWindowHint( GLFW_RESIZABLE, GLFW_TRUE );

        m_window = glfwCreateWindow( ( int )m_data.width, ( int )m_data.height, m_data.title.c_str(), nullptr, nullptr );
        DT_CORE_ASSERT( m_window, "Could not create GLFW window!" );

        glfwSetWindowUserPointer( m_window, &m_data );

        // Initialize Input context
        Input::SetContext( m_window );

        // Setup basic callbacks
        glfwSetWindowSizeCallback( m_window, []( GLFWwindow* window, int width, int height ) {
            WindowData& data = *( WindowData* )glfwGetWindowUserPointer( window );
            data.width       = width;
            data.height      = height;
        } );
    }

    void Window::Shutdown()
    {
        if( m_window )
        {
            glfwDestroyWindow( m_window );
            m_window = nullptr;
        }

        // Note: We don't terminate GLFW here because there might be other windows
        // or we might want to restart. Termination usually happens at app exit.
    }

    void Window::OnUpdate()
    {
        glfwPollEvents();
    }

    bool Window::IsClosed() const
    {
        return glfwWindowShouldClose( m_window );
    }
} // namespace DigitalTwin