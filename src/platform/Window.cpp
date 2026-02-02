#include "platform/Window.h"

#include "core/Log.h"
#include "platform/Input.h"
#include "platform/PlatformSystem.h"
#include <GLFW/glfw3.h>

namespace DigitalTwin
{
    Window::Window( const WindowDesc& config, Input* inputSystem, PlatformSystem* platformSystem )
        : m_inputSystem( inputSystem )
        , m_platformSystem( platformSystem )
    {
        Init( config );
    }

    Window::~Window()
    {
        if( m_platformSystem )
        {
            m_platformSystem->RemoveWindow( this );
        }

        Shutdown();
    }

    void Window::Init( const WindowDesc& config )
    {
        m_data.title  = config.title;
        m_data.width  = config.width;
        m_data.height = config.height;
        m_data.input  = m_inputSystem; // Store input pointer in data for callbacks

        DT_INFO( "Creating window {0} ({1}x{2})", config.title, config.width, config.height );

        // Note: glfwInit is now handled by PlatformSystem, not here.

        glfwWindowHint( GLFW_CLIENT_API, GLFW_NO_API );
        glfwWindowHint( GLFW_RESIZABLE, GLFW_TRUE );

        m_window = glfwCreateWindow( ( int )m_data.width, ( int )m_data.height, m_data.title.c_str(), nullptr, nullptr );
        DT_CORE_ASSERT( m_window, "Could not create GLFW window!" );

        glfwSetWindowUserPointer( m_window, &m_data );

        // Update Input context to this new window
        if( m_inputSystem )
        {
            m_inputSystem->SetEventContext( m_window );
        }

        // --- Callbacks ---
        glfwSetFramebufferSizeCallback( m_window, []( GLFWwindow* window, int width, int height ) {
            WindowData& data = *( WindowData* )glfwGetWindowUserPointer( window );
            data.width       = width;
            data.height      = height;
            data.wasResized  = true; // Mark as dirty
        } );

        glfwSetScrollCallback( m_window, []( GLFWwindow* window, double xoffset, double yoffset ) {
            WindowData& data = *( WindowData* )glfwGetWindowUserPointer( window );
            if( data.input )
            {
                data.input->SetScrollY( ( float )yoffset );
            }
        } );
    }

    void Window::Shutdown()
    {
        if( m_window )
        {
            glfwDestroyWindow( m_window );
            m_window = nullptr;
        }
    }

    VkSurfaceKHR Window::CreateSurface( VkInstance instance )
    {
        VkSurfaceKHR surface = VK_NULL_HANDLE;
        if( glfwCreateWindowSurface( instance, m_window, nullptr, &surface ) != VK_SUCCESS )
        {
            DT_ERROR( "Failed to create window surface!" );
            return VK_NULL_HANDLE;
        }
        return surface;
    }

    void Window::Show()
    {
        if( m_window )
            glfwShowWindow( m_window );
    }

    bool Window::IsClosed() const
    {
        return glfwWindowShouldClose( m_window );
    }

    void Window::GetFramebufferSize( uint32_t& width, uint32_t& height ) const
    {
        int w, h;
        glfwGetFramebufferSize( m_window, &w, &h );
        width  = static_cast<uint32_t>( w );
        height = static_cast<uint32_t>( h );
    }

} // namespace DigitalTwin