#include "platform/PlatformSystem.h"

#include "core/Log.h"
#include <GLFW/glfw3.h>
#include <algorithm> // For std::find

namespace DigitalTwin
{
    // Static error callback for GLFW
    static void GLFWErrorCallback( int error, const char* description )
    {
        DT_ERROR( "GLFW Error ({0}): {1}", error, description );
    }

    PlatformSystem::PlatformSystem()
    {
        // Initialize the Input system instance
        m_input = std::make_unique<Input>();
    }

    PlatformSystem::~PlatformSystem()
    {
        Shutdown();
    }

    Result PlatformSystem::Initialize()
    {
        if( m_initialized )
            return Result::SUCCESS;

        DT_INFO( "Initializing Platform System..." );

        if( !glfwInit() )
        {
            DT_ERROR( "Failed to initialize GLFW!" );
            return Result::FAIL;
        }

        glfwSetErrorCallback( GLFWErrorCallback );

        // Hint to GLFW that we are using Vulkan (no OpenGL context)
        // This applies to all subsequently created windows.
        glfwWindowHint( GLFW_CLIENT_API, GLFW_NO_API );

        m_initialized = true;
        return Result::SUCCESS;
    }

    void PlatformSystem::Shutdown()
    {
        if( !m_initialized )
            return;

        DT_INFO( "Shutting down Platform System..." );

        // SAFETY CHECK:
        // If there are still active windows tracked by the system, we must detach them.
        // This ensures that when the user's unique_ptr<Window> eventually goes out of scope,
        // the Window destructor won't try to call RemoveWindow() on this dead PlatformSystem.
        if( !m_windows.empty() )
        {
            DT_WARN( "PlatformSystem shutdown detecting {0} active windows. Detaching logic...", m_windows.size() );

            for( Window* window: m_windows )
            {
                // Notify the window that the system is gone.
                // The window will no longer try to unregister itself in its destructor.
                window->DetachSystem();
            }
            m_windows.clear();
        }

        // Terminate GLFW
        glfwTerminate();

        m_initialized = false;
    }

    Scope<Window> PlatformSystem::CreateWindow( const WindowDesc& config )
    {
        if( !m_initialized )
        {
            DT_ERROR( "Cannot create window: PlatformSystem is not initialized." );
            return nullptr;
        }

        // Create the window instance.
        // We use 'new' because the Window constructor requires access to 'this' (PlatformSystem*),
        // and might be private/protected due to friend class usage.
        Window* rawWindow = new Window( config, m_input.get(), this );

        // Track the window in our internal vector
        m_windows.push_back( rawWindow );

        // Return ownership to the user via unique_ptr
        return Scope<Window>( rawWindow );
    }

    void PlatformSystem::RemoveWindow( Window* window )
    {
        // This is called by ~Window().
        // We need to find the window pointer and remove it from the tracking list.

        auto it = std::find( m_windows.begin(), m_windows.end(), window );
        if( it != m_windows.end() )
        {
            // Efficient removal: swap with the last element and pop back
            // (Order of windows in the vector does not matter)
            if( it != m_windows.end() - 1 )
            {
                std::iter_swap( it, m_windows.end() - 1 );
            }
            m_windows.pop_back();
        }
    }

    void PlatformSystem::OnUpdate()
    {
        // Reset per-frame input states (like scroll delta)
        if( m_input )
        {
            m_input->ResetScroll();
        }

        // Poll for window events (keyboard, mouse, close requests)
        glfwPollEvents();
    }

    std::vector<const char*> PlatformSystem::GetRequiredVulkanExtensions() const
    {
        uint32_t     glfwExtensionCount = 0;
        const char** glfwExtensions     = glfwGetRequiredInstanceExtensions( &glfwExtensionCount );

        if( !glfwExtensions )
            return {};

        return std::vector<const char*>( glfwExtensions, glfwExtensions + glfwExtensionCount );
    }

} // namespace DigitalTwin