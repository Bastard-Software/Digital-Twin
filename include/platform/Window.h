#pragma once
#include "core/Core.h"
#include <string>
#include <volk.h>

struct GLFWwindow;

namespace DigitalTwin
{
    // Forward declaration
    class Input;
    class PlatformSystem;
    class Swapchain;

    struct WindowDesc
    {
        std::string title  = "Digital Twin Simulation";
        uint32_t    width  = 1280;
        uint32_t    height = 720;
    };

    class DT_API Window
    {
    public:
        ~Window();

        void Show();

        uint32_t GetWidth() const { return m_data.width; }
        uint32_t GetHeight() const { return m_data.height; }

        // Returns true if the user requested to close the window
        bool IsClosed() const;

        // Returns the actual framebuffer size (pixels)
        void GetFramebufferSize( uint32_t& width, uint32_t& height ) const;

        // Checks if the window was resized since the last reset
        bool WasResized() const { return m_data.wasResized; }

        // Acknowledges the resize event
        void ResetResizeFlag() { m_data.wasResized = false; }

        bool IsMinimized() const { return m_data.width == 0 || m_data.height == 0; }

        // Raw pointer for RHI / ImGui
        void*  GetNativeWindow() const { return m_window; }
        Input* GetInput() const { return m_inputSystem; }

        void DetachSystem() { m_platformSystem = nullptr; }

    protected:
        Window( const WindowDesc& config, Input* inputSystem, PlatformSystem* platformSystem );

        void Init( const WindowDesc& config );
        void Shutdown();

        VkSurfaceKHR CreateSurface( VkInstance instance );

        friend class PlatformSystem;
        friend class Swapchain;

    private:
        struct WindowData
        {
            std::string title;
            uint32_t    width, height;
            bool        wasResized = false;
            Input*      input      = nullptr; // Pointer to Input system for callbacks
        };

        WindowData      m_data;
        GLFWwindow*     m_window         = nullptr;
        Input*          m_inputSystem    = nullptr;
        PlatformSystem* m_platformSystem = nullptr;
    };
} // namespace DigitalTwin