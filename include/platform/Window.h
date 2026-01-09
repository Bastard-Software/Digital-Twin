#pragma once
#include "core/Core.h"
#include <string>

struct GLFWwindow;

namespace DigitalTwin
{
    // Forward declaration
    class Input;
    class PlatformSystem;

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

        // Raw pointer for RHI / ImGui
        void* GetNativeWindow() const { return m_window; }
        Input* GetInput() const { return m_inputSystem; }

        void DetachSystem() { m_platformSystem = nullptr; }

    protected:
        Window( const WindowDesc& config, Input* inputSystem, PlatformSystem* platformSystem );

        void Init( const WindowDesc& config );
        void Shutdown();

        friend class PlatformSystem;

    private:
        struct WindowData
        {
            std::string title;
            uint32_t    width, height;
            Input*      input = nullptr; // Pointer to Input system for callbacks
        };

        WindowData      m_data;
        GLFWwindow*     m_window         = nullptr;
        Input*          m_inputSystem    = nullptr;
        PlatformSystem* m_platformSystem = nullptr;
    };
} // namespace DigitalTwin