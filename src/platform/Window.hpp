#pragma once
#include "core/Base.hpp"
#include <string>

struct GLFWwindow;

namespace DigitalTwin
{
    struct WindowConfig
    {
        std::string title  = "Digital Twin Simulation";
        uint32_t    width  = 1280;
        uint32_t    height = 720;
        bool        vsync  = true;
    };

    class Window
    {
    public:
        Window( const WindowConfig& config );
        ~Window();

        void OnUpdate();

        uint32_t GetWidth() const { return m_data.width; }
        uint32_t GetHeight() const { return m_data.height; }

        // Returns true if the user requested to close the window (e.g. clicked X)
        bool IsClosed() const;

        // Raw pointer for RHI / ImGui
        void* GetNativeWindow() const { return m_window; }

    private:
        void Init( const WindowConfig& config );
        void Shutdown();

    private:
        struct WindowData
        {
            std::string title;
            uint32_t    width, height;
            bool        vsync;
        };

        WindowData  m_data;
        GLFWwindow* m_window = nullptr;
    };
} // namespace DigitalTwin