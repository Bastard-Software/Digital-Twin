#pragma once
#include "platform/Input.h"
#include "platform/Window.h"
#include <memory>
#include <vector>

namespace DigitalTwin
{
    class PlatformSystem
    {
    public:
        PlatformSystem();
        ~PlatformSystem();

        Result Initialize();
        void   Shutdown();

        // Creates a new window managed by the platform.
        // Injects the Input system into the window automatically.
        std::unique_ptr<Window> CreateWindow( const WindowDesc& config );

        // Polls system events (keyboard, mouse, window). Call this once per frame.
        void OnUpdate();

        void RemoveWindow( Window* window );

        // Helper to get Vulkan extensions required by the surface
        std::vector<const char*> GetRequiredVulkanExtensions() const;

        // Returns the Input system owned by the platform
        Input* GetInput() { return m_input.get(); }

    private:
        std::vector<Window*>   m_windows;
        std::unique_ptr<Input> m_input;
        bool_t                 m_initialized = false;
    };
} // namespace DigitalTwin