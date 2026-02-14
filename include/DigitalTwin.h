#pragma once
#include "DigitalTwinTypes.h"

#include "core/Core.h"
#include "platform/Window.h"
#include <string>

namespace DigitalTwin
{

    class DT_API DigitalTwin
    {
    public:
        DigitalTwin();
        ~DigitalTwin();

        Result Initialize( const DigitalTwinConfig& config );
        void   Shutdown();

        const FrameContext& BeginFrame();
        void                EndFrame();
        void                Step(); // TODO: For future use

        bool IsWindowClosed();

        void  RenderUI( std::function<void()> uiCallback );
        void* GetImGuiTextureID( TextureHandle handle );
        void* GetImGuiContext();

    public:
        FileSystem* GetFileSystem() const;

    private:
        struct Impl;
        std::unique_ptr<Impl> m_impl;
    };

} // namespace DigitalTwin