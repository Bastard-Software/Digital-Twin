#pragma once
#include "DigitalTwinTypes.h"

#include "platform/Window.h"

#include "core/Core.h"
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

        WindowHandle CreateWindow( const std::string& title, uint32_t width, uint32_t height );
        bool         IsWindowColsed( WindowHandle handle ) const;

        void OnUpdate();

    public:
        FileSystem* GetFileSystem() const;

    private:
        struct Impl;
        std::unique_ptr<Impl> m_impl;
    };

} // namespace DigitalTwin