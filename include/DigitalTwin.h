#pragma once
#include "DigitalTwinTypes.h"

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

        void Print();

    private:
        struct Impl;
        std::unique_ptr<Impl> m_impl;
    };

} // namespace DigitalTwin