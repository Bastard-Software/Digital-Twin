#pragma once

#include "core/Core.h"
#include <string>

namespace DigitalTwin
{

    class DT_API DigitalTwin
    {
    public:
        DigitalTwin();
        ~DigitalTwin();

        void Print();

    private:
        std::string m_name;
    };

} // namespace DT