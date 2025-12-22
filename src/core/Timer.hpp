#pragma once
#include <chrono>

namespace DigitalTwin
{

    class Timer
    {
    public:
        Timer() { Reset(); }

        void Reset() { m_start = std::chrono::high_resolution_clock::now(); }

        // Returns elapsed time in seconds
        float Elapsed()
        {
            return std::chrono::duration_cast<std::chrono::nanoseconds>( std::chrono::high_resolution_clock::now() - m_start ).count() * 1e-9f;
        }

        // Returns elapsed time in milliseconds
        float ElapsedMillis() { return Elapsed() * 1000.0f; }

    private:
        std::chrono::time_point<std::chrono::high_resolution_clock> m_start;
    };
} // namespace DigitalTwin