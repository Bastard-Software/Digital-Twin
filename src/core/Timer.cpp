#include "core/Timer.h"

#include <algorithm>

namespace DigitalTwin
{
    Timer::Timer()
    {
        Reset();
    }

    void Timer::Reset()
    {
        m_startTime = Clock::now();
        m_lastTime  = m_startTime;
        m_totalTime = 0.0f;
        m_deltaTime = 0.0f;
    }

    void Timer::Tick()
    {
        auto now = Clock::now();

        // Calculate duration in seconds
        std::chrono::duration<float> delta = now - m_lastTime;
        m_deltaTime                        = delta.count();

        m_lastTime = now;

        // Safety clamp: Prevent "spiral of death" if the app hangs or debugging pauses execution.
        // If we pause for 10 seconds, we don't want the next frame to think 10s passed instantly.
        if( m_deltaTime > 0.1f )
        {
            m_deltaTime = 0.1f;
        }

        m_totalTime += m_deltaTime;
    }

} // namespace DigitalTwin