#pragma once
#include "core/Core.h"
#include <chrono>

namespace DigitalTwin
{
    /**
     * @brief A simple high-resolution timer.
     * Responsible strictly for measuring time deltas and total elapsed time.
     * Does not handle scheduling or frame limiting.
     */
    class DT_API Timer
    {
    public:
        Timer();
        ~Timer() = default;

        /**
         * @brief Resets the timer to zero.
         */
        void Reset();

        /**
         * @brief Updates the timer state. Call this once at the beginning of the frame.
         * Calculates the time elapsed since the last Tick().
         */
        void Tick();

        /**
         * @brief Returns the time in seconds between the last two Tick() calls.
         */
        float GetDeltaTime() const { return m_deltaTime; }

        /**
         * @brief Returns the total time in seconds since the timer was started/reset.
         */
        float GetTotalTime() const { return m_totalTime; }

    private:
        using Clock = std::chrono::high_resolution_clock;

        Clock::time_point m_startTime;
        Clock::time_point m_lastTime;

        float m_totalTime = 0.0f;
        float m_deltaTime = 0.0f;
    };

} // namespace DigitalTwin