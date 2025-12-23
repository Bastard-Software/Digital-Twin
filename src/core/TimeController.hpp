#pragma once
#include <algorithm>
#include <cstdint>

namespace DigitalTwin
{
    /**
     * @brief Manages the conversion between Real Time (Application) and Simulation Time.
     * Handles time scaling, pausing, and frame counting.
     */
    class TimeController
    {
    public:
        /**
         * @brief Sets the speed multiplier for the simulation.
         * 1.0 = Real-time, 2.0 = 2x speed, 0.0 = Paused.
         */
        void  SetTimeScale( float scale ) { m_timeScale = scale; }
        float GetTimeScale() const { return m_timeScale; }

        /**
         * @brief Returns the total accumulated simulation time (in seconds).
         */
        float GetSimTime() const { return m_simTime; }

        /**
         * @brief Returns the index of the current simulation frame.
         */
        uint32_t GetFrameIndex() const { return m_frameIndex; }

        /**
         * @brief Returns the simulation delta time for the current frame.
         * Calculated as: RealDeltaTime * TimeScale.
         */
        float GetSimDeltaTime() const { return m_simDeltaTime; }

        /**
         * @brief Updates the internal timers based on the application's real delta time.
         * @param realDt The time elapsed since the last CPU frame (in seconds).
         */
        void Update( float realDt )
        {
            // Safety clamp to prevent "spiral of death" during debugging or massive lag spikes.
            // If the frame took longer than 100ms, we cap it to 100ms.
            if( realDt > 0.1f )
                realDt = 0.1f;

            m_realDeltaTime = realDt;
            m_simDeltaTime  = realDt * m_timeScale;

            m_simTime += m_simDeltaTime;
            m_frameIndex++;
        }

        bool IsPaused() const { return m_timeScale == 0.0f; }

    private:
        float    m_timeScale     = 1.0f;
        float    m_simTime       = 0.0f;
        float    m_realDeltaTime = 0.0f;
        float    m_simDeltaTime  = 0.0f;
        uint32_t m_frameIndex    = 0;
    };
} // namespace DigitalTwin