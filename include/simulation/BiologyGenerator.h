#pragma once
#include "simulation/Behaviours.h"

namespace DigitalTwin
{
    /**
     * @brief Factory class to generate pre-calculated GPU biological parameters
     * from real-world scientific units (hours, days, mmHg, MPa).
     */
    class BiologyGenerator
    {
    public:
        class StandardCellCycleBuilder
        {
        public:
            StandardCellCycleBuilder& SetBaseDoublingTime( float hours )
            {
                m_doublingTimeHours = hours;
                return *this;
            }
            StandardCellCycleBuilder& SetProliferationOxygenTarget( float o2_mmHg )
            {
                m_prolifO2 = o2_mmHg;
                return *this;
            }
            StandardCellCycleBuilder& SetArrestPressureThreshold( float pressure_MPa )
            {
                m_arrestPressure = pressure_MPa;
                return *this;
            }
            StandardCellCycleBuilder& SetNecrosisOxygenThreshold( float o2_mmHg )
            {
                m_necrosisO2 = o2_mmHg;
                return *this;
            }
            StandardCellCycleBuilder& SetApoptosisRate( float rate_per_day )
            {
                m_apoptosisRate = rate_per_day;
                return *this;
            }

            Behaviours::CellCycle Build() const
            {
                Behaviours::CellCycle cycle;

                // Convert Doubling Time (Hours) to a per-second growth rate.
                // Assuming biomass needs to go from 0.5 to 1.0 to trigger division.
                float totalSeconds     = m_doublingTimeHours * 3600.0f;
                cycle.growthRatePerSec = 1.0f / totalSeconds;

                // Convert Apoptosis rate (Per Day) to probability per second
                float secondsInDay        = 24.0f * 3600.0f;
                cycle.apoptosisProbPerSec = m_apoptosisRate / secondsInDay;

                // Direct mappings
                cycle.targetO2       = m_prolifO2;
                cycle.arrestPressure = m_arrestPressure;
                cycle.necrosisO2     = m_necrosisO2;

                return cycle;
            }

        private:
            float m_doublingTimeHours = 18.0f; // Hours to double volume
            float m_prolifO2          = 38.0f; // mmHg
            float m_arrestPressure    = 2.5f;  // MPa
            float m_necrosisO2        = 5.0f;  // mmHg
            float m_apoptosisRate     = 0.05f; // 5% chance per day
        };

        static StandardCellCycleBuilder StandardCellCycle() { return StandardCellCycleBuilder(); }
    };

} // namespace DigitalTwin