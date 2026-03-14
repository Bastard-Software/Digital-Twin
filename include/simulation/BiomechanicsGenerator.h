#pragma once
#include "simulation/Behaviours.h"

namespace DigitalTwin
{
    /**
     * @brief Factory class to generate pre-calculated GPU physics parameters
     * from real-world biological and physical units.
     */
    class BiomechanicsGenerator
    {
    public:
        class JKRBuilder
        {
        public:
            JKRBuilder& SetYoungsModulus( float e )
            {
                m_youngsModulus = e;
                return *this;
            }
            JKRBuilder& SetPoissonRatio( float nu )
            {
                m_poissonRatio = nu;
                return *this;
            }
            JKRBuilder& SetAdhesionEnergy( float w )
            {
                m_adhesionEnergy = w;
                return *this;
            }
            JKRBuilder& SetMaxInteractionRadius( float r )
            {
                m_maxRadius = r;
                return *this;
            }

            Behaviours::Biomechanics Build() const
            {
                Behaviours::Biomechanics b;

                // Calculate Effective Young's Modulus for two interacting identical spheres
                float effectiveStiffness = m_youngsModulus / ( 2.0f * ( 1.0f - m_poissonRatio * m_poissonRatio ) );

                b.repulsionStiffness = effectiveStiffness;
                b.adhesionStrength   = m_adhesionEnergy;
                b.maxRadius          = m_maxRadius;

                return b;
            }

        private:
            float m_youngsModulus  = 15.0f; // kPa (Stiffness)
            float m_poissonRatio   = 0.4f;  // Dimensionless (Compressibility)
            float m_adhesionEnergy = 2.0f;  // Adhesion strength
            float m_maxRadius      = 1.5f;  // micrometers
        };

        static JKRBuilder JKR() { return JKRBuilder(); }
    };

} // namespace DigitalTwin