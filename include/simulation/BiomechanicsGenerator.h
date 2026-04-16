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
            JKRBuilder& SetDampingCoefficient( float d )
            {
                m_damping = d;
                return *this;
            }
            // Cortical actomyosin tension (Maître et al. 2012). Adds an outward
            // per-neighbour force `k_T · overlap` linearly opposing adhesion.
            // Default 0 preserves pre-Phase-4 behaviour.
            JKRBuilder& SetCorticalTension( float t )
            {
                m_corticalTension = t;
                return *this;
            }
            // Phase 4.5-B — lateral (edge-to-edge) junctional adhesion. Hull-
            // pair translational pull modelling cadherin belt contacts.
            JKRBuilder& SetLateralAdhesionScale( float s )
            {
                m_lateralAdhesionScale = s;
                return *this;
            }

            Behaviours::Biomechanics Build() const
            {
                Behaviours::Biomechanics b;

                // Calculate Effective Young's Modulus for two interacting identical spheres
                float effectiveStiffness = m_youngsModulus / ( 2.0f * ( 1.0f - m_poissonRatio * m_poissonRatio ) );

                b.repulsionStiffness    = effectiveStiffness;
                b.adhesionStrength      = m_adhesionEnergy;
                b.maxRadius             = m_maxRadius;
                b.dampingCoefficient    = m_damping;
                b.corticalTension       = m_corticalTension;
                b.lateralAdhesionScale  = m_lateralAdhesionScale;

                return b;
            }

        private:
            float m_youngsModulus         = 15.0f; // kPa (Stiffness)
            float m_poissonRatio          = 0.4f;  // Dimensionless (Compressibility)
            float m_adhesionEnergy        = 2.0f;  // Adhesion strength
            float m_maxRadius             = 1.5f;  // micrometers
            float m_damping               = 0.0f;  // Velocity drag coefficient
            float m_corticalTension       = 0.0f;  // Actomyosin contractility (0 = off)
            float m_lateralAdhesionScale  = 0.0f;  // Cadherin-belt lateral pull (0 = off)
        };

        static JKRBuilder JKR() { return JKRBuilder(); }
    };

} // namespace DigitalTwin