#pragma once
#include <string>
#include <variant>

namespace DigitalTwin::Behaviours
{
    /**
     * @brief Represents random thermal motion of particles.
     * Pure data structure (AST node) for the Simulation Compiler.
     */
    struct BrownianMotion
    {
        float speed = 1.0f;
    };

    struct ConsumeField
    {
        std::string fieldName;
        float       rate = 1.0f;
    };

    struct SecreteField
    {
        std::string fieldName;
        float       rate = 1.0f;
    };

    /**
     * @brief Raw data structure for the GPU Compute Compiler.
     * Contains pre-calculated mathematical constants derived from biological parameters.
     */
    struct Biomechanics
    {
        float repulsionStiffness = 15.0f; // How hard cells push each other apart (Hertz force)
        float adhesionStrength   = 2.0f;  // How strongly cells stick together (JKR force)
        float maxRadius          = 1.5f;  // The interaction radius for collision detection
    };

    // Future mechanisms: Chemotaxis, Proliferate, etc.
} // namespace DigitalTwin::Behaviours

namespace DigitalTwin
{

    /**
     * @brief Factory class to generate pre-calculated GPU parameters
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
                // E* = E / (2 * (1 - nu^2))
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

    // A variant holding all possible behaviours the engine understands
    using BehaviourVariant = std::variant<Behaviours::BrownianMotion, Behaviours::ConsumeField, Behaviours::SecreteField, Behaviours::Biomechanics>;

    // Wrapper to attach execution parameters (like frequency) to a behaviour
    struct BehaviourRecord
    {
        BehaviourVariant behaviour;
        float            targetHz = 60.0f; // Default to 60 executions per second

        // Fluent API for frequency
        BehaviourRecord& SetHz( float hz )
        {
            targetHz = hz;
            return *this;
        }
    };
} // namespace DigitalTwin