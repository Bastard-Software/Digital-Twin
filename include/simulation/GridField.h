#pragma once
#include "core/Core.h"
#include <string>

namespace DigitalTwin
{
    /**
     * @brief Defines a continuous physical field (e.g., Oxygen, VEGF) solved via PDEs on a 3D grid.
     * Uses a fluent interface for configuration.
     */
    class DT_API GridField
    {
    public:
        explicit GridField( const std::string& name )
            : m_name( name )
        {
        }

        // --- Fluent Setters ---
        GridField& SetInitialConcentration( float v )
        {
            m_initialConcentration = v;
            return *this;
        }
        GridField& SetDiffusionCoefficient( float v )
        {
            m_diffusionCoefficient = v;
            return *this;
        }
        GridField& SetDecayRate( float v )
        {
            m_decayRate = v;
            return *this;
        }
        GridField& SetComputeHz( float v )
        {
            m_computeHz = v;
            return *this;
        }

        // --- Getters ---
        const std::string& GetName() const { return m_name; }
        float              GetInitialConcentration() const { return m_initialConcentration; }
        float              GetDiffusionCoefficient() const { return m_diffusionCoefficient; }
        float              GetDecayRate() const { return m_decayRate; }
        float              GetComputeHz() const { return m_computeHz; }

    private:
        std::string m_name;
        float       m_initialConcentration = 0.0f;
        float       m_diffusionCoefficient = 0.0f;
        float       m_decayRate            = 0.0f;
        float       m_computeHz            = 60.0f;
    };
} // namespace DigitalTwin