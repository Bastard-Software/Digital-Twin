#pragma once
#include "core/Core.h"
#include <functional>
#include <glm/glm.hpp>
#include <string>

namespace DigitalTwin
{
    using GridInitFunction = std::function<float( const glm::vec3& worldPos )>;

    /**
     * @brief Collection of mathematical generators to initialize 3D grid states.
     */
    class DT_API GridInitializer
    {
    public:
        /**
         * @brief Fills the entire grid with a constant background value.
         */
        static GridInitFunction Constant( float value )
        {
            return [ value ]( const glm::vec3& ) {
                return value;
            };
        }

        /**
         * @brief Creates a sphere of high concentration that sharply drops to the background value.
         */
        static GridInitFunction Sphere( const glm::vec3& center, float radius, float insideValue, float outsideValue )
        {
            return [ center, radius, insideValue, outsideValue ]( const glm::vec3& pos ) {
                float dist = glm::length( pos - center );
                return ( dist <= radius ) ? insideValue : outsideValue;
            };
        }

        /**
         * @brief Creates a smooth Gaussian gradient from the center. (Great for visualizing diffusion).
         */
        static GridInitFunction Gaussian( const glm::vec3& center, float sigma, float peakValue )
        {
            return [ center, sigma, peakValue ]( const glm::vec3& pos ) {
                float distSq = glm::dot( pos - center, pos - center );
                return peakValue * std::exp( -distSq / ( 2.0f * sigma * sigma ) );
            };
        }

        /**
         * @brief Creates a field with multiple overlapping sources (Gaussians).
         * Ideal for simulating a network of blood vessels.
         */
        static GridInitFunction MultiGaussian( const std::vector<glm::vec3>& centers, float sigma, float peakValue )
        {
            // Capture the 'centers' vector by value so the lambda holds its own copy
            return [ centers, sigma, peakValue ]( const glm::vec3& pos ) {
                float totalValue = 0.0f;
                float twoSigmaSq = 2.0f * sigma * sigma;

                for( const auto& center: centers )
                {
                    float distSq = glm::dot( pos - center, pos - center );
                    totalValue += peakValue * std::exp( -distSq / twoSigmaSq );
                }

                // Clamp to prevent exceeding the maximum physical concentration (e.g., 100.0)
                return std::min( totalValue, 100.0f );
            };
        }
    };

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
            m_initializer = []( const glm::vec3& ) {
                return 0.0f;
            };
        }

        // --- Fluent Setters ---
        GridField& SetInitializer( GridInitFunction fn )
        {
            m_initializer = fn;
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
        const std::string&      GetName() const { return m_name; }
        const GridInitFunction& GetInitializer() const { return m_initializer; }
        float                   GetDiffusionCoefficient() const { return m_diffusionCoefficient; }
        float                   GetDecayRate() const { return m_decayRate; }
        float                   GetComputeHz() const { return m_computeHz; }

    private:
        std::string      m_name;
        GridInitFunction m_initializer;
        float            m_diffusionCoefficient = 0.0f;
        float            m_decayRate            = 0.0f;
        float            m_computeHz            = 60.0f;
    };
} // namespace DigitalTwin