#pragma once
#include "core/Base.hpp"
#include <glm/glm.hpp>

namespace DigitalTwin
{

    /**
     * @brief Orbit Camera (Arcball style).
     * Rotates around a target point (Focal Point). ideal for inspecting biological structures.
     * Controls:
     * - Right Drag: Rotate (Orbit)
     * - Middle Drag (or Shift+Right): Pan
     * - Scroll: Zoom (Distance)
     */
    class Camera
    {
    public:
        Camera( float fov, float aspectRatio, float nearClip, float farClip );

        void OnUpdate( float dt );
        void OnResize( uint32_t width, uint32_t height );

        // Getters
        const glm::mat4& GetView() const { return m_viewMatrix; }
        const glm::mat4& GetProjection() const { return m_projectionMatrix; }
        const glm::mat4& GetViewProjection() const { return m_viewProjection; }
        const glm::vec3& GetPosition() const { return m_position; }

        // Setters for initial view
        void SetFocalPoint( const glm::vec3& point )
        {
            m_focalPoint = point;
            RecalculateView();
        }
        void SetDistance( float dist )
        {
            m_distance = dist;
            RecalculateView();
        }

    private:
        void RecalculateView();
        void RecalculateProjection();

        // Helper to get position from spherical coordinates
        glm::vec3 CalculatePosition() const;

    private:
        float m_fov, m_aspectRatio, m_nearClip, m_farClip;

        glm::mat4 m_viewMatrix;
        glm::mat4 m_projectionMatrix;
        glm::mat4 m_viewProjection;

        glm::vec3 m_position   = { 0.0f, 0.0f, 0.0f };
        glm::vec3 m_focalPoint = { 0.0f, 0.0f, 0.0f }; // Target to look at

        // Spherical coordinates
        float m_distance = 15.0f;
        float m_pitch    = 0.0f; // Radians roughly
        float m_yaw      = 0.0f; // Radians roughly

        glm::vec2 m_initialMousePos = { 0.0f, 0.0f };
    };
} // namespace DigitalTwin