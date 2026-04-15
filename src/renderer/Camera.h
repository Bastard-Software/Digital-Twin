#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

namespace DigitalTwin
{
    /**
     * @brief Orbital Camera (pure state + verb API).
     *
     * The camera polls no input. The editor (or a scripted demo) drives it via
     * Orbit/Pan/Zoom (relative, from mouse deltas) or SetFocalPoint/SetDistance/
     * SetOrientation (absolute, for cinematic sequences / Python automation).
     *
     * Orientation is stored as a quaternion, allowing unlimited rotation around
     * the focal point with no gimbal lock.
     */
    class Camera
    {
    public:
        Camera( float fov, float aspectRatio, float nearClip, float farClip );
        ~Camera() = default;

        void OnResize( uint32_t width, uint32_t height );

        // --- Interactive (relative, typically from editor mouse input) ---
        void Orbit( const glm::vec2& pixelDelta );
        void Pan( const glm::vec2& pixelDelta );
        void Zoom( float scrollAmount );

        // --- Absolute (scripted demos / Python) ---
        void SetFocalPoint( const glm::vec3& point );
        void SetDistance( float distance );
        void SetOrientation( const glm::quat& q );
        void FocusOn( const glm::vec3& point, float distance );

        // --- Getters ---
        const glm::mat4& GetView() const { return m_viewMatrix; }
        const glm::mat4& GetProjection() const { return m_projectionMatrix; }
        const glm::mat4& GetViewProjection() const { return m_viewProjection; }

        const glm::vec3& GetPosition() const { return m_position; }
        const glm::vec3& GetFocalPoint() const { return m_focalPoint; }
        const glm::quat& GetOrientation() const { return m_orientation; }
        float            GetDistance() const { return m_distance; }
        float            GetFov() const { return m_fov; }

    private:
        void RecalculateView();
        void RecalculateProjection();

    private:
        // Projection parameters
        float m_fov;
        float m_aspectRatio;
        float m_nearClip;
        float m_farClip;

        // Matrices
        glm::mat4 m_viewMatrix       = glm::mat4( 1.0f );
        glm::mat4 m_projectionMatrix = glm::mat4( 1.0f );
        glm::mat4 m_viewProjection   = glm::mat4( 1.0f );

        // Orbit state
        glm::vec3 m_position    = { 0.0f, 0.0f, 0.0f };
        glm::vec3 m_focalPoint  = { 0.0f, 0.0f, 0.0f };
        glm::quat m_orientation = glm::identity<glm::quat>();
        float     m_distance    = 10.0f;
    };

} // namespace DigitalTwin
