#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

namespace DigitalTwin
{
    class Input;

    /**
     * @brief Orbital Camera (Blender Style).
     * Rotates around a specific target (Focal Point).
     *
     * Controls:
     * - Middle Mouse Drag: Orbit around target.
     * - Shift + Middle Mouse Drag: Pan (move target).
     * - Scroll: Zoom (change distance to target).
     */
    class Camera
    {
    public:
        Camera( float fov, float aspectRatio, float nearClip, float farClip );
        ~Camera() = default;

        /**
         * @brief Updates camera logic based on input.
         * @param dt Delta time (seconds).
         * @param input Pointer to the input system to poll mouse/keyboard state.
         */
        void OnUpdate( float dt, const Input* input );

        /**
         * @brief Updates the projection matrix based on new window dimensions.
         */
        void OnResize( uint32_t width, uint32_t height );

        // --- Getters ---
        const glm::mat4& GetView() const { return m_viewMatrix; }
        const glm::mat4& GetProjection() const { return m_projectionMatrix; }
        const glm::mat4& GetViewProjection() const { return m_viewProjection; }

        const glm::vec3& GetPosition() const { return m_position; }
        const glm::vec3& GetFocalPoint() const { return m_focalPoint; }
        float            GetDistance() const { return m_distance; }

        // --- Setters ---
        void SetFocalPoint( const glm::vec3& point );
        void SetDistance( float distance );

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

        // Orbit parameters
        glm::vec3 m_position   = { 0.0f, 0.0f, 0.0f };
        glm::vec3 m_focalPoint = { 0.0f, 0.0f, 0.0f }; // Target center
        float     m_distance   = 10.0f;
        float     m_pitch      = 0.0f; // Radians
        float     m_yaw        = 0.0f; // Radians

        // Input state
        glm::vec2 m_lastMousePos = { 0.0f, 0.0f };
        bool      m_firstUpdate  = true;
    };

} // namespace DigitalTwin