#include "renderer/Camera.h"

#include "core/Log.h"
#include "platform/Input.h" // Assuming this path based on provided files

// GLM definitions for quaternion and matrix math
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/quaternion.hpp>

namespace DigitalTwin
{
    Camera::Camera( float fov, float aspectRatio, float nearClip, float farClip )
        : m_fov( fov )
        , m_aspectRatio( aspectRatio )
        , m_nearClip( nearClip )
        , m_farClip( farClip )
    {
        // Default orientation
        m_yaw   = glm::radians( 45.0f );
        m_pitch = glm::radians( -30.0f ); // Slight look down

        RecalculateProjection();
        RecalculateView();
    }

    void Camera::SetFocalPoint( const glm::vec3& point )
    {
        m_focalPoint = point;
        RecalculateView();
    }

    void Camera::SetDistance( float distance )
    {
        m_distance = std::max( distance, 0.1f ); // Prevent zero/negative distance
        RecalculateView();
    }

    void Camera::OnResize( uint32_t width, uint32_t height )
    {
        if( height == 0 )
            return;

        m_aspectRatio = ( float )width / ( float )height;
        RecalculateProjection();
    }

    void Camera::RecalculateProjection()
    {
        m_projectionMatrix = glm::perspective( glm::radians( m_fov ), m_aspectRatio, m_nearClip, m_farClip );

        // Vulkan clip space has inverted Y compared to OpenGL.
        // GLM is designed for OpenGL, so we flip the Y axis manually.
        m_projectionMatrix[ 1 ][ 1 ] *= -1.0f;

        m_viewProjection = m_projectionMatrix * m_viewMatrix;
    }

    void Camera::RecalculateView()
    {
        // Sphere coordinates to Cartesian conversion
        // This calculates the camera position on a sphere around the focal point.
        float x = m_distance * cos( m_pitch ) * cos( m_yaw );
        float y = m_distance * sin( m_pitch );
        float z = m_distance * cos( m_pitch ) * sin( m_yaw );

        m_position = m_focalPoint + glm::vec3( x, y, z );

        // Look at the focal point from the calculated position
        // Up vector is always global Y (0,1,0) to keep the horizon stable
        m_viewMatrix = glm::lookAt( m_position, m_focalPoint, glm::vec3( 0.0f, 1.0f, 0.0f ) );

        m_viewProjection = m_projectionMatrix * m_viewMatrix;
    }

    void Camera::OnUpdate( float dt, const Input* input )
    {
        if( !input )
            return;

        auto [ mouseX, mouseY ] = input->GetMousePosition();
        glm::vec2 mouse         = { mouseX, mouseY };

        if( m_firstUpdate )
        {
            m_lastMousePos = mouse;
            m_firstUpdate  = false;
        }

        glm::vec2 delta = ( mouse - m_lastMousePos ) * 0.003f; // Sensitivity
        m_lastMousePos  = mouse;

        // --- MOUSE INPUT ---
        bool isMiddleDown = input->IsMouseButtonPressed( Mouse::Middle );
        bool isShiftDown  = input->IsKeyPressed( Key::LeftShift ) || input->IsKeyPressed( Key::RightShift );

        if( isMiddleDown )
        {
            if( isShiftDown )
            {
                // --- PAN (Shift + MMB) ---
                // Move the focal point along the camera's local Right and Up vectors.

                // 1. Get Camera Basis Vectors
                glm::vec3 forward = glm::normalize( m_focalPoint - m_position );
                glm::vec3 right   = glm::normalize( glm::cross( forward, glm::vec3( 0, 1, 0 ) ) );
                glm::vec3 up      = glm::normalize( glm::cross( right, forward ) );

                // 2. Calculate Pan Speed
                // Speed increases with distance to keep panning feeling natural at different zooms.
                float panSpeed = m_distance * 0.5f;

                // 3. Apply movement (invert delta logic for "drag the world" feel)
                m_focalPoint -= right * delta.x * panSpeed;
                m_focalPoint += up * delta.y * panSpeed; // Y is inverted in 2D space vs 3D Up
            }
            else
            {
                // --- ORBIT (MMB) ---
                // Rotate around the focal point.

                float rotationSpeed = 2.0f;
                m_yaw += delta.x * rotationSpeed;
                m_pitch += delta.y * rotationSpeed;

                // Clamp pitch to avoid gimbal lock (camera flipping over at the poles)
                // Limit slightly less than 90 degrees (1.57 radians)
                constexpr float PITCH_LIMIT = 1.56f;
                if( m_pitch > PITCH_LIMIT )
                    m_pitch = PITCH_LIMIT;
                if( m_pitch < -PITCH_LIMIT )
                    m_pitch = -PITCH_LIMIT;
            }

            // Rebuild view if we moved
            RecalculateView();
        }

        // --- ZOOM (Scroll) ---
        float scroll = input->GetScrollY();
        if( scroll != 0.0f )
        {
            float zoomSpeed = 0.5f;
            float zoomLevel = scroll * m_distance * zoomSpeed;

            m_distance -= zoomLevel;

            // Prevent zooming through the target or into negative distance
            if( m_distance < 0.1f )
            {
                m_distance = 0.1f;

                // Optional: You could push the focal point forward here to create a "dolly" effect
                // instead of stopping, but for a strict orbit camera, clamping is safer.
            }

            RecalculateView();
        }
    }

} // namespace DigitalTwin