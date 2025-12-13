#include "renderer/Camera.hpp"

#include "platform/Input.hpp"
#include "platform/KeyCodes.hpp"
#include "platform/MouseCodes.hpp"
#include <glm/gtc/matrix_transform.hpp>

namespace DigitalTwin
{
    Camera::Camera( float fov, float aspectRatio, float nearClip, float farClip )
        : m_fov( fov )
        , m_aspectRatio( aspectRatio )
        , m_nearClip( nearClip )
        , m_farClip( farClip )
    {
        m_yaw   = glm::radians( 90.0f );
        m_pitch = 0.0f;

        RecalculateProjection();
        RecalculateView();
    }

    void Camera::OnResize( uint32_t width, uint32_t height )
    {
        if( height == 0 )
            return;
        m_aspectRatio = static_cast<float>( width ) / static_cast<float>( height );
        RecalculateProjection();
    }

    void Camera::OnUpdate( float dt )
    {
        // 1. Poll Mouse Input
        auto [ mouseX, mouseY ] = Input::GetMousePosition();
        glm::vec2 mouse         = { mouseX, mouseY };
        glm::vec2 delta         = ( mouse - m_initialMousePos ) * 0.003f; // Sensitivity factor
        m_initialMousePos       = mouse;

        // Check Input States
        bool isMiddleMouse = Input::IsMouseButtonPressed( ( int )Mouse::Middle );
        bool isShift       = Input::IsKeyPressed( ( int )Key::LeftShift );

        // 2. Blender Style Controls
        if( isMiddleMouse )
        {
            if( isShift )
            {
                // --- PAN: Shift + Middle Mouse ---
                // We need to move the Focal Point (Target) along the camera's local plane.
                // To do this correctly without zooming in/out, we calculate basis vectors
                // from the current view direction.

                // A. Calculate View Direction (Vector from Camera to Target)
                // We use the previous frame's positions to determine orientation.
                glm::vec3 viewDir = glm::normalize( m_focalPoint - m_position );

                // B. Calculate Right Vector
                // Cross Product of (ViewDir) and (WorldUp) gives a vector perpendicular to both.
                // This points perfectly to the "Right" of the screen.
                glm::vec3 right = glm::normalize( glm::cross( viewDir, glm::vec3( 0.0f, 1.0f, 0.0f ) ) );

                // C. Calculate Camera Up Vector
                // Cross Product of (Right) and (ViewDir) gives the local "Up" vector for the camera.
                // Note: We don't use WorldUp (0,1,0) here because the camera might be pitched down.
                glm::vec3 up = glm::normalize( glm::cross( right, viewDir ) );

                // Calculate Pan Speed based on distance (farther objects need faster movement)
                float panSpeed = m_distance * 0.002f;

                // Apply Movement
                // We move the focal point along these calculated vectors.
                // The multipliers (800.0f) are arbitrary to match the delta sensitivity.
                m_focalPoint -= right * delta.x * panSpeed * 800.0f;
                m_focalPoint += up * delta.y * panSpeed * 800.0f;
            }
            else
            {
                // --- ORBIT: Middle Mouse Only ---
                // Rotate around the focal point.

                m_yaw -= delta.x * 2.0f;   // Dragging left rotates view left
                m_pitch += delta.y * 2.0f; // Dragging up rotates view up

                // Clamp pitch to avoid gimbal lock at the poles (approx 89 degrees)
                const float PITCH_LIMIT = 1.55f;
                if( m_pitch > PITCH_LIMIT )
                    m_pitch = PITCH_LIMIT;
                if( m_pitch < -PITCH_LIMIT )
                    m_pitch = -PITCH_LIMIT;
            }
        }

        // 3. ZOOM: Scroll Wheel
        float scroll = Input::GetScrollY();
        if( scroll != 0.0f )
        {
            // Zoom proportional to distance for smooth approach
            m_distance -= scroll * m_distance * 0.1f;

            // Prevent going through the target or into negatives
            m_distance = std::max( m_distance, 0.1f );
        }

        RecalculateView();
    }

    void Camera::RecalculateView()
    {
        // 1. Calculate Cartesian position on a sphere based on Yaw/Pitch/Distance
        float x = m_distance * cos( m_pitch ) * cos( m_yaw );
        float y = m_distance * sin( m_pitch );
        float z = m_distance * cos( m_pitch ) * sin( m_yaw );

        // Position is relative to the Focal Point
        m_position = m_focalPoint + glm::vec3( x, y, z );

        // 2. LookAt Matrix
        // Constructs a view matrix looking from 'm_position' to 'm_focalPoint'
        // World Up is always (0,1,0) to keep horizon stable.
        m_viewMatrix = glm::lookAt( m_position, m_focalPoint, glm::vec3( 0.0f, 1.0f, 0.0f ) );

        m_viewProjection = m_projectionMatrix * m_viewMatrix;
    }

    void Camera::RecalculateProjection()
    {
        m_projectionMatrix = glm::perspective( glm::radians( m_fov ), m_aspectRatio, m_nearClip, m_farClip );
        m_projectionMatrix[ 1 ][ 1 ] *= -1.0f; // Vulkan Y flip
        m_viewProjection = m_projectionMatrix * m_viewMatrix;
    }

} // namespace DigitalTwin