#include "renderer/Camera.h"

#include "core/Log.h"

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/quaternion.hpp>

namespace DigitalTwin
{
    namespace
    {
        constexpr glm::vec3 kWorldUp    = { 0.0f, 1.0f, 0.0f };
        constexpr float     kOrbitSens  = 0.006f; // radians per pixel
        constexpr float     kPanSens    = 0.0015f;
        constexpr float     kZoomSens   = 0.1f;
        constexpr float     kMinDistance = 0.1f;
    } // namespace

    Camera::Camera( float fov, float aspectRatio, float nearClip, float farClip )
        : m_fov( fov )
        , m_aspectRatio( aspectRatio )
        , m_nearClip( nearClip )
        , m_farClip( farClip )
    {
        // Default orientation: 45° yaw around world-Y, 30° pitch around local X.
        // Matches the previous pitch/yaw defaults roughly — a slight overhead angle
        // looking toward the focal point.
        glm::quat qYaw   = glm::angleAxis( glm::radians( 45.0f ), kWorldUp );
        glm::quat qPitch = glm::angleAxis( glm::radians( 30.0f ), glm::vec3( 1.0f, 0.0f, 0.0f ) );
        m_orientation    = qYaw * qPitch;

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
        m_distance = std::max( distance, kMinDistance );
        RecalculateView();
    }

    void Camera::SetOrientation( const glm::quat& q )
    {
        m_orientation = glm::normalize( q );
        RecalculateView();
    }

    void Camera::FocusOn( const glm::vec3& point, float distance )
    {
        m_focalPoint = point;
        m_distance   = std::max( distance, kMinDistance );
        RecalculateView();
    }

    void Camera::OnResize( uint32_t width, uint32_t height )
    {
        if( height == 0 )
            return;

        m_aspectRatio = ( float )width / ( float )height;
        RecalculateProjection();
    }

    void Camera::Orbit( const glm::vec2& pixelDelta )
    {
        float yawDelta   = -pixelDelta.x * kOrbitSens;
        float pitchDelta = -pixelDelta.y * kOrbitSens;

        // Yaw around world-up (keeps the horizon horizontal).
        glm::quat qYaw = glm::angleAxis( yawDelta, kWorldUp );

        // Pitch around the camera's current local-right axis, expressed in world space.
        glm::vec3 localRight = m_orientation * glm::vec3( 1.0f, 0.0f, 0.0f );
        glm::quat qPitch     = glm::angleAxis( pitchDelta, localRight );

        m_orientation = glm::normalize( qYaw * qPitch * m_orientation );
        RecalculateView();
    }

    void Camera::Pan( const glm::vec2& pixelDelta )
    {
        glm::vec3 right = m_orientation * glm::vec3( 1.0f, 0.0f, 0.0f );
        glm::vec3 up    = m_orientation * glm::vec3( 0.0f, 1.0f, 0.0f );

        float panSpeed = m_distance * kPanSens;

        // Drag-the-world feel: cursor right → world pans right → focal moves left.
        m_focalPoint -= right * pixelDelta.x * panSpeed;
        m_focalPoint += up * pixelDelta.y * panSpeed;
        RecalculateView();
    }

    void Camera::Zoom( float scrollAmount )
    {
        m_distance -= scrollAmount * m_distance * kZoomSens * 5.0f;
        if( m_distance < kMinDistance )
            m_distance = kMinDistance;
        RecalculateView();
    }

    void Camera::RecalculateProjection()
    {
        m_projectionMatrix = glm::perspective( glm::radians( m_fov ), m_aspectRatio, m_nearClip, m_farClip );

        // Vulkan clip space has inverted Y compared to OpenGL.
        m_projectionMatrix[ 1 ][ 1 ] *= -1.0f;

        m_viewProjection = m_projectionMatrix * m_viewMatrix;
    }

    void Camera::RecalculateView()
    {
        // Camera sits at focalPoint + orientation * (0, 0, distance).
        // Orientation's Z axis points from focal toward camera.
        glm::vec3 offset = m_orientation * glm::vec3( 0.0f, 0.0f, m_distance );
        m_position       = m_focalPoint + offset;

        // Up vector derived from orientation — never parallel to view direction,
        // so lookAt stays non-degenerate for any rotation including past ±90° pitch.
        glm::vec3 up = m_orientation * glm::vec3( 0.0f, 1.0f, 0.0f );
        m_viewMatrix = glm::lookAt( m_position, m_focalPoint, up );

        m_viewProjection = m_projectionMatrix * m_viewMatrix;
    }

} // namespace DigitalTwin
