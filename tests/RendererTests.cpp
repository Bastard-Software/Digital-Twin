#include "platform/Input.h"
#include "renderer/Camera.h"
#include <glm/gtc/matrix_access.hpp>
#include <gtest/gtest.h>

using namespace DigitalTwin;

class CameraMathTest : public ::testing::Test
{
protected:
    // Standard 16:9 camera
    std::unique_ptr<Camera> m_camera;

    void SetUp() override { m_camera = std::make_unique<Camera>( 45.0f, 16.0f / 9.0f, 0.1f, 1000.0f ); }
};

// 1. Initial state test
TEST_F( CameraMathTest, InitialState )
{
    // Verify defaults
    EXPECT_GT( m_camera->GetDistance(), 0.0f );
    EXPECT_EQ( m_camera->GetFocalPoint(), glm::vec3( 0, 0, 0 ) );

    // Check View Matrix is not Identity (it should be looking from some offset)
    glm::mat4 view       = m_camera->GetView();
    bool      isIdentity = view == glm::mat4( 1.0f );
    EXPECT_FALSE( isIdentity );
}

// 2. Zoom logic test
TEST_F( CameraMathTest, ZoomLogic )
{
    float initialDist = m_camera->GetDistance();

    // Simulate Zoom In (Decrease distance)
    m_camera->SetDistance( initialDist * 0.5f );

    EXPECT_LT( m_camera->GetDistance(), initialDist );

    // Calculate expected position magnitude
    float posLen = glm::length( m_camera->GetPosition() - m_camera->GetFocalPoint() );
    EXPECT_NEAR( posLen, initialDist * 0.5f, 0.001f );
}

// 3. Pan logic test
TEST_F( CameraMathTest, PanLogic )
{
    glm::vec3 initialPos = m_camera->GetPosition();
    glm::vec3 newFocus   = { 10.0f, 5.0f, 0.0f };

    m_camera->SetFocalPoint( newFocus );

    // If we move the focal point, the camera position should move by the same amount
    // to maintain relative orbit.
    glm::vec3 newPos   = m_camera->GetPosition();
    glm::vec3 deltaPos = newPos - initialPos;

    EXPECT_NEAR( deltaPos.x, newFocus.x, 0.001f );
    EXPECT_NEAR( deltaPos.y, newFocus.y, 0.001f );
    EXPECT_NEAR( deltaPos.z, newFocus.z, 0.001f );
}

// 4. Projection matrix test}
TEST_F( CameraMathTest, ProjectionMatrix )
{
    // Check Vulkan Y-Flip in projection
    glm::mat4 proj = m_camera->GetProjection();

    // In standard GLM perspective, [1][1] is positive.
    // In our Vulkan fix, it should be negative (or flipped relative to standard).
    // Specifically, m_projectionMatrix[1][1] *= -1.0f;

    // We can verify aspect ratio impact
    m_camera->OnResize( 100, 100 ); // Aspect 1.0
    glm::mat4 projSquare = m_camera->GetProjection();

    EXPECT_NE( proj[ 0 ][ 0 ], projSquare[ 0 ][ 0 ] ); // X scale changes with aspect ratio
}