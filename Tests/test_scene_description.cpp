#include "Scene/SceneDescription.h"

#include <cmath>

#include <gtest/gtest.h>
#include <glm/gtc/matrix_transform.hpp>

namespace
{

mukki::SceneObject MakeObject(const glm::vec3& rotationDegrees)
{
    mukki::SceneObject object{};
    object.position = glm::vec3{1.0f, -2.0f, 3.0f};
    object.rotation = rotationDegrees;
    object.scale = glm::vec3{1.0f, 1.0f, 1.0f};
    return object;
}

} // namespace

TEST(SceneDescription, BuildObjectRotationMatchesTransformOrder)
{
    const glm::vec3 degrees{25.0f, -40.0f, 65.0f};
    const mukki::SceneObject object = MakeObject(degrees);

    glm::mat4 expected{1.0f};
    expected = glm::rotate(expected, glm::radians(degrees.y), glm::vec3{0.0f, 1.0f, 0.0f});
    expected = glm::rotate(expected, glm::radians(degrees.x), glm::vec3{1.0f, 0.0f, 0.0f});
    expected = glm::rotate(expected, glm::radians(degrees.z), glm::vec3{0.0f, 0.0f, 1.0f});

    const glm::mat4 actual = glm::mat4_cast(mukki::BuildObjectRotation(object));
    for (int column = 0; column < 4; ++column)
    {
        for (int row = 0; row < 4; ++row)
        {
            EXPECT_NEAR(actual[column][row], expected[column][row], 1.0e-5f);
        }
    }
}

TEST(SceneDescription, ObjectRotationRoundTripsThroughQuaternion)
{
    const glm::vec3 cases[] = {
        glm::vec3{0.0f, 0.0f, 0.0f},
        glm::vec3{15.0f, 0.0f, 0.0f},
        glm::vec3{0.0f, 90.0f, 0.0f},
        glm::vec3{0.0f, 180.0f, 0.0f},
        glm::vec3{-30.0f, 45.0f, 120.0f},
        glm::vec3{80.0f, -170.0f, -60.0f},
    };

    for (const glm::vec3& degrees : cases)
    {
        const mukki::SceneObject object = MakeObject(degrees);
        const glm::quat rotation = mukki::BuildObjectRotation(object);
        mukki::SceneObject restored = object;
        restored.rotation = mukki::ExtractObjectRotationEuler(rotation);

        // The quaternion is the canonical representation; angles may differ by full turns.
        const glm::quat restoredRotation = mukki::BuildObjectRotation(restored);
        EXPECT_NEAR(std::abs(glm::dot(rotation, restoredRotation)), 1.0f, 1.0e-4f);
    }
}

TEST(SceneDescription, ExtractObjectRotationEulerRecoversSimpleAngles)
{
    const glm::vec3 euler = mukki::ExtractObjectRotationEuler(glm::quat{1.0f, 0.0f, 0.0f, 0.0f});
    EXPECT_NEAR(euler.x, 0.0f, 1.0e-5f);
    EXPECT_NEAR(euler.y, 0.0f, 1.0e-5f);
    EXPECT_NEAR(euler.z, 0.0f, 1.0e-5f);

    const glm::vec3 yawOnly =
        mukki::ExtractObjectRotationEuler(mukki::BuildObjectRotation(MakeObject(glm::vec3{0.0f, 30.0f, 0.0f})));
    EXPECT_NEAR(yawOnly.x, 0.0f, 1.0e-4f);
    EXPECT_NEAR(yawOnly.y, 30.0f, 1.0e-4f);
    EXPECT_NEAR(yawOnly.z, 0.0f, 1.0e-4f);

    const glm::vec3 pitchOnly =
        mukki::ExtractObjectRotationEuler(mukki::BuildObjectRotation(MakeObject(glm::vec3{-25.0f, 0.0f, 0.0f})));
    EXPECT_NEAR(pitchOnly.x, -25.0f, 1.0e-4f);
    EXPECT_NEAR(pitchOnly.y, 0.0f, 1.0e-4f);
    EXPECT_NEAR(pitchOnly.z, 0.0f, 1.0e-4f);
}
