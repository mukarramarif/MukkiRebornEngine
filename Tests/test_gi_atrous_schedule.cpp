#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#include "Renderer/GlobalIllumination/GiAtrousSchedule.h"

namespace mukki
{
namespace
{

TEST(GiAtrousSchedule, SingleIterationWritesFiltered)
{
    const std::vector<GiAtrousStep> steps = BuildGiAtrousSchedule(1, 8);
    ASSERT_EQ(steps.size(), 1u);
    EXPECT_EQ(steps[0].input, GiAtrousTexture::History);
    EXPECT_EQ(steps[0].output, GiAtrousTexture::Filtered);
}

TEST(GiAtrousSchedule, EvenIterationsPingPong)
{
    const std::vector<GiAtrousStep> steps = BuildGiAtrousSchedule(4, 8);
    ASSERT_EQ(steps.size(), 4u);
    EXPECT_EQ(steps[0].input, GiAtrousTexture::History);
    EXPECT_EQ(steps[0].output, GiAtrousTexture::Noisy);
    EXPECT_EQ(steps[1].input, GiAtrousTexture::Noisy);
    EXPECT_EQ(steps[1].output, GiAtrousTexture::Filtered);
    EXPECT_EQ(steps[2].input, GiAtrousTexture::Filtered);
    EXPECT_EQ(steps[2].output, GiAtrousTexture::Noisy);
    EXPECT_EQ(steps[3].input, GiAtrousTexture::Noisy);
    EXPECT_EQ(steps[3].output, GiAtrousTexture::Filtered);
}

TEST(GiAtrousSchedule, AlwaysEndsOnFiltered)
{
    for (std::uint32_t iterations = 1; iterations <= 8; ++iterations)
    {
        const std::vector<GiAtrousStep> steps = BuildGiAtrousSchedule(iterations, 8);
        ASSERT_EQ(steps.size(), iterations);
        EXPECT_EQ(steps.back().output, GiAtrousTexture::Filtered);
    }
}

TEST(GiAtrousSchedule, ChainsOutputsAndNeverReadsTheWrittenTarget)
{
    const std::vector<GiAtrousStep> steps = BuildGiAtrousSchedule(6, 8);
    for (std::size_t index = 0; index < steps.size(); ++index)
    {
        EXPECT_NE(steps[index].input, steps[index].output);
        if (index > 0)
        {
            EXPECT_EQ(steps[index].input, steps[index - 1].output);
        }
    }
}

TEST(GiAtrousSchedule, ClampsIterationCount)
{
    EXPECT_EQ(BuildGiAtrousSchedule(0, 8).size(), 1u);
    EXPECT_EQ(BuildGiAtrousSchedule(100, 8).size(), 8u);
}

} // namespace
} // namespace mukki
