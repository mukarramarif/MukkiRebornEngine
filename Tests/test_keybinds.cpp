#include "Application/Keybinds.h"

#include <gtest/gtest.h>

namespace
{

constexpr int kKeyA = 65;
constexpr int kKeyD = 68;
constexpr int kMouseLeft = 0;

mukki::InputState StateWithKey(int key, bool down)
{
    mukki::InputState state{};
    state.keys[static_cast<std::size_t>(key)] = down;
    return state;
}

mukki::InputState StateWithMouseButton(int button, bool down)
{
    mukki::InputState state{};
    state.mouseButtons[static_cast<std::size_t>(button)] = down;
    return state;
}

TEST(Keybinds, PressedAndReleasedEdges)
{
    mukki::Keybinds keybinds{};
    keybinds.Bind("move.left", mukki::Binding::Key(kKeyA));

    keybinds.Update(StateWithKey(kKeyA, false));
    EXPECT_FALSE(keybinds.IsActionDown("move.left"));
    EXPECT_FALSE(keybinds.IsActionPressed("move.left"));

    keybinds.Update(StateWithKey(kKeyA, true));
    EXPECT_TRUE(keybinds.IsActionDown("move.left"));
    EXPECT_TRUE(keybinds.IsActionPressed("move.left"));
    EXPECT_FALSE(keybinds.IsActionReleased("move.left"));

    keybinds.Update(StateWithKey(kKeyA, true));
    EXPECT_TRUE(keybinds.IsActionDown("move.left"));
    EXPECT_FALSE(keybinds.IsActionPressed("move.left"));
    EXPECT_FALSE(keybinds.IsActionReleased("move.left"));

    keybinds.Update(StateWithKey(kKeyA, false));
    EXPECT_FALSE(keybinds.IsActionDown("move.left"));
    EXPECT_FALSE(keybinds.IsActionPressed("move.left"));
    EXPECT_TRUE(keybinds.IsActionReleased("move.left"));
}

TEST(Keybinds, MultipleBindingsOnOneAction)
{
    mukki::Keybinds keybinds{};
    keybinds.Bind("move.left", mukki::Binding::Key(kKeyA));
    keybinds.Bind("move.left", mukki::Binding::Mouse(kMouseLeft));

    keybinds.Update(StateWithMouseButton(kMouseLeft, true));
    EXPECT_TRUE(keybinds.IsActionDown("move.left"));
}

TEST(Keybinds, AxisFromActions)
{
    mukki::Keybinds keybinds{};
    keybinds.Bind("move.left", mukki::Binding::Key(kKeyA));
    keybinds.Bind("move.right", mukki::Binding::Key(kKeyD));

    keybinds.Update(StateWithKey(kKeyA, true));
    EXPECT_FLOAT_EQ(keybinds.GetAxis("move.left", "move.right"), -1.0f);

    keybinds.Update(StateWithKey(kKeyD, true));
    EXPECT_FLOAT_EQ(keybinds.GetAxis("move.left", "move.right"), 1.0f);

    mukki::InputState both{};
    both.keys[static_cast<std::size_t>(kKeyA)] = true;
    both.keys[static_cast<std::size_t>(kKeyD)] = true;
    keybinds.Update(both);
    EXPECT_FLOAT_EQ(keybinds.GetAxis("move.left", "move.right"), 0.0f);
}

TEST(Keybinds, UnknownActionIsInactive)
{
    mukki::Keybinds keybinds{};
    keybinds.Update(StateWithKey(kKeyA, true));

    EXPECT_FALSE(keybinds.IsActionDown("does.not.exist"));
    EXPECT_FALSE(keybinds.IsActionPressed("does.not.exist"));
    EXPECT_FALSE(keybinds.IsActionReleased("does.not.exist"));
}

TEST(Keybinds, UnbindRemovesAction)
{
    mukki::Keybinds keybinds{};
    keybinds.Bind("move.left", mukki::Binding::Key(kKeyA));
    keybinds.Unbind("move.left");

    keybinds.Update(StateWithKey(kKeyA, true));
    EXPECT_FALSE(keybinds.IsActionDown("move.left"));
}

} // namespace
