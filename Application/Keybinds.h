#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "Application/Input.h"

namespace mukki
{

struct Binding
{
    enum class Source : std::uint8_t
    {
        Key,
        MouseButton,
    };

    Source source = Source::Key;
    int code = 0;

    [[nodiscard]] static Binding Key(int key);
    [[nodiscard]] static Binding Mouse(int button);
};

class Keybinds
{
public:
    void Bind(std::string action, Binding binding);
    void Unbind(std::string_view action);
    void Clear();

    void Update(const InputState& state);

    [[nodiscard]] bool IsActionDown(std::string_view action) const;
    [[nodiscard]] bool IsActionPressed(std::string_view action) const;
    [[nodiscard]] bool IsActionReleased(std::string_view action) const;
    [[nodiscard]] float GetAxis(std::string_view negativeAction, std::string_view positiveAction) const;

private:
    struct Action
    {
        std::string name{};
        std::vector<Binding> bindings{};
    };

    [[nodiscard]] const Action* FindAction(std::string_view action) const;
    [[nodiscard]] static bool IsBindingDown(const Binding& binding, const InputState& state);

    std::vector<Action> m_actions{};
    InputState m_current{};
    InputState m_previous{};
};

} // namespace mukki
