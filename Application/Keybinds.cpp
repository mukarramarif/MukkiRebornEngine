#include "Application/Keybinds.h"

#include "Utilities/Assert.h"

#include <utility>

namespace mukki
{

Binding Binding::Key(int key)
{
    return Binding{Source::Key, key};
}

Binding Binding::Mouse(int button)
{
    return Binding{Source::MouseButton, button};
}

void Keybinds::Bind(std::string action, Binding binding)
{
    MRE_ASSERT(!action.empty(), "Keybinds::Bind called with an empty action name");

    for (Action& existing : m_actions)
    {
        if (existing.name == action)
        {
            existing.bindings.push_back(binding);
            return;
        }
    }

    m_actions.push_back(Action{std::move(action), {binding}});
}

void Keybinds::Unbind(std::string_view action)
{
    for (auto it = m_actions.begin(); it != m_actions.end(); ++it)
    {
        if (it->name == action)
        {
            m_actions.erase(it);
            return;
        }
    }
}

void Keybinds::Clear()
{
    m_actions.clear();
}

void Keybinds::Update(const InputState& state)
{
    m_previous = m_current;
    m_current = state;
}

bool Keybinds::IsActionDown(std::string_view action) const
{
    const Action* found = FindAction(action);
    if (found == nullptr)
    {
        return false;
    }

    for (const Binding& binding : found->bindings)
    {
        if (IsBindingDown(binding, m_current))
        {
            return true;
        }
    }

    return false;
}

bool Keybinds::IsActionPressed(std::string_view action) const
{
    const Action* found = FindAction(action);
    if (found == nullptr)
    {
        return false;
    }

    for (const Binding& binding : found->bindings)
    {
        if (IsBindingDown(binding, m_current) && !IsBindingDown(binding, m_previous))
        {
            return true;
        }
    }

    return false;
}

bool Keybinds::IsActionReleased(std::string_view action) const
{
    const Action* found = FindAction(action);
    if (found == nullptr)
    {
        return false;
    }

    for (const Binding& binding : found->bindings)
    {
        if (!IsBindingDown(binding, m_current) && IsBindingDown(binding, m_previous))
        {
            return true;
        }
    }

    return false;
}

float Keybinds::GetAxis(std::string_view negativeAction, std::string_view positiveAction) const
{
    const float negative = IsActionDown(negativeAction) ? 1.0f : 0.0f;
    const float positive = IsActionDown(positiveAction) ? 1.0f : 0.0f;
    return positive - negative;
}

const Keybinds::Action* Keybinds::FindAction(std::string_view action) const
{
    for (const Action& existing : m_actions)
    {
        if (existing.name == action)
        {
            return &existing;
        }
    }

    return nullptr;
}

bool Keybinds::IsBindingDown(const Binding& binding, const InputState& state)
{
    switch (binding.source)
    {
    case Binding::Source::Key:
        return state.IsKeyDown(binding.code);
    case Binding::Source::MouseButton:
        return state.IsMouseButtonDown(binding.code);
    }

    MRE_UNREACHABLE();
}

} // namespace mukki
