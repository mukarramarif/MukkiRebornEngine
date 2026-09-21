#include "Application/UI/UiStyles.h"

#include <imgui.h>

namespace mukki::ui
{

void ApplyDarkTheme()
{
    ImGui::StyleColorsDark();

    ImGuiStyle& style = ImGui::GetStyle();
    style.WindowRounding = 4.0f;
    style.FrameRounding = 3.0f;
    style.GrabRounding = 3.0f;
    style.WindowBorderSize = 1.0f;
    style.FrameBorderSize = 0.0f;
    style.WindowPadding = ImVec2(8.0f, 8.0f);

    ImVec4* colors = style.Colors;
    colors[ImGuiCol_WindowBg] = ImVec4(0.08f, 0.09f, 0.11f, 0.94f);
    colors[ImGuiCol_TitleBg] = ImVec4(0.06f, 0.07f, 0.09f, 1.00f);
    colors[ImGuiCol_TitleBgActive] = ImVec4(0.12f, 0.14f, 0.18f, 1.00f);
    colors[ImGuiCol_FrameBg] = ImVec4(0.14f, 0.16f, 0.20f, 1.00f);
    colors[ImGuiCol_Header] = ImVec4(0.18f, 0.21f, 0.27f, 1.00f);
    colors[ImGuiCol_HeaderHovered] = ImVec4(0.22f, 0.26f, 0.33f, 1.00f);
    colors[ImGuiCol_Button] = ImVec4(0.18f, 0.21f, 0.27f, 1.00f);
    colors[ImGuiCol_ButtonHovered] = ImVec4(0.24f, 0.28f, 0.36f, 1.00f);
}

} // namespace mukki::ui
