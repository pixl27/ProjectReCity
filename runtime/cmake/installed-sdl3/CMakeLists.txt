# Some SDL3 packages define shared and header targets even when the requested
# static component is missing. Keep imported targets in this directory so a
# subsequent source build can define its own SDL3 targets.
set(CMAKE_FIND_PACKAGE_TARGETS_GLOBAL FALSE)
if(NOT TARGET SDL3::SDL3-static)
  find_package(SDL3 3.2 CONFIG QUIET COMPONENTS SDL3-static)
endif()
if(TARGET SDL3::SDL3-static)
  add_library(recity_installed_sdl3 INTERFACE)
  target_link_libraries(recity_installed_sdl3 INTERFACE SDL3::SDL3-static)
endif()
