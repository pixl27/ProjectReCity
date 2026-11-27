cmake_minimum_required(VERSION 3.24)
project(gta6rt LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
  message(FATAL_ERROR "gta6rt must be built for x64 (run from an x64 Native Tools prompt or vcvars64.bat)")
endif()

if(WIN32)
  # The d3d12_xs shim and its tests use version 1.2 root signatures (D3D12_ROOT_SIGNATURE_DESC2), which the Windows
  # SDK declares from 10.0.26100 on. Stop here with the fix instead of failing halfway through the build.
  include(CheckCXXSourceCompiles)
  check_cxx_source_compiles([=[
    #include <d3d12.h>
    int main() { D3D12_VERSIONED_ROOT_SIGNATURE_DESC d{}; d.Version = D3D_ROOT_SIGNATURE_VERSION_1_2;
                 return static_cast<int>(sizeof(d.Desc_1_2)); }
  ]=] GTA6RT_SDK_HAS_ROOT_SIGNATURE_1_2)
  if(NOT GTA6RT_SDK_HAS_ROOT_SIGNATURE_1_2)
    if(CMAKE_VS_WINDOWS_TARGET_PLATFORM_VERSION)
      set(_gta6rt_sdk "${CMAKE_VS_WINDOWS_TARGET_PLATFORM_VERSION}")
    elseif(DEFINED ENV{WindowsSDKVersion})
      string(REGEX REPLACE "[\\/]+$" "" _gta6rt_sdk "$ENV{WindowsSDKVersion}")
    else()
      set(_gta6rt_sdk "an older one")
    endif()
    message(FATAL_ERROR
      "gta6rt needs the Windows SDK 10.0.26100 or newer (version 1.2 root signatures); "
      "this build uses ${_gta6rt_sdk}. Install it with "
      "`winget install Microsoft.WindowsSDK.10.0.26100` or the Visual Studio Installer, then delete the build "
      "folder and configure again.")
  endif()

  # Root-signature flag fix, shared by the d3d12_xs shim and its tests.
  add_library(gta6rt_d3d12_xs_rootsig STATIC
    shims/d3d12_xs_rootsig.cpp
  )
  target_include_directories(gta6rt_d3d12_xs_rootsig PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/shims)
  target_compile_definitions(gta6rt_d3d12_xs_rootsig PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
  target_compile_options(gta6rt_d3d12_xs_rootsig PRIVATE /W4 /permissive- /utf-8)

  # Portable root-signature model + packer to user SGPRs (no d3d12.h, no windows.h).
  add_library(gta6rt_root_signature STATIC
    src/root_signature.cpp
  )
  target_include_directories(gta6rt_root_signature PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_compile_definitions(gta6rt_root_signature PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
  target_compile_options(gta6rt_root_signature PRIVATE /W4 /permissive- /utf-8)

  # Portable contract test: model, packer, canonical codec and RTS0 hypothesis.
  # Needs neither d3d12.lib nor the game; runs on Windows CI like the rest here.
  add_executable(root_signature_test
    test/root_signature_test.cpp
  )
  target_link_libraries(root_signature_test PRIVATE gta6rt_root_signature)
  target_compile_definitions(root_signature_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
  target_compile_options(root_signature_test PRIVATE /W4 /permissive- /utf-8)
  target_link_options(root_signature_test PRIVATE /OPT:NOREF)

  # XTaskQueue/XAsync from microsoft/libHttpClient (MIT), vendored in third_party/xtaskqueue; shims/xtaskqueue/pch.h
  # stands in for libHttpClient's pch.h so only the task queue builds. Used by the xgameruntime shim.
  set(XTASKQUEUE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/xtaskqueue)
  add_library(gta6rt_xtaskqueue STATIC
    ${XTASKQUEUE_DIR}/Source/Task/TaskQueue.cpp
    ${XTASKQUEUE_DIR}/Source/Task/AsyncLib.cpp
    ${XTASKQUEUE_DIR}/Source/Task/ThreadPool_win32.cpp
    ${XTASKQUEUE_DIR}/Source/Task/WaitTimer_win32.cpp
  )
  target_include_directories(gta6rt_xtaskqueue
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/shims/xtaskqueue ${XTASKQUEUE_DIR}/Source/Task ${XTASKQUEUE_DIR}/Source/Common
    PUBLIC ${XTASKQUEUE_DIR}/Include)
  target_compile_definitions(gta6rt_xtaskqueue PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
  target_compile_options(gta6rt_xtaskqueue PRIVATE /utf-8)  # third-party code: its own warning level
  target_link_libraries(gta6rt_xtaskqueue PUBLIC ole32)

  # Reserves one loader-assigned module TLS index for mapped images. Loaded from the executable's directory by
  # Loader::initialize_tls; its trivial DllMain only keeps the TLS anchor in the image. Built in the same directory as
  # the executables that map images, because the loader looks for it next to the running EXE.
  add_library(gta6rt_tls_host SHARED src/tls_host.cpp)
  target_compile_options(gta6rt_tls_host PRIVATE /W4 /permissive- /utf-8 /GS-)
  target_compile_definitions(gta6rt_tls_host PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
  set_target_properties(gta6rt_tls_host PROPERTIES
    OUTPUT_NAME gta6rt_tls_host
    PREFIX ""
    MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>"
  )

  add_executable(gta6rt
    src/main.cpp
    src/pe_loader.cpp
    shims/registry.cpp
    shims/xmem.cpp
    shims/d3d12_xs.cpp
    shims/d3d12_xs_device.cpp
    shims/xgameruntime.cpp
    shims/process.cpp
  )
  add_dependencies(gta6rt gta6rt_tls_host)
  target_compile_definitions(gta6rt PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX UNICODE _UNICODE)
  target_compile_options(gta6rt PRIVATE /W4 /permissive- /utf-8)
  # Static shim tables register themselves at startup; keep the linker from dropping their objects.
  # GTA6RT_HOST_BASE keeps the loader's own image off the default x64 EXE base (0x140000000), where a mapped EXE
  # without relocations wants to go. Windows moves the EXE with ASLR anyway; Wine loads it at its preferred base.
  set(GTA6RT_HOST_BASE /BASE:0x10000000000)
  target_link_options(gta6rt PRIVATE /OPT:NOREF ${GTA6RT_HOST_BASE})
  target_link_libraries(gta6rt PRIVATE gta6rt_d3d12_xs_rootsig gta6rt_xtaskqueue psapi)

  # Root-signature forwarding test: needs a real PC d3d12.lib to serialize the synthetic descriptors.
  # find_library searches %LIB% (a Developer Prompt), not the Windows SDK the Visual Studio generator links
  # against, so also look in that SDK's um/x64 folder.
  set(_sdk_lib_hints)
  if(DEFINED ENV{WindowsSdkDir} AND DEFINED ENV{WindowsSDKLibVersion})
    list(APPEND _sdk_lib_hints "$ENV{WindowsSdkDir}Lib/$ENV{WindowsSDKLibVersion}um/x64")
  endif()
  get_filename_component(_kits_root
    "[HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Windows Kits\\Installed Roots;KitsRoot10]" ABSOLUTE)
  if(CMAKE_VS_WINDOWS_TARGET_PLATFORM_VERSION AND EXISTS "${_kits_root}")
    list(APPEND _sdk_lib_hints "${_kits_root}/Lib/${CMAKE_VS_WINDOWS_TARGET_PLATFORM_VERSION}/um/x64")
  endif()
  find_library(D3D12_LIBRARY NAMES d3d12 HINTS ${_sdk_lib_hints})
  find_library(DXGI_LIBRARY NAMES dxgi HINTS ${_sdk_lib_hints})
  if(D3D12_LIBRARY)
    add_executable(root_sig_test
      test/root_sig_test.cpp
      shims/registry.cpp
      shims/d3d12_xs.cpp
      shims/d3d12_xs_device.cpp
    )
    target_link_libraries(root_sig_test PRIVATE gta6rt_d3d12_xs_rootsig ${D3D12_LIBRARY})
    target_compile_definitions(root_sig_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
    target_compile_options(root_sig_test PRIVATE /W4 /permissive- /utf-8)
    target_link_options(root_sig_test PRIVATE /OPT:NOREF)

    # RTS0 parser oracle: synthetic descriptors only, serialized by the real PC
    # d3d12.dll and parsed back by rt::rootsig::parse_rts0. A layout mismatch
    # fails here with the blob bytes, never silently in a draw.
    add_executable(root_signature_oracle_test
      test/root_signature_oracle_test.cpp
    )
    target_link_libraries(root_signature_oracle_test PRIVATE gta6rt_root_signature ${D3D12_LIBRARY})
    target_compile_definitions(root_signature_oracle_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
    target_compile_options(root_signature_oracle_test PRIVATE /W4 /permissive- /utf-8)
    target_link_options(root_signature_oracle_test PRIVATE /OPT:NOREF)

    # Xbox-ABI device proxy test: WARP PC device, no game executable. Needs DXGI for the WARP adapter.
    if(DXGI_LIBRARY)
      add_executable(device_proxy_test
        test/device_proxy_test.cpp
        shims/registry.cpp
        shims/d3d12_xs.cpp
        shims/d3d12_xs_device.cpp
      )
      target_link_libraries(device_proxy_test PRIVATE gta6rt_d3d12_xs_rootsig ${D3D12_LIBRARY} ${DXGI_LIBRARY})
      target_compile_definitions(device_proxy_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
      target_compile_options(device_proxy_test PRIVATE /W4 /permissive- /utf-8)
      target_link_options(device_proxy_test PRIVATE /OPT:NOREF)
    else()
      message(STATUS "dxgi.lib not found: device_proxy_test is not built")
    endif()
  else()
    message(STATUS "d3d12.lib not found: root_sig_test and device_proxy_test are not built")
  endif()

  # xmem shim ABI test: resolves the registered xmem.dll exports itself, so xmem.cpp and registry.cpp are compiled
  # into this executable instead of reusing gta6rt's objects (which would duplicate their symbols).
  add_executable(xmem_test
    test/xmem_test.cpp
    shims/xmem.cpp
    shims/registry.cpp
  )
  target_include_directories(xmem_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/shims)
  target_compile_definitions(xmem_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
  target_compile_options(xmem_test PRIVATE /W4 /permissive- /utf-8)
  target_link_options(xmem_test PRIVATE /OPT:NOREF)
  target_link_libraries(xmem_test PRIVATE psapi)

  # Offline GDK runtime shim (XGameRuntime, XSystem, XTaskQueue, XAsync), driven through the shim registry the way
  # a title calls it at boot. Like xmem_test it compiles its own registry.cpp.
  add_executable(xgameruntime_test
    test/xgameruntime_test.cpp
    shims/xgameruntime.cpp
    shims/registry.cpp
  )
  target_include_directories(xgameruntime_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/shims)
  target_compile_definitions(xgameruntime_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
  target_compile_options(xgameruntime_test PRIVATE /W4 /permissive- /utf-8)
  target_link_options(xgameruntime_test PRIVATE /OPT:NOREF)
  target_link_libraries(xgameruntime_test PRIVATE gta6rt_xtaskqueue)

  # GameInput shim on SDL3 (zlib licence): any controller SDL supports, DualSense trigger effects and lightbar.
  # Nothing is downloaded unless asked. The shim is built when SDL3 is available:
  #   - an installed SDL3 3.2+ that find_package finds (SDL3_DIR / CMAKE_PREFIX_PATH), or
  #   - -DRECITY_SDL3_FETCH=ON: download the pinned release below once at configure time (hash-checked) and build it
  #     static, or -DFETCHCONTENT_SOURCE_DIR_SDL3=<extracted SDL3-3.4.20> to build that copy without downloading.
  # Otherwise, or with -DRECITY_SDL3=OFF, gta6rt is built without the shim and the loader binds GameInputCreate to the
  # host's GameInput.dll or the unimplemented thunk as before. docs/gameinput.md has the details.
  option(RECITY_SDL3 "Build the GameInput shim when SDL3 is available" ON)
  option(RECITY_SDL3_FETCH "Download the pinned SDL3 release if no installed SDL3 is found" OFF)
  if(RECITY_SDL3)
    add_subdirectory(cmake/installed-sdl3)
    if(TARGET recity_installed_sdl3)
      set(_recity_sdl3 recity_installed_sdl3)
    elseif(RECITY_SDL3_FETCH OR FETCHCONTENT_SOURCE_DIR_SDL3)
      include(FetchContent)
      FetchContent_Declare(SDL3
        URL https://github.com/libsdl-org/SDL/releases/download/release-3.4.20/SDL3-3.4.20.tar.gz
        URL_HASH SHA256=f3bb06f55bed9557227a700af2ea62e49497ebdf58037732a731a5fb3ef8581f
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
      # Only what gamepads need: the joystick, HIDAPI, haptic, sensor and power code, in a static library.
      set(SDL_SHARED OFF CACHE BOOL "" FORCE)
      set(SDL_STATIC ON CACHE BOOL "" FORCE)
      set(SDL_TEST_LIBRARY OFF CACHE BOOL "" FORCE)
      set(SDL_TESTS OFF CACHE BOOL "" FORCE)
      set(SDL_EXAMPLES OFF CACHE BOOL "" FORCE)
      set(SDL_INSTALL OFF CACHE BOOL "" FORCE)
      set(SDL_DISABLE_INSTALL ON CACHE BOOL "" FORCE)
      foreach(_sdl_off SDL_AUDIO SDL_VIDEO SDL_GPU SDL_RENDER SDL_CAMERA SDL_DIALOG SDL_TRAY)
        set(${_sdl_off} OFF CACHE BOOL "" FORCE)
      endforeach()
      FetchContent_MakeAvailable(SDL3)
      set(_recity_sdl3 SDL3::SDL3-static)
    endif()
  endif()
  if(RECITY_SDL3 AND _recity_sdl3)
    target_sources(gta6rt PRIVATE shims/gameinput.cpp)
    target_link_libraries(gta6rt PRIVATE ${_recity_sdl3})

    # GameInput through the shim registry on SDL virtual joysticks: no controller, no game data.
    add_executable(gameinput_test test/gameinput_test.cpp shims/gameinput.cpp shims/registry.cpp)
    # gameinput_probe: the same shim on a real controller, run by hand (not a CTest).
    add_executable(gameinput_probe test/gameinput_probe.cpp shims/gameinput.cpp shims/registry.cpp)
    foreach(_gi gameinput_test gameinput_probe)
      target_include_directories(${_gi} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/shims)
      target_compile_definitions(${_gi} PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
      target_compile_options(${_gi} PRIVATE /W4 /permissive- /utf-8)
      target_link_options(${_gi} PRIVATE /OPT:NOREF)
      target_link_libraries(${_gi} PRIVATE ${_recity_sdl3})
    endforeach()
  else()
    if(RECITY_SDL3)
      message(STATUS "GameInput shim not built: no installed SDL3 found. Set SDL3_DIR, or pass -DRECITY_SDL3_FETCH=ON "
                     "to download the pinned SDL3 release (see docs/gameinput.md)")
    else()
      message(STATUS "GameInput shim not built (RECITY_SDL3=OFF)")
    endif()
  endif()

else()
  message(STATUS "gta6rt requires Windows; only Vulkan tools are available on Linux")
endif()

include(CTest)
if(BUILD_TESTING)
  add_executable(pipeline_cache_file_test test/pipeline_cache_file_test.cpp)
  add_test(NAME pipeline_cache_file COMMAND pipeline_cache_file_test)
  # XDXR build descriptions (docs/xdxr.md); compared with d3d12.h where the Windows SDK provides it.
  add_executable(xdxr_descs_test test/xdxr_descs_test.cpp)
  add_test(NAME xdxr_descs COMMAND xdxr_descs_test)
  if(WIN32)
    # Loader tests. They build their own PE images, so they need neither the game nor any other input.
    add_executable(loader_test
      test/loader_test.cpp
      src/pe_loader.cpp
      shims/registry.cpp
    )
    target_compile_definitions(loader_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX UNICODE _UNICODE)
    target_compile_options(loader_test PRIVATE /W4 /permissive- /utf-8)
    target_link_options(loader_test PRIVATE /OPT:NOREF ${GTA6RT_HOST_BASE})
    add_dependencies(loader_test gta6rt_tls_host)
    add_test(NAME loader COMMAND loader_test)
    # PE loader tests use two locally built fixtures: a DLL with exports, imports and relocations, and an EXE that
    # imports it so Loader::load exercises recursive module loading. They contain no game data.
    set(TINY_FIXTURE_DIR "${CMAKE_CURRENT_BINARY_DIR}/fixtures/$<CONFIG>")

    add_library(tiny_dll SHARED test/fixtures/tiny_dll.cpp)
    target_compile_definitions(tiny_dll PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
    target_compile_options(tiny_dll PRIVATE /W4 /permissive- /utf-8 /GS-)
    # tiny_fault_caught faults inside its own __try. MSVC always catches that; clang-cl only with asynchronous
    # exceptions, otherwise the access violation escapes and the unwind-registration check fails.
    target_compile_options(tiny_dll PRIVATE $<$<CXX_COMPILER_ID:Clang>:/EHa>)
    set_target_properties(tiny_dll PROPERTIES
      OUTPUT_NAME tiny_dll
      PREFIX ""
      RUNTIME_OUTPUT_DIRECTORY "${TINY_FIXTURE_DIR}"
      ARCHIVE_OUTPUT_DIRECTORY "${TINY_FIXTURE_DIR}"
      MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>"
    )

    add_executable(tiny_exe test/fixtures/tiny_exe.cpp)
    target_link_libraries(tiny_exe PRIVATE tiny_dll)
    target_compile_definitions(tiny_exe PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
    target_compile_options(tiny_exe PRIVATE /W4 /permissive- /utf-8 /GS-)
    set_target_properties(tiny_exe PROPERTIES
      OUTPUT_NAME tiny_exe
      RUNTIME_OUTPUT_DIRECTORY "${TINY_FIXTURE_DIR}"
      MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>"
    )

    # pe_loader.cpp needs the shim registry for shims::find/make_unimplemented_thunk. The xmem and d3d12_xs tables
    # are linked too so the optional RECITY_GAME_DIR run classifies the shipped modules' imports exactly like
    # gta6rt does (no d3d12.lib is needed: d3d12_xs resolves its host entry points at run time).
    add_executable(pe_loader_test
      test/pe_loader_test.cpp
      src/pe_loader.cpp
      shims/registry.cpp
      shims/xmem.cpp
      shims/d3d12_xs.cpp
      shims/d3d12_xs_device.cpp
    )
    target_include_directories(pe_loader_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/src
      ${CMAKE_CURRENT_SOURCE_DIR}/shims
    )
    target_compile_definitions(pe_loader_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
    target_compile_options(pe_loader_test PRIVATE /W4 /permissive- /utf-8)
    target_link_options(pe_loader_test PRIVATE /OPT:NOREF)
    target_link_libraries(pe_loader_test PRIVATE gta6rt_d3d12_xs_rootsig psapi)
    add_dependencies(pe_loader_test tiny_dll tiny_exe gta6rt_tls_host)

    # `gta6rt --run` rehearsal: a stand-in game EXE (CRT startup, statics, heap, files, threads) that imports
    # xgameruntime.dll from a link-time stub; under --run the shim registry answers those imports instead.
    add_library(xgameruntime_stub SHARED test/fixtures/xgameruntime_stub.cpp)
    target_compile_definitions(xgameruntime_stub PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
    set_target_properties(xgameruntime_stub PROPERTIES
      OUTPUT_NAME xgameruntime
      PREFIX ""
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/fixtures/game/$<CONFIG>"
      ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/fixtures/game/$<CONFIG>"
    )
    add_executable(tiny_game test/fixtures/tiny_game.cpp)
    target_link_libraries(tiny_game PRIVATE xgameruntime_stub)
    target_compile_definitions(tiny_game PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX _CRT_SECURE_NO_WARNINGS)  # plain fopen, as games use
    target_compile_options(tiny_game PRIVATE /W4 /permissive- /utf-8 /EHsc)
    set_target_properties(tiny_game PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/fixtures/game/$<CONFIG>"
    )
    add_test(NAME pe_loader COMMAND pe_loader_test "$<TARGET_FILE:tiny_dll>" "$<TARGET_FILE:tiny_exe>")

    # Synthetic, hand-built images through Loader::load: range checks, wrapping RVAs, import-table termination and
    # base relocations (HIGHLOW included). No fixture DLL is needed; the images are written to a temporary folder.
    add_executable(test_pe_loader_bounds
      test/test_pe_loader_bounds.cpp
      src/pe_loader.cpp
      shims/registry.cpp
      shims/xmem.cpp
      shims/d3d12_xs.cpp
      shims/d3d12_xs_device.cpp
    )
    target_include_directories(test_pe_loader_bounds PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/src
      ${CMAKE_CURRENT_SOURCE_DIR}/shims
    )
    target_compile_definitions(test_pe_loader_bounds PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
    target_compile_options(test_pe_loader_bounds PRIVATE /W4 /permissive- /utf-8)
    target_link_options(test_pe_loader_bounds PRIVATE /OPT:NOREF)
    target_link_libraries(test_pe_loader_bounds PRIVATE gta6rt_d3d12_xs_rootsig psapi)
    add_dependencies(test_pe_loader_bounds gta6rt_tls_host)
    add_test(NAME pe_loader_bounds COMMAND test_pe_loader_bounds)

    # Static TLS fixture: a DLL with __declspec(thread) and a TLS callback. It is ordinary test code; no game data
    # is involved. loader_tls_test needs gta6rt_tls_host.dll next to its own EXE, because that is where
    # Loader::initialize_tls looks for the reserved module TLS index.
    add_library(tiny_tls SHARED test/fixtures/tiny_tls.cpp)
    target_compile_definitions(tiny_tls PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
    target_compile_options(tiny_tls PRIVATE /W4 /permissive- /utf-8 /GS-)
    set_target_properties(tiny_tls PROPERTIES
      OUTPUT_NAME tiny_tls
      PREFIX ""
      RUNTIME_OUTPUT_DIRECTORY "${TINY_FIXTURE_DIR}"
      ARCHIVE_OUTPUT_DIRECTORY "${TINY_FIXTURE_DIR}"
    )

    # pe_loader.cpp needs the shim registry for shims::find/make_unimplemented_thunk. The fixtures import only host
    # DLLs, so no shim tables are linked.
    add_executable(loader_tls_test
      test/loader_tls_test.cpp
      src/pe_loader.cpp
      shims/registry.cpp
    )
    target_include_directories(loader_tls_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_compile_definitions(loader_tls_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
    target_compile_options(loader_tls_test PRIVATE /W4 /permissive- /utf-8 /EHsc)
    add_dependencies(loader_tls_test tiny_tls gta6rt_tls_host)
    add_test(NAME loader_tls
      COMMAND loader_tls_test "$<TARGET_FILE:tiny_tls>"
    )

    # RECITY_UNIMPLEMENTED=log|fail|stop: builds its own image with two imports that bind to nothing, calls them in
    # process (log, fail) and in child processes (default/env parsing, stop). No game files, no GPU.
    add_executable(unimplemented_policy_test
      test/unimplemented_policy_test.cpp
      src/pe_loader.cpp
      shims/registry.cpp
    )
    # The register-preservation check needs a caller that sets rbx/rbp/rdi/rsi/r12-r15 and xmm6-15, which MSVC x64
    # cannot do inline: test/fixtures/abi_probe.asm (MASM). Without ml64 the rest of the test still runs.
    find_program(RECITY_ML64 ml64)  # CMake would otherwise pick the 32-bit ml with the Ninja generator
    if(RECITY_ML64)
      set(CMAKE_ASM_MASM_COMPILER "${RECITY_ML64}")
      enable_language(ASM_MASM)
      target_sources(unimplemented_policy_test PRIVATE test/fixtures/abi_probe.asm)
      target_compile_definitions(unimplemented_policy_test PRIVATE RECITY_HAVE_ABI_PROBE=1)
    else()
      message(STATUS "ml64 not found: unimplemented_policy_test skips the register-preservation check")
    endif()
    target_include_directories(unimplemented_policy_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_compile_definitions(unimplemented_policy_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX UNICODE _UNICODE _CRT_SECURE_NO_WARNINGS)
    target_compile_options(unimplemented_policy_test PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/W4 /permissive- /utf-8 /EHsc>)
    target_link_options(unimplemented_policy_test PRIVATE /OPT:NOREF)
    add_dependencies(unimplemented_policy_test gta6rt_tls_host)
    add_test(NAME unimplemented_policy COMMAND unimplemented_policy_test)
  endif()

  if(WIN32 AND TARGET root_sig_test)
    add_test(NAME root_sig_test COMMAND root_sig_test)
  endif()
  if(TARGET device_proxy_test)
    add_test(NAME device_proxy COMMAND device_proxy_test)
  endif()
  if(TARGET root_sig_test)
    add_test(NAME root_sig COMMAND root_sig_test)
  endif()
  add_executable(command_list_test test/command_list_test.cpp)
  add_test(NAME command_list COMMAND command_list_test)
  add_executable(resource_tiling_test test/resource_tiling_test.cpp)
  add_test(NAME resource_tiling COMMAND resource_tiling_test)
  add_executable(descriptor_table_test test/descriptor_table_test.cpp)
  add_test(NAME descriptor_table COMMAND descriptor_table_test)
  if(WIN32)
    # Optional: skips unless RECITY_GAME_DIR and RECITY_MOVIE are set.
    add_executable(bink_decode_test
      test/bink_decode_test.cpp
      src/pe_loader.cpp
      shims/registry.cpp
      shims/xmem.cpp
    )
    target_compile_definitions(bink_decode_test PRIVATE
      WIN32_LEAN_AND_MEAN NOMINMAX UNICODE _UNICODE _CRT_SECURE_NO_WARNINGS)
    target_compile_options(bink_decode_test PRIVATE /W4 /permissive- /EHsc /utf-8)
    target_link_options(bink_decode_test PRIVATE /OPT:NOREF)
    target_link_libraries(bink_decode_test PRIVATE psapi)
    add_dependencies(bink_decode_test gta6rt_tls_host)
    add_test(NAME bink_decode COMMAND bink_decode_test)

    # Optional: skips unless RECITY_GAME_DIR is set. Drives the package's FSR DLL on the Xbox-ABI device proxy.
    if(D3D12_LIBRARY AND DXGI_LIBRARY)
      add_executable(fsr_dispatch_test
        test/fsr_dispatch_test.cpp
        src/pe_loader.cpp
        shims/registry.cpp
        shims/xmem.cpp
        shims/d3d12_xs.cpp
        shims/d3d12_xs_device.cpp
      )
      target_compile_definitions(fsr_dispatch_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX _CRT_SECURE_NO_WARNINGS)
      target_compile_options(fsr_dispatch_test PRIVATE /W4 /permissive- /EHsc /utf-8)
      target_link_options(fsr_dispatch_test PRIVATE /OPT:NOREF)
      target_link_libraries(fsr_dispatch_test PRIVATE gta6rt_d3d12_xs_rootsig ${D3D12_LIBRARY} ${DXGI_LIBRARY} psapi)
      add_test(NAME fsr_dispatch COMMAND fsr_dispatch_test)

      # Optional: skips unless RECITY_GAME_DIR is set. Constructs the package's XboxBvhBuilder on the device proxy.
      add_executable(bvh_builder_test
        test/bvh_builder_test.cpp
        src/pe_loader.cpp
        shims/registry.cpp
        shims/xmem.cpp
        shims/d3d12_xs.cpp
        shims/d3d12_xs_device.cpp
      )
      target_compile_definitions(bvh_builder_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX _CRT_SECURE_NO_WARNINGS)
      target_compile_options(bvh_builder_test PRIVATE /W4 /permissive- /EHsc /utf-8)
      target_link_options(bvh_builder_test PRIVATE /OPT:NOREF)
      target_link_libraries(bvh_builder_test PRIVATE gta6rt_d3d12_xs_rootsig ${D3D12_LIBRARY} ${DXGI_LIBRARY} psapi)
      add_test(NAME bvh_builder COMMAND bvh_builder_test)
    endif()

    # Optional: skips unless RECITY_GAME_DIR and RECITY_MOVIE_AUDIO are set.
    add_executable(bink_audio_test
      test/bink_audio_test.cpp
      src/pe_loader.cpp
      shims/registry.cpp
      shims/xmem.cpp
    )
    target_compile_definitions(bink_audio_test PRIVATE
      WIN32_LEAN_AND_MEAN NOMINMAX UNICODE _UNICODE _CRT_SECURE_NO_WARNINGS)
    target_compile_options(bink_audio_test PRIVATE /W4 /permissive- /EHsc /utf-8)
    target_link_options(bink_audio_test PRIVATE /OPT:NOREF)
    target_link_libraries(bink_audio_test PRIVATE psapi)
    add_dependencies(bink_audio_test gta6rt_tls_host)
    add_test(NAME bink_audio COMMAND bink_audio_test)
  endif()
  if(WIN32 AND TARGET xmem_test)
    add_test(NAME xmem COMMAND xmem_test)
  endif()
  if(WIN32 AND TARGET xgameruntime_test)
    add_test(NAME xgameruntime COMMAND xgameruntime_test)
  endif()
  if(WIN32 AND TARGET gameinput_test)
    add_test(NAME gameinput COMMAND gameinput_test)
  elseif(WIN32)
    # Built without SDL3: listed as "Not Run (Disabled)" so it is visible that the shim was not tested.
    add_test(NAME gameinput COMMAND ${CMAKE_COMMAND} -E echo "built without SDL3")
    set_tests_properties(gameinput PROPERTIES DISABLED TRUE)
  endif()
  if(WIN32 AND TARGET gta6rt AND TARGET tiny_game)
    # The stand-in game passes only with its own arguments; with others it exits 1, which --run passes through.
    add_test(NAME run_tiny_game COMMAND gta6rt --run $<TARGET_FILE:tiny_game> -- hello 42)
    add_test(NAME run_tiny_game_exit_code COMMAND gta6rt --run $<TARGET_FILE:tiny_game> -- wrong args)
    set_tests_properties(run_tiny_game_exit_code PROPERTIES WILL_FAIL TRUE)
    # Mode flags are exclusive: two of them print the usage before anything is mapped or run.
    add_test(NAME gta6rt_rejects_two_modes COMMAND gta6rt --dry-run --run $<TARGET_FILE:tiny_game>)
    add_test(NAME gta6rt_rejects_init_and_run COMMAND gta6rt --init --run $<TARGET_FILE:tiny_game>)
    set_tests_properties(gta6rt_rejects_two_modes gta6rt_rejects_init_and_run PROPERTIES
                         PASS_REGULAR_EXPRESSION "usage: gta6rt" FAIL_REGULAR_EXPRESSION "running |module\\(s\\) mapped")
  endif()
  if(WIN32 AND TARGET root_signature_test)
    add_test(NAME root_signature COMMAND root_signature_test)
  endif()
  if(WIN32 AND TARGET root_signature_oracle_test)
    add_test(NAME root_signature_oracle COMMAND root_signature_oracle_test)
  endif()
endif()

if(WIN32)
option(RECITY_D3D12_SHIM_TESTS "Build the standalone D3D12 root-signature shim test" OFF)
if(RECITY_D3D12_SHIM_TESTS)
  enable_testing()
  add_executable(d3d12_xs_root_signature_test
    test/d3d12_xs_root_signature_test.cpp
    shims/d3d12_xs.cpp
    shims/registry.cpp
    shims/d3d12_xs_device.cpp
  )
  target_include_directories(d3d12_xs_root_signature_test PRIVATE shims)
  target_compile_definitions(d3d12_xs_root_signature_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
  target_compile_options(d3d12_xs_root_signature_test PRIVATE /W4 /permissive- /EHsc /utf-8)
  target_link_options(d3d12_xs_root_signature_test PRIVATE /OPT:NOREF)
  target_link_libraries(d3d12_xs_root_signature_test PRIVATE gta6rt_d3d12_xs_rootsig d3d12 dxgi)
  add_test(NAME d3d12_xs_root_signature COMMAND d3d12_xs_root_signature_test)
endif()
endif()

# GPU test and pipeline tools load Vulkan at run time, so only the Vulkan headers are needed at build time.
find_package(Vulkan QUIET)
if(NOT TARGET Vulkan::Headers)
  find_path(Vulkan_INCLUDE_DIR
    NAMES vulkan/vulkan.h
    HINTS "$ENV{VULKAN_SDK}/Include" "$ENV{VULKAN_SDK}/include"
  )
  if(Vulkan_INCLUDE_DIR AND EXISTS "${Vulkan_INCLUDE_DIR}/vulkan/vulkan.h")
    add_library(Vulkan::Headers INTERFACE IMPORTED)
    set_target_properties(Vulkan::Headers PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${Vulkan_INCLUDE_DIR}"
    )
  endif()
endif()
if(TARGET Vulkan::Headers)
  foreach(tool vk_harness vk_texture_test vk_compute_test vk_store_test vk_pipeline_check vk_command_replay_test
          vk_bindless_test vk_descriptor_bench)
    add_executable(${tool} test/${tool}.cpp)
    target_link_libraries(${tool} PRIVATE Vulkan::Headers ${CMAKE_DL_LIBS})
    if(WIN32)
      target_compile_definitions(${tool} PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
    endif()
  endforeach()
  # D3D12 COM objects over the Vulkan backend. Their method stubs are generated from the SDK's d3d12.h at build
  # time (tools/gen_d3d12_stubs.py), so nothing derived from the SDK header is checked in.
  find_package(Python3 COMPONENTS Interpreter QUIET)
  if(WIN32 AND Python3_FOUND)
    set(RECITY_STUBS ${CMAKE_CURRENT_BINARY_DIR}/generated/d3d12_stubs.h)
    add_custom_command(
      OUTPUT ${RECITY_STUBS}
      COMMAND Python3::Interpreter -I ${CMAKE_CURRENT_SOURCE_DIR}/../tools/gen_d3d12_stubs.py --out ${RECITY_STUBS}
              ID3D12GraphicsCommandList8 ID3D12CommandQueue ID3D12CommandAllocator ID3D12Fence1
              ID3D12Device11 ID3D12DescriptorHeap ID3D12Resource ID3D12QueryHeap
      DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/../tools/gen_d3d12_stubs.py
      COMMENT "Generating D3D12 COM stubs"
      VERBATIM)
    add_library(gta6rt_d3d12_com STATIC shims/d3d12_com.cpp shims/d3d12_vulkan_device.cpp shims/d3d12_xs_device.cpp
                ${RECITY_STUBS})   # d3d12_vulkan_device.cpp reads the PC device behind the proxy (#224)
    target_include_directories(gta6rt_d3d12_com PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/generated)
    target_link_libraries(gta6rt_d3d12_com PUBLIC Vulkan::Headers)
    target_compile_definitions(gta6rt_d3d12_com PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
    target_compile_options(gta6rt_d3d12_com PRIVATE /W4 /permissive- /utf-8)

    add_executable(d3d12_com_test test/d3d12_com_test.cpp)
    target_link_libraries(d3d12_com_test PRIVATE gta6rt_d3d12_com)
    target_compile_definitions(d3d12_com_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
    target_compile_options(d3d12_com_test PRIVATE /W4 /permissive- /utf-8)

    # GCL8's QueryInterface and its one method need an allocator and a list, not a Vulkan device.
    add_executable(d3d12_gcl8_test test/d3d12_gcl8_test.cpp)
    target_link_libraries(d3d12_gcl8_test PRIVATE gta6rt_d3d12_com)
    target_compile_definitions(d3d12_gcl8_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
    target_compile_options(d3d12_gcl8_test PRIVATE /W4 /permissive- /utf-8)
    add_test(NAME d3d12_gcl8 COMMAND d3d12_gcl8_test)
    add_test(NAME d3d12_trace COMMAND Python3::Interpreter -I
             "${CMAKE_CURRENT_SOURCE_DIR}/test/check_d3d12_trace.py" "$<TARGET_FILE:d3d12_gcl8_test>")

    # The Vulkan device behind the Xbox-ABI proxy, end to end.
    add_executable(d3d12_vulkan_device_test test/d3d12_vulkan_device_test.cpp)
    target_link_libraries(d3d12_vulkan_device_test PRIVATE gta6rt_d3d12_com)
    target_compile_definitions(d3d12_vulkan_device_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
    target_compile_options(d3d12_vulkan_device_test PRIVATE /W4 /permissive- /utf-8)
    add_test(NAME d3d12_vulkan_device COMMAND d3d12_vulkan_device_test)
    set_tests_properties(d3d12_vulkan_device PROPERTIES SKIP_RETURN_CODE 77)

    # ClearUnorderedAccessViewUint/Float on buffers and textures, read back through D3D12.
    add_executable(d3d12_uav_clear_test test/d3d12_uav_clear_test.cpp)
    target_link_libraries(d3d12_uav_clear_test PRIVATE gta6rt_d3d12_com)
    target_compile_definitions(d3d12_uav_clear_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
    target_compile_options(d3d12_uav_clear_test PRIVATE /W4 /permissive- /utf-8)

    # The 1/2/3 method variants (D3D-28) against the methods they wrap, on the Vulkan device directly.
    add_executable(d3d12_method_variants_test test/d3d12_method_variants_test.cpp)
    target_link_libraries(d3d12_method_variants_test PRIVATE gta6rt_d3d12_com)
    target_compile_definitions(d3d12_method_variants_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
    target_compile_options(d3d12_method_variants_test PRIVATE /W4 /permissive- /utf-8)

    # CreateSampler2 (D3D-28) against CreateSampler, and its refusals against the host's PC runtime.
    add_executable(d3d12_sampler2_test test/d3d12_sampler2_test.cpp)
    target_link_libraries(d3d12_sampler2_test PRIVATE gta6rt_d3d12_com)
    target_compile_definitions(d3d12_sampler2_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
    target_compile_options(d3d12_sampler2_test PRIVATE /W4 /permissive- /utf-8)

    # Descriptors through the proxy: copies between heaps, what texture views and samplers write into their slots,
    # and (where glslangValidator is found) a compute shader the test runs on the device's bindless set, which
    # finds a T# and an S# through the hash table and samples. The shader is GLSL, compiled here.
    add_executable(d3d12_descriptor_test test/d3d12_descriptor_test.cpp)
    target_link_libraries(d3d12_descriptor_test PRIVATE gta6rt_d3d12_com)
    target_compile_definitions(d3d12_descriptor_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
    target_compile_options(d3d12_descriptor_test PRIVATE /W4 /permissive- /utf-8)
    if(NOT Vulkan_GLSLANG_VALIDATOR_EXECUTABLE)
      find_program(Vulkan_GLSLANG_VALIDATOR_EXECUTABLE glslangValidator HINTS "$ENV{VULKAN_SDK}/Bin" "$ENV{VULKAN_SDK}/bin")
    endif()
    if(Vulkan_GLSLANG_VALIDATOR_EXECUTABLE)
      set(RECITY_TEST_CS_SAMPLE ${CMAKE_CURRENT_BINARY_DIR}/generated/cs_sample.spv)
      add_custom_command(
        OUTPUT ${RECITY_TEST_CS_SAMPLE}
        COMMAND ${Vulkan_GLSLANG_VALIDATOR_EXECUTABLE} -V --target-env vulkan1.3 -o ${RECITY_TEST_CS_SAMPLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/test/glsl/cs_sample.comp
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/test/glsl/cs_sample.comp
        COMMENT "Compiling the descriptor test shader"
        VERBATIM)
      target_sources(d3d12_descriptor_test PRIVATE ${RECITY_TEST_CS_SAMPLE})
      target_compile_definitions(d3d12_descriptor_test PRIVATE RECITY_TEST_CS_SAMPLE_SPV="${RECITY_TEST_CS_SAMPLE}")
    else()
      message(STATUS "glslangValidator not found: d3d12_descriptor_test skips its shader")
    endif()

    # Differential regression: run explicitly (requires Vulkan hardware and Windows WARP).
    add_executable(d3d12_clear_test test/d3d12_clear_test.cpp)
    target_link_libraries(d3d12_clear_test PRIVATE gta6rt_d3d12_com d3d12 dxgi)
    target_compile_definitions(d3d12_clear_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
    target_compile_options(d3d12_clear_test PRIVATE /W4 /permissive- /utf-8)

    # Synthetic typed-view readbacks against Windows WARP; run explicitly on a Vulkan GPU.
    add_executable(d3d12_typeless_rtv_test test/d3d12_typeless_rtv_test.cpp)
    target_link_libraries(d3d12_typeless_rtv_test PRIVATE gta6rt_d3d12_com d3d12 dxgi)
    target_compile_definitions(d3d12_typeless_rtv_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
    target_compile_options(d3d12_typeless_rtv_test PRIVATE /W4 /permissive- /utf-8)

    # ID3D12Resource::GetDesc against the host runtime and WARP; run explicitly on a Vulkan GPU.
    add_executable(d3d12_resource_desc_test test/d3d12_resource_desc_test.cpp)
    target_link_libraries(d3d12_resource_desc_test PRIVATE gta6rt_d3d12_com d3d12 dxgi)
    target_compile_definitions(d3d12_resource_desc_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
    target_compile_options(d3d12_resource_desc_test PRIVATE /W4 /permissive- /utf-8)

    # ID3D12Resource::GetHeapProperties and ID3D12DescriptorHeap::GetDesc (node masks, resolved heap flags) against the
    # host runtime and WARP; run explicitly on a Vulkan GPU.
    add_executable(d3d12_node_mask_test test/d3d12_node_mask_test.cpp)
    target_link_libraries(d3d12_node_mask_test PRIVATE gta6rt_d3d12_com d3d12 dxgi)
    target_compile_definitions(d3d12_node_mask_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
    target_compile_options(d3d12_node_mask_test PRIVATE /W4 /permissive- /utf-8)

    # The loader can create this device instead of the host's (D3D12XboxCreateDevice, RECITY_D3D12=vulkan).
    if(TARGET gta6rt)
      target_link_libraries(gta6rt PRIVATE gta6rt_d3d12_com)
      target_compile_definitions(gta6rt PRIVATE RECITY_VULKAN_DEVICE)
    endif()
    # bvh_builder_test can put the package's XboxBvhBuilder on this device too (RECITY_BVH_DEVICE=vulkan).
    if(TARGET bvh_builder_test)
      target_link_libraries(bvh_builder_test PRIVATE gta6rt_d3d12_com)
      target_compile_definitions(bvh_builder_test PRIVATE RECITY_VULKAN_DEVICE)
    endif()
  else()
    message(STATUS "Python 3 or Windows missing: the D3D12 COM objects and d3d12_com_test are not built")
  endif()

  # Needs only the Vulkan headers, not a GPU.
  if(BUILD_TESTING)
    add_executable(typeless_rtv_test test/typeless_rtv_test.cpp)
    target_link_libraries(typeless_rtv_test PRIVATE Vulkan::Headers)
    if(WIN32)
      target_compile_definitions(typeless_rtv_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
    endif()
    add_test(NAME typeless_rtv COMMAND typeless_rtv_test)
    add_executable(pipeline_cache_snapshot_test test/pipeline_cache_snapshot_test.cpp)
    target_link_libraries(pipeline_cache_snapshot_test PRIVATE Vulkan::Headers)
    add_test(NAME pipeline_cache_snapshot COMMAND pipeline_cache_snapshot_test)
    add_executable(clear_color_test test/clear_color_test.cpp)
    target_link_libraries(clear_color_test PRIVATE Vulkan::Headers)
    if(WIN32)
      target_compile_definitions(clear_color_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
    endif()
    add_test(NAME clear_color COMMAND clear_color_test)
    add_executable(d3d12_state_map_test test/d3d12_state_map_test.cpp)
    target_link_libraries(d3d12_state_map_test PRIVATE Vulkan::Headers)
    if(WIN32)
      target_compile_definitions(d3d12_state_map_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
    endif()
    add_test(NAME d3d12_state_map COMMAND d3d12_state_map_test)
    # GFX10 T#/S#/V# descriptor encoding and its Vulkan mappings (src/gfx10_descriptors.h).
    add_executable(gfx10_descriptors_test test/gfx10_descriptors_test.cpp)
    target_link_libraries(gfx10_descriptors_test PRIVATE Vulkan::Headers)
    add_test(NAME gfx10_descriptors COMMAND gfx10_descriptors_test)
    add_test(NAME vk_descriptor_bench_selftest COMMAND vk_descriptor_bench --self-test)
  endif()
else()
  message(STATUS "Vulkan headers not found: GPU test tools, vk_pipeline_check, d3d12_state_map_test and "
                 "gfx10_descriptors_test are not built")
endif()
