# Controller input: the GameInput shim on SDL3

The Xbox build reads controllers through Microsoft's GameInput API. `runtime/shims/gameinput.cpp` answers
`GameInputCreate` with its own implementation of that API, backed by [SDL3](https://libsdl.org) gamepads, so
every controller SDL supports works the way an Xbox pad would: DualSense, DualShock 4, Switch Pro, Xbox pads, and the
rest of SDL's controller database. On a DualSense the impulse-trigger rumble becomes adaptive-trigger effects, and
the lightbar colour can be set.

Without the shim the loader would bind `GameInputCreate` to the host's `GameInput.dll` if there is one.
That works for Xbox pads only, and gives no DualSense
features. The game executable is not available yet, so whether it imports GameInput this way, and under which
module, is unverified (see [VERIFY](#verify)).

## Building

The shim is optional, and the build downloads nothing unless asked. SDL3 is zlib-licensed. CMake builds the shim when
SDL3 is available in one of these ways:

- **Installed static SDL3**, 3.2 or newer, found by `find_package(SDL3)`: set `SDL3_DIR` or `CMAKE_PREFIX_PATH`, for example
  `-DSDL3_DIR=C:/SDL3/cmake`. No flag is needed.
- **`-DRECITY_SDL3_FETCH=ON`**: downloads the pinned release `SDL3-3.4.20.tar.gz` from SDL's GitHub releases once at
  configure time, checks its SHA-256, and builds it static with only the gamepad parts (no audio, video, GPU or
  render). That takes about a minute on the first build. `scripts/build_runtime.ps1 -FetchSdl3` passes it.
- **A local copy of that tarball**: extract it and pass `-DFETCHCONTENT_SOURCE_DIR_SDL3=<path to SDL3-3.4.20>`; it is
  built the same way, without a download.

If the Windows development package provides only the shared library, use either source-build option above.
An installed static library takes precedence over the source build.

Without any of these, or with `-DRECITY_SDL3=OFF`, configure prints a message and `gta6rt` is built without the shim:
it has no `GameInputCreate`, as before, so the loader binds it to the host's `GameInput.dll` or the unimplemented
thunk. `gameinput_test` and `gameinput_probe` are not built, and CTest lists `gameinput` as not run (disabled) rather
than failing.

`scripts/build_runtime.ps1` (what `make_game.py` runs) builds the shim only on its CMake path, when Visual Studio's
C++ CMake tools are installed. Its direct-`cl.exe` fallback compiles a fixed file list without the shim and warns
about it.

## Mapping

GameInput's buttons are positional, as on an Xbox pad, and SDL's are too, so the face buttons go by position, not by
label:

| GameInput | SDL3 | DualSense | Switch Pro |
|---|---|---|---|
| `A` | south | Cross | B |
| `B` | east | Circle | A |
| `X` | west | Square | Y |
| `Y` | north | Triangle | X |
| `View` | back | Create | Minus |
| `Menu` | start | Options | Plus |
| shoulders, stick clicks, d-pad | same | L1/R1, L3/R3, d-pad | L/R, stick clicks, d-pad |
| guide button callback | guide | PS | Home |

The DualSense touchpad click, mute button and paddles are not in GameInput's gamepad state; they are in the
controller-button array (`GetControllerButtonState`, indexed in SDL's `SDL_GamepadButton` order).

- Sticks: -1..1 with +Y up (SDL reports +Y down; the shim flips it). Triggers: 0..1. No deadzone is applied, as on
  console: the title applies its own.
- `GetUiNavigationState` is derived from the pad: A accept, B cancel, X/Y context 1/2, stick clicks context 3/4, d-pad
  or left stick past half for up/down/left/right, shoulders for page left/right, triggers past half for page up/down,
  right stick past half for scrolling, Menu and View as themselves.
- Controller axes (`GetControllerAxisState`): LX, LY, RX, RY, LT, RT in 0..1, sticks at rest at 0.5.
- Every pad is reported as family `GameInputFamilyXboxOne` with its real USB vendor and product id.
- Player LEDs (and SDL's default lightbar colour) follow connection order: the first pad is player 1, and a pad
  plugged in later takes the lowest free slot.

## Rumble and triggers

`IGameInputDevice::SetRumbleState` takes four motor levels, 0..1. The two body motors go to `SDL_RumbleGamepad` on
every pad that has rumble. The two trigger motors depend on the pad:

- Xbox One and Series pads (and anything else SDL reports trigger rumble for): `SDL_RumbleGamepadTriggers`, the same
  impulse triggers as on console.
- DualSense: the trigger level becomes an adaptive-trigger effect sent with `SDL_SendGamepadEffect`, chosen by
  `RECITY_DS_TRIGGERS`.
- Others (DualShock 4, Switch Pro): the trigger levels are dropped and the device reports no trigger motors.

**This is an approximation.** The Xbox title only sends an impulse-trigger vibration level per frame. The PS5 version
of a game has effects authored per weapon and vehicle (weapon resistance, a click at a trigger position, and so on);
none of that exists in the Xbox build, so the shim can only turn the level it gets into a generic effect. Expect it to
feel like the Xbox signal played through the DualSense's trigger motors, not like the PS5 release.

The DualSense effect is SDL's 47-byte DualSense effects block (`DS5EffectsState_t` in SDL's `SDL_hidapi_ps5.c`, the
format `test/testcontroller.c` uses): byte 0 holds the valid flags `0x04 | 0x08` (right and left trigger effect),
bytes 10-20 the right trigger effect and bytes 21-31 the left one. The effect modes are the three SDL's test program
uses:

| `RECITY_DS_TRIGGERS` | level 0 | level > 0 |
|---|---|---|
| `rumble` (default) | `05` (effect off) | `06 <freq> <level*255> 00`: vibration at `RECITY_DS_TRIGGER_FREQ` Hz from the start of the pull |
| `resistance` | `05` | `01 00 <level*255>`: constant resistance over the whole pull |
| `off` | nothing is sent; the triggers stay as they are | |

An effect is sent only when its bytes change, so a steady level costs one report.

## Environment variables

| variable | default | effect |
|---|---|---|
| `RECITY_DS_TRIGGERS` | `rumble` | DualSense trigger effects: `off`, `rumble` or `resistance` (above) |
| `RECITY_DS_TRIGGER_GAIN` | `1` | scales the trigger level before it becomes an effect, 0..4 |
| `RECITY_DS_TRIGGER_FREQ` | `30` | vibration frequency of the `rumble` mode, 1..255 |
| `RECITY_DS_LIGHTBAR` | unset | `RRGGBB` (hex) sets the DualSense/DualShock 4 lightbar when the pad connects; `off` turns it dark; unset keeps SDL's player colour |
| `RECITY_RUMBLE` | `1` | `0` turns all rumble and trigger output off |
| `RECITY_GAMEINPUT_THREAD` | `1` | `0` stops the background poll thread; only `rt::gameinput::poll()` then reads the pads (tests) |

SDL's own hints (`SDL_JOYSTICK_*`, see SDL's `SDL_hints.h`) also apply, for example `SDL_JOYSTICK_HIDAPI_PS5=0` to
leave a DualSense to the generic driver, or `SDL_GAMECONTROLLERCONFIG` to add a mapping for an unknown pad.

## How it runs

One thread owns SDL: a background thread started by the first `GameInputCreate` that polls every 2 ms. It opens new
pads, closes removed ones, takes a reading of each, and applies rumble requests. Title calls never touch SDL: they read
the latest reading or leave a rumble request, so a call costs a lock and a copy. Callbacks (device, reading, guide
button) run on that thread, as GameInput runs them on its own threads. Once the title creates a dispatcher
(`IGameInput::CreateDispatcher`), callbacks are queued instead, the dispatcher's wait handle is signalled, and
`IGameInputDispatcher::Dispatch` runs them on the title's thread.

The API surface:

| interface | implemented | not implemented |
|---|---|---|
| `IGameInput` | `GetCurrentTimestamp` (µs), `GetCurrentReading`, `GetNextReading`, `RegisterReadingCallback`, `RegisterDeviceCallback` (blocking and async enumeration), `RegisterGuideButtonCallback`, `RegisterKeyboardLayoutCallback` (never fires), `StopCallback`, `UnregisterCallback`, `CreateDispatcher`, `FindDeviceFromId`, `EnableOemDeviceSupport` (no-op), `SetFocusPolicy` (no-op) | `GetPreviousReading` and `GetTemporalReading` (no history kept), `CreateAggregateDevice`, `FindDeviceFromObject/PlatformHandle/PlatformString` |
| `IGameInputReading` | kind, sequence number, timestamp, device, gamepad state, UI navigation state, controller axes and buttons | raw report, keyboard, mouse, touch, motion, arcade/flight stick, racing wheel (all return nothing) |
| `IGameInputDevice` | device info, status, battery, `SetRumbleState` | force feedback and haptic motors (none reported), raw device I/O, power off, input synchronization (no-ops) |
| `IGameInputDispatcher` | `Dispatch`, `OpenWaitHandle` | the quota (the queue is short) |

A device object stays alive for the whole process. When a pad is unplugged its status drops to
`GameInputDeviceNoStatus` and its device callback runs; when the same pad is plugged in again (matched by SDL GUID and
serial or device path) the title gets the same object back, connected again.

## ABI

`runtime/shims/gameinput_abi.h` declares the v0 GameInput API as ReCity's own code: the interfaces in vtable order,
the enums the shim uses and the structs it has to lay out. It was written against Microsoft's public v0 header (the
GDK version, also published as the `Microsoft.GameInput` NuGet package 0.2303.22621.3038). The header is not in the
repository. Its struct offsets were checked by compiling both side by side with `offsetof` asserts for every field the
shim fills (`GameInputDeviceInfo` 320 bytes with `deviceFamily` at 96 and the info pointers from 160,
`GameInputGamepadState` 28 bytes, `GameInputControllerAxisInfo` 48, and the rest in `gameinput_abi.h`).

## Tests

- `gameinput_test` (CTest, no controller): turns SDL's real backends off and drives the shim through
  `rt::shims::find` with SDL virtual joysticks, an Xbox-style pad with trigger motors, a DualSense (Sony
  `054c:0ce6`) and a pad without trigger motors. It checks enumeration and device callbacks, the button, stick,
  trigger and UI-navigation mapping, reading callbacks and sequence numbers, the guide button, motor and trigger
  rumble as SDL receives them, the DualSense effect bytes in each `RECITY_DS_TRIGGERS` mode, the lightbar, a
  dispatcher, async and blocking enumeration, unregistering, and unplugging and replugging.
- `gameinput_probe [--seconds N] [--rumble]` (by hand, with a pad connected): lists the pads, plays each motor and
  trigger for a second with `--rumble`, then prints the live reading.

## VERIFY

These need the game executable or a test on hardware:

- **Which API version and module.** The shim is GameInput v0 and registers `GameInputCreate` under `gameinput.dll`
  and `xgameruntime.dll`. GameInput v1 and later (NuGet 1.x and up) changed the interfaces; if the executable was
  built against one of those, the vtables need a second set.
- **IIDs.** `QueryInterface` answers the v0 IIDs from the public header.
- **Reading history.** `GetNextReading` returns the newest reading when it is newer than the reference, and
  `GetPreviousReading` has no history. A title that replays every input frame between two polls would miss the
  intermediate states.
- **UI navigation mapping** and the **controller-axis convention** (0..1 with sticks at 0.5) are this shim's choice,
  not measured from a console.
- **Battery capacity** is reported as a fraction (remaining 0..1 of full 1), not in device units.
- **`UnregisterCallback` timeout** is not honoured; it waits for a callback that is running.
- **Device family**: every pad is reported as `GameInputFamilyXboxOne`; a title that filters on family sees only
  that.
- **DualSense effect parameters**: the `06` (vibration) and `01` (resistance) byte meanings follow SDL's
  `testcontroller.c`; how strong 255 feels and what frequency suits the Xbox signal needs a session on a real pad.
