# Running the GPU tests

The GPU tests are small hand-written RDNA2 programs in `runtime/test/shaders`. They are translated to SPIR-V and run
on your GPU, and the results are compared with what the RDNA2 hardware gives. They need no game files, and every
result helps: post it in [the GPU results issue](https://github.com/pixl27/ProjectReCity/issues/2), one comment per GPU and driver.
If your GPU is already covered there, testing an open pull request from the review queue in the [coordination
issue][claims] helps more.

This page collects the setup steps and the problems testers have run into so far.

[claims]: https://github.com/pixl27/ProjectReCity/issues/1

## Measuring graphics subgroup size

With `vk_harness`, `llvm-mc`, `spirv-as` and `spirv-val` available, run a synthetic wave64 ballot in each of VS, PS and GS:

```powershell
python -I runtime/test/shaders/graphics_subgroup/check.py runtime/build/vk_harness.exe runtime/build/graphics-subgroup
```

The driver selects the actual subgroup width. The test reads `SubgroupSize` and the ballot results in the same
shader invocation, first with the default pipeline and then with `VKH_VARYING_SUBGROUP_STAGE` set to the tested stage.
The expected result from 64 active lanes is 32 upper-half votes and 64 total votes. A physical wave32 sees 0 upper-half
votes and 32 total votes when full. The output also reports partially occupied subgroups. Contradictory subgroup
measurements are marked inconclusive and produce a nonzero exit status with `--strict`. The geometry test uses a
separate direct Vulkan ballot to check whether a contradiction also occurs without RDNA2 translation.

## What you need

- **Python 3.11+.** `scripts/run_gpu_tests.py` needs at least 3.10.
- **`llvm-mc` with the AMDGPU target** (LLVM 21+). The official LLVM Windows release has no AMDGPU target. On
  Windows, use one of:
  - `scripts\build_llvm_mc.ps1`, which builds a native `llvm-mc.exe` (needs git and the Visual Studio C++ tools);
  - conda-forge's `llvm-tools` package, whose Windows build includes the AMDGPU target;
  - `llvm-21` in WSL, with `LLVM_MC` set to the whole command:
    `LLVM_MC="wsl -d Ubuntu -e /usr/lib/llvm-21/bin/llvm-mc"`.

  On Linux and macOS, the distribution's or Homebrew's `llvm-mc` on `PATH` is enough, if it is 21 or newer.

  An older `llvm-mc` is not enough for everything. Ubuntu 24.04's `llvm-18` (18.1.3) runs `run_gpu_tests.py`,
  `valu_fuzz.py --family mem` and `lds_fuzz.py`, but rejects `op_sel` on the 16-bit integer VOP3 forms that
  `valu_fuzz.py --family f16` generates (`v_ashrrev_i16_e64 ... op_sel:[0,0,0]`: "not a valid operand"), so no f16
  program assembles. 23.1.6 assembles them.
- **SPIRV-Tools** (`spirv-as`, `spirv-val`): part of the Vulkan SDK, or a distribution package.
- **A C++20 compiler, CMake and the Vulkan headers** to build the test programs:
  - Windows: Visual Studio 2022 with "Desktop development with C++", the Windows SDK 10.0.26100 or newer, and the
    Vulkan SDK.
  - Linux: CMake 3.24+, GCC or Clang, and the Vulkan headers and loader. The headers should be recent enough
    for `VK_KHR_maintenance8`; older ones still build, but skip what needs it.
  - macOS: `brew install llvm molten-vk vulkan-headers vulkan-loader spirv-tools vulkan-tools`. The test programs
    load MoltenVK directly; set `RECITY_VULKAN_LOADER` to use the Vulkan loader instead.

## Build the test programs

Windows, from a "Developer PowerShell for VS 2022" or a plain shell:

```powershell
scripts\build_runtime.ps1            # CMake + Ninja into runtime\build
```

If you configure with CMake's Visual Studio generator instead, the programs end up in `runtime\build\Release` (or
`RelWithDebInfo`); pass that folder to `--build` below.

Linux and macOS:

```sh
cmake -S runtime -B runtime/build/linux
cmake --build runtime/build/linux --parallel
```

On macOS, add `-DVulkan_INCLUDE_DIR=/opt/homebrew/include` to the first command.

## Run them

The whole suite, assembled, translated and run in one go:

```sh
python -I scripts/run_gpu_tests.py --build runtime/build/linux     # Windows: the default runtime\build
```

`--filter <text>` runs only matching tests, and `--work <dir>` keeps the intermediate files.

The two-step form, which `CONTRIBUTING.md` describes:

```sh
python -I runtime/test/shaders/build.py <out dir>
python -I runtime/test/shaders/check.py <out dir> <folder with the test programs>
```

## GPUs with 32-wide subgroups (NVIDIA, Intel, Apple)

The game's shaders are mostly wave64. A GPU that can't run 64-wide compute subgroups skips the wave64 tests
(`cs_wavemask`, `cs_wave64_ballot`, `cs_wm_*`, `cs_w64x_*`, `ms_w64x_ballot`); that's expected, not a failure.

To test the wave64 emulation that makes them exact on these GPUs, set `RECITY_W64_EMULATE=1`, **for both
`build.py` and `check.py`**: `build.py` translates with emulation only when it's set. If you set it only for
`check.py`, the shaders are checked against the emulated expectations without being emulated, and every wave64 test
fails.

## Without a GPU: Mesa llvmpipe

Mesa's llvmpipe is a software Vulkan driver (`mesa-vulkan-drivers` on Debian and Ubuntu). CI runs the suite on it
for every pull request (`.github/workflows/runtime.yml`), and you can do the same on Linux without a GPU:

```sh
sudo apt-get install -y libvulkan-dev mesa-vulkan-drivers vulkan-tools spirv-tools llvm-18
cmake -S runtime -B runtime/build/linux && cmake --build runtime/build/linux --parallel
python -I scripts/run_gpu_tests.py --build runtime/build/linux --llvm-mc /usr/lib/llvm-18/bin/llvm-mc
RECITY_W64_EMULATE=1 python -I scripts/run_gpu_tests.py --build runtime/build/linux \
    --llvm-mc /usr/lib/llvm-18/bin/llvm-mc --filter cs_
```

With Mesa 25.2 (llvmpipe, LLVM 20) the first run passes 52 of 88 and skips 36, the second passes 37 of 39 and
skips 2. The skips are expected: llvmpipe has 8-wide subgroups (so every wave64 test skips without emulation, and
the 32-lane tests `ps_bitmask_perm`, `cs_wqm_w32` and `cs_w32_lane_wrap` skip in both runs), no
`VK_KHR_fragment_shader_barycentric` (`ps_bary`, `ps_uniform_branch`, `ps_scc_mask32`) and no
`VK_KHR_maintenance8` (`ps_texoff`). With
`RECITY_W64_EMULATE=1` the mesh tests abort inside llvmpipe's compiler ("LLVM ERROR: Cannot emit physreg copy
instruction"), which is why the second run is limited to compute. llvmpipe is not a substitute for a result from a
real GPU in [the GPU results issue](https://github.com/pixl27/ProjectReCity/issues/2): it catches regressions, not hardware
differences.

## Choosing the GPU

With more than one GPU, pick one through the Vulkan loader, for example
`VK_LOADER_DRIVERS_SELECT=*amd*` or `VK_LOADER_DRIVERS_SELECT=nv-vk*` (or `VK_DRIVER_FILES=<path to the ICD json>`).
`vulkaninfo --summary` shows which devices the loader then sees.

## Known problems

| What you see | Cause | What to do |
|---|---|---|
| Windows refuses to start freshly built test programs ("An Application Control policy has blocked this file"), and the runner stops with a traceback | Smart App Control on Windows 11 | Turn Smart App Control off for the run |
| Validation errors without a VUID, or messages from a layer you didn't ask for | An implicit Vulkan layer from other software (Rockstar Social Club, ReShade, MangoHud, an overlay) | Set `VK_LOADER_LAYERS_DISABLE=~implicit~` |
| Every test crashes inside `libnvidia-eglcore.so` | NVIDIA's 535 server driver (Ubuntu 24.04's default) | Use driver 615 or newer |
| `build.py`: 0 translated, reason `.text` | An old `build.py` with LLVM 19 | Update to current `main` |
| `vk_harness` doesn't compile: `VK_KHR_maintenance8` names unknown | Vulkan headers older than the SDK the code was written against | Update to current `main` (the code now builds without it), or use current Khronos Vulkan-Headers; conda-forge's package is too old |
| `ps_bary` and `ps_uniform_branch` fail by about 0.9 | The GPU runs pixel shaders wave64 without fragment barycentrics (GCN3 and RDNA1 on RADV), or an old AMD Windows driver (22.40) | Current `main` skips them on such devices; on Windows, update the driver |
| `gs_tri` and `ms_tri` fail on a Mac | MoltenVK has no geometry stage and no `VK_EXT_mesh_shader` | Current `main` reports them as skipped |
| CMake stops with "needs the Windows SDK 10.0.26100 or newer", or MSVC reports `'Desc_1_2': is not a member of 'D3D12_VERSIONED_ROOT_SIGNATURE_DESC'` | A Windows SDK older than 10.0.26100 (10.0.22621 is the common one), which has no version 1.2 root signatures | `winget install Microsoft.WindowsSDK.10.0.26100`, or add it in the Visual Studio Installer, then delete `runtime/build` and configure again |
| `run_gpu_tests.py` fails on a `dict \| None` annotation | Python older than 3.10 | Use Python 3.11+ |
| A GPU test that uses `image_load` (or lighting built from real shaders) reads zeros from its image; @Silliana saw every texel except pixel (0,0) come back zero in their own layer (#603, not reproduced here) | In this runtime the shader turns the T# words into an index through the descriptor hash table (`runtime/src/descriptor_table.h`: a miss gives index 0), and index 0 is a 1×1 image of zeros (`runtime/src/vulkan_bindless.h`), so a hand-built or zeroed T# that was never registered reads zeros even at (0,0). The translator does not compare coordinates with the T# size, and `cs_imgload.asm` only uses its T# words as a hash key | Register the descriptor through the runtime (for D3D12, create it with `CreateShaderResourceView`) so its T# hits the table, with the image's real width and height (stored as size − 1, see `runtime/src/gfx10_descriptors.h`) |

## Posting your result

Use the template at the top of [the GPU results issue](https://github.com/pixl27/ProjectReCity/issues/2): GPU, driver, subgroup
range, OS, the commit you built, the tool versions and the counts each script printed. Post counts and pass/fail
lines only, never shaders or logs from the game. A failure that isn't in the table above gets its own issue, linked
from your entry.
