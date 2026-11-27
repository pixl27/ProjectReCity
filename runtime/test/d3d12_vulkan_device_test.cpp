// The Vulkan-backed ID3D12Device behind the Xbox-ABI device proxy, end to end, on a GPU. Everything goes through
// the proxy the way the game will call it: device methods (forwarded slots), descriptor heaps from slot 14 with
// start handles read in the Xbox shape (RAX), a texture from slot 76 with the Xbox layout word, committed
// resources, views, command lists, a queue, a fence and query heaps. Results are read back through D3D12 itself
// (CopyTextureRegion into a readback heap, then Map). GetCopyableFootprints, GetResourceAllocationInfo and
// GetResourceTiling are compared with the host PC runtime's, and so are copies (d3d12_copy_compare.h).
// WriteToSubresource/ReadFromSubresource run on devices of their own, once per copy path. No shaders or game data.
// Set RECITY_VK_VALIDATION=1 to run under VK_LAYER_KHRONOS_validation; any validation error fails the test.
#include "vk_test_device.h"
#include "../shims/d3d12_com.h"
#include "../shims/d3d12_xs_device.h"
#include "d3d12_copy_compare.h"  // D3D-46: copies compared with the host runtime
#include "../src/vulkan_memory.h"  // D3D-96: resizable_bar

#include <dxgi1_4.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

namespace {

// The Xbox call shapes measured in the prior art (docs/d3d12_vulkan.md section 7).
using XboxHeapHandleFn = std::uint64_t(WINAPI*)(void* heap);  // heap slots 9/10: 8-byte handle in RAX
using XboxCreateResourceFn = HRESULT(WINAPI*)(void* device, const D3D12_HEAP_PROPERTIES*, D3D12_HEAP_FLAGS,
                                              const D3D12_RESOURCE_DESC*, D3D12_RESOURCE_STATES,
                                              const D3D12_CLEAR_VALUE*, const IID*, void**);  // device slot 76

template<class T>
struct Ref {
    T* p = nullptr;
    ~Ref() { reset(); }
    void reset() {
        if (p) p->Release();
        p = nullptr;
    }
    T* operator->() const { return p; }
};

void* slot(void* object, unsigned index) { return (*reinterpret_cast<void***>(object))[index]; }

D3D12_RESOURCE_DESC texture_desc(UINT width, UINT height, DXGI_FORMAT format, D3D12_RESOURCE_FLAGS flags,
                                 UINT16 mips = 1, UINT16 layers = 1) {
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width = width;
    desc.Height = height;
    desc.DepthOrArraySize = layers;
    desc.MipLevels = mips;
    desc.Format = format;
    desc.SampleDesc.Count = 1;
    desc.Flags = flags;
    return desc;
}

D3D12_RESOURCE_DESC buffer_desc(UINT64 bytes) {
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = bytes;
    desc.Height = 1;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    return desc;
}

D3D12_RESOURCE_DESC volume_desc(UINT16 mips) {
    auto desc = texture_desc(8, 6, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_NONE, mips, 16);
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE3D;
    return desc;
}

D3D12_TEXTURE_COPY_LOCATION subresource(ID3D12Resource* resource, UINT index) {
    D3D12_TEXTURE_COPY_LOCATION location{};
    location.pResource = resource;
    location.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    location.SubresourceIndex = index;
    return location;
}

D3D12_TEXTURE_COPY_LOCATION footprint(ID3D12Resource* buffer, const D3D12_PLACED_SUBRESOURCE_FOOTPRINT& placed) {
    D3D12_TEXTURE_COPY_LOCATION location{};
    location.pResource = buffer;
    location.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    location.PlacedFootprint = placed;
    return location;
}

// Places one subresource of `resource` in a buffer at the next aligned offset after `cursor`.
D3D12_PLACED_SUBRESOURCE_FOOTPRINT place(ID3D12Device* device, ID3D12Resource* resource, UINT index,
                                         UINT64& cursor) {
    const D3D12_RESOURCE_DESC desc = resource->GetDesc();
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT layout{};
    UINT64 total = 0;
    const UINT64 base = (cursor + D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT - 1) / D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT *
                        D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT;
    device->GetCopyableFootprints(&desc, index, 1, base, &layout, nullptr, nullptr, &total);
    cursor = base + total;
    return layout;
}

std::array<std::uint8_t, 4> pattern(UINT x, UINT y) {
    return {std::uint8_t(x * 16 + y), std::uint8_t(255 - y * 16), std::uint8_t(x * y), 255};
}

D3D12_RESOURCE_BARRIER transition(ID3D12Resource* resource, D3D12_RESOURCE_STATES before,
                                  D3D12_RESOURCE_STATES after) {
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = resource;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    barrier.Transition.StateBefore = before;
    barrier.Transition.StateAfter = after;
    return barrier;
}

std::uint8_t texel_byte(UINT x, UINT y, UINT z, UINT i) { return std::uint8_t(x * 31 + y * 7 + z * 61 + i * 3 + 1); }

// WriteToSubresource/ReadFromSubresource on `device`. Textures in a CPU-visible custom heap copy on the host when the
// device has VK_EXT_host_image_copy, through a staging submit when it doesn't; main runs this once per path. Covers
// zero-filled first reads, boxes with pitches unlike the texture's, a mip, a volume, BC1 blocks, a default-heap
// texture (always staged), the rejected calls, and both directions between the CPU copies and GPU copies.
int cpu_copies(ID3D12Device* device, bool bc, const char* path) {
    const D3D12_HEAP_PROPERTIES cpu_heap{D3D12_HEAP_TYPE_CUSTOM, D3D12_CPU_PAGE_PROPERTY_WRITE_BACK, D3D12_MEMORY_POOL_L0};
    const D3D12_HEAP_PROPERTIES default_heap{D3D12_HEAP_TYPE_DEFAULT};
    const D3D12_HEAP_PROPERTIES upload_heap{D3D12_HEAP_TYPE_UPLOAD};
    const D3D12_HEAP_PROPERTIES readback_heap{D3D12_HEAP_TYPE_READBACK};
    const auto create = [&](const D3D12_HEAP_PROPERTIES& heap, const D3D12_RESOURCE_DESC& desc,
                            D3D12_RESOURCE_STATES state, Ref<ID3D12Resource>& out) {
        return device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, state, nullptr, IID_PPV_ARGS(&out.p));
    };
    unsigned bad = 0;
    const auto expect_bytes = [&](const char* what, UINT x, UINT y, UINT z, const std::uint8_t* got,
                                  const std::array<std::uint8_t, 4>& want) {
        if (std::memcmp(got, want.data(), 4) == 0) return;
        if (bad++ < 8)
            std::fprintf(stderr, "%s, %s (%u,%u,%u): got %02x %02x %02x %02x, want %02x %02x %02x %02x\n", path, what,
                         x, y, z, got[0], got[1], got[2], got[3], want[0], want[1], want[2], want[3]);
    };
    const auto texel = [](UINT x, UINT y, UINT z) {
        return std::array<std::uint8_t, 4>{texel_byte(x, y, z, 0), texel_byte(x, y, z, 1), texel_byte(x, y, z, 2),
                                           texel_byte(x, y, z, 3)};
    };
    constexpr std::array<std::uint8_t, 4> zero{};

    Ref<ID3D12Resource> tex, vol, bc1, gpu_only, buffer, upload, readback;
    EXPECT(SUCCEEDED(create(cpu_heap, texture_desc(8, 8, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_NONE, 2),
                            D3D12_RESOURCE_STATE_COMMON, tex)));
    D3D12_RESOURCE_DESC volume = texture_desc(4, 4, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_NONE, 1, 4);
    volume.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE3D;
    EXPECT(SUCCEEDED(create(cpu_heap, volume, D3D12_RESOURCE_STATE_COMMON, vol)));
    if (bc)
        EXPECT(SUCCEEDED(create(cpu_heap, texture_desc(8, 8, DXGI_FORMAT_BC1_UNORM, D3D12_RESOURCE_FLAG_NONE),
                                D3D12_RESOURCE_STATE_COMMON, bc1)));
    EXPECT(SUCCEEDED(create(default_heap, texture_desc(4, 4, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_NONE),
                            D3D12_RESOURCE_STATE_COPY_SOURCE, gpu_only)));
    EXPECT(SUCCEEDED(create(cpu_heap, buffer_desc(4096), D3D12_RESOURCE_STATE_COMMON, buffer)));
    EXPECT(SUCCEEDED(create(upload_heap, buffer_desc(4096), D3D12_RESOURCE_STATE_GENERIC_READ, upload)));
    EXPECT(SUCCEEDED(create(readback_heap, buffer_desc(4096), D3D12_RESOURCE_STATE_COPY_DEST, readback)));

    // A texture in a CPU-visible heap maps only without a pointer; a default-heap texture not at all.
    void* pointer = nullptr;
    EXPECT(tex->Map(0, nullptr, nullptr) == S_OK);
    EXPECT(tex->Map(1, nullptr, nullptr) == S_OK);
    EXPECT(tex->Map(0, nullptr, &pointer) == E_INVALIDARG);
    EXPECT(tex->Map(2, nullptr, nullptr) == E_INVALIDARG);
    EXPECT(gpu_only->Map(0, nullptr, nullptr) == E_INVALIDARG);
    tex->Unmap(1, nullptr);

    // A new texture reads back zero-filled.
    std::vector<std::uint8_t> got(1024, 0xcd);
    EXPECT(SUCCEEDED(tex->ReadFromSubresource(got.data(), 32, 0, 0, nullptr)));
    for (UINT y = 0; y < 8; ++y)
        for (UINT x = 0; x < 8; ++x) expect_bytes("new texture", x, y, 0, &got[y * 32 + x * 4], zero);

    // A 5x4 box from rows 100 bytes apart, read back whole at a 40-byte pitch.
    std::vector<std::uint8_t> src(1024, 0);
    const D3D12_BOX box{2, 1, 0, 7, 5, 1};
    for (UINT y = 0; y < 4; ++y)
        for (UINT x = 0; x < 5; ++x) std::memcpy(&src[y * 100 + x * 4], texel(x + 2, y + 1, 0).data(), 4);
    EXPECT(SUCCEEDED(tex->WriteToSubresource(0, &box, src.data(), 100, 0)));
    std::fill(got.begin(), got.end(), std::uint8_t(0xcd));
    EXPECT(SUCCEEDED(tex->ReadFromSubresource(got.data(), 40, 0, 0, nullptr)));
    for (UINT y = 0; y < 8; ++y)
        for (UINT x = 0; x < 8; ++x) {
            const bool inside = x >= 2 && x < 7 && y >= 1 && y < 5;
            expect_bytes("boxed write", x, y, 0, &got[y * 40 + x * 4], inside ? texel(x, y, 0) : zero);
        }

    // Mip 1 (4x4) written whole, read back as a box.
    for (UINT y = 0; y < 4; ++y)
        for (UINT x = 0; x < 4; ++x) std::memcpy(&src[y * 16 + x * 4], texel(x, y, 1).data(), 4);
    EXPECT(SUCCEEDED(tex->WriteToSubresource(1, nullptr, src.data(), 16, 0)));
    const D3D12_BOX mip_box{1, 1, 0, 3, 4, 1};
    std::fill(got.begin(), got.end(), std::uint8_t(0xcd));
    EXPECT(SUCCEEDED(tex->ReadFromSubresource(got.data(), 8, 0, 1, &mip_box)));
    for (UINT y = 0; y < 3; ++y)
        for (UINT x = 0; x < 2; ++x) expect_bytes("mip 1", x + 1, y + 1, 1, &got[y * 8 + x * 4], texel(x + 1, y + 1, 1));

    // A volume: three slices of a 3x4 box, rows 12 and slices 64 bytes apart; read back whole at other pitches.
    std::fill(src.begin(), src.end(), std::uint8_t(0));
    const D3D12_BOX volume_box{1, 0, 1, 4, 4, 4};
    for (UINT z = 0; z < 3; ++z)
        for (UINT y = 0; y < 4; ++y)
            for (UINT x = 0; x < 3; ++x) std::memcpy(&src[z * 64 + y * 12 + x * 4], texel(x + 1, y, z + 1).data(), 4);
    EXPECT(SUCCEEDED(vol->WriteToSubresource(0, &volume_box, src.data(), 12, 64)));
    std::fill(got.begin(), got.end(), std::uint8_t(0xcd));
    EXPECT(SUCCEEDED(vol->ReadFromSubresource(got.data(), 20, 96, 0, nullptr)));
    for (UINT z = 0; z < 4; ++z)
        for (UINT y = 0; y < 4; ++y)
            for (UINT x = 0; x < 4; ++x)
                expect_bytes("volume", x, y, z, &got[z * 96 + y * 20 + x * 4], x >= 1 && z >= 1 ? texel(x, y, z) : zero);

    // BC1: 2x2 blocks of 8 bytes written whole, then one block replaced through a box.
    if (bc1.p) {
        std::uint8_t blocks[32];
        for (UINT i = 0; i < 32; ++i) blocks[i] = std::uint8_t(0x40 + i);
        EXPECT(SUCCEEDED(bc1->WriteToSubresource(0, nullptr, blocks, 16, 0)));
        const D3D12_BOX corner{4, 4, 0, 8, 8, 1};
        const std::uint8_t replaced[8] = {0xe0, 0xe1, 0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7};
        EXPECT(SUCCEEDED(bc1->WriteToSubresource(0, &corner, replaced, 8, 0)));
        std::memcpy(blocks + 24, replaced, 8);
        std::uint8_t back[64] = {};
        EXPECT(SUCCEEDED(bc1->ReadFromSubresource(back, 32, 0, 0, nullptr)));
        for (UINT row = 0; row < 2; ++row)
            if (std::memcmp(back + row * 32, blocks + row * 16, 16) != 0 && bad++ < 8)
                std::fprintf(stderr, "%s, BC1 block row %u differs\n", path, row);
        const D3D12_BOX cut{2, 0, 0, 8, 4, 1};  // starts inside a block
        EXPECT(bc1->WriteToSubresource(0, &cut, blocks, 16, 0) == E_INVALIDARG);
    }

    // Calls D3D12 rejects.
    const D3D12_BOX outside{0, 0, 0, 9, 1, 1}, empty{2, 2, 0, 2, 3, 1};
    EXPECT(tex->WriteToSubresource(0, &outside, src.data(), 100, 0) == E_INVALIDARG);
    EXPECT(tex->WriteToSubresource(0, &empty, src.data(), 100, 0) == E_INVALIDARG);
    EXPECT(tex->WriteToSubresource(2, nullptr, src.data(), 100, 0) == E_INVALIDARG);  // 2 mips, 1 layer
    EXPECT(tex->WriteToSubresource(0, nullptr, nullptr, 32, 0) == E_INVALIDARG);
    EXPECT(tex->ReadFromSubresource(nullptr, 32, 0, 0, nullptr) == E_INVALIDARG);
    EXPECT(tex->WriteToSubresource(0, nullptr, src.data(), 16, 0) == E_INVALIDARG);  // 32-byte rows 16 apart
    EXPECT(vol->WriteToSubresource(0, nullptr, src.data(), 16, 32) == E_INVALIDARG);  // 64-byte slices 32 apart
    EXPECT(buffer->WriteToSubresource(0, nullptr, src.data(), 16, 0) == E_INVALIDARG);

    // A default-heap texture: D3D12 wants a CPU-visible heap, but the copy still works (and is logged).
    for (UINT y = 0; y < 4; ++y)
        for (UINT x = 0; x < 4; ++x) std::memcpy(&src[y * 16 + x * 4], texel(x, y, 7).data(), 4);
    EXPECT(SUCCEEDED(gpu_only->WriteToSubresource(0, nullptr, src.data(), 16, 0)));

    // GPU copies see the CPU's writes (mip 1, the default-heap texture), and the CPU sees a GPU copy's (a 4x4
    // upload into the bottom left of mip 0, which is in COMMON and promoted).
    D3D12_COMMAND_QUEUE_DESC queue_desc{};
    Ref<ID3D12CommandQueue> queue;
    Ref<ID3D12CommandAllocator> allocator;
    Ref<ID3D12GraphicsCommandList> list;
    Ref<ID3D12Fence> fence;
    EXPECT(SUCCEEDED(device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue.p))));
    EXPECT(SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator.p))));
    EXPECT(SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.p, nullptr,
                                               IID_PPV_ARGS(&list.p))));
    EXPECT(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))));
    std::uint8_t* up = nullptr;
    std::uint8_t* down = nullptr;
    EXPECT(SUCCEEDED(upload->Map(0, nullptr, reinterpret_cast<void**>(&up))));
    EXPECT(SUCCEEDED(readback->Map(0, nullptr, reinterpret_cast<void**>(&down))));
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT square{0, {DXGI_FORMAT_R8G8B8A8_UNORM, 4, 4, 1, 256}};
    for (UINT y = 0; y < 4; ++y)
        for (UINT x = 0; x < 4; ++x) std::memcpy(up + y * 256 + x * 4, texel(x, y + 4, 0).data(), 4);
    std::memset(down, 0xcd, 4096);
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT second = square;
    second.Offset = 2048;
    {
        const auto dst = subresource(tex.p, 0);
        const auto in = footprint(upload.p, square);
        list->CopyTextureRegion(&dst, 0, 4, 0, &in, nullptr);
        const auto out = footprint(readback.p, square);
        const auto mip1 = subresource(tex.p, 1);
        list->CopyTextureRegion(&out, 0, 0, 0, &mip1, nullptr);
        const auto out2 = footprint(readback.p, second);
        const auto staged = subresource(gpu_only.p, 0);
        list->CopyTextureRegion(&out2, 0, 0, 0, &staged, nullptr);
    }
    EXPECT(list->Close() == S_OK);
    ID3D12CommandList* lists[] = {list.p};
    queue->ExecuteCommandLists(1, lists);
    EXPECT(rt::d3d12::last_execute_result(queue.p) == recity::ReplayResult::success);
    EXPECT(SUCCEEDED(queue->Signal(fence.p, 1)) && SUCCEEDED(fence->SetEventOnCompletion(1, nullptr)));
    for (UINT y = 0; y < 4; ++y)
        for (UINT x = 0; x < 4; ++x) {
            expect_bytes("GPU copy of mip 1", x, y, 1, down + y * 256 + x * 4, texel(x, y, 1));
            expect_bytes("GPU copy of the default-heap texture", x, y, 7, down + 2048 + y * 256 + x * 4, texel(x, y, 7));
        }
    const D3D12_BOX bottom_left{0, 4, 0, 4, 8, 1};
    std::fill(got.begin(), got.end(), std::uint8_t(0xcd));
    EXPECT(SUCCEEDED(tex->ReadFromSubresource(got.data(), 16, 0, 0, &bottom_left)));
    for (UINT y = 0; y < 4; ++y)
        for (UINT x = 0; x < 4; ++x) expect_bytes("CPU read of a GPU upload", x, y + 4, 0, &got[y * 16 + x * 4], texel(x, y + 4, 0));
    upload->Unmap(0, nullptr);
    readback->Unmap(0, nullptr);
    EXPECT(!bad);
    std::printf("WriteToSubresource/ReadFromSubresource (%s): boxes, pitches, a mip, a volume%s, rejected calls and "
                "GPU copies both ways matched\n", path, bc1.p ? ", BC1" : "");
    return 0;
}

// The host PC runtime's D3D12 device to compare answers with: the default adapter, or the one RECITY_HOST_ADAPTER
// names ("warp", or part of an adapter's name); false if there is none.
bool create_host_device(Ref<ID3D12Device>& host) {
    copycmp::Ptr<ID3D12Device> device;
    std::string name;
    if (!copycmp::create_host(device, name)) return false;
    host.p = device.p;
    device.p = nullptr;
    return true;
}

bool block_compressed(DXGI_FORMAT format) {
    switch (format) {
    case DXGI_FORMAT_BC1_UNORM:
    case DXGI_FORMAT_BC1_UNORM_SRGB:
    case DXGI_FORMAT_BC2_UNORM:
    case DXGI_FORMAT_BC2_UNORM_SRGB:
    case DXGI_FORMAT_BC3_UNORM:
    case DXGI_FORMAT_BC3_UNORM_SRGB:
    case DXGI_FORMAT_BC4_UNORM:
    case DXGI_FORMAT_BC4_SNORM:
    case DXGI_FORMAT_BC5_UNORM:
    case DXGI_FORMAT_BC5_SNORM:
    case DXGI_FORMAT_BC6H_UF16:
    case DXGI_FORMAT_BC6H_SF16:
    case DXGI_FORMAT_BC7_UNORM:
    case DXGI_FORMAT_BC7_UNORM_SRGB: return true;
    default: return false;
    }
}

// A block-compressed texture whose top mip is not a whole number of 4x4 blocks. The PC runtime only answers
// GetCopyableFootprints for these when D3D12_OPTIONS8.UnalignedBlockTexturesSupported is TRUE (Windows 10 does
// not); it returns -1 for every field otherwise.
bool uses_unaligned_blocks(const D3D12_RESOURCE_DESC& desc) {
    return block_compressed(desc.Format) && (desc.Width % 4 != 0 || desc.Height % 4 != 0);
}

bool host_supports_unaligned_block_textures(ID3D12Device* host) {
    D3D12_FEATURE_DATA_D3D12_OPTIONS8 options{};
    return SUCCEEDED(host->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS8, &options, sizeof(options))) &&
           options.UnalignedBlockTexturesSupported;
}

// ResolveSubresourceRegion is on ID3D12GraphicsCommandList7, which the Windows 10 runtime does not have.
bool host_has_graphics_command_list7(ID3D12Device* host) {
    Ref<ID3D12CommandAllocator> allocator;
    Ref<ID3D12GraphicsCommandList> list;
    Ref<ID3D12GraphicsCommandList7> list7;
    if (FAILED(host->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator.p))))
        return false;
    if (FAILED(host->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.p, nullptr,
                                       IID_PPV_ARGS(&list.p))))
        return false;
    if (FAILED(list->Close())) return false;
    return SUCCEEDED(list->QueryInterface(IID_PPV_ARGS(&list7.p)));
}

int compare_format_info(ID3D12Device* ours, const vktest::Device& vk) {
    VkPhysicalDeviceIDProperties id{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
    VkPhysicalDeviceProperties2 properties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &id};
    vk.instance_fn<PFN_vkGetPhysicalDeviceProperties2>("vkGetPhysicalDeviceProperties2")(vk.physical, &properties);
    LUID luid{};
    static_assert(sizeof(luid) == sizeof(id.deviceLUID));
    std::memcpy(&luid, id.deviceLUID, sizeof(luid));
    copycmp::Ptr<ID3D12Device> host;
    std::string adapter_name;
    if (!id.deviceLUIDValid || !copycmp::try_create_host(host, adapter_name, &luid)) {
        std::printf("FORMAT_INFO: no native D3D12 device on the Vulkan adapter, skipped comparison\n");
        return 0;
    }

    unsigned compared = 0, host_unknown = 0;
    int mismatches = 0;
    const auto compare = [&](DXGI_FORMAT format, UINT size) {
        D3D12_FEATURE_DATA_FORMAT_INFO expected{format, 0xff}, actual = expected;
        HRESULT expected_hr = host->CheckFeatureSupport(D3D12_FEATURE_FORMAT_INFO, &expected, size);
        // Some Windows runtimes predate these single-plane formats. Still require the layer to recognize them.
        if (expected_hr == E_INVALIDARG && size == sizeof(expected) &&
            (format == DXGI_FORMAT_SAMPLER_FEEDBACK_MIN_MIP_OPAQUE ||
             format == DXGI_FORMAT_SAMPLER_FEEDBACK_MIP_REGION_USED_OPAQUE || format == DXGI_FORMAT_A4B4G4R4_UNORM)) {
            expected_hr = S_OK;
            expected.PlaneCount = 1;
            ++host_unknown;
            std::printf("FORMAT_INFO format %u: host returned E_INVALIDARG, checking for one plane\n", UINT(format));
        }
        const HRESULT our_hr = ours->CheckFeatureSupport(D3D12_FEATURE_FORMAT_INFO, &actual, size);
        ++compared;
        if (our_hr != expected_hr || actual.PlaneCount != expected.PlaneCount || actual.Format != format) {
            std::fprintf(stderr, "FORMAT_INFO format %u size %u: HRESULT/planes %08lx/%u, expected %08lx/%u\n",
                         UINT(format), size, static_cast<unsigned long>(our_hr), UINT(actual.PlaneCount),
                         static_cast<unsigned long>(expected_hr), UINT(expected.PlaneCount));
            ++mismatches;
        }
    };
    constexpr UINT size = sizeof(D3D12_FEATURE_DATA_FORMAT_INFO);
    // Include UNKNOWN, holes between format families, and the value after the last named format.
    for (UINT format = 0; format <= UINT(DXGI_FORMAT_A4B4G4R4_UNORM) + 1; ++format)
        compare(static_cast<DXGI_FORMAT>(format), size);
    compare(DXGI_FORMAT_FORCE_UINT, size);
    compare(DXGI_FORMAT_D24_UNORM_S8_UINT, size - 1);
    std::printf("FORMAT_INFO on %s: %u queries, %u newer formats unknown to host, %d mismatches\n",
                adapter_name.c_str(), compared, host_unknown, mismatches);
    return mismatches;
}

// GetCopyableFootprints must match the PC runtime's exactly, since applications size and fill upload buffers from
// it. Unaligned block-compressed cases are skipped when the host runtime cannot answer them, so the rest still
// run. Returns the number of mismatches, or -1 when the host has no D3D12 device to compare with.
int compare_footprints(ID3D12Device* ours) {
    Ref<ID3D12Device> host;
    if (!create_host_device(host)) return -1;
    const bool unaligned_blocks = host_supports_unaligned_block_textures(host.p);
    unsigned skipped = 0;
    struct Case {
        D3D12_RESOURCE_DESC desc;
        UINT subresources;
    };
    const Case cases[] = {
        {volume_desc(5), 5},
        {buffer_desc(300), 1},
        {texture_desc(8, 8, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_NONE, 4, 2), 8},
        {texture_desc(13, 7, DXGI_FORMAT_R9G9B9E5_SHAREDEXP, D3D12_RESOURCE_FLAG_NONE, 4, 2), 8},
        {texture_desc(13, 7, DXGI_FORMAT_B5G5R5A1_UNORM, D3D12_RESOURCE_FLAG_NONE, 4, 2), 8},
        {texture_desc(13, 7, DXGI_FORMAT_A8_UNORM, D3D12_RESOURCE_FLAG_NONE, 4, 2), 8},
        {texture_desc(13, 7, DXGI_FORMAT_R16G16B16A16_FLOAT, D3D12_RESOURCE_FLAG_NONE, 4), 4},
        {texture_desc(10, 6, DXGI_FORMAT_BC1_UNORM, D3D12_RESOURCE_FLAG_NONE, 4), 4},
        {texture_desc(4, 4, DXGI_FORMAT_BC7_UNORM_SRGB, D3D12_RESOURCE_FLAG_NONE, 1, 3), 3},
        {texture_desc(16, 16, DXGI_FORMAT_BC5_UNORM, D3D12_RESOURCE_FLAG_NONE, 5), 5},
        {texture_desc(8, 8, DXGI_FORMAT_D32_FLOAT_S8X24_UINT, D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL, 2), 4},
        {texture_desc(8, 8, DXGI_FORMAT_D24_UNORM_S8_UINT, D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL, 2, 2), 8},
        {texture_desc(300, 5, DXGI_FORMAT_D32_FLOAT_S8X24_UINT, D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL), 2},
        {texture_desc(8, 8, DXGI_FORMAT_D32_FLOAT, D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL), 1},
        {texture_desc(8, 8, DXGI_FORMAT_D16_UNORM, D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL, 4), 4},
        {texture_desc(6, 6, DXGI_FORMAT_R32_TYPELESS, D3D12_RESOURCE_FLAG_NONE), 1},
        {texture_desc(16, 16, DXGI_FORMAT_R32G32_UINT, D3D12_RESOURCE_FLAG_NONE, 3), 3},
        {texture_desc(16, 16, DXGI_FORMAT_R32G32_SINT, D3D12_RESOURCE_FLAG_NONE, 3), 3},
        {texture_desc(16, 16, DXGI_FORMAT_R32G32_TYPELESS, D3D12_RESOURCE_FLAG_NONE, 3), 3},
        {texture_desc(16, 16, DXGI_FORMAT_R16G16B16A16_UINT, D3D12_RESOURCE_FLAG_NONE, 3), 3},
        {texture_desc(16, 16, DXGI_FORMAT_R16G16B16A16_SINT, D3D12_RESOURCE_FLAG_NONE, 3), 3},
        {texture_desc(16, 16, DXGI_FORMAT_R10G10B10A2_UINT, D3D12_RESOURCE_FLAG_NONE, 3), 3},
        {texture_desc(16, 16, DXGI_FORMAT_R10G10B10_XR_BIAS_A2_UNORM, D3D12_RESOURCE_FLAG_NONE, 1), 1},
        {texture_desc(16, 16, DXGI_FORMAT_R16G16_TYPELESS, D3D12_RESOURCE_FLAG_NONE, 3), 3},
        {texture_desc(16, 16, DXGI_FORMAT_R16G16_UINT, D3D12_RESOURCE_FLAG_NONE, 3), 3},
        {texture_desc(16, 16, DXGI_FORMAT_R16G16_SINT, D3D12_RESOURCE_FLAG_NONE, 3), 3},
        {texture_desc(16, 16, DXGI_FORMAT_R8G8_TYPELESS, D3D12_RESOURCE_FLAG_NONE, 3), 3},
        {texture_desc(16, 16, DXGI_FORMAT_R8G8_UINT, D3D12_RESOURCE_FLAG_NONE, 3), 3},
        {texture_desc(16, 16, DXGI_FORMAT_R8G8_SINT, D3D12_RESOURCE_FLAG_NONE, 3), 3},
    };
    int mismatches = 0;
    for (const Case& c : cases) {
        if (!unaligned_blocks && uses_unaligned_blocks(c.desc)) {
            ++skipped;
            continue;
        }
        for (const UINT64 base : {UINT64(0), UINT64(1024)}) {
            std::vector<D3D12_PLACED_SUBRESOURCE_FOOTPRINT> a(c.subresources), b(c.subresources);
            std::vector<UINT> rows_a(c.subresources), rows_b(c.subresources);
            std::vector<UINT64> size_a(c.subresources), size_b(c.subresources);
            UINT64 total_a = 0, total_b = 0;
            ours->GetCopyableFootprints(&c.desc, 0, c.subresources, base, a.data(), rows_a.data(), size_a.data(),
                                        &total_a);
            host->GetCopyableFootprints(&c.desc, 0, c.subresources, base, b.data(), rows_b.data(), size_b.data(),
                                        &total_b);
            for (UINT i = 0; i < c.subresources; ++i) {
                const auto &x = a[i], &y = b[i];
                if (x.Offset == y.Offset && x.Footprint.Format == y.Footprint.Format &&
                    x.Footprint.Width == y.Footprint.Width && x.Footprint.Height == y.Footprint.Height &&
                    x.Footprint.Depth == y.Footprint.Depth && x.Footprint.RowPitch == y.Footprint.RowPitch &&
                    rows_a[i] == rows_b[i] && size_a[i] == size_b[i])
                    continue;
                if (mismatches++ < 24)
                    std::fprintf(stderr,
                                 "footprint format %d base %llu subresource %u: ours offset %llu fmt %d %ux%ux%u pitch %u "
                                 "rows %u row %llu; host offset %llu fmt %d %ux%ux%u pitch %u rows %u row %llu\n",
                                 int(c.desc.Format), (unsigned long long)base, i, (unsigned long long)x.Offset,
                                 int(x.Footprint.Format), x.Footprint.Width, x.Footprint.Height, x.Footprint.Depth,
                                 x.Footprint.RowPitch, rows_a[i], (unsigned long long)size_a[i],
                                 (unsigned long long)y.Offset, int(y.Footprint.Format), y.Footprint.Width,
                                 y.Footprint.Height, y.Footprint.Depth, y.Footprint.RowPitch, rows_b[i],
                                 (unsigned long long)size_b[i]);
            }
            if (total_a != total_b && mismatches++ < 8)
                std::fprintf(stderr, "footprint format %d base %llu: total ours %llu host %llu\n", int(c.desc.Format),
                             (unsigned long long)base, (unsigned long long)total_a, (unsigned long long)total_b);
        }
    }
    if (skipped)
        std::printf("GetCopyableFootprints: %u unaligned block-compressed cases skipped (the host runtime has no "
                    "D3D12_OPTIONS8.UnalignedBlockTexturesSupported)\n", skipped);
    return mismatches;
}

// GetResourceAllocationInfo against the PC runtime. Buffer sizes, every alignment and which descriptions are invalid
// must match it exactly. A texture's size is the driver's choice on both sides, so ours is only checked to be a
// whole number of alignments and at least the texels of its top mip; both sizes are printed. Also checks the
// Device4/Device8 variants' per-resource layout. Returns the number of mismatches, or -1 without a host device.
int compare_allocation_info(ID3D12Device* ours) {
    Ref<ID3D12Device> host;
    if (!create_host_device(host)) return -1;
    const auto with = [](D3D12_RESOURCE_DESC desc, UINT64 alignment, UINT samples = 1) {
        desc.Alignment = alignment;
        desc.SampleDesc.Count = samples;
        return desc;
    };
    constexpr UINT64 k4 = D3D12_SMALL_RESOURCE_PLACEMENT_ALIGNMENT, k64 = D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT;
    constexpr D3D12_RESOURCE_FLAGS none = D3D12_RESOURCE_FLAG_NONE, rt = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
    constexpr DXGI_FORMAT rgba8 = DXGI_FORMAT_R8G8B8A8_UNORM;
    struct Case {
        const char* name;
        D3D12_RESOURCE_DESC desc;
        UINT64 top_mip_bytes;  // 0: the size must match the host's exactly (buffers, invalid descriptions)
    };
    D3D12_RESOURCE_DESC bad_buffer = buffer_desc(256);
    bad_buffer.Height = 2;
    D3D12_RESOURCE_DESC typed_buffer = buffer_desc(256);
    typed_buffer.Format = DXGI_FORMAT_R8_UNORM;
    D3D12_RESOURCE_DESC big_volume = volume_desc(1);
    big_volume.Width = big_volume.Height = 32;
    big_volume.DepthOrArraySize = 8;
    const Case cases[] = {
        {"buffer 300", buffer_desc(300), 0},
        {"buffer 64 KiB", buffer_desc(65536), 0},
        {"buffer 64 KiB + 1", buffer_desc(65537), 0},
        {"buffer aligned 64 KiB", with(buffer_desc(1000), k64), 0},
        {"buffer width 0", buffer_desc(0), 0},
        {"buffer height 2", bad_buffer, 0},
        {"buffer with a format", typed_buffer, 0},
        {"256x256 full chain", texture_desc(256, 256, rgba8, none, 0), 256 * 256 * 4},
        {"16x16 small", with(texture_desc(16, 16, rgba8, none), k4), 16 * 16 * 4},
        {"16x16 small render target", with(texture_desc(16, 16, rgba8, rt), k4), 16 * 16 * 4},
        {"1024x1024 small (too big)", with(texture_desc(1024, 1024, rgba8, none), k4), 1024 * 1024 * 4},
        {"64x64 BC1 small", with(texture_desc(64, 64, DXGI_FORMAT_BC1_UNORM, none, 3), k4), 16 * 16 * 8},
        {"64x64 depth", texture_desc(64, 64, DXGI_FORMAT_D32_FLOAT, D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL),
         64 * 64 * 4},
        {"8x8 array of 6", texture_desc(8, 8, rgba8, none, 4, 6), 8 * 8 * 4 * 6},
        {"32x32x8 volume", big_volume, 32 * 32 * 8 * 4},
        {"64x64 4x MSAA", with(texture_desc(64, 64, rgba8, rt), 0, 4), 64 * 64 * 4 * 4},
        {"64x64 32x MSAA", with(texture_desc(64, 64, rgba8, rt), 0, 32), 64 * 64 * 4 * 32},
        {"64x64 64x MSAA (invalid)", with(texture_desc(64, 64, rgba8, rt), 0, 64), 0},
        {"64x64 zero sample count (invalid)", with(texture_desc(64, 64, rgba8, rt), 0, 0), 0},
        {"64x64 4x MSAA small", with(texture_desc(64, 64, rgba8, rt), k64, 4), 64 * 64 * 4 * 4},
        {"1024x1024 4x MSAA small", with(texture_desc(1024, 1024, rgba8, rt), k64, 4), 1024 * 1024 * 4 * 4},
        {"zero width", texture_desc(0, 16, rgba8, none), 0},
        {"alignment 12345", with(texture_desc(16, 16, rgba8, none), 12345), 0},
        {"4x MSAA with 2 mips", with(texture_desc(64, 64, rgba8, rt, 2), 0, 4), 0},
        {"16x16 R32G32_UINT", texture_desc(16, 16, DXGI_FORMAT_R32G32_UINT, none, 1), 16 * 16 * 8},
        {"16x16 R32G32_SINT", texture_desc(16, 16, DXGI_FORMAT_R32G32_SINT, none, 1), 16 * 16 * 8},
        {"16x16 R32G32_TYPELESS", texture_desc(16, 16, DXGI_FORMAT_R32G32_TYPELESS, none, 1), 16 * 16 * 8},
        {"16x16 R16G16B16A16_UINT", texture_desc(16, 16, DXGI_FORMAT_R16G16B16A16_UINT, none, 1), 16 * 16 * 8},
        {"16x16 R16G16B16A16_SINT", texture_desc(16, 16, DXGI_FORMAT_R16G16B16A16_SINT, none, 1), 16 * 16 * 8},
        {"16x16 R10G10B10A2_UINT", texture_desc(16, 16, DXGI_FORMAT_R10G10B10A2_UINT, none, 1), 16 * 16 * 4},
        {"16x16 R10G10B10_XR_BIAS_A2_UNORM",
         texture_desc(16, 16, DXGI_FORMAT_R10G10B10_XR_BIAS_A2_UNORM, none, 1), 16 * 16 * 4},
        {"16x16 R16G16_TYPELESS", texture_desc(16, 16, DXGI_FORMAT_R16G16_TYPELESS, none, 1), 16 * 16 * 4},
        {"16x16 R16G16_UINT", texture_desc(16, 16, DXGI_FORMAT_R16G16_UINT, none, 1), 16 * 16 * 4},
        {"16x16 R16G16_SINT", texture_desc(16, 16, DXGI_FORMAT_R16G16_SINT, none, 1), 16 * 16 * 4},
        {"16x16 R8G8_TYPELESS", texture_desc(16, 16, DXGI_FORMAT_R8G8_TYPELESS, none, 1), 16 * 16 * 2},
        {"16x16 R8G8_UINT", texture_desc(16, 16, DXGI_FORMAT_R8G8_UINT, none, 1), 16 * 16 * 2},
        {"16x16 R8G8_SINT", texture_desc(16, 16, DXGI_FORMAT_R8G8_SINT, none, 1), 16 * 16 * 2},
    };
    int mismatches = 0;
    const auto mismatch = [&](const char* name, const D3D12_RESOURCE_ALLOCATION_INFO& a,
                              const D3D12_RESOURCE_ALLOCATION_INFO& b) {
        if (mismatches++ < 24)
            std::fprintf(stderr, "allocation info %s: ours size %llu alignment %llu, host size %llu alignment %llu\n",
                         name, (unsigned long long)a.SizeInBytes, (unsigned long long)a.Alignment,
                         (unsigned long long)b.SizeInBytes, (unsigned long long)b.Alignment);
    };
    for (const Case& c : cases) {
        const D3D12_RESOURCE_ALLOCATION_INFO a = ours->GetResourceAllocationInfo(0, 1, &c.desc);
        const D3D12_RESOURCE_ALLOCATION_INFO b = host->GetResourceAllocationInfo(0, 1, &c.desc);
        if (b.SizeInBytes == UINT64_MAX || !c.top_mip_bytes) {
            if (a.SizeInBytes != b.SizeInBytes || (b.SizeInBytes != UINT64_MAX && a.Alignment != b.Alignment))
                mismatch(c.name, a, b);
            continue;
        }
        if (a.Alignment != b.Alignment || a.SizeInBytes == UINT64_MAX || a.SizeInBytes % a.Alignment ||
            a.SizeInBytes < c.top_mip_bytes)
            mismatch(c.name, a, b);
        std::printf("allocation info %s: alignment %llu, size ours %llu host %llu\n", c.name,
                    (unsigned long long)a.Alignment, (unsigned long long)a.SizeInBytes,
                    (unsigned long long)b.SizeInBytes);
    }

    // Several resources at once: each at its own alignment after the one before, the largest alignment overall.
    // Buffers only for the host comparison (their sizes are fixed); the texture checks our own layout.
    const D3D12_RESOURCE_DESC buffers[] = {buffer_desc(300), buffer_desc(70000), buffer_desc(1)};
    const D3D12_RESOURCE_ALLOCATION_INFO a = ours->GetResourceAllocationInfo(0, 3, buffers);
    const D3D12_RESOURCE_ALLOCATION_INFO b = host->GetResourceAllocationInfo(0, 3, buffers);
    if (a.SizeInBytes != b.SizeInBytes || a.Alignment != b.Alignment) mismatch("three buffers", a, b);

    Ref<ID3D12Device8> ours8;
    if (FAILED(ours->QueryInterface(IID_PPV_ARGS(&ours8.p)))) return mismatches + 1;
    const D3D12_RESOURCE_DESC mixed[] = {with(texture_desc(16, 16, rgba8, none), k4), buffer_desc(300),
                                         with(texture_desc(64, 64, rgba8, rt), 0, 4)};
    D3D12_RESOURCE_DESC1 mixed1[3]{};
    for (int i = 0; i < 3; ++i) std::memcpy(&mixed1[i], &mixed[i], sizeof(mixed[i]));
    D3D12_RESOURCE_ALLOCATION_INFO1 each[3]{}, each2[3]{};
    const D3D12_RESOURCE_ALLOCATION_INFO total = ours8->GetResourceAllocationInfo1(0, 3, mixed, each);
    const D3D12_RESOURCE_ALLOCATION_INFO total2 = ours8->GetResourceAllocationInfo2(0, 3, mixed1, each2);
    const auto aligned = [](UINT64 v, UINT64 to) { return (v + to - 1) / to * to; };
    bool layout_ok = total.SizeInBytes == total2.SizeInBytes && total.Alignment == total2.Alignment &&
                     each[0].Offset == 0 && each[0].Alignment == k4 && each[1].Alignment == k64 &&
                     each[2].Alignment == D3D12_DEFAULT_MSAA_RESOURCE_PLACEMENT_ALIGNMENT &&
                     total.Alignment == D3D12_DEFAULT_MSAA_RESOURCE_PLACEMENT_ALIGNMENT &&
                     total.SizeInBytes == each[2].Offset + each[2].SizeInBytes;
    for (int i = 0; i < 3; ++i) {
        const D3D12_RESOURCE_ALLOCATION_INFO single = ours->GetResourceAllocationInfo(0, 1, &mixed[i]);
        layout_ok = layout_ok && each[i].SizeInBytes == single.SizeInBytes &&
                    each[i].Alignment == single.Alignment && each[i].Offset == each2[i].Offset &&
                    each[i].SizeInBytes == each2[i].SizeInBytes && each[i].Alignment == each2[i].Alignment &&
                    (i == 0 || each[i].Offset == aligned(each[i - 1].Offset + each[i - 1].SizeInBytes,
                                                         each[i].Alignment));
    }
    // An invalid description in the list makes the whole answer invalid.
    const D3D12_RESOURCE_DESC with_bad[] = {buffer_desc(300), buffer_desc(0)};
    layout_ok = layout_ok && ours8->GetResourceAllocationInfo1(0, 2, with_bad, each).SizeInBytes == UINT64_MAX &&
                host->GetResourceAllocationInfo(0, 2, with_bad).SizeInBytes == UINT64_MAX;
    if (!layout_ok) {
        ++mismatches;
        std::fprintf(stderr, "allocation info: Device4/Device8 per-resource layout wrong (total %llu/%llu)\n",
                     (unsigned long long)total.SizeInBytes, (unsigned long long)total.Alignment);
    }
    return mismatches;
}

D3D12_RESOURCE_DESC reserved_desc(D3D12_RESOURCE_DIMENSION dimension, UINT64 width, UINT height, UINT16 depth,
                                  UINT16 mips, DXGI_FORMAT format) {
    D3D12_RESOURCE_DESC desc = dimension == D3D12_RESOURCE_DIMENSION_BUFFER
                                   ? buffer_desc(width)
                                   : texture_desc(UINT(width), height, format, D3D12_RESOURCE_FLAG_NONE, mips, depth);
    desc.Dimension = dimension;
    if (dimension != D3D12_RESOURCE_DIMENSION_BUFFER) desc.Layout = D3D12_TEXTURE_LAYOUT_64KB_UNDEFINED_SWIZZLE;
    return desc;
}

// GetResourceTiling against the PC runtime, for the same reserved resources. What D3D12 defines must match: the
// tile shape, every standard mip's tile counts and its start within its array slice, and that no mip filling a
// whole tile is packed. Which smaller mips are packed (D3D11.3 spec 5.9.2.8.5: "may or may not be packed,
// depending on the IHV") and how many tiles they take is the driver's choice (src/resource_tiling.h), and with it
// the slice size and the total: compared only against a host that splits at the same mip. A layout the host
// refuses is skipped (drivers differ in what they accept); one the host accepts and ours refuses is a mismatch.
// Returns the number of mismatches, or -1 without a host D3D12 device with tiled resources or with every layout
// skipped.
int compare_tiling(ID3D12Device* ours) {
    Ref<ID3D12Device> host;
    if (!create_host_device(host)) return -1;
    D3D12_FEATURE_DATA_D3D12_OPTIONS options{};
    if (FAILED(host->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, &options, sizeof(options))) ||
        options.TiledResourcesTier == D3D12_TILED_RESOURCES_TIER_NOT_SUPPORTED)
        return -1;
    const bool volumes = options.TiledResourcesTier >= D3D12_TILED_RESOURCES_TIER_3;
    const auto B = D3D12_RESOURCE_DIMENSION_BUFFER, T2 = D3D12_RESOURCE_DIMENSION_TEXTURE2D,
               T3 = D3D12_RESOURCE_DIMENSION_TEXTURE3D;
    const D3D12_RESOURCE_DESC cases[] = {
        reserved_desc(B, 200000, 1, 1, 1, DXGI_FORMAT_UNKNOWN),
        reserved_desc(T2, 1024, 1024, 1, 0, DXGI_FORMAT_R8G8B8A8_UNORM),
        reserved_desc(T2, 1000, 300, 3, 0, DXGI_FORMAT_R8G8B8A8_UNORM_SRGB),
        reserved_desc(T2, 512, 512, 1, 0, DXGI_FORMAT_R8_UNORM),
        reserved_desc(T2, 512, 512, 1, 0, DXGI_FORMAT_R16_FLOAT),
        reserved_desc(T2, 512, 512, 1, 0, DXGI_FORMAT_R16G16B16A16_FLOAT),
        reserved_desc(T2, 512, 512, 2, 0, DXGI_FORMAT_R32G32B32A32_FLOAT),
        reserved_desc(T2, 2048, 2048, 1, 0, DXGI_FORMAT_BC1_UNORM),
        reserved_desc(T2, 2048, 1024, 2, 0, DXGI_FORMAT_BC7_UNORM),
        reserved_desc(T2, 1000, 600, 1, 0, DXGI_FORMAT_BC3_UNORM),
        reserved_desc(T2, 128, 128, 1, 1, DXGI_FORMAT_R8G8B8A8_UNORM),
        reserved_desc(T2, 100, 100, 1, 0, DXGI_FORMAT_R8G8B8A8_UNORM),
        reserved_desc(T2, 4096, 64, 1, 0, DXGI_FORMAT_B8G8R8A8_UNORM),
        reserved_desc(T2, 1024, 1024, 4, 3, DXGI_FORMAT_R32_FLOAT),
        reserved_desc(T3, 256, 256, 64, 0, DXGI_FORMAT_R8G8B8A8_UNORM),
        reserved_desc(T3, 128, 128, 128, 0, DXGI_FORMAT_R8_UNORM),
        reserved_desc(T3, 64, 64, 64, 0, DXGI_FORMAT_R16G16B16A16_FLOAT),
        reserved_desc(T3, 64, 64, 64, 0, DXGI_FORMAT_R32G32B32A32_FLOAT),
    };
    struct Tiling {
        UINT total = 0, count = 64;
        D3D12_PACKED_MIP_INFO packed{};
        D3D12_TILE_SHAPE shape{};
        D3D12_SUBRESOURCE_TILING subresources[64]{};
    };
    int mismatches = 0, compared = 0;
    const auto report = [&](int index, const char* what, UINT a, UINT b) {
        if (mismatches++ < 24) std::fprintf(stderr, "tiling case %d %s: ours %u host %u\n", index, what, a, b);
    };
    for (int index = 0; index < int(std::size(cases)); ++index) {
        const D3D12_RESOURCE_DESC& desc = cases[index];
        if (desc.Dimension == T3 && !volumes) continue;
        Ref<ID3D12Resource> a, b;
        const HRESULT ours_hr =
            ours->CreateReservedResource(&desc, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&a.p));
        const HRESULT host_hr =
            host->CreateReservedResource(&desc, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&b.p));
        if (FAILED(host_hr)) {
            // Which reserved resources a PC driver accepts varies: an RTX 3090 on driver 616.56 refuses texture
            // arrays whose mips reach the packed tail (cases 2, 6 and 8, at tier 3), an RTX 5070 on 621.01 accepts
            // them. The console is the reference for what the layer accepts, not the host, so there is nothing to
            // compare here.
            std::printf("GetResourceTiling case %d: the host runtime refuses it (0x%08lx), skipped\n", index,
                        static_cast<unsigned long>(host_hr));
            continue;
        }
        if (FAILED(ours_hr)) {
            report(index, "creation refused by ours, accepted by the host (HRESULT low word)", ours_hr & 0xffff, 0);
            continue;
        }
        ++compared;
        Tiling x, y;
        ours->GetResourceTiling(a.p, &x.total, &x.packed, &x.shape, &x.count, 0, x.subresources);
        host->GetResourceTiling(b.p, &y.total, &y.packed, &y.shape, &y.count, 0, y.subresources);
        if (x.shape.WidthInTexels != y.shape.WidthInTexels) report(index, "tile width", x.shape.WidthInTexels, y.shape.WidthInTexels);
        if (x.shape.HeightInTexels != y.shape.HeightInTexels) report(index, "tile height", x.shape.HeightInTexels, y.shape.HeightInTexels);
        if (x.shape.DepthInTexels != y.shape.DepthInTexels) report(index, "tile depth", x.shape.DepthInTexels, y.shape.DepthInTexels);
        if (x.count != y.count) {
            report(index, "subresource count", x.count, y.count);
            continue;
        }
        // Which small mips are packed is the driver's choice (D3D11.3 functional spec 5.9.2.8.5, D3D12_PACKED_MIP_INFO):
        // only a mip that fills at least one standard tile in every dimension must stay standard; a coarser mip with
        // any dimension below the tile "may or may not be packed, depending on the IHV", and once a mip is packed all
        // coarser ones are. Hosts differ: an RTX 5070 packs from the first mip below the tile (like ours), a Radeon
        // 880M keeps one or two more standard, WARP packs nothing. So the layer must keep every mip that fills a
        // tile standard and be self-consistent; the host's boundary is informational, and tile counts are compared
        // only for mips both treat as standard (and the packed totals only when both split at the same mip).
        const UINT layers = desc.Dimension == T2 ? desc.DepthOrArraySize : 1;
        const UINT mips = x.count / layers;
        const UINT slice_x = x.total / layers, slice_y = y.total / layers;
        UINT shared = 0;
        if (desc.Dimension != B) {
            UINT required = 0;  // leading mips that fill a whole tile in every dimension
            for (UINT m = 0; m < mips; ++m) {
                const UINT64 w = std::max<UINT64>(1, desc.Width >> m);
                const UINT h = std::max<UINT>(1, desc.Height >> m);
                const UINT d = desc.Dimension == T3 ? std::max<UINT>(1, desc.DepthOrArraySize >> m) : 1;
                if (w < x.shape.WidthInTexels || h < x.shape.HeightInTexels || d < x.shape.DepthInTexels) break;
                ++required;
            }
            if (x.packed.NumStandardMips < required || x.packed.NumStandardMips + x.packed.NumPackedMips != int(mips))
                report(index, "standard mips (a mip filling a tile must not be packed)", x.packed.NumStandardMips, required);
            if (y.packed.NumStandardMips < required)
                report(index, "host standard mips below the D3D12 minimum", y.packed.NumStandardMips, required);
            if (bool(x.packed.NumPackedMips) != bool(x.packed.NumTilesForPackedMips))
                report(index, "packed tiles", x.packed.NumPackedMips, x.packed.NumTilesForPackedMips);
            if (x.packed.NumStandardMips != y.packed.NumStandardMips)
                std::printf("GetResourceTiling case %d: standard mips ours %u, host %u (driver's choice, not compared)\n",
                            index, x.packed.NumStandardMips, y.packed.NumStandardMips);
            shared = std::min(x.packed.NumStandardMips, y.packed.NumStandardMips);
            if (x.packed.NumStandardMips == y.packed.NumStandardMips) {
                if (x.packed.NumPackedMips != y.packed.NumPackedMips) report(index, "packed mips", x.packed.NumPackedMips, y.packed.NumPackedMips);
                if (x.packed.StartTileIndexInOverallResource != y.packed.StartTileIndexInOverallResource)
                    report(index, "packed start", x.packed.StartTileIndexInOverallResource, y.packed.StartTileIndexInOverallResource);
                if (slice_x - x.packed.NumTilesForPackedMips != slice_y - y.packed.NumTilesForPackedMips)
                    report(index, "standard tiles per slice", slice_x - x.packed.NumTilesForPackedMips,
                           slice_y - y.packed.NumTilesForPackedMips);
            }
        } else if (x.total != y.total) {
            report(index, "buffer tiles", x.total, y.total);
        }
        for (UINT i = 0; i < x.count; ++i) {
            const auto &s = x.subresources[i], &t = y.subresources[i];
            const UINT layer = i / mips;
            if (desc.Dimension != B && i % mips >= x.packed.NumStandardMips) {
                // Packed: no tiles of its own. (On slices after the first the PC runtime writes a tile index here,
                // not the D3D12_PACKED_TILE its documentation gives; ours follows the documentation.)
                if (s.WidthInTiles || s.HeightInTiles || s.DepthInTiles || s.StartTileIndexInOverallResource != D3D12_PACKED_TILE)
                    report(index, "packed subresource", i, 0);
                continue;
            }
            if (desc.Dimension != B && i % mips >= shared) continue;  // standard in ours only: the host packs it
            if (s.WidthInTiles != t.WidthInTiles || s.HeightInTiles != t.HeightInTiles || s.DepthInTiles != t.DepthInTiles ||
                s.StartTileIndexInOverallResource - layer * slice_x != t.StartTileIndexInOverallResource - layer * slice_y)
                report(index, "standard subresource", i, 0);
        }
    }
    return compared ? mismatches : -1;
}


// whole_rectangle is a rectangle over the whole source: by the documentation the same as a null rectangle.
enum class ResolveScenario { full, rectangle, null_rectangle, whole_rectangle };

const char* resolve_scenario_name(ResolveScenario scenario) {
    switch (scenario) {
    case ResolveScenario::full: return "ResolveSubresource";
    case ResolveScenario::rectangle: return "ResolveSubresourceRegion(rect)";
    case ResolveScenario::null_rectangle: return "ResolveSubresourceRegion(null rect)";
    case ResolveScenario::whole_rectangle: return "ResolveSubresourceRegion(whole-source rect)";
    }
    return "unknown";
}

UINT resolve_pixel_size(DXGI_FORMAT format) {
    switch (format) {
    case DXGI_FORMAT_R8G8B8A8_UNORM:
    case DXGI_FORMAT_B8G8R8A8_UNORM:
    case DXGI_FORMAT_R10G10B10A2_UNORM: return 4;
    case DXGI_FORMAT_R16G16B16A16_FLOAT: return 8;
    case DXGI_FORMAT_R32G32B32A32_FLOAT: return 16;
    default: return 0;
    }
}

// Runs one deterministic resolve on either the host D3D12 device or the Vulkan-backed proxy and returns only
// packed texel bytes (row-pitch padding is discarded). The source/destination are cleared to solid colours so
// every format and sample count exercises the resolve path without requiring a shader or game data.
bool capture_resolve(ID3D12Device* device, DXGI_FORMAT format, UINT samples, ResolveScenario scenario,
                     std::vector<std::uint8_t>& pixels) {
#define RESOLVE_CHECK(condition) do { \
    if (!(condition)) { \
        std::fprintf(stderr, "resolve capture failed at line %d: %s (format %d, samples %u, %s)\n", \
                     __LINE__, #condition, int(format), samples, resolve_scenario_name(scenario)); \
        return false; \
    } \
} while (false)

    const bool proxy = rt::d3d12_xs::is_device_proxy(device);
    const UINT src_width = 8, src_height = 8;
    const UINT dst_width = scenario == ResolveScenario::full ? src_width : 12;
    const UINT dst_height = scenario == ResolveScenario::full ? src_height : 10;
    const UINT pixel_size = resolve_pixel_size(format);
    RESOLVE_CHECK(pixel_size != 0);

    const D3D12_HEAP_PROPERTIES default_heap{D3D12_HEAP_TYPE_DEFAULT};
    const D3D12_HEAP_PROPERTIES readback_heap{D3D12_HEAP_TYPE_READBACK};
    D3D12_RESOURCE_DESC src_desc =
        texture_desc(src_width, src_height, format, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET);
    src_desc.SampleDesc.Count = samples;
    D3D12_RESOURCE_DESC dst_desc =
        texture_desc(dst_width, dst_height, format, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET);

    Ref<ID3D12Resource> src, dst, readback;
    RESOLVE_CHECK(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &src_desc,
                                                             D3D12_RESOURCE_STATE_RENDER_TARGET, nullptr,
                                                             IID_PPV_ARGS(&src.p))));
    RESOLVE_CHECK(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &dst_desc,
                                                             D3D12_RESOURCE_STATE_RENDER_TARGET, nullptr,
                                                             IID_PPV_ARGS(&dst.p))));

    D3D12_DESCRIPTOR_HEAP_DESC heap_desc{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 2, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
    Ref<ID3D12DescriptorHeap> rtv_heap;
    RESOLVE_CHECK(SUCCEEDED(device->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&rtv_heap.p))));
    D3D12_CPU_DESCRIPTOR_HANDLE src_rtv{}, dst_rtv{};
    if (proxy) {
        const std::uint64_t start =
            reinterpret_cast<XboxHeapHandleFn>(slot(rtv_heap.p, 9))(rtv_heap.p);
        const UINT increment = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
        RESOLVE_CHECK(start != 0 && increment >= sizeof(recity::TargetView));
        src_rtv = {SIZE_T(start)};
        dst_rtv = {SIZE_T(start + increment)};
    } else {
        src_rtv = rtv_heap->GetCPUDescriptorHandleForHeapStart();
        dst_rtv = {src_rtv.ptr + device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV)};
    }
    D3D12_RENDER_TARGET_VIEW_DESC msaa_view{};
    msaa_view.Format = format;
    msaa_view.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2DMS;
    device->CreateRenderTargetView(src.p, &msaa_view, src_rtv);
    device->CreateRenderTargetView(dst.p, nullptr, dst_rtv);

    D3D12_COMMAND_QUEUE_DESC queue_desc{};
    queue_desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    Ref<ID3D12CommandQueue> queue;
    Ref<ID3D12CommandAllocator> allocator;
    Ref<ID3D12GraphicsCommandList> list;
    Ref<ID3D12GraphicsCommandList7> list7;
    Ref<ID3D12Fence> fence;
    RESOLVE_CHECK(SUCCEEDED(device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue.p))));
    RESOLVE_CHECK(SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator.p))));
    RESOLVE_CHECK(SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.p, nullptr,
                                                       IID_PPV_ARGS(&list.p))));
    // ResolveSubresourceRegion is on ID3D12GraphicsCommandList7; the full ResolveSubresource path does not need
    // it, so a host without list7 (Windows 10) can still be compared on the full form.
    if (scenario != ResolveScenario::full)
        RESOLVE_CHECK(SUCCEEDED(list->QueryInterface(IID_PPV_ARGS(&list7.p))));
    RESOLVE_CHECK(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))));

    UINT64 cursor = 0;
    const auto readback_layout = place(device, dst.p, 0, cursor);
    RESOLVE_CHECK(cursor != 0 && readback_layout.Footprint.RowPitch >= dst_width * pixel_size);
    const D3D12_RESOURCE_DESC readback_desc = buffer_desc(cursor);
    RESOLVE_CHECK(SUCCEEDED(device->CreateCommittedResource(&readback_heap, D3D12_HEAP_FLAG_NONE,
                                                             &readback_desc, D3D12_RESOURCE_STATE_COPY_DEST,
                                                             nullptr, IID_PPV_ARGS(&readback.p))));

    const FLOAT green[4] = {0, 1, 0, 1};
    const FLOAT red[4] = {1, 0, 0, 1};
    // Use a rectangle even for the full source: multisampled images are cleared as rendering attachments,
    // not with vkCmdClearColorImage (which is for single-sample transfer clears).
    const D3D12_RECT full_source{0, 0, LONG(src_width), LONG(src_height)};
    list->ClearRenderTargetView(src_rtv, green, 1, &full_source);
    list->ClearRenderTargetView(dst_rtv, red, 0, nullptr);
    const D3D12_RESOURCE_BARRIER before_resolve[] = {
        transition(src.p, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_RESOLVE_SOURCE),
        transition(dst.p, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_RESOLVE_DEST)};
    list->ResourceBarrier(2, before_resolve);
    if (scenario == ResolveScenario::full) {
        list->ResolveSubresource(dst.p, 0, src.p, 0, format);
    } else {
        D3D12_RECT rectangle{2, 1, 6, 5};
        if (scenario == ResolveScenario::whole_rectangle) rectangle = full_source;
        list7->ResolveSubresourceRegion(dst.p, 0, 3, 2, src.p, 0,
                                        scenario == ResolveScenario::null_rectangle ? nullptr : &rectangle,
                                        format, D3D12_RESOLVE_MODE_AVERAGE);
    }
    const auto to_copy = transition(dst.p, D3D12_RESOURCE_STATE_RESOLVE_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE);
    list->ResourceBarrier(1, &to_copy);
    const auto readback_location = footprint(readback.p, readback_layout);
    const auto resolved_location = subresource(dst.p, 0);
    list->CopyTextureRegion(&readback_location, 0, 0, 0, &resolved_location, nullptr);
    RESOLVE_CHECK(list->Close() == S_OK);

    ID3D12CommandList* lists[] = {list.p};
    queue->ExecuteCommandLists(1, lists);
    if (proxy)
        RESOLVE_CHECK(rt::d3d12::last_execute_result(queue.p) == recity::ReplayResult::success);
    RESOLVE_CHECK(SUCCEEDED(queue->Signal(fence.p, 1)) && SUCCEEDED(fence->SetEventOnCompletion(1, nullptr)));

    D3D12_RANGE read_range{0, SIZE_T(cursor)};
    std::uint8_t* mapped = nullptr;
    RESOLVE_CHECK(SUCCEEDED(readback->Map(0, &read_range, reinterpret_cast<void**>(&mapped))) && mapped);
    pixels.resize(size_t(dst_width) * dst_height * pixel_size);
    for (UINT row = 0; row < dst_height; ++row)
        std::memcpy(pixels.data() + size_t(row) * dst_width * pixel_size,
                    mapped + readback_layout.Offset + size_t(row) * readback_layout.Footprint.RowPitch,
                    size_t(dst_width) * pixel_size);
    const D3D12_RANGE written{0, 0};
    readback->Unmap(0, &written);
#undef RESOLVE_CHECK
    return true;
}

// Compare D3D12 multisample quality queries for depth/stencil formats with the host runtime. The Vulkan
// backend has one standard sample pattern, so every supported count reports one quality level (index 0).
// A count the Vulkan GPU cannot do for depth and stencil attachments (depth_sample_counts, from its limits) is not
// something the layer can offer: a software host such as WARP advertises 16x, an RTX or Radeon does not.
int compare_depth_multisample_quality_levels(ID3D12Device* ours, VkSampleCountFlags depth_sample_counts) {
    Ref<ID3D12Device> host;
    if (!create_host_device(host)) return -1;

    constexpr DXGI_FORMAT formats[] = {
        DXGI_FORMAT_D32_FLOAT,
        DXGI_FORMAT_D16_UNORM,
        DXGI_FORMAT_D24_UNORM_S8_UINT,
        DXGI_FORMAT_D32_FLOAT_S8X24_UINT,
    };
    constexpr UINT sample_counts[] = {2, 4, 8, 16, 32};
    unsigned compared = 0, skipped = 0;
    int mismatches = 0;
    for (DXGI_FORMAT format : formats) {
        for (UINT samples : sample_counts) {
            D3D12_FEATURE_DATA_MULTISAMPLE_QUALITY_LEVELS host_levels{};
            host_levels.Format = format;
            host_levels.SampleCount = samples;
            D3D12_FEATURE_DATA_MULTISAMPLE_QUALITY_LEVELS our_levels = host_levels;
            const HRESULT host_hr = host->CheckFeatureSupport(D3D12_FEATURE_MULTISAMPLE_QUALITY_LEVELS,
                                                               &host_levels, sizeof(host_levels));
            if (FAILED(host_hr) || host_levels.NumQualityLevels == 0 || !(depth_sample_counts & samples)) {
                ++skipped;
                continue;
            }
            ++compared;
            const HRESULT our_hr = ours->CheckFeatureSupport(D3D12_FEATURE_MULTISAMPLE_QUALITY_LEVELS,
                                                             &our_levels, sizeof(our_levels));
            if (FAILED(our_hr) || our_levels.NumQualityLevels != 1) {
                std::fprintf(stderr,
                             "depth MSAA quality mismatch: format %d samples %u (host %u quality levels, Vulkan "
                             "HRESULT 0x%08lx / %u levels)\n",
                             int(format), samples, host_levels.NumQualityLevels,
                             static_cast<unsigned long>(our_hr), our_levels.NumQualityLevels);
                ++mismatches;
            }
        }
    }
    std::printf("Depth/stencil MSAA quality queries: %u compared, %u host-unsupported skipped, %d mismatches\n",
                compared, skipped, mismatches);
    return mismatches;
}

// Compare the Vulkan-backed resolve output byte-for-byte with the host runtime for every common supported format,
// sample count and region form. A host-unsupported combination is a deliberate skip; a mismatch in advertised
// support or in the resulting pixels is a failure. The region forms need ID3D12GraphicsCommandList7 on the host;
// without it (Windows 10) only the full ResolveSubresource form is compared, so the later device checks still run.
int compare_resolves(ID3D12Device* ours, VkSampleCountFlags color_sample_counts) {
    Ref<ID3D12Device> host;
    if (!create_host_device(host)) return -1;
    const bool host_regions = host_has_graphics_command_list7(host.p);
    if (!host_regions)
        std::printf("Resolve comparison: the host runtime has no ID3D12GraphicsCommandList7; "
                    "ResolveSubresourceRegion cases are skipped\n");

    constexpr DXGI_FORMAT formats[] = {
        DXGI_FORMAT_R8G8B8A8_UNORM,
        DXGI_FORMAT_B8G8R8A8_UNORM,
        DXGI_FORMAT_R10G10B10A2_UNORM,
        DXGI_FORMAT_R16G16B16A16_FLOAT,
        DXGI_FORMAT_R32G32B32A32_FLOAT,
    };
    constexpr UINT sample_counts[] = {2, 4, 8, 16, 32};
    constexpr UINT required =
        UINT(D3D12_FORMAT_SUPPORT1_RENDER_TARGET) | UINT(D3D12_FORMAT_SUPPORT1_MULTISAMPLE_RENDERTARGET) |
        UINT(D3D12_FORMAT_SUPPORT1_MULTISAMPLE_RESOLVE);
    constexpr ResolveScenario scenarios[] = {
        ResolveScenario::full, ResolveScenario::rectangle, ResolveScenario::null_rectangle,
        ResolveScenario::whole_rectangle};
    int mismatches = 0;
    unsigned compared = 0, matched = 0, skipped = 0;

    for (DXGI_FORMAT format : formats) {
        D3D12_FEATURE_DATA_FORMAT_SUPPORT host_format{format}, our_format{format};
        const HRESULT host_format_hr =
            host->CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT, &host_format, sizeof(host_format));
        const HRESULT our_format_hr =
            ours->CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT, &our_format, sizeof(our_format));
        if (FAILED(host_format_hr) || (UINT(host_format.Support1) & required) != required) {
            std::printf("resolve format %d: host does not advertise multisample colour resolve, skipped\n", int(format));
            ++skipped;
            continue;
        }
        if (FAILED(our_format_hr) || (UINT(our_format.Support1) & required) != required) {
            std::fprintf(stderr, "resolve format %d: Vulkan device does not match host multisample format support\n",
                         int(format));
            ++mismatches;
            continue;
        }

        for (UINT samples : sample_counts) {
            D3D12_FEATURE_DATA_MULTISAMPLE_QUALITY_LEVELS host_levels{};
            host_levels.Format = format;
            host_levels.SampleCount = samples;
            D3D12_FEATURE_DATA_MULTISAMPLE_QUALITY_LEVELS our_levels = host_levels;
            const HRESULT host_hr = host->CheckFeatureSupport(D3D12_FEATURE_MULTISAMPLE_QUALITY_LEVELS,
                                                               &host_levels, sizeof(host_levels));
            const HRESULT our_hr = ours->CheckFeatureSupport(D3D12_FEATURE_MULTISAMPLE_QUALITY_LEVELS,
                                                              &our_levels, sizeof(our_levels));
            // Not a count the Vulkan GPU can do for colour attachments (a software host like WARP offers 16x).
            if (FAILED(host_hr) || host_levels.NumQualityLevels == 0 || !(color_sample_counts & samples)) {
                ++skipped;
                continue;
            }
            if (FAILED(our_hr) || our_levels.NumQualityLevels == 0) {
                std::fprintf(stderr, "resolve format %d samples %u: Vulkan reports unsupported, host supports it\n",
                             int(format), samples);
                ++mismatches;
                continue;
            }

            for (ResolveScenario scenario : scenarios) {
                if (!host_regions && scenario != ResolveScenario::full) {
                    ++skipped;
                    continue;
                }
                std::vector<std::uint8_t> host_pixels, our_pixels;
                if (!capture_resolve(host.p, format, samples, scenario, host_pixels) ||
                    !capture_resolve(ours, format, samples, scenario, our_pixels)) {
                    ++mismatches;
                    continue;
                }
                ++compared;
                if (host_pixels != our_pixels && scenario == ResolveScenario::null_rectangle) {
                    // A null rectangle is the whole source (ID3D12GraphicsCommandList1::ResolveSubresourceRegion).
                    // Some hosts do not honour that (the Radeon 880M on driver 26.11.2 and WARP resolve to the
                    // destination's origin and leave the rest unwritten; an RTX 5070 follows the documentation), so
                    // where the host disagrees with its own whole-source rectangle, the layer is held to the
                    // documented result: that rectangle's.
                    std::vector<std::uint8_t> host_whole;
                    if (capture_resolve(host.p, format, samples, ResolveScenario::whole_rectangle, host_whole) &&
                        host_whole != host_pixels && host_whole == our_pixels) {
                        std::printf("resolve format %d samples %u: the host's null-rectangle resolve differs from "
                                    "its whole-source rectangle (host quirk); ours follows the documentation\n",
                                    int(format), samples);
                        ++matched;
                        continue;
                    }
                }
                if (host_pixels != our_pixels) {
                    std::fprintf(stderr, "resolve mismatch: format %d samples %u %s (host %zu bytes, Vulkan %zu)\n",
                                 int(format), samples, resolve_scenario_name(scenario),
                                 host_pixels.size(), our_pixels.size());
                    ++mismatches;
                } else {
                    ++matched;
                }
            }
        }
    }
    std::printf("Resolve comparison: %u compared, %u matched, %u unsupported or host-unavailable cases skipped, "
                "%d mismatches\n", compared, matched, skipped, mismatches);
    if (compared == 0) {
        std::fprintf(stderr, "Resolve comparison: no common supported format/sample combination was exercised\n");
        ++mismatches;
    }
    return mismatches;
}

// D3D12_FEATURE_SHADER_MODEL and D3D12_FEATURE_ROOT_SIGNATURE must give the host PC runtime's HRESULT, with the
// highest supported value clamped to what the layer can honour (DXIL 1.0, root signature 1.1). Returns the number
// of mismatches, or -1 when the host has no D3D12 device.
int compare_feature_support(ID3D12Device* ours) {
    const HMODULE d3d12 = LoadLibraryExA("d3d12.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
    const auto create = d3d12 ? reinterpret_cast<PFN_D3D12_CREATE_DEVICE>(
                                    reinterpret_cast<void*>(GetProcAddress(d3d12, "D3D12CreateDevice")))
                              : nullptr;
    Ref<ID3D12Device> host;
    if (!create || FAILED(create(nullptr, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&host.p)))) return -1;
    int mismatches = 0;
    const auto report = [&](const char* what, UINT request, HRESULT got_hr, UINT got, HRESULT want_hr, UINT want) {
        if (got_hr == want_hr && (FAILED(got_hr) || got == want)) return;
        if (mismatches++ < 8)
            std::fprintf(stderr, "%s request 0x%02x: ours hr=0x%08lX value 0x%02x; host hr=0x%08lX value 0x%02x\n",
                         what, request, static_cast<unsigned long>(got_hr), got,
                         static_cast<unsigned long>(want_hr), want);
    };
    // A host runtime older than a request refuses it with E_INVALIDARG where a newer one clamps it: Windows 10's
    // d3d12.dll answers shader models up to 6.6 and root signatures up to 1.1. Such a request can't be compared with
    // the host, so the layer's answer is checked against its own clamp instead. The requests ascend, so `answered`
    // is the highest one the host has accepted so far.
    unsigned predated = 0;
    const auto host_predates = [&](const char* what, UINT request, UINT answered, HRESULT host_hr, HRESULT got_hr,
                                   UINT got, UINT clamp) {
        if (host_hr != E_INVALIDARG || !answered || request <= answered) return false;
        ++predated;
        if ((got_hr != S_OK || got != clamp) && mismatches++ < 8)
            std::fprintf(stderr, "%s request 0x%02x: ours hr=0x%08lX value 0x%02x; the host runtime predates it, "
                                 "the layer's clamp is S_OK 0x%02x\n",
                         what, request, static_cast<unsigned long>(got_hr), got, clamp);
        return true;
    };

    // 5.1 upwards: D3D12's own baseline, which every SDK names, then 6.0-6.9 by number (only 6.0 is guaranteed to be
    // named by every SDK that has the query at all). And only values the runtime defines go to the host: it crashed
    // the process on an undefined one (0x5f, and once on 0), so the layer's refusal of an undefined value is checked
    // in main.
    const D3D_SHADER_MODEL models[] = {
        D3D_SHADER_MODEL_5_1, D3D_SHADER_MODEL_6_0, static_cast<D3D_SHADER_MODEL>(0x61),
        static_cast<D3D_SHADER_MODEL>(0x62),
        static_cast<D3D_SHADER_MODEL>(0x63), static_cast<D3D_SHADER_MODEL>(0x64),
        static_cast<D3D_SHADER_MODEL>(0x65), static_cast<D3D_SHADER_MODEL>(0x66),
        static_cast<D3D_SHADER_MODEL>(0x67), static_cast<D3D_SHADER_MODEL>(0x68),
        static_cast<D3D_SHADER_MODEL>(0x69)};
    UINT host_model = 0;
    for (const D3D_SHADER_MODEL model : models) {
        D3D12_FEATURE_DATA_SHADER_MODEL ours_data{model}, host_data{model};
        const HRESULT ours_hr =
            ours->CheckFeatureSupport(D3D12_FEATURE_SHADER_MODEL, &ours_data, sizeof(ours_data));
        const HRESULT host_hr =
            host->CheckFeatureSupport(D3D12_FEATURE_SHADER_MODEL, &host_data, sizeof(host_data));
        if (SUCCEEDED(host_hr)) host_model = static_cast<UINT>(model);
        if (host_predates("shader model", static_cast<UINT>(model), host_model, host_hr, ours_hr,
                          static_cast<UINT>(ours_data.HighestShaderModel),
                          std::min<UINT>(static_cast<UINT>(model), static_cast<UINT>(D3D_SHADER_MODEL_6_0))))
            continue;
        const UINT want = std::min<UINT>(static_cast<UINT>(host_data.HighestShaderModel),
                                         static_cast<UINT>(D3D_SHADER_MODEL_6_0));
        report("shader model", static_cast<UINT>(model), ours_hr,
               static_cast<UINT>(ours_data.HighestShaderModel), host_hr, want);
    }

    // 0x3 is 1.2 by number: SDKs without the 1.2 headers do not name it (runtime/CMakeLists.txt probes for that).
    const D3D_ROOT_SIGNATURE_VERSION versions[] = {D3D_ROOT_SIGNATURE_VERSION_1_0, D3D_ROOT_SIGNATURE_VERSION_1_1,
                                                   static_cast<D3D_ROOT_SIGNATURE_VERSION>(0x3)};
    UINT host_version = 0;
    for (const D3D_ROOT_SIGNATURE_VERSION version : versions) {
        D3D12_FEATURE_DATA_ROOT_SIGNATURE ours_data{version}, host_data{version};
        const HRESULT ours_hr =
            ours->CheckFeatureSupport(D3D12_FEATURE_ROOT_SIGNATURE, &ours_data, sizeof(ours_data));
        const HRESULT host_hr =
            host->CheckFeatureSupport(D3D12_FEATURE_ROOT_SIGNATURE, &host_data, sizeof(host_data));
        if (SUCCEEDED(host_hr)) host_version = static_cast<UINT>(version);
        if (host_predates("root signature", static_cast<UINT>(version), host_version, host_hr, ours_hr,
                          static_cast<UINT>(ours_data.HighestVersion),
                          std::min<UINT>(static_cast<UINT>(version),
                                         static_cast<UINT>(D3D_ROOT_SIGNATURE_VERSION_1_1))))
            continue;
        const UINT want = std::min<UINT>(static_cast<UINT>(host_data.HighestVersion),
                                         static_cast<UINT>(D3D_ROOT_SIGNATURE_VERSION_1_1));
        report("root signature", static_cast<UINT>(version), ours_hr,
               static_cast<UINT>(ours_data.HighestVersion), host_hr, want);
    }

    if (predated)
        std::printf("CheckFeatureSupport: %u requests the host runtime predates (e.g. Windows 10) checked against the "
                    "layer's clamp instead of the host\n", predated);

    // A wrong DataSize is refused by both, in both queries.
    {
        D3D12_FEATURE_DATA_SHADER_MODEL ours_data{D3D_SHADER_MODEL_6_0};
        D3D12_FEATURE_DATA_ROOT_SIGNATURE ours_root{D3D_ROOT_SIGNATURE_VERSION_1_1};
        if (SUCCEEDED(ours->CheckFeatureSupport(D3D12_FEATURE_SHADER_MODEL, &ours_data, sizeof(ours_data) - 1)))
            ++mismatches;
        if (SUCCEEDED(ours->CheckFeatureSupport(D3D12_FEATURE_ROOT_SIGNATURE, &ours_root, sizeof(ours_root) - 1)))
            ++mismatches;
        if (SUCCEEDED(ours->CheckFeatureSupport(D3D12_FEATURE_SHADER_MODEL, nullptr, sizeof(ours_data))))
            ++mismatches;
        if (SUCCEEDED(ours->CheckFeatureSupport(D3D12_FEATURE_ROOT_SIGNATURE, nullptr, sizeof(ours_root))))
            ++mismatches;
    }
    return mismatches;
}

// Same-adapter host D3D12 device matched by VkPhysicalDeviceIDProperties (LUID first, then PCI IDs), falling back
// to the default adapter. False if the host has no D3D12 device.
bool create_host_device_for_physical(VkInstance instance, VkPhysicalDevice physical,
                                     PFN_vkGetInstanceProcAddr instance_proc, Ref<ID3D12Device>& host,
                                     std::string& adapter_name) {
    const HMODULE d3d12 = LoadLibraryExA("d3d12.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
    const auto create = d3d12 ? reinterpret_cast<PFN_D3D12_CREATE_DEVICE>(
                                    reinterpret_cast<void*>(GetProcAddress(d3d12, "D3D12CreateDevice")))
                              : nullptr;
    if (!create) return false;

    VkPhysicalDeviceIDProperties id_props{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
    VkPhysicalDeviceProperties2 props2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &id_props};
    if (instance && physical && instance_proc) {
        if (const auto get_props2 = reinterpret_cast<PFN_vkGetPhysicalDeviceProperties2>(
                instance_proc(instance, "vkGetPhysicalDeviceProperties2"))) {
            get_props2(physical, &props2);
        }
    }

    const HMODULE dxgi = LoadLibraryExA("dxgi.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
    using CreateFactoryFn = HRESULT(WINAPI*)(REFIID, void**);
    const auto create_factory = dxgi ? reinterpret_cast<CreateFactoryFn>(
                                           reinterpret_cast<void*>(GetProcAddress(dxgi, "CreateDXGIFactory1")))
                                     : nullptr;
    Ref<IDXGIAdapter1> luid_match, pci_match;
    std::string luid_name, pci_name;
    if (create_factory) {
        Ref<IDXGIFactory4> factory;
        if (SUCCEEDED(create_factory(IID_PPV_ARGS(&factory.p)))) {
            Ref<IDXGIAdapter1> candidate;
            for (UINT i = 0; factory->EnumAdapters1(i, &candidate.p) != DXGI_ERROR_NOT_FOUND; ++i) {
                DXGI_ADAPTER_DESC1 desc{};
                if (SUCCEEDED(candidate->GetDesc1(&desc))) {
                    char utf8[256]{};
                    WideCharToMultiByte(CP_UTF8, 0, desc.Description, -1, utf8, sizeof(utf8), nullptr, nullptr);
                    if (!luid_match.p && id_props.deviceLUIDValid &&
                        std::memcmp(&desc.AdapterLuid, id_props.deviceLUID, sizeof(desc.AdapterLuid)) == 0) {
                        luid_match.p = candidate.p;
                        candidate.p = nullptr;
                        luid_name = utf8;
                        break;
                    }
                    if (!pci_match.p && props2.properties.vendorID != 0 && desc.VendorId == props2.properties.vendorID &&
                        desc.DeviceId == props2.properties.deviceID) {
                        pci_match.p = candidate.p;
                        candidate.p = nullptr;
                        pci_name = utf8;
                    }
                }
                candidate.reset();
            }
        }
    }
    if (luid_match.p && SUCCEEDED(create(luid_match.p, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&host.p)))) {
        adapter_name = luid_name;
        return true;
    }
    if (pci_match.p && SUCCEEDED(create(pci_match.p, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&host.p)))) {
        adapter_name = pci_name;
        return true;
    }
    if (SUCCEEDED(create(nullptr, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&host.p)))) {
        adapter_name = "default adapter";
        return true;
    }
    return false;
}

// D3D-86: validate CheckFeatureSupport(D3D12_OPTIONS2..21 except 6/8/10/12/17), check size/null refusals and local
// invariants, and verify conservative compatibility against the host PC D3D12 runtime on the same adapter.
int compare_feature_options(ID3D12Device* ours, const rt::d3d12::VulkanDeviceInfo& info) {
    int mismatches = 0;
    const auto fail = [&](const char* what) {
        if (mismatches++ < 32) std::fprintf(stderr, "CheckFeatureSupport options: %s\n", what);
    };
    const auto query_ours = [&](auto tag_feature, auto& out) -> bool {
        using T = std::remove_reference_t<decltype(out)>;
        const char* name = tag_feature.first;
        const D3D12_FEATURE feature = tag_feature.second;
        if (ours->CheckFeatureSupport(feature, nullptr, sizeof(T)) != E_INVALIDARG) {
            std::fprintf(stderr, "%s: expected E_INVALIDARG for nullptr\n", name);
            ++mismatches;
            return false;
        }
        std::vector<std::uint8_t> scratch(sizeof(T) + 1, 0xAA);
        for (UINT bad_size : {0u, UINT(sizeof(T) - 1), UINT(sizeof(T) + 1)}) {
            if (ours->CheckFeatureSupport(feature, scratch.data(), bad_size) != E_INVALIDARG) {
                std::fprintf(stderr, "%s: expected E_INVALIDARG for size %u\n", name, bad_size);
                ++mismatches;
                return false;
            }
        }
        std::memset(&out, 0xAA, sizeof(out));
        if (FAILED(ours->CheckFeatureSupport(feature, &out, sizeof(out)))) {
            std::fprintf(stderr, "%s: CheckFeatureSupport failed on Vulkan device\n", name);
            ++mismatches;
            return false;
        }
        return true;
    };

    D3D12_FEATURE_DATA_D3D12_OPTIONS2 o2{};
    D3D12_FEATURE_DATA_D3D12_OPTIONS3 o3{};
    D3D12_FEATURE_DATA_D3D12_OPTIONS4 o4{};
    D3D12_FEATURE_DATA_D3D12_OPTIONS5 o5{};
    D3D12_FEATURE_DATA_D3D12_OPTIONS7 o7{};
    D3D12_FEATURE_DATA_D3D12_OPTIONS9 o9{};
    D3D12_FEATURE_DATA_D3D12_OPTIONS11 o11{};
    D3D12_FEATURE_DATA_D3D12_OPTIONS13 o13{};
    D3D12_FEATURE_DATA_D3D12_OPTIONS14 o14{};
    D3D12_FEATURE_DATA_D3D12_OPTIONS15 o15{};
    D3D12_FEATURE_DATA_D3D12_OPTIONS16 o16{};
    D3D12_FEATURE_DATA_D3D12_OPTIONS18 o18{};
    D3D12_FEATURE_DATA_D3D12_OPTIONS19 o19{};
    D3D12_FEATURE_DATA_D3D12_OPTIONS20 o20{};
    D3D12_FEATURE_DATA_D3D12_OPTIONS21 o21{};

    query_ours(std::pair{"OPTIONS2", D3D12_FEATURE_D3D12_OPTIONS2}, o2);
    query_ours(std::pair{"OPTIONS3", D3D12_FEATURE_D3D12_OPTIONS3}, o3);
    query_ours(std::pair{"OPTIONS4", D3D12_FEATURE_D3D12_OPTIONS4}, o4);
    query_ours(std::pair{"OPTIONS5", D3D12_FEATURE_D3D12_OPTIONS5}, o5);
    query_ours(std::pair{"OPTIONS7", D3D12_FEATURE_D3D12_OPTIONS7}, o7);
    query_ours(std::pair{"OPTIONS9", D3D12_FEATURE_D3D12_OPTIONS9}, o9);
    query_ours(std::pair{"OPTIONS11", D3D12_FEATURE_D3D12_OPTIONS11}, o11);
    query_ours(std::pair{"OPTIONS13", D3D12_FEATURE_D3D12_OPTIONS13}, o13);
    query_ours(std::pair{"OPTIONS14", D3D12_FEATURE_D3D12_OPTIONS14}, o14);
    query_ours(std::pair{"OPTIONS15", D3D12_FEATURE_D3D12_OPTIONS15}, o15);
    query_ours(std::pair{"OPTIONS16", D3D12_FEATURE_D3D12_OPTIONS16}, o16);
    query_ours(std::pair{"OPTIONS18", D3D12_FEATURE_D3D12_OPTIONS18}, o18);
    query_ours(std::pair{"OPTIONS19", D3D12_FEATURE_D3D12_OPTIONS19}, o19);
    query_ours(std::pair{"OPTIONS20", D3D12_FEATURE_D3D12_OPTIONS20}, o20);
    query_ours(std::pair{"OPTIONS21", D3D12_FEATURE_D3D12_OPTIONS21}, o21);

    std::uint32_t unknown_feature = 0;
    if (ours->CheckFeatureSupport(static_cast<D3D12_FEATURE>(9999), &unknown_feature, sizeof(unknown_feature)) !=
        E_INVALIDARG)
        fail("unknown feature unexpectedly answered");

    // Local invariants for what this layer actually implements.
    if (o2.DepthBoundsTestSupported ||
        o2.ProgrammableSamplePositionsTier != D3D12_PROGRAMMABLE_SAMPLE_POSITIONS_TIER_NOT_SUPPORTED)
        fail("OPTIONS2 unexpected support");
    if (o3.CopyQueueTimestampQueriesSupported || o3.CastingFullyTypedFormatSupported ||
        o3.WriteBufferImmediateSupportFlags != D3D12_COMMAND_LIST_SUPPORT_FLAG_NONE ||
        o3.ViewInstancingTier != D3D12_VIEW_INSTANCING_TIER_NOT_SUPPORTED || o3.BarycentricsSupported)
        fail("OPTIONS3 unexpected support");
    if (!o4.MSAA64KBAlignedTextureSupported ||
        o4.SharedResourceCompatibilityTier != D3D12_SHARED_RESOURCE_COMPATIBILITY_TIER_0 ||
        o4.Native16BitShaderOpsSupported)
        fail("OPTIONS4 mismatch");
    if (o5.SRVOnlyTiledResourceTier3 || o5.RenderPassesTier != D3D12_RENDER_PASS_TIER_0 ||
        o5.RaytracingTier != D3D12_RAYTRACING_TIER_NOT_SUPPORTED)
        fail("OPTIONS5 unexpected support");
    if (o7.MeshShaderTier != D3D12_MESH_SHADER_TIER_NOT_SUPPORTED ||
        o7.SamplerFeedbackTier != D3D12_SAMPLER_FEEDBACK_TIER_NOT_SUPPORTED)
        fail("OPTIONS7 unexpected support");
    if (o9.MeshShaderPipelineStatsSupported || o9.MeshShaderSupportsFullRangeRenderTargetArrayIndex ||
        o9.AtomicInt64OnTypedResourceSupported || o9.AtomicInt64OnGroupSharedSupported ||
        o9.DerivativesInMeshAndAmplificationShadersSupported ||
        o9.WaveMMATier != D3D12_WAVE_MMA_TIER_NOT_SUPPORTED)
        fail("OPTIONS9 unexpected support");
    if (o11.AtomicInt64OnDescriptorHeapResourceSupported) fail("OPTIONS11 unexpected support");
    if (o13.UnrestrictedBufferTextureCopyPitchSupported || o13.UnrestrictedVertexElementAlignmentSupported ||
        o13.InvertedViewportHeightFlipsYSupported || o13.InvertedViewportDepthFlipsZSupported ||
        o13.TextureCopyBetweenDimensionsSupported || o13.AlphaBlendFactorSupported)
        fail("OPTIONS13 unexpected support");
    if (o14.AdvancedTextureOpsSupported || o14.WriteableMSAATexturesSupported ||
        o14.IndependentFrontAndBackStencilRefMaskSupported)
        fail("OPTIONS14 unexpected support");
    if (o15.TriangleFanSupported || o15.DynamicIndexBufferStripCutSupported) fail("OPTIONS15 unexpected support");
    if (o16.DynamicDepthBiasSupported || o16.GPUUploadHeapSupported) fail("OPTIONS16 unexpected support");
    if (o18.RenderPassesValid) fail("OPTIONS18 unexpected support");
    if (o19.SupportedSampleCountsWithNoOutputs != 1u)
        fail("OPTIONS19 advertised multisampling for unsupported no-output draws");
    if (o19.MismatchingOutputDimensionsSupported || o19.PointSamplingAddressesNeverRoundUp ||
        o19.RasterizerDesc2Supported || o19.NarrowQuadrilateralLinesSupported ||
        o19.AnisoFilterWithPointMipSupported != (info.sampler_anisotropy ? TRUE : FALSE) ||
        o19.MaxSamplerDescriptorHeapSize != D3D12_MAX_SHADER_VISIBLE_SAMPLER_HEAP_SIZE ||
        o19.MaxSamplerDescriptorHeapSizeWithStaticSamplers != D3D12_MAX_SHADER_VISIBLE_SAMPLER_HEAP_SIZE ||
        o19.MaxViewDescriptorHeapSize != D3D12_MAX_SHADER_VISIBLE_DESCRIPTOR_HEAP_SIZE_TIER_2 ||
        o19.ComputeOnlyCustomHeapSupported)
        fail("OPTIONS19 mismatch");
    if (o20.ComputeOnlyWriteWatchSupported || o20.RecreateAtTier != D3D12_RECREATE_AT_TIER_NOT_SUPPORTED)
        fail("OPTIONS20 unexpected support");
    if (o21.WorkGraphsTier != D3D12_WORK_GRAPHS_TIER_NOT_SUPPORTED ||
        o21.ExecuteIndirectTier != D3D12_EXECUTE_INDIRECT_TIER_1_0 || o21.SampleCmpGradientAndBiasSupported ||
        o21.ExtendedCommandInfoSupported)
        fail("OPTIONS21 unexpected support");

    // Verify OPTIONS19.AnisoFilterWithPointMipSupported follows VulkanDeviceInfo::sampler_anisotropy.
    {
        rt::d3d12::VulkanDeviceInfo toggled_info = info;
        toggled_info.sampler_anisotropy = !info.sampler_anisotropy;
        Ref<ID3D12Device> toggled;
        if (FAILED(rt::d3d12::create_vulkan_device(toggled_info, IID_PPV_ARGS(&toggled.p)))) {
            fail("failed to create toggled sampler_anisotropy device");
        } else {
            D3D12_FEATURE_DATA_D3D12_OPTIONS19 toggled19{};
            if (FAILED(toggled->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS19, &toggled19, sizeof(toggled19))) ||
                toggled19.AnisoFilterWithPointMipSupported != (toggled_info.sampler_anisotropy ? TRUE : FALSE))
                fail("OPTIONS19.AnisoFilterWithPointMipSupported did not follow sampler_anisotropy");
        }
    }

    Ref<ID3D12Device> host;
    std::string adapter_name;
    if (!create_host_device_for_physical(info.instance, info.physical, info.get_instance_proc, host, adapter_name)) {
        std::printf("CheckFeatureSupport options: no host D3D12 device to compare with, skipped\n");
        return mismatches;
    }

    UINT host_compared = 0, host_skipped = 0;
    const auto check_bool = [&](const char* feature, const char* field, BOOL a, BOOL b) {
        if (!a || b) return;
        if (mismatches++ < 32)
            std::fprintf(stderr, "%s.%s: ours TRUE exceeds host FALSE\n", feature, field);
    };
    const auto check_tier = [&](const char* feature, const char* field, int a, int b) {
        if (a <= b) return;
        if (mismatches++ < 32)
            std::fprintf(stderr, "%s.%s: ours tier %d exceeds host %d\n", feature, field, a, b);
    };
    const auto compare_host = [&](const char* name, D3D12_FEATURE feature, const auto& ours_val, auto&& verify) {
        using T = std::remove_cv_t<std::remove_reference_t<decltype(ours_val)>>;
        T host_val{};
        if (FAILED(host->CheckFeatureSupport(feature, &host_val, sizeof(host_val)))) {
            ++host_skipped;
            return;
        }
        ++host_compared;
        verify(name, ours_val, host_val);
    };

    compare_host("OPTIONS2", D3D12_FEATURE_D3D12_OPTIONS2, o2,
                 [&](const char* n, const auto& a, const auto& b) {
                     check_bool(n, "DepthBoundsTestSupported", a.DepthBoundsTestSupported, b.DepthBoundsTestSupported);
                     check_tier(n, "ProgrammableSamplePositionsTier", int(a.ProgrammableSamplePositionsTier),
                                int(b.ProgrammableSamplePositionsTier));
                 });
    compare_host("OPTIONS3", D3D12_FEATURE_D3D12_OPTIONS3, o3,
                 [&](const char* n, const auto& a, const auto& b) {
                     check_bool(n, "CopyQueueTimestampQueriesSupported", a.CopyQueueTimestampQueriesSupported,
                                b.CopyQueueTimestampQueriesSupported);
                     check_bool(n, "CastingFullyTypedFormatSupported", a.CastingFullyTypedFormatSupported,
                                b.CastingFullyTypedFormatSupported);
                     if ((UINT(a.WriteBufferImmediateSupportFlags) & ~UINT(b.WriteBufferImmediateSupportFlags)) != 0 &&
                         mismatches++ < 32)
                         std::fprintf(stderr, "%s.WriteBufferImmediateSupportFlags: ours 0x%x not subset of host 0x%x\n",
                                      n, UINT(a.WriteBufferImmediateSupportFlags),
                                      UINT(b.WriteBufferImmediateSupportFlags));
                     check_tier(n, "ViewInstancingTier", int(a.ViewInstancingTier), int(b.ViewInstancingTier));
                     check_bool(n, "BarycentricsSupported", a.BarycentricsSupported, b.BarycentricsSupported);
                 });
    compare_host("OPTIONS4", D3D12_FEATURE_D3D12_OPTIONS4, o4,
                 [&](const char* n, const auto& a, const auto& b) {
                     check_bool(n, "MSAA64KBAlignedTextureSupported", a.MSAA64KBAlignedTextureSupported,
                                b.MSAA64KBAlignedTextureSupported);
                     check_tier(n, "SharedResourceCompatibilityTier", int(a.SharedResourceCompatibilityTier),
                                int(b.SharedResourceCompatibilityTier));
                     check_bool(n, "Native16BitShaderOpsSupported", a.Native16BitShaderOpsSupported,
                                b.Native16BitShaderOpsSupported);
                 });
    compare_host("OPTIONS5", D3D12_FEATURE_D3D12_OPTIONS5, o5,
                 [&](const char* n, const auto& a, const auto& b) {
                     check_bool(n, "SRVOnlyTiledResourceTier3", a.SRVOnlyTiledResourceTier3, b.SRVOnlyTiledResourceTier3);
                     check_tier(n, "RenderPassesTier", int(a.RenderPassesTier), int(b.RenderPassesTier));
                     check_tier(n, "RaytracingTier", int(a.RaytracingTier), int(b.RaytracingTier));
                 });
    compare_host("OPTIONS7", D3D12_FEATURE_D3D12_OPTIONS7, o7,
                 [&](const char* n, const auto& a, const auto& b) {
                     check_tier(n, "MeshShaderTier", int(a.MeshShaderTier), int(b.MeshShaderTier));
                     check_tier(n, "SamplerFeedbackTier", int(a.SamplerFeedbackTier), int(b.SamplerFeedbackTier));
                 });
    compare_host("OPTIONS9", D3D12_FEATURE_D3D12_OPTIONS9, o9,
                 [&](const char* n, const auto& a, const auto& b) {
                     check_bool(n, "MeshShaderPipelineStatsSupported", a.MeshShaderPipelineStatsSupported,
                                b.MeshShaderPipelineStatsSupported);
                     check_bool(n, "MeshShaderSupportsFullRangeRenderTargetArrayIndex",
                                a.MeshShaderSupportsFullRangeRenderTargetArrayIndex,
                                b.MeshShaderSupportsFullRangeRenderTargetArrayIndex);
                     check_bool(n, "AtomicInt64OnTypedResourceSupported", a.AtomicInt64OnTypedResourceSupported,
                                b.AtomicInt64OnTypedResourceSupported);
                     check_bool(n, "AtomicInt64OnGroupSharedSupported", a.AtomicInt64OnGroupSharedSupported,
                                b.AtomicInt64OnGroupSharedSupported);
                     check_bool(n, "DerivativesInMeshAndAmplificationShadersSupported",
                                a.DerivativesInMeshAndAmplificationShadersSupported,
                                b.DerivativesInMeshAndAmplificationShadersSupported);
                     check_tier(n, "WaveMMATier", int(a.WaveMMATier), int(b.WaveMMATier));
                 });
    compare_host("OPTIONS11", D3D12_FEATURE_D3D12_OPTIONS11, o11,
                 [&](const char* n, const auto& a, const auto& b) {
                     check_bool(n, "AtomicInt64OnDescriptorHeapResourceSupported",
                                a.AtomicInt64OnDescriptorHeapResourceSupported,
                                b.AtomicInt64OnDescriptorHeapResourceSupported);
                 });
    compare_host("OPTIONS13", D3D12_FEATURE_D3D12_OPTIONS13, o13,
                 [&](const char* n, const auto& a, const auto& b) {
                     check_bool(n, "UnrestrictedBufferTextureCopyPitchSupported",
                                a.UnrestrictedBufferTextureCopyPitchSupported,
                                b.UnrestrictedBufferTextureCopyPitchSupported);
                     check_bool(n, "UnrestrictedVertexElementAlignmentSupported",
                                a.UnrestrictedVertexElementAlignmentSupported,
                                b.UnrestrictedVertexElementAlignmentSupported);
                     check_bool(n, "InvertedViewportHeightFlipsYSupported", a.InvertedViewportHeightFlipsYSupported,
                                b.InvertedViewportHeightFlipsYSupported);
                     check_bool(n, "InvertedViewportDepthFlipsZSupported", a.InvertedViewportDepthFlipsZSupported,
                                b.InvertedViewportDepthFlipsZSupported);
                     check_bool(n, "TextureCopyBetweenDimensionsSupported", a.TextureCopyBetweenDimensionsSupported,
                                b.TextureCopyBetweenDimensionsSupported);
                     check_bool(n, "AlphaBlendFactorSupported", a.AlphaBlendFactorSupported,
                                b.AlphaBlendFactorSupported);
                 });
    compare_host("OPTIONS14", D3D12_FEATURE_D3D12_OPTIONS14, o14,
                 [&](const char* n, const auto& a, const auto& b) {
                     check_bool(n, "AdvancedTextureOpsSupported", a.AdvancedTextureOpsSupported,
                                b.AdvancedTextureOpsSupported);
                     check_bool(n, "WriteableMSAATexturesSupported", a.WriteableMSAATexturesSupported,
                                b.WriteableMSAATexturesSupported);
                     check_bool(n, "IndependentFrontAndBackStencilRefMaskSupported",
                                a.IndependentFrontAndBackStencilRefMaskSupported,
                                b.IndependentFrontAndBackStencilRefMaskSupported);
                 });
    compare_host("OPTIONS15", D3D12_FEATURE_D3D12_OPTIONS15, o15,
                 [&](const char* n, const auto& a, const auto& b) {
                     check_bool(n, "TriangleFanSupported", a.TriangleFanSupported, b.TriangleFanSupported);
                     check_bool(n, "DynamicIndexBufferStripCutSupported", a.DynamicIndexBufferStripCutSupported,
                                b.DynamicIndexBufferStripCutSupported);
                 });
    compare_host("OPTIONS16", D3D12_FEATURE_D3D12_OPTIONS16, o16,
                 [&](const char* n, const auto& a, const auto& b) {
                     check_bool(n, "DynamicDepthBiasSupported", a.DynamicDepthBiasSupported,
                                b.DynamicDepthBiasSupported);
                     check_bool(n, "GPUUploadHeapSupported", a.GPUUploadHeapSupported, b.GPUUploadHeapSupported);
                 });
    compare_host("OPTIONS18", D3D12_FEATURE_D3D12_OPTIONS18, o18,
                 [&](const char* n, const auto& a, const auto& b) {
                     check_bool(n, "RenderPassesValid", a.RenderPassesValid, b.RenderPassesValid);
                 });
    compare_host("OPTIONS19", D3D12_FEATURE_D3D12_OPTIONS19, o19,
                 [&](const char* n, const auto& a, const auto& b) {
                     check_bool(n, "MismatchingOutputDimensionsSupported", a.MismatchingOutputDimensionsSupported,
                                b.MismatchingOutputDimensionsSupported);
                     if (b.SupportedSampleCountsWithNoOutputs != 0 &&
                         (a.SupportedSampleCountsWithNoOutputs & ~b.SupportedSampleCountsWithNoOutputs) != 0 &&
                         mismatches++ < 32)
                         std::fprintf(stderr, "%s.SupportedSampleCountsWithNoOutputs: ours 0x%x not subset of host 0x%x\n",
                                      n, a.SupportedSampleCountsWithNoOutputs, b.SupportedSampleCountsWithNoOutputs);
                     check_bool(n, "PointSamplingAddressesNeverRoundUp", a.PointSamplingAddressesNeverRoundUp,
                                b.PointSamplingAddressesNeverRoundUp);
                     check_bool(n, "RasterizerDesc2Supported", a.RasterizerDesc2Supported, b.RasterizerDesc2Supported);
                     check_bool(n, "NarrowQuadrilateralLinesSupported", a.NarrowQuadrilateralLinesSupported,
                                b.NarrowQuadrilateralLinesSupported);
                     check_bool(n, "AnisoFilterWithPointMipSupported", a.AnisoFilterWithPointMipSupported,
                                b.AnisoFilterWithPointMipSupported);
                     if (b.MaxSamplerDescriptorHeapSize > 0 &&
                         a.MaxSamplerDescriptorHeapSize > b.MaxSamplerDescriptorHeapSize && mismatches++ < 32)
                         std::fprintf(stderr, "%s.MaxSamplerDescriptorHeapSize: ours %u > host %u\n", n,
                                      a.MaxSamplerDescriptorHeapSize, b.MaxSamplerDescriptorHeapSize);
                     if (b.MaxSamplerDescriptorHeapSizeWithStaticSamplers > 0 &&
                         a.MaxSamplerDescriptorHeapSizeWithStaticSamplers >
                             b.MaxSamplerDescriptorHeapSizeWithStaticSamplers &&
                         mismatches++ < 32)
                         std::fprintf(stderr,
                                      "%s.MaxSamplerDescriptorHeapSizeWithStaticSamplers: ours %u > host %u\n", n,
                                      a.MaxSamplerDescriptorHeapSizeWithStaticSamplers,
                                      b.MaxSamplerDescriptorHeapSizeWithStaticSamplers);
                     if (b.MaxViewDescriptorHeapSize > 0 &&
                         a.MaxViewDescriptorHeapSize > b.MaxViewDescriptorHeapSize && mismatches++ < 32)
                         std::fprintf(stderr, "%s.MaxViewDescriptorHeapSize: ours %u > host %u\n", n,
                                      a.MaxViewDescriptorHeapSize, b.MaxViewDescriptorHeapSize);
                     check_bool(n, "ComputeOnlyCustomHeapSupported", a.ComputeOnlyCustomHeapSupported,
                                b.ComputeOnlyCustomHeapSupported);
                 });
    compare_host("OPTIONS20", D3D12_FEATURE_D3D12_OPTIONS20, o20,
                 [&](const char* n, const auto& a, const auto& b) {
                     check_bool(n, "ComputeOnlyWriteWatchSupported", a.ComputeOnlyWriteWatchSupported,
                                b.ComputeOnlyWriteWatchSupported);
                     check_tier(n, "RecreateAtTier", int(a.RecreateAtTier), int(b.RecreateAtTier));
                 });
    compare_host("OPTIONS21", D3D12_FEATURE_D3D12_OPTIONS21, o21,
                 [&](const char* n, const auto& a, const auto& b) {
                     check_tier(n, "WorkGraphsTier", int(a.WorkGraphsTier), int(b.WorkGraphsTier));
                     check_tier(n, "ExecuteIndirectTier", int(a.ExecuteIndirectTier), int(b.ExecuteIndirectTier));
                     check_bool(n, "SampleCmpGradientAndBiasSupported", a.SampleCmpGradientAndBiasSupported,
                                b.SampleCmpGradientAndBiasSupported);
                     check_bool(n, "ExtendedCommandInfoSupported", a.ExtendedCommandInfoSupported,
                                b.ExtendedCommandInfoSupported);
                 });

    std::printf("CheckFeatureSupport options (%s): %u feature queries compared (%u host-unsupported skipped), "
                "MSAA64KB=%d, NoOutputsSamples=0x%x, MaxViewHeap=%u, %d mismatches\n",
                adapter_name.empty() ? "host D3D12" : adapter_name.c_str(), host_compared, host_skipped,
                int(o4.MSAA64KBAlignedTextureSupported), o19.SupportedSampleCountsWithNoOutputs,
                o19.MaxViewDescriptorHeapSize, mismatches);
    return mismatches;
}

// A software Vulkan device (llvmpipe on CI) and the host's WARP device answer about different machines, so the
// OPTIONS1 comparison is skipped there; on a GPU both runtimes see the same hardware.
bool software_vulkan_device(const vktest::Device& vk) {
    const auto get_properties = reinterpret_cast<PFN_vkGetPhysicalDeviceProperties>(
        vk.instance_proc(vk.instance, "vkGetPhysicalDeviceProperties"));
    if (!get_properties) return true;
    VkPhysicalDeviceProperties properties{};
    get_properties(vk.physical, &properties);
    return properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU;
}

// D3D12_OPTIONS1 (D3D-86): wave ops and the lane range come from the Vulkan device's subgroup properties, so the
// host PC runtime on the same GPU is the oracle. Returns the number of mismatches, -2 for a software Vulkan device,
// or -1 when the host has no D3D12 device.
int compare_options1(ID3D12Device* ours, const vktest::Device& vk) {
    if (software_vulkan_device(vk)) return -2;
    Ref<ID3D12Device> host;
    if (!create_host_device(host)) return -1;
    D3D12_FEATURE_DATA_D3D12_OPTIONS1 theirs{};
    if (FAILED(host->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS1, &theirs, sizeof(theirs)))) return -1;
    D3D12_FEATURE_DATA_D3D12_OPTIONS1 mine{};
    if (FAILED(ours->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS1, &mine, sizeof(mine)))) return 1;
    int mismatches = 0;
    const auto same = [&](const char* field, UINT got, UINT want) {
        if (got == want) return;
        if (mismatches++ < 4)
            std::fprintf(stderr, "D3D12_OPTIONS1 %s: ours %u, host %u\n", field, got, want);
    };
    same("WaveOps", mine.WaveOps, theirs.WaveOps);
    same("WaveLaneCountMin", mine.WaveLaneCountMin, theirs.WaveLaneCountMin);
    same("WaveLaneCountMax", mine.WaveLaneCountMax, theirs.WaveLaneCountMax);
    same("ExpandedComputeResourceStates", mine.ExpandedComputeResourceStates,
         theirs.ExpandedComputeResourceStates);
    // TotalLaneCount is deliberately not compared: the host counts every lane of the device (6144 on the RTX 5070)
    // and the layer answers one full wave (docs section 4.1).
    std::printf("D3D12_OPTIONS1: lanes %u..%u, total %u (host total %u)\n", mine.WaveLaneCountMin,
                mine.WaveLaneCountMax, mine.TotalLaneCount, theirs.TotalLaneCount);
    return mismatches;
}

}  // namespace

int main() {
    vktest::Device d;
    const auto status = d.initialize();
    if (status == vktest::DeviceStatus::unavailable) return 77;
    if (status == vktest::DeviceStatus::failed) return 1;
    EXPECT(d.depth_format_usable(VK_FORMAT_D32_SFLOAT_S8_UINT));
    VkSampleCountFlags depth_sample_counts = 0, color_sample_counts = 0;
    {
        VkPhysicalDeviceProperties properties{};
        d.instance_fn<PFN_vkGetPhysicalDeviceProperties>("vkGetPhysicalDeviceProperties")(d.physical, &properties);
        depth_sample_counts =
            properties.limits.framebufferDepthSampleCounts & properties.limits.framebufferStencilSampleCounts;
        color_sample_counts = properties.limits.framebufferColorSampleCounts;
    }
    constexpr UINT size = 8, small_size = 4;
    unsigned mismatches = 0;
    const auto report = [&](const char* what, UINT index, const std::uint8_t* got, const std::uint8_t* want, size_t n) {
        if (std::memcmp(got, want, n) == 0) return;
        if (mismatches++ < 8) {
            std::fprintf(stderr, "%s %u: got", what, index);
            for (size_t i = 0; i < n; ++i) std::fprintf(stderr, " %02x", got[i]);
            std::fprintf(stderr, ", want");
            for (size_t i = 0; i < n; ++i) std::fprintf(stderr, " %02x", want[i]);
            std::fprintf(stderr, "\n");
        }
    };

    {
        rt::d3d12::VulkanDeviceInfo info{};
        info.get_instance_proc = d.instance_proc;
        info.instance = d.instance;
        info.physical = d.physical;
        info.device = d.device;
        info.queue = d.queue;
        info.queue_family = d.family;
        info.texture_compression_bc = d.bc;
        info.occlusion_query_precise = d.occlusion_query_precise;
        info.pipeline_statistics_query = d.pipeline_statistics_query;
        ID3D12Device* inner = nullptr;
        EXPECT(SUCCEEDED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&inner))));
        Ref<ID3D12Device> device;
        EXPECT(SUCCEEDED(rt::d3d12_xs::create_device_proxy(inner, __uuidof(ID3D12Device),
                                                           reinterpret_cast<void**>(&device.p))));
        rt::d3d12::set_device_outer(inner, device.p);  // the proxy now owns `inner`
        EXPECT(rt::d3d12_xs::is_device_proxy(device.p));
        EXPECT(rt::d3d12_xs::device_proxy_slot_count(device.p) == 80);  // ID3D12Device11

        {
            Ref<ID3D12Device11> device11;
            Ref<IUnknown> device12;
            EXPECT(SUCCEEDED(device->QueryInterface(IID_PPV_ARGS(&device11.p))) && device11.p == device.p);
            EXPECT(device->QueryInterface(__uuidof(ID3D12Device12), reinterpret_cast<void**>(&device12.p)) ==
                   E_NOINTERFACE);
        }
        EXPECT(device->GetNodeCount() == 1);

        // BC1 can be sampled, but it cannot be an attachment or a storage image.
        if (d.bc) {
            const D3D12_HEAP_PROPERTIES heap{D3D12_HEAP_TYPE_DEFAULT};
            Ref<ID3D12Device> host;
            create_host_device(host);
            for (const auto flags : {D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
                                     D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS}) {
                const auto desc = texture_desc(8, 8, DXGI_FORMAT_BC1_UNORM, flags);
                Ref<ID3D12Resource> image;
                const auto create = [&](ID3D12Device* dev, Ref<ID3D12Resource>& out) {
                    return dev->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON,
                                                        nullptr, IID_PPV_ARGS(&out.p));
                };
                const HRESULT hr = create(device.p, image);
                const bool supported = flags == D3D12_RESOURCE_FLAG_NONE;
                EXPECT(hr == (supported ? S_OK : E_INVALIDARG));
                EXPECT((image.p != nullptr) == supported);
                if (host.p) {
                    Ref<ID3D12Resource> native;
                    EXPECT(create(host.p, native) == hr);
                    EXPECT((native.p != nullptr) == supported);
                }
                const auto allocation = device->GetResourceAllocationInfo(0, 1, &desc);
                EXPECT((allocation.SizeInBytes != UINT64_MAX) == supported);
            }
        }
        const D3D_FEATURE_LEVEL levels[] = {D3D_FEATURE_LEVEL_11_0, D3D_FEATURE_LEVEL_12_1, D3D_FEATURE_LEVEL_12_2};
        D3D12_FEATURE_DATA_FEATURE_LEVELS feature_levels{3, levels, D3D_FEATURE_LEVEL(0)};
        EXPECT(SUCCEEDED(device->CheckFeatureSupport(D3D12_FEATURE_FEATURE_LEVELS, &feature_levels,
                                                     sizeof(feature_levels))));
        EXPECT(feature_levels.MaxSupportedFeatureLevel == D3D_FEATURE_LEVEL_12_1);
        D3D12_FEATURE_DATA_ARCHITECTURE architecture{};
        EXPECT(SUCCEEDED(device->CheckFeatureSupport(D3D12_FEATURE_ARCHITECTURE, &architecture, sizeof(architecture))));
        D3D12_FEATURE_DATA_D3D12_OPTIONS options{};
        EXPECT(SUCCEEDED(device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, &options, sizeof(options))));
        EXPECT(options.ResourceBindingTier == D3D12_RESOURCE_BINDING_TIER_3);
        EXPECT(options.MaxGPUVirtualAddressBitsPerResource == 40);

        D3D12_FEATURE_DATA_GPU_VIRTUAL_ADDRESS_SUPPORT va_support{};
        EXPECT(SUCCEEDED(device->CheckFeatureSupport(D3D12_FEATURE_GPU_VIRTUAL_ADDRESS_SUPPORT, &va_support, sizeof(va_support))));
        EXPECT(va_support.MaxGPUVirtualAddressBitsPerResource == 40);
        EXPECT(va_support.MaxGPUVirtualAddressBitsPerProcess == 48);

        EXPECT(compare_format_info(device.p, d) == 0);
        // Use a real oversized buffer and check that rejecting it leaves the output untouched.
        D3D12_FEATURE_DATA_FORMAT_INFO format_info[2]{{DXGI_FORMAT_D24_UNORM_S8_UINT, 0xff}};
        EXPECT(device->CheckFeatureSupport(D3D12_FEATURE_FORMAT_INFO, nullptr, sizeof(format_info[0])) == E_INVALIDARG);
        EXPECT(device->CheckFeatureSupport(D3D12_FEATURE_FORMAT_INFO, format_info, sizeof(format_info)) == E_INVALIDARG);
        EXPECT(format_info[0].PlaneCount == 0xff);
        EXPECT(device->CheckFeatureSupport(D3D12_FEATURE_FORMAT_INFO, format_info, sizeof(format_info[0])) == S_OK);
        EXPECT(format_info[0].PlaneCount == 2);

        D3D12_FEATURE_DATA_FORMAT_SUPPORT format_support{DXGI_FORMAT_R8G8B8A8_UNORM};
        EXPECT(SUCCEEDED(device->CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT, &format_support, sizeof(format_support))));
        EXPECT(format_support.Support1 & D3D12_FORMAT_SUPPORT1_TEXTURE2D);
        EXPECT(format_support.Support1 & D3D12_FORMAT_SUPPORT1_RENDER_TARGET);
        EXPECT(format_support.Support1 & D3D12_FORMAT_SUPPORT1_SHADER_SAMPLE);

        D3D12_FEATURE_DATA_MULTISAMPLE_QUALITY_LEVELS invalid_sample_count{};
        invalid_sample_count.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        invalid_sample_count.SampleCount = D3D12_MAX_MULTISAMPLE_SAMPLE_COUNT * 2;
        EXPECT(device->CheckFeatureSupport(D3D12_FEATURE_MULTISAMPLE_QUALITY_LEVELS, &invalid_sample_count,
                                           sizeof(invalid_sample_count)) == E_INVALIDARG);
        EXPECT(invalid_sample_count.NumQualityLevels == 0);
        invalid_sample_count.SampleCount = 0;
        EXPECT(device->CheckFeatureSupport(D3D12_FEATURE_MULTISAMPLE_QUALITY_LEVELS, &invalid_sample_count,
                                           sizeof(invalid_sample_count)) == E_INVALIDARG);

        D3D12_RESOURCE_DESC unsupported_quality =
            texture_desc(size, size, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET);
        unsupported_quality.SampleDesc.Count = 4;
        unsupported_quality.SampleDesc.Quality = 1;
        EXPECT(device->GetResourceAllocationInfo(0, 1, &unsupported_quality).SizeInBytes == UINT64_MAX);

        D3D12_FEATURE_DATA_FORMAT_SUPPORT invalid_format{DXGI_FORMAT_UNKNOWN};
        EXPECT(FAILED(device->CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT, &invalid_format, sizeof(invalid_format))));

        // D3D-61: XR_BIAS is a display-only format with the R10G10B10A2 bit layout. The layer can create and
        // copy it, and its views can cast to the ordinary 10:10:10:2 format, but it does not claim display or
        // video-processor support.
        {
            const D3D12_HEAP_PROPERTIES default_heap{D3D12_HEAP_TYPE_DEFAULT};
            D3D12_FEATURE_DATA_FORMAT_SUPPORT xr{DXGI_FORMAT_R10G10B10_XR_BIAS_A2_UNORM};
            EXPECT(SUCCEEDED(device->CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT, &xr, sizeof(xr))));
            EXPECT(xr.Support1 & D3D12_FORMAT_SUPPORT1_TEXTURE2D);
            EXPECT(xr.Support1 & D3D12_FORMAT_SUPPORT1_CAST_WITHIN_BIT_LAYOUT);
            EXPECT(!(xr.Support1 & (D3D12_FORMAT_SUPPORT1_TEXTURE3D | D3D12_FORMAT_SUPPORT1_MIP |
                                    D3D12_FORMAT_SUPPORT1_SHADER_SAMPLE | D3D12_FORMAT_SUPPORT1_RENDER_TARGET |
                                    D3D12_FORMAT_SUPPORT1_BLENDABLE)));
            D3D12_FEATURE_DATA_MULTISAMPLE_QUALITY_LEVELS multsample{DXGI_FORMAT_R10G10B10_XR_BIAS_A2_UNORM, 4};
            EXPECT(SUCCEEDED(device->CheckFeatureSupport(D3D12_FEATURE_MULTISAMPLE_QUALITY_LEVELS, &multsample,
                                                         sizeof(multsample))));
            EXPECT(multsample.NumQualityLevels == 0);
            Ref<ID3D12Resource> tex;
            const D3D12_RESOURCE_DESC desc = texture_desc(
                16, 16, DXGI_FORMAT_R10G10B10_XR_BIAS_A2_UNORM, D3D12_RESOURCE_FLAG_NONE, 1);
            EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &desc,
                                                             D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                             IID_PPV_ARGS(&tex.p))));
            EXPECT(tex.p != nullptr);

            // The host refuses extra mips, multisampling and 3D; the layer must refuse the same descriptions.
            D3D12_RESOURCE_DESC invalid = desc;
            invalid.MipLevels = 2;
            EXPECT(FAILED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &invalid,
                                                          D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                          __uuidof(ID3D12Resource), nullptr)));
            EXPECT(device->GetResourceAllocationInfo(0, 1, &invalid).SizeInBytes == UINT64_MAX);
            D3D12_PLACED_SUBRESOURCE_FOOTPRINT layout{};
            UINT rows = 0;
            UINT64 total = 0;
            device->GetCopyableFootprints(&invalid, 0, 1, 0, &layout, &rows, nullptr, &total);
            EXPECT(total == UINT64_MAX);

            invalid = desc;
            invalid.SampleDesc.Count = 4;
            invalid.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
            EXPECT(FAILED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &invalid,
                                                          D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                          __uuidof(ID3D12Resource), nullptr)));
            EXPECT(device->GetResourceAllocationInfo(0, 1, &invalid).SizeInBytes == UINT64_MAX);

            invalid = desc;
            invalid.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE3D;
            invalid.DepthOrArraySize = 4;
            EXPECT(FAILED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &invalid,
                                                          D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                          __uuidof(ID3D12Resource), nullptr)));
            EXPECT(device->GetResourceAllocationInfo(0, 1, &invalid).SizeInBytes == UINT64_MAX);
        }

        // D3D-60: CheckFeatureSupport and committed resource creation for integer and typeless formats
        {
            const D3D12_HEAP_PROPERTIES default_heap{D3D12_HEAP_TYPE_DEFAULT};
            for (DXGI_FORMAT f : {
                     DXGI_FORMAT_R32G32_TYPELESS, DXGI_FORMAT_R32G32_UINT, DXGI_FORMAT_R32G32_SINT,
                     DXGI_FORMAT_R16G16B16A16_UINT, DXGI_FORMAT_R16G16B16A16_SINT,
                     DXGI_FORMAT_R10G10B10A2_UINT, DXGI_FORMAT_R16G16_TYPELESS, DXGI_FORMAT_R16G16_UINT,
                     DXGI_FORMAT_R16G16_SINT, DXGI_FORMAT_R8G8_TYPELESS, DXGI_FORMAT_R8G8_UINT,
                     DXGI_FORMAT_R8G8_SINT}) {
                D3D12_FEATURE_DATA_FORMAT_SUPPORT fs{f};
                EXPECT(SUCCEEDED(device->CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT, &fs, sizeof(fs))));
                EXPECT(fs.Support1 & D3D12_FORMAT_SUPPORT1_TEXTURE2D);
                Ref<ID3D12Resource> tex;
                const D3D12_RESOURCE_DESC desc = texture_desc(16, 16, f, D3D12_RESOURCE_FLAG_NONE, 1);
                EXPECT(SUCCEEDED(device->CreateCommittedResource(
                    &default_heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON, nullptr,
                    IID_PPV_ARGS(&tex.p))));
                EXPECT(tex.p != nullptr);
            }
        }

        const int options_mismatches = compare_feature_options(device.p, info);
        EXPECT(options_mismatches == 0);

        // Push constants limit check (D3D-14): Vulkan spec guarantees >= 128 bytes.
        EXPECT(rt::d3d12::device_max_push_constants_size(device.p) >= 128);
        EXPECT(rt::d3d12::device_max_push_constants_size(inner) == rt::d3d12::device_max_push_constants_size(device.p));
        // The test GPU may have either answer (Intel Arc and a Radeon 680M on driver 22.40.84.06 report 128 bytes);
        // the device must flag the fallback exactly when the 144-byte block does not fit.
        EXPECT(rt::d3d12::device_push_constants_need_fallback(device.p) ==
               (rt::d3d12::device_max_push_constants_size(device.p) < 144));

        // Simulated fallback path when maxPushConstantsSize < 144 bytes:
        {
            rt::d3d12::VulkanDeviceInfo fallback_info = info;
            fallback_info.texture_compression_bc = false;
            fallback_info.max_push_constants_size = 128;
            fallback_info.push_constants_need_fallback = true;
            ID3D12Device* fallback_inner = nullptr;
            EXPECT(SUCCEEDED(rt::d3d12::create_vulkan_device(fallback_info, IID_PPV_ARGS(&fallback_inner))));
            Ref<ID3D12Device> fallback_device;
            EXPECT(SUCCEEDED(rt::d3d12_xs::create_device_proxy(fallback_inner, __uuidof(ID3D12Device),
                                                               reinterpret_cast<void**>(&fallback_device.p))));
            rt::d3d12::set_device_outer(fallback_inner, fallback_device.p);
            D3D12_FEATURE_DATA_D3D12_OPTIONS8 options8{TRUE};
            EXPECT(fallback_device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS8, &options8,
                                                        sizeof(options8)) == S_OK);
            EXPECT(options8.UnalignedBlockTexturesSupported == FALSE);
            EXPECT(rt::d3d12::device_max_push_constants_size(fallback_device.p) == 128);
            EXPECT(rt::d3d12::device_push_constants_need_fallback(fallback_device.p));
        }

        // The Windows 10 footprint skip depends on this classification; check it on every host.
        EXPECT(uses_unaligned_blocks(texture_desc(10, 6, DXGI_FORMAT_BC1_UNORM, D3D12_RESOURCE_FLAG_NONE)));
        EXPECT(!uses_unaligned_blocks(texture_desc(8, 8, DXGI_FORMAT_BC1_UNORM, D3D12_RESOURCE_FLAG_NONE)));
        EXPECT(!uses_unaligned_blocks(texture_desc(10, 6, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_NONE)));

        // D3D12_OPTIONS1 (D3D-86): the lane range is the Vulkan device's own subgroup range and the device-wide lane
        // count is deliberately one wave, so those are pinned here rather than only compared. Wave ops can only be
        // promised by a GPU, and a software Vulkan device has no host GPU to compare with either.
        {
            D3D12_FEATURE_DATA_D3D12_OPTIONS1 wave{};
            EXPECT(SUCCEEDED(device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS1, &wave, sizeof(wave))));
            EXPECT(wave.WaveLaneCountMin >= 1);
            EXPECT(wave.WaveLaneCountMax >= wave.WaveLaneCountMin);
            EXPECT(wave.TotalLaneCount == wave.WaveLaneCountMax);
            EXPECT(wave.ExpandedComputeResourceStates);
            const int options1 = compare_options1(device.p, d);
            if (options1 == -2)
                std::printf("D3D12_OPTIONS1: software Vulkan device, host comparison skipped\n");
            else if (options1 < 0)
                std::printf("D3D12_OPTIONS1: no host D3D12 device to compare with, skipped\n");
            EXPECT(options1 <= 0);
            if (options1 != -2) EXPECT(wave.WaveOps);
            D3D12_FEATURE_DATA_D3D12_OPTIONS1 shrunk{};
            EXPECT(FAILED(device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS1, &shrunk, sizeof(shrunk) - 4)));
            EXPECT(FAILED(device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS1, nullptr, sizeof(shrunk))));
        }
        const int footprint_mismatches = compare_footprints(device.p);
        if (footprint_mismatches < 0)
            std::printf("GetCopyableFootprints: no host D3D12 device to compare with, skipped\n");
        EXPECT(footprint_mismatches <= 0);

        // Shader model and root signature (D3D-86): the answers are only useful if the game can rely on them, so
        // pin both the ceiling and the refusal cases rather than just comparing with the host.
        {
            D3D12_FEATURE_DATA_SHADER_MODEL model{D3D_SHADER_MODEL_6_0};
            EXPECT(SUCCEEDED(device->CheckFeatureSupport(D3D12_FEATURE_SHADER_MODEL, &model, sizeof(model))));
            EXPECT(model.HighestShaderModel == D3D_SHADER_MODEL_6_0);
            // 5.1 is D3D12's baseline and the host answers it, so a caller asking for the minimum must get it back.
            D3D12_FEATURE_DATA_SHADER_MODEL baseline{D3D_SHADER_MODEL_5_1};
            EXPECT(SUCCEEDED(device->CheckFeatureSupport(D3D12_FEATURE_SHADER_MODEL, &baseline, sizeof(baseline))));
            EXPECT(baseline.HighestShaderModel == D3D_SHADER_MODEL_5_1);
            D3D12_FEATURE_DATA_SHADER_MODEL above{static_cast<D3D_SHADER_MODEL>(0x69)};
            EXPECT(SUCCEEDED(device->CheckFeatureSupport(D3D12_FEATURE_SHADER_MODEL, &above, sizeof(above))));
            EXPECT(above.HighestShaderModel == D3D_SHADER_MODEL_6_0);  // clamped to what the layer accepts
            D3D12_FEATURE_DATA_SHADER_MODEL below{static_cast<D3D_SHADER_MODEL>(0)};
            EXPECT(FAILED(device->CheckFeatureSupport(D3D12_FEATURE_SHADER_MODEL, &below, sizeof(below))));

            D3D12_FEATURE_DATA_ROOT_SIGNATURE signature{D3D_ROOT_SIGNATURE_VERSION_1_1};
            EXPECT(SUCCEEDED(device->CheckFeatureSupport(D3D12_FEATURE_ROOT_SIGNATURE, &signature, sizeof(signature))));
            EXPECT(signature.HighestVersion == D3D_ROOT_SIGNATURE_VERSION_1_1);
            D3D12_FEATURE_DATA_ROOT_SIGNATURE newer{static_cast<D3D_ROOT_SIGNATURE_VERSION>(0x3)};
            EXPECT(SUCCEEDED(device->CheckFeatureSupport(D3D12_FEATURE_ROOT_SIGNATURE, &newer, sizeof(newer))));
            EXPECT(newer.HighestVersion == D3D_ROOT_SIGNATURE_VERSION_1_1);
            D3D12_FEATURE_DATA_ROOT_SIGNATURE undefined{static_cast<D3D_ROOT_SIGNATURE_VERSION>(0)};
            EXPECT(FAILED(device->CheckFeatureSupport(D3D12_FEATURE_ROOT_SIGNATURE, &undefined, sizeof(undefined))));
            // A wrong DataSize and a null buffer are refused for both queries.
            D3D12_FEATURE_DATA_SHADER_MODEL short_model{D3D_SHADER_MODEL_6_0};
            D3D12_FEATURE_DATA_ROOT_SIGNATURE short_signature{D3D_ROOT_SIGNATURE_VERSION_1_1};
            EXPECT(FAILED(device->CheckFeatureSupport(D3D12_FEATURE_SHADER_MODEL, &short_model,
                                                      sizeof(short_model) - 1)));
            EXPECT(FAILED(device->CheckFeatureSupport(D3D12_FEATURE_ROOT_SIGNATURE, &short_signature,
                                                      sizeof(short_signature) - 1)));
            EXPECT(FAILED(device->CheckFeatureSupport(D3D12_FEATURE_SHADER_MODEL, nullptr, sizeof(short_model))));
            EXPECT(FAILED(device->CheckFeatureSupport(D3D12_FEATURE_ROOT_SIGNATURE, nullptr, sizeof(short_signature))));
        }
        const int feature_mismatches = compare_feature_support(device.p);
        if (feature_mismatches < 0)
            std::printf("CheckFeatureSupport: no host D3D12 device to compare with, skipped\n");
        EXPECT(feature_mismatches <= 0);
        const int allocation_mismatches = compare_allocation_info(device.p);
        if (allocation_mismatches < 0)
            std::printf("GetResourceAllocationInfo: no host D3D12 device to compare with, skipped\n");
        EXPECT(allocation_mismatches <= 0);

        // D3D-46: copies against the host runtime (d3d12_copy_compare.h).
        const int copy_mismatches = copycmp::compare_copies(device.p, d.bc, d.depth_format_usable(VK_FORMAT_D24_UNORM_S8_UINT));
        if (copy_mismatches < 0) std::printf("Copies: no host D3D12 device to compare with, skipped\n");
        EXPECT(copy_mismatches <= 0);
        const int tiling_mismatches = compare_tiling(device.p);
        if (tiling_mismatches < 0)
            std::printf("GetResourceTiling: no host D3D12 device with tiled resources, or no layout it accepts; "
                        "skipped\n");
        EXPECT(tiling_mismatches <= 0);

        const int depth_quality_mismatches = compare_depth_multisample_quality_levels(device.p, depth_sample_counts);
        if (depth_quality_mismatches < 0)
            std::printf("Depth/stencil MSAA quality queries: no host D3D12 device to compare with, skipped\n");
        EXPECT(depth_quality_mismatches <= 0);

        const int resolve_mismatches = compare_resolves(device.p, color_sample_counts);
        if (resolve_mismatches < 0)
            std::printf("Resolve comparison: no host D3D12 device to compare with, skipped\n");
        EXPECT(resolve_mismatches <= 0);

        // Command objects; their GetDevice is the proxy, never the inner device.
        D3D12_COMMAND_QUEUE_DESC queue_desc{};
        queue_desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
        Ref<ID3D12CommandQueue> queue;
        Ref<ID3D12CommandAllocator> allocator;
        Ref<ID3D12GraphicsCommandList> list;
        Ref<ID3D12Fence> fence;
        EXPECT(SUCCEEDED(device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue.p))));
        EXPECT(SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator.p))));
        EXPECT(SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.p, nullptr,
                                                   IID_PPV_ARGS(&list.p))));
        EXPECT(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))));
        {
            Ref<ID3D12Device> parent;
            EXPECT(SUCCEEDED(queue->GetDevice(IID_PPV_ARGS(&parent.p))) && parent.p == device.p);
        }
        // Residency and power (D3D-33): no-ops on Vulkan that still check their arguments, through the proxy's
        // forwarded slots. EnqueueMakeResident must signal its fence, or a queue waiting on it would hang.
        {
            Ref<ID3D12Fence> residency;
            EXPECT(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&residency.p))));
            ID3D12Pageable* const pageables[] = {fence.p, allocator.p};
            EXPECT(device->MakeResident(2, pageables) == S_OK);
            EXPECT(device->MakeResident(0, nullptr) == S_OK);
            EXPECT(device->MakeResident(1, nullptr) == E_INVALIDARG);
            EXPECT(device->Evict(2, pageables) == S_OK);
            EXPECT(device->Evict(0, nullptr) == S_OK);
            EXPECT(device->Evict(1, nullptr) == E_INVALIDARG);
            EXPECT(device->SetStablePowerState(TRUE) == S_OK);
            EXPECT(device->SetStablePowerState(FALSE) == S_OK);

            Ref<ID3D12Device1> device1;
            EXPECT(SUCCEEDED(device->QueryInterface(IID_PPV_ARGS(&device1.p))));
            const D3D12_RESIDENCY_PRIORITY priorities[] = {D3D12_RESIDENCY_PRIORITY_MAXIMUM,
                                                           D3D12_RESIDENCY_PRIORITY(D3D12_RESIDENCY_PRIORITY_LOW + 5)};
            EXPECT(device1->SetResidencyPriority(2, pageables, priorities) == S_OK);
            EXPECT(device1->SetResidencyPriority(0, nullptr, nullptr) == S_OK);
            EXPECT(device1->SetResidencyPriority(1, pageables, nullptr) == E_INVALIDARG);
            EXPECT(device1->SetResidencyPriority(1, nullptr, priorities) == E_INVALIDARG);

            Ref<ID3D12Device3> device3;
            EXPECT(SUCCEEDED(device->QueryInterface(IID_PPV_ARGS(&device3.p))));
            EXPECT(device3->EnqueueMakeResident(D3D12_RESIDENCY_FLAG_NONE, 2, pageables, residency.p, 7) == S_OK);
            EXPECT(residency->GetCompletedValue() == 7);
            EXPECT(device3->EnqueueMakeResident(D3D12_RESIDENCY_FLAG_DENY_OVERBUDGET, 0, nullptr, residency.p, 9) ==
                   S_OK);
            EXPECT(residency->GetCompletedValue() == 9);
            // Refused calls leave the fence alone.
            EXPECT(device3->EnqueueMakeResident(D3D12_RESIDENCY_FLAG_NONE, 1, pageables, nullptr, 11) == E_INVALIDARG);
            EXPECT(device3->EnqueueMakeResident(D3D12_RESIDENCY_FLAG_NONE, 1, nullptr, residency.p, 11) ==
                   E_INVALIDARG);
            EXPECT(device3->EnqueueMakeResident(D3D12_RESIDENCY_FLAGS(0x2), 1, pageables, residency.p, 11) ==
                   E_INVALIDARG);
            EXPECT(residency->GetCompletedValue() == 9);
            // The GPU sees the signal: a queue waits on it, then signals past it.
            EXPECT(SUCCEEDED(queue->Wait(residency.p, 9)));
            EXPECT(SUCCEEDED(queue->Signal(residency.p, 10)));
            EXPECT(SUCCEEDED(residency->SetEventOnCompletion(10, nullptr)) && residency->GetCompletedValue() == 10);

            Ref<ID3D12Device6> device6;
            EXPECT(SUCCEEDED(device->QueryInterface(IID_PPV_ARGS(&device6.p))));
            const HANDLE done = CreateEventW(nullptr, TRUE, FALSE, nullptr);
            EXPECT(done);
            BOOL further = TRUE;
            const HRESULT background = device6->SetBackgroundProcessingMode(
                D3D12_BACKGROUND_PROCESSING_MODE_DISABLE_BACKGROUND_WORK, D3D12_MEASUREMENTS_ACTION_COMMIT_RESULTS,
                done, &further);
            const DWORD signalled = WaitForSingleObject(done, 0);
            CloseHandle(done);
            EXPECT(background == S_OK && signalled == WAIT_OBJECT_0 && further == FALSE);
            EXPECT(device6->SetBackgroundProcessingMode(D3D12_BACKGROUND_PROCESSING_MODE_ALLOWED,
                                                        D3D12_MEASUREMENTS_ACTION_KEEP_ALL, nullptr, nullptr) == S_OK);
            EXPECT(device6->SetBackgroundProcessingMode(D3D12_BACKGROUND_PROCESSING_MODE(4),
                                                        D3D12_MEASUREMENTS_ACTION_KEEP_ALL, nullptr, nullptr) ==
                   E_INVALIDARG);
            EXPECT(device6->SetBackgroundProcessingMode(D3D12_BACKGROUND_PROCESSING_MODE_ALLOWED,
                                                        D3D12_MEASUREMENTS_ACTION(4), nullptr, nullptr) ==
                   E_INVALIDARG);
        }
        {
            Ref<ID3D12Device4> device4;
            Ref<ID3D12GraphicsCommandList> closed;
            EXPECT(SUCCEEDED(device->QueryInterface(IID_PPV_ARGS(&device4.p))));
            EXPECT(SUCCEEDED(device4->CreateCommandList1(0, D3D12_COMMAND_LIST_TYPE_DIRECT,
                                                         D3D12_COMMAND_LIST_FLAG_NONE, IID_PPV_ARGS(&closed.p))));
            EXPECT(closed->Close() == E_FAIL);  // CreateCommandList1 returns a closed list
        }

        // Descriptor heaps from slot 14, start handles in the Xbox shape.
        D3D12_DESCRIPTOR_HEAP_DESC heap_desc{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 3, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
        Ref<ID3D12DescriptorHeap> rtv_heap, dsv_heap;
        EXPECT(SUCCEEDED(device->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&rtv_heap.p))));
        heap_desc = {D3D12_DESCRIPTOR_HEAP_TYPE_DSV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
        EXPECT(SUCCEEDED(device->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&dsv_heap.p))));
        EXPECT(rt::d3d12_xs::is_heap_proxy(rtv_heap.p) && rt::d3d12_xs::is_heap_proxy(dsv_heap.p));
        const std::uint64_t rtv_start = reinterpret_cast<XboxHeapHandleFn>(slot(rtv_heap.p, 9))(rtv_heap.p);
        const std::uint64_t dsv_start = reinterpret_cast<XboxHeapHandleFn>(slot(dsv_heap.p, 9))(dsv_heap.p);
        const UINT increment = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
        EXPECT(rtv_start && dsv_start && increment >= sizeof(recity::TargetView));
        const D3D12_CPU_DESCRIPTOR_HANDLE rtv0{SIZE_T(rtv_start)}, rtv1{SIZE_T(rtv_start + increment)},
            rtv565{SIZE_T(rtv_start + 2 * increment)}, dsv0{SIZE_T(dsv_start)};

        // Resources: the render target through slot 76 with the Xbox tiled layout word, the others through the
        // PC CreateCommittedResource.
        const D3D12_HEAP_PROPERTIES default_heap{D3D12_HEAP_TYPE_DEFAULT};
        const D3D12_HEAP_PROPERTIES upload_heap{D3D12_HEAP_TYPE_UPLOAD};
        const D3D12_HEAP_PROPERTIES readback_heap{D3D12_HEAP_TYPE_READBACK};
        const auto create = [&](const D3D12_HEAP_PROPERTIES& heap, const D3D12_RESOURCE_DESC& desc,
                                D3D12_RESOURCE_STATES state, Ref<ID3D12Resource>& out) {
            return device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, state, nullptr,
                                                   IID_PPV_ARGS(&out.p));
        };
        Ref<ID3D12Resource> refused_buffer;
        D3D12_RESOURCE_DESC invalid_buffer_samples = buffer_desc(256);
        invalid_buffer_samples.SampleDesc.Count = 0;
        EXPECT(create(default_heap, invalid_buffer_samples, D3D12_RESOURCE_STATE_COPY_DEST, refused_buffer) ==
               E_INVALIDARG && !refused_buffer.p);
        invalid_buffer_samples.SampleDesc.Count = 1;
        invalid_buffer_samples.SampleDesc.Quality = 1;
        EXPECT(create(default_heap, invalid_buffer_samples, D3D12_RESOURCE_STATE_COPY_DEST, refused_buffer) ==
               E_INVALIDARG && !refused_buffer.p);

        Ref<ID3D12Resource> rt_resource, packed565, ds, untouched, upload, readback, volume;
        D3D12_RESOURCE_DESC xbox_rt =
            texture_desc(size, size, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET);
        xbox_rt.Layout = static_cast<D3D12_TEXTURE_LAYOUT>(0x101);
        const IID resource_iid = __uuidof(ID3D12Resource);
        EXPECT(SUCCEEDED(reinterpret_cast<XboxCreateResourceFn>(slot(device.p, 76))(
            device.p, &default_heap, D3D12_HEAP_FLAG_NONE, &xbox_rt, D3D12_RESOURCE_STATE_COMMON, nullptr,
            &resource_iid, reinterpret_cast<void**>(&rt_resource.p))));
        EXPECT(SUCCEEDED(create(default_heap,
                                texture_desc(size, size, DXGI_FORMAT_B5G6R5_UNORM, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET),
                                D3D12_RESOURCE_STATE_RENDER_TARGET, packed565)));
        EXPECT(SUCCEEDED(create(default_heap,
                                texture_desc(size, size, DXGI_FORMAT_D32_FLOAT_S8X24_UINT,
                                             D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL),
                                D3D12_RESOURCE_STATE_DEPTH_WRITE, ds)));
        EXPECT(SUCCEEDED(create(default_heap,
                                texture_desc(small_size, small_size, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_NONE),
                                D3D12_RESOURCE_STATE_COPY_SOURCE, untouched)));
        EXPECT(SUCCEEDED(create(upload_heap, buffer_desc(65536), D3D12_RESOURCE_STATE_GENERIC_READ, upload)));
        EXPECT(SUCCEEDED(create(readback_heap, buffer_desc(65536), D3D12_RESOURCE_STATE_COPY_DEST, readback)));
        EXPECT(SUCCEEDED(create(default_heap, volume_desc(0), D3D12_RESOURCE_STATE_COPY_DEST, volume)));
        EXPECT(volume->GetDesc().MipLevels == 5);
        EXPECT(rt_resource->GetDesc().Width == size && rt_resource->GetDesc().Layout == D3D12_TEXTURE_LAYOUT_UNKNOWN);
        EXPECT(rt::d3d12::vulkan_image(rt_resource.p) && !rt::d3d12::vulkan_image(upload.p));

        // Upload buffers are mapped and zeroed.
        std::uint8_t* up = nullptr;
        std::uint8_t* down = nullptr;
        EXPECT(SUCCEEDED(upload->Map(0, nullptr, reinterpret_cast<void**>(&up))) && up);
        EXPECT(SUCCEEDED(readback->Map(0, nullptr, reinterpret_cast<void**>(&down))) && down);
        bool zeroed = true;
        for (int i = 0; i < 65536; ++i) zeroed = zeroed && up[i] == 0;
        EXPECT(zeroed);
        D3D12_HEAP_PROPERTIES heap_properties{};
        EXPECT(SUCCEEDED(upload->GetHeapProperties(&heap_properties, nullptr)) &&
               heap_properties.Type == D3D12_HEAP_TYPE_UPLOAD);

        // D3D-96: upload buffers are in VRAM exactly when the GPU has Resizable BAR (and RECITY_UPLOAD_SYSMEM is not
        // set), host-visible either way. The copies below read this buffer on the GPU, so both placements are checked.
        {
            VkPhysicalDeviceMemoryProperties memory{};
            d.instance_fn<PFN_vkGetPhysicalDeviceMemoryProperties>("vkGetPhysicalDeviceMemoryProperties")(d.physical,
                                                                                                          &memory);
            char text[4] = {};
            size_t length = 0;
            const bool sysmem = getenv_s(&length, text, "RECITY_UPLOAD_SYSMEM") == 0 && length > 1 && text[0] != '0';
            const bool in_vram = rt::d3d12::device_upload_in_vram(device.p);
            EXPECT(in_vram == (recity::resizable_bar(memory) && !sysmem));
            const VkMemoryPropertyFlags upload_flags = rt::d3d12::vulkan_memory_flags(upload.p);
            EXPECT((upload_flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) &&
                   (upload_flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT));
            if (in_vram) EXPECT(upload_flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            EXPECT(rt::d3d12::vulkan_memory_flags(readback.p) & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
            std::printf("upload buffer memory: %s\n", in_vram ? "VRAM (Resizable BAR)" : "system memory");
        }

        // Views into the heaps.
        device->CreateRenderTargetView(rt_resource.p, nullptr, rtv0);
        D3D12_RENDER_TARGET_VIEW_DESC rtv_desc{};
        rtv_desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        rtv_desc.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2D;
        device->CreateRenderTargetView(rt_resource.p, &rtv_desc, rtv1);
        device->CreateDepthStencilView(ds.p, nullptr, dsv0);
        device->CreateRenderTargetView(packed565.p, nullptr, rtv565);

        // List 1: clears and barriers on never-written resources (their first use zero-fills them), then copies of
        // the colour, depth plane, stencil plane and the untouched texture into the readback buffer.
        UINT64 cursor = 0;
        const auto rt_place = place(device.p, rt_resource.p, 0, cursor);
        const auto packed_place = place(device.p, packed565.p, 0, cursor);
        const auto depth_place = place(device.p, ds.p, 0, cursor);
        const auto stencil_place = place(device.p, ds.p, 1, cursor);
        const auto untouched_place = place(device.p, untouched.p, 0, cursor);
        EXPECT(rt_place.Footprint.RowPitch == 256 && depth_place.Footprint.Format == DXGI_FORMAT_R32_TYPELESS &&
               stencil_place.Footprint.Format == DXGI_FORMAT_R8_TYPELESS);
        std::memset(down, 0xcd, 65536);

        D3D12_RESOURCE_BARRIER barrier =
            transition(rt_resource.p, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_RENDER_TARGET);
        list->ResourceBarrier(1, &barrier);
        const FLOAT red[4] = {1, 0, 0, 1}, blue[4] = {0, 0, 1, 1};
        const D3D12_RECT square{2, 2, 6, 6}, top_half{0, 0, 8, 4};
        list->ClearRenderTargetView(rtv0, red, 0, nullptr);
        list->ClearRenderTargetView(rtv1, blue, 1, &square);
        const FLOAT packed_full[4] = {1, 0.25f, 0, 1}, packed_rect[4] = {0, 0.75f, 1, 0};
        list->ClearRenderTargetView(rtv565, packed_full, 0, nullptr);
        list->ClearRenderTargetView(rtv565, packed_rect, 1, &square);
        barrier = transition(packed565.p, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_SOURCE);
        list->ResourceBarrier(1, &barrier);
        list->ClearDepthStencilView(dsv0, D3D12_CLEAR_FLAG_DEPTH | D3D12_CLEAR_FLAG_STENCIL, 0.25f, 0x42, 1, &top_half);
        barrier = transition(rt_resource.p, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_SOURCE);
        list->ResourceBarrier(1, &barrier);
        barrier = transition(ds.p, D3D12_RESOURCE_STATE_DEPTH_WRITE, D3D12_RESOURCE_STATE_COPY_SOURCE);
        list->ResourceBarrier(1, &barrier);
        Ref<ID3D12GraphicsCommandList7> list7;
        EXPECT(SUCCEEDED(list->QueryInterface(IID_PPV_ARGS(&list7.p))));
        D3D12_TEXTURE_BARRIER readable{};  // an enhanced barrier that only makes the untouched texture readable
        readable.SyncBefore = D3D12_BARRIER_SYNC_NONE;
        readable.SyncAfter = D3D12_BARRIER_SYNC_COPY;
        readable.AccessBefore = D3D12_BARRIER_ACCESS_NO_ACCESS;
        readable.AccessAfter = D3D12_BARRIER_ACCESS_COPY_SOURCE;
        readable.LayoutBefore = D3D12_BARRIER_LAYOUT_COPY_SOURCE;
        readable.LayoutAfter = D3D12_BARRIER_LAYOUT_COPY_SOURCE;
        readable.pResource = untouched.p;
        readable.Subresources = {0xffffffff, 0, 0, 0, 0, 0};
        D3D12_BARRIER_GROUP group{};
        group.Type = D3D12_BARRIER_TYPE_TEXTURE;
        group.NumBarriers = 1;
        group.pTextureBarriers = &readable;
        list7->Barrier(1, &group);
        for (const auto& [resource, index, placed] :
             {std::tuple{rt_resource.p, 0u, rt_place}, std::tuple{packed565.p, 0u, packed_place},
              std::tuple{ds.p, 0u, depth_place},
              std::tuple{ds.p, 1u, stencil_place}, std::tuple{untouched.p, 0u, untouched_place}}) {
            const auto dst = footprint(readback.p, placed);
            const auto src = subresource(resource, index);
            list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
        }
        EXPECT(list->Close() == S_OK);

        ID3D12CommandList* lists[] = {list.p};
        queue->ExecuteCommandLists(1, lists);
        EXPECT(rt::d3d12::last_execute_result(queue.p) == recity::ReplayResult::success);
        EXPECT(SUCCEEDED(queue->Signal(fence.p, 1)));
        EXPECT(SUCCEEDED(fence->SetEventOnCompletion(1, nullptr)));  // null event: blocks until done
        EXPECT(fence->GetCompletedValue() == 1);

        for (UINT row = 0; row < size; ++row) {
            for (UINT col = 0; col < size; ++col) {
                const bool inside = col >= 2 && col < 6 && row >= 2 && row < 6;
                const std::uint8_t want[4] = {std::uint8_t(inside ? 0 : 255), 0, std::uint8_t(inside ? 255 : 0), 255};
                report("colour texel", row * size + col,
                       down + rt_place.Offset + row * rt_place.Footprint.RowPitch + col * 4, want, 4);
                // B is in bits 0..4, G in 5..10 and R in 11..15. The green levels round to 16 and 47.
                const std::uint16_t packed_word = inside ? 0x05ff : 0xfa00;
                report("B5G6R5 texel", row * size + col,
                       down + packed_place.Offset + row * packed_place.Footprint.RowPitch + col * 2,
                       reinterpret_cast<const std::uint8_t*>(&packed_word), sizeof(packed_word));
                const float depth = row < 4 ? 0.25f : 0.0f;  // the bottom half was zero-filled, never cleared
                report("depth texel", row * size + col,
                       down + depth_place.Offset + row * depth_place.Footprint.RowPitch + col * 4,
                       reinterpret_cast<const std::uint8_t*>(&depth), 4);
                const std::uint8_t stencil = row < 4 ? 0x42 : 0;
                report("stencil texel", row * size + col,
                       down + stencil_place.Offset + row * stencil_place.Footprint.RowPitch + col, &stencil, 1);
            }
        }
        const std::uint8_t zero[4] = {};
        for (UINT row = 0; row < small_size; ++row)
            for (UINT col = 0; col < small_size; ++col)
                report("untouched texel", row * small_size + col,
                       down + untouched_place.Offset + row * untouched_place.Footprint.RowPitch + col * 4, zero, 4);

        // List 2: uploads. A pattern goes from the upload buffer into a texture, which is copied whole into a
        // texture created in COMMON (implicit promotion to COPY_DEST), and in part (a boxed region) into a smaller
        // one; a BC1 texture makes a round trip; a buffer region is copied.
        Ref<ID3D12Resource> source, whole, part, bc;
        const D3D12_RESOURCE_DESC rgba = texture_desc(size, size, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_NONE);
        EXPECT(SUCCEEDED(create(default_heap, rgba, D3D12_RESOURCE_STATE_COPY_DEST, source)));
        EXPECT(SUCCEEDED(create(default_heap, rgba, D3D12_RESOURCE_STATE_COMMON, whole)));
        EXPECT(SUCCEEDED(create(default_heap,
                                texture_desc(small_size, small_size, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_NONE),
                                D3D12_RESOURCE_STATE_COPY_DEST, part)));
        if (d.bc)
            EXPECT(SUCCEEDED(create(default_heap,
                                    texture_desc(size, size, DXGI_FORMAT_BC1_UNORM, D3D12_RESOURCE_FLAG_NONE),
                                    D3D12_RESOURCE_STATE_COPY_DEST, bc)));

        UINT64 up_cursor = 0;
        const auto pattern_place = place(device.p, source.p, 0, up_cursor);
        for (UINT row = 0; row < size; ++row)
            for (UINT col = 0; col < size; ++col)
                std::memcpy(up + pattern_place.Offset + row * pattern_place.Footprint.RowPitch + col * 4,
                            pattern(col, row).data(), 4);
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT bc_up{}, bc_down{};
        if (bc.p) {
            bc_up = place(device.p, bc.p, 0, up_cursor);
            for (UINT block_row = 0; block_row < 2; ++block_row)
                for (UINT i = 0; i < 16; ++i)  // two 8-byte blocks per row
                    up[bc_up.Offset + block_row * bc_up.Footprint.RowPitch + i] = std::uint8_t(0x30 + block_row * 16 + i);
        }
        const UINT64 bytes_at = (up_cursor + 255) / 256 * 256;
        for (UINT i = 0; i < 64; ++i) up[bytes_at + i] = std::uint8_t(0xa0 + i);

        cursor = 0;
        const auto whole_place = place(device.p, whole.p, 0, cursor);
        const auto part_place = place(device.p, part.p, 0, cursor);
        if (bc.p) bc_down = place(device.p, bc.p, 0, cursor);
        const UINT64 bytes_down = (cursor + 255) / 256 * 256;
        std::memset(down, 0xcd, 65536);

        EXPECT(SUCCEEDED(allocator->Reset()));
        EXPECT(SUCCEEDED(list->Reset(allocator.p, nullptr)));
        {
            const auto dst = subresource(source.p, 0);
            const auto src = footprint(upload.p, pattern_place);
            list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
        }
        barrier = transition(source.p, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE);
        list->ResourceBarrier(1, &barrier);
        list->CopyResource(whole.p, source.p);
        {
            const auto dst = subresource(part.p, 0);
            const auto src = subresource(source.p, 0);
            const D3D12_BOX box{2, 2, 0, 6, 6, 1};
            list->CopyTextureRegion(&dst, 0, 0, 0, &src, &box);
        }
        barrier = transition(whole.p, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE);  // promoted
        list->ResourceBarrier(1, &barrier);
        barrier = transition(part.p, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE);
        list->ResourceBarrier(1, &barrier);
        if (bc.p) {
            const auto dst = subresource(bc.p, 0);
            const auto src = footprint(upload.p, bc_up);
            list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
            barrier = transition(bc.p, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE);
            list->ResourceBarrier(1, &barrier);
        }
        for (const auto& [resource, placed] : {std::pair{whole.p, whole_place}, std::pair{part.p, part_place},
                                               std::pair{bc.p, bc_down}}) {
            if (!resource) continue;
            const auto dst = footprint(readback.p, placed);
            const auto src = subresource(resource, 0);
            list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
        }
        list->CopyBufferRegion(readback.p, bytes_down, upload.p, bytes_at, 64);
        EXPECT(list->Close() == S_OK);
        queue->ExecuteCommandLists(1, lists);
        EXPECT(rt::d3d12::last_execute_result(queue.p) == recity::ReplayResult::success);
        EXPECT(SUCCEEDED(queue->Signal(fence.p, 2)));
        EXPECT(SUCCEEDED(fence->SetEventOnCompletion(2, nullptr)));

        for (UINT row = 0; row < size; ++row)
            for (UINT col = 0; col < size; ++col)
                report("whole-copy texel", row * size + col,
                       down + whole_place.Offset + row * whole_place.Footprint.RowPitch + col * 4,
                       pattern(col, row).data(), 4);
        for (UINT row = 0; row < small_size; ++row)
            for (UINT col = 0; col < small_size; ++col)
                report("boxed-copy texel", row * small_size + col,
                       down + part_place.Offset + row * part_place.Footprint.RowPitch + col * 4,
                       pattern(col + 2, row + 2).data(), 4);
        if (bc.p)
            for (UINT block_row = 0; block_row < 2; ++block_row)
                report("BC1 block row", block_row, down + bc_down.Offset + block_row * bc_down.Footprint.RowPitch,
                       up + bc_up.Offset + block_row * bc_up.Footprint.RowPitch, 16);
        else
            std::printf("BC1 round trip: no textureCompressionBC, skipped\n");
        report("buffer copy", 0, down + bytes_down, up + bytes_at, 64);

        // Mip 1 is 4x3x8. Move a two-row box across slices through each of the three copy paths.
        // The footprint has a third row, so using the box height as the slice stride corrupts every later slice.
        {
            Ref<ID3D12Resource> clone, cropped;
            EXPECT(SUCCEEDED(create(default_heap, volume_desc(0), D3D12_RESOURCE_STATE_COPY_DEST, clone)));
            EXPECT(SUCCEEDED(create(default_heap, volume_desc(0), D3D12_RESOURCE_STATE_COPY_DEST, cropped)));
            cursor = 0;
            const auto layout = place(device.p, volume.p, 1, cursor);
            EXPECT(layout.Footprint.Depth == 8 && layout.Footprint.Height == 3 && layout.Footprint.Width == 4);
            const auto offset = [&](UINT x, UINT y, UINT z) {
                return layout.Offset + (z * 3 + y) * layout.Footprint.RowPitch + x * 4;
            };
            for (UINT z = 0; z < 8; ++z)
                for (UINT y = 0; y < 3; ++y)
                    for (UINT x = 0; x < 4; ++x)
                        std::memcpy(up + offset(x, y, z), pattern(x, z + 8 * y).data(), 4);
            std::memset(down, 0xcd, 65536);
            const auto in = footprint(upload.p, layout), out = footprint(readback.p, layout);
            const auto v = subresource(volume.p, 1), c = subresource(clone.p, 1), target = subresource(cropped.p, 1);
            EXPECT(SUCCEEDED(allocator->Reset()) && SUCCEEDED(list->Reset(allocator.p, nullptr)));
            const D3D12_BOX upload_box{1, 1, 1, 4, 3, 5};
            list->CopyTextureRegion(&v, 0, 0, 2, &in, &upload_box);
            auto readable_volume = transition(volume.p, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE);
            list->ResourceBarrier(1, &readable_volume);
            list->CopyResource(clone.p, volume.p);
            readable_volume = transition(clone.p, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE);
            list->ResourceBarrier(1, &readable_volume);
            const D3D12_BOX clone_box{0, 0, 2, 3, 2, 6};
            list->CopyTextureRegion(&target, 1, 1, 3, &c, &clone_box);
            readable_volume = transition(cropped.p, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE);
            list->ResourceBarrier(1, &readable_volume);
            const D3D12_BOX output_box{1, 1, 3, 4, 3, 7};
            list->CopyTextureRegion(&out, 0, 0, 1, &target, &output_box);
            EXPECT(SUCCEEDED(list->Close()));
            queue->ExecuteCommandLists(1, lists);
            EXPECT(rt::d3d12::last_execute_result(queue.p) == recity::ReplayResult::success);
            EXPECT(SUCCEEDED(queue->Signal(fence.p, 3)) && SUCCEEDED(fence->SetEventOnCompletion(3, nullptr)));
            const std::array<std::uint8_t, 4> untouched_bytes{0xcd, 0xcd, 0xcd, 0xcd};
            for (UINT z = 0; z < 8; ++z)
                for (UINT y = 0; y < 3; ++y)
                    for (UINT x = 0; x < 4; ++x) {
                        const auto want = (x < 3 && y < 2 && z >= 1 && z < 5)
                            ? pattern(x + 1, z + 8 * (y + 1)) : untouched_bytes;
                        report("volume texel", (z * 3 + y) * 4 + x, down + offset(x, y, z), want.data(), 4);
                    }
        }
        volume.reset();

        // A resolve: a 4x multisampled target is cleared and resolved into a single-sampled texture, which is read
        // back. First the shapes D3D12 refuses: a multisampled texture that is not a single-mip render target or
        // depth-stencil, a sample count that is not a power of two, and a multisampled volume.
        {
            Ref<ID3D12Resource> msaa, resolved, refused;
            D3D12_RESOURCE_DESC shape =
                texture_desc(size, size, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET, 2);
            shape.SampleDesc.Count = 4;
            EXPECT(create(default_heap, shape, D3D12_RESOURCE_STATE_COMMON, refused) == E_INVALIDARG && !refused.p);
            shape.MipLevels = 1;
            shape.Flags = D3D12_RESOURCE_FLAG_NONE;
            EXPECT(create(default_heap, shape, D3D12_RESOURCE_STATE_COMMON, refused) == E_INVALIDARG && !refused.p);
            shape.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
            shape.SampleDesc.Count = 3;
            EXPECT(create(default_heap, shape, D3D12_RESOURCE_STATE_COMMON, refused) == E_INVALIDARG && !refused.p);
            shape.SampleDesc.Count = D3D12_MAX_MULTISAMPLE_SAMPLE_COUNT * 2;
            EXPECT(create(default_heap, shape, D3D12_RESOURCE_STATE_COMMON, refused) == E_INVALIDARG && !refused.p);
            shape.SampleDesc.Count = 0;
            EXPECT(create(default_heap, shape, D3D12_RESOURCE_STATE_COMMON, refused) == E_INVALIDARG && !refused.p);
            shape.SampleDesc.Count = 4;
            shape.SampleDesc.Quality = 1;  // The Vulkan backend exposes only quality level 0.
            EXPECT(create(default_heap, shape, D3D12_RESOURCE_STATE_COMMON, refused) == E_INVALIDARG && !refused.p);
            shape.SampleDesc.Quality = 0;
            shape = volume_desc(1);
            shape.SampleDesc.Count = 4;
            EXPECT(create(default_heap, shape, D3D12_RESOURCE_STATE_COMMON, refused) == E_INVALIDARG && !refused.p);

            D3D12_RESOURCE_DESC samples4 =
                texture_desc(size, size, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET);
            samples4.SampleDesc.Count = 4;
            EXPECT(SUCCEEDED(create(default_heap, samples4, D3D12_RESOURCE_STATE_RENDER_TARGET, msaa)));
            EXPECT(SUCCEEDED(create(default_heap, rgba, D3D12_RESOURCE_STATE_RESOLVE_DEST, resolved)));
            EXPECT(msaa->GetDesc().SampleDesc.Count == 4);
            D3D12_RENDER_TARGET_VIEW_DESC msaa_view{};
            msaa_view.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
            msaa_view.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2DMS;
            device->CreateRenderTargetView(msaa.p, &msaa_view, rtv1);  // the lists that used this slot have run
            cursor = 0;
            const auto resolved_place = place(device.p, resolved.p, 0, cursor);
            std::memset(down, 0xcd, 65536);

            EXPECT(SUCCEEDED(allocator->Reset()) && SUCCEEDED(list->Reset(allocator.p, nullptr)));
            const FLOAT green[4] = {0, 1, 0, 1};
            list->ClearRenderTargetView(rtv1, green, 0, nullptr);
            barrier = transition(msaa.p, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_RESOLVE_SOURCE);
            list->ResourceBarrier(1, &barrier);
            list->ResolveSubresource(resolved.p, 0, msaa.p, 0, DXGI_FORMAT_R8G8B8A8_UNORM);
            barrier = transition(resolved.p, D3D12_RESOURCE_STATE_RESOLVE_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE);
            list->ResourceBarrier(1, &barrier);
            {
                const auto dst = footprint(readback.p, resolved_place);
                const auto src = subresource(resolved.p, 0);
                list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
            }
            EXPECT(list->Close() == S_OK);
            queue->ExecuteCommandLists(1, lists);
            EXPECT(rt::d3d12::last_execute_result(queue.p) == recity::ReplayResult::success);
            EXPECT(SUCCEEDED(queue->Signal(fence.p, 4)) && SUCCEEDED(fence->SetEventOnCompletion(4, nullptr)));
            const std::uint8_t resolved_green[4] = {0, 255, 0, 255};
            for (UINT row = 0; row < size; ++row)
                for (UINT col = 0; col < size; ++col)
                    report("resolved texel", row * size + col,
                           down + resolved_place.Offset + row * resolved_place.Footprint.RowPitch + col * 4,
                           resolved_green, 4);

            // Move a cropped two-colour region into a larger red texture.
            constexpr UINT region_width = 10, region_height = 9;
            constexpr auto region_format = DXGI_FORMAT_R8G8B8A8_UNORM;
            const auto region_desc =
                texture_desc(region_width, region_height, region_format, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET);
            Ref<ID3D12Resource> region_target;
            EXPECT(SUCCEEDED(create(default_heap, region_desc, D3D12_RESOURCE_STATE_RENDER_TARGET, region_target)));
            device->CreateRenderTargetView(region_target.p, nullptr, rtv0);
            cursor = 0;
            const auto region_place = place(device.p, region_target.p, 0, cursor);
            EXPECT(SUCCEEDED(allocator->Reset()) && SUCCEEDED(list->Reset(allocator.p, nullptr)));
            barrier = transition(msaa.p, D3D12_RESOURCE_STATE_RESOLVE_SOURCE, D3D12_RESOURCE_STATE_RENDER_TARGET);
            list->ResourceBarrier(1, &barrier);
            const D3D12_RECT blue_patch{2, 3, 4, 5};
            list->ClearRenderTargetView(rtv1, blue, 1, &blue_patch);
            list->ClearRenderTargetView(rtv0, red, 0, nullptr);
            const D3D12_RESOURCE_BARRIER resolve_barriers[] = {
                transition(msaa.p, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_RESOLVE_SOURCE),
                transition(region_target.p, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_RESOLVE_DEST)};
            list->ResourceBarrier(2, resolve_barriers);
            D3D12_RECT region{1, 2, 5, 5};
            const auto region_copy = subresource(region_target.p, 0), region_out = footprint(readback.p, region_place);
            const auto resolve_and_readback = [&](UINT x, D3D12_RECT* rect, UINT64 fence_value) {
                list7->ResolveSubresourceRegion(region_target.p, 0, x, 1, msaa.p, 0, rect, region_format,
                                                D3D12_RESOLVE_MODE_AVERAGE);
                const auto to_copy = transition(region_target.p, D3D12_RESOURCE_STATE_RESOLVE_DEST,
                                                D3D12_RESOURCE_STATE_COPY_SOURCE);
                list->ResourceBarrier(1, &to_copy);
                list->CopyTextureRegion(&region_out, 0, 0, 0, &region_copy, nullptr);
                if (list->Close() != S_OK) return false;
                queue->ExecuteCommandLists(1, lists);
                return rt::d3d12::last_execute_result(queue.p) == recity::ReplayResult::success &&
                       SUCCEEDED(queue->Signal(fence.p, fence_value)) &&
                       SUCCEEDED(fence->SetEventOnCompletion(fence_value, nullptr));
            };
            EXPECT(resolve_and_readback(4, &region, 5));
            const std::uint8_t resolved_red[4] = {255, 0, 0, 255}, resolved_blue[4] = {0, 0, 255, 255};
            for (UINT row = 0; row < region_height; ++row)
                for (UINT col = 0; col < region_width; ++col) {
                    const bool in_region = row >= 1 && row < 4 && col >= 4 && col < 8;
                    const bool in_blue = row >= 2 && row < 4 && col >= 5 && col < 7;
                    report("partial resolve texel", row * region_width + col,
                           down + region_place.Offset + row * region_place.Footprint.RowPitch + col * 4,
                           in_blue ? resolved_blue : in_region ? resolved_green : resolved_red, 4);
                }

            // A null rectangle resolves the entire source at the destination offset.
            EXPECT(SUCCEEDED(allocator->Reset()) && SUCCEEDED(list->Reset(allocator.p, nullptr)));
            const auto to_resolve = transition(region_target.p, D3D12_RESOURCE_STATE_COPY_SOURCE,
                                               D3D12_RESOURCE_STATE_RESOLVE_DEST);
            list->ResourceBarrier(1, &to_resolve);
            EXPECT(resolve_and_readback(1, nullptr, 6));
            for (UINT row = 0; row < region_height; ++row)
                for (UINT col = 0; col < region_width; ++col) {
                    const bool in_source = row >= 1 && col >= 1 && col < 9;
                    const bool in_blue = row >= 4 && row < 6 && col >= 3 && col < 5;
                    report("whole source at offset", row * region_width + col,
                           down + region_place.Offset + row * region_place.Footprint.RowPitch + col * 4,
                           in_blue ? resolved_blue : in_source ? resolved_green : resolved_red, 4);
                }

            D3D12_RECT out_of_source{0, 0, LONG(size + 1), 1};
            for (auto [x, rect] : {std::pair{0u, &out_of_source}, std::pair{7u, &region}}) {
                EXPECT(SUCCEEDED(list->Reset(allocator.p, nullptr)));
                list->ResourceBarrier(1, &to_resolve);
                list7->ResolveSubresourceRegion(region_target.p, 0, x, 1, msaa.p, 0, rect, region_format,
                                                D3D12_RESOLVE_MODE_AVERAGE);
                EXPECT(list->Close() == S_OK);
                queue->ExecuteCommandLists(1, lists);
                EXPECT(rt::d3d12::last_execute_result(queue.p) == recity::ReplayResult::invalid_copy);
            }
            EXPECT(SUCCEEDED(list->Reset(allocator.p, nullptr)));
            list7->ResolveSubresourceRegion(region_target.p, 0, 4, 1, msaa.p, 0, &region, region_format,
                                            D3D12_RESOLVE_MODE_MAX);
            EXPECT(list->Close() == E_INVALIDARG);
            EXPECT(SUCCEEDED(list->Reset(allocator.p, nullptr)));
            list7->ResolveSubresourceRegion(region_target.p, 0, 4, 1, msaa.p, 0, &region,
                                            DXGI_FORMAT_R8G8B8A8_UNORM_SRGB, D3D12_RESOLVE_MODE_AVERAGE);
            EXPECT(list->Close() == E_INVALIDARG);

            // A resolve of a resource into itself is refused when it is recorded, one from a single-sampled
            // texture when it is executed; nothing is submitted.
            EXPECT(SUCCEEDED(list->Reset(allocator.p, nullptr)));
            list->ResolveSubresource(resolved.p, 0, resolved.p, 0, DXGI_FORMAT_R8G8B8A8_UNORM);
            EXPECT(list->Close() == E_INVALIDARG);
            EXPECT(SUCCEEDED(list->Reset(allocator.p, nullptr)));
            list->ResolveSubresource(resolved.p, 0, source.p, 0, DXGI_FORMAT_R8G8B8A8_UNORM);
            EXPECT(list->Close() == S_OK);
            queue->ExecuteCommandLists(1, lists);
            EXPECT(rt::d3d12::last_execute_result(queue.p) == recity::ReplayResult::invalid_copy);
        }

        // D3D-26 through COM: AtomicCopyBufferUINT/UINT64 from the upload buffer to the readback buffer, discards of a
        // texture and a buffer, and a WriteBufferImmediate whose address no resource holds, which fails at execution.
        {
            Ref<ID3D12GraphicsCommandList1> list1;
            Ref<ID3D12GraphicsCommandList2> list2;
            EXPECT(SUCCEEDED(list->QueryInterface(IID_PPV_ARGS(&list1.p))));
            EXPECT(SUCCEEDED(list->QueryInterface(IID_PPV_ARGS(&list2.p))));
            const std::uint32_t value32 = 0x600df00d;
            const std::uint64_t value64 = 0x0123456789abcdefull;
            std::memcpy(up + 4096, &value32, 4);
            std::memcpy(up + 4104, &value64, 8);
            std::memset(down, 0xcd, 65536);
            ID3D12Resource* const dependents[] = {upload.p};
            const D3D12_SUBRESOURCE_RANGE_UINT64 ranges[] = {{0, {4096, 4112}}};
            EXPECT(SUCCEEDED(allocator->Reset()) && SUCCEEDED(list->Reset(allocator.p, nullptr)));
            list1->AtomicCopyBufferUINT(readback.p, 256, upload.p, 4096, 1, dependents, ranges);
            list1->AtomicCopyBufferUINT64(readback.p, 264, upload.p, 4104, 1, dependents, ranges);
            list->DiscardResource(rt_resource.p, nullptr);
            const D3D12_DISCARD_REGION first_mip{0, nullptr, 0, 1};
            list->DiscardResource(rt_resource.p, &first_mip);
            list->DiscardResource(upload.p, nullptr);
            EXPECT(list->Close() == S_OK);
            queue->ExecuteCommandLists(1, lists);
            EXPECT(rt::d3d12::last_execute_result(queue.p) == recity::ReplayResult::success);
            EXPECT(SUCCEEDED(queue->Signal(fence.p, 7)) && SUCCEEDED(fence->SetEventOnCompletion(7, nullptr)));
            report("atomic copy UINT", 0, down + 256, reinterpret_cast<const std::uint8_t*>(&value32), 4);
            report("atomic copy UINT64", 0, down + 264, reinterpret_cast<const std::uint8_t*>(&value64), 8);
            const std::uint8_t cd[4] = {0xcd, 0xcd, 0xcd, 0xcd};
            report("next to the atomic copies", 0, down + 252, cd, 4);
            report("next to the atomic copies", 1, down + 272, cd, 4);

            // Misaligned or empty calls are refused when they are recorded.
            EXPECT(SUCCEEDED(list->Reset(allocator.p, nullptr)));
            list1->AtomicCopyBufferUINT64(readback.p, 260, upload.p, 4104, 1, dependents, ranges);
            EXPECT(list->Close() == E_INVALIDARG);
            EXPECT(SUCCEEDED(list->Reset(allocator.p, nullptr)));
            list1->AtomicCopyBufferUINT(readback.p, 256, upload.p, 4096, 0, nullptr, nullptr);
            EXPECT(list->Close() == E_INVALIDARG);
            EXPECT(SUCCEEDED(list->Reset(allocator.p, nullptr)));
            const D3D12_WRITEBUFFERIMMEDIATE_PARAMETER misaligned{0x10002, 1};
            list2->WriteBufferImmediate(1, &misaligned, nullptr);
            EXPECT(list->Close() == E_INVALIDARG);
            // A region with NumSubresources = UINT_MAX is an explicit range, not a null region.
            EXPECT(SUCCEEDED(list->Reset(allocator.p, nullptr)));
            const D3D12_DISCARD_REGION all_from_one{0, nullptr, 1, UINT_MAX};
            list->DiscardResource(rt_resource.p, &all_from_one);
            EXPECT(list->Close() == E_INVALIDARG);
            EXPECT(SUCCEEDED(list->Reset(allocator.p, nullptr)));
            const D3D12_DISCARD_REGION all_from_zero{0, nullptr, 0, UINT_MAX};
            list->DiscardResource(rt_resource.p, &all_from_zero);
            EXPECT(list->Close() == S_OK);
            queue->ExecuteCommandLists(1, lists);
            EXPECT(rt::d3d12::last_execute_result(queue.p) == recity::ReplayResult::invalid_subresource);

            // Nothing on main gives a buffer a GPU virtual address yet (GetGPUVirtualAddress, #157/#291), so an
            // immediate write has no target: the list is refused and nothing is submitted.
            EXPECT(SUCCEEDED(list->Reset(allocator.p, nullptr)));
            const D3D12_WRITEBUFFERIMMEDIATE_PARAMETER write{0x10000, 1};
            const D3D12_WRITEBUFFERIMMEDIATE_MODE mode = D3D12_WRITEBUFFERIMMEDIATE_MODE_MARKER_OUT;
            list2->WriteBufferImmediate(1, &write, &mode);
            EXPECT(list->Close() == S_OK);
            queue->ExecuteCommandLists(1, lists);
            EXPECT(rt::d3d12::last_execute_result(queue.p) == recity::ReplayResult::unresolved_resource);
        }

        // A depth-stencil view of a slice range of a 4x multisampled depth array (D3D12_DSV_DIMENSION_TEXTURE2DMSARRAY):
        // the view captures the range, a clear through it replays, and a range past the last slice is refused when
        // the list runs. D3D12 can't copy a multisampled texture to a buffer, so the clear itself isn't read back.
        {
            Ref<ID3D12Resource> ms_depth;
            Ref<ID3D12Fence> ms_fence;
            D3D12_RESOURCE_DESC ms_desc =
                texture_desc(size, size, DXGI_FORMAT_D32_FLOAT, D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL, 1, 4);
            ms_desc.SampleDesc.Count = 4;
            EXPECT(SUCCEEDED(create(default_heap, ms_desc, D3D12_RESOURCE_STATE_DEPTH_WRITE, ms_depth)));
            EXPECT(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&ms_fence.p))));
            D3D12_DEPTH_STENCIL_VIEW_DESC ms_view{};
            ms_view.Format = DXGI_FORMAT_D32_FLOAT;
            ms_view.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2DMSARRAY;
            ms_view.Texture2DMSArray.FirstArraySlice = 1;
            ms_view.Texture2DMSArray.ArraySize = 2;
            device->CreateDepthStencilView(ms_depth.p, &ms_view, dsv0);  // the lists that used this slot have run
            recity::TargetView captured{};
            std::memcpy(&captured, reinterpret_cast<const void*>(dsv0.ptr), sizeof(captured));
            EXPECT(captured.resource && captured.mip == 0 && captured.first_layer == 1 && captured.layer_count == 2);

            EXPECT(SUCCEEDED(allocator->Reset()) && SUCCEEDED(list->Reset(allocator.p, nullptr)));
            list->ClearDepthStencilView(dsv0, D3D12_CLEAR_FLAG_DEPTH, 0.5f, 0, 0, nullptr);
            EXPECT(list->Close() == S_OK);
            queue->ExecuteCommandLists(1, lists);
            EXPECT(rt::d3d12::last_execute_result(queue.p) == recity::ReplayResult::success);
            EXPECT(SUCCEEDED(queue->Signal(ms_fence.p, 1)) && SUCCEEDED(ms_fence->SetEventOnCompletion(1, nullptr)));

            ms_view.Texture2DMSArray.FirstArraySlice = 3;  // slices 3 and 4 of a 4-slice texture
            device->CreateDepthStencilView(ms_depth.p, &ms_view, dsv0);
            EXPECT(SUCCEEDED(list->Reset(allocator.p, nullptr)));
            list->ClearDepthStencilView(dsv0, D3D12_CLEAR_FLAG_DEPTH, 0.5f, 0, 0, nullptr);
            EXPECT(list->Close() == S_OK);
            queue->ExecuteCommandLists(1, lists);
            EXPECT(rt::d3d12::last_execute_result(queue.p) == recity::ReplayResult::invalid_subresource);
            EXPECT(SUCCEEDED(queue->Signal(ms_fence.p, 2)) && SUCCEEDED(ms_fence->SetEventOnCompletion(2, nullptr)));
        }

        // A render-target view of a slice range of a 4x multisampled colour array (D3D12_RTV_DIMENSION_TEXTURE2DMSARRAY):
        // all four slices are cleared red through a whole-array view, then slices 1-2 green through a two-slice view.
        // Every slice is resolved and read back, so only slices 1 and 2 may be green. A range past the last slice is
        // refused when the list runs.
        {
            Ref<ID3D12Resource> ms_colour, ms_resolved;
            Ref<ID3D12Fence> ms_fence;
            constexpr UINT16 slices = 4;
            D3D12_RESOURCE_DESC ms_desc = texture_desc(size, size, DXGI_FORMAT_R8G8B8A8_UNORM,
                                                       D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET, 1, slices);
            ms_desc.SampleDesc.Count = 4;
            EXPECT(SUCCEEDED(create(default_heap, ms_desc, D3D12_RESOURCE_STATE_RENDER_TARGET, ms_colour)));
            EXPECT(SUCCEEDED(create(default_heap,
                                    texture_desc(size, size, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_NONE, 1,
                                                 slices),
                                    D3D12_RESOURCE_STATE_RESOLVE_DEST, ms_resolved)));
            EXPECT(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&ms_fence.p))));
            D3D12_RENDER_TARGET_VIEW_DESC ms_view{};
            ms_view.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
            ms_view.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2DMSARRAY;
            ms_view.Texture2DMSArray.FirstArraySlice = 0;
            ms_view.Texture2DMSArray.ArraySize = slices;
            device->CreateRenderTargetView(ms_colour.p, &ms_view, rtv0);  // the lists that used these slots have run
            ms_view.Texture2DMSArray.FirstArraySlice = 1;
            ms_view.Texture2DMSArray.ArraySize = 2;
            device->CreateRenderTargetView(ms_colour.p, &ms_view, rtv1);
            recity::TargetView captured{};
            std::memcpy(&captured, reinterpret_cast<const void*>(rtv1.ptr), sizeof(captured));
            EXPECT(captured.resource && captured.mip == 0 && captured.first_layer == 1 && captured.layer_count == 2);

            cursor = 0;
            D3D12_PLACED_SUBRESOURCE_FOOTPRINT slice_place[slices];
            for (UINT s = 0; s < slices; ++s) slice_place[s] = place(device.p, ms_resolved.p, s, cursor);
            std::memset(down, 0xcd, 65536);
            EXPECT(SUCCEEDED(allocator->Reset()) && SUCCEEDED(list->Reset(allocator.p, nullptr)));
            const FLOAT slice_red[4] = {1, 0, 0, 1}, slice_green[4] = {0, 1, 0, 1};
            list->ClearRenderTargetView(rtv0, slice_red, 0, nullptr);
            list->ClearRenderTargetView(rtv1, slice_green, 0, nullptr);
            barrier = transition(ms_colour.p, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_RESOLVE_SOURCE);
            list->ResourceBarrier(1, &barrier);
            for (UINT s = 0; s < slices; ++s)
                list->ResolveSubresource(ms_resolved.p, s, ms_colour.p, s, DXGI_FORMAT_R8G8B8A8_UNORM);
            barrier = transition(ms_resolved.p, D3D12_RESOURCE_STATE_RESOLVE_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE);
            list->ResourceBarrier(1, &barrier);
            for (UINT s = 0; s < slices; ++s) {
                const auto dst = footprint(readback.p, slice_place[s]);
                const auto src = subresource(ms_resolved.p, s);
                list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
            }
            EXPECT(list->Close() == S_OK);
            queue->ExecuteCommandLists(1, lists);
            EXPECT(rt::d3d12::last_execute_result(queue.p) == recity::ReplayResult::success);
            EXPECT(SUCCEEDED(queue->Signal(ms_fence.p, 1)) && SUCCEEDED(ms_fence->SetEventOnCompletion(1, nullptr)));
            const std::uint8_t red_texel[4] = {255, 0, 0, 255}, green_texel[4] = {0, 255, 0, 255};
            for (UINT s = 0; s < slices; ++s)
                for (UINT row = 0; row < size; ++row)
                    for (UINT col = 0; col < size; ++col)
                        report("multisample array slice texel", (s * size + row) * size + col,
                               down + slice_place[s].Offset + row * slice_place[s].Footprint.RowPitch + col * 4,
                               s == 1 || s == 2 ? green_texel : red_texel, 4);

            ms_view.Texture2DMSArray.FirstArraySlice = 3;  // slices 3 and 4 of a 4-slice texture
            device->CreateRenderTargetView(ms_colour.p, &ms_view, rtv1);
            EXPECT(SUCCEEDED(list->Reset(allocator.p, nullptr)));
            barrier = transition(ms_colour.p, D3D12_RESOURCE_STATE_RESOLVE_SOURCE, D3D12_RESOURCE_STATE_RENDER_TARGET);
            list->ResourceBarrier(1, &barrier);
            list->ClearRenderTargetView(rtv1, slice_green, 0, nullptr);
            EXPECT(list->Close() == S_OK);
            queue->ExecuteCommandLists(1, lists);
            EXPECT(rt::d3d12::last_execute_result(queue.p) == recity::ReplayResult::invalid_subresource);
            EXPECT(SUCCEEDED(queue->Signal(ms_fence.p, 2)) && SUCCEEDED(ms_fence->SetEventOnCompletion(2, nullptr)));
        }

        // Reserved resources (docs/d3d12_vulkan.md section 4.2): fully backed at creation, so they copy like
        // committed ones and their first use reads zero; D3D12's description rules hold; tile-mapping calls are
        // accepted and change nothing.
        {
            const auto B = D3D12_RESOURCE_DIMENSION_BUFFER, T2 = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
            Ref<ID3D12Resource> tiled, tiled_buffer, xbox_tiled, refused;
            const D3D12_RESOURCE_DESC tiled_desc = reserved_desc(T2, size, size, 1, 1, DXGI_FORMAT_R8G8B8A8_UNORM);
            EXPECT(SUCCEEDED(device->CreateReservedResource(&tiled_desc, D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                            IID_PPV_ARGS(&tiled.p))));
            EXPECT(tiled->GetDesc().Layout == D3D12_TEXTURE_LAYOUT_64KB_UNDEFINED_SWIZZLE);
            D3D12_HEAP_PROPERTIES reserved_heap{};
            EXPECT(tiled->GetHeapProperties(&reserved_heap, nullptr) == E_INVALIDARG);  // no heap of its own
            EXPECT(device->CreateReservedResource(&tiled_desc, D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                  __uuidof(ID3D12Resource), nullptr) == S_FALSE);
            const auto refuses = [&](D3D12_RESOURCE_DESC desc) {
                const HRESULT hr = device->CreateReservedResource(&desc, D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                                  IID_PPV_ARGS(&refused.p));
                return hr == E_INVALIDARG && !refused.p;
            };
            auto bad = tiled_desc;
            bad.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;  // textures need a 64 KiB swizzle layout
            EXPECT(refuses(bad));
            bad.Layout = D3D12_TEXTURE_LAYOUT_64KB_STANDARD_SWIZZLE;  // StandardSwizzle64KBSupported is FALSE
            EXPECT(refuses(bad));
            EXPECT(refuses(reserved_desc(T2, 64, 64, 1, 1, DXGI_FORMAT_R32G32B32_FLOAT)));  // 12-byte texels: no tiles
            bad = reserved_desc(B, 65536, 1, 1, 1, DXGI_FORMAT_UNKNOWN);
            bad.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;  // buffers are ROW_MAJOR
            EXPECT(refuses(bad));
            const D3D12_RESOURCE_DESC tiled_buffer_desc = reserved_desc(B, 2 * 65536, 1, 1, 1, DXGI_FORMAT_UNKNOWN);
            EXPECT(SUCCEEDED(device->CreateReservedResource(&tiled_buffer_desc, D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                            IID_PPV_ARGS(&tiled_buffer.p))));
            auto xbox_desc = tiled_desc;
            xbox_desc.Layout = static_cast<D3D12_TEXTURE_LAYOUT>(0x101);  // the Xbox tiled layout word
            EXPECT(SUCCEEDED(device->CreateReservedResource(&xbox_desc, D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                            IID_PPV_ARGS(&xbox_tiled.p))));
            {
                Ref<ID3D12Device4> device4;
                Ref<ID3D12Device10> device10;
                Ref<ID3D12Resource> one, two;
                EXPECT(SUCCEEDED(device->QueryInterface(IID_PPV_ARGS(&device4.p))));
                EXPECT(SUCCEEDED(device->QueryInterface(IID_PPV_ARGS(&device10.p))));
                EXPECT(SUCCEEDED(device4->CreateReservedResource1(&tiled_desc, D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                                  nullptr, IID_PPV_ARGS(&one.p))));
                EXPECT(SUCCEEDED(device10->CreateReservedResource2(&tiled_desc, D3D12_BARRIER_LAYOUT_COMMON, nullptr,
                                                                   nullptr, 0, nullptr, IID_PPV_ARGS(&two.p))));
                UINT tiles = 0;
                device->GetResourceTiling(two.p, &tiles, nullptr, nullptr, nullptr, 0, nullptr);
                EXPECT(tiles == 1);  // 8x8 RGBA8: one packed tile
            }

            // Tiling: a resource that is not reserved has none, as on the PC runtime; a first index past the end
            // returns no subresources.
            UINT total = 7, count = 4;
            D3D12_PACKED_MIP_INFO packed{1, 1, 1, 1};
            D3D12_TILE_SHAPE shape{1, 1, 1};
            D3D12_SUBRESOURCE_TILING tilings[4]{};
            device->GetResourceTiling(upload.p, &total, &packed, &shape, &count, 0, tilings);
            EXPECT(total == 0 && count == 0 && shape.WidthInTexels == 0 && packed.NumStandardMips == 0 &&
                   packed.NumTilesForPackedMips == 0);
            count = 4;
            device->GetResourceTiling(tiled_buffer.p, &total, &packed, &shape, &count, 0, tilings);
            EXPECT(total == 2 && count == 1 && shape.WidthInTexels == 65536 && tilings[0].WidthInTiles == 2);
            count = 4;
            device->GetResourceTiling(tiled_buffer.p, nullptr, nullptr, nullptr, &count, 1, tilings);
            EXPECT(count == 0);

            // Mapping calls: accepted, logged once, and the resources keep their memory.
            D3D12_TILED_RESOURCE_COORDINATE origin{};
            D3D12_TILE_REGION_SIZE one_tile{1, FALSE, 0, 0, 0};
            const D3D12_TILE_RANGE_FLAGS unmap = D3D12_TILE_RANGE_FLAG_NULL;
            queue->UpdateTileMappings(tiled.p, 1, &origin, &one_tile, nullptr, 1, &unmap, nullptr, nullptr,
                                      D3D12_TILE_MAPPING_FLAG_NONE);
            queue->CopyTileMappings(tiled_buffer.p, &origin, tiled_buffer.p, &origin, &one_tile,
                                    D3D12_TILE_MAPPING_FLAG_NONE);

            // Upload a pattern into the reserved texture and bytes into the reserved buffer, read both back, and
            // read the never-written Xbox-layout texture, which is zero.
            UINT64 tiled_up_cursor = 0;
            const auto tiled_up = place(device.p, tiled.p, 0, tiled_up_cursor);
            for (UINT row = 0; row < size; ++row)
                for (UINT col = 0; col < size; ++col)
                    std::memcpy(up + tiled_up.Offset + row * tiled_up.Footprint.RowPitch + col * 4,
                                pattern(col, row).data(), 4);
            const UINT64 tiled_bytes_up = (tiled_up_cursor + 255) / 256 * 256;
            for (UINT i = 0; i < 256; ++i) up[tiled_bytes_up + i] = std::uint8_t(0x5a ^ i);
            UINT64 tiled_down_cursor = 0;
            const auto tiled_down = place(device.p, tiled.p, 0, tiled_down_cursor);
            const auto xbox_down = place(device.p, xbox_tiled.p, 0, tiled_down_cursor);
            const UINT64 tiled_bytes_down = (tiled_down_cursor + 255) / 256 * 256;
            std::memset(down, 0xcd, 65536);

            EXPECT(SUCCEEDED(allocator->Reset()) && SUCCEEDED(list->Reset(allocator.p, nullptr)));
            {
                const auto dst = subresource(tiled.p, 0);
                const auto src = footprint(upload.p, tiled_up);
                list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
            }
            barrier = transition(tiled.p, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE);
            list->ResourceBarrier(1, &barrier);
            barrier = transition(xbox_tiled.p, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE);
            list->ResourceBarrier(1, &barrier);
            for (const auto& [resource, placed] : {std::pair{tiled.p, tiled_down}, std::pair{xbox_tiled.p, xbox_down}}) {
                const auto dst = footprint(readback.p, placed);
                const auto src = subresource(resource, 0);
                list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
            }
            list->CopyBufferRegion(tiled_buffer.p, 65536 + 4096, upload.p, tiled_bytes_up, 256);  // into the second tile
            barrier = transition(tiled_buffer.p, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE);
            list->ResourceBarrier(1, &barrier);
            list->CopyBufferRegion(readback.p, tiled_bytes_down, tiled_buffer.p, 65536 + 4096, 256);
            EXPECT(list->Close() == S_OK);
            queue->ExecuteCommandLists(1, lists);
            EXPECT(rt::d3d12::last_execute_result(queue.p) == recity::ReplayResult::success);
            EXPECT(SUCCEEDED(queue->Signal(fence.p, 9)) && SUCCEEDED(fence->SetEventOnCompletion(9, nullptr)));
            for (UINT row = 0; row < size; ++row)
                for (UINT col = 0; col < size; ++col) {
                    report("reserved texel", row * size + col,
                           down + tiled_down.Offset + row * tiled_down.Footprint.RowPitch + col * 4,
                           pattern(col, row).data(), 4);
                    report("unwritten reserved texel", row * size + col,
                           down + xbox_down.Offset + row * xbox_down.Footprint.RowPitch + col * 4, zero, 4);
                }
            report("reserved buffer", 0, down + tiled_bytes_down, up + tiled_bytes_up, 256);
        }

        // Queries (docs/d3d12_vulkan.md section 4.6). The layer has no draws yet, so occlusion and pipeline
        // statistics count nothing and read zero; what is checked is that every query is reset and run, that
        // results land as UINT64s at the offsets asked for (over a 0xcd fill), that timestamps advance at the
        // reported frequency, that a query keeps its result across executions, that one never ended reads zero
        // instead of stalling the GPU, and that the D3D12 rules for the calls are kept.
        {
            UINT64 frequency = 0;
            EXPECT(SUCCEEDED(queue->GetTimestampFrequency(&frequency)) && frequency >= 1'000'000);
            EXPECT(queue->GetTimestampFrequency(nullptr) == E_INVALIDARG);
            Ref<ID3D12QueryHeap> occlusion, timestamps, statistics, refused;
            D3D12_QUERY_HEAP_DESC query_desc{D3D12_QUERY_HEAP_TYPE_OCCLUSION, 4, 0};
            EXPECT(SUCCEEDED(device->CreateQueryHeap(&query_desc, IID_PPV_ARGS(&occlusion.p))));
            EXPECT(device->CreateQueryHeap(&query_desc, __uuidof(ID3D12QueryHeap), nullptr) == S_FALSE);
            {
                Ref<ID3D12Device> parent;
                EXPECT(SUCCEEDED(occlusion->GetDevice(IID_PPV_ARGS(&parent.p))) && parent.p == device.p);
            }
            query_desc = {D3D12_QUERY_HEAP_TYPE_TIMESTAMP, 3, 0};
            EXPECT(SUCCEEDED(device->CreateQueryHeap(&query_desc, IID_PPV_ARGS(&timestamps.p))));
            query_desc = {D3D12_QUERY_HEAP_TYPE_OCCLUSION, 0, 0};
            EXPECT(device->CreateQueryHeap(&query_desc, IID_PPV_ARGS(&refused.p)) == E_INVALIDARG && !refused.p);
            query_desc = {D3D12_QUERY_HEAP_TYPE_SO_STATISTICS, 1, 0};  // no stream output path yet
            EXPECT(device->CreateQueryHeap(&query_desc, IID_PPV_ARGS(&refused.p)) == E_NOTIMPL && !refused.p);
            query_desc = {D3D12_QUERY_HEAP_TYPE_PIPELINE_STATISTICS, 2, 0};
            if (d.pipeline_statistics_query)
                EXPECT(SUCCEEDED(device->CreateQueryHeap(&query_desc, IID_PPV_ARGS(&statistics.p))));
            else
                std::printf("pipeline statistics: no pipelineStatisticsQuery, skipped\n");

            // Readback layout: timestamps at 0, occlusion at 64, statistics at 256, the second run's at 1024.
            constexpr UINT64 ts_at = 0, occlusion_at = 64, statistics_at = 256, again_at = 1024;
            const auto u64 = [&](UINT64 offset) {
                UINT64 value;
                std::memcpy(&value, down + offset, sizeof(value));
                return value;
            };
            std::memset(down, 0xcd, 65536);
            EXPECT(SUCCEEDED(allocator->Reset()) && SUCCEEDED(list->Reset(allocator.p, nullptr)));
            list->EndQuery(timestamps.p, D3D12_QUERY_TYPE_TIMESTAMP, 0);
            if (statistics.p) list->BeginQuery(statistics.p, D3D12_QUERY_TYPE_PIPELINE_STATISTICS, 0);
            // A rectangle clear runs in a render pass inside a query; clears count no samples.
            device->CreateRenderTargetView(rt_resource.p, &rtv_desc, rtv1);  // the resolve test reused it
            list->BeginQuery(occlusion.p, D3D12_QUERY_TYPE_OCCLUSION, 0);
            barrier = transition(rt_resource.p, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_RENDER_TARGET);
            list->ResourceBarrier(1, &barrier);
            list->ClearRenderTargetView(rtv1, red, 1, &square);
            list->EndQuery(occlusion.p, D3D12_QUERY_TYPE_OCCLUSION, 0);
            list->BeginQuery(occlusion.p, D3D12_QUERY_TYPE_BINARY_OCCLUSION, 1);
            list->CopyBufferRegion(readback.p, 8192, upload.p, 0, 32768);
            list->EndQuery(occlusion.p, D3D12_QUERY_TYPE_BINARY_OCCLUSION, 1);
            if (statistics.p) list->EndQuery(statistics.p, D3D12_QUERY_TYPE_PIPELINE_STATISTICS, 0);
            list->EndQuery(timestamps.p, D3D12_QUERY_TYPE_TIMESTAMP, 1);
            list->ResolveQueryData(timestamps.p, D3D12_QUERY_TYPE_TIMESTAMP, 0, 3, readback.p, ts_at);  // 2: never
            list->ResolveQueryData(occlusion.p, D3D12_QUERY_TYPE_OCCLUSION, 0, 1, readback.p, occlusion_at);
            list->ResolveQueryData(occlusion.p, D3D12_QUERY_TYPE_BINARY_OCCLUSION, 1, 1, readback.p, occlusion_at + 8);
            list->ResolveQueryData(occlusion.p, D3D12_QUERY_TYPE_OCCLUSION, 2, 2, readback.p, occlusion_at + 16);
            if (statistics.p)
                list->ResolveQueryData(statistics.p, D3D12_QUERY_TYPE_PIPELINE_STATISTICS, 0, 2, readback.p,
                                       statistics_at);
            EXPECT(list->Close() == S_OK);
            queue->ExecuteCommandLists(1, lists);
            EXPECT(rt::d3d12::last_execute_result(queue.p) == recity::ReplayResult::success);
            EXPECT(SUCCEEDED(queue->Signal(fence.p, 10)) && SUCCEEDED(fence->SetEventOnCompletion(10, nullptr)));

            const UINT64 t0 = u64(ts_at), t1 = u64(ts_at + 8);
            EXPECT(t0 != 0 && t0 != 0xcdcdcdcdcdcdcdcdull && t1 >= t0);
            EXPECT(double(t1 - t0) / double(frequency) < 10.0);  // the copy took less than ten seconds
            EXPECT(u64(ts_at + 16) == 0);                        // never written
            for (UINT i = 0; i < 4; ++i) EXPECT(u64(occlusion_at + 8 * i) == 0);
            for (UINT i = 0; statistics.p && i < 22; ++i) EXPECT(u64(statistics_at + 8 * i) == 0);
            EXPECT(u64(statistics.p ? statistics_at + 176 : statistics_at) == 0xcdcdcdcdcdcdcdcdull);  // not past it

            // A second execution reuses timestamp 0 (reset before it is written again) and resolves timestamp 1,
            // which keeps the result of the first execution.
            EXPECT(SUCCEEDED(allocator->Reset()) && SUCCEEDED(list->Reset(allocator.p, nullptr)));
            list->EndQuery(timestamps.p, D3D12_QUERY_TYPE_TIMESTAMP, 0);
            list->ResolveQueryData(timestamps.p, D3D12_QUERY_TYPE_TIMESTAMP, 0, 2, readback.p, again_at);
            EXPECT(list->Close() == S_OK);
            queue->ExecuteCommandLists(1, lists);
            EXPECT(rt::d3d12::last_execute_result(queue.p) == recity::ReplayResult::success);
            EXPECT(SUCCEEDED(queue->Signal(fence.p, 11)) && SUCCEEDED(fence->SetEventOnCompletion(11, nullptr)));
            EXPECT(u64(again_at) >= t1 && u64(again_at + 8) == t1);

            // Lists that break the rules. Close reports what it can see; the rest is refused at execution, and
            // nothing is submitted.
            const auto refuse = [&](HRESULT close, recity::ReplayResult result, auto&& record) {
                if (FAILED(allocator->Reset()) || FAILED(list->Reset(allocator.p, nullptr))) return false;
                record();
                if (list->Close() != close) return false;
                queue->ExecuteCommandLists(1, lists);
                return rt::d3d12::last_execute_result(queue.p) == result;
            };
            using R = recity::ReplayResult;
            EXPECT(refuse(E_INVALIDARG, R::invalid_query,  // begun and never ended
                          [&] { list->BeginQuery(occlusion.p, D3D12_QUERY_TYPE_OCCLUSION, 0); }));
            EXPECT(refuse(E_INVALIDARG, R::success,  // timestamps have no Begin; the call is dropped
                          [&] { list->BeginQuery(timestamps.p, D3D12_QUERY_TYPE_TIMESTAMP, 0); }));
            EXPECT(refuse(E_INVALIDARG, R::success,  // offset not a multiple of 8
                          [&] { list->ResolveQueryData(timestamps.p, D3D12_QUERY_TYPE_TIMESTAMP, 0, 1, readback.p, 4); }));
            EXPECT(refuse(S_OK, R::invalid_query,  // past the end of the heap
                          [&] { list->ResolveQueryData(occlusion.p, D3D12_QUERY_TYPE_OCCLUSION, 3, 2, readback.p, 0); }));
            EXPECT(refuse(S_OK, R::invalid_query,  // a type the heap does not hold
                          [&] { list->EndQuery(occlusion.p, D3D12_QUERY_TYPE_TIMESTAMP, 0); }));
            EXPECT(refuse(S_OK, R::invalid_query,  // into a texture
                          [&] { list->ResolveQueryData(timestamps.p, D3D12_QUERY_TYPE_TIMESTAMP, 0, 1, rt_resource.p, 0); }));
            EXPECT(refuse(S_OK, R::invalid_query,  // past the end of the buffer
                          [&] { list->ResolveQueryData(timestamps.p, D3D12_QUERY_TYPE_TIMESTAMP, 0, 2, readback.p, 65528); }));
            EXPECT(refuse(S_OK, R::unsupported, [&] {  // valid D3D12, but Vulkan runs one occlusion query at a time
                list->BeginQuery(occlusion.p, D3D12_QUERY_TYPE_OCCLUSION, 0);
                list->BeginQuery(occlusion.p, D3D12_QUERY_TYPE_BINARY_OCCLUSION, 1);
                list->EndQuery(occlusion.p, D3D12_QUERY_TYPE_BINARY_OCCLUSION, 1);
                list->EndQuery(occlusion.p, D3D12_QUERY_TYPE_OCCLUSION, 0);
            }));
            EXPECT(SUCCEEDED(allocator->Reset()) && SUCCEEDED(list->Reset(allocator.p, nullptr)));
            EXPECT(SUCCEEDED(list->Close()));
        }

        // Every child holds the proxy; once they are gone the proxy holds the only reference.
        upload->Unmap(0, nullptr);
        readback->Unmap(0, nullptr);
        for (auto* r : {&rt_resource, &packed565, &ds, &untouched, &upload, &readback, &source, &whole, &part, &bc}) r->reset();
        list7.reset();
        list.reset();
        allocator.reset();
        queue.reset();
        fence.reset();
        rtv_heap.reset();
        dsv_heap.reset();
        EXPECT(rt::d3d12_xs::device_proxy_ref_count(device.p) == 1);
    }  // the device is released here, before the Vulkan device it runs on
    EXPECT(!mismatches);

    // CPU texture copies, once per path. A second D3D12 device on the same Vulkan device, told the extension is
    // off, takes the staging path even on a GPU that has host image copy.
    for (const bool host : {true, false}) {
        if (host && !d.host_image_copy) {
            std::printf("WriteToSubresource/ReadFromSubresource (host image copy): no VK_EXT_host_image_copy, "
                        "skipped\n");
            continue;
        }
        rt::d3d12::VulkanDeviceInfo info{};
        info.get_instance_proc = d.instance_proc;
        info.instance = d.instance;
        info.physical = d.physical;
        info.device = d.device;
        info.queue = d.queue;
        info.queue_family = d.family;
        info.host_image_copy = host;
        Ref<ID3D12Device> device;
        EXPECT(SUCCEEDED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&device.p))));
        EXPECT(cpu_copies(device.p, d.bc, host ? "host image copy" : "staging copy") == 0);
    }

    // The path D3D12XboxCreateDevice takes with RECITY_D3D12=vulkan: a device on its own Vulkan instance.
    {
        Ref<ID3D12Device> standalone;
        EXPECT(SUCCEEDED(rt::d3d12::create_vulkan_device_standalone(IID_PPV_ARGS(&standalone.p))));
        D3D12_COMMAND_QUEUE_DESC queue_desc{};
        Ref<ID3D12CommandQueue> queue;
        Ref<ID3D12Fence> fence;
        EXPECT(SUCCEEDED(standalone->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue.p))));
        EXPECT(SUCCEEDED(standalone->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))));
        EXPECT(SUCCEEDED(queue->Signal(fence.p, 5)));
        EXPECT(SUCCEEDED(fence->SetEventOnCompletion(5, nullptr)) && fence->GetCompletedValue() == 5);
        UINT64 frequency = 0;
        Ref<ID3D12QueryHeap> timestamps;
        const D3D12_QUERY_HEAP_DESC query_desc{D3D12_QUERY_HEAP_TYPE_TIMESTAMP, 1, 0};
        EXPECT(SUCCEEDED(queue->GetTimestampFrequency(&frequency)) && frequency);
        EXPECT(SUCCEEDED(standalone->CreateQueryHeap(&query_desc, IID_PPV_ARGS(&timestamps.p))));
    }

    if (d.validation_errors()) {
        std::fprintf(stderr, "%u validation errors\n", d.validation_errors());
        return 1;
    }
    std::printf("Vulkan device behind the Xbox-ABI proxy: clears, zero-filled first use, uploads, texture, buffer and "
                "BC copies, a resolve, reserved resources, queries, and GetCopyableFootprints and GetResourceTiling all "
                "matched\n");
    return 0;
}
