#pragma once

// Replays a closed CommandList (command_list.h) into a Vulkan command buffer.
//
// Layouts. With a LayoutTracker, every image's current Vulkan layout is known per subresource, and barriers and
// clears start from it rather than from what the application claims. That is how D3D12's implicit state promotion
// and decay are honoured: a texture in `common` sits in GENERAL, which every promotable access (shader read, copy)
// can use directly, so a later barrier whose `before` names the promoted state still starts from GENERAL; decay
// back to `common` changes no layout. Buffers have no layouts, and simultaneous-access textures stay in GENERAL
// for their whole life, so their decay is free too. Without a tracker, `before` is trusted as the D3D12 runtime
// trusts it.
//
// Depth/stencil planes. With separate_depth_stencil_layouts (Vulkan 1.2 separateDepthStencilLayouts, enabled on
// the device), each plane has its own layout, so a transition or clear of one plane leaves the other alone. Without
// it, both planes always move together.
//
// Requirements on the device and images:
// - a queue with graphics support (the stage masks use ALL_GRAPHICS);
// - images cleared without rectangles have TRANSFER_DST usage; images cleared with rectangles have attachment usage
//   and image_view must return views for them (dynamicRendering, Vulkan 1.3); reinterpreted whole-view clears
//   use the same attachment path because vkCmdClearColorImage has no view-format argument;
// - enhanced barriers need pipeline_barrier2 (synchronization2, Vulkan 1.3);
// - unordered-access clears need fill_buffer, update_buffer and copy_buffer for buffers; texture views cleared with
//   rectangles need color_attachment usage on the image (ResourceBinding::color_attachment);
// - query commands need the query functions, fill_buffer and query_pool; resolve destinations have TRANSFER_DST usage.
// A list that needs a missing function fails with ReplayResult::unsupported before anything is written.

#include "command_list.h"
#include "uav_clear.h"

#include <vulkan/vulkan.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <limits>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace recity {

// What the resolver knows about a resource. image == VK_NULL_HANDLE means a buffer in the shared GPU address
// space: its transitions become global memory barriers.
struct ResourceBinding {
    VkImage image = VK_NULL_HANDLE;
    VkFormat format = VK_FORMAT_UNDEFINED;
    VkImageAspectFlags aspects = VK_IMAGE_ASPECT_COLOR_BIT;
    std::uint32_t width = 1;
    std::uint32_t height = 1;
    std::uint32_t mip_levels = 1;
    std::uint32_t array_layers = 1;
    bool presentable = false;          // a swapchain image: common/present maps to PRESENT_SRC_KHR
    bool simultaneous_access = false;  // D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS: always GENERAL
    bool not_zeroed = false;           // D3D12_HEAP_FLAG_CREATE_NOT_ZEROED: first use skips the zero fill
    // A descriptor in the bindless set 0 has one layout, but D3D12 lets a texture be sampled both in a
    // shader-resource state and while promoted from common (GENERAL). With this set, the shader-resource states
    // use GENERAL too, so the descriptor's layout is right in both.
    bool sampled_in_general = false;
    VkBuffer buffer = VK_NULL_HANDLE;  // a buffer resource, for copies
    VkDeviceSize buffer_offset = 0;    // where it starts in `buffer` (the GPU address arena holds many)
    VkDeviceSize size = 0;             // its size in bytes
    std::uint64_t va = 0;              // a buffer's console VA (GetGPUVirtualAddress), for vertex/index views
    std::uint32_t depth = 1;
    bool volume = false;               // one array layer, including at mips with a single slice
    std::uint32_t samples = 1;         // more than one: a multisampled 2D texture
    bool rgba8_typeless = false;       // Original resource was R8G8B8A8_TYPELESS; views may reinterpret it.
    bool color_attachment = false;     // has COLOR_ATTACHMENT usage: unordered-access rectangle clears need it
};

struct VulkanAccess {
    VkImageLayout layout;
    VkAccessFlags access;
    VkPipelineStageFlags stages;
};

inline constexpr VkPipelineStageFlags shader_stages =
    VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
inline constexpr VkImageAspectFlags depth_stencil_aspects = VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;

// Maps a legacy D3D12_RESOURCE_STATES combination to the Vulkan layout, accesses and stages that cover it.
// Read-only combinations share one layout when every bit agrees on it (depth_read with shader reads stays in
// DEPTH_STENCIL_READ_ONLY_OPTIMAL, which can be sampled); otherwise they fall back to GENERAL.
// Buffer-only and extension states map to GENERAL with conservative memory access.
inline VulkanAccess vulkan_access(std::uint32_t states, bool presentable) {
    namespace s = d3d12_state;
    constexpr VkAccessFlags any_memory = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    if (states == s::common) {
        if (presentable) return {VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, 0, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT};
        return {VK_IMAGE_LAYOUT_GENERAL, any_memory, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT};
    }

    struct Bit {
        std::uint32_t state;
        VkImageLayout layout;
        VkAccessFlags access;
        VkPipelineStageFlags stages;
    };
    constexpr VkPipelineStageFlags fragment_tests =
        VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
    static constexpr Bit bits[] = {
        {s::vertex_and_constant_buffer, VK_IMAGE_LAYOUT_GENERAL,
         VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT | VK_ACCESS_UNIFORM_READ_BIT,
         VK_PIPELINE_STAGE_VERTEX_INPUT_BIT | shader_stages},
        {s::index_buffer, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_INDEX_READ_BIT, VK_PIPELINE_STAGE_VERTEX_INPUT_BIT},
        {s::render_target, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
         VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
         VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT},
        {s::unordered_access, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
         shader_stages},
        {s::depth_write, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
         VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT, fragment_tests},
        {s::depth_read, VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL, VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT,
         fragment_tests},
        {s::non_pixel_shader_resource, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_ACCESS_SHADER_READ_BIT,
         shader_stages},
        {s::pixel_shader_resource, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_ACCESS_SHADER_READ_BIT,
         VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT},
        {s::stream_out, VK_IMAGE_LAYOUT_GENERAL, any_memory, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT},
        {s::indirect_argument, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_INDIRECT_COMMAND_READ_BIT,
         VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT},
        {s::copy_dest, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_ACCESS_TRANSFER_WRITE_BIT,
         VK_PIPELINE_STAGE_TRANSFER_BIT},
        {s::copy_source, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_ACCESS_TRANSFER_READ_BIT,
         VK_PIPELINE_STAGE_TRANSFER_BIT},
        {s::resolve_dest, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_ACCESS_TRANSFER_WRITE_BIT,
         VK_PIPELINE_STAGE_TRANSFER_BIT},
        {s::resolve_source, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_ACCESS_TRANSFER_READ_BIT,
         VK_PIPELINE_STAGE_TRANSFER_BIT},
        {s::raytracing_acceleration_structure, VK_IMAGE_LAYOUT_GENERAL, any_memory,
         VK_PIPELINE_STAGE_ALL_COMMANDS_BIT},
        {s::shading_rate_source, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_MEMORY_READ_BIT,
         VK_PIPELINE_STAGE_ALL_COMMANDS_BIT},
    };

    const auto merge = [](VkImageLayout a, VkImageLayout b) {
        if (a == VK_IMAGE_LAYOUT_UNDEFINED || a == b) return b;
        const auto pair = [&](VkImageLayout x, VkImageLayout y) { return (a == x && b == y) || (a == y && b == x); };
        if (pair(VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL))
            return VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
        return VK_IMAGE_LAYOUT_GENERAL;
    };

    VulkanAccess out{VK_IMAGE_LAYOUT_UNDEFINED, 0, 0};
    std::uint32_t known = 0;
    for (const auto& bit : bits) {
        if (!(states & bit.state)) continue;
        known |= bit.state;
        out.layout = merge(out.layout, bit.layout);
        out.access |= bit.access;
        out.stages |= bit.stages;
    }
    if (states & ~known) {
        out.layout = VK_IMAGE_LAYOUT_GENERAL;
        out.access |= any_memory;
        out.stages |= VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    }
    return out;
}

// Enhanced barriers: D3D12_BARRIER_SYNC -> VkPipelineStageFlags2. split (the end half of a split barrier) waits
// for everything, since the begin half's scope is not kept.
inline VkPipelineStageFlags2 vulkan_sync(std::uint32_t sync) {
    namespace y = d3d12_sync;
    if (sync == y::none) return VK_PIPELINE_STAGE_2_NONE;
    constexpr VkPipelineStageFlags2 pre_raster = VK_PIPELINE_STAGE_2_PRE_RASTERIZATION_SHADERS_BIT;
    struct Bit {
        std::uint32_t sync;
        VkPipelineStageFlags2 stages;
    };
    static constexpr Bit bits[] = {
        {y::all, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT},
        {y::draw, VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT},
        {y::index_input, VK_PIPELINE_STAGE_2_INDEX_INPUT_BIT},
        {y::vertex_shading, pre_raster | VK_PIPELINE_STAGE_2_VERTEX_ATTRIBUTE_INPUT_BIT},
        {y::pixel_shading, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT},
        {y::depth_stencil,
         VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT},
        {y::render_target, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT},
        {y::compute_shading, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT},
        {y::copy, VK_PIPELINE_STAGE_2_COPY_BIT},
        {y::resolve, VK_PIPELINE_STAGE_2_RESOLVE_BIT},
        {y::execute_indirect, VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT},
        {y::all_shading, pre_raster | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT},
        {y::non_pixel_shading, pre_raster | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT},
        {y::clear_unordered_access_view, VK_PIPELINE_STAGE_2_CLEAR_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT},
    };
    VkPipelineStageFlags2 out = 0;
    std::uint32_t known = 0;
    for (const auto& bit : bits) {
        if (!(sync & bit.sync)) continue;
        known |= bit.sync;
        out |= bit.stages;
    }
    // Ray tracing, video and split have no narrower equivalent without extensions.
    if (sync & ~known) out |= VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    return out;
}

// D3D12_BARRIER_ACCESS -> VkAccessFlags2. common means "any access the layout allows".
inline VkAccessFlags2 vulkan_barrier_access(std::uint32_t access) {
    namespace a = d3d12_access;
    constexpr VkAccessFlags2 any_memory = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
    if (access & a::no_access) return VK_ACCESS_2_NONE;
    if (access == a::common) return any_memory;
    struct Bit {
        std::uint32_t access;
        VkAccessFlags2 flags;
    };
    static constexpr Bit bits[] = {
        {a::vertex_buffer, VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT},
        {a::constant_buffer, VK_ACCESS_2_UNIFORM_READ_BIT},
        {a::index_buffer, VK_ACCESS_2_INDEX_READ_BIT},
        {a::render_target, VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT},
        {a::unordered_access, VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT},
        {a::depth_stencil_write,
         VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT},
        {a::depth_stencil_read, VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT},
        {a::shader_resource, VK_ACCESS_2_SHADER_READ_BIT},
        {a::stream_output, VK_ACCESS_2_MEMORY_WRITE_BIT},
        {a::indirect_argument, VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT},
        {a::copy_dest, VK_ACCESS_2_TRANSFER_WRITE_BIT},
        {a::copy_source, VK_ACCESS_2_TRANSFER_READ_BIT},
        {a::resolve_dest, VK_ACCESS_2_TRANSFER_WRITE_BIT},
        {a::resolve_source, VK_ACCESS_2_TRANSFER_READ_BIT},
        {a::raytracing_acceleration_structure_read, VK_ACCESS_2_MEMORY_READ_BIT},
        {a::raytracing_acceleration_structure_write, VK_ACCESS_2_MEMORY_WRITE_BIT},
        {a::shading_rate_source, VK_ACCESS_2_MEMORY_READ_BIT},
    };
    VkAccessFlags2 out = 0;
    std::uint32_t known = 0;
    for (const auto& bit : bits) {
        if (!(access & bit.access)) continue;
        known |= bit.access;
        out |= bit.flags;
    }
    if (access & ~known) out |= any_memory;  // video and unknown bits
    return out;
}

// For one VkAccessFlagBits2 that vulkan_barrier_access produces: the stages Vulkan lets perform it (the "supported
// access types" table of the synchronization2 chapter, with the meta stages that contain them), and the narrowest
// stage to add when a barrier's stages have none of those. MEMORY_READ/WRITE are valid in every stage.
struct AccessStages {
    VkAccessFlags2 access;
    VkPipelineStageFlags2 supported;
    VkPipelineStageFlags2 narrowest;
};

inline constexpr VkPipelineStageFlags2 shader_stages2 =
    VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_TESSELLATION_CONTROL_SHADER_BIT |
    VK_PIPELINE_STAGE_2_TESSELLATION_EVALUATION_SHADER_BIT | VK_PIPELINE_STAGE_2_GEOMETRY_SHADER_BIT |
    VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
    VK_PIPELINE_STAGE_2_PRE_RASTERIZATION_SHADERS_BIT | VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT |
    VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;

inline constexpr AccessStages access_stages[] = {
    {VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT,
     VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT | VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
     VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT},
    {VK_ACCESS_2_INDEX_READ_BIT,
     VK_PIPELINE_STAGE_2_INDEX_INPUT_BIT | VK_PIPELINE_STAGE_2_VERTEX_INPUT_BIT | VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT |
         VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
     VK_PIPELINE_STAGE_2_INDEX_INPUT_BIT},
    {VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT,
     VK_PIPELINE_STAGE_2_VERTEX_ATTRIBUTE_INPUT_BIT | VK_PIPELINE_STAGE_2_VERTEX_INPUT_BIT |
         VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT | VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
     VK_PIPELINE_STAGE_2_VERTEX_ATTRIBUTE_INPUT_BIT},
    // Shader accesses: D3D12 does not say which shader stage, so every one that can run (all shading).
    {VK_ACCESS_2_UNIFORM_READ_BIT, shader_stages2,
     VK_PIPELINE_STAGE_2_PRE_RASTERIZATION_SHADERS_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT |
         VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT},
    {VK_ACCESS_2_SHADER_READ_BIT, shader_stages2,
     VK_PIPELINE_STAGE_2_PRE_RASTERIZATION_SHADERS_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT |
         VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT},
    {VK_ACCESS_2_SHADER_WRITE_BIT, shader_stages2,
     VK_PIPELINE_STAGE_2_PRE_RASTERIZATION_SHADERS_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT |
         VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT},
    {VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT,
     VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT |
         VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
     VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT},
    {VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
     VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT |
         VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
     VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT},
    {VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT,
     VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT |
         VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT | VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
     VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT},
    {VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
     VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT |
         VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT | VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
     VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT},
    // Transfers: ALL_GRAPHICS does not contain the transfer stages, and CLEAR only writes.
    {VK_ACCESS_2_TRANSFER_READ_BIT,
     VK_PIPELINE_STAGE_2_COPY_BIT | VK_PIPELINE_STAGE_2_BLIT_BIT | VK_PIPELINE_STAGE_2_RESOLVE_BIT |
         VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT | VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
     VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT},
    {VK_ACCESS_2_TRANSFER_WRITE_BIT,
     VK_PIPELINE_STAGE_2_COPY_BIT | VK_PIPELINE_STAGE_2_BLIT_BIT | VK_PIPELINE_STAGE_2_RESOLVE_BIT |
         VK_PIPELINE_STAGE_2_CLEAR_BIT | VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT | VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
     VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT},
};

// One side of an enhanced barrier: D3D12 sync and access to a Vulkan stage mask and access mask that Vulkan accepts
// together. D3D12 lets an access be named with a sync scope whose Vulkan stages cannot perform it (SYNC_ALL_SHADING
// with ACCESS_VERTEX_BUFFER is valid D3D12, but VERTEX_ATTRIBUTE_READ needs VERTEX_ATTRIBUTE_INPUT). The stages are
// widened by the narrowest stage that can, never the access dropped: the dependency still covers every access the
// application named, only over a slightly larger scope. With no stages (SYNC_NONE) there is no access either.
struct BarrierScope {
    VkPipelineStageFlags2 stages;
    VkAccessFlags2 access;
};

inline BarrierScope vulkan_barrier_scope(std::uint32_t sync, std::uint32_t access) {
    BarrierScope out{vulkan_sync(sync), 0};
    if (!out.stages) return out;
    out.access = vulkan_barrier_access(access);
    for (const AccessStages& entry : access_stages)
        if ((out.access & entry.access) && !(out.stages & entry.supported)) out.stages |= entry.narrowest;
    return out;
}

// D3D12_BARRIER_LAYOUT -> VkImageLayout. Queue-specific layouts map like their generic forms.
inline VkImageLayout vulkan_layout(std::uint32_t layout, bool presentable) {
    namespace l = d3d12_layout;
    switch (layout) {
    case l::undefined: return VK_IMAGE_LAYOUT_UNDEFINED;
    case l::common: return presentable ? VK_IMAGE_LAYOUT_PRESENT_SRC_KHR : VK_IMAGE_LAYOUT_GENERAL;
    case l::render_target: return VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    case l::depth_stencil_write: return VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    case l::depth_stencil_read: return VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
    case l::shader_resource:
    case l::direct_queue_shader_resource:
    case l::compute_queue_shader_resource: return VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    case l::copy_source:
    case l::resolve_source:
    case l::direct_queue_copy_source:
    case l::compute_queue_copy_source: return VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    case l::copy_dest:
    case l::resolve_dest:
    case l::direct_queue_copy_dest:
    case l::compute_queue_copy_dest: return VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    default: return VK_IMAGE_LAYOUT_GENERAL;  // generic read, UAV, shading rate, video, *_common
    }
}

// The layout an image actually uses: simultaneous-access images live in GENERAL, and so do images in a
// shader-resource state when set 0 names them (sampled_in_general).
inline VkImageLayout image_layout(const ResourceBinding& resource, VkImageLayout layout) {
    if (resource.simultaneous_access && layout != VK_IMAGE_LAYOUT_UNDEFINED &&
        layout != VK_IMAGE_LAYOUT_PRESENT_SRC_KHR)
        return VK_IMAGE_LAYOUT_GENERAL;
    if (resource.sampled_in_general && layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
        return VK_IMAGE_LAYOUT_GENERAL;
    return layout;
}

// Bytes per texel, or per 4x4 block for block-compressed formats, of one aspect as copies lay it out in a buffer
// (D3D12 footprints and Vulkan buffer-image copies agree on these). bytes == 0: not a format copies support yet.
struct BlockFormat {
    std::uint32_t bytes = 0;
    std::uint32_t block = 1;  // texels per block edge
};

inline BlockFormat block_format(VkFormat format, VkImageAspectFlags aspect) {
    if (aspect == VK_IMAGE_ASPECT_STENCIL_BIT) return {1, 1};
    switch (format) {
    case VK_FORMAT_R32G32B32A32_SFLOAT:
    case VK_FORMAT_R32G32B32A32_UINT:
    case VK_FORMAT_R32G32B32A32_SINT: return {16, 1};
    case VK_FORMAT_R16G16B16A16_SFLOAT:
    case VK_FORMAT_R16G16B16A16_UNORM:
    case VK_FORMAT_R16G16B16A16_UINT:
    case VK_FORMAT_R16G16B16A16_SINT:
    case VK_FORMAT_R32G32_SFLOAT:
    case VK_FORMAT_R32G32_UINT:
    case VK_FORMAT_R32G32_SINT: return {8, 1};
    case VK_FORMAT_A2B10G10R10_UNORM_PACK32:
    case VK_FORMAT_A2B10G10R10_UINT_PACK32:
    case VK_FORMAT_E5B9G9R9_UFLOAT_PACK32:
    case VK_FORMAT_B10G11R11_UFLOAT_PACK32:
    case VK_FORMAT_R8G8B8A8_UNORM:
    case VK_FORMAT_R8G8B8A8_SRGB:
    case VK_FORMAT_R8G8B8A8_SNORM:
    case VK_FORMAT_R8G8B8A8_UINT:
    case VK_FORMAT_R8G8B8A8_SINT:
    case VK_FORMAT_B8G8R8A8_UNORM:
    case VK_FORMAT_B8G8R8A8_SRGB:
    case VK_FORMAT_R16G16_SFLOAT:
    case VK_FORMAT_R16G16_UINT:
    case VK_FORMAT_R16G16_SINT:
    case VK_FORMAT_R32_SFLOAT:
    case VK_FORMAT_R32_UINT:
    case VK_FORMAT_R32_SINT:
    case VK_FORMAT_D32_SFLOAT:
    case VK_FORMAT_D32_SFLOAT_S8_UINT:  // depth aspect
    case VK_FORMAT_D24_UNORM_S8_UINT: return {4, 1};
    case VK_FORMAT_R8G8_UNORM:
    case VK_FORMAT_R8G8_SNORM:
    case VK_FORMAT_R5G6B5_UNORM_PACK16:
    case VK_FORMAT_A1R5G5B5_UNORM_PACK16:
    case VK_FORMAT_R8G8_UINT:
    case VK_FORMAT_R8G8_SINT:
    case VK_FORMAT_R16_SFLOAT:
    case VK_FORMAT_R16_UNORM:
    case VK_FORMAT_A4R4G4B4_UNORM_PACK16:
    case VK_FORMAT_D16_UNORM: return {2, 1};
    case VK_FORMAT_R8_UNORM: return {1, 1};
    case VK_FORMAT_BC1_RGBA_UNORM_BLOCK:
    case VK_FORMAT_BC1_RGBA_SRGB_BLOCK:
    case VK_FORMAT_BC4_UNORM_BLOCK:
    case VK_FORMAT_BC4_SNORM_BLOCK: return {8, 4};
    case VK_FORMAT_BC2_UNORM_BLOCK:
    case VK_FORMAT_BC2_SRGB_BLOCK:
    case VK_FORMAT_BC3_UNORM_BLOCK:
    case VK_FORMAT_BC3_SRGB_BLOCK:
    case VK_FORMAT_BC5_UNORM_BLOCK:
    case VK_FORMAT_BC5_SNORM_BLOCK:
    case VK_FORMAT_BC6H_UFLOAT_BLOCK:
    case VK_FORMAT_BC6H_SFLOAT_BLOCK:
    case VK_FORMAT_BC7_UNORM_BLOCK:
    case VK_FORMAT_BC7_SRGB_BLOCK: return {16, 4};
    default: return {};
    }
}

inline bool two_planes(const ResourceBinding& resource) {
    return (resource.aspects & depth_stencil_aspects) == depth_stencil_aspects;
}

// Decodes a D3D12 subresource index. Combined depth-stencil images have two planes (0 depth, 1 stencil);
// every other image has one.
inline bool image_range(const ResourceBinding& resource, std::uint32_t subresource, VkImageSubresourceRange& out) {
    if (!resource.mip_levels || !resource.array_layers) return false;
    if (subresource == all_subresources) {
        out = {resource.aspects, 0, resource.mip_levels, 0, resource.array_layers};
        return true;
    }
    const std::uint64_t per_plane = std::uint64_t(resource.mip_levels) * resource.array_layers;
    const std::uint64_t plane = subresource / per_plane;
    if (plane >= (two_planes(resource) ? 2u : 1u)) return false;
    const auto within = static_cast<std::uint32_t>(subresource % per_plane);
    const VkImageAspectFlags aspect =
        two_planes(resource) ? (plane ? VK_IMAGE_ASPECT_STENCIL_BIT : VK_IMAGE_ASPECT_DEPTH_BIT) : resource.aspects;
    out = {aspect, within % resource.mip_levels, 1, within / resource.mip_levels, 1};
    return true;
}

// Decodes D3D12_BARRIER_SUBRESOURCE_RANGE.
inline bool barrier_range(const ResourceBinding& resource, const BarrierSubresourceRange& range,
                          VkImageSubresourceRange& out) {
    if (!range.mip_count) return image_range(resource, range.index_or_first_mip, out);
    const std::uint32_t planes = two_planes(resource) ? 2 : 1;
    if (range.index_or_first_mip >= resource.mip_levels ||
        range.mip_count > resource.mip_levels - range.index_or_first_mip)
        return false;
    if (range.first_layer >= resource.array_layers || !range.layer_count ||
        range.layer_count > resource.array_layers - range.first_layer)
        return false;
    if (range.first_plane >= planes || !range.plane_count || range.plane_count > planes - range.first_plane)
        return false;
    VkImageAspectFlags aspects = resource.aspects;
    if (planes == 2) {
        aspects = 0;
        for (std::uint32_t p = range.first_plane; p < range.first_plane + range.plane_count; ++p)
            aspects |= p ? VK_IMAGE_ASPECT_STENCIL_BIT : VK_IMAGE_ASPECT_DEPTH_BIT;
    }
    out = {aspects, range.index_or_first_mip, range.mip_count, range.first_layer, range.layer_count};
    return true;
}

// A part of an image range with one current layout.
struct LayoutPiece {
    VkImageSubresourceRange range;
    VkImageLayout layout;
};

// The current Vulkan layout of every tracked image subresource. Shared by all queues of a device; lists are
// emitted in submission order. Images are tracked from creation (their initial state) until destruction.
class LayoutTracker {
public:
    explicit LayoutTracker(bool separate_depth_stencil_layouts = false)
        : separate_(separate_depth_stencil_layouts) {}

    bool separate_depth_stencil_layouts() const { return separate_; }

    void track(ResourceId id, const ResourceBinding& resource, VkImageLayout layout) {
        Entry entry{resource.mip_levels, resource.array_layers, two_planes(resource) ? 2u : 1u,
                    separate_ && two_planes(resource), {}};
        entry.layouts.assign(std::size_t(entry.mips) * entry.layers * (entry.two_planes ? 2 : 1), layout);
        entries_[id] = std::move(entry);
    }

    void forget(ResourceId id) { entries_.erase(id); }
    bool tracked(ResourceId id) const { return entries_.count(id) != 0; }

    // Current layout of one D3D12 subresource; UNDEFINED when the resource or index is unknown.
    VkImageLayout layout(ResourceId id, std::uint32_t subresource) const {
        const auto it = entries_.find(id);
        if (it == entries_.end()) return VK_IMAGE_LAYOUT_UNDEFINED;
        const Entry& e = it->second;
        const std::uint32_t per_plane = e.mips * e.layers;
        const std::uint32_t plane = subresource / per_plane;
        if (plane >= e.planes) return VK_IMAGE_LAYOUT_UNDEFINED;
        return e.layouts[(e.two_planes ? plane : 0) * per_plane + subresource % per_plane];  // merged planes share
    }

    // The current layouts over a range: one piece when they all agree, else one per subresource.
    bool pieces(ResourceId id, const VkImageSubresourceRange& range, std::vector<LayoutPiece>& out) const {
        const auto it = entries_.find(id);
        if (it == entries_.end()) return false;
        const Entry& e = it->second;
        out.clear();
        bool uniform = true;
        VkImageLayout first = VK_IMAGE_LAYOUT_UNDEFINED;
        bool any = false;
        visit(e, range, [&](std::size_t index, VkImageAspectFlags, std::uint32_t, std::uint32_t) {
            if (!any) first = e.layouts[index];
            uniform = uniform && e.layouts[index] == first;
            any = true;
        });
        if (uniform) {
            out.push_back({range, first});
            return true;
        }
        visit(e, range, [&](std::size_t index, VkImageAspectFlags aspect, std::uint32_t mip, std::uint32_t layer) {
            out.push_back({{aspect, mip, 1, layer, 1}, e.layouts[index]});
        });
        return true;
    }

    void set(ResourceId id, const VkImageSubresourceRange& range, VkImageLayout layout) {
        const auto it = entries_.find(id);
        if (it == entries_.end()) return;
        Entry& e = it->second;
        visit(e, range, [&](std::size_t index, VkImageAspectFlags, std::uint32_t, std::uint32_t) {
            e.layouts[index] = layout;
        });
    }

private:
    struct Entry {
        std::uint32_t mips;
        std::uint32_t layers;
        std::uint32_t planes;  // of the resource
        bool two_planes;       // depth and stencil tracked apart
        std::vector<VkImageLayout> layouts;
    };

    template<typename F>
    static void visit(const Entry& e, const VkImageSubresourceRange& range, F&& f) {
        const std::uint32_t mip_end = range.levelCount == VK_REMAINING_MIP_LEVELS ? e.mips
                                                                                  : range.baseMipLevel + range.levelCount;
        const std::uint32_t layer_end = range.layerCount == VK_REMAINING_ARRAY_LAYERS
                                            ? e.layers
                                            : range.baseArrayLayer + range.layerCount;
        for (std::uint32_t plane = 0; plane < (e.two_planes ? 2u : 1u); ++plane) {
            VkImageAspectFlags aspect = range.aspectMask;
            if (e.two_planes) {
                aspect = plane ? VK_IMAGE_ASPECT_STENCIL_BIT : VK_IMAGE_ASPECT_DEPTH_BIT;
                if (!(range.aspectMask & aspect)) continue;
            }
            for (std::uint32_t layer = range.baseArrayLayer; layer < std::min(layer_end, e.layers); ++layer)
                for (std::uint32_t mip = range.baseMipLevel; mip < std::min(mip_end, e.mips); ++mip)
                    f((std::size_t(plane) * e.layers + layer) * e.mips + mip, aspect, mip, layer);
        }
    }

    bool separate_;
    std::unordered_map<ResourceId, Entry> entries_;
};

// A D3D12 query heap on Vulkan. `ended` is kept on the host: which queries hold a result, updated as lists are
// emitted in submission order. A resolve of a query that was never ended writes zeros instead of waiting on it.
struct QueryPool {
    VkQueryPool pool = VK_NULL_HANDLE;
    std::uint32_t heap_type = 0;  // d3d12_query_heap
    std::uint32_t count = 0;
    bool precise = false;         // occlusionQueryPrecise: D3D12_QUERY_TYPE_OCCLUSION counts exact samples
    std::vector<bool> ended;
};

struct ReplayFunctions {
    PFN_vkCmdPipelineBarrier pipeline_barrier = nullptr;
    PFN_vkCmdClearColorImage clear_color_image = nullptr;
    PFN_vkCmdClearDepthStencilImage clear_depth_stencil_image = nullptr;
    // Optional; a list that needs a missing one is rejected with ReplayResult::unsupported.
    PFN_vkCmdPipelineBarrier2 pipeline_barrier2 = nullptr;  // enhanced barriers
    PFN_vkCmdBeginRendering begin_rendering = nullptr;      // rectangle clears
    PFN_vkCmdEndRendering end_rendering = nullptr;
    PFN_vkCmdClearAttachments clear_attachments = nullptr;
    PFN_vkCmdCopyBuffer copy_buffer = nullptr;                  // copies
    PFN_vkCmdCopyBufferToImage copy_buffer_to_image = nullptr;
    PFN_vkCmdCopyImageToBuffer copy_image_to_buffer = nullptr;
    PFN_vkCmdCopyImage copy_image = nullptr;
    PFN_vkCmdResolveImage resolve_image = nullptr;              // resolves
    PFN_vkCmdFillBuffer fill_buffer = nullptr;                  // unordered-access buffer clears, with copy_buffer
    PFN_vkCmdUpdateBuffer update_buffer = nullptr;              // WriteBufferImmediate, unordered-access clears
    // The buffer and offset holding [va, va + size), from ResourceBinding::va; WriteBufferImmediate.
    std::function<bool(std::uint64_t va, std::uint64_t size, VkBuffer&, VkDeviceSize&)> buffer_at;
    // VK_EXT_conditional_rendering: SetPredication. The predicate buffer must have been created with
    // VK_BUFFER_USAGE_CONDITIONAL_RENDERING_BIT_EXT.
    PFN_vkCmdBeginConditionalRenderingEXT begin_conditional_rendering = nullptr;
    PFN_vkCmdEndConditionalRenderingEXT end_conditional_rendering = nullptr;
    PFN_vkCmdResetQueryPool reset_query_pool = nullptr;  // queries, with fill_buffer for unavailable results
    PFN_vkCmdBeginQuery begin_query = nullptr;
    PFN_vkCmdEndQuery end_query = nullptr;
    PFN_vkCmdWriteTimestamp write_timestamp = nullptr;
    PFN_vkCmdCopyQueryPoolResults copy_query_pool_results = nullptr;
    // Returns a 2D-array view of the range (caller-owned, alive until execution completes); rectangle clears.
    std::function<VkImageView(const ResourceBinding&, const VkImageSubresourceRange&)> image_view;
    // Returns the query heap with that ID, or nullptr; it stays valid while the lists that use it run.
    std::function<QueryPool*(QueryHeapId)> query_pool;
    bool separate_depth_stencil_layouts = false;  // must match the LayoutTracker, if one is used
};

enum class ReplayResult {
    success,
    list_open,
    unresolved_resource,
    invalid_subresource,
    invalid_clear,
    invalid_copy,
    invalid_predication,
    invalid_query,
    unsupported
};

namespace detail {

struct Prepared {
    ResourceBinding binding;
    VkImageSubresourceRange range{};  // the barrier range, or the cleared view (cleared aspects only)
    VkImageView view = VK_NULL_HANDLE;
    VkFormat clear_format = VK_FORMAT_UNDEFINED; // The RTV format, without changing the resource binding.
    std::vector<VkClearRect> rects;   // rectangle clears, clipped to the mip
    bool skip = false;                // every rectangle was empty
    ResourceBinding source;                   // copies: the source resource
    VkImageSubresourceRange source_range{};   // copies: the source subresources (range: the destination's)
    std::vector<VkBufferCopy> buffer_copies;
    std::vector<VkBufferImageCopy> buffer_image_copies;  // one direction, given by which side is an image
    std::vector<VkImageCopy> image_copies;
    VkImageResolve resolve{};  // resolves
    std::vector<VkImageSubresourceRange> discards;  // DiscardResource: the subresources to discard
    VkBuffer write_buffer = VK_NULL_HANDLE;          // WriteBufferImmediate: where the address lands
    VkDeviceSize write_offset = 0;
    uav_clear::Element element;                      // unordered-access clears: one element of the view's format
    VkClearColorValue clear_color{};                 // ... the value that writes it into the image
    std::vector<std::array<VkDeviceSize, 2>> fills;  // ... buffer byte ranges, offset and size
    QueryPool* queries = nullptr;  // query commands: the heap
};

// ClearRenderTargetView takes FLOATs even for integer RTVs. Windows D3D12 (WARP and AMD) clamps to the
// component range and truncates fractions. Vulkan instead interprets the matching union member as an integer
// already; copying float bits changes the value without a validation error.
inline VkClearColorValue render_target_clear_color(VkFormat format, const ClearColor& input) {
    // Integer formats convert the float numerically: NaN maps to zero, values below the range clamp to the low
    // end, positive overflow (and +infinity) to the high end, and fractions truncate toward zero. Unsigned uses
    // uint32, signed int32. A2B10G10R10 clamps RGB to 10 bits and alpha (channel 3) to 2 bits.
    constexpr double u32_max = 4294967295.0, s32_min = -2147483648.0, s32_max = 2147483647.0;
    double low = 0, high = 0, alpha_high = 0;
    bool is_integer = true, is_signed = false;
    switch (format) {
    case VK_FORMAT_R8_UINT:
    case VK_FORMAT_R8G8_UINT:
    case VK_FORMAT_R8G8B8A8_UINT: high = 255; break;
    case VK_FORMAT_R16_UINT:
    case VK_FORMAT_R16G16_UINT:
    case VK_FORMAT_R16G16B16A16_UINT: high = 65535; break;
    case VK_FORMAT_R32_UINT:
    case VK_FORMAT_R32G32_UINT:
    case VK_FORMAT_R32G32B32A32_UINT: high = u32_max; break;
    case VK_FORMAT_A2B10G10R10_UINT_PACK32: high = 1023; alpha_high = 3; break;
    case VK_FORMAT_R8_SINT:
    case VK_FORMAT_R8G8_SINT:
    case VK_FORMAT_R8G8B8A8_SINT: low = -128; high = 127; is_signed = true; break;
    case VK_FORMAT_R16_SINT:
    case VK_FORMAT_R16G16_SINT:
    case VK_FORMAT_R16G16B16A16_SINT: low = -32768; high = 32767; is_signed = true; break;
    case VK_FORMAT_R32_SINT:
    case VK_FORMAT_R32G32_SINT:
    case VK_FORMAT_R32G32B32A32_SINT: low = s32_min; high = s32_max; is_signed = true; break;
    default: is_integer = false; break;
    }
    if (format != VK_FORMAT_A2B10G10R10_UINT_PACK32) alpha_high = high;
    VkClearColorValue value{};
    for (std::size_t i = 0; i < input.size(); ++i) {
        if (!is_integer) {
            value.float32[i] = input[i];
            continue;
        }
        const double maximum = i == 3 ? alpha_high : high;
        const double v = double(input[i]);
        // In range the cast truncates toward zero; out-of-range values never reach a float-to-int cast.
        const double clamped = v != v ? 0.0 : v <= low ? low : v >= maximum ? maximum : v;
        if (is_signed) value.int32[i] = static_cast<std::int32_t>(clamped);
        else value.uint32[i] = static_cast<std::uint32_t>(clamped);
    }
    return value;
}

// Only the RGBA8 family is explicitly reinterpreted here. Legacy views retain their resource format.
inline VkFormat color_view_format(const ResourceBinding& resource, const TargetView& target) {
    if (target.format == ColorViewFormat::resource)
        return resource.rgba8_typeless ? VK_FORMAT_UNDEFINED : resource.format;
    VkFormat format = VK_FORMAT_UNDEFINED;
    switch (target.format) {
    case ColorViewFormat::rgba8_unorm: format = VK_FORMAT_R8G8B8A8_UNORM; break;
    case ColorViewFormat::rgba8_srgb: format = VK_FORMAT_R8G8B8A8_SRGB; break;
    case ColorViewFormat::rgba8_uint: format = VK_FORMAT_R8G8B8A8_UINT; break;
    default: return VK_FORMAT_UNDEFINED;
    }
    if (!resource.rgba8_typeless) return format == resource.format ? format : VK_FORMAT_UNDEFINED;
    if (resource.format != VK_FORMAT_R8G8B8A8_UNORM || resource.volume || resource.depth != 1 || resource.samples != 1)
        return VK_FORMAT_UNDEFINED;
    return format;
}

inline bool view_range(const ResourceBinding& resource, const TargetView& view, VkImageAspectFlags aspects,
                       VkImageSubresourceRange& out) {
    if (view.mip >= resource.mip_levels) return false;
    if (resource.volume) {
        const auto depth = std::max(1u, resource.depth >> view.mip);
        if (aspects != VK_IMAGE_ASPECT_COLOR_BIT || view.first_layer >= depth ||
            !view.layer_count || view.layer_count > depth - view.first_layer) return false;
        // Layouts and transfers address one array layer for the whole 3D mip.
        out = {aspects, view.mip, 1, 0, 1};
        return true;
    }
    if (view.first_layer >= resource.array_layers || view.layer_count > resource.array_layers - view.first_layer)
        return false;
    out = {aspects, view.mip, 1, view.first_layer, view.layer_count};
    return true;
}

// Without separate depth/stencil layouts both planes of a combined image move together.
inline VkImageSubresourceRange planes_for_layout(const ResourceBinding& resource, VkImageSubresourceRange range,
                                                 bool separate) {
    if (two_planes(resource) && !separate) range.aspectMask = depth_stencil_aspects;
    return range;
}

inline VkExtent2D mip_extent(const ResourceBinding& resource, std::uint32_t mip) {
    return {std::max(1u, resource.width >> mip), std::max(1u, resource.height >> mip)};
}

inline std::vector<VkClearRect> clip(std::span<const Rect> rects, VkExtent2D extent, std::uint32_t layers) {
    std::vector<VkClearRect> out;
    for (const Rect& r : rects) {
        const std::int64_t left = std::max<std::int64_t>(r.left, 0), top = std::max<std::int64_t>(r.top, 0);
        const std::int64_t right = std::min<std::int64_t>(r.right, extent.width);
        const std::int64_t bottom = std::min<std::int64_t>(r.bottom, extent.height);
        if (right <= left || bottom <= top) continue;
        out.push_back({{{std::int32_t(left), std::int32_t(top)}, {std::uint32_t(right - left), std::uint32_t(bottom - top)}},
                       0, layers});
    }
    return out;
}

// A D3D12 barrier on a volume mip covers every W slice, including on devices with maintenance9.
inline VkImageSubresourceRange layout_range(const ResourceBinding& resource, VkImageSubresourceRange range) {
    if (resource.volume) range.layerCount = VK_REMAINING_ARRAY_LAYERS;
    return range;
}

inline void image_barrier(const ReplayFunctions& vk, VkCommandBuffer command_buffer, const ResourceBinding& resource,
                          const VkImageSubresourceRange& range, VkImageLayout old_layout, VkImageLayout new_layout,
                          VkAccessFlags src_access, VkPipelineStageFlags src_stages, VkAccessFlags dst_access,
                          VkPipelineStageFlags dst_stages) {
    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.srcAccessMask = src_access;
    barrier.dstAccessMask = dst_access;
    barrier.oldLayout = old_layout;
    barrier.newLayout = new_layout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = resource.image;
    barrier.subresourceRange = layout_range(resource, range);
    vk.pipeline_barrier(command_buffer, src_stages, dst_stages, 0, 0, nullptr, 0, nullptr, 1, &barrier);
}

inline void memory_barrier(const ReplayFunctions& vk, VkCommandBuffer command_buffer, VkAccessFlags src_access,
                           VkPipelineStageFlags src_stages, VkAccessFlags dst_access, VkPipelineStageFlags dst_stages) {
    VkMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = src_access;
    barrier.dstAccessMask = dst_access;
    vk.pipeline_barrier(command_buffer, src_stages, dst_stages, 0, 1, &barrier, 0, nullptr, 0, nullptr);
}

// The current layouts of a range: from the tracker, or `assumed` for an untracked image.
inline std::vector<LayoutPiece> current(const LayoutTracker* tracker, ResourceId id, const VkImageSubresourceRange& range,
                                        VkImageLayout assumed) {
    std::vector<LayoutPiece> out;
    if (!tracker || !tracker->pieces(id, range, out)) out = {{range, assumed}};
    return out;
}

// Images registered with the tracker in UNDEFINED have never been used. D3D12 committed resources start zeroed,
// so each such piece is cleared to zero (colour 0, depth 0, stencil 0) and moved to `layout`, the layout the
// operation about to run expects, before that operation; with not_zeroed, or a block-compressed format (which
// Vulkan cannot clear), it is only moved. The tracker records it.
inline void initialize(const ReplayFunctions& vk, VkCommandBuffer command_buffer, const ResourceBinding& resource,
                       ResourceId id, std::vector<LayoutPiece>& pieces, VkImageLayout layout, LayoutTracker* tracker) {
    constexpr VkAccessFlags any_memory = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    for (auto& piece : pieces) {
        if (piece.layout != VK_IMAGE_LAYOUT_UNDEFINED) continue;
        // Vulkan cannot clear block-compressed images; those start undefined (applications upload them first).
        const bool compressed = block_format(resource.format, VK_IMAGE_ASPECT_COLOR_BIT).block > 1;
        if (resource.not_zeroed || compressed) {
            image_barrier(vk, command_buffer, resource, piece.range, VK_IMAGE_LAYOUT_UNDEFINED, layout, 0,
                          VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, any_memory, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
        } else {
            const VkImageLayout dst = image_layout(resource, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
            image_barrier(vk, command_buffer, resource, piece.range, VK_IMAGE_LAYOUT_UNDEFINED, dst, 0,
                          VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                          VK_PIPELINE_STAGE_TRANSFER_BIT);
            if (piece.range.aspectMask & VK_IMAGE_ASPECT_COLOR_BIT) {
                const VkClearColorValue zero{};
                vk.clear_color_image(command_buffer, resource.image, dst, &zero, 1, &piece.range);
            } else {
                const VkClearDepthStencilValue zero{};
                vk.clear_depth_stencil_image(command_buffer, resource.image, dst, &zero, 1, &piece.range);
            }
            image_barrier(vk, command_buffer, resource, piece.range, dst, layout, VK_ACCESS_TRANSFER_WRITE_BIT,
                          VK_PIPELINE_STAGE_TRANSFER_BIT, any_memory, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
        }
        piece.layout = layout;
        if (tracker) tracker->set(id, piece.range, layout);
    }
}

inline bool within(std::uint64_t offset, std::uint64_t size, std::uint64_t total) {
    return offset <= total && size <= total - offset;
}

// The bytes of a buffer UAV clear, and whether the transfer commands can write them. Elements of up to 4 bytes
// repeat in every aligned word, so vkCmdFillBuffer writes the aligned middle and copies from it the unaligned ends
// (vkCmdCopyBuffer has no alignment rule); that needs one whole word in the middle. Wider elements are written
// with vkCmdUpdateBuffer and doubled with copies, which needs a word-aligned range.
inline ReplayResult prepare_buffer_uav_clear(const ClearUnorderedAccess& clear, Prepared& p) {
    const UavView& view = clear.view;
    if (!p.binding.buffer || p.binding.image) return ReplayResult::invalid_clear;
    if (view.offset >= p.binding.size) return ReplayResult::invalid_subresource;
    // A V# may reach past the end of the buffer (the hardware bounds-checks it); the clear stops at the end.
    const std::uint64_t size = std::min(view.size, p.binding.size - view.offset);
    if (view.format) {
        if (!uav_clear::encode(VkFormat(view.format), clear.is_float, clear.values, p.element))
            return ReplayResult::unsupported;
        if (view.element_size != p.element.size) return ReplayResult::invalid_clear;
    } else {
        p.element = uav_clear::raw_element(clear.values);
    }
    const std::uint32_t e = p.element.size;
    const auto add = [&](std::uint64_t offset, std::uint64_t bytes) {
        if (!bytes) return ReplayResult::success;
        if (offset % std::min(e, 4u)) return ReplayResult::invalid_clear;
        if (e <= 4) {
            const std::uint64_t a0 = (offset + 3) / 4 * 4, a1 = (offset + bytes) / 4 * 4;
            if ((a0 != offset || a1 != offset + bytes) && a1 < a0 + 4) return ReplayResult::unsupported;
        } else if (bytes % 4) {
            return ReplayResult::unsupported;
        }
        p.fills.push_back({offset, bytes});
        return ReplayResult::success;
    };
    if (clear.rects.empty()) return add(view.offset, size);
    const std::uint64_t elements = size / view.element_size;
    for (const Rect& r : clear.rects) {
        if (r.top > 0 || r.bottom <= 0) continue;  // a buffer has one row
        const std::uint64_t left = std::uint64_t(std::max<std::int64_t>(r.left, 0));
        const std::uint64_t right = std::min<std::uint64_t>(std::uint64_t(std::max<std::int64_t>(r.right, 0)), elements);
        if (right <= left) continue;
        if (const auto result = add(view.offset + left * view.element_size, (right - left) * view.element_size);
            result != ReplayResult::success)
            return result;
    }
    if (p.fills.empty()) p.skip = true;
    return ReplayResult::success;
}

// Writes one prepared range; the caller has made the buffer writable by transfers.
inline void emit_buffer_fill(const ReplayFunctions& vk, VkCommandBuffer command_buffer, VkBuffer buffer,
                             const uav_clear::Element& element, VkDeviceSize offset, VkDeviceSize size) {
    const auto transfer_barrier = [&] {
        memory_barrier(vk, command_buffer, VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                       VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
    };
    if (element.size <= 4) {
        std::uint32_t word = 0;
        for (unsigned i = 0; i < 4; ++i) word |= std::uint32_t(element.bytes[i % element.size]) << (8 * i);
        const VkDeviceSize a0 = (offset + 3) / 4 * 4, a1 = (offset + size) / 4 * 4;
        if (a1 > a0) vk.fill_buffer(command_buffer, buffer, a0, a1 - a0, word);
        VkBufferCopy ends[2]{};
        std::uint32_t count = 0;
        if (a0 > offset) ends[count++] = {offset + 4, offset, a0 - offset};
        if (a1 < offset + size) ends[count++] = {a1 - 4, a1, offset + size - a1};
        if (count) {
            transfer_barrier();
            vk.copy_buffer(command_buffer, buffer, buffer, count, ends);
        }
        return;
    }
    constexpr VkDeviceSize update_limit = 65536;  // vkCmdUpdateBuffer's, a multiple of every element size
    const VkDeviceSize first = std::min(size, update_limit);
    std::vector<std::uint8_t> data(static_cast<std::size_t>(first));
    for (std::size_t i = 0; i < data.size(); ++i) data[i] = element.bytes[i % element.size];
    vk.update_buffer(command_buffer, buffer, offset, first, data.data());
    for (VkDeviceSize filled = first; filled < size;) {
        const VkDeviceSize step = std::min(filled, size - filled);
        transfer_barrier();
        const VkBufferCopy copy{offset, offset + filled, step};
        vk.copy_buffer(command_buffer, buffer, buffer, 1, &copy);
        filled += step;
    }
}

// The one plane of one subresource a texture copy location names.
inline bool copy_subresource(const ResourceBinding& resource, std::uint32_t subresource, VkImageSubresourceRange& out) {
    return resource.image && subresource != all_subresources && image_range(resource, subresource, out);
}

// CopyTextureRegion: texture to texture, footprint to texture (upload) or texture to footprint (readback).
// Block-compressed footprints are padded to whole blocks; the Vulkan extent stops at the mip edge.
inline ReplayResult prepare_texture_copy(const CopyTexture& copy, Prepared& p) {
    if (!copy.dst.footprint && !copy.src.footprint) {
        VkImageSubresourceRange dst{}, src{};
        if (!copy_subresource(p.binding, copy.dst.subresource, dst) ||
            !copy_subresource(p.source, copy.src.subresource, src))
            return ReplayResult::invalid_subresource;
        if (dst.aspectMask != src.aspectMask || p.binding.volume != p.source.volume ||
            p.binding.samples != p.source.samples)
            return ReplayResult::invalid_copy;
        const VkExtent2D from = mip_extent(p.source, src.baseMipLevel), to = mip_extent(p.binding, dst.baseMipLevel);
        const auto from_depth = std::max(1u, p.source.depth >> src.baseMipLevel);
        const auto to_depth = std::max(1u, p.binding.depth >> dst.baseMipLevel);
        const CopyBox box = copy.has_box ? copy.box : CopyBox{0, 0, 0, from.width, from.height, from_depth};
        // A block-compressed box is in whole blocks and may run to the block-padded edge of the mip (the host runtime
        // accepts it); Vulkan wants the extent to stop at the mip edge, as with a footprint.
        const std::uint32_t block = block_format(p.source.format, src.aspectMask).block;
        const auto padded = [block](std::uint32_t size) { return (size + block - 1) / block * block; };
        if (box.right > padded(from.width) || box.bottom > padded(from.height) || box.back > from_depth)
            return ReplayResult::invalid_copy;
        std::uint32_t w = box.right - box.left, h = box.bottom - box.top;
        const std::uint32_t d = box.back - box.front;
        if (block > 1) {
            // D3D12 copies whole blocks: the box is rounded up to blocks and must fit the destination's padded mip.
            // Vulkan takes an extent that is a block multiple or reaches the edge of each image separately, so the
            // extent is cut where it reaches an edge, and a copy that cannot satisfy both images is refused.
            if ((box.left | box.top | copy.x | copy.y) % block || box.left >= from.width || box.top >= from.height ||
                copy.x >= padded(to.width) || copy.y >= padded(to.height))
                return ReplayResult::invalid_copy;
            w = padded(w);
            h = padded(h);
            if (w > padded(to.width) - copy.x || h > padded(to.height) - copy.y) return ReplayResult::invalid_copy;
            w = std::min({w, from.width - box.left, to.width - copy.x});
            h = std::min({h, from.height - box.top, to.height - copy.y});
            const auto fits = [block](std::uint32_t offset, std::uint32_t extent, std::uint32_t size) {
                return extent % block == 0 || offset + extent == size;
            };
            if (!fits(box.left, w, from.width) || !fits(copy.x, w, to.width) || !fits(box.top, h, from.height) ||
                !fits(copy.y, h, to.height))
                return ReplayResult::invalid_copy;
        }
        if (!within(copy.x, w, to.width) || !within(copy.y, h, to.height) || !within(copy.z, d, to_depth))
            return ReplayResult::invalid_copy;
        p.image_copies.push_back({{src.aspectMask, src.baseMipLevel, src.baseArrayLayer, 1},
                                  {std::int32_t(box.left), std::int32_t(box.top), std::int32_t(box.front)},
                                  {dst.aspectMask, dst.baseMipLevel, dst.baseArrayLayer, 1},
                                  {std::int32_t(copy.x), std::int32_t(copy.y), std::int32_t(copy.z)},
                                  {w, h, d}});
        p.range = dst;
        p.source_range = src;
        return ReplayResult::success;
    }

    const bool upload = copy.src.footprint;
    const ResourceBinding& image = upload ? p.binding : p.source;
    const ResourceBinding& buffer = upload ? p.source : p.binding;
    const CopyLocation& texture = upload ? copy.dst : copy.src;
    const CopyLocation& placed = upload ? copy.src : copy.dst;
    if (!buffer.buffer) return ReplayResult::invalid_copy;
    VkImageSubresourceRange range{};
    if (!copy_subresource(image, texture.subresource, range)) return ReplayResult::invalid_subresource;
    if (image.samples != 1) return ReplayResult::invalid_copy;  // multisampled textures have no footprint
    const BlockFormat format = block_format(image.format, range.aspectMask);
    if (!format.bytes || placed.row_pitch % format.bytes) return ReplayResult::unsupported;
    const VkExtent2D mip = mip_extent(image, range.baseMipLevel);
    const auto mip_depth = std::max(1u, image.depth >> range.baseMipLevel);
    // The box is in the source's coordinates: the footprint for an upload, the texture for a readback.
    const CopyBox box = copy.has_box ? copy.box
                        : upload     ? CopyBox{0, 0, 0, placed.width, placed.height, placed.depth}
                                     : CopyBox{0, 0, 0, mip.width, mip.height, mip_depth};
    const std::uint32_t texture_x = upload ? copy.x : box.left, texture_y = upload ? copy.y : box.top;
    const std::uint32_t placed_x = upload ? box.left : copy.x, placed_y = upload ? box.top : copy.y;
    std::uint32_t w = box.right - box.left, h = box.bottom - box.top;
    const auto texture_z = upload ? copy.z : box.front, placed_z = upload ? box.front : copy.z;
    const auto depth = box.back - box.front;
    if (!depth || !within(placed_z, depth, placed.depth) || !within(texture_z, depth, mip_depth) ||
        !within(placed_x, w, placed.width) || !within(placed_y, h, placed.height)) return ReplayResult::invalid_copy;
    if (texture_x >= mip.width || texture_y >= mip.height) return ReplayResult::invalid_copy;
    if (format.block > 1) {
        w = std::min(w, mip.width - texture_x);
        h = std::min(h, mip.height - texture_y);
    }
    if (!within(texture_x, w, mip.width) || !within(texture_y, h, mip.height)) return ReplayResult::invalid_copy;
    if ((placed_x | placed_y | texture_x | texture_y) % format.block) return ReplayResult::invalid_copy;
    const auto slice_rows = (std::uint64_t(placed.height) + format.block - 1) / format.block;
    const std::uint64_t slice_pitch = slice_rows * placed.row_pitch;
    if (!slice_pitch || placed.offset > buffer.size || placed_z > (buffer.size - placed.offset) / slice_pitch)
        return ReplayResult::invalid_copy;
    const auto slice_start = placed.offset + placed_z * slice_pitch;
    const auto in_slice = std::uint64_t(placed_y / format.block) * placed.row_pitch +
                          std::uint64_t(placed_x / format.block) * format.bytes;
    if (!within(slice_start, in_slice, buffer.size)) return ReplayResult::invalid_copy;
    const auto offset = slice_start + in_slice;
    const std::uint32_t alignment = (range.aspectMask & depth_stencil_aspects) ? 4 : format.bytes;
    if (offset % alignment) return ReplayResult::unsupported;
    const std::uint64_t rows = (std::uint64_t(h) + format.block - 1) / format.block;
    const std::uint64_t row_bytes = (std::uint64_t(w) + format.block - 1) / format.block * format.bytes;
    if (!rows || !row_bytes || !within(std::uint64_t(placed_x / format.block) * format.bytes, row_bytes, placed.row_pitch) ||
        depth - 1 > (buffer.size - offset) / slice_pitch ||
        !within(offset + (depth - 1) * slice_pitch, (rows - 1) * placed.row_pitch + row_bytes, buffer.size))
        return ReplayResult::invalid_copy;
    VkBufferImageCopy region{};
    region.bufferOffset = offset;
    region.bufferRowLength = placed.row_pitch / format.bytes * format.block;
    region.bufferImageHeight = depth > 1 ? std::uint32_t(slice_rows * format.block) : 0;
    region.imageSubresource = {range.aspectMask, range.baseMipLevel, range.baseArrayLayer, 1};
    region.imageOffset = {std::int32_t(texture_x), std::int32_t(texture_y), std::int32_t(texture_z)};
    region.imageExtent = {w, h, depth};
    p.buffer_image_copies.push_back(region);
    (upload ? p.range : p.source_range) = range;
    return ReplayResult::success;
}

// CopyResource: the whole buffer, or every subresource of two textures of the same shape.
inline ReplayResult prepare_resource_copy(Prepared& p) {
    if (p.binding.buffer && p.source.buffer) {
        if (p.binding.size != p.source.size) return ReplayResult::invalid_copy;
        p.buffer_copies.push_back({0, 0, p.source.size});
        return ReplayResult::success;
    }
    const ResourceBinding &dst = p.binding, &src = p.source;
    if (!dst.image || !src.image || dst.width != src.width || dst.height != src.height ||
        dst.depth != src.depth || dst.volume != src.volume || dst.mip_levels != src.mip_levels ||
        dst.array_layers != src.array_layers || dst.aspects != src.aspects || dst.samples != src.samples ||
        block_format(dst.format, VK_IMAGE_ASPECT_COLOR_BIT).bytes != block_format(src.format, VK_IMAGE_ASPECT_COLOR_BIT).bytes)
        return ReplayResult::invalid_copy;
    for (const VkImageAspectFlags aspect : {VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_ASPECT_DEPTH_BIT, VK_IMAGE_ASPECT_STENCIL_BIT}) {
        if (!(dst.aspects & aspect)) continue;
        for (std::uint32_t mip = 0; mip < dst.mip_levels; ++mip) {
            const VkExtent2D extent = mip_extent(dst, mip);
            p.image_copies.push_back({{aspect, mip, 0, dst.array_layers}, {0, 0, 0}, {aspect, mip, 0, dst.array_layers},
                                      {0, 0, 0}, {extent.width, extent.height, std::max(1u, src.depth >> mip)}});
        }
    }
    p.range = {dst.aspects, 0, dst.mip_levels, 0, dst.array_layers};
    p.source_range = {src.aspects, 0, src.mip_levels, 0, src.array_layers};
    return ReplayResult::success;
}

// Resolve a multisampled colour image into a single-sampled image of the same format.
inline ReplayResult prepare_resolve(const ResolveSubresource& resolve, Prepared& p) {
    VkImageSubresourceRange dst{}, src{};
    if (!copy_subresource(p.binding, resolve.dst_subresource, dst) ||
        !copy_subresource(p.source, resolve.src_subresource, src))
        return ReplayResult::invalid_subresource;
    if (p.source.samples <= 1 || p.binding.samples != 1 || p.binding.volume) return ReplayResult::invalid_copy;
    const VkExtent2D from = mip_extent(p.source, src.baseMipLevel), to = mip_extent(p.binding, dst.baseMipLevel);
    VkOffset3D src_offset{}, dst_offset{};
    VkExtent3D extent{from.width, from.height, 1};
    if (const auto& region = resolve.region) {
        dst_offset = {std::int32_t(region->dst_x), std::int32_t(region->dst_y), 0};
        if (const auto& r = region->src_rect) {
            if (std::uint32_t(r->right) > from.width || std::uint32_t(r->bottom) > from.height)
                return ReplayResult::invalid_copy;
            src_offset = {r->left, r->top, 0};
            extent = {std::uint32_t(r->right - r->left), std::uint32_t(r->bottom - r->top), 1};
        }
        if (!within(region->dst_x, extent.width, to.width) || !within(region->dst_y, extent.height, to.height))
            return ReplayResult::invalid_copy;
    } else if (from.width != to.width || from.height != to.height) {
        return ReplayResult::invalid_copy;
    }
    if (src.aspectMask != VK_IMAGE_ASPECT_COLOR_BIT || dst.aspectMask != VK_IMAGE_ASPECT_COLOR_BIT ||
        p.source.format != p.binding.format)
        return ReplayResult::unsupported;
    p.resolve = {{src.aspectMask, src.baseMipLevel, src.baseArrayLayer, 1}, src_offset,
                 {dst.aspectMask, dst.baseMipLevel, dst.baseArrayLayer, 1}, dst_offset, extent};
    p.range = dst;
    p.source_range = src;
    return ReplayResult::success;
}

// The layout a copy uses for a range: D3D12 requires copy_dest/copy_source, or common (GENERAL), where an implicit
// promotion lets the copy run in place. Anything else is moved to `wanted` and tracked.
inline VkImageLayout copy_layout(const ReplayFunctions& vk, VkCommandBuffer command_buffer,
                                 const ResourceBinding& resource, ResourceId id, const VkImageSubresourceRange& range,
                                 VkImageLayout wanted, VkAccessFlags access, LayoutTracker* tracker) {
    const VkImageLayout target = image_layout(resource, wanted);
    auto pieces = current(tracker, id, range, target);
    initialize(vk, command_buffer, resource, id, pieces, target, tracker);
    // D3D12 orders back-to-back copies and resolves on a resource that stays in COPY_DEST or COPY_SOURCE without a
    // barrier from the application; Vulkan does not. A piece already in the transfer layout gets no layout
    // transition, so an earlier transfer write to it is ordered here instead (write-after-write and read-after-write,
    // which synchronization validation reports otherwise).
    const auto after_transfer_writes = [&] {
        memory_barrier(vk, command_buffer, VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, access,
                       VK_PIPELINE_STAGE_TRANSFER_BIT);
    };
    if (pieces.size() == 1 && (pieces[0].layout == target || pieces[0].layout == VK_IMAGE_LAYOUT_GENERAL)) {
        after_transfer_writes();
        return pieces[0].layout;
    }
    bool ordered = false;
    for (const auto& piece : pieces) {
        if (piece.layout == target) {
            if (!ordered) after_transfer_writes();
            ordered = true;
            continue;
        }
        image_barrier(vk, command_buffer, resource, piece.range, piece.layout, target,
                      VK_ACCESS_MEMORY_WRITE_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, access,
                      VK_PIPELINE_STAGE_TRANSFER_BIT);
    }
    if (tracker) tracker->set(id, range, target);
    return target;
}

// Moves every piece to `target` (src: the D3D12 state the operation requires), runs `body`, and moves the pieces
// back to where they were. Pieces already in `target` need no transition.
template<typename Body>
void round_trip(const ReplayFunctions& vk, VkCommandBuffer command_buffer, const ResourceBinding& resource,
                const std::vector<LayoutPiece>& pieces, VkImageLayout target, const VulkanAccess& state,
                VkAccessFlags target_access, VkPipelineStageFlags target_stages, Body&& body) {
    for (const auto& piece : pieces) {
        if (piece.layout == target) {
            memory_barrier(vk, command_buffer, state.access, state.stages, target_access, target_stages);
            continue;
        }
        image_barrier(vk, command_buffer, resource, piece.range, piece.layout, target, state.access, state.stages,
                      target_access, target_stages);
    }
    body();
    for (const auto& piece : pieces) {
        if (piece.layout == target) {
            memory_barrier(vk, command_buffer, target_access, target_stages, state.access, state.stages);
            continue;
        }
        image_barrier(vk, command_buffer, resource, piece.range, target, piece.layout, target_access, target_stages,
                      state.access, state.stages);
    }
}

} // namespace detail

// Destination and source of a copy or resolve command.
inline std::pair<ResourceId, ResourceId> copy_ids(const Command& command) {
    if (const auto* c = std::get_if<CopyBuffer>(&command)) return {c->dst, c->src};
    if (const auto* c = std::get_if<CopyTexture>(&command)) return {c->dst.resource, c->src.resource};
    if (const auto* c = std::get_if<CopyResource>(&command)) return {c->dst, c->src};
    if (const auto* c = std::get_if<ResolveSubresource>(&command)) return {c->dst, c->src};
    return {0, 0};
}

// The heap, type and query range of a query command; count 0 for any other command.
struct QueryRange {
    QueryHeapId heap = 0;
    std::uint32_t type = 0;
    std::uint32_t first = 0;
    std::uint32_t count = 0;
};

inline QueryRange query_range(const Command& command) {
    if (const auto* q = std::get_if<BeginQuery>(&command)) return {q->heap, q->type, q->index, 1};
    if (const auto* q = std::get_if<EndQuery>(&command)) return {q->heap, q->type, q->index, 1};
    if (const auto* q = std::get_if<ResolveQueries>(&command)) return {q->heap, q->type, q->first, q->count};
    return {};
}

// A list checked against the current resources, ready to emit.
class PreparedList {
public:
    const CommandList* list = nullptr;
    std::vector<detail::Prepared> commands;
    // Copies, resolves and UAV clears that run as transfers, recorded while a predicate was set. D3D12 skips them
    // when the predicate says so, but Vulkan's conditional rendering does not cover transfer commands, so they run
    // unconditionally (docs/d3d12_vulkan.md section 4.7). Counted so the caller can report it.
    std::size_t unpredicated_copies = 0;
};

// Resolves and checks every command. Nothing is written to a command buffer.
// Resolve maps a ResourceId to a const ResourceBinding* (nullptr: unknown).
template<typename Resolve>
ReplayResult prepare(const CommandList& list, Resolve&& resolve, const ReplayFunctions& vk, PreparedList& out) {
    if (!list.closed()) return ReplayResult::list_open;
    if (list.queries_open()) return ReplayResult::invalid_query;  // a Vulkan query must end in its command buffer
    const auto commands = list.commands();
    out.list = &list;
    out.commands.assign(commands.size(), {});
    const bool rendering = vk.begin_rendering && vk.end_rendering && vk.clear_attachments && vk.image_view;
    const bool copies = vk.copy_buffer && vk.copy_buffer_to_image && vk.copy_image_to_buffer && vk.copy_image;
    const bool conditional = vk.begin_conditional_rendering && vk.end_conditional_rendering;
    // While a predicate is set, a whole-view clear is replayed as one rectangle over the view: vkCmdClearColorImage
    // and vkCmdClearDepthStencilImage ignore conditional rendering, vkCmdClearAttachments does not.
    bool predicated = false;
    constexpr Rect whole_view{0, 0, INT32_MAX, INT32_MAX};
    const auto clear_rects = [&](std::span<const Rect> rects) {
        return predicated && rects.empty() ? std::span<const Rect>(&whole_view, 1) : rects;
    };
    out.unpredicated_copies = 0;
    const bool queries = vk.reset_query_pool && vk.begin_query && vk.end_query && vk.write_timestamp &&
                         vk.copy_query_pool_results && vk.fill_buffer && vk.query_pool;
    // Vulkan allows one active query of each type in a command buffer (VUID-vkCmdBeginQuery-queryPool-01922);
    // D3D12 lets queries overlap. Overlapping occlusion (either kind) or pipeline-statistics queries need virtual
    // queries (end the running Vulkan query, start another, sum on resolve); until then such a list is unsupported.
    bool occlusion_active = false, statistics_active = false;

    for (std::size_t i = 0; i < commands.size(); ++i) {
        const auto& command = commands[i];
        auto& p = out.commands[i];
        const auto bind = [&](ResourceId id) {
            const ResourceBinding* binding = resolve(id);
            if (binding) p.binding = *binding;
            return binding != nullptr;
        };
        const auto rect_view = [&](std::span<const Rect> rects, const TargetView& target, VkImageAspectFlags view_aspects) {
            if (rects.empty()) return ReplayResult::success;
            if (!rendering) return ReplayResult::unsupported;
            p.rects = detail::clip(rects, detail::mip_extent(p.binding, target.mip), target.layer_count);
            if (p.rects.empty()) {
                p.skip = true;
                return ReplayResult::success;
            }
            VkImageSubresourceRange range = p.range;
            range.aspectMask = view_aspects;
            if (p.binding.volume) {
                // A 2D-array attachment view of a 3D image selects W slices here.
                range.baseArrayLayer = target.first_layer;
                range.layerCount = target.layer_count;
            }
            ResourceBinding view_binding = p.binding;
            if (p.clear_format != VK_FORMAT_UNDEFINED) view_binding.format = p.clear_format;
            p.view = vk.image_view(view_binding, range);
            return p.view ? ReplayResult::success : ReplayResult::unsupported;
        };

        if (const auto* transition = std::get_if<TransitionBarrier>(&command)) {
            if (!bind(transition->resource)) return ReplayResult::unresolved_resource;
            if (p.binding.image == VK_NULL_HANDLE) {
                if (transition->subresource != 0 && transition->subresource != all_subresources)
                    return ReplayResult::invalid_subresource;
            } else if (!image_range(p.binding, transition->subresource, p.range)) {
                return ReplayResult::invalid_subresource;
            }
        } else if (std::holds_alternative<GlobalBarrier>(command) || std::holds_alternative<BufferBarrier>(command)) {
            if (!vk.pipeline_barrier2) return ReplayResult::unsupported;
            if (const auto* buffer_barrier = std::get_if<BufferBarrier>(&command))
                if (!bind(buffer_barrier->resource)) return ReplayResult::unresolved_resource;
        } else if (const auto* texture = std::get_if<TextureBarrier>(&command)) {
            if (!vk.pipeline_barrier2) return ReplayResult::unsupported;
            if (!bind(texture->resource)) return ReplayResult::unresolved_resource;
            if (!p.binding.image || !barrier_range(p.binding, texture->range, p.range))
                return ReplayResult::invalid_subresource;
        } else if (const auto* color_clear = std::get_if<ClearRenderTarget>(&command)) {
            if (!bind(color_clear->target.resource)) return ReplayResult::unresolved_resource;
            if (!p.binding.image || p.binding.aspects != VK_IMAGE_ASPECT_COLOR_BIT) return ReplayResult::invalid_clear;
            if (!detail::view_range(p.binding, color_clear->target, VK_IMAGE_ASPECT_COLOR_BIT, p.range))
                return ReplayResult::invalid_subresource;
            p.clear_format = detail::color_view_format(p.binding, color_clear->target);
            // Older ownerless replay callers omit the resource format for float-only clears. Preserve
            // that path; an explicit format or a typeless resource must still resolve a concrete RTV.
            if (p.clear_format == VK_FORMAT_UNDEFINED &&
                (p.binding.rgba8_typeless || color_clear->target.format != ColorViewFormat::resource))
                return ReplayResult::invalid_clear;
            const bool partial_volume = p.binding.volume &&
                color_clear->target.layer_count != std::max(1u, p.binding.depth >> color_clear->target.mip);
            if ((p.clear_format != p.binding.format || partial_volume) && color_clear->rects.empty()) {
                // vkCmdClearColorImage can neither reinterpret the format nor select W slices of a mip.
                const auto extent = detail::mip_extent(p.binding, color_clear->target.mip);
                const Rect full{0, 0, std::int32_t(extent.width), std::int32_t(extent.height)};
                if (const auto r = rect_view(std::span<const Rect>(&full, 1), color_clear->target,
                                            VK_IMAGE_ASPECT_COLOR_BIT); r != ReplayResult::success) return r;
            } else if (const auto r = rect_view(clear_rects(color_clear->rects), color_clear->target, VK_IMAGE_ASPECT_COLOR_BIT);
                       r != ReplayResult::success) return r;
        } else if (const auto* depth_clear = std::get_if<ClearDepthStencil>(&command)) {
            if (!bind(depth_clear->target.resource)) return ReplayResult::unresolved_resource;
            VkImageAspectFlags aspects = 0;
            if (depth_clear->flags & clear_depth) aspects |= VK_IMAGE_ASPECT_DEPTH_BIT;
            if (depth_clear->flags & clear_stencil) aspects |= VK_IMAGE_ASPECT_STENCIL_BIT;
            const VkImageAspectFlags available = p.binding.aspects & depth_stencil_aspects;
            if (!p.binding.image || !available || (aspects & ~available) || (p.binding.aspects & ~depth_stencil_aspects))
                return ReplayResult::invalid_clear;
            if (!detail::view_range(p.binding, depth_clear->target, aspects, p.range))
                return ReplayResult::invalid_subresource;
            if (const auto r = rect_view(clear_rects(depth_clear->rects), depth_clear->target, p.binding.aspects);
                r != ReplayResult::success)
                return r;
        } else if (const auto* uav = std::get_if<ClearUnorderedAccess>(&command)) {
            const UavView& view = uav->view;
            if (!bind(view.resource)) return ReplayResult::unresolved_resource;
            if (view.buffer) {
                if (!vk.fill_buffer || !vk.update_buffer || !vk.copy_buffer) return ReplayResult::unsupported;
                if (predicated) ++out.unpredicated_copies;  // fills and copies are transfer commands
                if (const auto r = detail::prepare_buffer_uav_clear(*uav, p); r != ReplayResult::success) return r;
                continue;
            }
            if (!p.binding.image || p.binding.aspects != VK_IMAGE_ASPECT_COLOR_BIT) return ReplayResult::invalid_clear;
            const TargetView target{view.resource, view.mip, view.first_layer, view.layer_count};
            if (!detail::view_range(p.binding, target, VK_IMAGE_ASPECT_COLOR_BIT, p.range))
                return ReplayResult::invalid_subresource;
            // The view's format turns the values into bytes; the image's own format clears them in.
            const VkFormat view_format = view.format ? VkFormat(view.format) : p.binding.format;
            if (!uav_clear::encode(view_format, uav->is_float, uav->values, p.element) ||
                !uav_clear::decode(p.binding.format, p.element, p.clear_color))
                return ReplayResult::unsupported;
            if (!uav->rects.empty() && (!p.binding.color_attachment || p.binding.volume))
                return ReplayResult::unsupported;
            // Under a predicate a whole view is cleared as one rectangle where the image can be an attachment, as
            // render targets are; otherwise vkCmdClearColorImage runs unconditionally and is counted.
            const bool attachment = p.binding.color_attachment && !p.binding.volume;
            if (predicated && uav->rects.empty() && !attachment) ++out.unpredicated_copies;
            const auto rects = attachment ? clear_rects(uav->rects) : std::span<const Rect>(uav->rects);
            if (const auto r = rect_view(rects, target, VK_IMAGE_ASPECT_COLOR_BIT); r != ReplayResult::success)
                return r;
        } else if (std::holds_alternative<CopyBuffer>(command) || std::holds_alternative<CopyTexture>(command) ||
                   std::holds_alternative<CopyResource>(command)) {
            if (!copies) return ReplayResult::unsupported;
            if (predicated) ++out.unpredicated_copies;
            const auto [dst_id, src_id] = copy_ids(command);
            const ResourceBinding* source = resolve(src_id);
            if (!bind(dst_id) || !source) return ReplayResult::unresolved_resource;
            p.source = *source;
            ReplayResult result = ReplayResult::success;
            if (const auto* buffer_copy = std::get_if<CopyBuffer>(&command)) {
                if (!p.binding.buffer || !p.source.buffer ||
                    !detail::within(buffer_copy->dst_offset, buffer_copy->size, p.binding.size) ||
                    !detail::within(buffer_copy->src_offset, buffer_copy->size, p.source.size))
                    return ReplayResult::invalid_copy;
                p.buffer_copies.push_back({buffer_copy->src_offset, buffer_copy->dst_offset, buffer_copy->size});
            } else if (const auto* texture_copy = std::get_if<CopyTexture>(&command)) {
                result = detail::prepare_texture_copy(*texture_copy, p);
            } else {
                result = detail::prepare_resource_copy(p);
            }
            if (result != ReplayResult::success) return result;
        } else if (const auto* resolve_command = std::get_if<ResolveSubresource>(&command)) {
            if (!vk.resolve_image) return ReplayResult::unsupported;
            if (predicated) ++out.unpredicated_copies;  // a transfer command, like the copies
            const ResourceBinding* source = resolve(resolve_command->src);
            if (!bind(resolve_command->dst) || !source) return ReplayResult::unresolved_resource;
            p.source = *source;
            if (const auto result = detail::prepare_resolve(*resolve_command, p); result != ReplayResult::success)
                return result;
        } else if (const auto* discard = std::get_if<DiscardResource>(&command)) {
            // Buffers and partial (rectangle) discards keep their contents, which "undefined" allows.
            if (!bind(discard->resource)) return ReplayResult::unresolved_resource;
            const std::uint64_t end = std::uint64_t(discard->first_subresource) + discard->count;
            if (!p.binding.image) {
                if (!discard->whole && end > 1) return ReplayResult::invalid_subresource;
            } else if (discard->whole) {
                VkImageSubresourceRange range{};
                if (!image_range(p.binding, all_subresources, range)) return ReplayResult::invalid_subresource;
                if (!discard->partial) p.discards.push_back(range);
            } else {
                for (std::uint64_t i = discard->first_subresource; i < end; ++i) {
                    const auto s = static_cast<std::uint32_t>(i);
                    VkImageSubresourceRange range{};
                    if (!image_range(p.binding, s, range)) return ReplayResult::invalid_subresource;
                    if (!discard->partial) p.discards.push_back(range);
                }
            }
        } else if (const auto* write = std::get_if<WriteImmediate>(&command)) {
            if (!vk.update_buffer || !vk.buffer_at) return ReplayResult::unsupported;
            if (!vk.buffer_at(write->address, 4, p.write_buffer, p.write_offset)) return ReplayResult::unresolved_resource;
            if (p.write_offset % 4) return ReplayResult::invalid_copy;
        } else if (const auto* atomic = std::get_if<AtomicCopyBuffer>(&command)) {
            if (!vk.copy_buffer) return ReplayResult::unsupported;
            if (predicated) ++out.unpredicated_copies;  // a transfer command, like the copies
            const ResourceBinding* source = resolve(atomic->src);
            if (!bind(atomic->dst) || !source) return ReplayResult::unresolved_resource;
            p.source = *source;
            if (!p.binding.buffer || !p.source.buffer || !detail::within(atomic->dst_offset, atomic->bytes, p.binding.size) ||
                !detail::within(atomic->src_offset, atomic->bytes, p.source.size))
                return ReplayResult::invalid_copy;
            for (const auto& d : atomic->dependencies) {
                const ResourceBinding* dependent = resolve(d.resource);
                if (!dependent) return ReplayResult::unresolved_resource;
                if (!dependent->image && (d.subresource != 0 || d.begin > d.end || d.end > dependent->size))
                    return ReplayResult::invalid_copy;
            }
            p.buffer_copies.push_back({p.source.buffer_offset + atomic->src_offset,
                                       p.binding.buffer_offset + atomic->dst_offset, atomic->bytes});
        } else if (const auto* predication = std::get_if<Predication>(&command)) {
            predicated = predication->buffer != 0;
            if (!predicated) continue;
            if (!conditional) return ReplayResult::unsupported;
            if (!bind(predication->buffer)) return ReplayResult::unresolved_resource;
            // Vulkan reads one 32-bit word at a 4-byte aligned offset; D3D12's value is 64 bits at an 8-byte
            // aligned one, so its low word is read (docs/d3d12_vulkan.md section 4.7).
            if (!p.binding.buffer || !detail::within(predication->offset, 4, p.binding.size))
                return ReplayResult::invalid_predication;
        } else if (const QueryRange range = query_range(command); range.count) {
            // BeginQuery, EndQuery and ResolveQueryData are not among D3D12's predicated calls, and Vulkan's
            // conditional rendering leaves query commands and transfers alone, so a predicate changes nothing here.
            if (!queries) return ReplayResult::unsupported;
            p.queries = vk.query_pool(range.heap);
            if (!p.queries) return ReplayResult::unresolved_resource;
            if (!query_fits_heap(p.queries->heap_type, range.type)) return ReplayResult::invalid_query;
            if (range.first >= p.queries->count || range.count > p.queries->count - range.first)
                return ReplayResult::invalid_query;
            if (!query_result_size(range.type)) return ReplayResult::unsupported;
            if (range.type != d3d12_query::timestamp && !std::holds_alternative<ResolveQueries>(command)) {
                bool& active = range.type == d3d12_query::pipeline_statistics ? statistics_active : occlusion_active;
                const bool begin = std::holds_alternative<BeginQuery>(command);
                if (begin && active) return ReplayResult::unsupported;
                active = begin;  // the list pairs every End with its Begin
            }
            if (const auto* resolve_queries = std::get_if<ResolveQueries>(&command)) {
                // The destination is a buffer, as D3D12 requires, large enough for every result.
                if (!bind(resolve_queries->dst)) return ReplayResult::unresolved_resource;
                if (!p.binding.buffer ||
                    !detail::within(resolve_queries->offset, std::uint64_t(range.count) * query_result_size(range.type),
                                    p.binding.size))
                    return ReplayResult::invalid_query;
            }
        }
    }
    return ReplayResult::success;
}

// Writes a prepared list. With a tracker, barriers start from the tracked layouts and update them.
inline void emit(const PreparedList& prepared, VkCommandBuffer command_buffer, const ReplayFunctions& vk,
                 LayoutTracker* tracker = nullptr) {
    const auto commands = prepared.list->commands();
    const bool separate = vk.separate_depth_stencil_layouts;
    const VulkanAccess transfer_dst = vulkan_access(d3d12_state::copy_dest, false);
    bool conditional_rendering = false;  // inside vkCmdBeginConditionalRenderingEXT

    for (std::size_t i = 0; i < commands.size(); ++i) {
        const auto& command = commands[i];
        const auto& p = prepared.commands[i];
        if (const auto* predication = std::get_if<Predication>(&command)) {
            // Conditional rendering cannot nest, so a new predicate ends the previous one first.
            if (conditional_rendering) vk.end_conditional_rendering(command_buffer);
            conditional_rendering = predication->buffer != 0;
            if (conditional_rendering) {
                VkConditionalRenderingBeginInfoEXT begin{VK_STRUCTURE_TYPE_CONDITIONAL_RENDERING_BEGIN_INFO_EXT};
                begin.buffer = p.binding.buffer;
                begin.offset = predication->offset;
                // Vulkan discards the commands when the word is 0, which is EQUAL_ZERO; INVERTED gives NOT_EQUAL_ZERO.
                begin.flags = predication->op == d3d12_predication::not_equal_zero
                                  ? VK_CONDITIONAL_RENDERING_INVERTED_BIT_EXT
                                  : 0;
                vk.begin_conditional_rendering(command_buffer, &begin);
            }
            continue;
        }
        const ResourceId id = std::visit(
            [](const auto& c) -> ResourceId {
                using T = std::decay_t<decltype(c)>;
                if constexpr (std::is_same_v<T, ClearRenderTarget> || std::is_same_v<T, ClearDepthStencil>)
                    return c.target.resource;
                else if constexpr (std::is_same_v<T, ClearUnorderedAccess>)
                    return c.view.resource;
                else if constexpr (requires { c.resource; })
                    return c.resource;
                else
                    return 0;
            },
            command);

        if (const auto* transition = std::get_if<TransitionBarrier>(&command)) {
            const VulkanAccess from = vulkan_access(transition->before, p.binding.presentable);
            const VulkanAccess to = vulkan_access(transition->after, p.binding.presentable);
            if (!p.binding.image) {
                detail::memory_barrier(vk, command_buffer, from.access, from.stages, to.access, to.stages);
                continue;
            }
            const auto range = detail::planes_for_layout(p.binding, p.range, separate);
            const VkImageLayout new_layout = image_layout(p.binding, to.layout);
            auto pieces = detail::current(tracker, id, range, image_layout(p.binding, from.layout));
            detail::initialize(vk, command_buffer, p.binding, id, pieces, image_layout(p.binding, from.layout), tracker);
            for (const auto& piece : pieces)
                detail::image_barrier(vk, command_buffer, p.binding, piece.range, piece.layout, new_layout,
                                      from.access, from.stages, to.access, to.stages);
            if (tracker) tracker->set(id, range, new_layout);
        } else if (std::holds_alternative<UavBarrier>(command)) {
            const VulkanAccess uav = vulkan_access(d3d12_state::unordered_access, false);
            detail::memory_barrier(vk, command_buffer, VK_ACCESS_SHADER_WRITE_BIT, uav.stages, uav.access, uav.stages);
        } else if (std::holds_alternative<AliasingBarrier>(command)) {
            detail::memory_barrier(vk, command_buffer, VK_ACCESS_MEMORY_WRITE_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                   VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
                                   VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
        } else if (std::holds_alternative<GlobalBarrier>(command) || std::holds_alternative<BufferBarrier>(command)) {
            const auto [sync_before, sync_after, access_before, access_after] = std::visit(
                [](const auto& c) -> std::array<std::uint32_t, 4> {
                    if constexpr (requires { c.sync_before; })
                        return {c.sync_before, c.sync_after, c.access_before, c.access_after};
                    else
                        return {};
                },
                command);
            const BarrierScope src = vulkan_barrier_scope(sync_before, access_before);
            const BarrierScope dst = vulkan_barrier_scope(sync_after, access_after);
            VkMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
            barrier.srcStageMask = src.stages;
            barrier.dstStageMask = dst.stages;
            barrier.srcAccessMask = src.access;
            barrier.dstAccessMask = dst.access;
            VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
            dependency.memoryBarrierCount = 1;
            dependency.pMemoryBarriers = &barrier;
            vk.pipeline_barrier2(command_buffer, &dependency);
        } else if (const auto* texture = std::get_if<TextureBarrier>(&command)) {
            const auto range = detail::planes_for_layout(p.binding, p.range, separate);
            const bool discard = (texture->flags & texture_barrier_discard) ||
                                 texture->layout_before == d3d12_layout::undefined;
            const VkImageLayout new_layout =
                image_layout(p.binding, vulkan_layout(texture->layout_after, p.binding.presentable));
            std::vector<LayoutPiece> pieces;
            if (discard)
                pieces = {{range, VK_IMAGE_LAYOUT_UNDEFINED}};
            else {
                const VkImageLayout before =
                    image_layout(p.binding, vulkan_layout(texture->layout_before, p.binding.presentable));
                pieces = detail::current(tracker, id, range, before);
                detail::initialize(vk, command_buffer, p.binding, id, pieces, before, tracker);
            }
            const BarrierScope src = vulkan_barrier_scope(texture->sync_before, texture->access_before);
            const BarrierScope dst = vulkan_barrier_scope(texture->sync_after, texture->access_after);
            std::vector<VkImageMemoryBarrier2> barriers;
            for (const auto& piece : pieces) {
                VkImageMemoryBarrier2 b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
                b.srcStageMask = src.stages;
                b.dstStageMask = dst.stages;
                b.srcAccessMask = src.access;
                b.dstAccessMask = dst.access;
                b.oldLayout = piece.layout;
                b.newLayout = new_layout;
                b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                b.image = p.binding.image;
                b.subresourceRange = detail::layout_range(p.binding, piece.range);
                barriers.push_back(b);
            }
            VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
            dependency.imageMemoryBarrierCount = std::uint32_t(barriers.size());
            dependency.pImageMemoryBarriers = barriers.data();
            vk.pipeline_barrier2(command_buffer, &dependency);
            if (tracker) tracker->set(id, range, new_layout);
        } else if (const auto* color_clear = std::get_if<ClearRenderTarget>(&command)) {
            if (p.skip) continue;
            const VulkanAccess state = vulkan_access(d3d12_state::render_target, false);
            auto pieces = detail::current(tracker, id, p.range, image_layout(p.binding, state.layout));
            detail::initialize(vk, command_buffer, p.binding, id, pieces, image_layout(p.binding, state.layout), tracker);
            const VkClearColorValue color = detail::render_target_clear_color(p.clear_format, color_clear->color);
            if (p.rects.empty()) {
                detail::round_trip(vk, command_buffer, p.binding, pieces, image_layout(p.binding, transfer_dst.layout),
                                   state, transfer_dst.access, transfer_dst.stages, [&] {
                    vk.clear_color_image(command_buffer, p.binding.image, image_layout(p.binding, transfer_dst.layout),
                                         &color, 1, &p.range);
                });
                continue;
            }
            const VkImageLayout attachment = image_layout(p.binding, state.layout);
            detail::round_trip(vk, command_buffer, p.binding, pieces, attachment, state, state.access,
                               state.stages, [&] {
                VkRenderingAttachmentInfo info{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
                info.imageView = p.view;
                info.imageLayout = attachment;
                info.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
                info.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
                VkRenderingInfo rendering{VK_STRUCTURE_TYPE_RENDERING_INFO};
                rendering.renderArea = {{0, 0}, detail::mip_extent(p.binding, color_clear->target.mip)};
                rendering.layerCount = color_clear->target.layer_count;
                rendering.colorAttachmentCount = 1;
                rendering.pColorAttachments = &info;
                vk.begin_rendering(command_buffer, &rendering);
                VkClearAttachment attachment_clear{VK_IMAGE_ASPECT_COLOR_BIT, 0, {}};
                attachment_clear.clearValue.color = color;
                vk.clear_attachments(command_buffer, 1, &attachment_clear, std::uint32_t(p.rects.size()), p.rects.data());
                vk.end_rendering(command_buffer);
            });
        } else if (const auto* depth_clear = std::get_if<ClearDepthStencil>(&command)) {
            if (p.skip) continue;
            // With separate layouts only the cleared planes move; otherwise both do.
            const auto range = detail::planes_for_layout(p.binding, p.range, separate);
            const VulkanAccess state = vulkan_access(d3d12_state::depth_write, false);
            auto pieces = detail::current(tracker, id, range, image_layout(p.binding, state.layout));
            detail::initialize(vk, command_buffer, p.binding, id, pieces, image_layout(p.binding, state.layout), tracker);
            const VkClearDepthStencilValue value{depth_clear->depth, depth_clear->stencil};
            if (p.rects.empty()) {
                detail::round_trip(vk, command_buffer, p.binding, pieces, image_layout(p.binding, transfer_dst.layout),
                                   state, transfer_dst.access, transfer_dst.stages, [&] {
                    vk.clear_depth_stencil_image(command_buffer, p.binding.image,
                                                 image_layout(p.binding, transfer_dst.layout), &value, 1, &p.range);
                });
                continue;
            }
            const VkImageLayout attachment = image_layout(p.binding, state.layout);
            detail::round_trip(vk, command_buffer, p.binding, pieces, attachment, state, state.access,
                               state.stages, [&] {
                VkRenderingAttachmentInfo info{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
                info.imageView = p.view;
                info.imageLayout = attachment;
                info.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
                info.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
                VkRenderingInfo rendering{VK_STRUCTURE_TYPE_RENDERING_INFO};
                rendering.renderArea = {{0, 0}, detail::mip_extent(p.binding, depth_clear->target.mip)};
                rendering.layerCount = depth_clear->target.layer_count;
                if (p.range.aspectMask & VK_IMAGE_ASPECT_DEPTH_BIT) rendering.pDepthAttachment = &info;
                if (p.range.aspectMask & VK_IMAGE_ASPECT_STENCIL_BIT) rendering.pStencilAttachment = &info;
                vk.begin_rendering(command_buffer, &rendering);
                VkClearAttachment attachment_clear{p.range.aspectMask, 0, {}};
                attachment_clear.clearValue.depthStencil = value;
                vk.clear_attachments(command_buffer, 1, &attachment_clear, std::uint32_t(p.rects.size()), p.rects.data());
                vk.end_rendering(command_buffer);
            });
        } else if (const auto* uav = std::get_if<ClearUnorderedAccess>(&command)) {
            if (p.skip) continue;
            // The UAV state, widened to transfers: a later barrier from SYNC_CLEAR_UNORDERED_ACCESS_VIEW names the
            // clear stage, which is a transfer stage in the legacy masks.
            VulkanAccess state = vulkan_access(d3d12_state::unordered_access, false);
            state.access |= VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
            state.stages |= VK_PIPELINE_STAGE_TRANSFER_BIT;
            if (uav->view.buffer) {
                constexpr VkAccessFlags transfer = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
                detail::memory_barrier(vk, command_buffer, state.access, state.stages, transfer,
                                       VK_PIPELINE_STAGE_TRANSFER_BIT);
                for (std::size_t f = 0; f < p.fills.size(); ++f) {
                    if (f)  // rectangles may overlap
                        detail::memory_barrier(vk, command_buffer, VK_ACCESS_TRANSFER_WRITE_BIT,
                                               VK_PIPELINE_STAGE_TRANSFER_BIT, transfer, VK_PIPELINE_STAGE_TRANSFER_BIT);
                    detail::emit_buffer_fill(vk, command_buffer, p.binding.buffer, p.element, p.fills[f][0],
                                             p.fills[f][1]);
                }
                detail::memory_barrier(vk, command_buffer, VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                       state.access, state.stages);
                continue;
            }
            // Cleared in place in GENERAL, the UAV layout, which transfer clears and attachments both accept.
            const VkImageLayout layout = image_layout(p.binding, state.layout);
            auto pieces = detail::current(tracker, id, p.range, layout);
            detail::initialize(vk, command_buffer, p.binding, id, pieces, layout, tracker);
            if (p.rects.empty()) {
                detail::round_trip(vk, command_buffer, p.binding, pieces, layout, state,
                                   VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, [&] {
                    vk.clear_color_image(command_buffer, p.binding.image, layout, &p.clear_color, 1, &p.range);
                });
                continue;
            }
            detail::round_trip(vk, command_buffer, p.binding, pieces, layout, state,
                               VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                               VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, [&] {
                VkRenderingAttachmentInfo info{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
                info.imageView = p.view;
                info.imageLayout = layout;
                info.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
                info.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
                VkRenderingInfo rendering{VK_STRUCTURE_TYPE_RENDERING_INFO};
                rendering.renderArea = {{0, 0}, detail::mip_extent(p.binding, uav->view.mip)};
                rendering.layerCount = uav->view.layer_count;
                rendering.colorAttachmentCount = 1;
                rendering.pColorAttachments = &info;
                vk.begin_rendering(command_buffer, &rendering);
                VkClearAttachment attachment_clear{VK_IMAGE_ASPECT_COLOR_BIT, 0, {}};
                attachment_clear.clearValue.color = p.clear_color;
                vk.clear_attachments(command_buffer, 1, &attachment_clear, std::uint32_t(p.rects.size()), p.rects.data());
                vk.end_rendering(command_buffer);
            });
        } else if (std::holds_alternative<CopyBuffer>(command) || std::holds_alternative<CopyTexture>(command) ||
                   std::holds_alternative<CopyResource>(command)) {
            const auto [dst_id, src_id] = copy_ids(command);
            if (!p.buffer_copies.empty()) {
                vk.copy_buffer(command_buffer, p.source.buffer, p.binding.buffer, std::uint32_t(p.buffer_copies.size()),
                               p.buffer_copies.data());
            } else if (!p.image_copies.empty()) {
                const VkImageLayout src_layout = detail::copy_layout(
                    vk, command_buffer, p.source, src_id, detail::planes_for_layout(p.source, p.source_range, separate),
                    VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_ACCESS_TRANSFER_READ_BIT, tracker);
                const VkImageLayout dst_layout = detail::copy_layout(
                    vk, command_buffer, p.binding, dst_id, detail::planes_for_layout(p.binding, p.range, separate),
                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_ACCESS_TRANSFER_WRITE_BIT, tracker);
                const auto copy_regions = [&](std::uint32_t count, const VkImageCopy* regions) {
                    vk.copy_image(command_buffer, p.source.image, src_layout, p.binding.image, dst_layout, count, regions);
                };
                if (std::holds_alternative<CopyResource>(command) &&
                    p.binding.aspects == (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT)) {
                    // prepare_resource_copy groups mips by aspect. Order writes to the two aspects of the
                    // packed destination just as consecutive per-plane CopyTextureRegion calls are ordered.
                    const auto mips = p.binding.mip_levels;
                    copy_regions(mips, p.image_copies.data());
                    detail::memory_barrier(vk, command_buffer, VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                           VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
                    copy_regions(mips, p.image_copies.data() + mips);
                } else {
                    copy_regions(std::uint32_t(p.image_copies.size()), p.image_copies.data());
                }
            } else if (p.binding.image) {
                const VkImageLayout dst_layout = detail::copy_layout(
                    vk, command_buffer, p.binding, dst_id, detail::planes_for_layout(p.binding, p.range, separate),
                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_ACCESS_TRANSFER_WRITE_BIT, tracker);
                vk.copy_buffer_to_image(command_buffer, p.source.buffer, p.binding.image, dst_layout,
                                        std::uint32_t(p.buffer_image_copies.size()), p.buffer_image_copies.data());
            } else {
                const VkImageLayout src_layout = detail::copy_layout(
                    vk, command_buffer, p.source, src_id, detail::planes_for_layout(p.source, p.source_range, separate),
                    VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_ACCESS_TRANSFER_READ_BIT, tracker);
                vk.copy_image_to_buffer(command_buffer, p.source.image, src_layout, p.binding.buffer,
                                        std::uint32_t(p.buffer_image_copies.size()), p.buffer_image_copies.data());
            }
        } else if (std::holds_alternative<ResolveSubresource>(command)) {
            // resolve_source and resolve_dest are the transfer layouts, as copy_source and copy_dest are.
            const auto [dst_id, src_id] = copy_ids(command);
            const VkImageLayout src_layout = detail::copy_layout(
                vk, command_buffer, p.source, src_id, detail::planes_for_layout(p.source, p.source_range, separate),
                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_ACCESS_TRANSFER_READ_BIT, tracker);
            const VkImageLayout dst_layout = detail::copy_layout(
                vk, command_buffer, p.binding, dst_id, detail::planes_for_layout(p.binding, p.range, separate),
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_ACCESS_TRANSFER_WRITE_BIT, tracker);
            vk.resolve_image(command_buffer, p.source.image, src_layout, p.binding.image, dst_layout, 1, &p.resolve);
        } else if (std::holds_alternative<DiscardResource>(command)) {
            // oldLayout = UNDEFINED tells the driver the contents may go; the image stays in the layout the tracker
            // has for it, so later commands see no change. Without a tracker the layout is unknown, and a part
            // never used yet is still to be zero-filled: both are left alone. Without separate depth/stencil
            // layouts, discarding one plane would also discard the other, so that is left alone too.
            constexpr VkAccessFlags any_memory = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
            for (const auto& range : p.discards) {
                if (!tracker || detail::planes_for_layout(p.binding, range, separate).aspectMask != range.aspectMask)
                    continue;
                std::vector<LayoutPiece> pieces;
                if (!tracker->pieces(id, range, pieces)) continue;
                for (const auto& piece : pieces)
                    if (piece.layout != VK_IMAGE_LAYOUT_UNDEFINED)
                        detail::image_barrier(vk, command_buffer, p.binding, piece.range, VK_IMAGE_LAYOUT_UNDEFINED,
                                              piece.layout, VK_ACCESS_MEMORY_WRITE_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                              any_memory, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
            }
        } else if (const auto* write = std::get_if<WriteImmediate>(&command)) {
            // vkCmdUpdateBuffer runs in the transfer stage. MARKER_OUT waits for every earlier command to finish
            // and sees its writes. MARKER_IN only needs them started, which Vulkan cannot express (TOP_OF_PIPE in a
            // first scope is no stage at all), so it waits for them to finish too, without their writes. DEFAULT is
            // ordered like any copy, by the application's barriers.
            if (write->mode == write_immediate_mode::marker_out)
                detail::memory_barrier(vk, command_buffer, VK_ACCESS_MEMORY_WRITE_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                       VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
            else if (write->mode == write_immediate_mode::marker_in)
                detail::memory_barrier(vk, command_buffer, 0, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0,
                                       VK_PIPELINE_STAGE_TRANSFER_BIT);
            vk.update_buffer(command_buffer, p.write_buffer, p.write_offset, 4, &write->value);
        } else if (std::holds_alternative<AtomicCopyBuffer>(command)) {
            // Every earlier write, to the dependent resources among others, finishes and is visible before the copy.
            // Vulkan promises no atomicity for a transfer; that a naturally aligned 4- or 8-byte copy is one write
            // is an assumption about the hardware (docs/d3d12_vulkan.md, section 4.2).
            detail::memory_barrier(vk, command_buffer, VK_ACCESS_MEMORY_WRITE_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                   VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
                                   VK_PIPELINE_STAGE_TRANSFER_BIT);
            vk.copy_buffer(command_buffer, p.source.buffer, p.binding.buffer, 1, p.buffer_copies.data());
        } else if (const auto* begin = std::get_if<BeginQuery>(&command)) {
            // D3D12 queries need no reset; a Vulkan query is reset before every use. Nothing here runs inside a
            // render pass, where a reset is not allowed: a draw path that opens one around queries must reset
            // them before it.
            QueryPool& heap = *p.queries;
            vk.reset_query_pool(command_buffer, heap.pool, begin->index, 1);
            heap.ended[begin->index] = false;
            const bool precise = begin->type == d3d12_query::occlusion && heap.precise;
            vk.begin_query(command_buffer, heap.pool, begin->index, precise ? VK_QUERY_CONTROL_PRECISE_BIT : 0);
        } else if (const auto* end = std::get_if<EndQuery>(&command)) {
            QueryPool& heap = *p.queries;
            if (end->type == d3d12_query::timestamp) {
                vk.reset_query_pool(command_buffer, heap.pool, end->index, 1);
                vk.write_timestamp(command_buffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, heap.pool, end->index);
            } else {
                vk.end_query(command_buffer, heap.pool, end->index);
            }
            heap.ended[end->index] = true;
        } else if (const auto* resolve = std::get_if<ResolveQueries>(&command)) {
            // Runs of ended queries are copied as UINT64s, waiting for their results; D3D12 leaves a query that
            // was never ended undefined, and it is written as zeros here rather than waited on forever.
            const QueryPool& heap = *p.queries;
            const std::uint32_t size = query_result_size(resolve->type);
            std::uint32_t index = resolve->first;
            const std::uint32_t last = resolve->first + resolve->count;
            while (index < last) {
                const bool ended = heap.ended[index];
                std::uint32_t run = 1;
                while (index + run < last && heap.ended[index + run] == ended) ++run;
                const VkDeviceSize offset = resolve->offset + VkDeviceSize(index - resolve->first) * size;
                if (ended)
                    vk.copy_query_pool_results(command_buffer, heap.pool, index, run, p.binding.buffer, offset, size,
                                               VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
                else
                    vk.fill_buffer(command_buffer, p.binding.buffer, offset, VkDeviceSize(run) * size, 0);
                index += run;
            }
        }
    }
    // Predication does not carry over to the next command list.
    if (conditional_rendering) vk.end_conditional_rendering(command_buffer);
}

// prepare + emit for one list. A failed replay leaves the command buffer and the tracker untouched.
template<typename Resolve>
ReplayResult replay(const CommandList& list, VkCommandBuffer command_buffer, Resolve&& resolve,
                    const ReplayFunctions& vk, LayoutTracker* tracker = nullptr) {
    if (tracker && tracker->separate_depth_stencil_layouts() != vk.separate_depth_stencil_layouts)
        return ReplayResult::unsupported;
    PreparedList prepared;
    if (const auto result = prepare(list, resolve, vk, prepared); result != ReplayResult::success) return result;
    emit(prepared, command_buffer, vk, tracker);
    return ReplayResult::success;
}

} // namespace recity
