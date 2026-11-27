// Headless Vulkan harness: draws one triangle (3 vertices, no vertex buffers) with a given VS + FS into a
// W x H RGBA32F image and writes the raw floats to a file. Used to check translated shaders on the real GPU.
//
//   vk_harness <vs.spv> <fs.spv> <out.raw> [width height [clip-test-s0 clip-test-s1]]
//   vk_harness <ms.spv> <fs.spv> <out.raw> --mesh-group-readback X Y Z
// The latter captures one full-screen triangle per mesh workgroup, in Vulkan rasterization order.
//
// Push constants follow the contract in docs/design.md: s0..s31, then the heap base lo/hi. They are zero unless
//   VKH_SGPRS=w0,w1,...          sets s0, s1, ... (decimal or 0x hex)
//   VKH_HEAP=addr=word,...       allocates a zeroed 64 KiB heap (Xbox VA 0 = its device address), writes each word at
//                                its address and passes the heap base, so a V# in VKH_SGPRS can point into it.
//
#define VK_NO_PROTOTYPES
#include "vulkan_loader.h"

#include <cctype>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <utility>
#include <vector>

static PFN_vkGetInstanceProcAddr gipa;
#define LOADI(name) auto name = reinterpret_cast<PFN_##name>(gipa(inst, #name))
#define CHECK(x)                                                                 \
    do {                                                                         \
        VkResult r_ = (x);                                                       \
        if (r_ != VK_SUCCESS) {                                                  \
            std::fprintf(stderr, "FAILED %s -> %d (line %d)\n", #x, r_, __LINE__); \
            return 1;                                                            \
        }                                                                        \
    } while (0)

static std::vector<uint32_t> read_spv(const char* path) {
    std::ifstream f(path, std::ios::binary);
    std::vector<char> b((std::istreambuf_iterator<char>(f)), {});
    std::vector<uint32_t> w(b.size() / 4);
    std::memcpy(w.data(), b.data(), w.size() * 4);
    return w;
}

static bool parse_u32(const char*& s, uint32_t& v) {
    const bool hex = s[0] == '0' && (s[1] == 'x' || s[1] == 'X');
    const char* digits = hex ? s + 2 : s;
    if (!(hex ? std::isxdigit(uint8_t(*digits)) : std::isdigit(uint8_t(*digits)))) return false;   // no sign or space
    char* end;
    unsigned long long x = std::strtoull(digits, &end, hex ? 16 : 10);
    if (x > 0xFFFFFFFFull) return false;
    v = uint32_t(x); s = end;
    return true;
}

// A list separator must be followed by another item: "1,2," is rejected.
static bool next_item(const char*& s) {
    if (!*s) return true;
    return *s++ == ',' && *s;
}

static bool parse_sgprs(const char* s, uint32_t* sgprs) {
    for (uint32_t i = 0; *s; i++) {
        if (i == 32 || !parse_u32(s, sgprs[i]) || !next_item(s)) return false;
    }
    return true;
}

static bool parse_heap(const char* s, std::vector<std::pair<uint32_t, uint32_t>>& words, uint32_t heap_bytes) {
    while (*s) {
        uint32_t addr, word;
        if (!parse_u32(s, addr) || *s++ != '=' || !parse_u32(s, word) || !next_item(s)) return false;
        if (addr % 4 || addr > heap_bytes - 4) return false;
        words.emplace_back(addr, word);
    }
    return true;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: vk_harness vs.spv fs.spv out.raw [w h [clip-test-s0 clip-test-s1]]\n"
                             "       vk_harness ms.spv fs.spv out.raw --mesh-group-readback X Y Z\n");
        return 2;
    }
    const bool group_readback = argc > 4 && !std::strcmp(argv[4], "--mesh-group-readback");
    const bool clip_shifts = !group_readback && argc == 8;
    uint32_t groups[3] = {1, 1, 1};
    uint32_t W = 64, H = 64;
    if (group_readback) {
        if (argc != 8) {
            std::fprintf(stderr, "--mesh-group-readback requires X Y Z\n");
            return 2;
        }
        W = H = 1;
        for (unsigned i = 0; i < 3; ++i) {
            char* end;
            long n = std::strtol(argv[5 + i], &end, 10);
            if (*end || n < 1 || n > 256 / W) {
                std::fprintf(stderr, "mesh grid must contain 1..256 groups (8-bit stencil)\n");
                return 2;
            }
            groups[i] = uint32_t(n);
            W *= groups[i];
        }
    } else if (argc > 5) {
        W = atoi(argv[4]); H = atoi(argv[5]);
    }
    uint32_t pc[36] = {};
    const char* sgprs_env = std::getenv("VKH_SGPRS");
    if (sgprs_env && !parse_sgprs(sgprs_env, pc)) {
        std::fprintf(stderr, "VKH_SGPRS: expected up to 32 comma-separated words, got \"%s\"\n", sgprs_env);
        return 2;
    }
    const char* heap_dump = std::getenv("VKH_HEAP_DUMP");
    void* heap_mapping = nullptr;
    const char* replay_stage = std::getenv("VKH_BALLOT_REPLAY_STAGE");
    uint32_t replay_passes = 2;
    uint32_t heap_bytes = 64 * 1024;
    if (replay_stage) {
        if (const char* passes = std::getenv("VKH_BALLOT_PASSES")) {
            if (!parse_u32(passes, replay_passes) || *passes || replay_passes < 2) return 2;
        }
        // Preserve the original lane-record space as longer programs add ballot tables.
        if (replay_passes > 7) {
            const uint64_t extra = uint64_t(replay_passes - 7) * 512 * sizeof(uint32_t);
            if (extra > UINT32_MAX - heap_bytes) return 2;
            heap_bytes += uint32_t(extra);
        }
        pc[34] = replay_passes;
    }
    const char* heap_env = std::getenv("VKH_HEAP");
    if (heap_env && !*heap_env) heap_env = nullptr;
    if (heap_dump && !heap_env) return 2;
    std::vector<std::pair<uint32_t, uint32_t>> heap_words;
    if (heap_env && !parse_heap(heap_env, heap_words, heap_bytes)) {
        std::fprintf(stderr, "VKH_HEAP: expected addr=word pairs, 4-byte aligned below 0x%x, got \"%s\"\n",
                     heap_bytes, heap_env);
        return 2;
    }
    gipa = load_vulkan();
    if (!gipa) return 1;
    VkInstance inst = VK_NULL_HANDLE;
    LOADI(vkCreateInstance);
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    CHECK(vkCreateInstance(&ici, nullptr, &inst));
    LOADI(vkEnumeratePhysicalDevices); LOADI(vkGetPhysicalDeviceProperties);
    LOADI(vkGetPhysicalDeviceProperties2); LOADI(vkGetPhysicalDeviceFeatures2);
    LOADI(vkGetPhysicalDeviceQueueFamilyProperties); LOADI(vkGetPhysicalDeviceMemoryProperties);
    LOADI(vkCreateDevice); LOADI(vkGetDeviceProcAddr);
    uint32_t n = 0;
    vkEnumeratePhysicalDevices(inst, &n, nullptr);
    std::vector<VkPhysicalDevice> pds(n);
    vkEnumeratePhysicalDevices(inst, &n, pds.data());
    VkPhysicalDevice pd = VK_NULL_HANDLE;
    for (auto d : pds) {   // prefer the discrete GPU
        VkPhysicalDeviceProperties p; vkGetPhysicalDeviceProperties(d, &p);
        if (p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU || !pd) pd = d;
    }
    VkPhysicalDeviceProperties props; vkGetPhysicalDeviceProperties(pd, &props);
    VkPhysicalDeviceSubgroupSizeControlProperties sgProps{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES};
    VkPhysicalDeviceProperties2 props2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    props2.pNext = &sgProps;
    vkGetPhysicalDeviceProperties2(pd, &props2);
    std::fprintf(stderr, "device: %s (subgroup %u..%u)\n", props.deviceName, sgProps.minSubgroupSize, sgProps.maxSubgroupSize);
    uint32_t qn = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &qn, nullptr);
    std::vector<VkQueueFamilyProperties> qf(qn);
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &qn, qf.data());
    uint32_t qi = 0;
    while (qi < qn && !(qf[qi].queueFlags & VK_QUEUE_GRAPHICS_BIT)) qi++;
    float prio = 1;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = qi; qci.queueCount = 1; qci.pQueuePriorities = &prio;
    // Optional features some translated shaders need (enabled only when the device has them):
    // raw barycentric reads (BaryCoordKHR), POPS ordering (pixel interlock), non-constant texel offsets (maintenance8).
    auto vkEnumerateDeviceExtensionProperties = reinterpret_cast<PFN_vkEnumerateDeviceExtensionProperties>(
        gipa(inst, "vkEnumerateDeviceExtensionProperties"));
    uint32_t en = 0;
    vkEnumerateDeviceExtensionProperties(pd, nullptr, &en, nullptr);
    std::vector<VkExtensionProperties> exts(en);
    vkEnumerateDeviceExtensionProperties(pd, nullptr, &en, exts.data());
    auto has = [&](const char* e) {
        for (auto& x : exts) if (!std::strcmp(x.extensionName, e)) return true;
        return false;
    };
    // A translated pixel shader that reads BaryCoordKHR aborts the driver when the device lacks barycentrics
    // (llvmpipe and other software devices), so the suite reports it as unsupported instead. 77 is the existing
    // "this device cannot run the test" code (runtime/test/shaders/check.py, scripts/run_gpu_tests.py).
    const std::vector<uint32_t> fs_code = read_spv(argv[2]);
    bool needs_barycentric = false, needs_float64 = false;
    for (std::size_t i = 5; i + 1 < fs_code.size();) {   // instructions follow the 5-word SPIR-V header
        const uint32_t words = fs_code[i] >> 16, op = fs_code[i] & 0xffffu;
        if (!words) break;
        if (op == 17 && fs_code[i + 1] == 5284)          // OpCapability FragmentBarycentricKHR
            needs_barycentric = true;
        if (op == 17 && fs_code[i + 1] == 10)            // OpCapability Float64 (v_add_f64 and friends)
            needs_float64 = true;
        i += words;
    }
    if (needs_barycentric && !has(VK_KHR_FRAGMENT_SHADER_BARYCENTRIC_EXTENSION_NAME)) {
        std::fprintf(stderr, "unsupported: %s reads BaryCoordKHR and the device has no %s\n",
                     argv[2], VK_KHR_FRAGMENT_SHADER_BARYCENTRIC_EXTENSION_NAME);
        return 77;
    }
    std::vector<const char*> enable;
    void* chain = nullptr;
    VkPhysicalDeviceFragmentShaderBarycentricFeaturesKHR bary{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADER_BARYCENTRIC_FEATURES_KHR};
    VkPhysicalDeviceFragmentShaderInterlockFeaturesEXT lock{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADER_INTERLOCK_FEATURES_EXT};
    if (has(VK_KHR_FRAGMENT_SHADER_BARYCENTRIC_EXTENSION_NAME)) {
        enable.push_back(VK_KHR_FRAGMENT_SHADER_BARYCENTRIC_EXTENSION_NAME);
        bary.fragmentShaderBarycentric = VK_TRUE; bary.pNext = chain; chain = &bary;
    }
    if (has(VK_EXT_FRAGMENT_SHADER_INTERLOCK_EXTENSION_NAME)) {
        enable.push_back(VK_EXT_FRAGMENT_SHADER_INTERLOCK_EXTENSION_NAME);
        lock.fragmentShaderPixelInterlock = VK_TRUE; lock.pNext = chain; chain = &lock;
    }
    // VKH_MESH=1: the first shader is a mesh shader (VK_EXT_mesh_shader), drawn with vkCmdDrawMeshTasksEXT(1,1,1).
    const bool mesh = group_readback || std::getenv("VKH_MESH") != nullptr;
    VkPhysicalDeviceMeshShaderFeaturesEXT meshf{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT};
    if (mesh) {
        // Query the feature, not just the extension. 77: the device cannot run this test (as vk_compute_test and
        // vk_store_test), so the runners report SKIP.
        VkPhysicalDeviceFeatures2 mesh_query{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
        mesh_query.pNext = &meshf;
        vkGetPhysicalDeviceFeatures2(pd, &mesh_query);
        if (!has(VK_EXT_MESH_SHADER_EXTENSION_NAME) || !meshf.meshShader) {
            std::fprintf(stderr, "unsupported: no VK_EXT_mesh_shader\n");
            return 77;
        }
        enable.push_back(VK_EXT_MESH_SHADER_EXTENSION_NAME);
        meshf = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT};
        meshf.meshShader = VK_TRUE; meshf.pNext = chain; chain = &meshf;
    }
#ifdef VK_KHR_MAINTENANCE_8_EXTENSION_NAME
    VkPhysicalDeviceMaintenance8FeaturesKHR m8{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_8_FEATURES_KHR};
    if (has(VK_KHR_MAINTENANCE_8_EXTENSION_NAME)) {
        enable.push_back(VK_KHR_MAINTENANCE_8_EXTENSION_NAME);
        m8.maintenance8 = VK_TRUE; m8.pNext = chain; chain = &m8;
    }
#endif
    VkPhysicalDeviceVulkan12Features f12q{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    VkPhysicalDeviceVulkan13Features f13q{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    f13q.pNext = &f12q;
    VkPhysicalDeviceFeatures2 f2q{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    f2q.pNext = &f13q;
    vkGetPhysicalDeviceFeatures2(pd, &f2q);
    VkPhysicalDeviceVulkan13Features f13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    if (f13q.subgroupSizeControl) {
        f13.subgroupSizeControl = VK_TRUE;
        f13.pNext = chain;
        chain = &f13;
    }
    VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    if (heap_env) {
        // 77 is the "this device cannot run the test" code that check.py and run_gpu_tests.py report as SKIP.
        if (!f12q.bufferDeviceAddress) { std::fprintf(stderr, "unsupported: VKH_HEAP needs bufferDeviceAddress\n"); return 77; }
        f12.bufferDeviceAddress = VK_TRUE;
        f12.pNext = chain;
        chain = &f12;
    }
    for (auto e : enable) std::fprintf(stderr, "enabled %s\n", e);
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.queueCreateInfoCount = 1; dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = uint32_t(enable.size()); dci.ppEnabledExtensionNames = enable.data();
    dci.pNext = chain;
    // VKH_GS=<geometry.spv>: insert a geometry stage between the vertex and fragment shaders.
    const char* gs_path = std::getenv("VKH_GS");
    if (gs_path && !f2q.features.geometryShader) {   // MoltenVK: Metal has no geometry stage
        std::fprintf(stderr, "unsupported: no geometryShader\n");
        return 77;
    }
    uint32_t replay_vertices = 3;
    VkPipelineStageFlags replay_access = 0;
    VkPhysicalDeviceFeatures feats{};
    if (replay_stage) {
        if (!heap_env || mesh || gs_path || clip_shifts) return 2;
        if (!std::strcmp(replay_stage, "vs")) {
            if (!f2q.features.vertexPipelineStoresAndAtomics) return 77;
            // Enough unique vertices for at least one full graphics subgroup in each capture draw.
            replay_vertices = 192;
            feats.vertexPipelineStoresAndAtomics = VK_TRUE;
            replay_access = VK_PIPELINE_STAGE_VERTEX_SHADER_BIT;
        } else if (!std::strcmp(replay_stage, "ps")) {
            if (!f2q.features.fragmentStoresAndAtomics) return 77;
            feats.fragmentStoresAndAtomics = VK_TRUE;
            replay_access = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        } else return 2;
    }
    if (clip_shifts && !f2q.features.shaderClipDistance) {
        std::fprintf(stderr, "unsupported: shaderClipDistance\n");
        return 77;
    }
    feats.geometryShader = gs_path ? VK_TRUE : VK_FALSE;
    feats.shaderClipDistance = f2q.features.shaderClipDistance;
    feats.shaderCullDistance = f2q.features.shaderCullDistance;
    if (needs_float64) {
        if (!f2q.features.shaderFloat64) {
            std::fprintf(stderr, "unsupported: %s uses Float64 and the device has no shaderFloat64\n", argv[2]);
            return 77;
        }
        feats.shaderFloat64 = VK_TRUE;
    }
    if (heap_env) {
        if (!f2q.features.shaderInt64) { std::fprintf(stderr, "unsupported: VKH_HEAP needs shaderInt64\n"); return 77; }
        feats.shaderInt64 = VK_TRUE;   // heap addresses are 64-bit
    }
    dci.pEnabledFeatures = &feats;
    VkDevice dev;
    CHECK(vkCreateDevice(pd, &dci, nullptr, &dev));
#define LOADD(name) auto name = reinterpret_cast<PFN_##name>(vkGetDeviceProcAddr(dev, #name))
    LOADD(vkGetDeviceQueue); LOADD(vkCreateImage); LOADD(vkGetImageMemoryRequirements); LOADD(vkAllocateMemory);
    LOADD(vkBindImageMemory); LOADD(vkCreateImageView); LOADD(vkCreateRenderPass); LOADD(vkCreateFramebuffer);
    LOADD(vkCreateShaderModule); LOADD(vkCreatePipelineLayout); LOADD(vkCreateGraphicsPipelines);
    LOADD(vkCreateCommandPool); LOADD(vkAllocateCommandBuffers); LOADD(vkBeginCommandBuffer);
    LOADD(vkCmdBeginRenderPass); LOADD(vkCmdBindPipeline); LOADD(vkCmdDraw); LOADD(vkCmdEndRenderPass);
    LOADD(vkCmdCopyImageToBuffer); LOADD(vkEndCommandBuffer); LOADD(vkQueueSubmit); LOADD(vkQueueWaitIdle);
    LOADD(vkCreateBuffer); LOADD(vkGetBufferMemoryRequirements); LOADD(vkBindBufferMemory); LOADD(vkMapMemory);
    VkQueue q; vkGetDeviceQueue(dev, qi, 0, &q);
    VkPhysicalDeviceMemoryProperties mp; vkGetPhysicalDeviceMemoryProperties(pd, &mp);
    auto memtype = [&](uint32_t bits, VkMemoryPropertyFlags want) {
        for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
            if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want) == want) return i;
        return 0u;
    };
    auto attachment = [&](VkFormat format, VkImageUsageFlags usage, VkImageAspectFlags aspect,
                          VkImage& image, VkImageView& view) -> int {
        VkImageCreateInfo ii{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        ii.imageType = VK_IMAGE_TYPE_2D; ii.format = format; ii.extent = {W, H, 1}; ii.mipLevels = 1; ii.arrayLayers = 1;
        ii.samples = VK_SAMPLE_COUNT_1_BIT; ii.tiling = VK_IMAGE_TILING_OPTIMAL; ii.usage = usage;
        CHECK(vkCreateImage(dev, &ii, nullptr, &image));
        VkMemoryRequirements mr; vkGetImageMemoryRequirements(dev, image, &mr);
        VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        mai.allocationSize = mr.size; mai.memoryTypeIndex = memtype(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        VkDeviceMemory mem; CHECK(vkAllocateMemory(dev, &mai, nullptr, &mem));
        CHECK(vkBindImageMemory(dev, image, mem, 0));
        VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vi.image = image; vi.viewType = VK_IMAGE_VIEW_TYPE_2D; vi.format = format;
        vi.subresourceRange = {aspect, 0, 1, 0, 1};
        CHECK(vkCreateImageView(dev, &vi, nullptr, &view));
        return 0;
    };
    const VkFormat fmt = VK_FORMAT_R32G32B32A32_SFLOAT;
    VkImage img;
    VkImageView views[2]{};
    if (attachment(fmt, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                   VK_IMAGE_ASPECT_COLOR_BIT, img, views[0])) return 1;
    VkAttachmentDescription attachments[2]{};
    auto& ad = attachments[0];
    ad.format = fmt; ad.samples = VK_SAMPLE_COUNT_1_BIT; ad.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    ad.storeOp = VK_ATTACHMENT_STORE_OP_STORE; ad.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    ad.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE; ad.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    ad.finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    VkAttachmentReference ar{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkSubpassDescription sd{}; sd.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    sd.colorAttachmentCount = 1; sd.pColorAttachments = &ar;
    VkAttachmentReference stencil_ref{1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
    if (group_readback) {
        LOADI(vkGetPhysicalDeviceFormatProperties);
        auto& ds = attachments[1];
        for (auto format : {VK_FORMAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT, VK_FORMAT_D32_SFLOAT_S8_UINT}) {
            VkFormatProperties fp;
            vkGetPhysicalDeviceFormatProperties(pd, format, &fp);
            if (fp.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) {
                ds.format = format;
                break;
            }
        }
        if (!ds.format) { std::fprintf(stderr, "no stencil attachment format\n"); return 1; }
        VkImage stencil;
        VkImageAspectFlags aspects = VK_IMAGE_ASPECT_STENCIL_BIT;
        if (ds.format != VK_FORMAT_S8_UINT) aspects |= VK_IMAGE_ASPECT_DEPTH_BIT;
        if (attachment(ds.format, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, aspects, stencil, views[1])) return 1;
        ds.samples = VK_SAMPLE_COUNT_1_BIT;
        ds.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE; ds.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        ds.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_CLEAR; ds.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        ds.finalLayout = stencil_ref.layout;
        sd.pDepthStencilAttachment = &stencil_ref;
    }
    VkRenderPassCreateInfo rpi{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
    rpi.attachmentCount = group_readback ? 2 : 1; rpi.pAttachments = attachments; rpi.subpassCount = 1; rpi.pSubpasses = &sd;
    VkSubpassDependency copy_dep{0, VK_SUBPASS_EXTERNAL, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                                 VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT};
    rpi.dependencyCount = 1; rpi.pDependencies = &copy_dep;
    VkRenderPass rp; CHECK(vkCreateRenderPass(dev, &rpi, nullptr, &rp));
    VkFramebufferCreateInfo fbi{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
    fbi.renderPass = rp; fbi.attachmentCount = rpi.attachmentCount; fbi.pAttachments = views; fbi.width = W; fbi.height = H; fbi.layers = 1;
    VkFramebuffer fb; CHECK(vkCreateFramebuffer(dev, &fbi, nullptr, &fb));
    auto module = [&](const char* p, VkShaderModule* m) {
        auto code = read_spv(p);
        VkShaderModuleCreateInfo smi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        smi.codeSize = code.size() * 4; smi.pCode = code.data();
        return vkCreateShaderModule(dev, &smi, nullptr, m);
    };
    VkShaderModule vs, fs;
    CHECK(module(argv[1], &vs)); CHECK(module(argv[2], &fs));
    VkPipelineShaderStageCreateInfo stages[3]{{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO},
                                              {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO},
                                              {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO}};
    stages[0].stage = mesh ? VK_SHADER_STAGE_MESH_BIT_EXT : VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vs; stages[0].pName = "main";
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT; stages[1].module = fs; stages[1].pName = "main";
    uint32_t nstages = 2;
    if (gs_path) {
        VkShaderModule gsm;
        CHECK(module(gs_path, &gsm));
        stages[2].stage = VK_SHADER_STAGE_GEOMETRY_BIT; stages[2].module = gsm; stages[2].pName = "main";
        nstages = 3;
    }
    const char* sg_env = std::getenv("VKH_SUBGROUP_SIZE");
    const uint32_t req_sg = sg_env ? uint32_t(atoi(sg_env)) : 0;
    VkPipelineShaderStageRequiredSubgroupSizeCreateInfo reqSg{
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO};
    reqSg.requiredSubgroupSize = req_sg;
    if (replay_stage && pc[1] == 0) {
        if (req_sg || !f13q.subgroupSizeControl) return 77;
        const unsigned stage_index = replay_access == VK_PIPELINE_STAGE_VERTEX_SHADER_BIT ? 0 : 1;
        stages[stage_index].flags |= VK_PIPELINE_SHADER_STAGE_CREATE_ALLOW_VARYING_SUBGROUP_SIZE_BIT;
    }
    if (req_sg > 0 && f13q.subgroupSizeControl &&
        req_sg >= sgProps.minSubgroupSize && req_sg <= sgProps.maxSubgroupSize) {
        if (mesh && (sgProps.requiredSubgroupSizeStages & VK_SHADER_STAGE_MESH_BIT_EXT))
            stages[0].pNext = &reqSg;
        else if (gs_path && (sgProps.requiredSubgroupSizeStages & VK_SHADER_STAGE_GEOMETRY_BIT))
            stages[2].pNext = &reqSg;
        else if (!mesh && !gs_path && (sgProps.requiredSubgroupSizeStages & VK_SHADER_STAGE_FRAGMENT_BIT))
            stages[1].pNext = &reqSg;
    }
    VkPipelineVertexInputStateCreateInfo vin{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkViewport vp{0, 0, float(W), float(H), 0, 1}; VkRect2D sc{{0, 0}, {W, H}};
    VkPipelineViewportStateCreateInfo vps{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    vps.viewportCount = 1; vps.pViewports = &vp; vps.scissorCount = 1; vps.pScissors = &sc;
    VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    rs.cullMode = VK_CULL_MODE_NONE; rs.lineWidth = 1;
    VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    VkPipelineColorBlendAttachmentState cba{}; cba.colorWriteMask = 0xF;
    VkPipelineColorBlendStateCreateInfo cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    cb.attachmentCount = 1; cb.pAttachments = &cba;
    // Translated shaders declare a 136-byte push-constant block (user SGPRs + heap base); see VKH_SGPRS / VKH_HEAP.
    const VkShaderStageFlags pcs = (mesh ? VK_SHADER_STAGE_MESH_BIT_EXT : VK_SHADER_STAGE_VERTEX_BIT) |
                                   VK_SHADER_STAGE_FRAGMENT_BIT | (gs_path ? VK_SHADER_STAGE_GEOMETRY_BIT : 0);
    VkPushConstantRange pcr{pcs, 0, 144};
    VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pli.pushConstantRangeCount = 1; pli.pPushConstantRanges = &pcr;
    VkPipelineLayout pl; CHECK(vkCreatePipelineLayout(dev, &pli, nullptr, &pl));
    VkGraphicsPipelineCreateInfo gpi{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    gpi.stageCount = nstages; gpi.pStages = stages;
    gpi.pVertexInputState = mesh ? nullptr : &vin; gpi.pInputAssemblyState = mesh ? nullptr : &ia;
    gpi.pViewportState = &vps; gpi.pRasterizationState = &rs; gpi.pMultisampleState = &ms; gpi.pColorBlendState = &cb;
    VkPipelineDepthStencilStateCreateInfo ds_state{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    if (group_readback) {
        // Pixel i starts at stencil i. Every triangle decrements it, but only the i-th sees zero and writes color.
        ds_state.stencilTestEnable = VK_TRUE;
        auto& face = ds_state.front;
        face.compareOp = VK_COMPARE_OP_EQUAL;
        face.compareMask = face.writeMask = 0xff;
        face.failOp = face.passOp = VK_STENCIL_OP_DECREMENT_AND_WRAP;
        ds_state.back = face;
        gpi.pDepthStencilState = &ds_state;
    }
    gpi.layout = pl; gpi.renderPass = rp;
    VkPipeline pipe; CHECK(vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &gpi, nullptr, &pipe));
    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.size = VkDeviceSize(W) * H * 16; bi.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    VkBuffer buf; CHECK(vkCreateBuffer(dev, &bi, nullptr, &buf));
    VkMemoryRequirements br; vkGetBufferMemoryRequirements(dev, buf, &br);
    VkMemoryAllocateInfo bai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    bai.allocationSize = br.size;
    bai.memoryTypeIndex = memtype(br.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    VkDeviceMemory bmem; CHECK(vkAllocateMemory(dev, &bai, nullptr, &bmem));
    CHECK(vkBindBufferMemory(dev, buf, bmem, 0));
    if (heap_env) {
        LOADD(vkGetBufferDeviceAddress);
        VkBufferCreateInfo hbi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        hbi.size = heap_bytes;
        hbi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
        VkBuffer heap; CHECK(vkCreateBuffer(dev, &hbi, nullptr, &heap));
        VkMemoryRequirements hr; vkGetBufferMemoryRequirements(dev, heap, &hr);
        VkMemoryAllocateFlagsInfo haf{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
        haf.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
        VkMemoryAllocateInfo hai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        hai.pNext = &haf; hai.allocationSize = hr.size;
        hai.memoryTypeIndex = memtype(hr.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        VkDeviceMemory hmem; CHECK(vkAllocateMemory(dev, &hai, nullptr, &hmem));
        CHECK(vkBindBufferMemory(dev, heap, hmem, 0));
        void* hp; CHECK(vkMapMemory(dev, hmem, 0, heap_bytes, 0, &hp));
        heap_mapping = hp;
        std::memset(hp, 0, heap_bytes);
        for (auto [addr, word] : heap_words) std::memcpy(static_cast<char*>(hp) + addr, &word, 4);
        VkBufferDeviceAddressInfo hda{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
        hda.buffer = heap;
        const uint64_t base = vkGetBufferDeviceAddress(dev, &hda);
        pc[32] = uint32_t(base); pc[33] = uint32_t(base >> 32);
    }
    VkCommandPoolCreateInfo cpi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO}; cpi.queueFamilyIndex = qi;
    VkCommandPool pool; CHECK(vkCreateCommandPool(dev, &cpi, nullptr, &pool));
    VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cai.commandPool = pool; cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cai.commandBufferCount = 1;
    VkCommandBuffer cmd; CHECK(vkAllocateCommandBuffers(dev, &cai, &cmd));
    VkCommandBufferBeginInfo cbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    CHECK(vkBeginCommandBuffer(cmd, &cbi));
    VkClearValue clears[2]{};
    for (float& c : clears[0].color.float32) c = -1;   // marks untouched pixels
    VkRenderPassBeginInfo rbi{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    rbi.renderPass = rp; rbi.framebuffer = fb; rbi.renderArea = sc; rbi.clearValueCount = rpi.attachmentCount; rbi.pClearValues = clears;
    if (replay_stage) {
        LOADD(vkCmdPushConstants);
        LOADD(vkCmdPipelineBarrier);
        // Draw one physical half per pass, with a device-side dependency before reading the shared heap.
        for (uint32_t phase = 0; phase < replay_passes; ++phase) {
            pc[0] = phase;
            vkCmdBeginRenderPass(cmd, &rbi, VK_SUBPASS_CONTENTS_INLINE);
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
            vkCmdPushConstants(cmd, pl, pcs, 0, sizeof(pc), pc);
            vkCmdDraw(cmd, replay_vertices, 1, 0, 0);
            vkCmdEndRenderPass(cmd);
            VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
            barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
            const auto stages = replay_access | VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
            vkCmdPipelineBarrier(cmd, stages, stages, 0, 1, &barrier, 0, nullptr, 0, nullptr);
        }
        pc[0] = replay_passes;
    }
    vkCmdBeginRenderPass(cmd, &rbi, VK_SUBPASS_CONTENTS_INLINE);
    if (group_readback) {
        LOADD(vkCmdClearAttachments);
        VkClearAttachment seed{VK_IMAGE_ASPECT_STENCIL_BIT};
        for (uint32_t i = 1; i < W; ++i) {
            seed.clearValue.depthStencil.stencil = i;
            VkClearRect pixel{{{int32_t(i), 0}, {1, 1}}, 0, 1};
            vkCmdClearAttachments(cmd, 1, &seed, 1, &pixel);
        }
        std::fprintf(stderr, "mesh workgroups: %u x %u x %u\n", groups[0], groups[1], groups[2]);
    }
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
    if (clip_shifts) {
        const float shifts[] = {float(std::atof(argv[6])), float(std::atof(argv[7]))};
        std::memcpy(pc, shifts, sizeof(shifts));
    }
    reinterpret_cast<PFN_vkCmdPushConstants>(vkGetDeviceProcAddr(dev, "vkCmdPushConstants"))(
        cmd, pl, pcs, 0, sizeof(pc), pc);
    if (mesh)
        reinterpret_cast<PFN_vkCmdDrawMeshTasksEXT>(vkGetDeviceProcAddr(dev, "vkCmdDrawMeshTasksEXT"))(cmd, groups[0], groups[1], groups[2]);
    else
        vkCmdDraw(cmd, replay_vertices, 1, 0, 0);
    vkCmdEndRenderPass(cmd);
    VkBufferImageCopy reg{}; reg.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}; reg.imageExtent = {W, H, 1};
    vkCmdCopyImageToBuffer(cmd, img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, buf, 1, &reg);
    if (heap_dump) {
        LOADD(vkCmdPipelineBarrier);
        VkMemoryBarrier visible{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        visible.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        visible.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT, VK_PIPELINE_STAGE_HOST_BIT,
                             0, 1, &visible, 0, nullptr, 0, nullptr);
    }
    CHECK(vkEndCommandBuffer(cmd));
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount = 1; si.pCommandBuffers = &cmd;
    CHECK(vkQueueSubmit(q, 1, &si, VK_NULL_HANDLE));
    CHECK(vkQueueWaitIdle(q));
    void* p; CHECK(vkMapMemory(dev, bmem, 0, bi.size, 0, &p));
    if (heap_dump) {
        std::ofstream dump(heap_dump, std::ios::binary);
        dump.write(static_cast<const char*>(heap_mapping), heap_bytes);
        dump.close();
        if (!dump) return 1;
    }
    std::ofstream output(argv[3], std::ios::binary);
    output.write(static_cast<const char*>(p), std::streamsize(bi.size));
    output.close();
    if (!output) { std::fprintf(stderr, "failed to write %s\n", argv[3]); return 1; }
    std::fprintf(stderr, "ok %ux%u -> %s\n", W, H, argv[3]);
    return 0;
}
