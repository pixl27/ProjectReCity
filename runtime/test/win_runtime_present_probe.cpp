#define WIN32_LEAN_AND_MEAN
#define VK_USE_PLATFORM_WIN32_KHR
#define VK_NO_PROTOTYPES
#include <windows.h>
#include <vulkan/vulkan.h>
#include <cstdio>
#include <vector>
#include <cstring>
#include <cstdint>
#include <wrl/client.h>
#include "../shims/d3d12_com.h"

using Microsoft::WRL::ComPtr;
#define CHECK_HR(call) do { HRESULT hr = (call); if (FAILED(hr)) { std::fprintf(stderr, "%s failed: %08lx\n", #call, (unsigned long)hr); return 1; } } while (0)

static unsigned errors = 0, warnings = 0, loaderWarnings = 0;
static VKAPI_ATTR VkBool32 VKAPI_CALL debug_cb(VkDebugUtilsMessageSeverityFlagBitsEXT severity, VkDebugUtilsMessageTypeFlagsEXT type, const VkDebugUtilsMessengerCallbackDataEXT* data, void*) {
    if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) ++errors;
    else if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
        if (type == VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT && data->pMessageIdName &&
            std::strcmp(data->pMessageIdName, "Loader Message") == 0)
            ++loaderWarnings;
        else
            ++warnings;
    }
    std::fprintf(stderr, "validation: %s\n", data->pMessage);
    return VK_FALSE;
}
#define CHECK(x) do { VkResult result_ = (x); if (result_ != VK_SUCCESS) { std::fprintf(stderr, "%s: VkResult %d at line %d\n", #x, result_, __LINE__); return 1; } } while (0)
#define GLOBAL(name) auto name = reinterpret_cast<PFN_##name>(get(VK_NULL_HANDLE, #name)); if (!name) return 2
#define INSTANCE(name) auto name = reinterpret_cast<PFN_##name>(get(instance, #name)); if (!name) { std::fprintf(stderr, "missing %s\n", #name); return 2; }
#define DEVICE(name) auto name = reinterpret_cast<PFN_##name>(getDevice(device, #name)); if (!name) { std::fprintf(stderr, "missing %s\n", #name); return 2; }

int main() {
    HMODULE loader = LoadLibraryW(L"vulkan-1.dll");
    if (!loader) { std::fprintf(stderr, "no Vulkan loader\n"); return 1; }
    auto get = reinterpret_cast<PFN_vkGetInstanceProcAddr>(GetProcAddress(loader, "vkGetInstanceProcAddr"));
    if (!get) return 1;
    GLOBAL(vkCreateInstance); GLOBAL(vkEnumerateInstanceLayerProperties);
    uint32_t n = 0; CHECK(vkEnumerateInstanceLayerProperties(&n, nullptr));
    std::vector<VkLayerProperties> layers(n); CHECK(vkEnumerateInstanceLayerProperties(&n, layers.data()));
    bool found = false; for (auto& l : layers) if (!std::strcmp(l.layerName, "VK_LAYER_KHRONOS_validation")) found = true;
    if (!found) { std::fprintf(stderr, "validation layer unavailable\n"); return 1; }
    const char* layer = "VK_LAYER_KHRONOS_validation";
    const char* exts[] = {VK_KHR_SURFACE_EXTENSION_NAME, VK_KHR_WIN32_SURFACE_EXTENSION_NAME, VK_EXT_DEBUG_UTILS_EXTENSION_NAME};
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO}; app.pApplicationName = "ReCity D3D12 frame presentation probe"; app.apiVersion = VK_API_VERSION_1_3;
    VkDebugUtilsMessengerCreateInfoEXT debug{VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
    debug.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    debug.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT; debug.pfnUserCallback = debug_cb;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO}; ici.pNext = &debug; ici.pApplicationInfo = &app; ici.enabledLayerCount = 1; ici.ppEnabledLayerNames = &layer; ici.enabledExtensionCount = 3; ici.ppEnabledExtensionNames = exts;
    VkInstance instance{}; CHECK(vkCreateInstance(&ici, nullptr, &instance));
    std::printf("validation layer requested and instance creation succeeded: %s\n", layer);
    INSTANCE(vkCreateDebugUtilsMessengerEXT); INSTANCE(vkDestroyDebugUtilsMessengerEXT); INSTANCE(vkDestroyInstance); INSTANCE(vkCreateWin32SurfaceKHR); INSTANCE(vkDestroySurfaceKHR);
    INSTANCE(vkEnumeratePhysicalDevices); INSTANCE(vkGetPhysicalDeviceProperties); INSTANCE(vkGetPhysicalDeviceQueueFamilyProperties); INSTANCE(vkGetPhysicalDeviceSurfaceSupportKHR); INSTANCE(vkGetPhysicalDeviceSurfaceCapabilitiesKHR); INSTANCE(vkGetPhysicalDeviceSurfaceFormatsKHR); INSTANCE(vkGetPhysicalDeviceMemoryProperties); INSTANCE(vkCreateDevice); INSTANCE(vkGetDeviceProcAddr);
    auto getDevice = vkGetDeviceProcAddr;
    VkDebugUtilsMessengerEXT messenger{}; CHECK(vkCreateDebugUtilsMessengerEXT(instance, &debug, nullptr, &messenger));
    WNDCLASSW wc{}; wc.lpfnWndProc = DefWindowProcW; wc.hInstance = GetModuleHandleW(nullptr); wc.lpszClassName = L"ReCityVulkanProbe";
    if (!RegisterClassW(&wc)) { std::fprintf(stderr, "RegisterClass failed %lu\n", GetLastError()); return 1; }
    HWND window = CreateWindowExW(0, wc.lpszClassName, L"ReCity Vulkan swapchain diagnostic", WS_OVERLAPPEDWINDOW | WS_VISIBLE, CW_USEDEFAULT, CW_USEDEFAULT, 640, 480, nullptr, nullptr, wc.hInstance, nullptr);
    if (!window) { std::fprintf(stderr, "CreateWindow failed %lu\n", GetLastError()); return 1; }
    VkWin32SurfaceCreateInfoKHR sci{VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR}; sci.hinstance = wc.hInstance; sci.hwnd = window;
    VkSurfaceKHR surface{}; CHECK(vkCreateWin32SurfaceKHR(instance, &sci, nullptr, &surface));
    uint32_t count = 0; CHECK(vkEnumeratePhysicalDevices(instance, &count, nullptr)); if (!count) return 1;
    std::vector<VkPhysicalDevice> devices(count); CHECK(vkEnumeratePhysicalDevices(instance, &count, devices.data()));
    VkPhysicalDevice physical{}; uint32_t family = 0;
    for (auto p : devices) { uint32_t qn = 0; vkGetPhysicalDeviceQueueFamilyProperties(p, &qn, nullptr); std::vector<VkQueueFamilyProperties> qs(qn); vkGetPhysicalDeviceQueueFamilyProperties(p, &qn, qs.data()); for (uint32_t i = 0; i < qn; ++i) { VkBool32 supports = VK_FALSE; CHECK(vkGetPhysicalDeviceSurfaceSupportKHR(p, i, surface, &supports)); if (supports && (qs[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) { physical = p; family = i; break; } } if (physical) break; }
    if (!physical) { std::fprintf(stderr, "no graphics/present queue\n"); return 1; }
    VkPhysicalDeviceProperties props{}; vkGetPhysicalDeviceProperties(physical, &props); std::printf("adapter: %s, graphics/present queue %u\n", props.deviceName, family);
    float priority = 1.f; VkDeviceQueueCreateInfo qi{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO}; qi.queueFamilyIndex = family; qi.queueCount = 1; qi.pQueuePriorities = &priority;
    const char* devext = VK_KHR_SWAPCHAIN_EXTENSION_NAME; VkDeviceCreateInfo di{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO}; di.queueCreateInfoCount = 1; di.pQueueCreateInfos = &qi; di.enabledExtensionCount = 1; di.ppEnabledExtensionNames = &devext;
    VkPhysicalDeviceVulkan13Features features13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    features13.dynamicRendering = VK_TRUE;
    features13.synchronization2 = VK_TRUE;
    VkPhysicalDeviceVulkan12Features features12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    features12.timelineSemaphore = VK_TRUE;
    features12.pNext = &features13;
    di.pNext = &features12;
    VkDevice device{}; CHECK(vkCreateDevice(physical, &di, nullptr, &device));
    DEVICE(vkGetDeviceQueue); DEVICE(vkDestroyDevice); DEVICE(vkCreateSwapchainKHR); DEVICE(vkDestroySwapchainKHR); DEVICE(vkGetSwapchainImagesKHR); DEVICE(vkAcquireNextImageKHR); DEVICE(vkQueueSubmit); DEVICE(vkQueuePresentKHR); DEVICE(vkQueueWaitIdle); DEVICE(vkCreateSemaphore); DEVICE(vkDestroySemaphore); DEVICE(vkCreateCommandPool); DEVICE(vkDestroyCommandPool); DEVICE(vkAllocateCommandBuffers); DEVICE(vkResetCommandBuffer); DEVICE(vkBeginCommandBuffer); DEVICE(vkEndCommandBuffer); DEVICE(vkCmdPipelineBarrier); DEVICE(vkCmdCopyImage); DEVICE(vkCmdCopyImageToBuffer); DEVICE(vkCreateBuffer); DEVICE(vkDestroyBuffer); DEVICE(vkGetBufferMemoryRequirements); DEVICE(vkAllocateMemory); DEVICE(vkFreeMemory); DEVICE(vkBindBufferMemory); DEVICE(vkMapMemory); DEVICE(vkUnmapMemory);
    VkQueue queue{}; vkGetDeviceQueue(device, family, 0, &queue);
    VkSurfaceCapabilitiesKHR caps{}; CHECK(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physical, surface, &caps));
    if ((caps.supportedUsageFlags & (VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT)) != (VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT)) { std::fprintf(stderr, "surface does not support transfer src/dst\n"); return 1; }
    count = 0; CHECK(vkGetPhysicalDeviceSurfaceFormatsKHR(physical, surface, &count, nullptr)); std::vector<VkSurfaceFormatKHR> formats(count); CHECK(vkGetPhysicalDeviceSurfaceFormatsKHR(physical, surface, &count, formats.data()));
    VkSurfaceFormatKHR format{}; for (auto f : formats) if (f.format == VK_FORMAT_B8G8R8A8_UNORM || f.format == VK_FORMAT_R8G8B8A8_UNORM) { format = f; break; } if (!format.format) { std::fprintf(stderr, "no RGBA/BGRA UNORM format\n"); return 1; }
    VkExtent2D extent = caps.currentExtent; if (extent.width == UINT32_MAX) { extent.width = 640; extent.height = 480; }
    uint32_t imagesWanted = caps.minImageCount + 1; if (caps.maxImageCount && imagesWanted > caps.maxImageCount) imagesWanted = caps.maxImageCount;
    VkCompositeAlphaFlagBitsKHR alpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR; if (!(caps.supportedCompositeAlpha & alpha)) { for (uint32_t bit = 1; bit <= 8; bit <<= 1) if (caps.supportedCompositeAlpha & bit) { alpha = static_cast<VkCompositeAlphaFlagBitsKHR>(bit); break; } }
    VkSwapchainCreateInfoKHR ci{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR}; ci.surface = surface; ci.minImageCount = imagesWanted; ci.imageFormat = format.format; ci.imageColorSpace = format.colorSpace; ci.imageExtent = extent; ci.imageArrayLayers = 1; ci.imageUsage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT; ci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE; ci.preTransform = caps.currentTransform; ci.compositeAlpha = alpha; ci.presentMode = VK_PRESENT_MODE_FIFO_KHR; ci.clipped = VK_TRUE;
    VkSwapchainKHR swap{}; CHECK(vkCreateSwapchainKHR(device, &ci, nullptr, &swap)); count = 0; CHECK(vkGetSwapchainImagesKHR(device, swap, &count, nullptr)); std::vector<VkImage> images(count); CHECK(vkGetSwapchainImagesKHR(device, swap, &count, images.data()));
    std::printf("swapchain: %ux%u, format=%d, images=%u, FIFO\n", extent.width, extent.height, int(format.format), count);
    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO}; bi.size = VkDeviceSize(extent.width) * extent.height * 4; bi.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT; bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE; VkBuffer buffer{}; CHECK(vkCreateBuffer(device, &bi, nullptr, &buffer));
    VkMemoryRequirements req{}; vkGetBufferMemoryRequirements(device, buffer, &req); VkPhysicalDeviceMemoryProperties mem{}; vkGetPhysicalDeviceMemoryProperties(physical, &mem); uint32_t type = UINT32_MAX; for (uint32_t i = 0; i < mem.memoryTypeCount; ++i) if ((req.memoryTypeBits & (1u << i)) && (mem.memoryTypes[i].propertyFlags & (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) == (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) { type = i; break; } if (type == UINT32_MAX) return 1;
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO}; ai.allocationSize = req.size; ai.memoryTypeIndex = type; VkDeviceMemory memory{}; CHECK(vkAllocateMemory(device, &ai, nullptr, &memory)); CHECK(vkBindBufferMemory(device, buffer, memory, 0)); void* mapped{}; CHECK(vkMapMemory(device, memory, 0, VK_WHOLE_SIZE, 0, &mapped));
    VkCommandPoolCreateInfo pi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO}; pi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT; pi.queueFamilyIndex = family; VkCommandPool pool{}; CHECK(vkCreateCommandPool(device, &pi, nullptr, &pool)); VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO}; cai.commandPool = pool; cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cai.commandBufferCount = 1; VkCommandBuffer cmd{}; CHECK(vkAllocateCommandBuffers(device, &cai, &cmd));
    VkSemaphoreCreateInfo seminfo{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO}; VkSemaphore acquired{}, finished{}; CHECK(vkCreateSemaphore(device, &seminfo, nullptr, &acquired)); CHECK(vkCreateSemaphore(device, &seminfo, nullptr, &finished));
    unsigned acqs = 0, submits = 0, presents = 0, pixels = 0; MSG msg{};
    {
    rt::d3d12::VulkanDeviceInfo info{get, instance, physical, device, queue, family};
    info.separate_depth_stencil_layouts = false;
    ComPtr<ID3D12Device> renderer;
    CHECK_HR(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(renderer.GetAddressOf())));
    ComPtr<ID3D12CommandQueue> renderQueue;
    D3D12_COMMAND_QUEUE_DESC queueDesc{};
    CHECK_HR(renderer->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(renderQueue.GetAddressOf())));
    ComPtr<ID3D12CommandAllocator> allocator;
    CHECK_HR(renderer->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(allocator.GetAddressOf())));
    ComPtr<ID3D12GraphicsCommandList> list;
    CHECK_HR(renderer->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.Get(), nullptr, IID_PPV_ARGS(list.GetAddressOf())));
    ComPtr<ID3D12Resource> target;
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width = extent.width;
    desc.Height = extent.height;
    desc.DepthOrArraySize = desc.MipLevels = desc.SampleDesc.Count = 1;
    desc.Format = format.format == VK_FORMAT_B8G8R8A8_UNORM ? DXGI_FORMAT_B8G8R8A8_UNORM : DXGI_FORMAT_R8G8B8A8_UNORM;
    desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
    D3D12_HEAP_PROPERTIES heapProps{};
    heapProps.Type = D3D12_HEAP_TYPE_DEFAULT;
    CHECK_HR(renderer->CreateCommittedResource(&heapProps, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_RENDER_TARGET,
                                               nullptr, IID_PPV_ARGS(target.GetAddressOf())));
    const VkImage sourceImage = rt::d3d12::vulkan_image(target.Get());
    if (!sourceImage) return 1;
    ComPtr<ID3D12DescriptorHeap> descriptors;
    D3D12_DESCRIPTOR_HEAP_DESC heapDesc{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1};
    CHECK_HR(renderer->CreateDescriptorHeap(&heapDesc, IID_PPV_ARGS(descriptors.GetAddressOf())));
    const auto rtv = descriptors->GetCPUDescriptorHandleForHeapStart();
    renderer->CreateRenderTargetView(target.Get(), nullptr, rtv);
    D3D12_RESOURCE_BARRIER transition{};
    transition.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    transition.Transition.pResource = target.Get();
    transition.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    for (unsigned frame = 0; frame < 60; ++frame) {
        while (PeekMessageW(&msg, nullptr, 0, 0, PM_REMOVE)) { TranslateMessage(&msg); DispatchMessageW(&msg); }
        if (frame) {
            CHECK_HR(allocator->Reset());
            CHECK_HR(list->Reset(allocator.Get(), nullptr));
            transition.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_SOURCE;
            transition.Transition.StateAfter = D3D12_RESOURCE_STATE_RENDER_TARGET;
            list->ResourceBarrier(1, &transition);
        }
        const unsigned char tag = static_cast<unsigned char>(frame + 1);
        const float color[]{tag / 255.f, (255 - tag) / 255.f, 0.f, 1.f};
        list->ClearRenderTargetView(rtv, color, 0, nullptr);
        transition.Transition.StateBefore = D3D12_RESOURCE_STATE_RENDER_TARGET;
        transition.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
        list->ResourceBarrier(1, &transition);
        CHECK_HR(list->Close());
        ID3D12CommandList* commands[]{list.Get()};
        renderQueue->ExecuteCommandLists(1, commands);
        uint32_t index = 0; CHECK(vkAcquireNextImageKHR(device, swap, UINT64_MAX, acquired, VK_NULL_HANDLE, &index)); ++acqs;
        CHECK(vkResetCommandBuffer(cmd, 0)); VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}; begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT; CHECK(vkBeginCommandBuffer(cmd, &begin));
        VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER}; barrier.srcQueueFamilyIndex = barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED; barrier.image = images[index]; barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}; barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED; barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL; barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);
        VkImageCopy copy{};
        copy.srcSubresource = copy.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        copy.extent = {extent.width, extent.height, 1};
        vkCmdCopyImage(cmd, sourceImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, images[index],
                       VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
        barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL; barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL; barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT; barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT; vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);
        VkBufferImageCopy region{}; region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}; region.imageExtent = {extent.width, extent.height, 1}; vkCmdCopyImageToBuffer(cmd, images[index], VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, buffer, 1, &region);
        barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL; barrier.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR; barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT; barrier.dstAccessMask = 0; vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier); CHECK(vkEndCommandBuffer(cmd));
        VkPipelineStageFlags wait = VK_PIPELINE_STAGE_TRANSFER_BIT; VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO}; submit.waitSemaphoreCount = 1; submit.pWaitSemaphores = &acquired; submit.pWaitDstStageMask = &wait; submit.commandBufferCount = 1; submit.pCommandBuffers = &cmd; submit.signalSemaphoreCount = 1; submit.pSignalSemaphores = &finished; CHECK(vkQueueSubmit(queue, 1, &submit, VK_NULL_HANDLE)); ++submits;
        VkPresentInfoKHR present{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR}; present.waitSemaphoreCount = 1; present.pWaitSemaphores = &finished; present.swapchainCount = 1; present.pSwapchains = &swap; present.pImageIndices = &index; CHECK(vkQueuePresentKHR(queue, &present)); ++presents; CHECK(vkQueueWaitIdle(queue));
        const auto* data = static_cast<const unsigned char*>(mapped);
        const unsigned red = format.format == VK_FORMAT_B8G8R8A8_UNORM ? 2 : 0;
        for (size_t i = 0; i < size_t(extent.width) * extent.height; ++i) {
            const auto* p = data + 4 * i;
            if (p[red] != tag || p[1] != 255 - tag || p[2 - red] != 0 || p[3] != 255) {
                std::fprintf(stderr, "frame %u pixel %zu mismatch: %u %u %u %u\n", frame, i, p[red], p[1], p[2-red], p[3]);
                return 1;
            }
        }
        ++pixels;
    }
    } // Release ReCity objects while their Vulkan device is alive.
    vkDestroySemaphore(device, finished, nullptr); vkDestroySemaphore(device, acquired, nullptr); vkDestroyCommandPool(device, pool, nullptr); vkUnmapMemory(device, memory); vkDestroyBuffer(device, buffer, nullptr); vkFreeMemory(device, memory, nullptr); vkDestroySwapchainKHR(device, swap, nullptr); vkDestroyDevice(device, nullptr); vkDestroySurfaceKHR(instance, surface, nullptr); DestroyWindow(window); vkDestroyDebugUtilsMessengerEXT(instance, messenger, nullptr); vkDestroyInstance(instance, nullptr);
    std::printf("RESULT acquire=%u submit=%u present=%u full_frame_matches=%u validation_errors=%u validation_warnings=%u loader_warnings=%u\n", acqs, submits, presents, pixels, errors, warnings, loaderWarnings);
    return acqs == 60 && submits == 60 && presents == 60 && pixels == 60 && errors == 0 && warnings == 0 ? 0 : 1;
}
