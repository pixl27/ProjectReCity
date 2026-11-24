// Native PE mapper for Xbox Series (x86-64) images.
// Maps an image, applies relocations, binds every import through the shim registry, registers its unwind table and
// installs static TLS for the calling thread.
// Mapping never runs code from the image. Loader::initialise() is the one call that does: it sets up the module's TLS
// (slot, template and callbacks) and calls a DLL's entry point as DllMain.
// gta6rt_tls_host.dll must sit next to the executable; the loader reuses the module TLS index the OS assigns it.
#pragma once
#include <cstdint>
#include <filesystem>
#include <map>
#include <string>
#include <vector>

namespace rt {

enum class Binding { Shim, Host, Module, Unimplemented };

// No module TLS index has been assigned (matches TLS_OUT_OF_INDEXES without pulling <windows.h> into this header).
inline constexpr uint32_t kNoTlsIndex = 0xFFFFFFFFu;

struct ImportRecord {
    std::string dll;
    std::string name;   // "#123" for ordinals
    Binding binding;
};

struct Module {
    std::string name;
    std::filesystem::path path;
    uint8_t* base = nullptr;
    uint64_t preferred_base = 0;
    uint32_t size = 0;
    uint32_t entry_rva = 0;
    bool relocated = false;
    bool initialised = false;
    bool attach_failed = false;

    // The image's exception directory (.pdata), registered with RtlAddFunctionTable by load() so that exception
    // dispatch and stack walks can unwind through its frames. Removed when the loader is destroyed.
    void* function_table = nullptr;
    uint32_t function_count = 0;

    std::vector<ImportRecord> imports;
    std::map<std::string, void*> exports;           // by name, plus "#ordinal"
    std::map<std::string, std::string> forwarders;  // same keys, holding "module.symbol" for re-exports

    // Static TLS (IMAGE_DIRECTORY_ENTRY_TLS). load() installs tls_block in the calling thread and writes tls_index to
    // the image's _tls_index. tls_index names TEB->ThreadLocalStoragePointer[tls_index], the array
    // __declspec(thread) reads; it is not a TlsAlloc slot. The loader reserves the index by loading
    // gta6rt_tls_host.dll, so only one manually mapped TLS module can be live at a time; a second one is rejected
    // rather than sharing (and corrupting) the slot. install_tls_for_thread() must be called for every other thread
    // the runtime creates.
    uint32_t tls_index = kNoTlsIndex;
    void* tls_block = nullptr;         // calling thread's copy of the image's template
    void* tls_saved_block = nullptr;   // helper DLL block replaced in that thread, restored by ~Loader
    uint32_t tls_block_size = 0;       // raw template + zero fill
    const uint8_t* tls_template = nullptr;  // raw template in the mapped image
    uint32_t tls_raw_size = 0;
    uint32_t tls_zero_fill = 0;
};

class Loader {
public:
    // Also registers the module list with the shim registry, so an unimplemented import called under
    // RECITY_UNIMPLEMENTED=stop can name the module and RVA of its caller.
    explicit Loader(std::filesystem::path game_dir);
    Loader(const Loader&) = delete;
    Loader& operator=(const Loader&) = delete;
    // Restores the TLS slot for the calling thread. Destroy the Loader on the thread
    // that called load() while any mapped TLS module is still live.
    ~Loader();

    // Loads an image and (recursively) the bundled modules it imports. Throws std::runtime_error on failure.
    // Static TLS is installed here, before any image code can run; initialise() does not install it again.
    Module& load(const std::filesystem::path& image);

    const std::vector<Module*>& modules() const { return order_; }

    // The order initialise() runs modules in: each module after every module it imports (a cycle is broken at the
    // module that closed it). modules() is mapping order, which puts an importer before its imports.
    const std::vector<Module*>& initialisation_order() const { return init_order_; }

    // Runs each loaded module's TLS callbacks and then its DllMain(DLL_PROCESS_ATTACH), dependencies first.
    // Only DLLs get DllMain called: an EXE's entry point is the caller's to run. Throws on a corrupt TLS
    // directory, a TLS callback outside its image, or a DllMain that returns FALSE.
    // TLS was already installed by load(), so this is safe to call on the loading thread without a second install.
    void initialise();

    // Calls the image's IMAGE_TLS_DIRECTORY64 callbacks in order with (base, reason, nullptr). This is opt-in:
    // load() never runs image code, and initialise() is the only call inside the Loader that uses it. No-op for
    // images without TLS callbacks. The loader does not deliver thread attach/detach events: install the thread's
    // TLS first, then call this with DLL_THREAD_ATTACH/DETACH, and uninstall after the detach callbacks.
    void run_tls_callbacks(Module& m, uint32_t reason);

    // Installs/removes this module's template copy in the calling thread's TEB slot. load() does this for its caller;
    // the runtime must call install before mapped code runs on any other thread and uninstall before that thread
    // exits. Only one TLS module can be installed in a thread. These do not run TLS callbacks (use run_tls_callbacks
    // with DLL_THREAD_ATTACH/DETACH for that).
    void install_tls_for_thread(Module& m);
    void uninstall_tls_for_thread(Module& m);

private:
    void initialise(Module& m);
    Module& map_image(const std::filesystem::path& path);
    void initialize_tls(Module& m);
    void bind_imports(Module& m);
    void* resolve(Module& m, const std::string& dll, const std::string& name, uint16_t ordinal, bool by_ordinal);
    void* resolve_forwarder(const std::string& spec, Binding& binding, int depth);
    void parse_exports(Module& m);
    void protect_sections(Module& m);
    void register_unwind(Module& m);

    std::filesystem::path game_dir_;
    std::map<std::string, Module*> by_name_;  // lower-case file name
    std::vector<Module*> order_;       // mapping order; owns the modules
    std::vector<Module*> init_order_;  // dependencies first, see initialisation_order()
};

std::string lower(std::string s);

}  // namespace rt
