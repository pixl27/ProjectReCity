// Process-level imports of a mapped executable: its command line and its way out.
//
// `gta6rt --run` calls a mapped EXE's entry point inside the gta6rt process. The EXE's CRT startup asks for its
// arguments and finally calls exit(); both go through the EXE's own import table, so they can be answered here:
//   - argv/argc and the command line are the game's (set by run_entry), not gta6rt's;
//   - exit(), _exit(), quick_exit(), ExitProcess() and abort() from the game end the game's run, not the host:
//     the exit code is recorded, the run is signalled as finished and the calling thread ends. atexit handlers
//     and static destructors of the game are not run (its CRT shares the host's, whose process-wide atexit table
//     also holds gta6rt's own).
// Covered CRTs: the Universal CRT (api-ms-win-crt-runtime-l1-1-0.dll / ucrtbase.dll, what MSVC links; ucrtbased.dll in Debug) and
// msvcrt.dll (MinGW). Calls made inside host DLLs, not through the game's imports, are not seen here.
#include <windows.h>
#include <process.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <new>
#include <utility>
#include <string>
#include <vector>

#include "../src/pe_loader.h"
#include "process_shim.h"
#include "shims.h"

namespace rt::process {
namespace {

std::vector<std::string> g_args;
std::vector<std::wstring> g_wargs;
std::vector<char*> g_argv;
std::vector<wchar_t*> g_wargv;
std::string g_cmdline;
std::wstring g_wcmdline;
int g_argc = 0;
char** g_argv_ptr = nullptr;
wchar_t** g_wargv_ptr = nullptr;
char** g_env = nullptr;
wchar_t** g_wenv = nullptr;

std::atomic<bool> g_exited{false};
std::atomic<int> g_exit_code{0};
HANDLE g_done = nullptr;

std::string quote(const std::string& a) {
    if (!a.empty() && a.find_first_of(" \t\"") == std::string::npos) return a;
    std::string out = "\"";
    for (char c : a) {
        if (c == '"') out += '\\';
        out += c;
    }
    return out + "\"";
}

std::wstring widen(const std::string& s) {
    int n = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, nullptr, 0);
    std::wstring w(n > 0 ? n - 1 : 0, L'\0');
    if (n > 1) MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, w.data(), n);
    return w;
}

// Only a mapped entry or a worker it starts inherits this context. The loader owns
// at most one TLS module, and --run keeps its images alive until process exit.
struct GameTls {
    Loader* loader = nullptr;
    Module* module = nullptr;
    bool notify_detach = false;
};
thread_local GameTls current_tls;

bool detach_thread_tls() {
    const GameTls tls = std::exchange(current_tls, GameTls{});
    if (!tls.module) return true;
    bool ok = true;
    try {
        if (tls.notify_detach) tls.loader->run_tls_callbacks(*tls.module, DLL_THREAD_DETACH);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[rt] TLS thread detach failed: %s\n", e.what());
        ok = false;
    }
    tls.loader->uninstall_tls_for_thread(*tls.module);
    return ok;
}

[[noreturn]] void end_run(int code) {
    if (!detach_thread_tls()) code = 1;  // ExitThread does not unwind the mapped call stack.
    bool expected = false;
    if (g_exited.compare_exchange_strong(expected, true)) {
        g_exit_code = code;
        if (g_done) SetEvent(g_done);
        ExitThread(static_cast<DWORD>(code));
    }
    // The run is already over: this is game code running during host shutdown (an atexit handler the game
    // registered with the shared CRT calling exit again). Finish the process with the game's exit code; ending
    // just this thread here would make the host's last thread, and so the process, exit with `code` instead.
    ExitProcess(static_cast<UINT>(g_exit_code.load()));
}

void attach_thread_tls(GameTls tls, bool worker) {
    if (!tls.module) return;
    try {
        tls.loader->install_tls_for_thread(*tls.module);
        current_tls = tls;
        current_tls.notify_detach = worker;
        if (worker) tls.loader->run_tls_callbacks(*tls.module, DLL_THREAD_ATTACH);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[rt] TLS thread attach failed: %s\n", e.what());
        end_run(1);
    }
}

// The native API owns thread creation, suspension, stack size and the returned handle.
// Copy the start arguments before calling mapped code, which can exit without returning.
template<class Result>
struct ThreadStart {
    using Routine = Result (WINAPI*)(void*);
    Routine routine;
    void* argument;
    GameTls tls;
    static Result WINAPI invoke(void* p) {
        auto* start = static_cast<ThreadStart*>(p);
        const ThreadStart call = *start;
        delete start;
        attach_thread_tls(call.tls, true);
        const Result result = call.routine(call.argument);
        if (!detach_thread_tls()) end_run(1);
        return result;
    }
};

HANDLE WINAPI game_create_thread(LPSECURITY_ATTRIBUTES sa, SIZE_T stack, LPTHREAD_START_ROUTINE fn,
                                 void* arg, DWORD flags, LPDWORD id) {
    if (!current_tls.module || !fn) return CreateThread(sa, stack, fn, arg, flags, id);
    auto* start = new (std::nothrow) ThreadStart<DWORD>{fn, arg, current_tls};
    if (!start) { SetLastError(ERROR_NOT_ENOUGH_MEMORY); return nullptr; }
    HANDLE h = CreateThread(sa, stack, ThreadStart<DWORD>::invoke, start, flags, id);
    if (!h) {
        const DWORD error = GetLastError();
        delete start;
        SetLastError(error);
    }
    return h;
}

void WINAPI game_exit_thread(DWORD code) {
    if (!detach_thread_tls()) end_run(1);
    ExitThread(code);
}

// Forward to the game's CRT even when it and the launcher use different configurations.
template<bool Debug>
HMODULE thread_crt() {
    static HMODULE crt = LoadLibraryExW(Debug ? L"ucrtbased.dll" : L"ucrtbase.dll", nullptr,
                                        LOAD_LIBRARY_SEARCH_SYSTEM32);
    return crt;
}

template<bool Debug>
uintptr_t __cdecl game_beginthreadex(void* sa, unsigned stack, ThreadStart<unsigned>::Routine fn,
                                     void* arg, unsigned flags, unsigned* id) {
    static auto begin = reinterpret_cast<decltype(&_beginthreadex)>(GetProcAddress(thread_crt<Debug>(), "_beginthreadex"));
    if (!begin) { errno = ENOSYS; return 0; }
    if (!current_tls.module || !fn) return begin(sa, stack, fn, arg, flags, id);
    auto* start = new (std::nothrow) ThreadStart<unsigned>{fn, arg, current_tls};
    if (!start) { errno = ENOMEM; return 0; }
    uintptr_t h = begin(sa, stack, ThreadStart<unsigned>::invoke, start, flags, id);
    if (!h) delete start;
    return h;
}

template<bool Debug>
void __cdecl game_endthreadex(unsigned code) {
    if (!detach_thread_tls()) end_run(1);
    static auto end = reinterpret_cast<decltype(&_endthreadex)>(GetProcAddress(thread_crt<Debug>(), "_endthreadex"));
    if (end) end(code);
    ExitThread(code);
}

// ---- the way out
void __cdecl game_exit(int code) { end_run(code); }
void __stdcall game_exit_process(UINT code) { end_run(static_cast<int>(code)); }
void __cdecl game_abort() {
    std::fprintf(stderr, "[rt] process: the game called abort()\n");
    end_run(3);  // the CRT's abort exit code
}

// ---- arguments
int* __cdecl game_p_argc() { return &g_argc; }
char*** __cdecl game_p_argv() { return &g_argv_ptr; }
wchar_t*** __cdecl game_p_wargv() { return &g_wargv_ptr; }
int __cdecl game_configure_argv(int) { return 0; }  // already parsed: run_entry set them
LPSTR __stdcall game_get_command_line_a() { return g_cmdline.data(); }
LPWSTR __stdcall game_get_command_line_w() { return g_wcmdline.data(); }
char** __cdecl game_p_acmdln_storage() {
    static char* p = nullptr;
    p = g_cmdline.data();
    return &p;
}
wchar_t** __cdecl game_p_wcmdln_storage() {
    static wchar_t* p = nullptr;
    p = g_wcmdline.data();
    return &p;
}

// msvcrt.dll: int __getmainargs(int* argc, char*** argv, char*** env, int expand, void* startinfo)
int __cdecl game_getmainargs(int* argc, char*** argv, char*** env, int, void*) {
    *argc = g_argc;
    *argv = g_argv_ptr;
    if (env) *env = g_env;
    return 0;
}
int __cdecl game_wgetmainargs(int* argc, wchar_t*** argv, wchar_t*** env, int, void*) {
    *argc = g_argc;
    *argv = g_wargv_ptr;
    if (env) *env = g_wenv;
    return 0;
}

struct EntryCall {
    void* entry;
    GameTls tls;
};

DWORD WINAPI game_thread(void* p) {
    const EntryCall call = *static_cast<EntryCall*>(p);
    // An EXE entry point (mainCRTStartup and friends) takes the PEB pointer and returns the process exit code.
    auto entry = reinterpret_cast<DWORD(WINAPI*)(void*)>(call.entry);
    attach_thread_tls(call.tls, false);
    try {
        // PROCESS_ATTACH and the entry point must see the same per-thread data.
        call.tls.loader->initialise();
        std::printf("%zu module(s) initialised\n", call.tls.loader->modules().size());
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[rt] initialisation failed: %s\n", e.what());
        end_run(1);
    }
    const DWORD code = entry(nullptr);
    end_run(static_cast<int>(code));  // returning from the entry point is an exit too
}

}  // namespace

void set_command_line(const std::string& program, const std::vector<std::string>& args) {
    g_args.clear();
    g_args.push_back(program);
    g_args.insert(g_args.end(), args.begin(), args.end());
    g_wargs.clear();
    g_cmdline.clear();
    for (auto& a : g_args) {
        g_wargs.push_back(widen(a));
        if (!g_cmdline.empty()) g_cmdline += ' ';
        g_cmdline += quote(a);
    }
    g_wcmdline = widen(g_cmdline);
    g_argv.clear();
    g_wargv.clear();
    for (auto& a : g_args) g_argv.push_back(a.data());
    for (auto& a : g_wargs) g_wargv.push_back(a.data());
    g_argv.push_back(nullptr);
    g_wargv.push_back(nullptr);
    g_argc = static_cast<int>(g_args.size());
    g_argv_ptr = g_argv.data();
    g_wargv_ptr = g_wargv.data();
    g_env = _environ;
    g_wenv = _wenviron;
}

int run_entry(Loader& loader, void* entry, size_t stack_bytes) {
    EntryCall call{entry, {&loader}};
    for (Module* m : loader.modules()) {
        if (m->tls_index != kNoTlsIndex) call.tls.module = m;
    }
    g_exited = false;
    g_exit_code = 0;
    g_done = CreateEventW(nullptr, TRUE, FALSE, nullptr);
    if (!g_done) return -1;
    HANDLE thread = CreateThread(nullptr, stack_bytes, game_thread, &call, STACK_SIZE_PARAM_IS_A_RESERVATION, nullptr);
    if (!thread) {
        CloseHandle(g_done);
        g_done = nullptr;
        return -1;
    }
    // The run ends when any game thread exits the process (exit/ExitProcess/abort) or the entry point returns.
    HANDLE waits[] = {g_done, thread};
    DWORD code = 0;
    const DWORD wait = WaitForMultipleObjects(2, waits, FALSE, INFINITE);
    if (wait == WAIT_OBJECT_0) code = static_cast<DWORD>(g_exit_code.load());
    else if (wait != WAIT_OBJECT_0 + 1 || !GetExitCodeThread(thread, &code)) code = static_cast<DWORD>(-1);
    CloseHandle(thread);
    CloseHandle(g_done);
    g_done = nullptr;
    return static_cast<int>(code);
}

}  // namespace rt::process

namespace {
using namespace rt::process;
#define UCRT(dll, debug) \
    {dll, "_beginthreadex", reinterpret_cast<void*>(&game_beginthreadex<debug>)}, \
    {dll, "_endthreadex", reinterpret_cast<void*>(&game_endthreadex<debug>)}, \
    {dll, "exit", reinterpret_cast<void*>(&game_exit)}, {dll, "_exit", reinterpret_cast<void*>(&game_exit)}, \
    {dll, "_Exit", reinterpret_cast<void*>(&game_exit)}, {dll, "quick_exit", reinterpret_cast<void*>(&game_exit)}, \
    {dll, "abort", reinterpret_cast<void*>(&game_abort)}, {dll, "__p___argc", reinterpret_cast<void*>(&game_p_argc)}, \
    {dll, "__p___argv", reinterpret_cast<void*>(&game_p_argv)}, \
    {dll, "__p___wargv", reinterpret_cast<void*>(&game_p_wargv)}, \
    {dll, "_configure_narrow_argv", reinterpret_cast<void*>(&game_configure_argv)}, \
    {dll, "_configure_wide_argv", reinterpret_cast<void*>(&game_configure_argv)}, \
    {dll, "__p__acmdln", reinterpret_cast<void*>(&game_p_acmdln_storage)}, \
    {dll, "__p__wcmdln", reinterpret_cast<void*>(&game_p_wcmdln_storage)}
}  // namespace

SHIM_TABLE(process,
    UCRT("api-ms-win-crt-runtime-l1-1-0.dll", false),
    UCRT("ucrtbase.dll", false),
    UCRT("ucrtbased.dll", true),  // the Debug UCRT: a Debug build of the stand-in game (run_tiny_game) links it
    {"msvcrt.dll", "exit", reinterpret_cast<void*>(&game_exit)},
    {"msvcrt.dll", "_exit", reinterpret_cast<void*>(&game_exit)},
    {"msvcrt.dll", "abort", reinterpret_cast<void*>(&game_abort)},
    {"msvcrt.dll", "__getmainargs", reinterpret_cast<void*>(&game_getmainargs)},
    {"msvcrt.dll", "__wgetmainargs", reinterpret_cast<void*>(&game_wgetmainargs)},
    {"kernel32.dll", "CreateThread", reinterpret_cast<void*>(&game_create_thread)},
    {"kernel32.dll", "ExitThread", reinterpret_cast<void*>(&game_exit_thread)},
    {"kernel32.dll", "ExitProcess", reinterpret_cast<void*>(&game_exit_process)},
    {"kernel32.dll", "GetCommandLineA", reinterpret_cast<void*>(&game_get_command_line_a)},
    {"kernel32.dll", "GetCommandLineW", reinterpret_cast<void*>(&game_get_command_line_w)},
);
