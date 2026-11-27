// A stand-in game executable for `gta6rt --run`: what a real title does in its first second, run through the
// loader instead of the Windows one. Test code, not game content. Run as
//   gta6rt --run tiny_game.exe -- hello 42
// It checks its arguments, a static initializer, the heap, a file round trip, a second thread, and the
// xgameruntime.dll imports (answered by the shim registry: the stub DLL it links against would fail them), then
// calls exit() with 0 when everything held. CTest sets RECITY_RUN_TLS=1 to check static TLS.
// A C++ exception (RECITY_RUN_EH=1) needs more than the unwind tables #176 registers on MSVC:
// _CxxThrowException finds the throwing image with RtlPcToFileHeader, which only knows modules the Windows loader
// mapped. With MinGW (libgcc's SEH unwinder) the #176 tables are enough and the check passes.
#include <windows.h>
#include <process.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

extern "C" __declspec(dllimport) HRESULT __stdcall XGameRuntimeInitialize();
extern "C" __declspec(dllimport) void __stdcall XGameRuntimeUninitialize();
extern "C" __declspec(dllimport) uint32_t __stdcall XSystemGetDeviceType();

namespace {

int g_failures = 0;

void check(bool ok, const char* what) {
    std::printf("tiny_game: %s %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) ++g_failures;
}

struct StaticInit {
    int value;
    StaticInit() : value(1234) {}
};
StaticInit g_static;

bool env_on(const char* name) {
    char buf[8] = {};
    return GetEnvironmentVariableA(name, buf, sizeof buf) > 0 && buf[0] == '1';
}

// Force actual reads and writes to the compiler-emitted TLS block in optimized builds.
thread_local volatile int t_counter = 5;
thread_local volatile int t_process_seed = 17;
bool g_check_tls = false;
std::atomic<int> g_win32_detaches{0};
std::atomic<int> g_crt_detaches{0};

void NTAPI tls_callback(PVOID, DWORD reason, PVOID) {
    if (reason == DLL_PROCESS_ATTACH) {
        g_check_tls = env_on("RECITY_RUN_TLS");
        if (g_check_tls) t_process_seed = 23;
    } else if (reason == DLL_THREAD_DETACH && g_check_tls) {
        if (t_counter == 11) g_win32_detaches.fetch_add(1, std::memory_order_relaxed);
        if (t_counter == 13) g_crt_detaches.fetch_add(1, std::memory_order_relaxed);
    }
}

bool check_worker_tls(int value) {
    const bool fresh = t_counter == 5 && t_process_seed == 17;
    t_counter = value;
    return fresh && t_counter == value;
}

DWORD WINAPI win32_worker(LPVOID) {
    ExitThread(check_worker_tls(11) ? 31 : 1);
}

unsigned __stdcall crt_worker(void*) {
    _endthreadex(check_worker_tls(13) ? 37 : 1);
    return 1;  // The CRT declaration allows a return; the worker must have exited above.
}

bool exits_with(HANDLE thread, DWORD expected) {
    if (!thread) return false;
    DWORD code = 0;
    const bool exited = WaitForSingleObject(thread, INFINITE) == WAIT_OBJECT_0 &&
                        GetExitCodeThread(thread, &code) && code == expected;
    CloseHandle(thread);
    return exited;
}

}  // namespace

#pragma section(".CRT$XLB", read)
#pragma comment(linker, "/include:tiny_game_tls_callback_entry")
extern "C" {
__declspec(allocate(".CRT$XLB")) PIMAGE_TLS_CALLBACK tiny_game_tls_callback_entry = tls_callback;
}

int main(int argc, char** argv) {
    std::printf("tiny_game: started, argc=%d\n", argc);
    check(argc == 3 && std::strcmp(argv[1], "hello") == 0 && std::strcmp(argv[2], "42") == 0,
          "argv is the game's command line (\"hello\" \"42\"), not gta6rt's");
    check(std::strstr(GetCommandLineA(), "hello 42") != nullptr, "GetCommandLineA is the game's command line");
    check(g_static.value == 1234, "a static initializer ran before main");

    std::vector<int> v(1 << 20);
    for (size_t i = 0; i < v.size(); ++i) v[i] = int(i);
    check(v[123456] == 123456, "a 4 MiB heap allocation works");

    char tmp[MAX_PATH] = {};
    GetTempPathA(MAX_PATH, tmp);
    std::string path = std::string(tmp) + "recity_tiny_game_" + std::to_string(GetCurrentProcessId()) + ".txt";
    bool file_ok = false;
    if (FILE* f = std::fopen(path.c_str(), "wb")) {
        std::fputs("recity", f);
        std::fclose(f);
        char back[16] = {};
        if (FILE* r = std::fopen(path.c_str(), "rb")) {
            std::fread(back, 1, 7, r);
            std::fclose(r);
            file_ok = std::strcmp(back, "recity") == 0;
        }
        std::remove(path.c_str());
    }
    check(file_ok, "a file round trip through the CRT works");

    long long sum = 0;
    std::thread worker([&] { for (int i = 1; i <= 1000; ++i) sum += i; });
    worker.join();
    check(sum == 500500, "a second thread ran and joined");

    check(XGameRuntimeInitialize() == S_OK, "XGameRuntimeInitialize is answered by the gta6rt shim (S_OK)");
    check(XSystemGetDeviceType() == 7, "XSystemGetDeviceType is answered by the gta6rt shim (Series X)");
    XGameRuntimeUninitialize();

    if (env_on("RECITY_RUN_TLS")) {
        const int main_initial = t_counter;
        check(t_process_seed == 23, "main sees the process-attach TLS write");
        int worker_initial = -1;
        int worker_updated = -1;
        int worker_seed = -1;
        std::thread t([&] {
            worker_initial = t_counter;
            worker_seed = t_process_seed;
            t_counter = 9;
            worker_updated = t_counter;
        });
        t.join();
        const int main_after_join = t_counter;
        std::printf("tiny_game: TLS main %d -> %d, worker %d -> %d\n",
                    main_initial, main_after_join, worker_initial, worker_updated);
        check(main_initial == 5, "main thread_local starts at 5");
        check(worker_initial == 5, "worker thread_local starts at 5");
        check(worker_seed == 17, "worker sees the TLS template value after process attach");
        check(worker_updated == 9, "worker thread_local reads back 9");
        check(main_after_join == 5, "main thread_local remains at 5 after worker joins");
        check(exits_with(CreateThread(nullptr, 0, win32_worker, nullptr, 0, nullptr), 31),
              "CreateThread gets fresh TLS and ExitThread preserves the exit code");
        check(g_win32_detaches.load(std::memory_order_relaxed) == 1, "ExitThread runs the TLS detach callback once");
        const auto crt_thread = reinterpret_cast<HANDLE>(_beginthreadex(nullptr, 0, crt_worker, nullptr, 0, nullptr));
        check(exits_with(crt_thread, 37), "_beginthreadex gets fresh TLS and _endthreadex preserves the exit code");
        check(g_crt_detaches.load(std::memory_order_relaxed) == 1, "_endthreadex runs the TLS detach callback once");
        check(t_counter == 5 && t_process_seed == 23, "explicit thread exits leave main TLS unchanged");
    } else {
        std::printf("tiny_game: SKIP thread_local (set RECITY_RUN_TLS=1 to enable)\n");
    }
    if (env_on("RECITY_RUN_EH")) {
        bool caught = false;
        try {
            throw std::string("boom");
        } catch (const std::string& e) {
            caught = e == "boom";
        }
        check(caught, "a C++ exception is thrown and caught inside the image");
    } else {
        std::printf("tiny_game: SKIP C++ exception (MSVC needs the image known to RtlPcToFileHeader; RECITY_RUN_EH=1)\n");
    }

    std::printf("tiny_game: %s\n", g_failures ? "FAILED" : "OK");
    std::fflush(stdout);
    std::exit(g_failures ? 1 : 0);  // through the shimmed exit(): ends the run, not gta6rt
}
