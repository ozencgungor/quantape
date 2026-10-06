#ifndef QUANTAPE_TESTS_SUPPORT_BINARY_RUNNER_H
#define QUANTAPE_TESTS_SUPPORT_BINARY_RUNNER_H

// Fork/exec runner for tests that spawn built tool binaries (POSIX: macOS and
// Linux). There is deliberately no std::system() path: arguments are passed as
// an argv array, so paths with spaces or quotes survive unchanged, and stdout
// and stderr are captured separately.
//
// `status` follows the shell convention: the child's exit code for a normal
// exit, 128 + signal number when the child dies from a signal (a timeout kill
// therefore reports 137), and -1 when the process could not be spawned/reaped.

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <filesystem>
#include <string>
#include <vector>

#include <poll.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

struct RunResult {
    int status = -1;
    std::string out;
    std::string err;
};

namespace quantape::tests::detail {

/// One non-blocking-friendly read; appends data, clears `open` at EOF/error.
inline void appendAvailable(int fd, bool& open, std::string& sink) {
    char buffer[4096];
    const ssize_t count = ::read(fd, buffer, sizeof(buffer));
    if (count > 0) {
        sink.append(buffer, static_cast<std::size_t>(count));
    } else if (count == 0) {
        open = false;
    } else if (errno != EINTR && errno != EAGAIN && errno != EWOULDBLOCK) {
        open = false;
    }
}

} // namespace quantape::tests::detail

/// Runs `binary` with `args` (argv[0] is the binary path), capturing stdout
/// and stderr. Kills the child with SIGKILL when `timeout` elapses.
inline RunResult runTool(const std::filesystem::path& binary, const std::vector<std::string>& args,
                         std::chrono::seconds timeout = std::chrono::seconds(120)) {
    RunResult result;

    // Build argv before forking: between fork and exec the child may only call
    // async-signal-safe functions, so no allocation happens there.
    std::vector<std::string> storage;
    storage.reserve(args.size() + 1);
    storage.push_back(binary.string());
    storage.insert(storage.end(), args.begin(), args.end());
    std::vector<char*> argv;
    argv.reserve(storage.size() + 1);
    for (std::string& argument : storage) {
        argv.push_back(argument.data());
    }
    argv.push_back(nullptr);

    int outPipe[2] = {-1, -1};
    int errPipe[2] = {-1, -1};
    if (::pipe(outPipe) != 0) {
        return result;
    }
    if (::pipe(errPipe) != 0) {
        ::close(outPipe[0]);
        ::close(outPipe[1]);
        return result;
    }

    const pid_t pid = ::fork();
    if (pid < 0) {
        ::close(outPipe[0]);
        ::close(outPipe[1]);
        ::close(errPipe[0]);
        ::close(errPipe[1]);
        return result;
    }
    if (pid == 0) {
        ::close(outPipe[0]);
        ::close(errPipe[0]);
        if (::dup2(outPipe[1], STDOUT_FILENO) < 0 || ::dup2(errPipe[1], STDERR_FILENO) < 0) {
            ::_exit(126);
        }
        ::close(outPipe[1]);
        ::close(errPipe[1]);
        ::execv(binary.c_str(), argv.data());
        constexpr char kExecFailed[] = "runTool: execv failed\n";
        const ssize_t ignored = ::write(STDERR_FILENO, kExecFailed, sizeof(kExecFailed) - 1);
        static_cast<void>(ignored);
        ::_exit(127);
    }

    ::close(outPipe[1]);
    ::close(errPipe[1]);

    bool outOpen = true;
    bool errOpen = true;
    bool timedOut = false;
    const auto deadline = std::chrono::steady_clock::now() + timeout;

    while (outOpen || errOpen) {
        const auto now = std::chrono::steady_clock::now();
        if (now >= deadline) {
            timedOut = true;
            break;
        }
        struct pollfd watched[2];
        int count = 0;
        if (outOpen) {
            watched[count++] = {outPipe[0], POLLIN, 0};
        }
        if (errOpen) {
            watched[count++] = {errPipe[0], POLLIN, 0};
        }
        const auto remaining =
            std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now).count();
        const int ready = ::poll(watched, static_cast<nfds_t>(count),
                                 static_cast<int>(std::min<long long>(remaining, 1000)));
        if (ready < 0) {
            if (errno == EINTR) {
                continue;
            }
            break;
        }
        for (int i = 0; i < count; ++i) {
            if ((watched[i].revents & (POLLIN | POLLHUP | POLLERR)) == 0) {
                continue;
            }
            const bool isStdout = watched[i].fd == outPipe[0];
            quantape::tests::detail::appendAvailable(watched[i].fd, isStdout ? outOpen : errOpen,
                                                     isStdout ? result.out : result.err);
        }
    }

    if (timedOut) {
        ::kill(pid, SIGKILL);
    }
    int waitStatus = 0;
    while (::waitpid(pid, &waitStatus, 0) < 0) {
        if (errno != EINTR) {
            return result;
        }
    }

    // The child was the only writer: after reaping it, reads drain the buffer
    // and then return EOF without blocking.
    for (int i = 0; i < 2; ++i) {
        const int fd = i == 0 ? outPipe[0] : errPipe[0];
        bool& open = i == 0 ? outOpen : errOpen;
        std::string& sink = i == 0 ? result.out : result.err;
        while (open) {
            quantape::tests::detail::appendAvailable(fd, open, sink);
        }
        ::close(fd);
    }

    if (WIFEXITED(waitStatus)) {
        result.status = WEXITSTATUS(waitStatus);
    } else if (WIFSIGNALED(waitStatus)) {
        result.status = 128 + WTERMSIG(waitStatus);
    }
    return result;
}

#endif // QUANTAPE_TESTS_SUPPORT_BINARY_RUNNER_H
