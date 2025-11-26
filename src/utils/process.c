#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <limits.h>
#include <errno.h>

/* #include "include/log_utils.h" */ // Not used, replaced with printf
#include "utils/process.h"

static char g_exe_path[PATH_MAX] = {0};

const char* get_executable_path(void) {
    if (g_exe_path[0] == 0) {
        ssize_t len = readlink("/proc/self/exe", g_exe_path, sizeof(g_exe_path) - 1);
        if (len != -1) {
            g_exe_path[len] = '\0';
        } else {
            perror("readlink");
            return NULL;
        }
    }
    return g_exe_path;
}

int launch_worker(const char *test_id, const char *args, pid_t *pid_out, int *pipe_read_fd, int *pipe_write_fd) {
    const char *exe = get_executable_path();
    if (!exe) return -1;

    int pipe_fds[2] = {-1, -1};
    if (pipe_read_fd || pipe_write_fd) {
        if (pipe(pipe_fds) == -1) {
            perror("pipe");
            return -1;
        }
    }

    pid_t pid = fork();
    if (pid == -1) {
        perror("fork");
        if (pipe_fds[0] != -1) { close(pipe_fds[0]); close(pipe_fds[1]); }
        return -1;
    }

    if (pid == 0) {
        // Child process
        
        // Handle pipe ends
        if (pipe_read_fd || pipe_write_fd) {
            // If parent wants to read, child writes (and vice versa)
            // But usually we want a specific direction.
            // Let's assume if pipe_read_fd is requested by parent, parent reads, child writes.
            // If pipe_write_fd is requested by parent, parent writes, child reads.
            // This simple helper assumes a single pipe for now.
            // Let's standardize: 
            // If pipe_read_fd is passed, Parent Reads, Child Writes (stdout/custom).
            // Actually, let's just pass the FDs as args if needed?
            // Or simpler: dup2 to a standard FD?
            
            // For simplicity in this benchmark:
            // We will pass the pipe FDs as command line arguments to the worker if needed.
            // But execv closes non-cloexec FDs? No, they stay open.
            // We just need to know the number.
            
            // Let's close the ends we don't use.
            if (pipe_read_fd) {
                // Parent reads, Child writes.
                close(pipe_fds[0]); // Close read end
                // We can pass the write FD number to the child via args, 
                // OR we can dup2 it to a known FD (like 3).
                // Let's dup2 to 3 (if not already 3).
                if (pipe_fds[1] != 3) {
                    dup2(pipe_fds[1], 3);
                    close(pipe_fds[1]);
                }
            } else if (pipe_write_fd) {
                // Parent writes, Child reads.
                close(pipe_fds[1]); // Close write end
                if (pipe_fds[0] != 3) {
                    dup2(pipe_fds[0], 3);
                    close(pipe_fds[0]);
                }
            }
        }

        // Prepare arguments
        // argv[0] = exe
        // argv[1] = --worker
        // argv[2] = test_id
        // argv[3] = --worker-args
        // argv[4] = args (if present)
        // NULL
        
        char *argv[6];
        int argc = 0;
        argv[argc++] = (char*)exe;
        argv[argc++] = "--worker";
        argv[argc++] = (char*)test_id;
        
        if (args) {
            argv[argc++] = "--worker-args";
            argv[argc++] = (char*)args;
        }
        argv[argc++] = NULL;

        execv(exe, argv);
        perror("execv");
        exit(127);
    }

    // Parent process
    if (pipe_read_fd) {
        close(pipe_fds[1]); // Close write end
        *pipe_read_fd = pipe_fds[0];
    }
    if (pipe_write_fd) {
        close(pipe_fds[0]); // Close read end
        *pipe_write_fd = pipe_fds[1];
    }

    if (pid_out) *pid_out = pid;
    return 0;
}

int wait_for_worker(pid_t pid) {
    int status;
    if (waitpid(pid, &status, 0) == -1) {
        perror("waitpid");
        return -1;
    }
    if (WIFEXITED(status)) {
        return WEXITSTATUS(status);
    }
    return -1;
}
