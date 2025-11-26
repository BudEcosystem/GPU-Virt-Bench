#ifndef GPU_VIRT_BENCH_PROCESS_H
#define GPU_VIRT_BENCH_PROCESS_H

#include <sys/types.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Launch a worker process of this same executable.
 * 
 * @param test_id The test ID to run (e.g., "IS-005-CHILD")
 * @param args Additional arguments string (can be NULL)
 * @param pid_out Pointer to store the child PID
 * @param pipe_read_fd Pointer to store the read end of a pipe (if not NULL)
 * @param pipe_write_fd Pointer to store the write end of a pipe (if not NULL)
 * @return 0 on success, -1 on failure
 */
int launch_worker(const char *test_id, const char *args, pid_t *pid_out, int *pipe_read_fd, int *pipe_write_fd);

/*
 * Wait for a child process to exit.
 * 
 * @param pid The PID to wait for
 * @return Exit code of the child, or -1 on error
 */
int wait_for_worker(pid_t pid);

/*
 * Get the path to the current executable.
 */
const char* get_executable_path(void);

#ifdef __cplusplus
}
#endif

#endif // GPU_VIRT_BENCH_PROCESS_H
