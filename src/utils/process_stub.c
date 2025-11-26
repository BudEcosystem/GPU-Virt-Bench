/* Stub implementations for process functions */
#include <stddef.h>
#include "utils/process.h"

int launch_worker(const char *test_id, const char *args, pid_t *pid_out, int *pipe_read_fd, int *pipe_write_fd) {
    (void)test_id; (void)args; (void)pid_out; (void)pipe_read_fd; (void)pipe_write_fd;
    return -1; /* Not implemented - multi-process tests will be skipped */
}

int wait_for_worker(pid_t pid) {
    (void)pid;
    return -1; /* Not implemented */
}

const char* get_executable_path(void) {
    return NULL;
}
