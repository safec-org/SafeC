// driver.h — services each driver (host / wasm) provides to the core.
#pragma once
#include "scc.h"

// Loads a file from the host file system by path; returns its file index,
// or 0 if there is no such file (always 0 in the browser build).
int host_load_file(int path_sid);
// Searches the host's -I directories for <name>; 0 if not found.
int host_find_include(int name_sid);
// Bundled playground headers + runtime (vfs.sc, generated).
int vfs_load(int name_sid);
