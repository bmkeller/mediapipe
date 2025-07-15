cc_import(
    name = "luxonis_depthai_lib",
    static_library = "build/libdepthai-core.a",
)

cc_import(
    name = "luxonis_depthai_resources_lib",
    static_library = "build/libdepthai-resources.a",
)

cc_import(
    name = "luxonis_depthai_core_dylib",
    shared_library = "build/libdepthai-core.dylib",
)

# Define the library with headers
cc_library(
    name = "luxonis_depthai",
    hdrs = 
        glob(["src/mkel_wrappers/**/*.h"]) + 
        glob(["include/depthai/**/*.hpp"]) + 
        glob(["shared/depthai-shared/include/depthai-shared/**/*.hpp"]),
    includes = [ 
        "src/mkel_wrappers/", 
        "include/depthai/", 
        "shared/depthai-shared/include/"
    ],
    #deps = [":luxonis_depthai_lib", ":luxonis_depthai_resources_lib"],
    deps = [":luxonis_depthai_core_dylib"],
    visibility = ["//visibility:public"],
)
