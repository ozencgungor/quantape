# ============================================================================
# Shared target helpers for quantape executables.
# ============================================================================

include_guard(GLOBAL)

# Standard warning set (used by targets that had warnings before the split).
function(quantape_set_warnings target)
    if(MSVC)
        target_compile_options(${target} PRIVATE /W4)
    else()
        target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic)
        # GCC disables column tracking on huge headers (Stan) and prints a note
        # per TU even when those headers are system includes; turning the check
        # off removes the note. Clang has no misleading-indentation diagnostic.
        if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
            target_compile_options(${target} PRIVATE -Wno-misleading-indentation)
        endif()
    endif()
endfunction()

# Make Stan-backed executables find libtbb at runtime (macOS/Linux).
function(quantape_set_tbb_rpath target)
    if((APPLE OR UNIX) AND TARGET TBB::tbb)
        set_target_properties(${target} PROPERTIES
            INSTALL_RPATH "$<TARGET_FILE_DIR:TBB::tbb>"
            BUILD_WITH_INSTALL_RPATH TRUE)
        if(UNIX AND NOT APPLE)
            set_target_properties(${target} PROPERTIES
                INSTALL_RPATH "$ORIGIN/../lib:$<TARGET_FILE_DIR:TBB::tbb>")
        endif()
    endif()
endfunction()

# Thread-safe compile definitions required by Stan Math.
function(quantape_enable_stan_defs target)
    target_compile_definitions(${target} PRIVATE
        STAN_THREADS=true
        STAN_NO_RANGE_CHECKS
        _REENTRANT
        TBB_INTERFACE_NEW
    )
endfunction()

# Full Stan Math wiring: TBB + SUNDIALS (CVODES) + thread defs + rpath.
function(quantape_enable_stan target)
    target_link_libraries(${target} PUBLIC
        TBB::tbb
        SUNDIALS::nvecserial
        SUNDIALS::cvodes
    )
    quantape_enable_stan_defs(${target})
    quantape_set_tbb_rpath(${target})
endfunction()

# Register a test executable with CTest (executables remain runnable directly).
#
# Tests are numerical loops and run far too slowly unoptimized, so optimized
# configurations build them with -O2 on top of the configuration flags.
# Debug is never optimized: tests must be steppable like any other target.
function(quantape_register_test target)
    if(QUANTAPE_TEST_OPTIMIZATION)
        if(MSVC)
            target_compile_options(${target} PRIVATE $<$<NOT:$<CONFIG:Debug>>:/O2>)
        else()
            target_compile_options(${target} PRIVATE $<$<NOT:$<CONFIG:Debug>>:-O2>)
        endif()
    endif()
    add_test(NAME ${target} COMMAND ${target})
endfunction()

# Collect all executable targets of the current directory into a global list,
# so the root can define one build-all target for IDE build configurations.
function(quantape_collect_executables)
    get_property(_targets DIRECTORY PROPERTY BUILDSYSTEM_TARGETS)
    foreach(_target IN LISTS _targets)
        get_target_property(_type ${_target} TYPE)
        if(_type STREQUAL "EXECUTABLE")
            set_property(GLOBAL APPEND PROPERTY QUANTAPE_EXECUTABLE_TARGETS ${_target})
        endif()
    endforeach()
endfunction()
