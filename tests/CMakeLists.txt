add_executable(tensaxon_tests tests.cpp)

# Pull in include paths and the C++20 requirement from the interface library.
target_link_libraries(tensaxon_tests PRIVATE TensAxon::tensaxon)

# Strict compiler warnings.
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(tensaxon_tests PRIVATE
        -Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion -Wshadow
    )
elseif(MSVC)
    target_compile_options(tensaxon_tests PRIVATE /W4 /permissive-)
endif()

# Register the binary with CTest.
add_test(NAME TensAxonUnitTests COMMAND tensaxon_tests)
