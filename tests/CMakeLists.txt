add_executable(tensaxon_tests tests.cpp)

# Pull in include paths and the C++20 requirement from the interface library.
target_link_libraries(tensaxon_tests PRIVATE TensAxon::tensaxon)

# Strict compiler warnings.
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(tensaxon_tests PRIVATE
        -Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion -Wshadow -Werror
    )
    if(TENSAXON_ENABLE_SANITIZERS)
        target_compile_options(tensaxon_tests PRIVATE
            -fsanitize=address,undefined -fno-omit-frame-pointer -fno-sanitize-recover=all
        )
        target_link_options(tensaxon_tests PRIVATE
            -fsanitize=address,undefined
        )
    endif()
elseif(MSVC)
    target_compile_options(tensaxon_tests PRIVATE /W4 /WX /permissive-)
endif()

# Register the binary with CTest.
add_test(NAME TensAxonUnitTests COMMAND tensaxon_tests)
