add_executable(tensaxon_example basic.cpp)
target_link_libraries(tensaxon_example PRIVATE TensAxon::tensaxon)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(tensaxon_example PRIVATE
        -Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion -Wshadow -Werror
    )
    if(TENSAXON_ENABLE_SANITIZERS)
        target_compile_options(tensaxon_example PRIVATE
            -fsanitize=address,undefined -fno-omit-frame-pointer -fno-sanitize-recover=all
        )
        target_link_options(tensaxon_example PRIVATE
            -fsanitize=address,undefined
        )
    endif()
elseif(MSVC)
    target_compile_options(tensaxon_example PRIVATE /W4 /WX /permissive- /Zc:preprocessor)
endif()
