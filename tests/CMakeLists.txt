add_executable(linalg_tests tests.cpp)

# Подтягиваем пути к хедерам и стандарт C++ из интерфейсной библиотеки
target_link_libraries(linalg_tests PRIVATE LinAlg::linalg)

# Строгие флаги предупреждений компилятора
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(linalg_tests PRIVATE 
        -Wall -Wextra -Wpedantic -Wconversion -Wshadow
    )
elseif(MSVC)
    target_compile_options(linalg_tests PRIVATE /W4 /permissive-)
endif()

# Регистрируем бинарник в инфраструктуре CTest
add_test(NAME LinAlgUnitTests COMMAND linalg_tests)