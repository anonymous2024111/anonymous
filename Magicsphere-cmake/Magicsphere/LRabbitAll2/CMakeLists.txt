# add_compile_options(-std=c++17)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wcast-align -Wcast-qual -Wconversion -Wfloat-equal -Wformat=2 -Winit-self -Wmissing-declarations -Wmissing-include-dirs -Wpointer-arith -Wredundant-decls -Wswitch-default -Wuninitialized -Wwrite-strings -Wno-sign-conversion -Wno-unused-function -Wno-missing-declarations -fopenmp -mcx16 -O3 -DNDEBUG -fPIC")

find_package(TCMalloc)
find_package(Numa)

file(GLOB_RECURSE CURRENT_HEADERS  *.hpp)
file(GLOB CURRENT_SOURCES  reorder1.cpp)
message(${CURRENT_HEADERS} ${CURRENT_SOURCES})
add_library(MRabbitAll STATIC ${CURRENT_HEADERS} ${CURRENT_SOURCES})

# 添加 OpenMP 编译选项
target_link_libraries(MRabbitAll "${TORCH_LIBRARIES}" ${TCMALLOCLIB} ${NUMALIB})

install(TARGETS MRabbitAll ARCHIVE DESTINATION "LRabbitAll2/lib")
install(FILES ${CURRENT_HEADERS} DESTINATION "LRabbitAll2/include")
set_property(TARGET MRabbitAll PROPERTY CXX_STANDARD 17)