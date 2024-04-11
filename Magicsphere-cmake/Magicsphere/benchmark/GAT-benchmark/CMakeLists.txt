

file(GLOB_RECURSE CURRENT_HEADERS  sddmm_utils/*.h)
# file(GLOB CURRENT_SOURCES  src/*.cpp src/*.cu)
file(GLOB CURRENT_SOURCES  src/mGATKernel.cu)

add_library(MGAT1 STATIC ${CURRENT_HEADERS} ${CURRENT_SOURCES})
target_link_libraries(MGAT1 "${TORCH_LIBRARIES}" ${CUDA_LIBRARIES})

install(TARGETS MGAT1 ARCHIVE DESTINATION "benchmark/GAT-benchmark/lib")
install(FILES ${CURRENT_HEADERS} DESTINATION "benchmark/GAT-benchmark/include")
set_property(TARGET MGAT1 PROPERTY CXX_STANDARD 17)