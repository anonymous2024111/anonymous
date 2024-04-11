

file(GLOB_RECURSE CURRENT_HEADERS  spmm_utils/*.h)
# file(GLOB CURRENT_SOURCES  src/*.cpp src/*.cu)
file(GLOB CURRENT_SOURCES  src/mGCNKernel.cu)

add_library(MGCN1 STATIC ${CURRENT_HEADERS} ${CURRENT_SOURCES})
target_link_libraries(MGCN1 "${TORCH_LIBRARIES}" ${CUDA_LIBRARIES})

install(TARGETS MGCN1 ARCHIVE DESTINATION "benchmark/GCN-benchmark/lib")
install(FILES ${CURRENT_HEADERS} DESTINATION "benchmark/GCN-benchmark/include")
set_property(TARGET MGCN1 PROPERTY CXX_STANDARD 17)