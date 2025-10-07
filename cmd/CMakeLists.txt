
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#> main lib
find_package(Threads REQUIRED)
link_libraries(edges_curves_reconstruction)

add_executable(curvix_main main.cpp )

set_target_properties(curvix_main PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(curvix_main Threads::Threads)
