SET(TEST_SOURCES test_functions)

FOREACH (test_src ${TEST_SOURCES})
    ADD_EXECUTABLE(${test_src} ${test_src}.cpp)
    target_include_directories(${test_src} PRIVATE test_include) 
    TARGET_LINK_LIBRARIES(${test_src} edges_curves_reconstruction)
    ADD_TEST(${test_src} ${test_src})
ENDFOREACH (test_src)