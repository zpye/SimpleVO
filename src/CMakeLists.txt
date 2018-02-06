# source file
file(GLOB DIR_HEADER "*.h" "*.hpp")
file(GLOB DIR_SOURCE "*.cpp")

source_group(header FILES ${DIR_HEADER})
source_group(source FILES ${DIR_SOURCE})

# target
add_executable(SimpleVO ${DIR_SOURCE} ${DIR_HEADER})
target_link_libraries(SimpleVO ${CERES_LIBRARIES} ${OpenCV_LIBS} ${LIBRARIES})
if(MSVC)
target_compile_definitions(SimpleVO
    PRIVATE _CRT_NONSTDC_NO_WARNINGS _USE_MATH_DEFINES GOOGLE_GLOG_DLL_DECL=)
endif(MSVC)

add_subdirectory(test test)

if(NOT MSVC)
    add_subdirectory(evaluate evaluate)
endif()