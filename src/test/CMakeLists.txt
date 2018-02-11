# source file
set(test_HEADER "../local_parameterization_se3.hpp")
set(test_SOURCE "test_ceres_se3.cpp")

source_group(header FILES ${test_HEADER})
source_group(source FILES ${test_SOURCE})

# target
add_executable(test_ceres_se3 ${test_SOURCE} ${test_HEADER})
target_link_libraries(test_ceres_se3 ${CERES_LIBRARIES} ${LIBRARIES})
if(MSVC)
target_compile_definitions(test_ceres_se3 
    PRIVATE _MSC_VER _CRT_NONSTDC_NO_WARNINGS _USE_MATH_DEFINES GOOGLE_GLOG_DLL_DECL=)
endif(MSVC)