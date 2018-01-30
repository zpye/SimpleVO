# source file
file(GLOB DIR_HEADER "*.h" "*.hpp")
file(GLOB DIR_SOURCE "*.cpp")

source_group(header FILES ${DIR_HEADER})
source_group(source FILES ${DIR_SOURCE})

# target
add_executable(SimpleVO ${DIR_SOURCE} ${DIR_HEADER})

add_subdirectory(test test)

if(NOT MSVC)
    add_subdirectory(evaluate evaluate)
endif()