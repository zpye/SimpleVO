# source file
file(GLOB evaluate_HEADER "*.h" "*.hpp")
file(GLOB evaluate_SOURCE "*.cpp")

source_group(header FILES ${evaluate_HEADER})
source_group(source FILES ${evaluate_SOURCE})

# target
add_executable(evaluate_odometry ${evaluate_SOURCE} ${evaluate_HEADER})
