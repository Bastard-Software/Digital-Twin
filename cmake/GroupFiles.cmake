cmake_minimum_required(VERSION 3.24)

# Recursively find all header and source files in specified directories
file(GLOB_RECURSE INCLUDE_FILES ${INC_DIRS})
file(GLOB_RECURSE SOURCE_FILES ${SRC_DIRS})

# Combine all files into one list
set(ALL_FILES ${INCLUDE_FILES} ${SOURCE_FILES})

foreach(f ${ALL_FILES})
    # Get the file path relative to TWIN_DIR
    file(RELATIVE_PATH relative_path ${TWIN_DIR} ${f})
    
    # Extract only the directory part (without filename)
    get_filename_component(dir_path ${relative_path} DIRECTORY)
    
    # Check if dir_path is not empty before replacing separators
    if(NOT "${dir_path}" STREQUAL "")
        # Replace / with \ for Visual Studio
        string(REPLACE "/" "\\" group_name ${dir_path})
        
        # Files in subdirectories preserve their folder structure
        source_group("${group_name}" FILES ${f})
    else()
        # Files in root directory - no additional grouping
        source_group("" FILES ${f})
    endif()
endforeach()

# Add include directories for compilation
include_directories("${TWIN_DIR}")