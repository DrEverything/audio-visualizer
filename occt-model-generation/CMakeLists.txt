cmake_minimum_required(VERSION 3.15)
project(OpenCascadeExample CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 1. Point CMake to the OpenCASCADE directory you downloaded (must be the cmake subfolder for CMake Config file)
set(OpenCASCADE_DIR "C:/libs/occt/opencascade-8.0.0-vc14-64/cmake")

# Create dummy VTK targets to satisfy OpenCASCADE's internal dependency checks.
# Since we only use core CAD modules (not VTK visualization), these dummy targets allow
# CMake to configure successfully without needing to download or install the massive VTK library.
set(VTK_DUMMY_TARGETS
    VTK::CommonCore
    VTK::RenderingCore
    VTK::RenderingFreeType
    VTK::FiltersGeneral
    VTK::InteractionStyle
    VTK::RenderingOpenGL2
    VTK::RenderingGL2PSOpenGL2
    VTK::IOImage
    VTK::ImagingCore
)
foreach(target ${VTK_DUMMY_TARGETS})
    if(NOT TARGET ${target})
        add_library(${target} INTERFACE IMPORTED)
    endif()
endforeach()

# 2. Find OpenCASCADE
find_package(OpenCASCADE REQUIRED)

add_executable(occt_example main.cpp)

# 3. Include and Link OpenCASCADE
target_include_directories(occt_example PRIVATE ${OpenCASCADE_INCLUDE_DIR})
target_link_libraries(occt_example PRIVATE ${OpenCASCADE_LIBRARIES})

add_custom_target(run
    COMMAND $<TARGET_FILE:occt_example>
    DEPENDS occt_example
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running the compiled executable..."
)
