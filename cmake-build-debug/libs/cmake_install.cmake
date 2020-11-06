# Install script for directory: C:/Users/Vergi/CLionProjects/GPGPUTasks2020/libs

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "C:/Program Files (x86)/libclew")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Debug")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("C:/Users/Vergi/CLionProjects/GPGPUTasks2020/cmake-build-debug/libs/clew/cmake_install.cmake")
  include("C:/Users/Vergi/CLionProjects/GPGPUTasks2020/cmake-build-debug/libs/gpu/cmake_install.cmake")
  include("C:/Users/Vergi/CLionProjects/GPGPUTasks2020/cmake-build-debug/libs/images/cmake_install.cmake")
  include("C:/Users/Vergi/CLionProjects/GPGPUTasks2020/cmake-build-debug/libs/utils/cmake_install.cmake")

endif()

