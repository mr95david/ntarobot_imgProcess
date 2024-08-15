cmake_minimum_required(VERSION 3.8)
project(ntarobot_camera)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Importe de paquetes requeridos
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)

# Include pkg include
include_directories(include)

# Instalacion de source para ejecucion c++ y python
ament_python_install_package(${PROJECT_NAME})

# Declaracion e ingreso de includes para paquetes de cpp
install(
  DIRECTORY include
  DESTINATION include
)

# Instalacion de paquetes 
install(
  DIRECTORY launch config models
  DESTINATION share/${PROJECT_NAME}
)

# Orden de creacion de ejecutables de nodos python
# install(PROGRAMS
#   ${PROJECT_NAME}/simple_controller.py
#   ${PROJECT_NAME}/noisy_controller.py
#   DESTINATION lib/${PROJECT_NAME}
# )

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
