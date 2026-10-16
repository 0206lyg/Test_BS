cmake_minimum_required(VERSION 3.20)
project(slurry_lammps_plugin VERSION 0.1.0 LANGUAGES CXX)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
  message(FATAL_ERROR "Build outside the source directory (use cmake -B build).")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(SLURRY_ENABLE_LU "Build Eigen full and incomplete Jacobian preconditioners" ON)
add_library(slurry_solver_options INTERFACE)
if(SLURRY_ENABLE_LU)
  set(SLURRY_EIGEN_DIR "${CMAKE_CURRENT_SOURCE_DIR}/vendor/eigen3")
  if(NOT EXISTS "${SLURRY_EIGEN_DIR}/Eigen/SparseLU")
    message(FATAL_ERROR "The source bundle is missing vendor/eigen3/Eigen/SparseLU.")
  endif()
  target_include_directories(slurry_solver_options SYSTEM INTERFACE "${SLURRY_EIGEN_DIR}")
  target_compile_definitions(slurry_solver_options INTERFACE SLURRY_HAVE_EIGEN=1 EIGEN_DONT_PARALLELIZE)
endif()

find_package(LAMMPS CONFIG REQUIRED)
set(LAMMPS_SOURCE_DIR "" CACHE PATH "The pristine LAMMPS source directory")
if(NOT EXISTS "${LAMMPS_SOURCE_DIR}/src/lammpsplugin.h")
  message(FATAL_ERROR "Set LAMMPS_SOURCE_DIR to the matching LAMMPS source checkout.")
endif()

set(SLURRY_KERNEL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/vendor/gr_re2")
foreach(SLURRY_HEADER particleMath.h ellipsoidGap.h nearFieldResistance.h)
  if(NOT EXISTS "${SLURRY_KERNEL_DIR}/${SLURRY_HEADER}")
    message(FATAL_ERROR "Incomplete slurry source bundle: missing ${SLURRY_KERNEL_DIR}/${SLURRY_HEADER}. Re-extract the complete bundle.")
  endif()
endforeach()
add_library(slurry MODULE
  src/plugin.cpp
  src/pair_slurry_spheroid.cpp
  src/fix_slurry_drag.cpp
  src/fix_slurry_implicit.cpp)
target_link_libraries(slurry PRIVATE LAMMPS::lammps slurry_solver_options)
target_include_directories(slurry PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/src"
  "${LAMMPS_SOURCE_DIR}/src"
  "${LAMMPS_SOURCE_DIR}/src/ASPHERE"
  "${SLURRY_KERNEL_DIR}")
# Serial LAMMPS installations do not export the bundled MPI stub header.
get_target_property(SLURRY_LAMMPS_LINK_LIBRARIES LAMMPS::lammps INTERFACE_LINK_LIBRARIES)
if(NOT "${SLURRY_LAMMPS_LINK_LIBRARIES}" MATCHES "MPI::")
  target_include_directories(slurry PRIVATE "${LAMMPS_SOURCE_DIR}/src/STUBS")
endif()
set_target_properties(slurry PROPERTIES
  PREFIX "lib"
  OUTPUT_NAME "slurry"
  SUFFIX ".so")
install(TARGETS slurry LIBRARY DESTINATION lib/lammps/plugins)

# Geometry-only preparation. This executable does not link to MPI/LAMMPS and
# its artificial relaxation steps have no physical time interpretation.
add_executable(slurry_pack src/pack_spheroids.cpp)
install(TARGETS slurry_pack RUNTIME DESTINATION bin)

include(CTest)
if(BUILD_TESTING)
  if(SLURRY_ENABLE_LU)
    add_executable(slurry_eigen_lu_solver_test tests/test_eigen_lu_solver.cpp)
    target_include_directories(slurry_eigen_lu_solver_test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(slurry_eigen_lu_solver_test PRIVATE slurry_solver_options)
    add_test(NAME slurry_exact_sparse_lu COMMAND slurry_eigen_lu_solver_test)
    add_test(NAME slurry_exact_jacobian_ilu COMMAND slurry_eigen_lu_solver_test ilu)
    add_test(NAME slurry_adaptive_forcing_ilu COMMAND slurry_eigen_lu_solver_test ilu adaptive)
    add_test(NAME slurry_cached_ilu_equivalence COMMAND slurry_eigen_lu_solver_test ilu cache)
    add_executable(slurry_eigen_ilu_solver_test tests/test_eigen_ilu_solver.cpp)
    target_include_directories(slurry_eigen_ilu_solver_test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(slurry_eigen_ilu_solver_test PRIVATE slurry_solver_options)
    add_test(NAME slurry_incomplete_sparse_lu COMMAND slurry_eigen_ilu_solver_test)
  endif()
  add_executable(slurry_sparse_solver_test tests/test_sparse_solver.cpp)
  target_link_libraries(slurry_sparse_solver_test PRIVATE slurry_solver_options)
  target_include_directories(slurry_sparse_solver_test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
  add_test(NAME slurry_sparse_preconditioner COMMAND slurry_sparse_solver_test)
  add_executable(slurry_spheroid_hydro_math_test tests/test_spheroid_hydro_math.cpp)
  target_include_directories(slurry_spheroid_hydro_math_test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src" "${SLURRY_KERNEL_DIR}")
  add_test(NAME slurry_spheroid_lubrication_algebra COMMAND slurry_spheroid_hydro_math_test)
  add_executable(slurry_pair_hydro_math_test tests/test_pair_hydro_math.cpp)
  target_include_directories(slurry_pair_hydro_math_test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
  add_test(NAME slurry_mari_pair_lubrication COMMAND slurry_pair_hydro_math_test)
  add_executable(slurry_hydro_math_test tests/test_hydro_math.cpp)
  target_include_directories(slurry_hydro_math_test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
  add_test(NAME slurry_hydro_analytic_limits COMMAND slurry_hydro_math_test)
  add_test(NAME slurry_packing_gradients COMMAND slurry_pack --self-test)
  add_executable(slurry_implicit_math_test tests/test_implicit_math.cpp)
  target_link_libraries(slurry_implicit_math_test PRIVATE slurry_solver_options)
  target_include_directories(slurry_implicit_math_test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
  add_test(NAME slurry_implicit_resistance_contact COMMAND slurry_implicit_math_test)
  add_executable(slurry_stress_math_test tests/test_stress_math.cpp)
  target_include_directories(slurry_stress_math_test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
  add_test(NAME slurry_stress_analytic_limits COMMAND slurry_stress_math_test)
  add_executable(slurry_history_math_test tests/test_history_math.cpp)
  target_include_directories(slurry_history_math_test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src" "${SLURRY_KERNEL_DIR}")
  add_test(NAME slurry_history_objectivity COMMAND slurry_history_math_test)
  add_executable(slurry_friction_regression_test tests/test_friction_regression.cpp)
  target_link_libraries(slurry_friction_regression_test PRIVATE slurry_solver_options)
  target_include_directories(slurry_friction_regression_test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
  add_test(NAME slurry_dense_friction_kink COMMAND slurry_friction_regression_test
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/data/dense_friction_kink.dat")
  add_test(NAME slurry_dense_friction_stall COMMAND slurry_friction_regression_test
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/data/dense_friction_stall.dat")
  if(SLURRY_ENABLE_LU)
    add_test(NAME slurry_dense_friction_kink_lu COMMAND slurry_friction_regression_test
      "${CMAKE_CURRENT_SOURCE_DIR}/tests/data/dense_friction_kink.dat" lu)
    add_test(NAME slurry_dense_friction_stall_lu COMMAND slurry_friction_regression_test
      "${CMAKE_CURRENT_SOURCE_DIR}/tests/data/dense_friction_stall.dat" lu)
    add_test(NAME slurry_dense_friction_kink_ilu COMMAND slurry_friction_regression_test
      "${CMAKE_CURRENT_SOURCE_DIR}/tests/data/dense_friction_kink.dat" ilu)
    add_test(NAME slurry_dense_friction_stall_ilu COMMAND slurry_friction_regression_test
      "${CMAKE_CURRENT_SOURCE_DIR}/tests/data/dense_friction_stall.dat" ilu)
  endif()
endif()
