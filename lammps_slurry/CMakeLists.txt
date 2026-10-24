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

# Production bridge backends. OFF is an explicit developer/reference build;
# normal build.py always requests both optimized backends.
option(SLURRY_USE_FFTW "Use FFTW for the periodic bridge mobility" ON)
option(SLURRY_USE_MKL_PARDISO "Use threaded oneMKL PARDISO for bridge preconditioning" ON)
set(SLURRY_FFTW_ROOT "" CACHE PATH "FFTW installation prefix (optional)")
set(SLURRY_MKL_ROOT "" CACHE PATH "oneMKL installation prefix (optional)")
add_library(slurry_bridge_numerics INTERFACE)
target_link_libraries(slurry_bridge_numerics INTERFACE slurry_solver_options)
set(SLURRY_FFT_BACKEND "builtin-reference")
set(SLURRY_SPARSE_BACKEND "Eigen-reference")
if(SLURRY_ENABLE_LU)
  find_package(OpenMP REQUIRED COMPONENTS CXX)
  find_package(Threads REQUIRED)
  target_link_libraries(slurry_bridge_numerics INTERFACE OpenMP::OpenMP_CXX)
  if(SLURRY_USE_FFTW)
    find_path(SLURRY_FFTW_INCLUDE_DIR fftw3.h
      HINTS "${SLURRY_FFTW_ROOT}" "$ENV{FFTW_ROOT}" "$ENV{RCAC_FFTW_ROOT}"
      PATH_SUFFIXES include)
    find_library(SLURRY_FFTW_LIBRARY NAMES fftw3
      HINTS "${SLURRY_FFTW_ROOT}" "$ENV{FFTW_ROOT}" "$ENV{RCAC_FFTW_ROOT}"
      PATH_SUFFIXES lib lib64 lib/x86_64-linux-gnu)
    find_library(SLURRY_FFTW_THREADS_LIBRARY NAMES fftw3_omp fftw3_threads
      HINTS "${SLURRY_FFTW_ROOT}" "$ENV{FFTW_ROOT}" "$ENV{RCAC_FFTW_ROOT}"
      PATH_SUFFIXES lib lib64 lib/x86_64-linux-gnu)
    if(NOT SLURRY_FFTW_INCLUDE_DIR OR NOT SLURRY_FFTW_LIBRARY OR NOT SLURRY_FFTW_THREADS_LIBRARY)
      message(FATAL_ERROR "Production bridge requires FFTW headers, fftw3 and fftw3_omp (or fftw3_threads). Load the FFTW module or set SLURRY_FFTW_ROOT. No slow fallback was selected.")
    endif()
    target_compile_definitions(slurry_bridge_numerics INTERFACE SLURRY_HAVE_FFTW=1)
    target_include_directories(slurry_bridge_numerics SYSTEM INTERFACE "${SLURRY_FFTW_INCLUDE_DIR}")
    target_link_libraries(slurry_bridge_numerics INTERFACE
      "${SLURRY_FFTW_THREADS_LIBRARY}" "${SLURRY_FFTW_LIBRARY}" Threads::Threads)
    set(SLURRY_FFT_BACKEND "FFTW")
  endif()
  if(SLURRY_USE_MKL_PARDISO)
    # Static MKL includes CPU-dispatch kernels in the plugin itself. Dynamic
    # mkl_rt dlopens kernels invisible to ldd, defeating a private runtime copy.
    # GNU threading shares libgomp with the plugin and FFTW OpenMP backend.
    if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
      message(FATAL_ERROR "This production configuration uses MKL GNU threading; configure with the GCC compiler (Negishi: gcc/12.2.0).")
    endif()
    find_path(SLURRY_MKL_INCLUDE_DIR mkl.h
      HINTS "${SLURRY_MKL_ROOT}" "$ENV{MKLROOT}" "$ENV{RCAC_INTEL_MKL_ROOT}"
      PATH_SUFFIXES include mkl/include)
    set(_slurry_saved_suffixes "${CMAKE_FIND_LIBRARY_SUFFIXES}")
    set(CMAKE_FIND_LIBRARY_SUFFIXES .a)
    foreach(_part intel_lp64 gnu_thread core)
      find_library(SLURRY_MKL_${_part}_LIBRARY NAMES mkl_${_part}
        HINTS "${SLURRY_MKL_ROOT}" "$ENV{MKLROOT}" "$ENV{RCAC_INTEL_MKL_ROOT}"
        PATH_SUFFIXES lib lib/intel64 lib/intel64_lin lib64 mkl/lib/intel64)
    endforeach()
    set(CMAKE_FIND_LIBRARY_SUFFIXES "${_slurry_saved_suffixes}")
    if(NOT SLURRY_MKL_INCLUDE_DIR OR NOT SLURRY_MKL_intel_lp64_LIBRARY OR
       NOT SLURRY_MKL_gnu_thread_LIBRARY OR NOT SLURRY_MKL_core_LIBRARY)
      message(FATAL_ERROR "Production bridge requires static oneMKL LP64/GNU-thread/core libraries and headers. Load intel-mkl or set SLURRY_MKL_ROOT/MKLROOT to a full installation. No sequential fallback was selected.")
    endif()
    # Keep GNU OpenMP after the static MKL archives even when --as-needed
    # drops CMake's earlier libgomp entry (e.g. a PARDISO-only test program).
    # The GCC driver places its OpenMP runtime libraries at the end of linking.
    target_link_options(slurry_bridge_numerics INTERFACE -fopenmp)
    target_compile_definitions(slurry_bridge_numerics INTERFACE SLURRY_HAVE_MKL_PARDISO=1)
    target_include_directories(slurry_bridge_numerics SYSTEM INTERFACE "${SLURRY_MKL_INCLUDE_DIR}")
    target_link_libraries(slurry_bridge_numerics INTERFACE
      "-Wl,--start-group" "${SLURRY_MKL_intel_lp64_LIBRARY}"
      "${SLURRY_MKL_gnu_thread_LIBRARY}" "${SLURRY_MKL_core_LIBRARY}"
      "-Wl,--end-group" Threads::Threads "${CMAKE_DL_LIBS}" m)
    set(SLURRY_SPARSE_BACKEND "MKL-PARDISO-static-GNU-LP64")
  endif()
endif()
message(STATUS "Slurry bridge FFT backend: ${SLURRY_FFT_BACKEND}")
message(STATUS "Slurry bridge sparse backend: ${SLURRY_SPARSE_BACKEND}")
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/slurry-numerics.json"
  "{\n  \"fft\": \"${SLURRY_FFT_BACKEND}\",\n  \"sparse\": \"${SLURRY_SPARSE_BACKEND}\"\n}\n")

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
if(SLURRY_ENABLE_LU)
  target_sources(slurry PRIVATE src/fix_slurry_bridge.cpp)
endif()
target_link_libraries(slurry PRIVATE LAMMPS::lammps slurry_bridge_numerics)
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
target_include_directories(slurry_pack SYSTEM PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/vendor/eigen3")
if(OpenMP_CXX_FOUND)
  target_link_libraries(slurry_pack PRIVATE OpenMP::OpenMP_CXX)
endif()
install(TARGETS slurry_pack RUNTIME DESTINATION bin)

# Offline generation only. Runtime finite_gap_fast loads the resulting table;
# it does not call this executable or its Stokes/film evaluators per timestep.
if(SLURRY_ENABLE_LU)
  # Bind numerical validation reports to the headers actually compiled into
  # each probe. A current-tree hash plus an unrelated executable hash is not
  # sufficient evidence that a standalone probe was rebuilt.
  find_package(Python3 3.6 REQUIRED COMPONENTS Interpreter)
  file(GLOB SLURRY_NUMERICAL_INPUTS CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/vendor/gr_re2/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/tools/basis_*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/tools/slurry_generate_basis.cpp")
  list(APPEND SLURRY_NUMERICAL_INPUTS
    "${CMAKE_CURRENT_SOURCE_DIR}/tools/precompute_fast.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/tools/prepare_case.py")
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${SLURRY_NUMERICAL_INPUTS})
  execute_process(COMMAND "${Python3_EXECUTABLE}" -c
    "import importlib.util,sys; s=importlib.util.spec_from_file_location('slurry_prepare',sys.argv[1]); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); print(m.fast_source_fingerprint())"
    "${CMAKE_CURRENT_SOURCE_DIR}/tools/prepare_case.py"
    OUTPUT_VARIABLE SLURRY_NUMERICAL_SOURCE_FINGERPRINT OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE SLURRY_FINGERPRINT_STATUS)
  string(LENGTH "${SLURRY_NUMERICAL_SOURCE_FINGERPRINT}" SLURRY_FINGERPRINT_LENGTH)
  if(NOT SLURRY_FINGERPRINT_STATUS EQUAL 0 OR NOT SLURRY_FINGERPRINT_LENGTH EQUAL 64 OR
     NOT SLURRY_NUMERICAL_SOURCE_FINGERPRINT MATCHES "^[0-9a-f]+$")
    message(FATAL_ERROR "Could not fingerprint numerical sources for validation probes.")
  endif()
  add_executable(slurry_precompute_fast tools/precompute_fast.cpp)
  target_include_directories(slurry_precompute_fast PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
  target_link_libraries(slurry_precompute_fast PRIVATE slurry_solver_options)
  find_package(OpenMP QUIET COMPONENTS CXX)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(slurry_precompute_fast PRIVATE OpenMP::OpenMP_CXX)
  endif()
  install(TARGETS slurry_precompute_fast RUNTIME DESTINATION bin)
  add_executable(slurry_generate_basis tools/slurry_generate_basis.cpp)
  target_include_directories(slurry_generate_basis PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
  target_link_libraries(slurry_generate_basis PRIVATE slurry_solver_options)
  install(TARGETS slurry_generate_basis RUNTIME DESTINATION bin)
  add_executable(slurry_basis_physical_probe tests/test_basis_physical.cpp)
  target_include_directories(slurry_basis_physical_probe PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
  target_link_libraries(slurry_basis_physical_probe PRIVATE slurry_solver_options)
  target_compile_definitions(slurry_basis_physical_probe PRIVATE
    "SLURRY_NUMERICAL_SOURCE_FINGERPRINT=\"${SLURRY_NUMERICAL_SOURCE_FINGERPRINT}\"")
  install(TARGETS slurry_basis_physical_probe RUNTIME DESTINATION bin)
  add_executable(slurry_table_interpolation_probe tests/test_table_interpolation.cpp)
  target_include_directories(slurry_table_interpolation_probe PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
  target_link_libraries(slurry_table_interpolation_probe PRIVATE slurry_solver_options)
  target_compile_definitions(slurry_table_interpolation_probe PRIVATE
    "SLURRY_NUMERICAL_SOURCE_FINGERPRINT=\"${SLURRY_NUMERICAL_SOURCE_FINGERPRINT}\"")
  install(TARGETS slurry_table_interpolation_probe RUNTIME DESTINATION bin)
endif()

include(CTest)
if(BUILD_TESTING)
  if(SLURRY_ENABLE_LU AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/models/ar10_matched.ssb")
    add_executable(slurry_matched_hybrid_test tests/test_matched_hybrid.cpp)
    target_include_directories(slurry_matched_hybrid_test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(slurry_matched_hybrid_test PRIVATE slurry_solver_options)
    add_test(NAME slurry_matched_hybrid_golden COMMAND slurry_matched_hybrid_test
      "${CMAKE_CURRENT_SOURCE_DIR}/models/ar10_matched.ssb"
      "${CMAKE_CURRENT_SOURCE_DIR}/tests/data/matched_hybrid_golden.txt")
    add_executable(slurry_independent_pair_reference_test tests/test_independent_pair_reference.cpp)
    target_include_directories(slurry_independent_pair_reference_test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(slurry_independent_pair_reference_test PRIVATE slurry_solver_options)
    add_test(NAME slurry_independent_stokes_reference COMMAND slurry_independent_pair_reference_test
      "${CMAKE_CURRENT_SOURCE_DIR}/models/ar10_matched.ssb"
      "${CMAKE_CURRENT_SOURCE_DIR}/tests/data/independent_pair_reference.txt")
  endif()
  foreach(SLURRY_GRAND_TEST grand_pair_math grand_pair_lookup grand_reduced)
    add_executable(slurry_${SLURRY_GRAND_TEST}_test tests/test_${SLURRY_GRAND_TEST}.cpp)
    target_include_directories(slurry_${SLURRY_GRAND_TEST}_test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
    add_test(NAME slurry_${SLURRY_GRAND_TEST} COMMAND slurry_${SLURRY_GRAND_TEST}_test)
  endforeach()
  if(SLURRY_ENABLE_LU)
    add_executable(slurry_grand_system_check_test tests/test_grand_system_check.cpp)
    target_include_directories(slurry_grand_system_check_test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(slurry_grand_system_check_test PRIVATE slurry_solver_options)
    add_test(NAME slurry_grand_system_passivity COMMAND slurry_grand_system_check_test)
  endif()
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

# Compact bridge path: no offline basis or resistance-table generation.
if(SLURRY_ENABLE_LU)
  find_package(OpenMP QUIET COMPONENTS CXX)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(slurry PRIVATE OpenMP::OpenMP_CXX)
  endif()
  if(BUILD_TESTING)
    foreach(BRIDGE_TEST compact_bridge bridge_images bridge_solver bridge_step bridge_flake cylinder_geometry cylinder_film cylinder_pair periodic_mobility periodic_mesh contact_condensation bridge_workspace)
      if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_${BRIDGE_TEST}.cpp")
        add_executable(slurry_${BRIDGE_TEST}_test tests/test_${BRIDGE_TEST}.cpp)
        target_include_directories(slurry_${BRIDGE_TEST}_test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
        target_link_libraries(slurry_${BRIDGE_TEST}_test PRIVATE slurry_bridge_numerics)
        if(OpenMP_CXX_FOUND)
          target_link_libraries(slurry_${BRIDGE_TEST}_test PRIVATE OpenMP::OpenMP_CXX)
        endif()
        if(BRIDGE_TEST STREQUAL "compact_bridge")
          add_test(NAME slurry_${BRIDGE_TEST} COMMAND slurry_${BRIDGE_TEST}_test "${CMAKE_CURRENT_SOURCE_DIR}/tests/data/compact_bridge_v4.dat")
        else()
          add_test(NAME slurry_${BRIDGE_TEST} COMMAND slurry_${BRIDGE_TEST}_test)
        endif()
      endif()
    endforeach()
  endif()
endif()

if(BUILD_TESTING)
  find_package(Python3 3.6 REQUIRED COMPONENTS Interpreter)
  add_test(NAME slurry_bridge_tools COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_bridge_tools.py")
  set_tests_properties(slurry_bridge_tools PROPERTIES ENVIRONMENT "SLURRY_TEST_PACKER=$<TARGET_FILE:slurry_pack>")
  add_test(NAME slurry_packing_clearance COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_packing_clearance.py")
  set_tests_properties(slurry_packing_clearance PROPERTIES ENVIRONMENT "SLURRY_TEST_PACKER=$<TARGET_FILE:slurry_pack>")
  add_test(NAME slurry_runtime_link_paths COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_runtime_link_paths.py")
endif()
