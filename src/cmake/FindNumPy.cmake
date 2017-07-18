# Find folder where the NumPy header are located.
#
# NUMPY_INCLUDE_DIR - Location of NumPy C-API headers.

# Check if there's a bin directory corresponding to the PythonLib that is found.
find_package(PythonLibs REQUIRED)
find_program(PYTHON_EXECUTABLE NAMES python HINTS ${PYTHON_LIBRARY}/../bin ${PYTHON_INCLUDE_DIR}/../bin)

# If not find the system Python interpreter.
if     (PYTHON_EXECUTABLE)
else   (PYTHON_EXECUTABLE)
  find_package(PythonInterp REQUIRED)
endif  (PYTHON_EXECUTABLE)

execute_process(
  COMMAND ${PYTHON_EXECUTABLE} find_numpy.py
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/cmake
  OUTPUT_VARIABLE NUMPY_INCLUDE_DIR
  ERROR_VARIABLE  ERROR
)
