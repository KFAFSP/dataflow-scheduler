add_lit_testsuite(check-scheduler-python 
  "Running the scheduler Python regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS KTIRPythonModules
)

add_dependencies(check-all check-scheduler-python)
