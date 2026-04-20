configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
  )

add_lit_testsuite(check-scheduler
  "Running scheduler tests"
  ${CMAKE_CURRENT_BINARY_DIR})

add_dependencies(check-scheduler scheduler)

# Add check-all target that runs all tests
add_custom_target(check-all
  COMMENT "Running all tests")

add_dependencies(check-all check-scheduler)