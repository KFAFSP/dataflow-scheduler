# FIXME: Either evict this library or fix its linking.

add_ktir_library(KTIRSchedulerDriverLib
  scheduler-main.cpp
)

mlir_target_link_libraries(KTIRSchedulerDriverLib PRIVATE
  MLIROptLib
)
