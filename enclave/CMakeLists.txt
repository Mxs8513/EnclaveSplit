cmake_minimum_required(VERSION 3.16)
project(enclavesplit LANGUAGES C)

find_package(OpenEnclave CONFIG REQUIRED)

add_custom_command(
  OUTPUT enclave_t.h enclave_t.c enclave_u.h enclave_u.c
  DEPENDS ${CMAKE_SOURCE_DIR}/enclave.edl
  COMMAND openenclave::oeedger8r ${CMAKE_SOURCE_DIR}/enclave.edl
          --trusted --untrusted --search-path ${OE_INCLUDEDIR}
)

add_executable(enclavesplit_enc enclave.c enclave_t.c)
target_include_directories(enclavesplit_enc PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(enclavesplit_enc
  openenclave::oeenclave openenclave::oelibc
)

if(EXISTS ${CMAKE_SOURCE_DIR}/private.pem)
  add_custom_command(TARGET enclavesplit_enc POST_BUILD
    COMMAND openenclave::oesign sign -e $<TARGET_FILE:enclavesplit_enc>
            -c ${CMAKE_SOURCE_DIR}/enc.conf -k ${CMAKE_SOURCE_DIR}/private.pem
  )
else()
  message(WARNING "private.pem not found; enclave binary will not be signed by this template build")
endif()

add_executable(enclavesplit_host host.c enclave_u.c)
target_include_directories(enclavesplit_host PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(enclavesplit_host openenclave::oehost)
