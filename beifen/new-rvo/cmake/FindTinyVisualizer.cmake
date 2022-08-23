IF(NOT TinyVisualizer_FOUND)

FIND_PATH(TinyVisualizer_INCLUDE_DIR TinyVisualizer/Drawer.h
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}
  $ENV{TinyVisualizer_ROOT}/include
  $ENV{TinyVisualizer_ROOT}
  /usr/include
  /usr/local/include
  C:/TinyVisualizer/include
  )

FIND_PATH(ImGui_INCLUDE_DIR imgui/imgui.h
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}
  $ENV{TinyVisualizer_ROOT}/include
  $ENV{TinyVisualizer_ROOT}
  /usr/include
  /usr/local/include
  C:/TinyVisualizer/include
  )

FIND_LIBRARY(TinyVisualizer_LIBRARIES NAMES TinyVisualizer glfw3 PATHS 
  ${PROJECT_SOURCE_DIR}/lib
  ${PROJECT_SOURCE_DIR}
  $ENV{TinyVisualizer_ROOT}/lib
  $ENV{TinyVisualizer_ROOT}
  /usr/lib
  /usr/local/lib
  C:/TinyVisualizer/lib
  )

IF(TinyVisualizer_INCLUDE_DIR AND TinyVisualizer_LIBRARIES)
  set(TinyVisualizer_FOUND TRUE)
  set(TinyVisualizer_INCLUDE_DIRS ${TinyVisualizer_INCLUDE_DIR} ${ImGui_INCLUDE_DIR})
ENDIF(TinyVisualizer_INCLUDE_DIR AND TinyVisualizer_LIBRARIES)

MARK_AS_ADVANCED(TinyVisualizer_INCLUDE_DIR TinyVisualizer_INCLUDE_DIRS TinyVisualizer_LIBRARIES)

ENDIF(NOT TinyVisualizer_FOUND)
