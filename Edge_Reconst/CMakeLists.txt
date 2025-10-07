#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
set ( control_sources
  
  #> Multiview Geometry Utilities
  util.hpp
  util.cpp
  PairEdgeHypo.hpp
  PairEdgeHypo.cpp
  getReprojectedEdgel.hpp
  getReprojectedEdgel.cpp
  getSupportedEdgels.hpp
  getSupportedEdgels.cpp
  getOrientationList.hpp
  getOrientationList.cpp

  file_reader.hpp
  file_reader.cpp
  edge_mapping.hpp
  edge_mapping.cpp

  EdgeSketch_Core.cpp
  EdgeSketch_Core.hpp

  #>MVT Utilities
  #mvt.hpp
  #mvt.cpp
  #NViewsCertifier.h
  #NViewsCertifier.cpp
  #NViewsClass.h
  #NViewsClass.cpp
  #NViewsTypes.h
  #NViewsUtils.h
  #NViewsUtils.cpp
  #generatePointCloud.h
  #generatePointCloud.cpp

  EdgeClusterer.hpp
  EdgeClusterer.cpp
)

include_directories(${CMAKE_CURRENT_BINARY_DIR})

add_library( edges_curves_reconstruction ${control_sources} )
target_link_libraries(edges_curves_reconstruction 
  -L/gpfs/data/bkimia/zqiwu/3D/yaml-cpp/bin/lib64 yaml-cpp
  -L/usr/lib64 pthread
)

set_target_properties(edges_curves_reconstruction PROPERTIES LINKER_LANGUAGE CXX)