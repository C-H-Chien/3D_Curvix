#ifndef FILE_READER_CPP
#define FILE_READER_CPP

#include "file_reader.hpp"

//> Constructor: defining the paths 
file_reader::file_reader( std::string dataset_path, std::string dataset_name, std::string scene_name, int Num_Of_Total_Imgs )
  : Dataset_Total_Num_Of_Images(Num_Of_Total_Imgs)
{
  dataset_name_sequence_path = dataset_path + dataset_name + "/" + scene_name;

  Edge_File_Path_First_Half = dataset_name_sequence_path + "/Edges/Edge_";
  Curvelet_File_Path_First_Half = dataset_name_sequence_path + "/curvelets/Curvelets_";
  Kmatrix_File_Path = dataset_name_sequence_path + "/RnT/K_matrix.txt";

//> Read camera poses. For ABC-NEF dataset, read the refined absolute camera poses.
#if USE_REFINED_CAM_POSES
  Rmatrix_File_Path = dataset_name_sequence_path + "/RnT/refined_R.txt";
  Tmatrix_File_Path = dataset_name_sequence_path + "/RnT/refined_T.txt";
#else
  Rmatrix_File_Path = dataset_name_sequence_path + "/RnT/R_matrix.txt";
  Tmatrix_File_Path = dataset_name_sequence_path + "/RnT/T_matrix.txt";
#endif

  //> Read GT edge correspondences for precision-recall experiments
  GT_File_Path = dataset_name_sequence_path + "/GT_edge_pairs_indices_0006.txt";

  Image_File_Source_Path = dataset_name_sequence_path + "/train_img/";
}

//> Read all edgel files
void file_reader::read_All_Edgels( std::vector<Eigen::MatrixXd> &All_Edgels, int thresh_EDG ) 
{
  int file_idx = 0;
  All_Edgels.clear();

  //> Looping over all frames and read corresponding edgels
  while( file_idx < Dataset_Total_Num_Of_Images ) 
  {
    All_Edgels.push_back( read_Edgels_Of_a_File( file_idx, thresh_EDG ) );
    file_idx++;    
  }
  
#if SHOW_DATA_LOADING_INFO
  std::cout << "All edgel files are loaded successfully" << std::endl;
#endif
}

//> Read only one image
bool file_reader::read_an_image( int file_idx, cv::Mat &grayscale_img )
{
  std::string img_path = Image_File_Source_Path + std::to_string(file_idx) + "_colors.png";
  cv::Mat original_img = cv::imread(img_path, cv::IMREAD_COLOR);

  if (original_img.empty()) {
    std::string err_msg = "Image in path " + img_path + " not found!";
    LOG_ERROR(err_msg);
    return false;
  }

  //> Check if converting to a gray-scale image is necessary
  if (original_img.channels() == 1) {
    grayscale_img = original_img;
  }
  if (original_img.channels() == 3) {
    cv::cvtColor(original_img, grayscale_img, cv::COLOR_BGR2GRAY);
  }  
  return true;
}

//> Read edgels of a file specified by the file_idx
Eigen::MatrixXd file_reader::read_Edgels_Of_a_File( int file_idx, int thresh_EDG ) 
{  
  Eigen::MatrixXd Edgels;
  Eigen::Vector4d row_edge;
  std::string Edge_File_Path = Edge_File_Path_First_Half + std::to_string(file_idx) + "_t" + std::to_string(thresh_EDG) + ".txt";
  std::fstream Edge_File;
  Edge_File.open(Edge_File_Path, std::ios_base::in);
  if (!Edge_File) {
    LOG_FILE_ERROR(Edge_File_Path); 
    exit(1);
  }
  else {
    int d = 0, q = 0;
    double rd_data;
    Edgels.resize(1,4);
    while (Edge_File >> rd_data) {
      row_edge(q) = rd_data;
      q++;
      if(q>3){
        Edgels.conservativeResize(d+1,4);
        Edgels.row(d) = row_edge;
        q = 0;
        d++;
      }
    }
  }
  Edge_File.close();
  return Edgels;
}

//> Read all edge curvelets
void file_reader::read_All_Curvelets( std::vector<EdgeCurvelet> &All_Curvelets, int thresh_EDG ) {
  int file_idx = 0;
  All_Curvelets.clear();

  //> Looping over all frames and read corresponding edgels
  while( file_idx < Dataset_Total_Num_Of_Images ) 
  {
    std::string Curvelet_File_Path = Curvelet_File_Path_First_Half + std::to_string(file_idx) + "_t" + std::to_string(thresh_EDG) + ".txt";
    std::fstream Curvelet_File;
    Curvelet_File.open(Curvelet_File_Path, std::ios_base::in);
    if (!Curvelet_File) {
      LOG_FILE_ERROR(Curvelet_File_Path); 
      exit(1);
    }
    else {
      int rd_data_anchor, rd_data_chain1, rd_data_chain2;
      while (Curvelet_File >> rd_data_anchor >> rd_data_chain1 >> rd_data_chain2 ) {
        EdgeCurvelet curvelet;
        curvelet.image_number = file_idx;
        curvelet.self_edge_index = rd_data_anchor;
        curvelet.neighbor_edge_index = rd_data_chain1;
        All_Curvelets.push_back(curvelet);
      }
    }
    file_idx++;    
  }
#if SHOW_DATA_LOADING_INFO
  std::cout << "All edge curvelets are loaded successfully" << std::endl;
#endif
}

//> Read rotation matrices of all cameras
void file_reader::readRmatrix( std::vector<Eigen::Matrix3d> &All_R )
{
  Eigen::Matrix3d R_matrix;
  Eigen::Vector3d row_R;
  double rd_data;
  int d = 0, q = 0;
  std::fstream Rmatrix_File;
  Rmatrix_File.open(Rmatrix_File_Path, std::ios_base::in);
  if (!Rmatrix_File) { 
    LOG_FILE_ERROR(Rmatrix_File_Path); exit(1); 
  }
  else {
    while (Rmatrix_File >> rd_data) {
      row_R(q) = rd_data;
      q++;
      if(q>2){
        R_matrix.row(d) = row_R;
        row_R = {};
        q = 0;
        d++;
      }
      if(d>2){
        All_R.push_back(R_matrix);
        R_matrix = {};
        d = 0;
      }
    }
  }
  Rmatrix_File.close();
#if SHOW_DATA_LOADING_INFO
  std::cout<< "R matrix loading finished" <<std::endl;
#endif
}

//> Read translation vectors of all cameras
void file_reader::readTmatrix( std::vector<Eigen::Vector3d> &All_T )
{
  Eigen::Vector3d T_matrix;
  double rd_data;
  int d = 0;
  std::fstream Tmatrix_File;
  Tmatrix_File.open(Tmatrix_File_Path, std::ios_base::in);
  if (!Tmatrix_File) { 
    LOG_FILE_ERROR(Tmatrix_File_Path); exit(1); 
  }
  else {
    //> TODO: make this simpler
    while (Tmatrix_File >> rd_data) {
      T_matrix(d) = rd_data;
      d++;
      if(d > 2) {
          All_T.push_back(T_matrix);
          T_matrix = {};
          d = 0;
      }
    }
  }
  Tmatrix_File.close();
#if SHOW_DATA_LOADING_INFO
  std::cout<< "T matrix loading finished" <<std::endl;
#endif
}

//> Read calibration matrices only if each image has individual calibration matrix K
void file_reader::readK( std::vector<Eigen::Matrix3d> &All_K )
{
  std::fstream Kmatrix_File;
  Eigen::Matrix3d K_matrix;
  Eigen::Vector3d row_K;
  double rd_data;
  int d = 0, q = 0;
  Kmatrix_File.open(Kmatrix_File_Path, std::ios_base::in);
  if (!Kmatrix_File) { 
    LOG_FILE_ERROR(Kmatrix_File_Path); exit(1);
  }
  else {
    while (Kmatrix_File >> rd_data) {
      row_K(q) = rd_data;
      q++;
      if( q > 2 )
      {
        K_matrix.row(d) = row_K;
        row_K = {};
        q = 0;
        d++;
      }
      if( d > 2 )
      {
        All_K.push_back(K_matrix);
        K_matrix = {};
        d = 0;
      }
    }
    
  }
  Kmatrix_File.close();
#if SHOW_DATA_LOADING_INFO
  std::cout << "Multiple intrinsic matrices are loaded" <<std::endl;
#endif
}

//> Read ground truth edge correspondence pairs indices
void file_reader::readGT_EdgePairs( std::vector<std::vector<int>> &GT_EdgePairs )
{
  std::fstream GT_EdgePairs_File;
  GT_EdgePairs_File.open(GT_File_Path, std::ios_base::in);
  
  if (!GT_EdgePairs_File) {
    LOG_FILE_ERROR(GT_File_Path); 
    exit(1);
  }
  else {
    std::string line;
    GT_EdgePairs.clear();
    
    while (std::getline(GT_EdgePairs_File, line)) {
      std::istringstream iss(line);
      std::vector<int> row;
      int value;
      
      //> Read all values in the current line
      while (iss >> value) {
        row.push_back(value);
      }
      
      //> Sanity check: Only add non-empty rows
      if (!row.empty()) {
        GT_EdgePairs.push_back(row);
      }
    }
  }
  
  GT_EdgePairs_File.close();
  
#if SHOW_DATA_LOADING_INFO
  std::cout << "GT edge pairs indices loaded successfully. Total 3D points: " \
            << GT_EdgePairs.size() << std::endl;
  std::vector<int> one_row_of_data = GT_EdgePairs[10];
  for (int i = 0; i < one_row_of_data.size(); i++) std::cout << one_row_of_data[i] << ", ";
  std::cout << std::endl;
#endif
}

#endif

