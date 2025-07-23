//> Macros

//> Whether to delete output files automatically
#define OUTPUT_FOLDER_NAME              std::string("../../outputs")
#define DELETE_ALL_FILES_UNDER_OUTPUTS  (true)

//> Write to the files
#define WRITE_3D_EDGES                  (true)
#define WRITE_3D_EDGE_GRAPH             (true)

//> NCC on Edges Settings
#define EDGE_ORTHOGONAL_SHIFT_MAG       (5)         //> in pixels
#define PATCH_SIZE                      (7)         //> in pixels

//> Print out in terminal
#define SHOW_EDGE_SKETCH_SETTINGS       (false)

//> hypotheses formation settings
#define EPIP_TANGENCY_DISPL_THRESH      (3)         //> in pixels
#define LOCATION_PERTURBATION           (0.3)         //> in pixels
#define ORIENT_PERTURBATION             (0.174533)  //> in radians. 0.174533 is 10 degrees
#define CLUSTER_DIST_THRESH             (1)         //> τc, in pixels
#define CLUSTER_ORIENT_THRESH           (20.0)      //> in degrees
#define MAX_CLUSTER_SIZE                (10)        //> max number of edges per cluster
#define CLUSTER_ORIENT_GAUSS_SIGMA      (2.0)
#define ORTHOGONAL_SHIFT_MAG            (5)         //> in pixels

//> Edge graph pruning parameters
#define PRUNE_3D_EDGE_GRAPH_LAMBDA1     (1)
#define PRUNE_3D_EDGE_GRAPH_LAMBDA2     (1)
#define PRUNE_3D_EDGE_GRAPH_LAMBDA3     (1)
#define PRUNE_BY_PROJ_PROX_THRESH       (6)     //> in pixels

//> Edge graph alignment parameters
#define NUM_OF_ITERATIONS               (1000)
#define INIT_FROCE_STEP_SIZE            (0.01)
#define INIT_TORQUE_STEP_SIZE           (0.01)
#define ENABLE_EXPO_FORCE_AND_TORQUE    (false)
#define EXPO_INCREASE_FACTOR            (sqrt(2))

//> Precision-Recall evaluation parameters
#define GT_PROXIMITY_THRESH             (1) //> in pixels

//> Debugging purpose
#define DEBUG                      (0)
#define DEBUG_READ_FILES           (false)
#define DEBUG_PAIRED_EDGES         (true)
#define SHOW_DATA_LOADING_INFO     (false)
#define SHOW_OMP_NUM_OF_THREADS    (true)
#define ISOLATE_DATA               (false)

//> Constant values (no change)
#define PI                            (3.1415926)

//> Some useful macros
#define LOG_GEN_MESG(info_msg)          printf("\033[1;32m %s\033[0m\n", std::string(info_msg).c_str() );
#define LOG_INFOR_MESG(info_msg)        printf("\033[1;32m[INFO] %s\033[0m\n", std::string(info_msg).c_str() );
#define LOG_TIMEIMGS(time_msg)          printf("\033[1;35m[TIME] %s\033[0m\n", std::string(time_msg).c_str() );
#define LOG_FILE_ERROR(err_msg)         printf("\033[1;31m[ERROR] File %s not found!\033[0m\n", std::string(err_msg).c_str() );
#define LOG_ERROR(err_msg)              printf("\033[1;31m[ERROR] %s\033[0m", std::string(err_msg).c_str() );
#define LOG_WARNING(warn_msg)           printf("\033[1;35m[WARNING] %s\033[0m\n", std::string(warn_msg).c_str() );
#define LOG_DATA_LOAD_ERROR(err_msg)    printf("\033[1;31m[DATA LOAD ERROR] %s not loaded successfully!\033[0m\n", std::string(err_msg).c_str() );

//> MVT definitions

//> General Settings
#define USE_REFINED_CAM_POSES           (true)
#define FIX_RANDOMNESS                  (true)
#define RUN_CERES_SOLVER_ON             (false)

//> Thresholds Settings
#define CERTIFY_THRESH                  (-1e-09)   //> -1e-09 accroding to the paper. Negative because of numerical error.
#define DIFF_CONSECUTIVE_SOLS_THRESH    (3e-10)    //> 3e-10 according to the paper

#define PRINT_VECTOR3D(name, vec)       printf("%s = [%f, %f, %f]\n", std::string(name).c_str(), vec(0), vec(1), vec(2));
#define PRINT_VECTORXD(name, vec)       printf("%s = [", std::string(name).c_str()); for(int i = 0; i < vec.size(); i++) {printf("%f ", vec(i));} printf("]\n");

#define PRINT_ESSENTIAL(id1, id2, mat)  printf("E%d%d = [\n", id1, id2); \
                                        for(int i = 0; i < 3; i++) { \
                                            for(int j = 0; j < 3; j++) printf("%10.7f ", mat(i,j)); \
                                            printf(";\n"); \
                                        } printf("]\n");

