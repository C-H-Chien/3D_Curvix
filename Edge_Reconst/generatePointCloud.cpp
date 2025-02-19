#include "generatePointCloud.h"


#include <Eigen/Core>
#include <Eigen/Dense>

#include <vector>
#include <string>
#include <iostream>
#include <iomanip>
#include <random>
#include "definitions.h"

namespace NViewsTrian
{


Vector3 addNoise(int rand_seed, double noise, double focal, size_t size_img, Vector3& obs);

PCRes generatePointCloud(PCParams & options, 
                         const GenerateTranslation& generateTranslation, 
                         const GenerateRotation& generateRotation,
                         const PerturbTranslation& perturbTranslation, 
                         const PerturbRotation& perturbRotation)
{
        
        // std::cout << "[PC] Creating output struct\n"; 
        PCRes res = PCRes(options.N_points);
        
        int rand_seed = 0;
#if FIX_RANDOMNESS
#else
        srand(time(NULL));
#endif
        /** 1. GENERATE POINT CLOUD **/
        res.points_3D.setZero();
        
        for (int i = 0; i < options.N_points; i++)
        {
                bool coord_ok = false; 
                double px = options.max_side, py = options.max_side, pz = options.max_side; 
                int n_iter = 0; 
                
                /* For X component */
                while ((!coord_ok) && (n_iter < options.max_iter))
                {
#if FIX_RANDOMNESS
                        srand(rand_seed);
                        rand_seed++;
#endif
                        px = ((((double) rand()) / ((double) RAND_MAX)) - 0.5) * options.std_pc * 2.0;
                        
                        if ((px <= options.max_side * 0.5) && (-options.max_side * 0.5 <= px)) 
                        {
                                // save data 
                                coord_ok = true;
                        }
                        n_iter++; 
                }
                // Check return 
                if ((!coord_ok) && (n_iter >= options.max_iter))
                {
                        px = options.max_side;
                }
                
                /* For Y component */
                // reset flags
                n_iter = 0;
                coord_ok = false;
                while ((!coord_ok) && (n_iter < options.max_iter))
                {
#if FIX_RANDOMNESS
                        srand(rand_seed);
                        rand_seed++;
#endif
                        py = ((((double) rand()) / ((double) RAND_MAX)) - 0.5) * options.std_pc * 2.0;
                        
                        if ((py <= options.max_side * 0.5) && (-options.max_side * 0.5 <= py)) 
                        {
                                // save data 
                                coord_ok = true;
                        }
                        n_iter++;
                }
                
                // Check return 
                if ((!coord_ok) && (n_iter >= options.max_iter))
                {
                        py = options.max_side;
                }
                
                /* For Z component */
                // reset flags
                n_iter = 0;
                coord_ok = false;
                while ((!coord_ok) && (n_iter < options.max_iter))
                {
#if FIX_RANDOMNESS
                        srand(rand_seed);
                        rand_seed++;
#endif
                        pz = (((double) rand()) / ((double) RAND_MAX) - 0.5) * 2.0 * options.std_pc + options.dist_center;
                        
                        if (pz >= 0) 
                        {
                                // save data 
                                coord_ok = true;
                        }
                        n_iter++;
                }
                // Check return 
                if ((!coord_ok) && (n_iter >= options.max_iter))
                {
                        px = 0;
                }
                
                // save point 
                res.points_3D.col(i) << px, py, pz; 
                
         }  // end: for each point
                
                
        //> Previous pose
        // res.set_rot.empty(); 
        // res.set_trans.empty();
        
        for (int j=0; j < options.M_cameras; j++)
        {
                /** 2. GENERATE POSE **/
                // a. Generate relative pose 

                Vector3 translation = generateTranslation(rand_seed, options.max_parallax, options.dir_parallax);
                rand_seed++; 

                Matrix3 rotation = generateRotation(rand_seed, options.max_angle, options.dir_rotation, translation); 
                rand_seed++;

                // b. Perturb pose 
                translation = perturbTranslation(rand_seed, options.noise_trans, translation); 
                rand_seed++;
                
                rotation = perturbRotation(rand_seed, options.noise_rot, rotation); 
                rand_seed++;
              
                res.set_rot.push_back(rotation); 
                res.set_trans.push_back(translation);
        }
        
        
        
        /** 3. GENERATE CORRESPONDENCES **/
        // res.obs.empty();
                

        for (int i=0; i < options.N_points; i++)
        {
                Vector3 p_3d = res.points_3D.col(i);
                Eigen::MatrixXd obs_i(3, options.M_cameras);
                obs_i.setZero();
                for (int j=0; j < options.M_cameras; j++)
                {
                        
                        Matrix3 rot_j = res.set_rot[j]; 
                        Vector3 trans_j = res.set_trans[j]; 
                        Vector3 qj = rot_j * p_3d + trans_j;
                                               
                
                        // add noise 
                        Vector3 obs_noisy = addNoise(rand_seed, options.noise, options.focal_length, options.size_img, qj); 
                        rand_seed++;
                        
                        // save data 
                        obs_i.col(j) = obs_noisy;
                }
                // save matrix in array 
                res.obs.push_back(obs_i);
        }


        return res;
}  // end of function 

Vector3 addNoise(int rand_seed, double noise, double focal, size_t size_img, Vector3& obs)
{
        Vector3 noisy_obs = obs; 
        
        noisy_obs(0) /= noisy_obs(2);
        noisy_obs(1) /= noisy_obs(2);
        noisy_obs(2) = 1;
        
        Matrix3 K = Matrix3::Identity(); 
        K(0,0) = focal; 
        K(1,1) = focal; 
        K(0,2) = (double)size_img; 
        K(1,2) = (double)size_img; 
        Vector3 obs_no_noise = K * noisy_obs; 
        noisy_obs = obs_no_noise; 

#if FIX_RANDOMNESS
        srand(rand_seed);
#endif
        noisy_obs(0) += (((double) rand()) / ((double) RAND_MAX)-0.5)*2.0 * noise;
#if FIX_RANDOMNESS
        srand(rand_seed + 36);
#endif
        noisy_obs(1) += (((double) rand()) / ((double) RAND_MAX)-0.5)*2.0 * noise;

        return noisy_obs;
}


Vector3 generateRandomVector(int rand_seed, double std_vec)
{
    Vector3 v;
    v.setZero(); 

#if FIX_RANDOMNESS
     srand(rand_seed);
#endif
    v(0) = (((double) rand()) / ((double) RAND_MAX)-0.5)*2.0 * std_vec;

#if FIX_RANDOMNESS
    srand(rand_seed + 99);
#endif
    v(1) = (((double) rand()) / ((double) RAND_MAX)-0.5)*2.0 * std_vec; 

#if FIX_RANDOMNESS
    srand(rand_seed + 69);
#endif
    v(2) = (((double) rand()) / ((double) RAND_MAX)-0.5)*2.0 * std_vec; 

    v.normalize();
    return v; 
}

Matrix3 generateRotationRodrigues(int rand_seed, double max_angle, const Vector3 & dir_r)
{
#if FIX_RANDOMNESS
        srand(rand_seed);
#endif
        double angle = max_angle * (((double) rand())/ ((double) RAND_MAX)-0.5)*2.0;
        
        Matrix3 rot; 
        Vector3 dir_rot = dir_r.normalized();
        Matrix3 T; 
        // cross matrix
        T << 0, -dir_rot(2), dir_rot(1), 
                dir_rot(2), 0, -dir_rot(0), 
                -dir_rot(1), dir_rot(0), 0;
        
        rot = Matrix3::Identity() + sin(angle) * T + (1 - cos(angle)) * T * T;
                
        return rot;
}


// generate random translation with the specified norm
Vector3 generateRandomTranslationDefault( int rand_seed, double max_parallax, const Vector3 & dir_parallax)
{
#if FIX_RANDOMNESS
        srand(rand_seed);
#endif
        double par_rnd = (((double) rand()) / ((double) RAND_MAX)-0.5)*2.0 * max_parallax;
        rand_seed++;
        return (par_rnd * generateRandomVector(rand_seed, 1.0));

}
Matrix3 generateRandomRotation( int rand_seed, double maxAngle )
{
        Vector3 rpy;

#if FIX_RANDOMNESS
    srand(rand_seed);
#endif
        rpy[0] = ((double) std::rand())/ ((double) RAND_MAX);
#if FIX_RANDOMNESS
    srand(rand_seed + 66);
#endif
        rpy[1] = ((double) std::rand())/ ((double) RAND_MAX);

#if FIX_RANDOMNESS
    srand(rand_seed + 88);
#endif
        rpy[2] = ((double) std::rand())/ ((double) RAND_MAX);

        rpy[0] = maxAngle*2.0*(rpy[0]-0.5);
        rpy[1] = maxAngle*2.0*(rpy[1]-0.5);
        rpy[2] = maxAngle*2.0*(rpy[2]-0.5);

        Matrix3 R1;
        R1(0,0) = 1.0;
        R1(0,1) = 0.0;
        R1(0,2) = 0.0;
        R1(1,0) = 0.0;
        R1(1,1) = cos(rpy[0]);
        R1(1,2) = -sin(rpy[0]);
        R1(2,0) = 0.0;
        R1(2,1) = -R1(1,2);
        R1(2,2) = R1(1,1);

        Matrix3 R2;
        R2(0,0) = cos(rpy[1]);
        R2(0,1) = 0.0;
        R2(0,2) = sin(rpy[1]);
        R2(1,0) = 0.0;
        R2(1,1) = 1.0;
        R2(1,2) = 0.0;
        R2(2,0) = -R2(0,2);
        R2(2,1) = 0.0;
        R2(2,2) = R2(0,0);

        Matrix3 R3;
        R3(0,0) = cos(rpy[2]);
        R3(0,1) = -sin(rpy[2]);
        R3(0,2) = 0.0;
        R3(1,0) =-R3(0,1);
        R3(1,1) = R3(0,0);
        R3(1,2) = 0.0;
        R3(2,0) = 0.0;
        R3(2,1) = 0.0;
        R3(2,2) = 1.0;

        Matrix3 rotation = R3 * R2 * R1;

        rotation.col(0) = rotation.col(0) / rotation.col(0).norm();
        rotation.col(2) = rotation.col(0).cross(rotation.col(1));
        rotation.col(2) = rotation.col(2) / rotation.col(2).norm();
        rotation.col(1) = rotation.col(2).cross(rotation.col(0));
        rotation.col(1) = rotation.col(1) / rotation.col(1).norm();
        return rotation;
}



// generate random rotation with maxAngle
Matrix3 generateRandomRotationDefault( int rand_seed, double max_angle, const Vector3 & dir_rot, const Vector3 & trans)
{
        return (generateRandomRotation( rand_seed, max_angle ));
} 

// generate orbital rotation with maxAngle
Matrix3 generateOrbitalRotation( int rand_seed, double d, const Vector3 & dir_rot, const Vector3 & trans)
{
        // here: 
        // d: distance to center point cloud 
        // trans: translation vector
        // compute angle
        //> input argument rand_seed is not in use, but is required for function namespace
        Vector3 d_c; 
        d_c << 0, 0, d; 
        
        Vector3 r = trans - d_c ; 
        
        double theta = atan2(r(2),  r(0)) + PI * 0.5; 
        
        // Construct matrix
        Matrix3 R = Matrix3::Identity(); 
        R(0, 0) = cos(theta); 
        R(0, 2) = sin(theta); 
        R(2, 0) = -sin(theta); 
        R(2, 2) = cos(theta); 
        
        return (R);
} 

Vector3 generateOrbitalTranslation( int rand_seed, double max_parallax, const Vector3 & dir_parallax)
{
        // here max parallax is the radius of the circle
#if FIX_RANDOMNESS
        srand(rand_seed);
#else
        srand(time(NULL));
#endif
        double angle_in_circle =  (((double) rand()) / ((double) RAND_MAX)-0.5)* 2 * PI;
        
        double x_c = cos(angle_in_circle) * max_parallax; 
        double z_c = sin(angle_in_circle) * max_parallax;
        Vector3 trans; 
        trans << x_c, 0, dir_parallax(0) + z_c; 
        
        return (trans);
}; 


// generate random perturbation for translation 
Vector3 perturbRandomTranslationDefault( int rand_seed, double noise, const Vector3 & trans)
{
        double n_trans = trans.norm();        
        Vector3 res = trans; 

        // perturbation 
#if FIX_RANDOMNESS
        srand(rand_seed);
#endif
        res(0) +=  (((double) rand()) / ((double) RAND_MAX)-0.5)*2.0 * noise;

#if FIX_RANDOMNESS
        srand(rand_seed + 22);
#endif
        res(1) +=  (((double) rand()) / ((double) RAND_MAX)-0.5)*2.0 * noise;
#if FIX_RANDOMNESS
        srand(rand_seed + 44);
#endif
        res(2) +=  (((double) rand()) / ((double) RAND_MAX)-0.5)*2.0 * noise;
        
        return (n_trans * res.normalized());
}
// generate random perturbation for translation 
Matrix3 perturbRandomRotationDefault( int rand_seed, double noise, const Matrix3 & rot)
{
     Matrix3 noise_rot = generateRandomRotation(rand_seed, noise); 
     
     return (noise_rot * rot);
}


Vector3 generateTranslationForward( double max_parallax, const Vector3 & dir_parallax)
{
        Vector3 t; 
        t << 0, 0, 1; 
        
        return (max_parallax * t);

}; 

Vector3 generateTranslationStereo( int rand_seed, double max_parallax, const Vector3 & dir_parallax)
{
        Vector3 t;
#if FIX_RANDOMNESS
        srand(rand_seed);
#else
        srand(time(NULL));
#endif

        double par_rnd = (((double) rand()) / ((double) RAND_MAX)-0.5)*max_parallax; 
        t << par_rnd, 0, 0; 
        
        return (t);

}; 


Vector3 generateTranslationSideways( int rand_seed, double max_parallax, const Vector3 & dir_parallax)
{
        Vector3 t;

#if FIX_RANDOMNESS
        srand(rand_seed);
#endif
        t(0) = (((double) rand()) / ((double) RAND_MAX)-0.5)*2.0; 
#if FIX_RANDOMNESS
        srand(rand_seed + 100);
#endif
        t(1) = (((double) rand()) / ((double) RAND_MAX)-0.5)*2.0; 
        t(2) = 0; 
        t.normalize(); 
        
        return (max_parallax * t);

}; 

Vector3 generateTranslationOblique( int rand_seed, double max_parallax, const Vector3 & dir_parallax)
{
        //> input argument rand_seed is not in use, but is required for function namespace
        Vector3 t;
        t << 1, 1, 1; 
        t.normalize(); 
        
        return (max_parallax * t);

}; 

//> Generate N choose 2 constraints. M is the number of cameras.
int generateM2Comb(const int M, Eigen::MatrixXd & comb_idx)
{
        // probably not the best way
        int M_comb = M * (M-1) / 2;
        comb_idx.resize(2, M_comb); 
        comb_idx.setZero();
        int col_id = 0; 
        for (int i=0;i<M;i++)
        {
                for (int j=i+1;j<M;j++)
                {
                        comb_idx.col(col_id) << i,j;  
                        col_id++;     
                }
        } 
        
        return M_comb;
}                     
                      
}  // end of namespace UtilsTwoView
