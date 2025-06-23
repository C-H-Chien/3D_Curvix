#ifndef PAIREDGEHYPO_CPP
#define PAIREDGEHYPO_CPP
// ====================================================================================================
//
// =====================================================================================================
#include <cmath>
#include <math.h>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <iomanip>
#include <string.h>
#include <assert.h>
#include <vector>
#include <chrono>

#include "PairEdgeHypo.hpp"
#include "definitions.h"

//> Eigen library
#include <Eigen/Core>
#include <Eigen/Dense>

#include <stdio.h>
#include <stdlib.h>

//using namespace std;

namespace PairEdgeHypothesis {
    
    pair_edge_hypothesis::pair_edge_hypothesis( double Reproj_Err_Thresh )
    : reproj_dist_thresh(Reproj_Err_Thresh) {
        util = std::shared_ptr<MultiviewGeometryUtil::multiview_geometry_util>(new MultiviewGeometryUtil::multiview_geometry_util());
    }

    Eigen::MatrixXd pair_edge_hypothesis::getAp_Bp(Eigen::MatrixXd Edges_HYPO2, Eigen::Vector3d pt_edgel_HYPO1, Eigen::Matrix3d F ) {
        Eigen::Vector3d coeffs;
        coeffs = F * pt_edgel_HYPO1;
        Eigen::MatrixXd Ap_Bp;
        Ap_Bp.conservativeResize(Edges_HYPO2.rows(),2);
        Ap_Bp.col(0) = coeffs(0) * Edges_HYPO2.col(0);
        Ap_Bp.col(1) = coeffs(1) * Edges_HYPO2.col(1);
        return Ap_Bp;
    }

    Eigen::MatrixXd pair_edge_hypothesis::getAp_Bp_Dist(Eigen::MatrixXd Edges_HYPO2, Eigen::Vector3d pt_edgel_HYPO1, Eigen::Matrix3d F ) {
        Eigen::Vector3d coeffs;
        coeffs = F * pt_edgel_HYPO1;
        Eigen::MatrixXd Ap_Bp;
        Ap_Bp.conservativeResize(Edges_HYPO2.rows(),2);
        Ap_Bp.col(0) = coeffs(0) * Edges_HYPO2.col(0);
        Ap_Bp.col(1) = coeffs(1) * Edges_HYPO2.col(1);
        Eigen::MatrixXd numerOfDist = Ap_Bp.col(0) + Ap_Bp.col(1) + Eigen::VectorXd::Ones(Edges_HYPO2.rows())*coeffs(2);
        Eigen::MatrixXd denomOfDist = Eigen::VectorXd::Ones(Edges_HYPO2.rows())*(coeffs(0)*coeffs(0)+coeffs(1)*coeffs(1));
        denomOfDist = denomOfDist.array().sqrt();
        return numerOfDist.cwiseAbs()/denomOfDist(0);
    }

    Eigen::MatrixXd pair_edge_hypothesis::getHYPO2_idx(Eigen::MatrixXd Edges_HYPO2, Eigen::MatrixXd numerOfDist) {
        int idx_hypopair = 0;
        Eigen::MatrixXd HYPO2_idx;
        for(int idx_HYPO2 = 0; idx_HYPO2 < numerOfDist.rows(); idx_HYPO2++){
            double dist = numerOfDist(idx_HYPO2,0);
            if(dist < reproj_dist_thresh){
                HYPO2_idx.conservativeResize(idx_hypopair+1,1);
                HYPO2_idx.row(idx_hypopair) << double(idx_HYPO2);
                idx_hypopair++;
            }
        }
        return HYPO2_idx;
    }

    Eigen::MatrixXd pair_edge_hypothesis::getedgels_HYPO2(Eigen::MatrixXd Edges_HYPO2, Eigen::MatrixXd numerOfDist) {
        int idx_hypopair = 0;
        Eigen::MatrixXd edgels_HYPO2;
        for(int idx_HYPO2 = 0; idx_HYPO2 < numerOfDist.rows(); idx_HYPO2++){
            double dist = numerOfDist(idx_HYPO2,0);
            if(dist < reproj_dist_thresh){
                edgels_HYPO2.conservativeResize(idx_hypopair+1,4);
                edgels_HYPO2.row(idx_hypopair) = Edges_HYPO2.row(idx_HYPO2);
                idx_hypopair++;
            }
        }
        return edgels_HYPO2;
    }

    //> Find edges in the view whose epipolar angle fall within the epipolar angle range epip_angle_range = [thresh_ore21_1, thresh_ore21_2]
    Eigen::MatrixXd pair_edge_hypothesis::getHYPO2_idx_Ore(Eigen::MatrixXd OreListdegree, std::pair<double,double> epip_angle_range) {
        double thresh_ore21_1 = epip_angle_range.first;
        double thresh_ore21_2 = epip_angle_range.second;
        int idx_hypopair = 0;
        Eigen::MatrixXd HYPO2_idx;
        std::vector<double> Ore_List1Bar(OreListdegree.data(), OreListdegree.data() + OreListdegree.rows());
        auto it = std::find_if(std::begin(Ore_List1Bar), std::end(Ore_List1Bar), [thresh_ore21_1, thresh_ore21_2](double i){return i > thresh_ore21_1 && i <thresh_ore21_2;});
        while (it != std::end(Ore_List1Bar)) {
            HYPO2_idx.conservativeResize(idx_hypopair+1,1);
            HYPO2_idx.row(idx_hypopair) << double(std::distance(std::begin(Ore_List1Bar), it));
            idx_hypopair++;
            it = std::find_if(std::next(it), std::end(Ore_List1Bar), [thresh_ore21_1, thresh_ore21_2](double i){return i > thresh_ore21_1 && i <thresh_ore21_2;});
        }
        return HYPO2_idx;
    }

    Eigen::MatrixXd pair_edge_hypothesis::getedgels_HYPO2_Ore(Eigen::MatrixXd Edges_HYPO2, Eigen::MatrixXd OreListdegree, std::pair<double,double> epip_angle_range) {
        double thresh_ore21_1 = epip_angle_range.first;
        double thresh_ore21_2 = epip_angle_range.second;
        int idx_hypopair = 0;
        Eigen::MatrixXd edgels_HYPO2;
        std::vector<double> Ore_List1Bar(OreListdegree.data(), OreListdegree.data() + OreListdegree.rows());
        auto it = std::find_if(std::begin(Ore_List1Bar), std::end(Ore_List1Bar), [thresh_ore21_1, thresh_ore21_2](double i){return i > thresh_ore21_1 && i <thresh_ore21_2;});
        while (it != std::end(Ore_List1Bar)) {
            edgels_HYPO2.conservativeResize(idx_hypopair+1,4);
            //edgels_HYPO2.row(idx_hypopair) << double(std::distance(std::begin(Ore_List1Bar), it));
            edgels_HYPO2.row(idx_hypopair) = Edges_HYPO2.row(std::distance(std::begin(Ore_List1Bar), it));
            idx_hypopair++;
            it = std::find_if(std::next(it), std::end(Ore_List1Bar), [thresh_ore21_1, thresh_ore21_2](double i){return i > thresh_ore21_1 && i <thresh_ore21_2;});
        }
        return edgels_HYPO2;
    }

    Eigen::MatrixXd pair_edge_hypothesis::getHYPO2_idx_Ore_sted(Eigen::MatrixXd OreListdegree, std::pair<double,double> epip_angle_range) {
        double thresh_ore21_1 = epip_angle_range.first;
        double thresh_ore21_2 = epip_angle_range.second;
        Eigen::MatrixXd HYPO2_idx;
        HYPO2_idx.conservativeResize(2,1);
        std::vector<double> Ore_List1Bar(OreListdegree.data(), OreListdegree.data() + OreListdegree.rows());
        auto itst = std::find_if(std::begin(Ore_List1Bar), std::end(Ore_List1Bar), [thresh_ore21_1](double i){return i > thresh_ore21_1;});
        HYPO2_idx.row(0) << double(std::distance(std::begin(Ore_List1Bar), itst));
        auto ited = std::find_if(std::begin(Ore_List1Bar), std::end(Ore_List1Bar), [thresh_ore21_2](double i){return i > thresh_ore21_2;});
        HYPO2_idx.row(1) << double(std::distance(std::begin(Ore_List1Bar), ited)-1);
        
        return HYPO2_idx;
    }

    Eigen::MatrixXd pair_edge_hypothesis::getHYPO2_idx_Ore_fixed(Eigen::MatrixXd OreListdegree, std::pair<double,double> epip_angle_range) {
        double thresh_ore21_1 = epip_angle_range.first;
        double thresh_ore21_2 = epip_angle_range.second;
        Eigen::MatrixXd HYPO2_idx;
        HYPO2_idx.conservativeResize(20,1);
        std::vector<double> Ore_List1Bar(OreListdegree.data(), OreListdegree.data() + OreListdegree.rows());
        auto itst = std::find_if(std::begin(Ore_List1Bar), std::end(Ore_List1Bar), [thresh_ore21_1](double i){return i > thresh_ore21_1;});
        auto ited = std::find_if(std::begin(Ore_List1Bar), std::end(Ore_List1Bar), [thresh_ore21_2](double i){return i > thresh_ore21_2;});
        int idx_start = std::distance(std::begin(Ore_List1Bar), itst);
        int idx_end   = std::distance(std::begin(Ore_List1Bar), ited)-1;
        int midpoint  = std::round((idx_start+idx_end)/2);
        for(int idx = 0; idx < 20; idx++){
            HYPO2_idx.row(idx) << double(midpoint-11+idx);
        }
        return HYPO2_idx;
    }

    Eigen::MatrixXd pair_edge_hypothesis::getedgels_HYPO2_Ore_fixed(Eigen::MatrixXd Edges_HYPO2, Eigen::MatrixXd OreListdegree, std::pair<double,double> epip_angle_range) {
        double thresh_ore21_1 = epip_angle_range.first;
        double thresh_ore21_2 = epip_angle_range.second;
        Eigen::MatrixXd edgels_HYPO2;
        edgels_HYPO2.conservativeResize(20,4);
        std::vector<double> Ore_List1Bar(OreListdegree.data(), OreListdegree.data() + OreListdegree.rows());
        auto itst = std::find_if(std::begin(Ore_List1Bar), std::end(Ore_List1Bar), [thresh_ore21_1](double i){return i > thresh_ore21_1;});
        auto ited = std::find_if(std::begin(Ore_List1Bar), std::end(Ore_List1Bar), [thresh_ore21_2](double i){return i > thresh_ore21_2;});
        int idx_start = std::distance(std::begin(Ore_List1Bar), itst);
        int idx_end   = std::distance(std::begin(Ore_List1Bar), ited)-1;
        int midpoint  = std::round((idx_start+idx_end)/2);
        for(int idx = 0; idx < 20; idx++){
            edgels_HYPO2.row(idx) = Edges_HYPO2.row(midpoint-11+idx);
        }        
        return edgels_HYPO2;
    }

    Eigen::MatrixXd pair_edge_hypothesis::edgelsHYPO2correct(
        Eigen::MatrixXd edgels_HYPO2,  Eigen::MatrixXd edgel_HYPO1, 
        Eigen::Matrix3d F21, Eigen::Matrix3d F12, Eigen::MatrixXd HYPO2_idx_raw)
    {
        Eigen::MatrixXd edgels_HYPO2_corrected;
        Eigen::MatrixXd xy1_H1;
        xy1_H1.conservativeResize(1,3);
        xy1_H1(0,0) = edgel_HYPO1(0,0);
        xy1_H1(0,1) = edgel_HYPO1(0,1);
        xy1_H1(0,2) = 1;
        Eigen::MatrixXd coeffspt1T = F21 * xy1_H1.transpose();
        Eigen::MatrixXd coeffspt1  = coeffspt1T.transpose();
        Eigen::MatrixXd Apixel_1   = coeffspt1.col(0);
        Eigen::MatrixXd Bpixel_1   = coeffspt1.col(1);
        Eigen::MatrixXd Cpixel_1   = coeffspt1.col(2);
        double a1_line  = -Apixel_1(0,0)/Bpixel_1(0,0);
        double b1_line  = -1;
        double c1_line  = -Cpixel_1(0,0)/Bpixel_1(0,0);
        double a_edgeH1    = tan(edgel_HYPO1(0,2));
        double b_edgeH1    = -1;
        double c_edgeH1    = -(a_edgeH1*edgel_HYPO1(0,0)-edgel_HYPO1(0,1));
        double idx_correct = 0;
        for(int idx_hypo2 = 0; idx_hypo2 < edgels_HYPO2.rows(); idx_hypo2++){
            double a_edgeH2 = tan(edgels_HYPO2(idx_hypo2,2));
            double b_edgeH2 = -1;
            double c_edgeH2 = -(a_edgeH2*edgels_HYPO2(idx_hypo2,0)-edgels_HYPO2(idx_hypo2,1));
            double x_currH2 = ((b1_line*c_edgeH2-b_edgeH2*c1_line)/(a1_line*b_edgeH2-a_edgeH2*b1_line) + edgels_HYPO2(idx_hypo2,0))/2;
            double y_currH2 = ((c1_line*a_edgeH2-c_edgeH2*a1_line)/(a1_line*b_edgeH2-a_edgeH2*b1_line) + edgels_HYPO2(idx_hypo2,1))/2;
            double dist2    = sqrt((x_currH2 - edgels_HYPO2(idx_hypo2,0))*(x_currH2 - edgels_HYPO2(idx_hypo2,0))+(y_currH2 - edgels_HYPO2(idx_hypo2,1))*(y_currH2 - edgels_HYPO2(idx_hypo2,1)));

            Eigen::MatrixXd xy1_H2;
            xy1_H2.conservativeResize(1,3);
            xy1_H2(0,0) = x_currH2;
            xy1_H2(0,1) = y_currH2;
            xy1_H2(0,2) = 1;
            Eigen::MatrixXd coeffspt2T = F12 * xy1_H2.transpose();
            Eigen::MatrixXd coeffspt2  = coeffspt2T.transpose();
            Eigen::MatrixXd Apixel_2   = coeffspt2.col(0);
            Eigen::MatrixXd Bpixel_2   = coeffspt2.col(1);
            Eigen::MatrixXd Cpixel_2   = coeffspt2.col(2);
            double a2_line  = -Apixel_2(0,0)/Bpixel_2(0,0);
            double b2_line  = -1;
            double c2_line  = -Cpixel_2(0,0)/Bpixel_2(0,0);
            double x_currH1 = (b2_line*c_edgeH1-b_edgeH1*c2_line)/(a2_line*b_edgeH1-a_edgeH1*b2_line);
            double y_currH1 = (c2_line*a_edgeH1-c_edgeH1*a2_line)/(a2_line*b_edgeH1-a_edgeH1*b2_line);
            double dist1    = sqrt((x_currH1 - edgel_HYPO1(0,0))*(x_currH1 - edgel_HYPO1(0,0))+(y_currH1 - edgel_HYPO1(0,1))*(y_currH1 - edgel_HYPO1(0,1)));

            edgels_HYPO2_corrected.conservativeResize(idx_correct+1,10);
            edgels_HYPO2_corrected.row(idx_correct) << edgel_HYPO1(0,0), edgel_HYPO1(0,1), edgel_HYPO1(0,2), edgel_HYPO1(0,3), \
                                                        edgels_HYPO2(idx_hypo2,0), edgels_HYPO2(idx_hypo2,1), edgels_HYPO2(idx_hypo2,2), edgels_HYPO2(idx_hypo2,3), \
                                                        HYPO2_idx_raw(idx_hypo2), idx_hypo2;
            idx_correct++;
        }
        return edgels_HYPO2_corrected;
    }

    Eigen::MatrixXd pair_edge_hypothesis::edgelsHYPO2_epipolar_correction(
        Eigen::MatrixXd edgels_HYPO2, Eigen::MatrixXd edgel_HYPO1, \
        Eigen::Matrix3d F21, Eigen::Matrix3d F12, Eigen::MatrixXd HYPO2_idx_raw ) 
    {
        Eigen::MatrixXd edgels_HYPO2_corrected(0, 10);
        int idx_correct = 0;

        //> Get the epipolar line coefficients corresponding to the H1 edge
        Eigen::Vector3d epip_coeffs = util->getEpipolarLineCoeffs( edgel_HYPO1, 0, F21 );
        
        for(int idx_hypo2 = 0; idx_hypo2 < edgels_HYPO2.rows(); idx_hypo2++)
        {
            Eigen::Vector3d xy1_H2( edgels_HYPO2(idx_hypo2,0), edgels_HYPO2(idx_hypo2,1), 1.0 );
            
            double corrected_x, corrected_y, corrected_theta;
            double epiline_x, epiline_y;
            double normal_distance_epiline = util->getNormalDistance2EpipolarLine( epip_coeffs, xy1_H2, epiline_x, epiline_y );

            if (normal_distance_epiline < LOCATION_PERTURBATION){
                //> If normal distance is small, move directly to epipolar line
                corrected_x = epiline_x;
                corrected_y = epiline_y;
                corrected_theta = edgels_HYPO2(idx_hypo2, 2);
            }
            else {
                double x_intersection, y_intersection;
                Eigen::Vector3d isolated_H2( edgels_HYPO2(idx_hypo2,0), edgels_HYPO2(idx_hypo2,1), edgels_HYPO2(idx_hypo2,2) );
                double dist_diff_edg2 = util->getTangentialDistance2EpipolarLine( epip_coeffs, isolated_H2, x_intersection, y_intersection );
                double theta = edgels_HYPO2(idx_hypo2,2);

                if (dist_diff_edg2 < EPIP_TANGENCY_DISPL_THRESH) {
                    //> (i) if the displacement after epipolar shift is less than EPIP_TANGENCY_DISPL_THRESH, then feel free to shift it along its direction vector
                    corrected_x = x_intersection;
                    corrected_y = y_intersection;
                    corrected_theta = edgels_HYPO2(idx_hypo2, 2);
                }
                else {
                    //> (ii) if not, then perturb the edge orientation first before shifting the edge along its direction vector
                    double p_theta = epip_coeffs(0) * cos(theta) + epip_coeffs(1) * sin(theta);
                    double derivative_p_theta = -epip_coeffs(0) * sin(theta) + epip_coeffs(1) * cos(theta);

                    //> Decide how theta should be perturbed by observing the signs of p_theta and derivative_p_theta
                    if (p_theta > 0 && derivative_p_theta < 0) theta -= ORIENT_PERTURBATION;
                    else if (p_theta < 0 && derivative_p_theta < 0) theta -= ORIENT_PERTURBATION;
                    else if (p_theta > 0 && derivative_p_theta > 0) theta += ORIENT_PERTURBATION;
                    else if (p_theta < 0 && derivative_p_theta > 0) theta += ORIENT_PERTURBATION;

                    //> Calculate the intersection between the tangent and epipolar line
                    Eigen::Vector3d isolated_H2_( edgels_HYPO2(idx_hypo2,0), edgels_HYPO2(idx_hypo2,1), theta );
                    dist_diff_edg2 = util->getTangentialDistance2EpipolarLine( epip_coeffs, isolated_H2_, x_intersection, y_intersection );

                    if (dist_diff_edg2 < EPIP_TANGENCY_DISPL_THRESH) {
                        corrected_x = x_intersection;
                        corrected_y = y_intersection;
                        corrected_theta = theta;
                    } 
                    else {
                        continue;
                    }
                }
            }
            edgels_HYPO2_corrected.conservativeResize(idx_correct+1,10);
            edgels_HYPO2_corrected.row(idx_correct) << edgel_HYPO1(0,0), edgel_HYPO1(0,1), edgel_HYPO1(0,2), edgel_HYPO1(0,3), \
                                                       corrected_x, corrected_y, edgels_HYPO2(idx_hypo2,2), edgels_HYPO2(idx_hypo2,3), \
                                                       HYPO2_idx_raw(idx_hypo2), idx_hypo2;
            idx_correct++;
        }
        return edgels_HYPO2_corrected;
    }
   
    // Eigen::MatrixXd pair_edge_hypothesis::edgelsHYPO2correct_post_validation(
    //     Eigen::MatrixXd edgels_HYPO2,  Eigen::MatrixXd edgel_HYPO1, 
    //     Eigen::Matrix3d F21, Eigen::Matrix3d F12, Eigen::MatrixXd HYPO2_idx_raw)
    // {
    //     Eigen::MatrixXd edgels_HYPO2_corrected(0, 10);  

    //     ////////////// find cofficients for tangent and epipolar line for edge 2 //////////////
    //     Eigen::MatrixXd xy1_H1;
    //     xy1_H1.conservativeResize(1,3);
    //     xy1_H1(0,0) = edgel_HYPO1(0,0);
    //     xy1_H1(0,1) = edgel_HYPO1(0,1);
    //     xy1_H1(0,2) = 1;
    //     Eigen::MatrixXd coeffspt1T = F21 * xy1_H1.transpose();
    //     Eigen::MatrixXd coeffspt1  = coeffspt1T.transpose();
    //     Eigen::MatrixXd Apixel_1   = coeffspt1.col(0);
    //     Eigen::MatrixXd Bpixel_1   = coeffspt1.col(1);
    //     Eigen::MatrixXd Cpixel_1   = coeffspt1.col(2);
    //     double a1_line  = -Apixel_1(0,0)/Bpixel_1(0,0);
    //     double b1_line  = -1;
    //     double c1_line  = -Cpixel_1(0,0)/Bpixel_1(0,0);
    //     double idx_correct = 0;
    //     ////////////// find cofficients for tangent and epipolar line for edge 2 //////////////
        
    //     for(int idx_hypo2 = 0; idx_hypo2 < edgels_HYPO2.rows(); idx_hypo2++){

    //         Eigen::Vector3d xy1_H2( edgels_HYPO2(idx_hypo2,0), edgels_HYPO2(idx_hypo2,1), 1.0 );
    //         double corrected_x, corrected_y;

    //         ////////////// calculate normal distance between edge and epipolar line //////////////
    //         double epiline_x = xy1_H2(0) - a1_line * (a1_line * xy1_H2(0) + b1_line *xy1_H2(1) + c1_line)/(pow(a1_line,2) + pow(b1_line,2));
    //         double epiline_y = xy1_H2(1) - b1_line* (a1_line * xy1_H2(0) + b1_line * xy1_H2(1) + c1_line)/(pow(a1_line,2) + pow(b1_line,2));
    //         double distance_epiline = sqrt(pow(xy1_H2(0) - epiline_x, 2) + pow(xy1_H2(1) - epiline_y, 2));
    //         ////////////// calculate normal distance between edge and epipolar line //////////////

    //         // double tangential_distance;
    //         if (distance_epiline < LOCATION_PERTURBATION){
    //             // If normal distance is small, move directly to epipolar line
    //             corrected_x = epiline_x;
    //             corrected_y = epiline_y;
    //         }
    //         else {
    //             ////////////// calculate the intersection between the tangent and epipolar line //////////////
    //             double theta = edgels_HYPO2(idx_hypo2,2);
    //             double a_edgeH2 = tan(theta); //tan(theta2)
    //             double b_edgeH2 = -1;
    //             double c_edgeH2 = -(a_edgeH2*edgels_HYPO2(idx_hypo2,0)-edgels_HYPO2(idx_hypo2,1)); //−(a⋅x2−y2)

    //             double x_intersection = (b1_line * c_edgeH2 - b_edgeH2 * c1_line) / (a1_line * b_edgeH2 - a_edgeH2 * b1_line);
    //             double y_intersection = (c1_line * a_edgeH2 - c_edgeH2 * a1_line) / (a1_line * b_edgeH2 - a_edgeH2 * b1_line);
    //             double dist_diff_edg2 = sqrt((x_intersection - edgels_HYPO2(idx_hypo2,0))*(x_intersection - edgels_HYPO2(idx_hypo2,0))+(y_intersection -  edgels_HYPO2(idx_hypo2,1))*(y_intersection - edgels_HYPO2(idx_hypo2,1)));
                
    //             //> Inner two cases: 
    //             if (dist_diff_edg2 < EPIP_TANGENCY_DISPL_THRESH) {
    //                 //> (i) if the displacement after epipolar shift is less than EPIP_TANGENCY_DISPL_THRESH, then feel free to shift it along its direction vector
    //                 corrected_x = x_intersection;
    //                 corrected_y = y_intersection;
    //             }
    //             else {
    //                 //> (ii) if not, then perturb the edge orientation first before shifting the edge along its direction vector

    //                 //////////////// rotate the edge ////////////////
    //                 double p_theta = a1_line * cos(theta) + b1_line * sin(theta);
    //                 double derivative_p_theta = -a1_line * sin(theta) + b1_line * cos(theta);
    //                 // double theta_bar = 0.174533; //10 degrees
                
    //                 if (p_theta * derivative_p_theta > 0){
    //                     theta = theta - ORIENT_PERTURBATION;
    //                 }else{
    //                     theta = theta + ORIENT_PERTURBATION;
    //                 }
    //                 //////////////// rotate the edge ////////////////

    //                 ////////////// calculate the intersection between the tangent and epipolar line //////////////
    //                 double a_edgeH2 = tan(theta); //tan(theta2)
    //                 double b_edgeH2 = -1;
    //                 double c_edgeH2 = -(a_edgeH2*edgels_HYPO2(idx_hypo2,0)-edgels_HYPO2(idx_hypo2,1)); //−(a⋅x2−y2)

    //                 double x_intersection = (b1_line * c_edgeH2 - b_edgeH2 * c1_line) / (a1_line * b_edgeH2 - a_edgeH2 * b1_line);
    //                 double y_intersection = (c1_line * a_edgeH2 - c_edgeH2 * a1_line) / (a1_line * b_edgeH2 - a_edgeH2 * b1_line);
    //                 double dist_diff_edg2 = sqrt((x_intersection - edgels_HYPO2(idx_hypo2,0))*(x_intersection - edgels_HYPO2(idx_hypo2,0))+(y_intersection -  edgels_HYPO2(idx_hypo2,1))*(y_intersection - edgels_HYPO2(idx_hypo2,1)));

    //                 // check if H2 edge's tangent line is almost parallal to epipolar line
    //                 double m_epipolar_edg2 = -a1_line / b1_line;            // Slope of epipolar line
    //                 double angle_diff_rad_edg2 = abs(theta - atan(m_epipolar_edg2));
    //                 double angle_diff_deg_edg2 = angle_diff_rad_edg2 * (180.0 / M_PI);
    //                 if (angle_diff_deg_edg2 > 180){
    //                     angle_diff_deg_edg2 -= 180;
    //                 }
    //                 double angle_diff_original = abs(edgels_HYPO2(idx_hypo2,2) - atan(m_epipolar_edg2))  * (180.0 / M_PI);
    //                 if (angle_diff_original > 180){
    //                     angle_diff_original -= 180;
    //                 }
    //                 if (dist_diff_edg2 < EPIP_TANGENCY_DISPL_THRESH) {
    //                     corrected_x = x_intersection;
    //                     corrected_y = y_intersection;
    //                 } 
    //                 else {
    //                     continue;
    //                 }
    //             }


    //         }

    //         edgels_HYPO2_corrected.conservativeResize(idx_correct+1,10);
    //         edgels_HYPO2_corrected.row(idx_correct) << edgel_HYPO1(0,0), edgel_HYPO1(0,1), edgel_HYPO1(0,2), edgel_HYPO1(0,3), \
    //                                                 corrected_x, corrected_y, edgels_HYPO2(idx_hypo2,2), edgels_HYPO2(idx_hypo2,3), \
    //                                                 HYPO2_idx_raw(idx_hypo2), idx_hypo2;
    //         idx_correct++;

    //         // if (abs(edgel_HYPO1(0,0)-462.853)<0.001  && abs(edgel_HYPO1(0,1)-434.987)<0.001){
    //         //     std::cout<<"distance_epiline for target edge is: "<<distance_epiline<<std::endl;
    //         //     std::cout<<"tengential distance to epipolar line is: "<<tangential_distance<<std::endl;
    //         //     std::cout<< "epipolar line is: "<< a1_line<<" " << b1_line <<" " <<c1_line <<std::endl;
    //         //     std::cout<< "tangent line is: "<< a_edgeH2<<" " << b_edgeH2 <<" " <<c_edgeH2 <<" " <<std::endl;
    //         //     std::cout<< "corrected location is: " <<x_intersection<< ", "<< y_intersection <<std::endl;
    //         //     std::cout<< "angle diff: " << angle_diff_deg_edg2 << " degrees" <<std::endl;
    //         //     std::cout<< "dist diff: " << dist_diff_edg2 << " pixels" <<std::endl;
    //         //     std::cout<< "angle_edgeH2: "<<angle_edgeH2*(180.0 / M_PI)<<std::endl;
    //         //     std::cout<< "atan(m_epipolar): "<<atan(m_epipolar_edg1)*(180.0 / M_PI)<<std::endl;
    //         //     exit(0);
    //         // }
    //     }

    //     return edgels_HYPO2_corrected;
    // }



}

#endif
