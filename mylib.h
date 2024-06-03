// mylib.h
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/Geometry>
// #include <eigen3/MatrixXd>
#include <cmath>
#include <pybind11/stl.h>
#include <vector>
#include <iostream>
#include <chrono>
#include "open3d/Open3D.h"
#include "open3d/t/geometry/RaycastingScene.h"
#include "open3d/io/TriangleMeshIO.h"
#include "open3d/core/Tensor.h"
#include "open3d/core/TensorKey.h"
// #include <TriangleMeshIO.h>

using Eigen::Matrix, Eigen::Dynamic;
using Sparse_matrix = Eigen::SparseMatrix<double>;

typedef Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic> myMatrix;
typedef Eigen::Triplet<double> T;
using namespace std::chrono;

class NearSolver
{

    int height;
    int width;
    int numImages;

    // Eigen::MatrixXd light_poses;
    // Eigen::MatrixXd measurement_matrix;

    Eigen::SparseMatrix<double> A;
    Eigen::MatrixXd intrinsics;
    Eigen::VectorXd B;
    std::vector<std::vector<int>> underconstrainedPoints;
    Eigen::MatrixXd A_pixel;
    Eigen::VectorXd B_pixel;
    Eigen::VectorXd X;

    // create mesh
    open3d::geometry::TriangleMesh legacy_mesh;
    open3d::t::geometry::TriangleMesh mesh;
    open3d::t::geometry::RaycastingScene raycasting_scene;

public:
    NearSolver(int h, int w, int n, std::string mesh_file)
    {
        height = h;
        width = w;
        numImages = n;
        underconstrainedPoints = {};

        // A = Eigen::SparseMatrix<double>(height*width*numImages,height*width*3);
        // B = Eigen::VectorXd(width*height*numImages);
        // X = Eigen::VectorXd(width*height*3);

        height = h;
        width = w;
        numImages = n;

        A = Eigen::SparseMatrix<double>(height * width * numImages, height * width * 3);
        B = Eigen::VectorXd(width * height * numImages);

        // Set all elements of B to zero
        B.setZero();

        X = Eigen::VectorXd(width * height * 3);
        A_pixel = Eigen::MatrixXd(10, 3);
        B_pixel = Eigen::VectorXd(10);

        A.reserve(Eigen::VectorXi::Constant(height * width * 3, numImages));

        // using namespace open3d;
        // PrintOpen3DVersion();
        // raycasting_scene = open3d::t::geometry::RaycastingScene::RaycastingScene();

        /*
        open3d::io::ReadTriangleMeshOptions mesh_options;
        bool validRead = open3d::io::ReadTriangleMeshFromPLY(mesh_file, legacy_mesh, mesh_options);

        if (!validRead)
        {
            std::cout << "Error reading mesh" << std::endl;
        }
        else
        {
            // std::cout << "Mesh read successfully" << std::endl;
        }

        // define const trianglemesh
        mesh = open3d::t::geometry::TriangleMesh::FromLegacy(legacy_mesh);

        // add triangle mesh to raycasting scene
        raycasting_scene.AddTriangles(mesh);

        // print number of triangles
        open3d::core::Tensor point = open3d::core::Tensor::Zeros({1, 3}, open3d::core::Dtype::Float32, open3d::core::Device("CPU:0"));
        open3d::core::Tensor distance_to_mesh = raycasting_scene.ComputeDistance(point);

        float distance = distance_to_mesh[0].Item<float>();
        */

        // std::cout << "Number of triangles: " << mesh.GetTriangleIndices().GetLength() << std::endl;
        // std::cout << "Distance to mesh: " << distance << std::endl;

        // core::Tensor open3d::t::geometry::RaycastingScene::CreateRaysPinhole()
    }

    void setCameraMatrix(Eigen::MatrixXd cameraMatrix)
    {
        intrinsics = cameraMatrix;
    }

    /*
    Eigen::MatrixXd getVisiblePixels(Eigen::Vector3d light_position, Eigen::MatrixXd depth_image)
    {
        Eigen::MatrixXd visiblePixels = Eigen::MatrixXd::Zero(height, width);
        open3d::core::Tensor points = open3d::core::Tensor::Zeros({height * width, 3}, open3d::core::Dtype::Float32, open3d::core::Device("CPU:0"));

        for (int i = 0; i < height; i++)
        {
            for (int j = 0; j < width; j++)
            {
                double depth = depth_image.coeff(i, j);

                if (depth < 0.01)
                {
                    continue;
                }

                double x = ((j - intrinsics.coeff(0, 2)) / intrinsics.coeff(0, 0)) * depth;
                double y = ((i - intrinsics.coeff(1, 2)) / intrinsics.coeff(1, 1)) * depth;
                double z = depth;

                Eigen::Vector3d point(x, y, z);
                open3d::core::Tensor pointTensor = open3d::core::eigen_converter::EigenMatrixToTensor(point);
                points[i * width + j] = pointTensor.To(open3d::core::Dtype::Float32).Reshape({3});
            }
        }

        // std::cout << "After generating points" << std::endl;
        open3d::core::Tensor lightTensor = open3d::core::eigen_converter::EigenMatrixToTensor(light_position);
        lightTensor = lightTensor.To(open3d::core::Dtype::Float32);
        lightTensor = lightTensor.Reshape({1, 3});
        open3d::core::Tensor light_diff = points - lightTensor;
        open3d::core::Tensor ones = open3d::core::Tensor::Ones({height * width, 3}, open3d::core::Dtype::Float32, open3d::core::Device("CPU:0")) * lightTensor;

        // std::cout << "After generating light diff" << std::endl;

        // convert to eigen
        Eigen::MatrixXd light_diff_eigen = open3d::core::eigen_converter::TensorToEigenMatrixXd(light_diff);
        // std::cout << "After converting to eigen" << std::endl;

        // normalize light_diff row wise
        light_diff_eigen = light_diff_eigen.rowwise().normalized();
        Eigen::MatrixXd ones_eigen = open3d::core::eigen_converter::TensorToEigenMatrixXd(ones);

        // concatenate light_diff and ones
        Eigen::MatrixXd combined_light_diff(height * width, 6);
        combined_light_diff << ones_eigen, light_diff_eigen;

        // std::cout << combined_light_diff << std::endl;

        // convert back to tensor
        open3d::core::Tensor combined_light_diff_tensor = open3d::core::eigen_converter::EigenMatrixToTensor(combined_light_diff);

        combined_light_diff_tensor = combined_light_diff_tensor.To(open3d::core::Dtype::Float32);

        // cast rays
        // std::cout << "Castin rays" << std::endl;
        std::unordered_map<std::string, open3d::core::Tensor> result = raycasting_scene.CastRays(combined_light_diff_tensor);
        open3d::core::Tensor t_hit = result["t_hit"];
        std::unordered_map<std::string, open3d::core::Tensor> closest_point = raycasting_scene.ComputeClosestPoints(points);

        // std::cout << "After casting rays" << std::endl;

        // print out t_hit size
        // std::cout << "t_hit size: " << t_hit.GetLength() << std::endl;
        // std::cout << "Closest point size: " << closest_point["points"].GetLength() << std::endl;

        open3d::core::Tensor closest_triangle_points = closest_point["points"];

        // convert to eigen
        Eigen::MatrixXd closest_triangle_points_eigen = open3d::core::eigen_converter::TensorToEigenMatrixXd(closest_triangle_points);

        open3d::core::Tensor isInf = t_hit.IsInf();

        for (int i = 0; i < height; i++)
        {
            for (int j = 0; j < width; j++)
            {
                if (depth_image.coeff(i, j) < 0.01)
                {
                    continue;
                }

                if (isInf[i * width + j].Item<bool>())
                {
                    visiblePixels(i, j) = 1;
                    continue;
                }

                float distance = t_hit[i * width + j].Item<float>();
                Eigen::Vector3d closest_point_eigen = closest_triangle_points_eigen.row(i * width + j);

                double depth_original = depth_image.coeff(i, j);
                double x_og = ((j - intrinsics.coeff(0, 2)) / intrinsics.coeff(0, 0)) * depth_original;
                double y_og = ((i - intrinsics.coeff(1, 2)) / intrinsics.coeff(1, 1)) * depth_original;
                double z_og = depth_original;

                Eigen::Vector3d point_og(x_og, y_og, z_og);

                float norm = (light_position - point_og).norm(); // Predicted distance between depth point and light source
                float difference = norm - distance;

                if (difference > 0.005)
                {
                    continue;
                }
                else
                {
                    visiblePixels(i, j) = 1;
                }
            }
        }
        return visiblePixels;
    }
    */
    /*
    bool isOccluded(Eigen::Vector3d point, Eigen::Vector3d light_pos)
    {
        // open3d::core::Tensor intrinsics_tensor = open3d::core::eigen_converter::EigenMatrixToTensor(intrinsics);
        // open3d::core::Tensor extrinsics_tensor = open3d::core::Tensor::Eye(4,open3d::core::Dtype::Float64,open3d::core::Device("CPU:0"));
        // open3d::core::Tensor rays = raycasting_scene.CreateRaysPinhole(intrinsics_tensor,extrinsics_tensor,width,height);

        open3d::core::Tensor pointTensor = open3d::core::eigen_converter::EigenMatrixToTensor(point);
        pointTensor = pointTensor.To(open3d::core::Dtype::Float32);
        Eigen::Vector3d light_vector = (point - light_pos).normalized();
        Eigen::Vector3d origin = light_pos;
        Eigen::VectorXd vec_joined(light_vector.size() + origin.size());
        vec_joined << origin, light_vector;
        open3d::core::Tensor light_vector_tensor = open3d::core::eigen_converter::EigenMatrixToTensor(vec_joined);

        // Transpose
        light_vector_tensor = light_vector_tensor.T();

        // Convert to float32
        light_vector_tensor = light_vector_tensor.To(open3d::core::Dtype::Float32);

        // open3d::core::Tensor occluded =  raycasting_scene.TestOcclusions(light_vector_tensor);
        // bool is_occluded = occluded[0].Item<bool>();

        std::unordered_map<std::string, open3d::core::Tensor> result = raycasting_scene.CastRays(light_vector_tensor);
        open3d::core::Tensor t_hit = result["t_hit"];

        // check if t_hit is INF
        if (t_hit.IsInf()[0].Item<bool>())
        {
            return false;
        }

        float distance = t_hit[0].Item<float>();
        std::unordered_map<std::string, open3d::core::Tensor> closest_point = raycasting_scene.ComputeClosestPoints(pointTensor.T());
        Eigen::Vector3d closest_point_eigen = open3d::core::eigen_converter::TensorToEigenVector3dVector(closest_point["points"])[0];

        float norm = (light_pos - closest_point_eigen).norm(); // Predicted distance between depth point and light source
        float difference = norm - distance;

        if (difference > 0.001)
        {
            return true;
        }
        else
        {
            return false;
        }
    }
    */
    /*
    Eigen::MatrixXd getValidLights(Eigen::Vector3d point, Eigen::MatrixXd light_pos)
    {
        Eigen::MatrixXd validLights = Eigen::MatrixXd::Zero(light_pos.rows(), light_pos.cols());

        open3d::core::Tensor pointTensor = open3d::core::eigen_converter::EigenMatrixToTensor(point);

        std::unordered_map<std::string, open3d::core::Tensor> closest_triangle = raycasting_scene.ComputeClosestPoints(pointTensor);

        open3d::core::Tensor closest_point = closest_triangle["points"][0];

        open3d::core::Tensor light_pos_tensor = open3d::core::eigen_converter::EigenMatrixToTensor(light_pos);

        light_pos_tensor = light_pos_tensor.To(open3d::core::Dtype::Float32);

        pointTensor = pointTensor.To(open3d::core::Dtype::Float32);

        open3d::core::Tensor light_diff = pointTensor - light_pos_tensor;

        // Convert to eigen

        Eigen::MatrixXd light_diff_eigen = open3d::core::eigen_converter::TensorToEigenMatrixXd(light_diff);

        // normalize light_diff

        light_diff_eigen = light_diff_eigen.normalized();

        // concatenate light_diff and ones

        Eigen::MatrixXd combined_light_diff(light_diff_eigen.rows(), light_diff_eigen.cols() + 3);

        combined_light_diff << light_pos, light_diff_eigen;

        // convert back to tensor

        open3d::core::Tensor combined_light_diff_tensor = open3d::core::eigen_converter::EigenMatrixToTensor(combined_light_diff);

        // cast rays

        std::unordered_map<std::string, open3d::core::Tensor> result = raycasting_scene.CastRays(combined_light_diff_tensor);

        open3d::core::Tensor t_hit = result["t_hit"];

        for (int i = 0; i < light_pos.rows(); i++)
        {
            if (t_hit[i].IsInf()[0].Item<bool>())
            {
                continue;
            }

            float distance = t_hit[i].Item<float>();

            Eigen::VectorXd closest_point_eigen = open3d::core::eigen_converter::TensorToEigenVector3dVector(closest_triangle["points"])[0];

            float norm = (light_pos.row(i) - closest_point_eigen).norm(); // Predicted distance between depth point and light source

            float difference = norm - distance;

            if (difference > 0.005)
            {
                continue;
            }
            else
            {

                validLights.row(i) = light_pos.row(i);
            }
        }

        return validLights;
    }
    */

    Eigen::MatrixXd attenuateM(Eigen::MatrixXd light_poses, Eigen::MatrixXd depth_image, Eigen::MatrixXd measurement)
    {

        //create eigen matrix with same size as measurement

        Eigen::MatrixXd attenuatedMeasurement = Eigen::MatrixXd::Zero(measurement.rows(),measurement.cols());
        
        open3d::core::Tensor points = open3d::core::Tensor::Zeros({height*width,3},open3d::core::Dtype::Float32,open3d::core::Device("CPU:0"));

       
        for(int i = 0; i < height;i++)
        {
            for(int j = 0; j < width;j++)
            {
                
                double depth = depth_image.coeff(i,j);

                if(depth < 0.01)
                {
                    continue;
                }

                double x =((j - intrinsics.coeff(0, 2))/intrinsics.coeff(0, 0))*depth;
                double y = ((i - intrinsics.coeff(1, 2))/intrinsics.coeff(1, 1))*depth;
                double z = depth;

              

                Eigen::Vector3d point(x,y,z);

               
                open3d::core::Tensor pointTensor = open3d::core::eigen_converter::EigenMatrixToTensor(point);

                points[i*width+j] = pointTensor.To(open3d::core::Dtype::Float32).Reshape({3});


            }
        }

        std::unordered_map< std::string, open3d::core::Tensor > closest_points = raycasting_scene.ComputeClosestPoints(points);

        open3d::core::Tensor normals = closest_points["primitive_normals"];

        Eigen::MatrixXd normals_eigen = open3d::core::eigen_converter::TensorToEigenMatrixXd(normals);

        Eigen::MatrixXd points_eigen = open3d::core::eigen_converter::TensorToEigenMatrixXd(points);

  

        int point_index = 0;
        int lightPerPoint = 0;
        std::vector<int> underconstrained;
        int b_index = 0;

        int counter_temp = 0;

   
        for(int i = 0; i < height;i++)
        {
            for(int j = 0; j < width;j++)
            {



                Eigen::Vector3d normal = normals_eigen.row(i*width+j);

           
                Eigen::Vector3d point = points_eigen.row(i*width+j);

             
                if(point[2] < 0.01)
                {
                    point_index++;
                    continue;
                }


              
                for(int k = 0; k < numImages; k++)
                {

                    b_index = point_index*numImages + k;

                   
                    if (measurement.coeff(i*width+j, k) == 0)
                    {
                        // counter_temp++;
                        
                        //B(b_index) = 0;
                        
                        continue;
                    }
                    

                    
                 
                    Eigen::Ref<Eigen::Vector3d> light_position(Eigen::Map<Eigen::Vector3d>(light_poses.block(4*k,3,3,1).data()));
                
                    int col = point_index*3;
                    int row = point_index*numImages;

                    double norm =(light_position-point).norm();

                    Eigen::Vector3d normalized = (light_position-point)/norm;

                    Eigen::Vector3d point2cam = -point.normalized();

           
             
                    Eigen::Ref<Eigen::Vector3d> z_axis(Eigen::Map<Eigen::Vector3d>(light_poses.block(4*k,3,3,1).data()));

                    //double angularAttenuationCoefficient = 1/normalized.dot(z_axis);
                    double angularAttenuationCoefficient = 1;
                    double radialAttenuationCoefficient = norm*norm;


                    attenuatedMeasurement.coeffRef(i*width+j,k) = measurement.coeff(i*width+j,k)*radialAttenuationCoefficient*angularAttenuationCoefficient;

                }

               
              
                point_index++;


               
            }

        }
        
        return attenuatedMeasurement;
    

    }


    void generateMatrices(Eigen::MatrixXd light_poses, Eigen::MatrixXd depth_image, Eigen::MatrixXd measurement)
    {

        open3d::core::Tensor points = open3d::core::Tensor::Zeros({height * width, 3}, open3d::core::Dtype::Float32, open3d::core::Device("CPU:0"));

        int temp_point_counter = 0;

        for (int i = 0; i < height; i++)
        {
            for (int j = 0; j < width; j++)
            {

                double depth = depth_image.coeff(i, j);

                if (depth < 0.01)
                {
                    continue;
                }

                double x = ((j - intrinsics.coeff(0, 2)) / intrinsics.coeff(0, 0)) * depth;
                double y = ((i - intrinsics.coeff(1, 2)) / intrinsics.coeff(1, 1)) * depth;
                double z = depth;

                Eigen::Vector3d point(x, y, z);

                open3d::core::Tensor pointTensor = open3d::core::eigen_converter::EigenMatrixToTensor(point);

                points[i * width + j] = pointTensor.To(open3d::core::Dtype::Float32).Reshape({3});
            }
        }

        //std::cout << "After generating points" << std::endl;
        //std::cout << "Number of points: " << temp_point_counter << std::endl;
        //std::cout << "points generated" << std::endl;

        std::unordered_map<std::string, open3d::core::Tensor> closest_points = raycasting_scene.ComputeClosestPoints(points);

        open3d::core::Tensor normals = closest_points["primitive_normals"];

        Eigen::MatrixXd normals_eigen = open3d::core::eigen_converter::TensorToEigenMatrixXd(normals);

        Eigen::MatrixXd points_eigen = open3d::core::eigen_converter::TensorToEigenMatrixXd(points);

        // print out normals tensor size

        //std::cout << "Normals size: " << normals.GetLength() << std::endl;
        //std::cout << "Normals generated" << std::endl;

        // Light poses is numberImages*4x4
        // depth is height*width
        // N_0 is height*width x 3

        // int point_nonzero_counter = 0;

        // for  (int i = 0; i < width*height; i++)
        // {
        //     if(measurement.coeff(i,1) != 0)
        //     {
        //         point_nonzero_counter++;
        //     }
        // }

        //std::cout << "Number of non zero points: " << point_nonzero_counter << std::endl;
        //std::cout << "Starting matrix generation" << std::endl;

        std::vector<T> tripletList;
        tripletList.reserve(height * width * numImages * 3);

        //std::cout << "After reserve" << std::endl;

        int point_index = 0;
        int lightPerPoint = 0;
        std::vector<int> underconstrained;
        int b_index = 0;

        int counter_temp = 0;

        for (int i = 0; i < height; i++)
        {
            for (int j = 0; j < width; j++)
            {

                Eigen::Vector3d normal = normals_eigen.row(i * width + j);

                Eigen::Vector3d point = points_eigen.row(i * width + j);

                if (point[2] < 0.01)
                {
                    point_index++;
                    continue;
                }

                for (int k = 0; k < numImages; k++)
                {

                    b_index = point_index * numImages + k;

                    if (measurement.coeff(i * width + j, k) == 0)
                    {
                        // counter_temp++;

                        // B(b_index) = 0;

                        continue;
                    }

                    Eigen::Ref<Eigen::Vector3d> light_position(Eigen::Map<Eigen::Vector3d>(light_poses.block(4 * k, 3, 3, 1).data()));

                    int col = point_index * 3;
                    int row = point_index * numImages;

                    double norm = (light_position - point).norm();

                    Eigen::Vector3d normalized = (light_position - point) / norm;

                    Eigen::Vector3d point2cam = -point.normalized();

                    // Find angle between normal and normalized

                    double angle = std::acos(normal.dot(normalized));

                    // convert angle to degree
                    //! Test
                    angle = angle * 180 / 3.14159;

                    // if (angle > 80)
                    // {
                    //     B(b_index) = 0;
                    //     continue;
                    // }

                    // rotate normalized around normal by 180 degrees

                    Eigen::AngleAxisd rotation(3.14159, normal);

                    Eigen::Vector3d rotated_normalized = rotation.toRotationMatrix() * normalized;

                    double specular_angle = std::acos(point2cam.dot(rotated_normalized));

                    // convert angle to degree

                    specular_angle = specular_angle * 180 / 3.14159;
                    // //! Test
                    // if (specular_angle < 20)
                    // {
                    //     B(b_index) = 0;
                    //     continue;
                    // }

                    // int* ptr = nullptr;
                    // *ptr = 10;

                    tripletList.push_back(T(row + k, col, normalized.coeff(0)));
                    tripletList.push_back(T(row + k, col + 1, normalized.coeff(1)));
                    tripletList.push_back(T(row + k, col + 2, normalized.coeff(2)));

                    Eigen::Ref<Eigen::Vector3d> z_axis(Eigen::Map<Eigen::Vector3d>(light_poses.block(4 * k, 3, 3, 1).data()));

                    // A_pixel.row(k) = normalized.transpose();

                    //! Change for RPCAnear
                    //double angularAttenuationCoefficient = 1 / normalized.dot(z_axis);
                    double angularAttenuationCoefficient = 1;
                    //double radialAttenuationCoefficient = norm * norm;

                    double radialAttenuationCoefficient = 1;

                    // B_pixel(k) = measurement(b_index)*radialAttenuationCoefficient*angularAttenuationCoefficient;

                    B(b_index) = measurement.coeff(i * width + j, k)*radialAttenuationCoefficient*angularAttenuationCoefficient;

                    lightPerPoint++;

                    // std::cout << "B: " << B(b_index) << std::endl;
                }

                //if (lightPerPoint < 11)
                //{
                    // std::cout << "Underconstrained point: " << i << " ," << j << std::endl;
                    // std::cout << "Number of lights: " << lightPerPoint << std::endl;

                underconstrained.push_back(i);
                underconstrained.push_back(j);
                underconstrained.push_back(lightPerPoint);
                underconstrainedPoints.push_back(underconstrained);
                underconstrained.clear();
                //}

                // Eigen::JacobiSVD<Eigen::MatrixXd> svd;
                // svd.compute(A_pixel, Eigen::ComputeThinV | Eigen::ComputeThinU);

                // Eigen::VectorXd res = svd.solve(B_pixel);

                // X.coeffRef(point_index*3) = res.coeff(0);
                // X.coeffRef(point_index*3+1) = res.coeff(1);
                // X.coeffRef(point_index*3+2) = res.coeff(2);

                lightPerPoint = 0;
                point_index++;
            }
        }

        A.setFromTriplets(tripletList.begin(), tripletList.end());

        A.makeCompressed();

        //std::cout << "After set from triplets" << std::endl;

        // std::cout << "counter temp: " << counter_temp << std::endl;

        //std::cout << "Number of underconstrained points: " << underconstrainedPoints.size() << std::endl;
    }

    Eigen::SparseMatrix<double> &getA()
    {
        return A;
    }

    Eigen::VectorXd &getB()
    {
        return B;
    }

    Eigen::VectorXd &getX()
    {
        return X;
    }

    std::vector<std::vector<int>> &getUc()
    {
        return underconstrainedPoints;
    }
};
