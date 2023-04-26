#include "stdafx.h"
#include <teaser/teaser_registration.h>

// __declspec(dllexport) void __cdecl
extern "C" int run(int n_src, int n_tgt, float* src, float* tgt, float* g)
{
	Eigen::Matrix<double, 3, Eigen::Dynamic> src_mat(3, n_src);
	Eigen::Matrix<double, 3, Eigen::Dynamic> tgt_mat(3, n_tgt);
	for (int i = 0; i < n_src; i++)
		for (int j = 0; j < 3; j++)
			src_mat(j, i) = src[i * 6 + j];
	for (int i = 0; i < n_tgt; i++)
		for (int j = 0; j < 3; j++)
			tgt_mat(j, i) = tgt[i * 6 + j];
	teaser::RobustRegistrationSolver::Params params;
	params.noise_bound = 0.02;
	params.cbar2 = 1;
	params.estimate_scaling = false;
	params.rotation_max_iterations = 100;
	params.rotation_gnc_factor = 1.4;
	params.rotation_estimation_algorithm =
		teaser::RobustRegistrationSolver::ROTATION_ESTIMATION_ALGORITHM::GNC_TLS;
	params.rotation_cost_threshold = 0.005;

	// Solve with TEASER++
	teaser::RobustRegistrationSolver solver(params);
	auto solution = solver.solve(src_mat, tgt_mat);
	if (!solution.valid)
		return 0;

	Eigen::Matrix4f Tf0 = Eigen::Matrix4f::Identity();
	
	Tf0.block(0, 0, 3, 3) = solution.rotation.cast<float>();
	Tf0.block(0, 3, 3, 1) = solution.translation.cast<float>();
	for (int i = 0; i < 4; i++)
		for (int j = 0; j < 4; j++)
			g[4 * i + j] = Tf0(i, j);
	return (int)solver.getTranslationInliers().size();
}

/*int deprecate() {
	// Load the .ply file. Both load_bunny() and load_kitti() are OK. 
	// string sFile_src, sFile_tgt; 
	// load_bunny(sFile_src, sFile_tgt);
	pcl::PointCloud<pcl::PointNormal>::Ptr pcl_cloud_src(new pcl::PointCloud<pcl::PointNormal>());
	pcl::PointCloud<pcl::PointNormal>::Ptr pcl_cloud_tgt(new pcl::PointCloud<pcl::PointNormal>());
    // read_cloud(sFile_src, pcl_cloud_src);
	// read_cloud(sFile_tgt, pcl_cloud_tgt);
	//// compute normals if necessary. the default method is KNN-based normal computation.
	if ( false == CNormalEstimator::is_valid_normal(pcl_cloud_src) ) {
		CNormalEstimator neEstimator; 
		printf_s("compute normals, ptsNum = %05d....\n", pcl_cloud_src->points.size() );
		neEstimator.loadInPlace(pcl_cloud_src);
	}
	if ( false == CNormalEstimator::is_valid_normal(pcl_cloud_tgt) ) {
		CNormalEstimator neEstimator;
		printf_s("compute normals, ptsNum = %05d....\n", pcl_cloud_tgt->points.size());
		neEstimator.loadInPlace(pcl_cloud_tgt);
	}
	//// start match. 
	std::shared_ptr<CTEASER> pTeaser(new CTEASER());
	pTeaser->setParams( string("Teaser") ); 
	//// set the sampling and feature radius. 
	double dRes = computeModelResFun(pcl_cloud_tgt);
	printf_s("model resolution = %.4f\n", dRes); 
	pTeaser->setRadius(5.0 * dRes, 10.0 * dRes); 
	//// set input cloud. 
	pTeaser->setInputSource(pcl_cloud_src);
	pTeaser->setInputTarget(pcl_cloud_tgt); 
	pcl::PointCloud<pcl::PointNormal>::Ptr cloud_aligned(new pcl::PointCloud<pcl::PointNormal>());
	Eigen::Matrix4f Tf0 = Eigen::Matrix4f::Identity(); 
	pTeaser->align(*cloud_aligned, Tf0); 

	//// save for visualization. 
	pcl::PCDWriter pcdWriter;
	string sSaveFile = "cloud_src.pcd";
	pcdWriter.writeBinary(sSaveFile, *pcl_cloud_src);
	sSaveFile = "cloud_tgt.pcd";
	pcdWriter.writeBinary(sSaveFile, *pcl_cloud_tgt);
	sSaveFile = "cloud_aft.pcd";
	pcdWriter.writeBinary(sSaveFile, *cloud_aligned);
	return 0; 
}*/
