#include "MG.h"
#include <fstream>
#include "Eigen/IterativeLinearSolvers"
#include "matlab/matlab_utils.h"
#include "tictoc.h"
#include "utils.h"
#include "cpuFramework.h"

std::shared_ptr<homo::Grid_H> homo::MG_H::getRootGrid(void)
{
	return grids[0];
}

void homo::MG_H::build(MGConfig config)
{
	mgConfig = config;

	std::shared_ptr<Grid_H> rootGrid(new Grid_H());
	if (!rootGrid) { throw std::runtime_error("failed to create root Grid"); }

	GridConfig gcon;
	gcon.enableManagedMem = config.enableManagedMem;
	gcon.namePrefix = config.namePrefix;

	rootGrid->buildRoot(config.reso[0], config.reso[1], config.reso[2], gcon);

	grids.emplace_back(rootGrid);

	// coarse grid until enough
	auto coarseGrid = rootGrid->coarse2(gcon);

	if (!coarseGrid) { throw std::runtime_error("failed to create coarse Grid"); }

	while (coarseGrid) {
		grids.push_back(coarseGrid);
		coarseGrid = coarseGrid->coarse2(gcon);
	}

	// reset all vectors
	for (int i = 0; i < grids.size(); i++) {
		grids[i]->reset_residual();
		grids[i]->reset_displacement();
		grids[i]->reset_force();
	}
}
void homo::MG_H::unregist()
{
	for (int t = 0; t < 3; t++) {
		cudaHostUnregister(grids[0]->uchar[t].data());
	}
	for (int i = 0; i < grids.size(); i++) {
		cudaHostUnregister(grids[i]->f_h.data());
		cudaHostUnregister(grids[i]->r_h.data());
	}
}
void homo::MG_H::Grid0(int block_num) {
	for (int i = 0; i < block_num; i++) {
		// give in rho_g u_g
		grids[0]->use_block_rho(i);
		grids[0]->use_block_u_g(i);
		grids[0]->use_block_f_g(i);
		cudaDeviceSynchronize();
		float* tmp;
		tmp = getMem().getBuffer("temp_rho0")->data<float>();
		grids[0]->update_host(tmp);
		grids[0]->gs_relaxation_host(i);
		grids[0]->write_block_u_g(i);
		cudaDeviceSynchronize();
	}
	grids[0]->enforce_vertex_boundary(grids[0]->u_h);
	for (int i = 0; i < block_num; i++) {
		grids[0]->use_block_rho(i);
		grids[0]->use_block_u_g(i);
		grids[0]->use_block_f_g(i);
		cudaDeviceSynchronize();
		float* tmp;
		tmp = getMem().getBuffer("temp_rho0")->data<float>();
		grids[0]->update_host(tmp);
		grids[0]->update_residual_host(i);
		grids[0]->write_block_r_g(i);
		cudaDeviceSynchronize();
	}
	grids[0]->enforce_vertex_boundary(grids[0]->r_h);
}
void homo::MG_H::CountU(int block_num) {
	grids[0]->u_h;
	int count233 = 0;
	int count433 = 0;
	for (auto i : grids[0]->u_h) {
		if (abs(i - 233.) < 1e-6) {
			count233++;
		}
		if (abs(i - 433.) < 1e-6) {
			count433++;
		}
	}
	std::cout << "233:" << count233 << "433:" << count433 << std::endl;
}
void homo::MG_H::gsGrid0(int block_num) {
	grids[0]->enforce_vertex_boundary_block(grids[0]->u_h, 0);
	grids[0]->joint_vertex_boundary_block();
	grids[0]->use_block_rhogs(0);
	grids[0]->use_block_u_ggs(0);
	grids[0]->use_block_f_ggs(0);
	grids[0]->enforce_vertex_boundary_block(grids[0]->u_h, 1);
	for (int i = 0; i < block_num - 1; i++) {
		// give in rho_g u_g
		cudaDeviceSynchronize();
		grids[0]->joint_vertex_boundary_block();
		if (i + 2 < block_num) {
			grids[0]->enforce_vertex_boundary_block(grids[0]->u_h, i + 2);
		}
		if (i >= 1) {
			grids[0]->write_block_u_ggs(i - 1);
		}
		grids[0]->use_block_rhogs(i + 1);
		grids[0]->use_block_u_ggs(i + 1);
		grids[0]->use_block_f_ggs(i + 1);
		float* tmp;
		if (grids[0]->current) {
			tmp = getMem().getBuffer("temp_rho0")->data<float>();
		}
		else {
			tmp = getMem().getBuffer("temp_rho1")->data<float>();
		}
		grids[0]->update_hostgs(tmp);
		grids[0]->gs_relaxation_host(i);
		std::swap(grids[0]->current, grids[0]->next);
	}
	cudaDeviceSynchronize();
	grids[0]->write_block_u_ggs(block_num - 2);

	float* tmp;
	if (grids[0]->current) {
		tmp = getMem().getBuffer("temp_rho0")->data<float>();
	}
	else {
		tmp = getMem().getBuffer("temp_rho1")->data<float>();
	}
	grids[0]->update_hostgs(tmp);
	grids[0]->gs_relaxation_host(block_num - 1);
	grids[0]->write_block_u_ggs(block_num - 1, false);
	cudaDeviceSynchronize();
}
void homo::MG_H::gsGrid1(int block_num) {
	grids[0]->enforce_vertex_boundary_block(grids[0]->u_h, 0);
	grids[0]->joint_vertex_boundary_block();
	grids[0]->use_block_rhogs(0);
	grids[0]->use_block_u_ggs(0);
	grids[0]->use_block_f_ggs(0);
	grids[0]->enforce_vertex_boundary_block(grids[0]->u_h, 1);
	// prepare for padding of block 1
	for (int i = 0; i < block_num - 1; i++) {
		cudaDeviceSynchronize();
		grids[0]->joint_vertex_boundary_block();
		// joint the padding of i+1
		// hang the request for prepare padding for block i+2
		if (i + 2 < block_num) {
			grids[0]->enforce_vertex_boundary_block(grids[0]->u_h, i+2);
		}
		if (i >= 1) {
			grids[0]->write_block_r_ggs(i - 1);
		}
		grids[0]->use_block_rhogs(i + 1);
		grids[0]->use_block_u_ggs(i + 1);
		grids[0]->use_block_f_ggs(i + 1);
		float* tmp;
		if (grids[0]->current) {
			tmp = getMem().getBuffer("temp_rho0")->data<float>();
		}
		else {
			tmp = getMem().getBuffer("temp_rho1")->data<float>();
		}
		grids[0]->update_hostgs(tmp);
		grids[0]->update_residual_host(i);
		std::swap(grids[0]->current, grids[0]->next);
	}
	cudaDeviceSynchronize();
	grids[0]->write_block_r_ggs(block_num - 2);
	float* tmp;
	if (grids[0]->current) {
		tmp = getMem().getBuffer("temp_rho0")->data<float>();
	}
	else {
		tmp = getMem().getBuffer("temp_rho1")->data<float>();
	}
	grids[0]->update_hostgs(tmp);
	grids[0]->update_residual_host(block_num - 1);
	grids[0]->write_block_r_ggs(block_num - 1, false);
	cudaDeviceSynchronize();
}
void homo::MG_H::v_cycle(float w_SOR /*= 1.f*/, int pre /*= 1*/, int post /*= 1*/)
{
	if (!grids[0]->use_host_memory) {
		grids[0]->gs_relaxation(w_SOR);
		grids[0]->update_residual();

		grids[1]->restrict_residual();
		grids[1]->reset_displacement();
	}
	else {
		// for blocks use gs_relaxation
		auto cellReso = grids[0]->cellReso;
		int block_numx = (cellReso[0] / MIN_TRANSFER);
		int block_numy = (cellReso[1] / MIN_TRANSFER);
		int block_numz = (cellReso[2] / MIN_TRANSFER);
		int block_num = block_numx * block_numy * block_numz;
		int block_len = grids[0]->n_gsvertices();
		grids[0]->useGrid_g();
		for (int i = 0; i < 1000; i++) {
			CountU(block_num);
			gsGrid0(block_num);
			gsGrid1(block_num);

			double res = norm_host(grids[0]->r_h);
			printf("residual is:%lf\n", res);
		}
	}
	return;
}

void homo::MG_H::reset_displacement(void)
{
	grids[0]->uchar.resize(3);
	for (auto i : { 0, 1, 2 }) {
		grids[0]->uchar[i].resize(grids[0]->uchar[i].size(), 0);
	}
	for (int i = 0; i < grids.size(); i++) {
		grids[i]->reset_displacement();
	}
}

double homo::MG_H::solveEquation(double tol /*= 1e-2*/, bool with_guess /*= true*/)
{
	double rel_res = 1;
	int iter = 0;
	if (!with_guess) { grids[0]->reset_displacement(); }
#if 1
	double fnorm;
	if (!grids[0]->use_host_memory) {
		fnorm = grids[0]->v_norm(grids[0]->f_g[0]);
	}
	else {
		fnorm = norm_host(grids[0]->f_h);
	}
	int overflow_counter = 2;
	bool enable_translate_displacement = false;
	std::vector<double> errlist;
	double uch = 1e-7;
	while ((rel_res > 1e-2 || uch > 1e-6) && iter++ < 20) {
		v_cycle(1);
#else
		while (1) {
			grids[0]->gs_relaxation(1.);
			grids[0]->update_residual();
			rel_res = grids[0]->residual();
			printf("residual:%f\n", rel_res);
		}
#endif
		if (enable_translate_displacement) grids[0]->translateForce(2, grids[0]->u_g);
		if (grids[0]->use_host_memory) {
			rel_res = norm_host(grids[0]->r_h) / (fnorm + 1e-10);
		}
		else {
			rel_res = grids[0]->residual() / (fnorm + 1e-10);
		}
		errlist.emplace_back(rel_res);
		printf("rel_res = %4.2lf%%    It.%d       \r", rel_res * 100, iter);
	}
	//grids[0]->enforce_vertex_boundary(grids[0]->u_h);
	printf("\n");
	if (iter >= 200) { printf(" - r_rel = %le\n", rel_res); }

	return rel_res;
}

void homo::MG_H::updateStencils(void) {
	for (int i = 1; i < grids.size(); i++) {
		grids[i]->restrict_stencil();
		if (i == grids.size() - 1) {
			grids[i]->assembleHostMatrix();
		}
	}
}
