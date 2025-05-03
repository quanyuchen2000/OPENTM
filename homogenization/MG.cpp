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

		gsGrid0(block_num);
		gsGrid1(block_num);


		double res = norm_host(grids[0]->r_h);
		printf("residual is:%lf\n", res);
		grids[1]->reset_force();
		grids[1]->useGrid_g();
		{
			grids[0]->enforce_vertex_boundary_block(grids[0]->r_h, 0);
			grids[0]->joint_vertex_boundary_block();
			grids[0]->use_block_r_ggs(0);
			grids[0]->enforce_vertex_boundary_block(grids[0]->r_h, 1);
			for (int i = 0; i < block_num - 1; i++) {
				cudaDeviceSynchronize();
				grids[0]->joint_vertex_boundary_block();
				if (i + 2 < block_num) {
					grids[0]->enforce_vertex_boundary_block(grids[0]->r_h, i + 2);
				}
				grids[0]->use_block_r_ggs(i + 1);
				grids[1]->restrict_residual(i);
				std::swap(grids[0]->current, grids[0]->next);
			}
			grids[1]->restrict_residual(block_num - 1);
			grids[1]->pad_vertex_data(grids[1]->f_g);
			grids[1]->reset_displacement();
		}
	}
	grids[1]->gs_relaxation(w_SOR);
	for (int i = 2; i < grids.size(); i++) {
		grids[i - 1]->update_residual();
		grids[i]->restrict_residual();
		grids[i]->reset_displacement();
		if (i == grids.size() - 1) {
			grids[i]->solveHostEquation();
			grids[i]->update_residual();
		}
		else {
			grids[i]->gs_relaxation(w_SOR);
		}
	}

	for (int i = grids.size() - 2; i > 0; i--) {
		grids[i]->prolongate_correction();
		grids[i]->gs_relaxation(w_SOR);
		grids[i]->update_residual();
	}
	if (!grids[0]->use_host_memory) {
		grids[0]->prolongate_correction();
		grids[0]->gs_relaxation(w_SOR);
		grids[0]->update_residual();
	}
	else {
		auto cellReso = grids[0]->cellReso;
		int block_numx = (cellReso[0] / MIN_TRANSFER);
		int block_numy = (cellReso[1] / MIN_TRANSFER);
		int block_numz = (cellReso[2] / MIN_TRANSFER);
		int block_num = block_numx * block_numy * block_numz;
		int block_len = grids[0]->n_gsvertices();
		grids[0]->useGrid_g();

		grids[0]->use_block_u_ggs(0);
		for (int i = 0; i < block_num - 1; i++) {
			cudaDeviceSynchronize();
			if (i >= 1) { grids[0]->write_block_u_ggs(i - 1); }
			grids[0]->use_block_u_ggs(i + 1);
			grids[0]->prolongate_correction(i);
			std::swap(grids[0]->current, grids[0]->next);
		}
		cudaDeviceSynchronize();
		grids[0]->write_block_u_ggs(block_num - 2);
		cudaDeviceSynchronize();
		grids[0]->prolongate_correction(block_num - 1);
		grids[0]->write_block_u_ggs(block_num - 1, false);
		cudaDeviceSynchronize();

		grids[0]->enforce_vertex_boundary(grids[0]->u_h);
		//Grid0(block_num);
		gsGrid0(block_num);
		gsGrid1(block_num);
		grids[0]->enforce_vertex_boundary(grids[0]->r_h);
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
	while ((rel_res > 1e-2 || uch > 1e-6) && iter++ < 200) {
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
		if (rel_res > 10 || iter >= 199) {
			//throw std::runtime_error("numerical failure");
			if (rel_res > 10) {
				printf("\033[31m\nnumerical explode, resetting initial guess...\033[0m\n");
				std::cerr << "\033[31m\nnumerical explode, resetting initial guess...\033[0m\n";
			}
			else {
				printf("\033[31mFailed to converge\033[0m\n");
				std::cerr << "\033[31m\nnumerical explode, resetting initial guess...\033[0m\n";
			}
			overflow_counter--;
			if (overflow_counter > 0) {
			}
			else {
				printf("\033[31mFailed\033[0m\n");
				throw std::runtime_error("MG numerical explode");
			}
			enable_translate_displacement = true;

			auto& gc = *grids.rbegin();
			// write coarsest force
			gc->v_write(getPath("berr"), gc->f_g[0], true);
			// write coarsest system matrix
			std::ofstream ofs(getPath("Khosterr")); ofs << gc->Khost; ofs.close();
			// write solved x
			gc->v_write(getPath("xerr"), gc->u_g[0], true);
			// write gs pos
			gc->writeGsVertexPos(getPath("poserr"));
			grids[0]->reset_displacement();
			grids[0]->writeDensity(getPath("rhoerr"), VoxelIOFormat::openVDB);
			grids[0]->reset_displacement();
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
