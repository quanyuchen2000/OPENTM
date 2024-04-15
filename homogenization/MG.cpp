#include "MG.h"
#include <fstream>
#include "Eigen/IterativeLinearSolvers"
#include "matlab/matlab_utils.h"
#include "tictoc.h"
#include "utils.h"

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

void homo::MG_H::v_cycle(float w_SOR /*= 1.f*/, int pre /*= 1*/, int post /*= 1*/)
{
	for (int i = 0; i < grids.size(); i++) {
		if (i != 0) {
			grids[i - 1]->update_residual();
			grids[i]->restrict_residual();
			grids[i]->reset_displacement();
		}
		if (i == grids.size() - 1) {
			grids[i]->solveHostEquation();
		}
		else {
			grids[i]->gs_relaxation(w_SOR);
		}
	}

	for (int i = grids.size() - 2; i >= 0; i--) {
		grids[i]->prolongate_correction();
		grids[i]->gs_relaxation(w_SOR);
	}

	// not necessary here
	//grids[0]->update_residual();

	return /*grids[0]->relative_residual()*/;
}

void homo::MG_H::reset_displacement(void)
{
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
	double fnorm = grids[0]->v_norm(grids[0]->f_g[0]);
	int overflow_counter = 2;
	bool enable_translate_displacement = false;
	std::vector<double> errlist;
	double uch = 1e-7;
	while ((rel_res > tol || uch > 1e-6) && iter++ < 200) {
#if 1
		v_cycle(1);
#else
		grids[0]->gs_relaxation(1.6);
		grids[0]->update_residual();
		rel_res = grids[0]->relative_residual();
#endif
		if (enable_translate_displacement) grids[0]->translateForce(2, grids[0]->u_g);
		rel_res = grids[0]->residual() / (fnorm + 1e-10);
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
	printf("\n");
	if (iter >= 200) { printf(" - r_rel = %le\n", rel_res); }
#else
	rel_res = pcg();
#endif
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
