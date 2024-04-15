#pragma once

#include "grid.h"
#include <vector>

namespace homo {

	struct MGConfig{
		int reso[3];
		bool enableManagedMem = true;
		std::string namePrefix;
	};

struct MG_H {
	// fine to coarse
	std::vector<std::shared_ptr<Grid_H>> grids;

	MGConfig mgConfig;

	std::shared_ptr<Grid_H> getRootGrid(void);

	void build(MGConfig config);

	void v_cycle(float w_SOR = 1.f, int pre = 1, int post = 1);

	void reset_displacement(void);

	double solveEquation(double tol = 1e-2, bool with_guess = true);

	void updateStencils(void);

};
}
