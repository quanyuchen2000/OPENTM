#include "cmdline.h"
#include <fstream>

void cfg::HomoConfig::init()
{
	reso[0] = 128; // resolution
	reso[1] = 128;
	reso[2] = 128;
	sym = Symmetry::NONE;
	winit = InitWay::IWP;
	volRatio = 0.3; // used for initialize rho but not constraint
	heatRatio[0] = 1; // heatRatio[0] is the ratio when rho = 1, heatRatio[1] for rho = 0
	heatRatio[1] = 0;
	outprefix = "";
	logrho = 0;
	logc = 0;
	logsens = 0;
	logobj = 0;
	testname = "none";
	useManagedMemory = true;
	inputrho = "init rand";
	max_iter = 500;
	initperiod = 10;
	finthres = 5e-4; // threshold of change ratio used for objective convergence check
	filterRadius = 2.0;
	designStep = 0.05; // design step for oc
	dampRatio = 0.5; // damp ratio for oc
	femRelThres = 0.01; // relative residual threshold for FEM
	target_tensor[0] = 0.3;
	target_tensor[1] = 0.2;
	target_tensor[2] = 0.1;
	target_tensor[3] = 0.;
	target_tensor[4] = 0.;
	target_tensor[5] = 0.;
}
