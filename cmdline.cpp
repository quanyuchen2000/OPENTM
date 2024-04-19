#include "cmdline.h"
#include <fstream>

void cfg::HomoConfig::init()
{
	reso[0] = 64;
	reso[1] = 64;
	reso[2] = 64;
	obj = Objective::bulk;
	sym = Symmetry::NONE;
	winit = InitWay::IWP;
	volRatio = 0.3;
	heatRatio[0] = 1;
	heatRatio[1] = 1e-4;
	outprefix = "";
	logrho = 0;
	logc = 0;
	logsens = 0;
	logobj = 0;
	testname = "none";
	useManagedMemory = true;
	inputrho = "";
	max_iter = 100;
	initperiod = 10;
	finthres = 5e-4; // threshold of change ratio used for objective convergence check
	filterRadius = 2.0;
	designStep = 0.05; // design step for oc
	dampRatio = 0.5; // damp ratio for oc
	femRelThres = 0.01; // relative residual threshold for FEM
	target_tensor[0] = 0.5;
	target_tensor[1] = 0.3;
	target_tensor[2] = 0.4;
	target_tensor[3] = 0.1;
	target_tensor[4] = 0.1;
	target_tensor[5] = 0.1;

	//// print parsed configuration
	//printf("Configuration : \n");
	//printf(" = reso  - - - - - - - - - - - - - - - - - - - - - - - %d\n", reso[0]);
	//printf(" = obj   - - - - - - - - - - - - - - - - - - - - - - - %s\n", FLAGS_obj.c_str());
	//printf(" = init  - - - - - - - - - - - - - - - - - - - - - - - %s\n", FLAGS_init.c_str());
	//printf(" = sym   - - - - - - - - - - - - - - - - - - - - - - - %s\n", FLAGS_sym.c_str());
	//printf(" = vol   - - - - - - - - - - - - - - - - - - - - - - - %4.2f\n", float(FLAGS_vol));
	//printf(" = output    - - - - - - - - - - - - - - - - - - - - - %s\n", FLAGS_prefix.c_str());
	//printf(" = logrho    - - - - - - - - - - - - - - - - - - - - - %s\n", b2s(FLAGS_logrho));
	//printf(" = logc    - - - - - - - - - - - - - - - - - - - - - - %s\n", b2s(FLAGS_logc));
	//printf(" = logsens   - - - - - - - - - - - - - - - - - - - - - %s\n", b2s(FLAGS_logsens));
	//printf(" = test    - - - - - - - - - - - - - - - - - - - - - - %s\n", FLAGS_test.c_str());
	//printf(" = useManagedMem   - - - - - - - - - - - - - - - - - - %s\n", b2s(FLAGS_managedmem));
	//printf(" = maxIter - - - - - - - - - - - - - - - - - - - - - - %d\n", FLAGS_N);
	//printf(" = initPeriod  - - - - - - - - - - - - - - - - - - - - %d\n", FLAGS_initperiod);
	//printf(" = finthres    - - - - - - - - - - - - - - - - - - - - %e\n", float(FLAGS_finthres));
	//printf(" = filterRadius  - - - - - - - - - - - - - - - - - - - %4.2f\n", float(FLAGS_filter));
	//printf(" = dampRatio     - - - - - - - - - - - - - - - - - - - %4.2f\n", float(FLAGS_damp));
	//printf(" = designStep    - - - - - - - - - - - - - - - - - - - %4.2f\n", float(FLAGS_step));
	//printf(" = femRelThres   - - - - - - - - - - - - - - - - - - - %4.2e\n", float(FLAGS_relthres));
	//printf(" = input(optional) - - - - - - - - - - - - - - - - - - %s\n", FLAGS_in.c_str());
}
