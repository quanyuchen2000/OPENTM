#include "homogenization.h"
#include "templateMatrix.h"
#include "matlab/matlab_utils.h"
#include <chrono>
#include <iomanip>

extern void uploadTemplaceMatrix(const double* ke, float penal);
extern void uploadTemplaceMatrix_H(const double* ke, float penal, const float* feMu, const float* disp);
void uploadTemplateLameMatrix(const char* kelam72, const char* kemu72, float Lam, float Mu);

std::string outpath = "C:/Users/zhangdi/Documents/temp/homo/";

namespace homo {
	std::string getPath(const std::string& str) {
		return outpath + str;
	}
	std::string setPathPrefix(const std::string& str) {
		outpath = str;
		return outpath;
	}

	size_t uid_t::uid;

}

homo::Homogenization_H::Homogenization_H(cfg::HomoConfig config)
{
	build(config);
}

void homo::Homogenization_H::build(cfg::HomoConfig homconfig)
{
	config = homconfig;

	printf("[Homo] building domain with resolution [%d, %d, %d], H = %.4le\n",
		config.reso[0], config.reso[1], config.reso[2], config.heatRatio);
	mg_.reset(new MG_H());
	MGConfig mgconf;
	mgconf.namePrefix = getName();
	std::copy(config.reso, config.reso + 3, mgconf.reso);
	mgconf.enableManagedMem = config.useManagedMemory;
	//mgconf.namePrefix;
	mg_->build(mgconf);
	grid = mg_->getRootGrid();

	//grid->test();
	initTemplateMatrix_H(1.0/config.reso[0]/2, getMem(), config.heatRatio);
	auto mtt = getFeMatrix();
	float cst[24];
	for (int i = 0; i < 8; i++) {
		for (int j = 0; j < 3; j++) {
			cst[j + i * 3] = mtt(i, j);
		}
	}
	auto mmp = getDispMatrix();
	float css[24];
	for (int i = 0; i < 8; i++) {
		for (int j = 0; j < 3; j++) {
			css[j + i * 3] = mmp(i, j);
		}
	}
	uploadTemplaceMatrix_H(getTemplateMatrixFp64_H().data(), power_penal, cst, css);
}

void homo::Homogenization_H::heatMatrix(float C[3][3])
{
	double c[3][3];
	heatMatrix(c);
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			C[i][j] = c[i][j];
		}
	}
}

void homo::Homogenization_H::update(float* rho, int pitchT)
{
	if (rho != nullptr) { grid->update(rho, pitchT); }
	grid->pad_cell_data(grid->rho_g);
	mg_->updateStencils();
}

void homo::Homogenization_H::ConfigDiagPrecondition(float strength)
{
	diag_strength = strength;
	grid->diagPrecondition(diag_strength);
}

std::string homo::Homogenization_H::getName(void)
{
	char buf[1000];
	sprintf_s(buf, "H%zu", getUid());
	return buf;
}
