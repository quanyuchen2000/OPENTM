#pragma once
#include "MG.h"
#include "cmdline.h"
#include <fstream>

namespace homo {
	struct uid_t {
	private:
		static size_t uid;
	protected:
		static void setUid(void) { uid++; };
		uid_t(void) { setUid(); }
		size_t getUid(void) { return uid; }
	};
	struct Homogenization_H : public uid_t {
		Homogenization_H(cfg::HomoConfig config);
		void build(cfg::HomoConfig config);
		void update(float* rho = nullptr, int pitchT = -1);
		void heatMatrix(double C[3][3]);
		void heatMatrix(float C[3][3]);
		std::shared_ptr<Grid_H> grid;
		std::unique_ptr<MG_H> mg_;
		float power_penal = 1;
		float diag_strength = 1e6;
		cfg::HomoConfig config;
		SymmetryType sym;

		void ConfigDiagPrecondition(float strength);
		void Sensitivity(float dC[3][3], float* sens, int pitchT, bool lexiOrder = false);
	private:
		std::string getName(void);
	};
}
