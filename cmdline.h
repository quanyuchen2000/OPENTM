#pragma once
#include "stdint.h"
#include <iostream>

#define b2s(boolValue) (boolValue?"Yes":"No")

namespace cfg {
	enum class Objective : uint8_t { bulk, shear, npr, custom };
	enum class Symmetry : uint8_t { reflect3, reflect6, rotate3, rotate2, NONE };
	enum class InitWay : uint8_t { random, randcenter, noise, manual, interp, rep_randcenter, P, G, D, IWP, example };
	enum class Model : uint8_t { mma, oc };
	struct HomoConfig {
		Symmetry sym;
		InitWay winit;
		Model model;
		int reso[3];
		double volRatio;
		double heatRatio[2];
		double finthres;
		double filterRadius;
		double designStep;
		double dampRatio;
		double femRelThres;
		double target_tensor[6];
		std::string outprefix;
		std::string testname;
		std::string inputrho;
		bool useManagedMemory = true;
		int logrho, logc, logsens, logobj;
		int max_iter = 100;
		int initperiod;
		void init();
	};
}


