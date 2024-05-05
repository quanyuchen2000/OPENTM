#include "homogenization/Framework.cuh"

using namespace homo;
using namespace culib;


template<typename CH>
void logIter(int iter, cfg::HomoConfig config, TensorVar<>& rho, CH& Ch, double obj) {
	/// fixed log 
	if (iter % 5 == 0) {
		rho.value().toVdb(getPath("rho"));
		//rho.diff().toVdb(getPath("sens"));
		Ch.writeTo(getPath("C"));
	}
	Ch.domain_.logger() << "finished iteration " << iter << std::endl;

	/// optional log
	char namebuf[100];
	if (config.logrho != 0 && iter % config.logrho == 0) {
		sprintf_s(namebuf, "rho_%04d", iter);
		rho.value().toVdb(getPath(namebuf));
	}

	if (config.logc != 0 && iter % config.logc == 0) {
		sprintf_s(namebuf, "Clog");
		//Ch.writeTo(getPath(namebuf));
		auto ch = Ch.data();
		std::ofstream ofs;
		if (iter == 0) {
			ofs.open(getPath(namebuf));
		} else {
			ofs.open(getPath(namebuf), std::ios::app);
		}
		ofs << "iter " << iter << " ";
		for (int i = 0; i < 36; i++) { ofs << ch[i] << " "; }
		ofs << std::endl;
		ofs.close();
	}

	if (config.logsens != 0 && iter % config.logsens == 0) {
		sprintf_s(namebuf, "sens_%04d", iter);
		//rho.diff().graft(sens.data());
		rho.diff().toVdb(getPath(namebuf));
	}

	if (config.logobj != 0 && iter % config.logobj == 0) {
		sprintf_s(namebuf, "objlog");
		std::ofstream ofs;
		if (iter == 0) {
			ofs.open(getPath(namebuf));
		}
		else {
			ofs.open(getPath(namebuf), std::ios::app);
		}
		ofs << "iter " << iter << " ";
		ofs << "obj = " << obj << std::endl;
		ofs.close();
	}
}

void initDensity(var_tsexp_t<>& rho, cfg::HomoConfig config) {
	int resox = rho.value().length(0);
	int resoy = rho.value().length(1);
	int resoz = rho.value().length(2);
	constexpr float pi = 3.1415926;
	if (config.winit == cfg::InitWay::random || config.winit == cfg::InitWay::randcenter) {
		randTri(rho.value(), config);
	} else if (config.winit == cfg::InitWay::manual) {
		rho.value().fromVdb(config.inputrho, false);
	} else if (config.winit == cfg::InitWay::interp) {
		rho.value().fromVdb(config.inputrho, true);
	} else if (config.winit == cfg::InitWay::rep_randcenter) {
		randTri(rho.value(), config);
	} else if (config.winit == cfg::InitWay::noise) {
		rho.value().rand(0.f, 1.f);
		symmetrizeField(rho.value(), config.sym);
		rho.value().proj(20.f, 0.5f);
		auto view = rho.value().view();
		auto ker = [=] __device__(int id) { return  view(id); };
		float s = config.volRatio / (sequence_sum(ker, view.size(), 0.f) / view.size());
		rho.value().mapInplace([=] __device__(int x, int y, int z, float val) {
			float newval = val * s;
			if (newval < 0.001f) newval = 0.001;
			if (newval >= 1.f) newval = 1.f;
			return newval;
		});
	} else if (config.winit == cfg::InitWay::P) {
		rho.rvalue().setValue([=]__device__(int i, int j, int k) {
			float p[3] = { float(i) / resox, float(j) / resoy , float(k) / resoz };
			float val = cosf(2 * pi * p[0]) + cosf(2 * pi * p[1]) + cosf(2 * pi * p[2]);
			auto newval = tanproj(-val, 20);
			newval = max(min(newval, 1.f), 0.001f);
			return newval;
		});
	} else if (config.winit == cfg::InitWay::G) {
		rho.rvalue().setValue([=]__device__(int i, int j, int k) {
			float p[3] = { float(i) / resox, float(j) / resoy, float(k) / resoz };
			float s[3], c[3];
			for (int i = 0; i < 3; i++) {
				s[i] = sin(2 * pi * p[i]);
				c[i] = cos(2 * pi * p[i]);
			}
			float val = s[0] * c[1] + s[2] * c[0] + s[1] * c[2];
			auto newval = tanproj(val, 20);
			newval = max(min(newval, 1.f), 0.001f);
			return newval;
		});
	} else if (config.winit == cfg::InitWay::D) {
		rho.rvalue().setValue([=] __device__(int i, int j, int k) {
			float p[3] = { float(i) / resox, float(j) / resoy, float(k) / resoz };
			float x = p[0], y = p[1], z = p[2];
			float val = cos(2 * pi * x) * cos(2 * pi * y) * cos(2 * pi * z) - sin(2 * pi * x) * sin(2 * pi * y) * sin(2 * pi * z);
			float newval = tanproj(val, 20);
			newval = max(min(newval, 1.f), 0.001f);
			return newval;
		});
	} else if (config.winit == cfg::InitWay::IWP) {
		rho.rvalue().setValue([=] __device__(int i, int j, int k) {
			float p[3] = { float(i) / resox, float(j) / resoy, float(k) / resoz };
			float x = p[0], y = p[1], z = p[2];
			float val = 2 * (cos(2 * pi * x) * cos(2 * pi * y) + cos(2 * pi * y) * cos(2 * pi * z) + cos(2 * pi * z) * cos(2 * pi * x)) -
				(cos(2 * 2 * pi * x) + cos(2 * 2 * pi * y) + cos(2 * 2 * pi * z));
			float newval = tanproj(val, 20);
			newval = max(min(newval, 1.f), 0.001f);
			return newval;
		});
	}
	// the initway example is what we've done in matlab to compare and check
	else if (config.winit == cfg::InitWay::example) {
		rho.rvalue().setValue_H([=] __device__(int i, int j, int k) {
			if (sqrt(pow(i - float(resox) / 4 + 0.5, 2) + pow(j - float(resoy) / 4 + 0.5, 2) + pow(k - float(resoz) / 4 + 0.5, 2)) < float(min(min(resox, resoy), resoz)) / 6.0)
			{
				return 0.;
			}
			else
				return 0.3;
			// return 0.5;
		});
	}
	// symmetrize density field
	symmetrizeField(rho.value(), config.sym);

	// clamp density value to [rho_min, 1]
	rho.value().clamp(0.1, 1);
}

std::vector<float> runCustom(cfg::HomoConfig config, std::vector<float> *rho0 = nullptr) {
	int reso = config.reso[0];
	int ne = pow(reso, 3);
	auto tt = config.target_tensor;
	Homogenization_H hom_H(config);
	hom_H.ConfigDiagPrecondition(0);
	var_tsexp_t<> rho_H(reso, reso, reso);
	if (!rho0) {
		initDensity(rho_H, config);
	}
	else {
		rho_H.value().fromHost(rho0[0]);
	}
	auto rhop_H = rho_H.conv(radial_convker_t<float, Spline4>(1.5, 0)).pow(3) * (config.heatRatio[0] - config.heatRatio[1]) + config.heatRatio[1];
	heat_tensor_t <float, decltype(rhop_H)> Hh(hom_H, rhop_H);
	auto objective = (Hh(0, 0) - tt[0]).pow(2) + (Hh(1, 1) - tt[1]).pow(2) +
		(Hh(2, 2) - tt[2]).pow(2) + (Hh(0, 1) - tt[3]).pow(2) +
		(Hh(2, 1) - tt[4]).pow(2) + (Hh(0, 2) - tt[5]).pow(2) - 1e-4;
	ConvergeChecker criteria(config.finthres);
	if (config.model == cfg::Model::mma) {
		MMAOptimizer mma(1, ne, 1, 0, 1e8, 1);
		mma.setBound(0.0001, 1);
		for (int itn = 0; itn < config.max_iter; itn++) {
			//clock_t start = clock();
			float f0val = objective.eval();
			objective.backward(1);
			if (criteria.is_converge(itn, f0val)) { printf("converged\n"); break; }
			auto rhoArray = rho_H.value().flatten();
			auto dfdx = rho_H.diff().flatten();
			//dfdx.toMatlab("dfdx");
			gv::gVector<float> dvdx(ne);
			dvdx.set(1.0);
			gv::gVector<float> gval(1.0);
			float* dgdx = dvdx.data();
			float curVol = gv::gVectorMap(rhoArray.data(), ne).sum();
			gval[0] = f0val;
			printf("\033[32m \n* Iter %d  obj = %.4e  vol = %4.2f%%\033[0m\n", itn, f0val + 0.0001, curVol / ne * 100);
			float* dfdx_s = dfdx.data();
			//mma.update(itn, rhoArray.data(), dfdx.data(), gval.data(), &dgdx);
			mma.update(itn, rhoArray.data(), dgdx, gval.data(), &dfdx_s);
			rho_H.rvalue().graft(rhoArray.data());
		}
		hom_H.grid->writeDensity(getPath("density"), VoxelIOFormat::openVDB);
	}
	else if (config.model == cfg::Model::oc) {
		OCOptimizer oc(ne, 0.001, 0.02, 0.5);
		VolumeGovernor governor;
		for (int itn = 0; itn < config.max_iter; itn++) {
			float val = objective.eval();
			printf("\033[32m\n * Iter %d   obj = %.4e  vb = %.4e\033[0m\n", itn, val, governor.get_volume_bound());
			float lowerBound = rhop_H.sum().eval_imp() / pow(reso, 3);
			float volfrac = rho_H.sum().eval_imp() / pow(reso, 3);
			auto it = governor.volume_check(val, lowerBound, volfrac, itn);
			if (it) {
				printf("converged"); break;
			}
			objective.backward(1);
			if (criteria.is_converge(itn, val) && governor.get_current_decrease() < 1e-4) { printf("converged\n"); break; }
			auto sens = rho_H.diff().flatten();
			auto rhoarray = rho_H.value().flatten();
			int ereso[3] = { reso,reso,reso };
			oc.filterSens(sens.data(), rhoarray.data(), reso, ereso);
			oc.update(sens.data(), rhoarray.data(), governor.get_volume_bound());
			rho_H.value().graft(rhoarray.data());
		}
		hom_H.grid->writeDensity(getPath("density"), VoxelIOFormat::openVDB);
	}
	std::vector<float> rho(reso * reso * reso);
	rho_H.eval().toHost(rho);
	cuda_error_check;
	return rho;
}

