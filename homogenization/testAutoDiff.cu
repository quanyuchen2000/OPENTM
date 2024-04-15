//#define _TUPLE_
//#define __CUDACC__
#include "Framework.cuh"

using namespace homo;

template<typename Scalar, typename RhoPhys>
void logIter(int iter, cfg::HomoConfig config, var_tsexp_t<>& rho, Tensor<Scalar> sens, elastic_tensor_t<Scalar, RhoPhys>& Ch, double obj) {
	// fixed log 
	if (iter % 5 == 0) {
		rho.value().toVdb(getPath("rho"));
		rho.diff().graft(sens.data());
		//rho.diff().toVdb(getPath("sens"));
		Ch.writeTo(getPath("C"));
	}
	Ch.domain_.logger() << "finished iteration " << iter << std::endl;

	// optional log
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
		}
		else {
			ofs.open(getPath(namebuf), std::ios::app);
		}
		ofs << "iter " << iter << " ";
		for (int i = 0; i < 36; i++) { ofs << ch[i] << " "; }
		ofs << std::endl;
		ofs.close();
	}

	if (config.logsens != 0 && iter % config.logsens == 0) {
		sprintf_s(namebuf, "sens_%04d", iter);
		rho.diff().graft(sens.data());
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

void test_MMA(cfg::HomoConfig config, int mode) {
	int reso = config.reso[0];
	config.sym = cfg::Symmetry::NONE;
	int ne = pow(reso, 3);
	float goalVolRatio = 0.3;
	//ConvergeChecker criteria(config.finthres);
	if (mode == 1) {
		// auto rhop_H = rho_H.pow(3);
		Homogenization_H hom_H(config);
		hom_H.ConfigDiagPrecondition(0);
		var_tsexp_t<> rho_H(reso, reso, reso);
		initDensity(rho_H, config);
		auto rhop_H = rho_H.conv(radial_convker_t<float, Spline4>(1.5, 0)).pow(3);
		heat_tensor_t <float, decltype(rhop_H)> Hh(hom_H, rhop_H);
		auto objective1 = (Hh(0, 0) - 0.5).pow(2) + (Hh(1, 1) - 0.3).pow(2) + (Hh(2, 2) - 0.4).pow(2) - 0.0001;
		for (int itn = 0; itn < 200; itn++) {
			//clock_t start = clock();
			float f0val = objective1.eval();
			objective1.backward(1);
			auto rhoArray = rho_H.value().flatten();
			auto dfdx = rho_H.diff().flatten();
			//dfdx.toMatlab("dfdx");
			gv::gVector<float> dvdx(ne);
			dvdx.set(1);
			gv::gVector<float> gval(1);
			float* dgdx = dvdx.data();
			float curVol = gv::gVectorMap(rhoArray.data(), ne).sum();
			// gval[0] = curVol - ne * goalVolRatio;
			gval[0] = f0val;
			printf("\033[32m \n* Iter %d  obj = %.4e  vol = %4.2f%%\033[0m\n", itn, f0val + 0.0001, curVol / ne * 100);
			//rhoArray.toMatlab("dfdx");
			float* dfdx_s = dfdx.data();
			MMAOptimizer mma(1, ne, 1, 0, 1e8, 1);
			mma.setBound(0.001, 1);
			//mma.update(itn, rhoArray.data(), dfdx.data(), gval.data(), &dgdx);
			mma.update(itn, rhoArray.data(), dgdx, gval.data(), &dfdx_s);
			rho_H.rvalue().graft(rhoArray.data());
			//clock_t end = clock();
			//std::cout << "cost" << (double)(end - start) / CLOCKS_PER_SEC << "s" << std::endl;
		}
		rhop_H.value().toMatlab("rhop");
		hom_H.grid->writeDensity(getPath("density"), VoxelIOFormat::openVDB);
	}
	else if (mode == 2) {
		auto tg = config.target_tensor;
		var_tsexp_t<> rho_H(reso, reso, reso);
		initDensity(rho_H, config);
		auto rhop_H = rho_H.conv(radial_convker_t<float, Spline4>(1.5, 0)).pow(3);
		// auto rhop_H = rho_H.pow(3);
		Homogenization_H hom_H(config);
		hom_H.ConfigDiagPrecondition(0);
		heat_tensor_t <float, decltype(rhop_H)> Hh(hom_H, rhop_H);
		auto objective1 = (Hh(0, 0) - tg[0]).pow(2) + (Hh(1, 1) - tg[1]).pow(2) + (Hh(2, 2) - tg[2]).pow(2) + 
			(Hh(1, 0) - tg[3]).pow(2) + (Hh(1, 2) - tg[4]).pow(2) + (Hh(2, 0) - tg[5]).pow(2) - 0.0001;
		OCOptimizer oc(ne, 0.001, 0.02, 0.5);
		float volume_bound = 1.0;
		float decrease = 0;
		float decrease_factor = 1.;
		float val_last = 1;
		int count = 0;
		for (int itn = 0; itn < 500; itn++) {
			float val = objective1.eval();
			printf("\033[32m\n * Iter %d   obj = %.4e  vb = %.4e\033[0m\n", itn, val, volume_bound);
			if (val + 0.0001 < 1e-4 || (itn + 1)%50 == 0) {
				float num_bound = rhop_H.sum().eval_imp()/pow(reso, 3);
				decrease = volume_bound - num_bound;
				volume_bound = volume_bound - decrease * decrease_factor;
				decrease_factor *= 0.8;
			}
			if (val_last - val < 1e-4 && val + 0.0001 > 1e-4) {
				count++;
			}
			else {
				count = 0;
			}
			if (count >= 5) {
				volume_bound += 0.3 * decrease * decrease_factor;
			}
			objective1.backward(1);
			auto sens = rho_H.diff().flatten();
			auto rhoarray = rho_H.value().flatten();
			int ereso[3] = { reso,reso,reso };
			oc.filterSens(sens.data(), rhoarray.data(), reso, ereso);
			oc.update(sens.data(), rhoarray.data(), volume_bound);
			rho_H.value().graft(rhoarray.data());
			val_last = val;
		}
		rhop_H.value().toMatlab("rhop");
		hom_H.grid->writeDensity(getPath("density"), VoxelIOFormat::openVDB);
	}
}

#include "homoCommon.cuh"

void cudaTest(void) {
#if 0
	Tensor<float> x = Tensor<float>::range(0, 1, 10000);;
	Tensor<float> fx(x.getDim());
	fx.copy(x);
	fx.mapInplace([=] __device__(int x, int y, int z, float val) {
		auto newval = tanproj(val, 20, 0.6);
		return newval;
	});
	fx.toMatlab("fx");
#elif 0
	Tensor<float> x(100, 100, 100);
	int n_period = 10;
	int nbasis1st = n_period * 3;
	int nbasis2nd = n_period * n_period * 3;
	int nbasis = nbasis1st + nbasis2nd;
	float* coeffs;
	cudaMalloc(&coeffs, sizeof(float)* nbasis);
	randArray(&coeffs, 1, nbasis, -1.f, 1.f);
	auto view = x.view();
	size_t block_size = 256;
	size_t grid_size = ceilDiv(view.size(), 32);
	randTribase_cos_kernel << <grid_size, block_size >> > (view, n_period, coeffs);
	cudaDeviceSynchronize();
	cuda_error_check;
	x.toVdb(getPath("x"));
	x.toMatlab("x");
	cudaFree(coeffs);
#elif 0
	Tensor<float> x(32, 32, 32);
	x.fromVdb("C:/Users/zhangdi/Documents/temp/homo/rho");
	x.symmetrize(Rotate3);
	x.toVdb("C:/Users/zhangdi/Documents/temp/homo/rho1");
#else
	Tensor<float> x(128, 128, 128);
	int n_period = 10;
	int nbasis1st = n_period * 6;
	int nbasis2nd = n_period * n_period * 36;
	int nbasis = nbasis1st + nbasis2nd;
	float* coeffs;
	cudaMalloc(&coeffs, sizeof(float)* nbasis);
	randArray(&coeffs, 1, nbasis, -1.f, 1.f);
	size_t block_size = 256;
	size_t grid_size = ceilDiv(x.view().size(), 32);
	randTribase_sincos_kernel << <grid_size, block_size >> > (x.view(), n_period, coeffs);
	cudaDeviceSynchronize();
	cudaFree(coeffs);
	cuda_error_check;
	x.toVdb("rho0");
	x.symmetrize(Rotate3);
	x.toVdb("rho1");
#endif
}

extern void runCustom(cfg::HomoConfig config);
