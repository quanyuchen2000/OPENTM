#include "homogenization/Framework.cuh"
#include "homogenization/cpuFramework.h"
#include "voxelIO/openvdb_wrapper_t.h"
#include <cmath>
#include <algorithm>
#include <omp.h>
#include <thread>
#include <execution>
#include <numeric>
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
			if (sqrt(pow(i - float(resox) / 2 + 0.5, 2) + pow(j - float(resoy) /2 + 0.5, 2) + pow(k - float(resoz) / 2 + 0.5, 2)) < float(min(min(resox, resoy), resoz)) / 6.0)
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
	// rho.value().clamp(0.001, 1);
}


std::vector<float> runCustom(cfg::HomoConfig config, std::vector<float> *rho0 = nullptr) {
	std::ofstream ofs;
	int reso = config.reso[0];
	std::string filename;
	filename = config.testname;
	filename.erase(std::remove(filename.begin(), filename.end(), '\t'), filename.end());
	ofs.open("time512.txt", std::ios::app);
	int ne = pow(reso, 3);
	auto tt = config.target_tensor;
	Homogenization_H hom_H(config);
	hom_H.ConfigDiagPrecondition(0);
	int total_ne = (reso / MIN_TRANSFER * reso / MIN_TRANSFER * reso / MIN_TRANSFER) * pow(MIN_TRANSFER + 2, 3);
	if (reso >= MIN_TRANSFER) {
		std::vector<float> rho(pow(reso, 3));

		if (!rho0) {
			initDensity_Host(rho, config);
		}
		else {
			std::copy(rho0->begin(), rho0->end(), rho.begin());
		}

		std::vector<float> rhop(total_ne);
		cudaHostRegister(rhop.data(), rhop.size() * sizeof(float), cudaHostRegisterPortable);

		heat_tensor_host_t <float> Hh(hom_H, &rhop);
		auto objective = (Hh(0, 0) - tt[0]).pow(2) + (Hh(1, 1) - tt[1]).pow(2) +
			(Hh(2, 2) - tt[2]).pow(2) + (Hh(0, 1) - tt[3]).pow(2) +
			(Hh(2, 1) - tt[4]).pow(2) + (Hh(0, 2) - tt[5]).pow(2) - 1e-2;

		{
			ConvergeChecker criteria(config.finthres);
			OCOptimizer oc(reso * reso * reso, 0.001, 0.02, 0.5);

			VolumeGovernor governor;
			float final_val;
			int itn;
			clock_t start = clock();
			for (itn = 0; itn < 200; itn++) {

				caculate_rhop(rho, rhop, config);

				update_density_boundary(rhop, config);

				float val = objective.eval();

				final_val = val;
				printf("\033[32m\n * Iter %d   obj = %.4e  vb = %.4e\033[0m\n", itn, val, governor.get_volume_bound());
				printf("%f %f %f %f %f %f", Hh.H_[0][0], Hh.H_[1][1], Hh.H_[2][2], Hh.H_[0][1], Hh.H_[1][2], Hh.H_[0][2]);
				float sum = 0.0;
				sum = std::transform_reduce(
					std::execution::par_unseq,
					rho.begin(), rho.end(),
					0.0,
					std::plus<double>(),
					[](float x) { return x * x * x; }
				);
				float lowerBound = sum / pow(reso, 3);
				sum = 0;
				sum = std::transform_reduce(
					std::execution::par_unseq,
					rho.begin(), rho.end(),
					0.0,
					std::plus<double>(),
					[](float x) { return x; }
				);
				float volfrac = sum / pow(reso, 3);
				auto it = governor.volume_check(val, lowerBound, volfrac, itn, rho, Hh.H_);
				if (it) {
					printf("converged"); break;
				}
				objective.backward(1);
				if (criteria.is_converge(itn, val) && governor.get_current_decrease() < 1e-2) { printf("converged\n"); break; }

				std::vector<float>* sensp = &Hh.sensitiveField;
				std::vector<float> sens;
				caculate_sens(sens, *sensp, rho, config);

				//// block optimization
				//// pay attention to 8
				//int blockne = MIN_TRANSFER * MIN_TRANSFER * MIN_TRANSFER;
				//int ereso[3] = { MIN_TRANSFER,MIN_TRANSFER,MIN_TRANSFER };
				//for (int i = 0; i < 8; i++) {
				//	float* grho = getMem().getBuffer("temp_rho0")->data<float>();
				//	float* gsens = getMem().getBuffer("temp_rho1")->data<float>();
				//	cudaMemcpy(grho, rho.data()+ i * blockne, blockne * sizeof(float), cudaMemcpyHostToDevice);
				//	cudaMemcpy(gsens, sens.data()+ i * blockne, blockne * sizeof(float), cudaMemcpyHostToDevice);
				//	oc.filterSens(gsens, grho, MIN_TRANSFER, ereso);
				//	cudaMemcpy(sens.data() + i * blockne, gsens, blockne * sizeof(float), cudaMemcpyDeviceToHost);
				//}
				//std::vector<float> newrho = rho;
				//float volratio = governor.get_volume_bound();
				//float maxSens = abs(find_max_abs(sens));
				//printf("max sens = %f\n", maxSens);
				//float minSens = 0;
				//for (int itn = 0; itn < 20; itn++) {
				//	float gSens = (maxSens + minSens) / 2;
				//	std::for_each(std::execution::par_unseq, newrho.begin(), newrho.end(),
				//		[&](auto& nr) {
				//			int i = &nr - &newrho[0];
				//			float r = rho[i];
				//			float B = -sens[i] / gSens;
				//			if (B < 0) B = 0.01f;
				//			float newr = sqrt(B) * r;
				//			if (newr - r < -0.02) newr = r - 0.02;
				//			if (newr - r > 0.02) newr = r + 0.02;
				//			if (newr < 0.001) newr = 0.001;
				//			if (newr > 1) newr = 1;
				//			newrho[i] = newr;
				//		});

				//	float curVol = std::transform_reduce(
				//		std::execution::par_unseq,
				//		newrho.begin(), newrho.end(),
				//		0.0,
				//		std::plus<double>(),
				//		[](float x) { return x; }
				//	) / ne;
				//	printf("[OC] : g = %.4e   vol = %4.2f%% (Goal %4.2f%%)       \r", gSens, curVol * 100, volratio * 100);
				//	if (curVol < volratio - 0.0001) {
				//		maxSens = gSens;
				//	}
				//	else if (curVol > volratio + 0.0001) {
				//		minSens = gSens;
				//	}
				//	else {
				//		break;
				//	}
				//}
				//printf("\n");
				//rho = newrho;

				// GPU OC updation faster but memory comsume more
				int ereso[3] = { reso,reso,reso };
				std::vector<float> vectortemp(rho.size());
				block2lexi(rho, vectortemp, config);
				rho = vectortemp;
				block2lexi(sens, vectortemp, config);
				sens = vectortemp;
				auto tmpname = getMem().addBuffer(pow(reso, 3) * sizeof(float));
				float* gsens = getMem().getBuffer(tmpname)->data<float>();
				tmpname = getMem().addBuffer(pow(reso, 3) * sizeof(float));
				float* grho = getMem().getBuffer(tmpname)->data<float>();
				cudaMemcpy(grho, rho.data(), pow(reso, 3) * sizeof(float), cudaMemcpyHostToDevice);
				cudaMemcpy(gsens, sens.data(), pow(reso, 3) * sizeof(float), cudaMemcpyHostToDevice);
				oc.filterSens(gsens, grho, reso, ereso);
				oc.update(gsens, grho, governor.get_volume_bound());
				cudaMemcpy(rho.data(), grho, pow(reso, 3) * sizeof(float), cudaMemcpyDeviceToHost);
				lexi2block(rho, vectortemp, config);
				rho = vectortemp;
				getMem().deleteBuffer(gsens);
				getMem().deleteBuffer(grho);

				clock_t end = clock();
				double elapsed_time = static_cast<double>(end - start) / CLOCKS_PER_SEC;
				std::cout << elapsed_time << std::endl;
			}
			// ofs << itn << "\n";
			if (governor.best_res != 100000 && governor.best_res < final_val) {
				ofs << governor.best_res << "\n";
				ofs << governor.val_last << "\n";
				ofs << governor.hh[0][0] << " " << governor.hh[1][1] << " " << governor.hh[2][2] << " " << governor.hh[0][1] << " " << governor.hh[1][2] << " " << governor.hh[0][2] << "\n";
				ofs << governor.best_vol << "\n";
				rho = governor.best_rho_h;
			}
			else {
				ofs << final_val << "\n";
				ofs << governor.val_last << "\n";
				ofs << Hh.H_[0][0] << " " << Hh.H_[1][1] << " " << Hh.H_[2][2] << " " << Hh.H_[0][1] << " " << Hh.H_[1][2] << " " << Hh.H_[0][2] << "\n";
			}
		}

		int gsize[3] = { reso, reso, reso };
		std::vector<float> vectorsave(rho.size());
		block2lexi(rho, vectorsave, config);
		openvdb_wrapper_t<float>::lexicalGrid2openVDBfile("128 block.vdb", gsize, vectorsave);
		ofs.close();
	}
	else {
		var_tsexp_t<> rho_H(reso, reso, reso);
		if (!rho0) {
			initDensity(rho_H, config);
		}
		else {
			rho_H.value().fromHost(rho0[0]);
		}
		// auto rhop_H = rho_H.conv(radial_convker_t<float, Spline4>(1.5, 0)).pow(3) * (config.heatRatio[0] - config.heatRatio[1]) + config.heatRatio[1];
		auto rhop_H = rho_H.pow(3) * (config.heatRatio[0] - config.heatRatio[1]) + config.heatRatio[1];
		heat_tensor_t <float, decltype(rhop_H)> Hh(hom_H, rhop_H);
		//auto objective = ((Hh(0, 0) - tt[0]).abs() + (Hh(1, 1) - tt[1]).abs() +
		//	(Hh(2, 2) - tt[2]).abs() + (Hh(0, 1) - tt[3]).abs() +
		//	(Hh(2, 1) - tt[4]).abs() + (Hh(0, 2) - tt[5]).abs()) - 1e-1;
		auto objective = (Hh(0, 0) - tt[0]).pow(2) + (Hh(1, 1) - tt[1]).pow(2) +
			(Hh(2, 2) - tt[2]).pow(2) + (Hh(0, 1) - tt[3]).pow(2) +
			(Hh(2, 1) - tt[4]).pow(2) + (Hh(0, 2) - tt[5]).pow(2) - 1e-2;
		//auto objective = (Hh(0, 0)/tt[0] - 1.).pow(2) + (Hh(1, 1)/tt[1] - 1.).pow(2) +
		//	(Hh(2, 2)/tt[2] - 1.).pow(2) + (Hh(0, 1)/tt[3] - 1.).pow(2) +
		//	(Hh(2, 1)/tt[4] - 1.).pow(2) + (Hh(0, 2)/tt[5] - 1.).pow(2) - 1e-2;

		ConvergeChecker criteria(config.finthres);
		if (config.model == cfg::Model::mma) {
			MMAOptimizer mma(1, ne, 1, 0, 1e6, 1);
			mma.setBound(0.0001, 1);
			clock_t start = clock();
			for (int itn = 0; itn < 200; itn++) {
				//clock_t start = clock();
				float f0val = objective.eval();
				objective.backward(1);
				if (criteria.is_converge(itn, f0val)) { printf("converged\n"); break; }
				auto rhoArray = rho_H.value().flatten();
				auto dfdx = rho_H.diff().flatten();
				//dfdx.toMatlab("dfdx");
				gv::gVector<float> dvdx(ne);
				dvdx.set(1.0 / (reso * reso * reso));
				gv::gVector<float> gval(1.0);
				float* dgdx = dvdx.data();
				float curVol = gv::gVectorMap(rhoArray.data(), ne).sum();
				gval[0] = f0val;
				printf("\033[32m \n* Iter %d  obj = %.4e  vol = %4.2f%%\033[0m\n", itn, f0val + 0.01, curVol / ne * 100);
				float* dfdx_s = dfdx.data();
				//mma.update(itn, rhoArray.data(), dfdx.data(), gval.data(), &dgdx);
				mma.update(itn, rhoArray.data(), dgdx, gval.data(), &dfdx_s);
				rho_H.rvalue().graft(rhoArray.data());
				clock_t end = clock();
				double elapsed_time = static_cast<double>(end - start) / CLOCKS_PER_SEC;
				ofs << elapsed_time << " " << f0val << "\n";
			}
			ofs.close();
			// rho_H.value().toVdb(filename);
		}
		else if (config.model == cfg::Model::oc) {
			OCOptimizer oc(ne, 0.001, 0.02, 0.5);
			VolumeGovernor governor;
			clock_t start = clock();
			float final_val;
			int itn;
			for (itn = 0; itn < 1; itn++) {
				float val = objective.eval();
				final_val = val;
				printf("\033[32m\n * Iter %d   obj = %.4e  vb = %.4e\033[0m\n", itn, val, governor.get_volume_bound());
				printf("%f %f %f %f %f %f", Hh.H_[0][0], Hh.H_[1][1], Hh.H_[2][2], Hh.H_[0][1], Hh.H_[1][2], Hh.H_[0][2]);
				float lowerBound = rho_H.pow(3).sum().eval_imp() / pow(reso, 3);
				float volfrac = rho_H.sum().eval_imp() / pow(reso, 3);
				auto it = governor.volume_check(val, lowerBound, volfrac, itn, rho_H, Hh.H_);
				if (it) {
					printf("converged"); break;
				}
				objective.backward(1);
				if (criteria.is_converge(itn, val) && governor.get_current_decrease() < 1e-2) { printf("converged\n"); break; }
				auto sens = rho_H.diff().flatten();
				auto rhoarray = rho_H.value().flatten();
				int ereso[3] = { reso,reso,reso };
				oc.filterSens(sens.data(), rhoarray.data(), reso, ereso);
				oc.update(sens.data(), rhoarray.data(), governor.get_volume_bound());
				rho_H.value().graft(rhoarray.data());
				clock_t end = clock();
				double elapsed_time = static_cast<double>(end - start) / CLOCKS_PER_SEC;
				ofs << elapsed_time << " " << val + 0.01 << "\n";
			}
			// ofs << itn << "\n";
			if (governor.best_res != 100000 && governor.best_res < final_val) {
				ofs << governor.best_res << "\n";
				ofs << governor.val_last << "\n";
				ofs << governor.hh[0][0] << " " << governor.hh[1][1] << " " << governor.hh[2][2] << " " << governor.hh[0][1] << " " << governor.hh[1][2] << " " << governor.hh[0][2] << "\n";
				ofs << governor.best_vol << "\n";
				governor.best_rho.toVdb("nofilter256");
			}
			else {
				ofs << final_val << "\n";
				ofs << governor.val_last << "\n";
				ofs << Hh.H_[0][0] << " " << Hh.H_[1][1] << " " << Hh.H_[2][2] << " " << Hh.H_[0][1] << " " << Hh.H_[1][2] << " " << Hh.H_[0][2] << "\n";
				ofs << rho_H.sum().eval_imp() / pow(reso, 3) << "\n";
				rho_H.value().toVdb("nofilter256");
			}
			ofs.close();
		}
		else if (config.model == cfg::Model::moo) {
			MMAOptimizer mma(1, ne, 1, 0, 1e4, 1);
			mma.setBound(0.001, 1);
			VolumeGovernor governor;
			clock_t start = clock();
			float final_val;
			for (int itn = 0; itn < 500; itn++) {
				float val = objective.eval();
				final_val = val;
				float lowerBound = rho_H.pow(3).sum().eval_imp() / pow(reso, 3);
				float volfrac = rho_H.sum().eval_imp() / pow(reso, 3);
				printf("\033[32m\n * Iter %d   obj = %.4e vf = %.4e vb = %.4e\033[0m\n", itn, val, volfrac, governor.get_volume_bound());
				auto it = governor.volume_check(val, lowerBound, volfrac, itn, rho_H, Hh.H_);
				objective.backward(1);
				if (criteria.is_converge(itn, val) && governor.get_current_decrease() < 1e-2) { printf("converged\n"); break; }
				symmetrizeField(rho_H.value(), config.sym);
				symmetrizeField(rho_H.diff(), config.sym);
				auto rhoArray = rho_H.value().flatten();
				auto dfdx = rho_H.diff().flatten();
				gv::gVector<float> dvdx(ne);
				dvdx.set(1.0 / (reso * reso * reso));
				gv::gVector<float> gval(1.0);
				gval[0] = volfrac - governor.get_volume_bound();
				std::cout << volfrac - governor.get_volume_bound();
				float* dfdx_s[1] = { dvdx.data() };
				mma.update(itn, rhoArray.data(), dfdx.data(), gval.data(), dfdx_s);
				rho_H.rvalue().graft(rhoArray.data());
				clock_t end = clock();
				double elapsed_time = static_cast<double>(end - start) / CLOCKS_PER_SEC;
				ofs << elapsed_time << " " << val << "\n";
			}
			// hom_H.grid->writeDensity(getPath("mmaonocdensity"), VoxelIOFormat::openVDB);
			ofs.close();
		}
	}
	freeMem();
	return {};
}

