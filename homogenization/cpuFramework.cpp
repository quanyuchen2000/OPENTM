#include "homogenization/cpuFramework.h"
#include "homogenization/grid.h"
#include <vector>
#include <omp.h> 
#include <thread>
#include <execution>
#include <numeric>
inline float tanproj(float val, float beta, float tau = 0.5f) {
	const float tbtau = tanhf(beta * tau);
	float newval = (tbtau + tanhf(beta * (val - tau))) / (tbtau + tanhf(beta * (1.f - tau)));
	return newval;
}
void initDensity_Host(std::vector<float>& rho, cfg::HomoConfig config) {
	const int resox = config.reso[0];
	const int resoy = config.reso[1];
	const int resoz = config.reso[2];
	constexpr float pi = 3.1415926f;

	if (config.winit != cfg::InitWay::IWP) {
		throw std::runtime_error("NO SUPPORT");
	}

	const int block_numx = resox / MIN_TRANSFER;
	const int block_numy = resoy / MIN_TRANSFER;
	const int block_numz = resoz / MIN_TRANSFER;
	const int block_num = block_numx * block_numy * block_numz;
	const int block_len = MIN_TRANSFER * MIN_TRANSFER * MIN_TRANSFER;

	const unsigned int num_threads = std::thread::hardware_concurrency();
	std::vector<std::thread> workers;

	auto thread_task = [&](int start_id, int end_id) {
		for (int block_id = start_id; block_id < end_id; ++block_id) {
			const int off_set = block_len * block_id;
			const int off_setx = block_id % block_numx;
			const int off_sety = (block_id / block_numx) % block_numy;
			const int off_setz = block_id / (block_numx * block_numy);

			for (int k = 0; k < MIN_TRANSFER; ++k) {
				for (int j = 0; j < MIN_TRANSFER; ++j) {
					for (int i = 0; i < MIN_TRANSFER; ++i) {
						const float x = float(i + off_setx * MIN_TRANSFER) / resox;
						const float y = float(j + off_sety * MIN_TRANSFER) / resoy;
						const float z = float(k + off_setz * MIN_TRANSFER) / resoz;

						const float val = 2 * (std::cos(2 * pi * x) * std::cos(2 * pi * y) +
							std::cos(2 * pi * y) * std::cos(2 * pi * z) +
							std::cos(2 * pi * z) * std::cos(2 * pi * x))
							- (std::cos(4 * pi * x) + std::cos(4 * pi * y) + std::cos(4 * pi * z));

						const int id = off_set + k * (MIN_TRANSFER * MIN_TRANSFER)
							+ j * MIN_TRANSFER + i;
						rho[id] = std::clamp(tanproj(val, 20), 0.001f, 1.0f);
					}
				}
			}
		}
		};

	const int blocks_per_thread = block_num / num_threads;
	int remaining_blocks = block_num % num_threads;
	int start_id = 0;

	for (unsigned int t = 0; t < num_threads; ++t) {
		int end_id = start_id + blocks_per_thread + (t < remaining_blocks ? 1 : 0);
		workers.emplace_back(thread_task, start_id, end_id);
		start_id = end_id;
	}

	for (auto& th : workers) {
		if (th.joinable()) th.join();
	}
}
//void initDensity_Host(std::vector<float>& rho, cfg::HomoConfig config) {
//	int resox = config.reso[0];
//	int resoy = config.reso[1];
//	int resoz = config.reso[2];
//	constexpr float pi = 3.1415926;
//
//	if (config.winit == cfg::InitWay::IWP) {
//		int off_set = 0;
//		// for each block init block
//		int block_numx = (resox / MIN_TRANSFER);
//		int block_numy = (resoy / MIN_TRANSFER);
//		int block_numz = (resoz / MIN_TRANSFER);
//		int block_num = block_numx * block_numy * block_numz;
//		int block_len = pow(MIN_TRANSFER, 3);
//
//		for (int block_id = 0; block_id < block_num; block_id++) {
//			off_set = block_len * block_id;
//			int off_setx = block_id % block_numx, off_sety = block_id / block_numx % block_numy, off_setz = block_id / (block_numx * block_numy);
//			for (int k = 0; k < MIN_TRANSFER; k++) {
//				for (int j = 0; j < MIN_TRANSFER; j++) {
//					for (int i = 0; i < MIN_TRANSFER; i++) {
//						float x = float(i + off_setx * MIN_TRANSFER) / resox, y = float(j + off_sety * MIN_TRANSFER) / resoy, z = float(k + off_setz * MIN_TRANSFER) / resoz;
//						float val = 2 * (cos(2 * pi * x) * cos(2 * pi * y) + cos(2 * pi * y) * cos(2 * pi * z) + cos(2 * pi * z) * cos(2 * pi * x)) -
//							(cos(2 * 2 * pi * x) + cos(2 * 2 * pi * y) + cos(2 * 2 * pi * z));
//						val = tanproj(val, 20);
//						val = std::max(std::min(val, 1.f), 0.001f);
//						int id = off_set + k * MIN_TRANSFER * MIN_TRANSFER + j * MIN_TRANSFER + i;
//						rho[id] = val;
//					}
//				}
//			}
//		}
//	}
//	else {
//		// Other initialization methods are not supported
//		throw std::runtime_error("NO SUPPORT");
//	}
//}
// auto rhop_H = rho_H.conv(radial_convker_t<float, Spline4>(1.5, 0)).pow(3) * (config.heatRatio[0] - config.heatRatio[1]) + config.heatRatio[1];
//void caculate_rhop(std::vector<float>& rho, std::vector<float>& rhop, cfg::HomoConfig config) {
//	int resox = config.reso[0];
//	int resoy = config.reso[1];
//	int resoz = config.reso[2];
//	int off_set = 0;
//	// for each block init block
//	int block_numx = (resox / MIN_TRANSFER);
//	int block_numy = (resoy / MIN_TRANSFER);
//	int block_numz = (resoz / MIN_TRANSFER);
//	int block_num = block_numx * block_numy * block_numz;
//	int block_len = pow(MIN_TRANSFER + 2, 3);
//	for (int block_id = 0; block_id < block_num; block_id++) {
//		off_set = block_len * block_id;
//		int off_set0 = block_id * pow(MIN_TRANSFER, 3);
//		int off_setx = block_id % block_numx, off_sety = block_id / block_numx % block_numy, off_setz = block_id / (block_numx * block_numy);
//		for (int k = 1; k < MIN_TRANSFER + 1; k++) {
//			for (int j = 1; j < MIN_TRANSFER + 1; j++) {
//				for (int i = 1; i < MIN_TRANSFER + 1; i++) {
//					int id = off_set + k * (MIN_TRANSFER + 2) * (MIN_TRANSFER + 2) + j * (MIN_TRANSFER + 2) + i;
//					int id0 = (i - 1) + (j - 1) * MIN_TRANSFER + (k - 1) * MIN_TRANSFER * MIN_TRANSFER + off_set0;			
//					rhop[id] = pow(rho[id0], 3) * (config.heatRatio[0] - config.heatRatio[1]) + config.heatRatio[1];
//				}
//			}
//		}
//	}
//}
void caculate_rhop(std::vector<float>& rho, std::vector<float>& rhop, cfg::HomoConfig config) {
	const int resox = config.reso[0];
	const int resoy = config.reso[1];
	const int resoz = config.reso[2];

	const int block_numx = resox / MIN_TRANSFER;
	const int block_numy = resoy / MIN_TRANSFER;
	const int block_numz = resoz / MIN_TRANSFER;
	const int block_num = block_numx * block_numy * block_numz;
	const int block_len = std::pow(MIN_TRANSFER + 2, 3);

	const unsigned int num_threads = std::thread::hardware_concurrency();
	std::vector<std::thread> workers;

	auto thread_task = [&](int start_id, int end_id) {
		for (int block_id = start_id; block_id < end_id; ++block_id) {
			const int off_set = block_len * block_id;
			const int off_set0 = block_id * (MIN_TRANSFER * MIN_TRANSFER * MIN_TRANSFER);
			const int off_setx = block_id % block_numx;
			const int off_sety = (block_id / block_numx) % block_numy;
			const int off_setz = block_id / (block_numx * block_numy);

			for (int k = 1; k < MIN_TRANSFER + 1; ++k) {
				for (int j = 1; j < MIN_TRANSFER + 1; ++j) {
					for (int i = 1; i < MIN_TRANSFER + 1; ++i) {
						const int id = off_set + k * (MIN_TRANSFER + 2) * (MIN_TRANSFER + 2)
							+ j * (MIN_TRANSFER + 2) + i;
						const int id0 = (i - 1) + (j - 1) * MIN_TRANSFER
							+ (k - 1) * MIN_TRANSFER * MIN_TRANSFER + off_set0;

						rhop[id] = rho[id0] * rho[id0] * rho[id0] * (config.heatRatio[0] - config.heatRatio[1])
							+ config.heatRatio[1];
					}
				}
			}
		}
		};

	const int blocks_per_thread = block_num / num_threads;
	int remaining_blocks = block_num % num_threads;
	int start_id = 0;

	for (unsigned int t = 0; t < num_threads; ++t) {
		const int end_id = start_id + blocks_per_thread + (t < remaining_blocks ? 1 : 0);
		workers.emplace_back(thread_task, start_id, end_id);
		start_id = end_id;
	}

	for (auto& th : workers) {
		if (th.joinable()) th.join();
	}
}
void caculate_sens(std::vector<float>& rhosens, std::vector<float>& rhopsens, std::vector<float> &rho, cfg::HomoConfig config) {
	rhosens.resize(rhopsens.size());
	for (int i = 0; i < rhosens.size(); i++) {
		rhosens[i] = rhopsens[i] * (3 * rho[i] * rho[i]) * (config.heatRatio[0] - config.heatRatio[1]);
	}
}

void update_density_boundary(std::vector<float>& rho, cfg::HomoConfig config) {
	const int resox = config.reso[0];
	const int resoy = config.reso[1];
	const int resoz = config.reso[2];

	const int block_numx = resox / MIN_TRANSFER;
	const int block_numy = resoy / MIN_TRANSFER;
	const int block_numz = resoz / MIN_TRANSFER;
	const int block_num = block_numx * block_numy * block_numz;
	const int block_len = (MIN_TRANSFER + 2) * (MIN_TRANSFER + 2) * (MIN_TRANSFER + 2);
	const unsigned int num_threads = std::thread::hardware_concurrency();
	std::vector<std::thread> workers;
	auto thread_task = [&](int start_block, int end_block) {
		for (int block_id = start_block; block_id < end_block; ++block_id) {
			const int off_set = block_len * block_id;
			const int off_setx = block_id % block_numx;
			const int off_sety = (block_id / block_numx) % block_numy;
			const int off_setz = block_id / (block_numx * block_numy);

			for (int i : {0, MIN_TRANSFER + 1}) {
				for (int j = 0; j < MIN_TRANSFER + 2; ++j) {
					for (int k = 0; k < MIN_TRANSFER + 2; ++k) {
						const int tox = (i == 0) ?
							(off_setx - 1 + block_numx) % block_numx :
							(off_setx + 1) % block_numx;
						const int ti = (i == 0) ? MIN_TRANSFER : 1;

						const int toy = (j == 0) ?
							(off_sety - 1 + block_numy) % block_numy :
							(j == MIN_TRANSFER + 1) ? (off_sety + 1) % block_numy : off_sety;
						const int tj = (j == 0) ? MIN_TRANSFER :
							(j == MIN_TRANSFER + 1) ? 1 : j;

						const int toz = (k == 0) ?
							(off_setz - 1 + block_numz) % block_numz :
							(k == MIN_TRANSFER + 1) ? (off_setz + 1) % block_numz : off_setz;
						const int tk = (k == 0) ? MIN_TRANSFER :
							(k == MIN_TRANSFER + 1) ? 1 : k;

						const int id = off_set + k * (MIN_TRANSFER + 2) * (MIN_TRANSFER + 2)
							+ j * (MIN_TRANSFER + 2) + i;
						const int tid = (tox + toy * block_numx + toz * block_numx * block_numy)
							* block_len + tk * (MIN_TRANSFER + 2) * (MIN_TRANSFER + 2)
							+ tj * (MIN_TRANSFER + 2) + ti;
						rho[id] = rho[tid];
					}
				}
			}

			for (int j : {0, MIN_TRANSFER + 1}) {
				for (int i = 1; i < MIN_TRANSFER + 1; ++i) {
					for (int k = 0; k < MIN_TRANSFER + 2; ++k) {
						const int toy = (j == 0) ?
							(off_sety - 1 + block_numy) % block_numy :
							(off_sety + 1) % block_numy;
						const int tj = (j == 0) ? MIN_TRANSFER : 1;

						const int tox = off_setx;
						const int ti = i;

						const int toz = (k == 0) ?
							(off_setz - 1 + block_numz) % block_numz :
							(k == MIN_TRANSFER + 1) ? (off_setz + 1) % block_numz : off_setz;
						const int tk = (k == 0) ? MIN_TRANSFER :
							(k == MIN_TRANSFER + 1) ? 1 : k;

						const int id = off_set + k * (MIN_TRANSFER + 2) * (MIN_TRANSFER + 2)
							+ j * (MIN_TRANSFER + 2) + i;
						const int tid = (tox + toy * block_numx + toz * block_numx * block_numy)
							* block_len + tk * (MIN_TRANSFER + 2) * (MIN_TRANSFER + 2)
							+ tj * (MIN_TRANSFER + 2) + ti;
						rho[id] = rho[tid];
					}
				}
			}

			for (int k : {0, MIN_TRANSFER + 1}) {
				for (int i = 1; i < MIN_TRANSFER + 1; ++i) {
					for (int j = 1; j < MIN_TRANSFER + 1; ++j) {
						const int toz = (k == 0) ?
							(off_setz - 1 + block_numz) % block_numz :
							(off_setz + 1) % block_numz;
						const int tk = (k == 0) ? MIN_TRANSFER : 1;
						const int tox = off_setx;
						const int ti = i;
						const int toy = off_sety;
						const int tj = j;

						const int id = off_set + k * (MIN_TRANSFER + 2) * (MIN_TRANSFER + 2)
							+ j * (MIN_TRANSFER + 2) + i;
						const int tid = (tox + toy * block_numx + toz * block_numx * block_numy)
							* block_len + tk * (MIN_TRANSFER + 2) * (MIN_TRANSFER + 2)
							+ tj * (MIN_TRANSFER + 2) + ti;
						rho[id] = rho[tid];
					}
				}
			}
		}
		};
	const int blocks_per_thread = block_num / num_threads;
	int remaining_blocks = block_num % num_threads;
	int start_block = 0;

	for (unsigned int t = 0; t < num_threads; ++t) {
		const int end_block = start_block + blocks_per_thread + (t < remaining_blocks ? 1 : 0);
		workers.emplace_back(thread_task, start_block, end_block);
		start_block = end_block;
	}

	for (auto& th : workers) {
		if (th.joinable()) th.join();
	}
}

void subtract_mean_parallel(std::vector<float>& A) {
	if (A.empty()) return;

	const double sum = std::transform_reduce(
		std::execution::par_unseq,
		A.begin(), A.end(),
		0.0,
		std::plus<>(),
		[](float x) { return static_cast<double>(x); }
	);

	const float mean = static_cast<float>(sum / A.size());

	std::for_each(
		std::execution::par_unseq,
		A.begin(), A.end(),
		[mean](float& x) { x -= mean; }
	);
}
double norm_host(std::vector<float>& A) {
	double sum = std::transform_reduce(
		std::execution::par_unseq,
		A.begin(), A.end(), 
		0.0,  
		std::plus<double>(),
		[](float x) { return static_cast<double>(x) * x; }
	);
	return std::sqrt(sum);
}
void block2lexi(std::vector<float>& rho, std::vector<float>& lexirho, cfg::HomoConfig config) {
	int resox = config.reso[0];
	int resoy = config.reso[1];
	int resoz = config.reso[2];
	int off_set = 0;
	// for each block init block
	int block_numx = (resox / MIN_TRANSFER);
	int block_numy = (resoy / MIN_TRANSFER);
	int block_numz = (resoz / MIN_TRANSFER);
	int block_num = block_numx * block_numy * block_numz;
	int block_len = pow(MIN_TRANSFER, 3);

	for (int block_id = 0; block_id < block_num; block_id++) {
		off_set = block_len * block_id;
		int off_setx = block_id % block_numx, off_sety = block_id / block_numx % block_numy, off_setz = block_id / (block_numx * block_numy);
		for (int k = 0; k < MIN_TRANSFER; k++) {
			for (int j = 0; j < MIN_TRANSFER; j++) {
				for (int i = 0; i < MIN_TRANSFER; i++) {
					int x = i + off_setx * MIN_TRANSFER, y = j + off_sety * MIN_TRANSFER, z = k + off_setz * MIN_TRANSFER;
					int id = off_set + k * MIN_TRANSFER * MIN_TRANSFER + j * MIN_TRANSFER + i;
					int idlexi = x + (y + z * resoy) * resox;
					lexirho[idlexi] = rho[id];
				}
			}
		}
	}
}
void lexi2block(std::vector<float>& lexirho, std::vector<float>& rho, cfg::HomoConfig config) {
	int resox = config.reso[0];
	int resoy = config.reso[1];
	int resoz = config.reso[2];
	int off_set = 0;

	int block_numx = resox / MIN_TRANSFER;
	int block_numy = resoy / MIN_TRANSFER;
	int block_numz = resoz / MIN_TRANSFER;
	int block_num = block_numx * block_numy * block_numz;
	int block_len = pow(MIN_TRANSFER, 3);

	for (int block_id = 0; block_id < block_num; block_id++) {
		off_set = block_len * block_id;

		int off_setx = block_id % block_numx;
		int off_sety = (block_id / block_numx) % block_numy;
		int off_setz = block_id / (block_numx * block_numy);

		for (int k = 0; k < MIN_TRANSFER; k++) {
			for (int j = 0; j < MIN_TRANSFER; j++) {
				for (int i = 0; i < MIN_TRANSFER; i++) {

					int x = i + off_setx * MIN_TRANSFER;
					int y = j + off_sety * MIN_TRANSFER;
					int z = k + off_setz * MIN_TRANSFER;

					int idlexi = x + (y + z * resoy) * resox;

					int id = off_set + k * MIN_TRANSFER * MIN_TRANSFER
						+ j * MIN_TRANSFER + i;

					rho[id] = lexirho[idlexi];
				}
			}
		}
	}
}
float find_max_abs(const std::vector<float>& sens) {
	if (sens.empty()) {
		throw std::invalid_argument("Vector is empty");
	}
	auto it = std::max_element(
		sens.begin(), sens.end(),
		[](float a, float b) {
			return std::abs(a) < std::abs(b);
		}
	);
	return *it;
}