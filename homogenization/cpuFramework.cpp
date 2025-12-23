#include "homogenization/cpuFramework.h"
#include "homogenization/grid.h"
#include <vector>
#include <omp.h> 
#include <thread>
#include <execution>
#include <numeric>
#include <mutex>
#include "voxelIO/openvdb_wrapper_t.h"
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

	if (config.winit != cfg::InitWay::IWP && config.winit != cfg::InitWay::manual) {
		throw std::runtime_error("NO SUPPORT");
	}
	else if (config.winit == cfg::InitWay::manual) {
		int subreso = 256;
		std::string fname = "64_1208040402020.vdb";
		printf("reading density %s...", fname.c_str());
		std::vector<int> pos[3];
		std::vector<float> value;
		// openEXR not compatible :<
		//openvdb_wrapper_t<float>::openVDBfile2grid(fname, pos, value);
		openvdb_wrapper_t<float>::openVDBfile2grid(fname, pos, value);
		//auto pv = std::tie(pos, value);
		int origin[3];
		int reso[3];
		for (int j = 0; j < 3; j++) {
			origin[j] = *std::min_element(pos[j].begin(), pos[j].end());
			reso[j] = 1 + *std::max_element(pos[j].begin(), pos[j].end()) - origin[j];
		}
		printf(" reso = (%d, %d, %d)\n", reso[0], reso[1], reso[2]);
		int ne = reso[0] * reso[1] * reso[2];
		std::vector<float> newvalues(ne, 0);
		for (int i = 0; i < value.size(); i++) {
			int p[3] = { pos[0][i] - origin[0], pos[1][i] - origin[1], pos[2][i] - origin[2] };
			int lexid = p[0] + p[1] * reso[0] + p[2] * reso[0] * reso[1];
			newvalues[lexid] = value[i];
		}
		const int block_numx = resox / MIN_TRANSFER;
		const int block_numy = resoy / MIN_TRANSFER;
		const int block_numz = resoz / MIN_TRANSFER;
		const int block_num = block_numx * block_numy * block_numz;
		const int block_len = MIN_TRANSFER * MIN_TRANSFER * MIN_TRANSFER;
		int ratio = config.reso[0] / reso[0];
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
							const int xc = (i + off_setx * MIN_TRANSFER)/ratio;
							const int yc = (j + off_sety * MIN_TRANSFER)/ratio;
							const int zc = (k + off_setz * MIN_TRANSFER)/ratio;
							float filtered_val = 0.0f;
							float total_weight = 0.0f;
							for (int dz : { -1, 0, 1}) {
								for (int dy : {-1, 0, 1}) {
									for (int dx : {-1, 0, 1}) {
										const int x = (xc + dx + reso[0]) % reso[0];
										const int y = (yc + dy + reso[1]) % reso[1];
										const int z = (zc + dz + reso[2]) % reso[2];
										const float distance = sqrtf(dx * dx + dy * dy + dz * dz);
										const float weight = (distance <= 1.5) ? (1.5 - distance) : 0.0f;
										if (weight > 0) {
											const float val = newvalues[x + y * reso[0] + z * reso[0] * reso[1]];
											filtered_val += val * weight;
											total_weight += weight;
										}
									}
								}
							}
							const int id = off_set + k * (MIN_TRANSFER * MIN_TRANSFER) + j * MIN_TRANSFER + i;
							rho[id] = (total_weight != 0) ? (filtered_val / total_weight) : 0.001f;
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

		//std::string fname = "512_1208040404040.vdb";
		//printf("reading density %s...", fname.c_str());
		//std::vector<int> pos[3];
		//std::vector<float> value;
		//openvdb_wrapper_t<float>::openVDBfile2grid(fname, pos, value);
		//int origin[3];
		//int reso[3];
		//for (int j = 0; j < 3; j++) {
		//	origin[j] = *std::min_element(pos[j].begin(), pos[j].end());
		//	reso[j] = 1 + *std::max_element(pos[j].begin(), pos[j].end()) - origin[j];
		//}
		//printf(" reso = (%d, %d, %d)\n", reso[0], reso[1], reso[2]);
		//int ne = reso[0] * reso[1] * reso[2];
		//std::vector<float> newvalues(ne, 0);
		//for (int i = 0; i < value.size(); i++) {
		//	int p[3] = { pos[0][i] - origin[0], pos[1][i] - origin[1], pos[2][i] - origin[2] };
		//	int lexid = p[0] + p[1] * reso[0] + p[2] * reso[0] * reso[1];
		//	newvalues[lexid] = value[i] < 0.8 ? 0.0001:1;
		//}
		//lexi2block(newvalues, rho, config);
	}
	else {
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
void build_filter_block(std::vector<float>& rho, std::vector<float>& padded, int blockid, int filter_radius, int blocksize) {
	int fr = filter_radius;
	int oldsz = blocksize - 2 * fr;
	int offsetnum = oldsz * oldsz * oldsz;
	int block_numx = 2;
	int block_numy = 2;
	int block_numz = 2;
	int offset[3] = { blockid % block_numx, blockid / block_numx % block_numy, blockid / (block_numx * block_numy) };
	for (int i = fr; i < blocksize - fr; i++) {
		for (int j = fr; j < blocksize - fr; j++) {
			for (int k = fr; k < blocksize - fr; k++) {
				padded[(k * blocksize + j) * blocksize + i] = rho[offsetnum * blockid + ((k - fr) * oldsz + (j - fr)) * oldsz + (i - fr)];
			}
		}
	}
	for (int k = 0; k < blocksize; k++) {
		for (int j = 0; j < blocksize; j++) {
#pragma unroll
			for (int i = 0; i < filter_radius; i++) {
				int pos[3] = { i - fr, j - fr, k - fr };
				int id_tar = i + blocksize * (j + blocksize * k);
				auto offsetx = offset;
				offsetx[0] = (offsetx[0] - 1 + block_numx) % block_numx;
				pos[0] = MIN_TRANSFER - fr + i;

				if (j < filter_radius) {
					offsetx[1] = (offsetx[1] - 1 + block_numy) % block_numy;
					pos[1] = MIN_TRANSFER - fr + j;
				}
				else if (blocksize - fr <= j) {
					offsetx[1] = (offsetx[1] + 1) % block_numy;
					pos[1] = j - blocksize + fr;
				}

				if (k < filter_radius) {
					offsetx[2] = (offsetx[2] - 1 + block_numz) % block_numz;
					pos[2] = MIN_TRANSFER - fr + k;
				}
				else if (blocksize - fr <= k) {
					offsetx[2] = (offsetx[2] + 1) % block_numz;
					pos[2] = k - blocksize + fr;
				}

				int src_bid = offsetx[0] + offsetx[1] * block_numx + offsetx[2] * block_numx * block_numy;
				int gsid_src = src_bid * offsetnum + pos[0] + oldsz * (pos[1] + oldsz * pos[2]);
				padded[id_tar] = rho[gsid_src];
			}
		}
	}
	for (int k = 0; k < blocksize; k++) {
		for (int j = 0; j < blocksize; j++) {
#pragma unroll
			for (int i = blocksize - fr; i < blocksize; i++) {
				int pos[3] = { i - fr, j - fr, k - fr };
				int id_tar = i + blocksize * (j + blocksize * k);
				auto offsetx = offset;
				offsetx[0] = (offsetx[0] + 1) % block_numx;
				pos[0] = i - blocksize + fr;

				if (j < filter_radius) {
					offsetx[1] = (offsetx[1] - 1 + block_numy) % block_numy;
					pos[1] = MIN_TRANSFER - fr + j;
				}
				else if (blocksize - fr <= j) {
					offsetx[1] = (offsetx[1] + 1) % block_numy;
					pos[1] = j - blocksize + fr;
				}


				if (k < filter_radius) {
					offsetx[2] = (offsetx[2] - 1 + block_numz) % block_numz;
					pos[2] = MIN_TRANSFER - fr + k;
				}
				else if (blocksize - fr <= k) {
					offsetx[2] = (offsetx[2] + 1) % block_numz;
					pos[2] = k - blocksize + fr;
				}

				int src_bid = offsetx[0] + offsetx[1] * block_numx + offsetx[2] * block_numx * block_numy;
				int gsid_src = src_bid * offsetnum + pos[0] + oldsz * (pos[1] + oldsz * pos[2]);
				padded[id_tar] = rho[gsid_src];
			}
		}
	}
	for (int k = 0; k < blocksize; k++) {
#pragma unroll
		for (int j = 0; j < fr; j++) {
			for (int i = fr; i < blocksize - fr; i++) {
				int pos[3] = { i - fr, j - fr, k - fr };
				int id_tar = i + blocksize * (j + blocksize * k);
				auto offsetx = offset;

				offsetx[1] = (offsetx[1] - 1 + block_numy) % block_numy;
				pos[1] = MIN_TRANSFER - fr + j;

				if (k < filter_radius) {
					offsetx[2] = (offsetx[2] - 1 + block_numz) % block_numz;
					pos[2] = MIN_TRANSFER - fr + k;
				}
				else if (blocksize - fr <= k) {
					offsetx[2] = (offsetx[2] + 1) % block_numz;
					pos[2] = k - blocksize + fr;
				}

				int src_bid = offsetx[0] + offsetx[1] * block_numx + offsetx[2] * block_numx * block_numy;
				int gsid_src = src_bid * offsetnum + pos[0] + oldsz * (pos[1] + oldsz * pos[2]);
				padded[id_tar] = rho[gsid_src];
			}
		}
	}
	for (int k = 0; k < blocksize; k++) {
#pragma unroll
		for (int j = blocksize - fr; j < blocksize; j++) {
			for (int i = fr; i < blocksize - fr; i++) {
				int pos[3] = { i - fr, j - fr, k - fr };
				int id_tar = i + blocksize * (j + blocksize * k);
				auto offsetx = offset;
				offsetx[1] = (offsetx[1] + 1) % block_numy;
				pos[1] = j - blocksize + fr;
				if (k < filter_radius) {
					offsetx[2] = (offsetx[2] - 1 + block_numz) % block_numz;
					pos[2] = MIN_TRANSFER - fr + k;
				}
				else if (blocksize - fr <= k) {
					offsetx[2] = (offsetx[2] + 1) % block_numz;
					pos[2] = k - blocksize + fr;
				}
				int src_bid = offsetx[0] + offsetx[1] * block_numx + offsetx[2] * block_numx * block_numy;
				int gsid_src = src_bid * offsetnum + pos[0] + oldsz * (pos[1] + oldsz * pos[2]);
				padded[id_tar] = rho[gsid_src];
			}
		}
	}
#pragma unroll
	for (int k = 0; k < fr; k++) {
		for (int j = fr; j < blocksize - fr; j++) {
			for (int i = fr; i < blocksize - fr; i++) {
				int pos[3] = { i - fr, j - fr, k - fr };
				int id_tar = i + blocksize * (j + blocksize * k);
				auto offsetx = offset;
				offsetx[2] = (offsetx[2] - 1 + block_numz) % block_numz;
				pos[2] = MIN_TRANSFER - fr + k;
				int src_bid = offsetx[0] + offsetx[1] * block_numx + offsetx[2] * block_numx * block_numy;
				int gsid_src = src_bid * offsetnum + pos[0] + oldsz * (pos[1] + oldsz * pos[2]);
				padded[id_tar] = rho[gsid_src];
			}
		}
	}
#pragma unroll
	for (int k = blocksize - fr; k < blocksize; k++) {
		for (int j = fr; j < blocksize - fr; j++) {
			for (int i = fr; i < blocksize - fr; i++) {
				int pos[3] = { i - fr, j - fr, k - fr };
				int id_tar = i + blocksize * (j + blocksize * k);
				auto offsetx = offset;
				offsetx[2] = (offsetx[2] + 1) % block_numz;
				pos[2] = k - blocksize + fr;
				int src_bid = offsetx[0] + offsetx[1] * block_numx + offsetx[2] * block_numx * block_numy;
				int gsid_src = src_bid * offsetnum + pos[0] + oldsz * (pos[1] + oldsz * pos[2]);
				padded[id_tar] = rho[gsid_src];
			}
		}
	}
}

void out_filter_block(std::vector<float>& rho, std::vector<float>& padded, int blockid, int filter_radius, int blocksize) {
	int fr = filter_radius;
	int oldsz = blocksize - 2 * fr;
	int offsetnum = oldsz * oldsz * oldsz;
	int block_numx = 2;
	int block_numy = 2;
	int block_numz = 2;
#pragma omp parallel for collapse(3) schedule(static)
	for (int i = fr; i < blocksize - fr; i++) {
		for (int j = fr; j < blocksize - fr; j++) {
			for (int k = fr; k < blocksize - fr; k++) {
				rho[offsetnum * blockid + ((k - fr) * oldsz + (j - fr)) * oldsz + (i - fr)] = padded[(k * blocksize + j) * blocksize + i];
			}
		}
	}
}

//void filterneighbor(int pos[3], int blockpos[3], int radius, float wsum, std::vector<float>& rho, std::vector<float>& sens) {
//	float sum = 0;
//	for (int nei = 0; nei < wfunc.size(); nei++) {
//		int offset[3];
//		ker.neigh(nei, offset);
//		float w = ker.weight(offset);
//		int neighpos[3] = { epos[0] + offset[0], epos[1] + offset[1], epos[2] + offset[2] };
//		if (ker.is_period()) {
//			for (int i = 0; i < 3; i++) neighpos[i] = (neighpos[i] + reso[i]) % reso[i];
//		}
//		if (is_bounded(neighpos, ereso)) {
//			int neighid = neighpos[0] + (neighpos[1] + neighpos[2] * ereso[1]) * pitchT;
//			//w /= weightSum[neighid];
//			sum += sens[neighid] * rho[neighid] * w;
//			wsum += w;
//		}
//	}
//	int eid = epos[0] + (epos[1] + epos[2] * ereso[1]) * pitchT;
//	sum /= wsum * rho[eid];
//	newsens[eid] = sum;
//
//	float sum = 0;
//	for (int k = -radius; k <= radius; k++) {
//		for (int j = -radius; j <= radius; j++) {
//			for (int i = -radius; i <= radius; i++) {
//				float w = 1 - sqrt(float(i * i + j * j + k * k) / (radius * radius));
//				int neighpos[3] = {pos[0] + i, pos[1] + j, pos[2] + k};
//				int blockpos[3] = {};
//
//				sum += sens[neighid] * rho[neighid] * w;
//			}
//		}
//	}
//	sum /= wsum * rho[eid];
//
//}
float caculatewsum(int radius) {
	float wsum = 0;
	for (int k = -radius; k <= radius; k++) {
		for (int j = -radius; j <= radius; j++) {
			for (int i = -radius; i <= radius; i++) {
				float roff = sqrt(float(i * i + j * j + k * k) / (radius * radius));
				float w = roff < 1 ? 1 - roff : 0;
				wsum += w;
			}
		}
	}
	return wsum;
}
void calboundary(std::vector<float>& rho, std::vector<float>& sens, std::vector<float>& boundary, int blockid, int filter_radius, std::vector<std::thread> &workers, std::atomic<int> &counter) {
	int fr = filter_radius;
	int block[3] = { blockid % 2, blockid / 2 % 2, blockid / 4 };
	int tsknum = MIN_TRANSFER * MIN_TRANSFER * MIN_TRANSFER - pow(MIN_TRANSFER - 2 * fr, 3);
	int blocksize = MIN_TRANSFER * MIN_TRANSFER * MIN_TRANSFER;
	float wsum = caculatewsum(fr);
	int offset0 = MIN_TRANSFER * MIN_TRANSFER * fr;
	int offset1 = MIN_TRANSFER * (MIN_TRANSFER - 2 * fr) * fr;
	int offset2 = (MIN_TRANSFER - 2 * fr) * (MIN_TRANSFER - 2 * fr) * fr;
	const unsigned num_thread = std::max(4u, std::thread::hardware_concurrency());


	for (unsigned t = 0; t < num_thread; ++t) {
		workers.emplace_back([&]() {
			while (true) {
				const int i = counter.fetch_add(1, std::memory_order_relaxed);
				if (i >= tsknum) break;
				std::vector<int> pos(3);
				if (i < offset0) {
					pos = { i % MIN_TRANSFER, i / MIN_TRANSFER % MIN_TRANSFER, i / (MIN_TRANSFER * MIN_TRANSFER) };
				}
				else if (i < 2 * offset0) {
					int id = i - offset0;
					pos = { id % MIN_TRANSFER, id / MIN_TRANSFER % MIN_TRANSFER, MIN_TRANSFER - 1 - (id / (MIN_TRANSFER * MIN_TRANSFER)) };
				}
				else if (i < 2 * offset0 + offset1) {
					int id = i - 2 * offset0;
					pos = { id % MIN_TRANSFER, id / MIN_TRANSFER % fr, id / (MIN_TRANSFER * fr) + fr };
				}
				else if (i < 2 * offset0 + 2 * offset1) {
					int id = i - 2 * offset0 - offset1;
					pos = { id % MIN_TRANSFER, MIN_TRANSFER - 1 - id / MIN_TRANSFER % fr, id / (MIN_TRANSFER * fr) + fr };
				}
				else if (i < 2 * offset0 + 2 * offset1 + offset2) {
					int id = i - 2 * (offset0 + offset1);
					pos = { id % fr, id / fr % (MIN_TRANSFER - 2 * fr) + fr, id / (fr * (MIN_TRANSFER - 2 * fr)) + fr };
				}
				else {
					int id = i - 2 * (offset0 + offset1) - offset2;
					pos = { MIN_TRANSFER - 1 - id % fr, id / fr % (MIN_TRANSFER - 2 * fr) + fr, id / (fr * (MIN_TRANSFER - 2 * fr)) + fr };
				}
				float sum = 0;
				float wsum0 = 0;
				for (int k = -fr; k <= fr; k++) {
					for (int j = -fr; j <= fr; j++) {
						for (int ix = -fr; ix <= fr; ix++) {
							float roff = float(ix * ix + j * j + k * k) / (fr * fr);
							float w = roff < 1 ? 1 - sqrt(roff) : 0;
							w /= wsum;
							int neighpos[3] = { pos[0] + ix + MIN_TRANSFER, pos[1] + j + MIN_TRANSFER, pos[2] + k + MIN_TRANSFER };
							int neighpost[3] = { neighpos[0] % MIN_TRANSFER, (neighpos[1] % MIN_TRANSFER), (neighpos[2] % MIN_TRANSFER) };
							int neighid = neighpos[0] % MIN_TRANSFER + (neighpos[1] % MIN_TRANSFER) * MIN_TRANSFER + (neighpos[2] % MIN_TRANSFER) * MIN_TRANSFER * MIN_TRANSFER;
							int blockpos[3] = { block[0] - 1 + neighpos[0] / MIN_TRANSFER, block[1] - 1 + neighpos[1] / MIN_TRANSFER ,block[2] - 1 + neighpos[2] / MIN_TRANSFER };
							int blockpost[3] = { (blockpos[0] + 2) % 2 , (blockpos[1] + 2) % 2 , (blockpos[2] + 2) % 2 };
							int bid = (blockpos[0] + 2) % 2 + (blockpos[1] + 2) % 2 * 2 + (blockpos[2] + 2) % 2 * 4;
							int nid = bid * blocksize + neighid;
							sum += sens[nid] * rho[nid] * w;
							wsum0 += w;
						}
					}
				}
				sum /= wsum0 * rho[(block[0] + block[1] * 2 + block[2] * 4) * blocksize + pos[0] + pos[1] * MIN_TRANSFER + pos[2] * MIN_TRANSFER * MIN_TRANSFER];
				boundary[i] = sum;
			}
			});
	}
	for (auto& t : workers) {
		if (t.joinable()) t.join();
	}
	//for (int i = 0; i < tsknum; i++) {
	//	std::vector<int> pos(3);
	//	if (i < offset0) {
	//		pos = { i % MIN_TRANSFER, i / MIN_TRANSFER % MIN_TRANSFER, i / (MIN_TRANSFER * MIN_TRANSFER) };
	//	}
	//	else if (i < 2 * offset0) {
	//		int id = i - offset0;
	//		pos = { id % MIN_TRANSFER, id / MIN_TRANSFER % MIN_TRANSFER, MIN_TRANSFER - 1 - (id / (MIN_TRANSFER * MIN_TRANSFER)) };
	//	}
	//	else if (i < 2 * offset0 + offset1) {
	//		int id = i - 2 * offset0;
	//		pos = { id % MIN_TRANSFER, id / MIN_TRANSFER % fr, id / (MIN_TRANSFER * fr) + fr};
	//	}
	//	else if (i < 2 * offset0 + 2 * offset1) {
	//		int id = i - 2 * offset0 - offset1;
	//		pos = { id % MIN_TRANSFER, MIN_TRANSFER - 1 - id / MIN_TRANSFER % fr, id / (MIN_TRANSFER * fr) + fr };
	//	}
	//	else if (i < 2 * offset0 + 2 * offset1 + offset2) {
	//		int id = i - 2 * (offset0 + offset1);
	//		pos = { id % fr, id / fr % (MIN_TRANSFER - 2 * fr) + fr, id / (fr * (MIN_TRANSFER - 2 * fr)) + fr};
	//	}
	//	else {
	//		int id = i - 2 * (offset0 + offset1) - offset2;
	//		pos = { MIN_TRANSFER - 1 - id % fr, id / fr % (MIN_TRANSFER - 2 * fr) + fr, id / (fr * (MIN_TRANSFER - 2 * fr)) + fr };
	//	}
	//	// debug 
	//	float sum = 0;
	//	float wsum0 = 0;
	//	for (int k = -fr; k <= fr; k++) {
	//		for (int j = -fr; j <= fr; j++) {
	//			for (int ix = -fr; ix <= fr; ix++) {
	//				float roff = float(ix * ix + j * j + k * k) / (fr * fr);
	//				float w = roff < 1 ? 1 - sqrt(roff): 0;
	//				w /= wsum;
	//				int neighpos[3] = { pos[0] + ix + MIN_TRANSFER, pos[1] + j + MIN_TRANSFER, pos[2] + k +MIN_TRANSFER};
	//				int neighpost[3] = { neighpos[0] % MIN_TRANSFER, (neighpos[1] % MIN_TRANSFER), (neighpos[2] % MIN_TRANSFER) };
	//				int neighid = neighpos[0] % MIN_TRANSFER + (neighpos[1] % MIN_TRANSFER) * MIN_TRANSFER + (neighpos[2] % MIN_TRANSFER) * MIN_TRANSFER * MIN_TRANSFER;
	//				int blockpos[3] = {block[0] - 1 + neighpos[0] / MIN_TRANSFER, block[1] - 1 + neighpos[1] / MIN_TRANSFER ,block[2] - 1 + neighpos[2] / MIN_TRANSFER };
	//				int blockpost[3] = { (blockpos[0] + 2) % 2 , (blockpos[1] + 2) % 2 , (blockpos[2] + 2) % 2};
	//				int bid = (blockpos[0] + 2) % 2 + (blockpos[1] + 2) % 2 * 2 + (blockpos[2] + 2) % 2 * 4;
	//				int nid = bid * blocksize + neighid;
	//				sum += sens[nid] * rho[nid] * w;
	//				wsum0 += w;
	//			}
	//		}
	//	}
	//	sum /= wsum0 * rho[blockid * blocksize + pos[0] + pos[1] * MIN_TRANSFER + pos[2] * MIN_TRANSFER * MIN_TRANSFER];
	//	boundary[i] = sum;
	//	// std::cout << sum;
	//}
}

void reboundary(std::vector<float>& sens, std::vector<float>& boundary, int blockid, int fr) {
	int tsknum = MIN_TRANSFER * MIN_TRANSFER * MIN_TRANSFER - pow(MIN_TRANSFER - 2 * fr, 3);
	int offset0 = MIN_TRANSFER * MIN_TRANSFER * fr;
	int offset1 = MIN_TRANSFER * (MIN_TRANSFER - 2 * fr) * fr;
	int offset2 = (MIN_TRANSFER - 2 * fr) * (MIN_TRANSFER - 2 * fr) * fr;
	int blocksize = MIN_TRANSFER * MIN_TRANSFER * MIN_TRANSFER;
	const unsigned num_thread = std::max(4u, std::thread::hardware_concurrency());
	std::vector<std::thread> workers;
	std::atomic<int> counter(0);

	for (unsigned t = 0; t < num_thread; ++t) {
		workers.emplace_back([&]() {
			while (true) {
				const int i = counter.fetch_add(1, std::memory_order_relaxed);
				if (i >= tsknum) break;
				std::vector<int> pos(3);
				if (i < offset0) {
					pos = { i % MIN_TRANSFER, i / MIN_TRANSFER % MIN_TRANSFER, i / (MIN_TRANSFER * MIN_TRANSFER) };
				}
				else if (i < 2 * offset0) {
					int id = i - offset0;
					pos = { id % MIN_TRANSFER, id / MIN_TRANSFER % MIN_TRANSFER, MIN_TRANSFER - 1 - (id / (MIN_TRANSFER * MIN_TRANSFER)) };
				}
				else if (i < 2 * offset0 + offset1) {
					int id = i - 2 * offset0;
					pos = { id % MIN_TRANSFER, id / MIN_TRANSFER % fr, id / (MIN_TRANSFER * fr) + fr };
				}
				else if (i < 2 * offset0 + 2 * offset1) {
					int id = i - 2 * offset0 - offset1;
					pos = { id % MIN_TRANSFER, MIN_TRANSFER - 1 - id / MIN_TRANSFER % fr, id / (MIN_TRANSFER * fr) + fr };
				}
				else if (i < 2 * offset0 + 2 * offset1 + offset2) {
					int id = i - 2 * (offset0 + offset1);
					pos = { id % fr, id / fr % (MIN_TRANSFER - 2 * fr) + fr, id / (fr * (MIN_TRANSFER - 2 * fr)) + fr };
				}
				else {
					int id = i - 2 * (offset0 + offset1) - offset2;
					pos = { MIN_TRANSFER - 1 - id % fr, id / fr % (MIN_TRANSFER - 2 * fr) + fr, id / (fr * (MIN_TRANSFER - 2 * fr)) + fr };
				}
				sens[blockid * blocksize + pos[0] + pos[1] * MIN_TRANSFER + pos[2] * MIN_TRANSFER * MIN_TRANSFER] = boundary[i];
			}
		});
	}
	for (auto& t : workers) {
		if (t.joinable()) t.join();
	}
	//for (int i = 0; i < tsknum; i++) {
	//	std::vector<int> pos(3);
	//	if (i < offset0) {
	//		pos = { i % MIN_TRANSFER, i / MIN_TRANSFER % MIN_TRANSFER, i / (MIN_TRANSFER * MIN_TRANSFER) };
	//	}
	//	else if (i < 2 * offset0) {
	//		int id = i - offset0;
	//		pos = { id % MIN_TRANSFER, id / MIN_TRANSFER % MIN_TRANSFER, MIN_TRANSFER - 1 - (id / (MIN_TRANSFER * MIN_TRANSFER)) };
	//	}
	//	else if (i < 2 * offset0 + offset1) {
	//		int id = i - 2 * offset0;
	//		pos = { id % MIN_TRANSFER, id / MIN_TRANSFER % fr, id / (MIN_TRANSFER * fr) + fr };
	//	}
	//	else if (i < 2 * offset0 + 2 * offset1) {
	//		int id = i - 2 * offset0 - offset1;
	//		pos = { id % MIN_TRANSFER, MIN_TRANSFER - 1 - id / MIN_TRANSFER % fr, id / (MIN_TRANSFER * fr) + fr };
	//	}
	//	else if (i < 2 * offset0 + 2 * offset1 + offset2) {
	//		int id = i - 2 * (offset0 + offset1);
	//		pos = { id % fr, id / fr % (MIN_TRANSFER - 2 * fr) + fr, id / (fr * (MIN_TRANSFER - 2 * fr)) + fr };
	//	}
	//	else {
	//		int id = i - 2 * (offset0 + offset1) - offset2;
	//		pos = { MIN_TRANSFER - 1 - id % fr, id / fr % (MIN_TRANSFER - 2 * fr) + fr, id / (fr * (MIN_TRANSFER - 2 * fr)) + fr };
	//	}
	//	sens[blockid * blocksize + pos[0] + pos[1] * MIN_TRANSFER + pos[2] * MIN_TRANSFER * MIN_TRANSFER] = boundary[i];
	//}
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
void block2lexi(std::vector<float>& rho, std::vector<float>& lexirho, int reso) {
	int resox = reso;
	int resoy = reso;
	int resoz = reso;
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
	//const unsigned num_thread = std::max(4u, std::thread::hardware_concurrency());
	//std::vector<std::thread> workers;
	//std::atomic<int> counter(0);
	//const int tsknum = block_num;
	//for (unsigned t = 0; t < num_thread; ++t) {
	//	workers.emplace_back([&]() {
	//		while (true) {
	//			const int block_id = counter.fetch_add(1, std::memory_order_relaxed);
	//			if (block_id >= tsknum) break;
	//			off_set = block_len * block_id;
	//			int off_setx = block_id % block_numx, off_sety = block_id / block_numx % block_numy, off_setz = block_id / (block_numx * block_numy);
	//			for (int k = 0; k < MIN_TRANSFER; k++) {
	//				for (int j = 0; j < MIN_TRANSFER; j++) {
	//					for (int i = 0; i < MIN_TRANSFER; i++) {
	//						int x = i + off_setx * MIN_TRANSFER, y = j + off_sety * MIN_TRANSFER, z = k + off_setz * MIN_TRANSFER;
	//						int id = off_set + k * MIN_TRANSFER * MIN_TRANSFER + j * MIN_TRANSFER + i;
	//						int idlexi = x + (y + z * resoy) * resox;
	//						lexirho[idlexi] = rho[id];
	//					}
	//				}
	//			}
	//		}
	//		});
	//}
	//for (auto& t : workers) {
	//	if (t.joinable()) t.join();
	//}
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