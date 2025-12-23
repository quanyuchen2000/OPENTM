#define _USE_MATH_DEFINES
#include "grid.h"
#include <stdexcept>
#include <algorithm>
#include <numeric>
#include <thread>
#include "utils.h"
#include "matlab/matlab_utils.h"
#include <exception>
#include <Eigen/Eigen>
#include <Eigen/Eigenvalues>
#include "templateMatrix.h"
#include "voxelIO/openvdb_wrapper_t.h"
#include <map>
#include "cuda_profiler_api.h"
#include "tictoc.h"

using namespace homo;
using VT = homo::Grid_H::VT;
void Grid_H::buildRoot(int xreso, int yreso, int zreso, GridConfig config) 
{
	gridConfig = config;

	if (xreso > 1024 || yreso > 1024 || zreso > 1024) {
		throw std::runtime_error("axis resolution cannot exceed 1024");
	}

	double xlog2 = log2(xreso);
	double ylog2 = log2(yreso);
	double zlog2 = log2(zreso);

	int baseCoarsestReso = std::pow(2, 2);

	int largestCoarseLevel[3] = {
		(std::max)(std::floor(xlog2) - 2, 0.),
		(std::max)(std::floor(ylog2) - 2, 0.),
		(std::max)(std::floor(zlog2) - 2, 0.),
	};

	std::cout << "Largest coarse level " << largestCoarseLevel[0]
		<< ", " << largestCoarseLevel[1]
		<< ", " << largestCoarseLevel[2] << std::endl;

	double xc = xlog2 - largestCoarseLevel[0];
	double yc = ylog2 - largestCoarseLevel[1];
	double zc = zlog2 - largestCoarseLevel[2];

	printf("(xc, yc, zc) = (%lf, %lf, %lf)\n", xc, yc, zc);

	int xcreso = std::ceil(pow(2, xc));
	int ycreso = std::ceil(pow(2, yc));
	int zcreso = std::ceil(pow(2, zc));

	availCoarseReso[0] = xcreso;
	availCoarseReso[1] = ycreso;
	availCoarseReso[2] = zcreso;

	// assemble stencil on the fly
	assemb_otf = true;

	// corrected resolution
	xreso = xcreso * pow(2, largestCoarseLevel[0]);
	yreso = ycreso * pow(2, largestCoarseLevel[1]);
	zreso = zcreso * pow(2, largestCoarseLevel[2]);

	rootCellReso[0] = cellReso[0] = xreso;
	rootCellReso[1] = cellReso[1] = yreso;
	rootCellReso[2] = cellReso[2] = zreso;

	is_root = true;

	for (int i = 0; i < 3; i++) {
		upCoarse[i] = 0;
		totalCoarse[i] = 0;
	}

	std::pair<int, int> nv_ne;
	if (xreso <= MIN_TRANSFER) {
		nv_ne = countGS();
	}
	else {
		nv_ne = countGS_template();
	}

	// allocate buffer
	size_t totalMem = allocateBuffer(nv_ne.first, nv_ne.second);
	// for root we set flags for blocks
	setFlags_g();
}

std::string Grid_H::getName(void)
{
	char buf[1000];
	sprintf_s(buf, "<%s_Grid_%d_%d_%d>", gridConfig.namePrefix.c_str(), cellReso[0], cellReso[1], cellReso[2]);
	return buf;
}

std::shared_ptr<Grid_H> Grid_H::coarse2(GridConfig config)
{
	std::shared_ptr<Grid_H> coarseGrid(new Grid_H());
	coarseGrid->fine = this;
	Coarse = coarseGrid.get();
	coarseGrid->is_root = false;
	coarseGrid->assemb_otf = false;

	coarseGrid->availCoarseReso = availCoarseReso;
	coarseGrid->rootCellReso = rootCellReso;
	bool has_coarse = false;
	for (int i = 0; i < 3; i++) {
		if (cellReso[i] <= availCoarseReso[i]) {
			coarseGrid->upCoarse[i] = 0;
			downCoarse[i] = 0;
		}
		else if (cellReso[i] >= 256) {
			has_coarse = true;
			coarseGrid->upCoarse[i] = 2;
			while (cellReso[i] / coarseGrid->upCoarse[i] >= 256) {
				coarseGrid->upCoarse[i] *= 2;
			}
			downCoarse[i] = coarseGrid->upCoarse[i];
		}
		else {
			has_coarse = true;
			coarseGrid->upCoarse[i] = 2;
			downCoarse[i] = 2;
		}
	}
	if (!has_coarse) {
		Coarse = nullptr;
		return {};
	}

	for (int i = 0; i < 3; i++) {
		coarseGrid->cellReso[i] = cellReso[i] / downCoarse[i];
	}
	// determine eight colored GS nodes number
	auto nv_ne = coarseGrid->countGS();
	// allocate buffer
	size_t totalMem = coarseGrid->allocateBuffer(nv_ne.first, nv_ne.second);

	coarseGrid->setFlags_g();

	return coarseGrid;
}
// padding left and right one element for data alignment 
// ** depends on cellReso[3]
std::pair<int, int> Grid_H::countGS(void)
{
	printf("%s Enumerating GS...\n", getName().c_str());
	printf("cell = [%d, %d, %d]\n", cellReso[0], cellReso[1], cellReso[2]);
	int n_gsvertex[8] = {};
	for (int i = 0; i < 8; i++) {
		int org[3] = { i % 2, i / 2 % 2, i / 4 };
		for (int k = 0; k < 3; k++) {
			gsVertexReso[k][i] = (cellReso[k] - org[k] + 2) / 2 + 1;
		}
		n_gsvertex[i] = gsVertexReso[0][i] * gsVertexReso[1][i] * gsVertexReso[2][i];
		gsVertexSetValid[i] = n_gsvertex[i];
		// ceil to multiple of 32
		n_gsvertex[i] = 32 * (n_gsvertex[i] / 32 + bool(n_gsvertex[i] % 32));
		printf("gv[%d] = %d (%d)\n", i, gsVertexSetValid[i], n_gsvertex[i]);
		gsVertexSetRound[i] = n_gsvertex[i];
		int endid = 0;
		for (int j = 0; j < i + 1; j++) {
			endid += n_gsvertex[j];
		}
		gsVertexSetEnd[i] = endid;
	}
	int nv = std::accumulate(n_gsvertex, n_gsvertex + 8, 0);
	printf("Total rounded vertex %d\n", nv);

	int n_gscell[8] = {};
	for (int i = 0; i < 8; i++) {
		int org[3] = { i % 2, i / 2 % 2, i / 4 };
		for (int k = 0; k < 3; k++) {
			gsCellReso[k][i] = ((cellReso[k] + 1 - org[k]) / 2 + 1);
		}
		n_gscell[i] = gsCellReso[0][i] * gsCellReso[1][i] * gsCellReso[2][i];
		gsCellSetValid[i] = n_gscell[i];
		n_gscell[i] = 32 * (n_gscell[i] / 32 + bool(n_gscell[i] % 32));
		printf("ge[%d] = %d (%d)\n", i, gsCellSetValid[i], n_gscell[i]);
		gsCellSetRound[i] = n_gscell[i];
		int endid = 0;
		for (int j = 0; j < i + 1; j++) endid += n_gscell[j];
		gsCellSetEnd[i] = endid;
	}
	int ne = std::accumulate(n_gscell, n_gscell + 8, 0);
	printf("Total rounded cell %d\n", ne);
	return { nv,ne };
}

std::pair<int, int> Grid_H::countGS_template(void)
{
	printf("%s Enumerating GS...\n", getName().c_str());
	printf("template cell = [%d, %d, %d]\n", MIN_TRANSFER, MIN_TRANSFER, MIN_TRANSFER);
	int n_gsvertex[8] = {};
	for (int i = 0; i < 8; i++) {
		int org[3] = { i % 2, i / 2 % 2, i / 4 };
		for (int k = 0; k < 3; k++) {
			gsVertexReso[k][i] = (MIN_TRANSFER - org[k] + 2) / 2 + 1;
		}
		n_gsvertex[i] = gsVertexReso[0][i] * gsVertexReso[1][i] * gsVertexReso[2][i];
		gsVertexSetValid[i] = n_gsvertex[i];
		// ceil to multiple of 32
		n_gsvertex[i] = 32 * (n_gsvertex[i] / 32 + bool(n_gsvertex[i] % 32));
		printf("gv[%d] = %d (%d)\n", i, gsVertexSetValid[i], n_gsvertex[i]);
		gsVertexSetRound[i] = n_gsvertex[i];
		int endid = 0;
		for (int j = 0; j < i + 1; j++) {
			endid += n_gsvertex[j];
		}
		gsVertexSetEnd[i] = endid;
	}
	int nv = std::accumulate(n_gsvertex, n_gsvertex + 8, 0);
	printf("Total rounded vertex %d\n", nv);

	int n_gscell[8] = {};
	for (int i = 0; i < 8; i++) {
		int org[3] = { i % 2, i / 2 % 2, i / 4 };
		for (int k = 0; k < 3; k++) {
			gsCellReso[k][i] = ((MIN_TRANSFER + 1 - org[k]) / 2 + 1);
		}
		n_gscell[i] = gsCellReso[0][i] * gsCellReso[1][i] * gsCellReso[2][i];
		gsCellSetValid[i] = n_gscell[i];
		n_gscell[i] = 32 * (n_gscell[i] / 32 + bool(n_gscell[i] % 32));
		printf("ge[%d] = %d (%d)\n", i, gsCellSetValid[i], n_gscell[i]);
		gsCellSetRound[i] = n_gscell[i];
		int endid = 0;
		for (int j = 0; j < i + 1; j++) endid += n_gscell[j];
		gsCellSetEnd[i] = endid;
	}
	int ne = std::accumulate(n_gscell, n_gscell + 8, 0);
	printf("Total rounded cell %d\n", ne);
	return { nv,ne };
}

void Grid_H::v_reset_h(VT* v, int len)
{
	memset(v, 0, sizeof(VT) * len);
}
//Grid_H::~Grid_H() {
//	cudaHostUnregister(f_h.data());
//	cudaHostUnregister(r_h.data());
//	for (int i = 0; i < 3; i++) {
//		cudaHostUnregister(uchar[i].data());
//	}
//}
size_t Grid_H::allocateBuffer(int nv, int ne) 
{
	size_t total_gpu = 0;
	size_t total_cpu = 0;
	// judge whether to use host memory
	use_host_memory = (cellReso[0] >= MIN_TRANSFER);
	if (use_host_memory){
		int total_nv = (cellReso[0]/MIN_TRANSFER * cellReso[1]/MIN_TRANSFER * cellReso[2]/MIN_TRANSFER) * nv;
		// the total data
		if (is_root) {
			f_h.resize(total_nv, 0);
			cudaHostRegister(f_h.data(), f_h.size() * sizeof(float), cudaHostRegisterPortable);
			r_h.resize(total_nv, 0);
			cudaHostRegister(r_h.data(), r_h.size() * sizeof(float), cudaHostRegisterPortable);
		}
		// the block used on device
		u_g[0] = getMem().addBuffer(homoutils::formated("%s_u_%d", getName().c_str()), nv * sizeof(VT))->data<VT>();
		f_g[0] = getMem().addBuffer(homoutils::formated("%s_f_%d", getName().c_str()), nv * sizeof(VT))->data<VT>();
		r_g[0] = getMem().addBuffer(homoutils::formated("%s_r_%d", getName().c_str()), nv * sizeof(VT))->data<VT>();
		u_g[1] = getMem().addBuffer(homoutils::formated("%s_u1_%d", getName().c_str()), nv * sizeof(VT))->data<VT>();
		f_g[1] = getMem().addBuffer(homoutils::formated("%s_f1_%d", getName().c_str()), nv * sizeof(VT))->data<VT>();
		r_g[1] = getMem().addBuffer(homoutils::formated("%s_r1_%d", getName().c_str()), nv * sizeof(VT))->data<VT>();
		total_gpu += nv * 6 * sizeof(VT);
		total_cpu += total_nv * 3 * sizeof(VT);
	}
	else{
		u_g[0] = getMem().addBuffer(homoutils::formated("%s_u_%d", getName().c_str()), nv * sizeof(VT))->data<VT>();
		f_g[0] = getMem().addBuffer(homoutils::formated("%s_f_%d", getName().c_str()), nv * sizeof(VT))->data<VT>();
		r_g[0] = getMem().addBuffer(homoutils::formated("%s_r_%d", getName().c_str()), nv * sizeof(VT))->data<VT>();
		total_gpu += nv * 3 * sizeof(VT);
	}

	// warning and todo
	if (!is_root) {
		for (int i = 0; i < 27; i++) {
			stencil_g[i] = getMem().addBuffer(homoutils::formated("%s_st_%d", getName().c_str(), i), nv * sizeof(VT))->data<VT>();
		}
		total_gpu += nv * sizeof(VT) * 27;
	}
	if (!use_host_memory) {
		if (gridConfig.enableManagedMem) {
			for (int i = 0; i < 3; i++) {
				// here we use useless memory should be cut off
				uchar_h[i] = getMem().addBuffer(homoutils::formated("%s_uchost_%d_%d", getName().c_str(), i), nv * sizeof(VT), Managed)->data<VT>();
				v_reset(uchar_h[i], nv);
				total_gpu += nv * sizeof(VT);
				total_cpu += nv * sizeof(VT);
			}
		}
		else {
			for (int i = 0; i < 3; i++) {
				uchar_h[i] = getMem().addBuffer(homoutils::formated("%s_uchost_%d_%d", getName().c_str(), i), nv * sizeof(VT), Hostheap)->data<VT>();
				memset(uchar_h[i], 0, sizeof(VT) * nv);
				total_cpu += nv * sizeof(VT);
			}
		}
	}
	else {
		int total_nv = (cellReso[0] / MIN_TRANSFER * cellReso[1] / MIN_TRANSFER * cellReso[2] / MIN_TRANSFER) * nv;
		if (is_root) {
			for (int i = 0; i < 3; i++) {
				uchar.push_back(std::vector<VT>(total_nv, 0));
				cudaHostRegister(uchar[i].data(), uchar[i].size() * sizeof(float), cudaHostRegisterPortable);
				uchar_h[i] = getMem().addBuffer(homoutils::formated("%s_uchost_%d_%d", getName().c_str(), i), nv * sizeof(VT), Managed)->data<VT>();
				v_reset(uchar_h[i], nv);
				total_gpu += nv * sizeof(VT);
				total_cpu += nv * sizeof(VT);
			}
		}
	}
	vertflag = getMem().addBuffer<VertexFlags>(homoutils::formated("%s_vflag", getName().c_str()), nv)->data<VertexFlags>();
	cellflag = getMem().addBuffer<CellFlags>(homoutils::formated("%s_cflag", getName().c_str()), ne)->data<CellFlags>();
	total_gpu += nv * sizeof(VertexFlags);
	total_gpu += ne * sizeof(CellFlags);

	if (is_root) {
		total_gpu += ne * sizeof(float);
		rho_g = getMem().addBuffer(homoutils::formated("%s_rho", getName().c_str()), ne * sizeof(float))->data<float>();
	}
	// tempBuffer is defined here for transfer
	if (is_root && use_host_memory) {
		getMem().addBuffer("temp_rho0", pow(MIN_TRANSFER + 2, 3) * sizeof(VT));
		getMem().addBuffer("temp_rho1", pow(MIN_TRANSFER + 2, 3) * sizeof(VT));
		getMem().addBuffer("filter_rho0", pow(MIN_TRANSFER + 4, 3) * sizeof(VT));
		getMem().addBuffer("filter_rho1", pow(MIN_TRANSFER + 4, 3) * sizeof(VT));
		total_gpu += 2 * pow(MIN_TRANSFER + 2, 3) * sizeof(VT);
		stream.resize(2);
		cudaStreamCreate(&stream[0]);
		cudaStreamCreate(&stream[1]);
		cudaEventCreate(&ready_event);
	}
	printf("%s allocated %zd MB GPU memory, %zd MB CPU memory\n", getName().c_str(), total_gpu / 1024 / 1024, total_cpu / 1024 / 1024);
	return total_gpu;
}


inline int lexi2gs(int lexpos[3], int gsreso[3][8], int gsend[8], bool padded = false) {
	int pos[3] = { lexpos[0], lexpos[1], lexpos[2] };
	if (!padded) {
		pos[0] += 1; pos[1] += 1; pos[2] += 1;
	}
	int org[3] = { pos[0] % 2, pos[1] % 2, pos[2] % 2 };
	int gscolor = org[0] + org[1] * 2 + org[2] * 4;
	pos[0] /= 2; pos[1] /= 2; pos[2] /= 2;
	int gsid = (gscolor == 0 ? 0 : gsend[gscolor - 1]) +
		pos[0] +
		pos[1] * gsreso[0][gscolor] +
		pos[2] * gsreso[0][gscolor] * gsreso[1][gscolor];
	return gsid;
}

VT* homo::Grid_H::getDisplacement(void)
{
	return u_g[0];
}

void homo::Grid_H::update(std::vector<float> &rho) {
	rho_h = &rho;
}
double homo::Grid_H::residual(void)
{
	return v_norm(r_g[0]);
}
void Grid_H::loadu() {
	std::ifstream fin("64_uh.txt");
	std::string line;
	std::vector<float> inputvec(pow(cellReso[0] + 1, 3));
	while (getline(fin, line)) {
		std::istringstream iss(line);
		for (auto& x : inputvec) {
			float value;
			iss >> value;
			x = value;
		}
	}
	fin.close();
	int blockx, blocky, blockz;
	blockx = cellReso[0] / MIN_TRANSFER;
	blocky = cellReso[1] / MIN_TRANSFER;
	blockz = cellReso[2] / MIN_TRANSFER;
	for (int blockid = 0; blockid < blockx * blocky * blockz; blockid++) {
		int bx = blockid % blockx;
		int by = (blockid / blockx) % blocky;
		int bz = blockid / (blockx * blocky);
		for (int posx = 0; posx < MIN_TRANSFER + 1; posx++) {
			for (int posy = 0; posy < MIN_TRANSFER + 1; posy++) {
				for (int posz = 0; posz < MIN_TRANSFER + 1; posz++) {
					int pos[3] = { posx, posy, posz };
					int gsid = lexi2gs(pos, gsVertexReso, gsVertexSetEnd);
					int offset = blockid * n_gsvertices();
					int lexid = (bx * MIN_TRANSFER + posx) + cellReso[0] * (by * MIN_TRANSFER + posy) + cellReso[0] * cellReso[1] * (bz * MIN_TRANSFER + posz);
					u_h[gsid + offset] = inputvec[lexid];
				}
			}
		}
	}
}
void Grid_H::lexiufile(int direct) {
	int blockx, blocky, blockz;
	blockx = cellReso[0] / MIN_TRANSFER;
	blocky = cellReso[1] / MIN_TRANSFER;
	blockz = cellReso[2] / MIN_TRANSFER;
	std::vector<float> outputvec(pow(cellReso[0]+1, 3));
	for (int blockid = 0; blockid < blockx * blocky * blockz; blockid++) {
		int bx = blockid % blockx;
		int by = (blockid / blockx) % blocky;
		int bz = blockid / (blockx * blocky);
		for (int posx = 0; posx < MIN_TRANSFER+1; posx++) {
			for (int posy = 0; posy < MIN_TRANSFER+1; posy++) {
				for (int posz = 0; posz < MIN_TRANSFER+1; posz++) {
					int pos[3] = { posx, posy, posz };
					int gsid = lexi2gs(pos, gsVertexReso, gsVertexSetEnd);
					int offset = blockid * n_gsvertices();
					float u = uchar[direct][gsid+offset];
					int lexid = (bx * MIN_TRANSFER + posx) + cellReso[0] * (by * MIN_TRANSFER + posy) + cellReso[0] * cellReso[1] * (bz * MIN_TRANSFER + posz);
					outputvec[lexid] = u;
				}
			}
		}
	}

	std::ofstream fout("32_uh.txt");
	for (const auto& x : outputvec) {
		fout << x << " ";
	}
	fout << "\n";
	fout.close();
}
void Grid_H::useF() {
	enforce_unit_macro_strain_host();
}

void Grid_H::useFchar(int k)
{
	useGrid_g();
	if (cellReso[0] >= MIN_TRANSFER) {
		enforce_unit_macro_strain_host(k);
		pad_vertex_data_host(f_h);
	}
	else {
		enforce_unit_macro_strain(k);
		pad_vertex_data(f_g);
	}
}
void Grid_H::useU()
{
	useGrid_g();
	if (cellReso[0] >= MIN_TRANSFER) {
		enforce_U();
	}
}

//void Grid_H::pad_vertex_data_host(std::vector<float>& vec) {
//	int off_set = 0;
//	// for each block init block
//	int block_numx = (cellReso[0] / MIN_TRANSFER);
//	int block_numy = (cellReso[1] / MIN_TRANSFER);
//	int block_numz = (cellReso[2] / MIN_TRANSFER);
//
//	int block_num = block_numx * block_numy * block_numz;
//	int block_len = n_gscells();
//
//	for (int block_id = 0; block_id < block_num; block_id++) {
//		off_set = block_len * block_id;
//		int off_setx = block_id % block_numx, off_sety = block_id / block_numx % block_numy, off_setz = block_id / (block_numx * block_numy);
//		for (int k = 0; k < MIN_TRANSFER + 3; k++) {
//			for (int j = 0; j < MIN_TRANSFER + 3; j++) {
//				for (int i = 0; i < MIN_TRANSFER + 3; i++) {
//					if (i == 0 || i == MIN_TRANSFER + 2 || j == 0 || j == MIN_TRANSFER + 2 || k == 0 || k == MIN_TRANSFER + 2) {
//						int pos[3] = { i, j, k };
//						int boundary = lexi2gs(pos, gsVertexReso, gsVertexSetEnd, true);
//						int gsid_tar = block_id * n_gsvertices() + lexi2gs(pos, gsVertexReso, gsVertexSetEnd, true);
//						if (i == 0) {
//							off_setx = (off_setx - 1 + block_numx) % block_numx;
//							pos[0] = MIN_TRANSFER;
//						}
//						if (i == MIN_TRANSFER + 2) {
//							off_setx = (off_setx + 1) % block_numx;
//							pos[0] = 2;
//						}
//						if (j == 0) {
//							off_sety = (off_sety - 1 + block_numy) % block_numy;
//							pos[1] = MIN_TRANSFER;
//						}
//						if (j == MIN_TRANSFER + 2) {
//							off_sety = (off_sety + 1) % block_numy;
//							pos[1] = 2;
//						}
//						if (k == 0) {
//							off_setz = (off_setz - 1 + block_numz) % block_numz;
//							pos[2] = MIN_TRANSFER;
//						}
//						if (k == MIN_TRANSFER + 2) {
//							off_setz = (off_setz + 1) % block_numz;
//							pos[2] = 2;
//						}
//						int src_bid = off_setx + off_sety * block_numx + off_setz * block_numx * block_numy;
//						int gsid_src = src_bid * n_gsvertices() + lexi2gs(pos, gsVertexReso, gsVertexSetEnd, true);
//						vec[gsid_tar] = vec[gsid_src];
//					}
//				}
//			}
//		}
//	}
//}
void Grid_H::pad_vertex_data_host(std::vector<float>& vec) {
	int block_numx = cellReso[0] / MIN_TRANSFER;
	int block_numy = cellReso[1] / MIN_TRANSFER;
	int block_numz = cellReso[2] / MIN_TRANSFER;
	int block_num = block_numx * block_numy * block_numz;
	const unsigned int num_threads = std::thread::hardware_concurrency();
	std::vector<std::thread> workers;

	const int blocks_per_thread = block_num / num_threads;
	int remaining_blocks = block_num % num_threads;
	int start_block = 0;

	for (unsigned int t = 0; t < num_threads; ++t) {
		int end_block = start_block + blocks_per_thread + (t < remaining_blocks ? 1 : 0);
		workers.emplace_back([&, start_block, end_block]() {
			for (int block_id = start_block; block_id < end_block; ++block_id) {
				process_single_block(block_id, vec, block_numx, block_numy, block_numz);
			}
			});
		start_block = end_block;
	}

	for (auto& th : workers) {
		th.join();
	}
}

void Grid_H::process_single_block(int block_id, std::vector<float>& vec,
	int block_numx, int block_numy, int block_numz) {
	int off_setx = block_id % block_numx;
	int off_sety = (block_id / block_numx) % block_numy;
	int off_setz = block_id / (block_numx * block_numy);

	auto process_i_face = [&](int i_boundary) {
		int i = (i_boundary == 0) ? 0 : MIN_TRANSFER + 2;
		for (int k = 0; k < MIN_TRANSFER + 3; ++k) {
			for (int j = 0; j < MIN_TRANSFER + 3; ++j) {
				int pos[3] = { i, j, k };
				int gsid_tar = block_id * n_gsvertices() + lexi2gs(pos, gsVertexReso, gsVertexSetEnd, true);

				int new_offx = (i == 0) ?
					(off_setx - 1 + block_numx) % block_numx :
					(off_setx + 1) % block_numx;
				pos[0] = (i == 0) ? MIN_TRANSFER : 2;

				int src_bid = new_offx + off_sety * block_numx + off_setz * block_numx * block_numy;
				int gsid_src = src_bid * n_gsvertices() + lexi2gs(pos, gsVertexReso, gsVertexSetEnd, true);
				vec[gsid_tar] = vec[gsid_src];
			}
		}
		};
	process_i_face(0);
	process_i_face(1);

	auto process_j_face = [&](int j_boundary) {
		int j = (j_boundary == 0) ? 0 : MIN_TRANSFER + 2;
		for (int k = 0; k < MIN_TRANSFER + 3; ++k) {
			for (int i = 1; i < MIN_TRANSFER + 2; ++i) {
				int pos[3] = { i, j, k };
				int gsid_tar = block_id * n_gsvertices() + lexi2gs(pos, gsVertexReso, gsVertexSetEnd, true);

				int new_offy = (j == 0) ?
					(off_sety - 1 + block_numy) % block_numy :
					(off_sety + 1) % block_numy;
				pos[1] = (j == 0) ? MIN_TRANSFER : 2;

				int src_bid = off_setx + new_offy * block_numx + off_setz * block_numx * block_numy;
				int gsid_src = src_bid * n_gsvertices() + lexi2gs(pos, gsVertexReso, gsVertexSetEnd, true);
				vec[gsid_tar] = vec[gsid_src];
			}
		}
		};
	process_j_face(0);
	process_j_face(1);

	auto process_k_face = [&](int k_boundary) {
		int k = (k_boundary == 0) ? 0 : MIN_TRANSFER + 2;
		for (int j = 1; j < MIN_TRANSFER + 2; ++j) {
			for (int i = 1; i < MIN_TRANSFER + 2; ++i) {
				int pos[3] = { i, j, k };
				int gsid_tar = block_id * n_gsvertices() + lexi2gs(pos, gsVertexReso, gsVertexSetEnd, true);

				int new_offz = (k == 0) ?
					(off_setz - 1 + block_numz) % block_numz :
					(off_setz + 1) % block_numz;
				pos[2] = (k == 0) ? MIN_TRANSFER : 2;

				int src_bid = off_setx + off_sety * block_numx + new_offz * block_numx * block_numy;
				int gsid_src = src_bid * n_gsvertices() + lexi2gs(pos, gsVertexReso, gsVertexSetEnd, true);
				vec[gsid_tar] = vec[gsid_src];
			}
		}
		};
	process_k_face(0);
	process_k_face(1);
}
void Grid_H::process_block_boundary(int bid, std::vector<VT>& v) {
	const int block_numx = cellReso[0] / MIN_TRANSFER;
	const int block_numy = cellReso[1] / MIN_TRANSFER;
	const int block_numz = cellReso[2] / MIN_TRANSFER;

	int off_setx = bid % block_numx;
	int off_sety = (bid / block_numx) % block_numy;
	int off_setz = bid / (block_numx * block_numy);

	auto process_right_face = [&](int axis) { // axis:0=x,1=y,2=z
		for (int k = 1; k < MIN_TRANSFER + 2; ++k) {
			for (int j = 1; j < MIN_TRANSFER + 2; ++j) {
				for (int i = 1; i < MIN_TRANSFER + 2; ++i) {
					int pos[3] = { i, j, k };
					if (pos[axis] != MIN_TRANSFER + 1) continue;

					int src_offx = off_setx + (axis == 0);
					int src_offy = off_sety + (axis == 1);
					int src_offz = off_setz + (axis == 2);

					src_offx = (src_offx + block_numx) % block_numx;
					src_offy = (src_offy + block_numy) % block_numy;
					src_offz = (src_offz + block_numz) % block_numz;

					int gsid_tar = bid * n_gsvertices() +
						lexi2gs(pos, gsVertexReso, gsVertexSetEnd, true);

					pos[axis] = 1;
					int src_bid = src_offx + src_offy * block_numx +
						src_offz * block_numx * block_numy;
					int gsid_src = src_bid * n_gsvertices() +
						lexi2gs(pos, gsVertexReso, gsVertexSetEnd, true);

					v[gsid_tar] = v[gsid_src];
				}
			}
		}
		};

	auto process_padding_layer = [&](int axis, int layer) {
		for (int k = 0; k < MIN_TRANSFER + 3; ++k) {
			for (int j = 0; j < MIN_TRANSFER + 3; ++j) {
				for (int i = 0; i < MIN_TRANSFER + 3; ++i) {
					int pos[3] = { i, j, k };
					if (pos[axis] != layer) continue;

					int src_offx = off_setx - (axis == 0 && layer == 0) +
						(axis == 0 && layer == MIN_TRANSFER + 2);
					int src_offy = off_sety - (axis == 1 && layer == 0) +
						(axis == 1 && layer == MIN_TRANSFER + 2);
					int src_offz = off_setz - (axis == 2 && layer == 0) +
						(axis == 2 && layer == MIN_TRANSFER + 2);

					src_offx = (src_offx + block_numx) % block_numx;
					src_offy = (src_offy + block_numy) % block_numy;
					src_offz = (src_offz + block_numz) % block_numz;

					int gsid_tar = bid * n_gsvertices() +
						lexi2gs(pos, gsVertexReso, gsVertexSetEnd, true);

					pos[axis] = (layer == 0) ? MIN_TRANSFER : 2;
					int src_bid = src_offx + src_offy * block_numx +
						src_offz * block_numx * block_numy;
					int gsid_src = src_bid * n_gsvertices() +
						lexi2gs(pos, gsVertexReso, gsVertexSetEnd, true);

					v[gsid_tar] = v[gsid_src];
				}
			}
		}
		};

	process_right_face(0);
	process_right_face(1);
	process_right_face(2);

	process_padding_layer(0, 0);
	process_padding_layer(0, MIN_TRANSFER + 2);
	process_padding_layer(1, 0);
	process_padding_layer(1, MIN_TRANSFER + 2);
	process_padding_layer(2, 0);
	process_padding_layer(2, MIN_TRANSFER + 2);
}

void homo::Grid_H::enforce_vertex_boundary(std::vector<VT>& v) {
	// seam align to right 
	// update padding
	int block_num = asp.block_numx * asp.block_numy * asp.block_numz;
	// enforce_period_vertex
	const unsigned num_threads = std::thread::hardware_concurrency();

	next_bid.store(0);

	for (unsigned t = 0; t < num_threads; ++t) {
		asp.workers.emplace_back([&, block_num] {
			while (true) {
				const int bid = next_bid.fetch_add(1, std::memory_order_relaxed);
				if (bid >= block_num) break;
				const int block_numx = asp.block_numx;
				const int block_numy = asp.block_numy;
				const int block_numz = asp.block_numz;
				const std::vector<int> offset = {
					bid % block_numx,
					(bid / block_numx) % block_numy,
					bid / (block_numx * block_numy)
				};
				const int ngsv = n_gsvertices();
				for (int i = 1; i < MIN_TRANSFER + 2; i++) {
					for (int j = 1; j < MIN_TRANSFER + 2; j++) {
						{
							int pos[3] = { MIN_TRANSFER + 1, i, j };
							const int gsid_tar = bid * ngsv + lexi2gs(pos, gsVertexReso, gsVertexSetEnd, true);
							auto new_offset = offset;
							new_offset[0] = (new_offset[0] + 1) % block_numx;
							pos[0] = 1;
							if (i == MIN_TRANSFER + 1) {
								pos[1] = 1;
								new_offset[1] = (new_offset[1] + 1) % block_numy;
							}
							if (j == MIN_TRANSFER + 1) {
								pos[2] = 1;
								new_offset[2] = (new_offset[2] + 1) % block_numz;
							}
							const int src_bid = new_offset[0] + new_offset[1] * block_numx + new_offset[2] * block_numx * block_numy;
							const int gsid_src = src_bid * ngsv + lexi2gs(pos, gsVertexReso, gsVertexSetEnd, true);
							v[gsid_tar] = v[gsid_src];
						}
						{
							int pos[3] = { i, MIN_TRANSFER + 1, j };
							const int gsid_tar = bid * ngsv + lexi2gs(pos, gsVertexReso, gsVertexSetEnd, true);
							auto new_offset = offset;
							new_offset[1] = (new_offset[1] + 1) % block_numy;
							pos[1] = 1;
							if (i == MIN_TRANSFER + 1) {
								pos[0] = 1;
								new_offset[0] = (new_offset[0] + 1) % block_numx;
							}
							if (j == MIN_TRANSFER + 1) {
								pos[2] = 1;
								new_offset[2] = (new_offset[2] + 1) % block_numz;
							}
							const int src_bid = new_offset[0] + new_offset[1] * block_numx + new_offset[2] * block_numx * block_numy;
							const int gsid_src = src_bid * ngsv + lexi2gs(pos, gsVertexReso, gsVertexSetEnd, true);
							v[gsid_tar] = v[gsid_src];
						}
						{
							int pos[3] = { i, j, MIN_TRANSFER + 1 };
							const int gsid_tar = bid * ngsv + lexi2gs(pos, gsVertexReso, gsVertexSetEnd, true);
							auto new_offset = offset;
							new_offset[2] = (new_offset[2] + 1) % block_numz;
							pos[2] = 1;
							if (i == MIN_TRANSFER + 1) {
								pos[0] = 1;
								new_offset[0] = (new_offset[0] + 1) % block_numx;
							}
							if (j == MIN_TRANSFER + 1) {
								pos[1] = 1;
								new_offset[1] = (new_offset[1] + 1) % block_numy;
							}
							const int src_bid = new_offset[0] + new_offset[1] * block_numx + new_offset[2] * block_numx * block_numy;
							const int gsid_src = src_bid * ngsv + lexi2gs(pos, gsVertexReso, gsVertexSetEnd, true);
							v[gsid_tar] = v[gsid_src];
						}
					}
				}

				for (int i : {0, MIN_TRANSFER + 2}) {
					for (int j = 0; j < MIN_TRANSFER + 3; j++) {
						for (int k = 0; k < MIN_TRANSFER + 3; k++) {
							int pos[3] = { i, j, k };
							int gsid_tar = bid * ngsv + lexi2gs(pos, gsVertexReso, gsVertexSetEnd, true);
							auto offsetx = offset;
							offsetx[0] = (i == 0) ?
								((offsetx[0] - 1 + block_numx) % block_numx) :
								(offsetx[0] + 1) % block_numx;
							pos[0] = (i == 0) ? MIN_TRANSFER : 2;

							offsetx[1] = (j == 0) ?
								((offsetx[1] - 1 + block_numy) % block_numy) :
								(j == MIN_TRANSFER + 2) ? (offsetx[1] + 1) % block_numy : offsetx[1];
							pos[1] = (j == 0) ? MIN_TRANSFER : (j == MIN_TRANSFER + 2) ? 2 : j;

							offsetx[2] = (k == 0) ?
								((offsetx[2] - 1 + block_numz) % block_numz) :
								(k == MIN_TRANSFER + 2) ? (offsetx[2] + 1) % block_numz : offsetx[2];
							pos[2] = (k == 0) ? MIN_TRANSFER : (k == MIN_TRANSFER + 2) ? 2 : k;
							int src_bid = offsetx[0] + offsetx[1] * block_numx + offsetx[2] * block_numx * block_numy;
							int gsid_src = src_bid * ngsv + lexi2gs(pos, gsVertexReso, gsVertexSetEnd, true);
							v[gsid_tar] = v[gsid_src];
						}
					}
				}

				for (int j : {0, MIN_TRANSFER + 2}) {
					for (int i = 0; i < MIN_TRANSFER + 3; i++) {
						for (int k = 0; k < MIN_TRANSFER + 3; k++) {
							int pos[3] = { i, j, k };
							int gsid_tar = bid * ngsv + lexi2gs(pos, gsVertexReso, gsVertexSetEnd, true);
							auto offsety = offset;

							offsety[1] = (j == 0) ?
								((offsety[1] - 1 + block_numy) % block_numy) :
								(offsety[1] + 1) % block_numy;
							pos[1] = (j == 0) ? MIN_TRANSFER : 2;

							offsety[0] = (i == 0) ?
								((offsety[0] - 1 + block_numx) % block_numx) :
								(i == MIN_TRANSFER + 2) ? (offsety[0] + 1) % block_numx : offsety[0];
							pos[0] = (i == 0) ? MIN_TRANSFER : (i == MIN_TRANSFER + 2) ? 2 : i;

							offsety[2] = (k == 0) ?
								((offsety[2] - 1 + block_numz) % block_numz) :
								(k == MIN_TRANSFER + 2) ? (offsety[2] + 1) % block_numz : offsety[2];
							pos[2] = (k == 0) ? MIN_TRANSFER : (k == MIN_TRANSFER + 2) ? 2 : k;

							const int src_bid = offsety[0] + offsety[1] * block_numx + offsety[2] * (block_numx * block_numy);
							const int gsid_src = src_bid * ngsv + lexi2gs(pos, gsVertexReso, gsVertexSetEnd, true);
							v[gsid_tar] = v[gsid_src];
						}
					}
				}

				for (int k : {0, MIN_TRANSFER + 2}) {
					for (int i = 0; i < MIN_TRANSFER + 3; i++) {
						for (int j = 0; j < MIN_TRANSFER + 3; j++) {
							int pos[3] = { i, j, k };
							int gsid_tar = bid * ngsv + lexi2gs(pos, gsVertexReso, gsVertexSetEnd, true);
							auto offsetz = offset;

							offsetz[2] = (k == 0) ?
								((offsetz[2] - 1 + block_numz) % block_numz) :
								(offsetz[2] + 1) % block_numz;
							pos[2] = (k == 0) ? MIN_TRANSFER : 2;

							offsetz[0] = (i == 0) ?
								((offsetz[0] - 1 + block_numx) % block_numx) :
								(i == MIN_TRANSFER + 2) ? (offsetz[0] + 1) % block_numx : offsetz[0];
							pos[0] = (i == 0) ? MIN_TRANSFER : (i == MIN_TRANSFER + 2) ? 2 : i;

							offsetz[1] = (j == 0) ?
								((offsetz[1] - 1 + block_numy) % block_numy) :
								(j == MIN_TRANSFER + 2) ? (offsetz[1] + 1) % block_numy : offsetz[1];
							pos[1] = (j == 0) ? MIN_TRANSFER : (j == MIN_TRANSFER + 2) ? 2 : j;

							const int src_bid = offsetz[0] + offsetz[1] * block_numx + offsetz[2] * (block_numx * block_numy);
							const int gsid_src = src_bid * ngsv + lexi2gs(pos, gsVertexReso, gsVertexSetEnd, true);
							v[gsid_tar] = v[gsid_src];
						}
					}
				}
			}
		});
	}

	for (auto& th : asp.workers) {
		if (th.joinable()) th.join();
	}
	asp.workers.clear();
}

void homo::Grid_H::enforce_vertex_boundary_block(std::vector<VT>& v, int blockid) {
	asp.workers.clear();
	asp.blockid = blockid;
	next_bid.store(0);
	for (unsigned t = 0; t < 9; ++t) {
		asp.workers.emplace_back([&] {
			while (true) {
				const int taskid = next_bid.fetch_add(1, std::memory_order_relaxed);
				if (taskid >= 9) break;
				const int block_numx = asp.block_numx;
				const int block_numy = asp.block_numy;
				const int block_numz = asp.block_numz;
				const int bid = asp.blockid;
				const std::vector<int> offset = {
					bid % block_numx,
					(bid / block_numx) % block_numy,
					bid / (block_numx * block_numy)
				};
				const int ngsv = n_gsvertices();
				switch (taskid) {
				case 0:
					for (int i = 1; i < MIN_TRANSFER + 2; i++) {
						for (int j = 1; j < MIN_TRANSFER + 2; j++) {
							int pos[3] = { MIN_TRANSFER + 1, i, j };
							const int gsid_tar = bid * ngsv + lexi2gs(pos, gsVertexReso, gsVertexSetEnd, true);
							auto new_offset = offset;
							new_offset[0] = (new_offset[0] + 1) % block_numx;
							pos[0] = 1;
							if (i == MIN_TRANSFER + 1) {
								pos[1] = 1;
								new_offset[1] = (new_offset[1] + 1) % block_numy;
							}
							if (j == MIN_TRANSFER + 1) {
								pos[2] = 1;
								new_offset[2] = (new_offset[2] + 1) % block_numz;
							}
							const int src_bid = new_offset[0] + new_offset[1] * block_numx + new_offset[2] * block_numx * block_numy;
							const int gsid_src = src_bid * ngsv + lexi2gs(pos, gsVertexReso, gsVertexSetEnd, true);
							v[gsid_tar] = v[gsid_src];
						}
					}
					break;
				case 1:
					for (int j = 1; j < MIN_TRANSFER + 2; j++) {
						for (int i = 1; i < MIN_TRANSFER + 2; i++) {
							int pos[3] = { i, MIN_TRANSFER + 1, j };
							const int gsid_tar = bid * ngsv + lexi2gs(pos, gsVertexReso, gsVertexSetEnd, true);
							auto new_offset = offset;
							new_offset[1] = (new_offset[1] + 1) % block_numy;
							pos[1] = 1;
							if (i == MIN_TRANSFER + 1) {
								pos[0] = 1;
								new_offset[0] = (new_offset[0] + 1) % block_numx;
							}
							if (j == MIN_TRANSFER + 1) {
								pos[2] = 1;
								new_offset[2] = (new_offset[2] + 1) % block_numz;
							}
							const int src_bid = new_offset[0] + new_offset[1] * block_numx + new_offset[2] * block_numx * block_numy;
							const int gsid_src = src_bid * ngsv + lexi2gs(pos, gsVertexReso, gsVertexSetEnd, true);
							v[gsid_tar] = v[gsid_src];
						}
					}
					break;
				case 2:
					for (int j = 1; j < MIN_TRANSFER + 2; j++) {
						for (int i = 1; i < MIN_TRANSFER + 2; i++) {
							int pos[3] = { i, j, MIN_TRANSFER + 1 };
							const int gsid_tar = bid * ngsv + lexi2gs(pos, gsVertexReso, gsVertexSetEnd, true);
							auto new_offset = offset;
							new_offset[2] = (new_offset[2] + 1) % block_numz;
							pos[2] = 1;
							if (i == MIN_TRANSFER + 1) {
								pos[0] = 1;
								new_offset[0] = (new_offset[0] + 1) % block_numx;
							}
							if (j == MIN_TRANSFER + 1) {
								pos[1] = 1;
								new_offset[1] = (new_offset[1] + 1) % block_numy;
							}
							const int src_bid = new_offset[0] + new_offset[1] * block_numx + new_offset[2] * block_numx * block_numy;
							const int gsid_src = src_bid * ngsv + lexi2gs(pos, gsVertexReso, gsVertexSetEnd, true);
							v[gsid_tar] = v[gsid_src];
						}
					}
					break;
				case 3:
					for (int k = 0; k < MIN_TRANSFER + 3; k++) {
						for (int j = 0; j < MIN_TRANSFER + 3; j++) {
							int pos[3] = { 0, j, k };
							int gsid_tar = bid * ngsv + lexi2gs(pos, gsVertexReso, gsVertexSetEnd, true);
							auto offsetx = offset;
							offsetx[0] = (offsetx[0] - 1 + block_numx) % block_numx;
							pos[0] = MIN_TRANSFER;

							offsetx[1] = (j == 0) ?
								((offsetx[1] - 1 + block_numy) % block_numy) :
								(j == MIN_TRANSFER + 2) ? (offsetx[1] + 1) % block_numy : offsetx[1];
							pos[1] = (j == 0) ? MIN_TRANSFER : (j == MIN_TRANSFER + 2) ? 2 : j;

							offsetx[2] = (k == 0) ?
								((offsetx[2] - 1 + block_numz) % block_numz) :
								(k == MIN_TRANSFER + 2) ? (offsetx[2] + 1) % block_numz : offsetx[2];
							pos[2] = (k == 0) ? MIN_TRANSFER : (k == MIN_TRANSFER + 2) ? 2 : k;
							int src_bid = offsetx[0] + offsetx[1] * block_numx + offsetx[2] * block_numx * block_numy;
							int gsid_src = src_bid * ngsv + lexi2gs(pos, gsVertexReso, gsVertexSetEnd, true);
							v[gsid_tar] = v[gsid_src];
						}
					}
					break;
				case 4:
					for (int k = 0; k < MIN_TRANSFER + 3; k++) {
						for (int j = 0; j < MIN_TRANSFER + 3; j++) {
							int pos[3] = { MIN_TRANSFER + 2, j, k };
							int gsid_tar = bid * ngsv + lexi2gs(pos, gsVertexReso, gsVertexSetEnd, true);
							auto offsetx = offset;
							offsetx[0] = (offsetx[0] + 1) % block_numx;
							pos[0] = 2;

							offsetx[1] = (j == 0) ?
								((offsetx[1] - 1 + block_numy) % block_numy) :
								(j == MIN_TRANSFER + 2) ? (offsetx[1] + 1) % block_numy : offsetx[1];
							pos[1] = (j == 0) ? MIN_TRANSFER : (j == MIN_TRANSFER + 2) ? 2 : j;

							offsetx[2] = (k == 0) ?
								((offsetx[2] - 1 + block_numz) % block_numz) :
								(k == MIN_TRANSFER + 2) ? (offsetx[2] + 1) % block_numz : offsetx[2];
							pos[2] = (k == 0) ? MIN_TRANSFER : (k == MIN_TRANSFER + 2) ? 2 : k;
							int src_bid = offsetx[0] + offsetx[1] * block_numx + offsetx[2] * block_numx * block_numy;
							int gsid_src = src_bid * ngsv + lexi2gs(pos, gsVertexReso, gsVertexSetEnd, true);
							v[gsid_tar] = v[gsid_src];
						}
					}
					break;
				case 5:
					for (int k = 0; k < MIN_TRANSFER + 3; k++) {
						for (int i = 0; i < MIN_TRANSFER + 3; i++) {
							int pos[3] = { i, 0, k };
							int gsid_tar = bid * ngsv + lexi2gs(pos, gsVertexReso, gsVertexSetEnd, true);
							auto offsety = offset;

							offsety[1] = ((offsety[1] - 1 + block_numy) % block_numy);
							pos[1] = MIN_TRANSFER;

							offsety[0] = (i == 0) ?
								((offsety[0] - 1 + block_numx) % block_numx) :
								(i == MIN_TRANSFER + 2) ? (offsety[0] + 1) % block_numx : offsety[0];
							pos[0] = (i == 0) ? MIN_TRANSFER : (i == MIN_TRANSFER + 2) ? 2 : i;

							offsety[2] = (k == 0) ?
								((offsety[2] - 1 + block_numz) % block_numz) :
								(k == MIN_TRANSFER + 2) ? (offsety[2] + 1) % block_numz : offsety[2];
							pos[2] = (k == 0) ? MIN_TRANSFER : (k == MIN_TRANSFER + 2) ? 2 : k;

							const int src_bid = offsety[0] + offsety[1] * block_numx + offsety[2] * (block_numx * block_numy);
							const int gsid_src = src_bid * ngsv + lexi2gs(pos, gsVertexReso, gsVertexSetEnd, true);
							v[gsid_tar] = v[gsid_src];
						}
					}
					break;
				case 6:
					for (int k = 0; k < MIN_TRANSFER + 3; k++) {
						for (int i = 0; i < MIN_TRANSFER + 3; i++) {
							int pos[3] = { i, MIN_TRANSFER + 2, k };
							int gsid_tar = bid * ngsv + lexi2gs(pos, gsVertexReso, gsVertexSetEnd, true);
							auto offsety = offset;

							offsety[1] = (offsety[1] + 1) % block_numy;
							pos[1] = 2;

							offsety[0] = (i == 0) ?
								((offsety[0] - 1 + block_numx) % block_numx) :
								(i == MIN_TRANSFER + 2) ? (offsety[0] + 1) % block_numx : offsety[0];
							pos[0] = (i == 0) ? MIN_TRANSFER : (i == MIN_TRANSFER + 2) ? 2 : i;

							offsety[2] = (k == 0) ?
								((offsety[2] - 1 + block_numz) % block_numz) :
								(k == MIN_TRANSFER + 2) ? (offsety[2] + 1) % block_numz : offsety[2];
							pos[2] = (k == 0) ? MIN_TRANSFER : (k == MIN_TRANSFER + 2) ? 2 : k;

							const int src_bid = offsety[0] + offsety[1] * block_numx + offsety[2] * (block_numx * block_numy);
							const int gsid_src = src_bid * ngsv + lexi2gs(pos, gsVertexReso, gsVertexSetEnd, true);
							v[gsid_tar] = v[gsid_src];
						}
					}
					break;
				case 7:
					for (int j = 0; j < MIN_TRANSFER + 3; j++) {
						for (int i = 0; i < MIN_TRANSFER + 3; i++) {
							int pos[3] = { i, j, 0 };
							int gsid_tar = bid * ngsv + lexi2gs(pos, gsVertexReso, gsVertexSetEnd, true);
							auto offsetz = offset;

							offsetz[2] = ((offsetz[2] - 1 + block_numz) % block_numz);
							pos[2] = MIN_TRANSFER;

							offsetz[0] = (i == 0) ?
								((offsetz[0] - 1 + block_numx) % block_numx) :
								(i == MIN_TRANSFER + 2) ? (offsetz[0] + 1) % block_numx : offsetz[0];
							pos[0] = (i == 0) ? MIN_TRANSFER : (i == MIN_TRANSFER + 2) ? 2 : i;

							offsetz[1] = (j == 0) ?
								((offsetz[1] - 1 + block_numy) % block_numy) :
								(j == MIN_TRANSFER + 2) ? (offsetz[1] + 1) % block_numy : offsetz[1];
							pos[1] = (j == 0) ? MIN_TRANSFER : (j == MIN_TRANSFER + 2) ? 2 : j;

							const int src_bid = offsetz[0] + offsetz[1] * block_numx + offsetz[2] * (block_numx * block_numy);
							const int gsid_src = src_bid * ngsv + lexi2gs(pos, gsVertexReso, gsVertexSetEnd, true);
							v[gsid_tar] = v[gsid_src];
						}
					}
					break;
				case 8:
					for (int j = 0; j < MIN_TRANSFER + 3; j++) {
						for (int i = 0; i < MIN_TRANSFER + 3; i++) {
							int pos[3] = { i, j, MIN_TRANSFER + 2 };
							int gsid_tar = bid * ngsv + lexi2gs(pos, gsVertexReso, gsVertexSetEnd, true);
							auto offsetz = offset;

							offsetz[2] = (offsetz[2] + 1) % block_numz;
							pos[2] = 2;

							offsetz[0] = (i == 0) ?
								((offsetz[0] - 1 + block_numx) % block_numx) :
								(i == MIN_TRANSFER + 2) ? (offsetz[0] + 1) % block_numx : offsetz[0];
							pos[0] = (i == 0) ? MIN_TRANSFER : (i == MIN_TRANSFER + 2) ? 2 : i;

							offsetz[1] = (j == 0) ?
								((offsetz[1] - 1 + block_numy) % block_numy) :
								(j == MIN_TRANSFER + 2) ? (offsetz[1] + 1) % block_numy : offsetz[1];
							pos[1] = (j == 0) ? MIN_TRANSFER : (j == MIN_TRANSFER + 2) ? 2 : j;

							const int src_bid = offsetz[0] + offsetz[1] * block_numx + offsetz[2] * (block_numx * block_numy);
							const int gsid_src = src_bid * ngsv + lexi2gs(pos, gsVertexReso, gsVertexSetEnd, true);
							v[gsid_tar] = v[gsid_src];
						}
					}
					break;
				default:
					break;
				}
			}
			});
	}
}

void homo::Grid_H::enforce_vertex_boundary_block_seperate_version(std::vector<VT>& v, int blockid) {
	asp.workers.clear();
	asp.blockid = blockid;
	next_bid.store(0);
	asp.tasknum = (MIN_TRANSFER + 1) * (MIN_TRANSFER + 1) * 3 + 6 * (MIN_TRANSFER + 3) * (MIN_TRANSFER + 3);
	for (unsigned t = 0; t < asp.tasknum; ++t) {
		asp.workers.emplace_back([&] {
			while (true) {
				const int taskid = next_bid.fetch_add(1, std::memory_order_relaxed);
				if (taskid >= asp.tasknum) break;
				const int block_numx = asp.block_numx;
				const int block_numy = asp.block_numy;
				const int block_numz = asp.block_numz;
				const int bid = asp.blockid;
				const int tnum1 = (MIN_TRANSFER + 1) * (MIN_TRANSFER + 1);
				const int tnum2 = (MIN_TRANSFER + 3) * (MIN_TRANSFER + 3);
				const std::vector<int> offset = {
					bid % block_numx,
					(bid / block_numx) % block_numy,
					bid / (block_numx * block_numy)
				};
				const int ngsv = n_gsvertices();
				if (taskid < tnum1) {
					int i = 1 + taskid % (MIN_TRANSFER + 1);
					int j = 1 + taskid / (MIN_TRANSFER + 1);
					int pos[3] = { MIN_TRANSFER + 1, i, j};
					const int gsid_tar = bid * ngsv + lexi2gs(pos, gsVertexReso, gsVertexSetEnd, true);
					auto new_offset = offset;
					new_offset[0] = (new_offset[0] + 1) % block_numx;
					pos[0] = 1;
					if (i == MIN_TRANSFER + 1) {
						pos[1] = 1;
						new_offset[1] = (new_offset[1] + 1) % block_numy;
					}
					if (j == MIN_TRANSFER + 1) {
						pos[2] = 1;
						new_offset[2] = (new_offset[2] + 1) % block_numz;
					}
					const int src_bid = new_offset[0] + new_offset[1] * block_numx + new_offset[2] * block_numx * block_numy;
					const int gsid_src = src_bid * ngsv + lexi2gs(pos, gsVertexReso, gsVertexSetEnd, true);
					v[gsid_tar] = v[gsid_src];
				}
				else if (taskid < 2 * tnum1) {
					int id = taskid - tnum1;
					int i = 1 + id % (MIN_TRANSFER + 1);
					int j = 1 + id / (MIN_TRANSFER + 1);
					int pos[3] = { i, MIN_TRANSFER + 1, j };
					const int gsid_tar = bid * ngsv + lexi2gs(pos, gsVertexReso, gsVertexSetEnd, true);
					auto new_offset = offset;
					new_offset[1] = (new_offset[1] + 1) % block_numy;
					pos[1] = 1;
					if (i == MIN_TRANSFER + 1) {
						pos[0] = 1;
						new_offset[0] = (new_offset[0] + 1) % block_numx;
					}
					if (j == MIN_TRANSFER + 1) {
						pos[2] = 1;
						new_offset[2] = (new_offset[2] + 1) % block_numz;
					}
					const int src_bid = new_offset[0] + new_offset[1] * block_numx + new_offset[2] * block_numx * block_numy;
					const int gsid_src = src_bid * ngsv + lexi2gs(pos, gsVertexReso, gsVertexSetEnd, true);
					v[gsid_tar] = v[gsid_src];
				}
				else if (taskid < 3 * tnum1) {
					int id = taskid - 2 * tnum1;
					int i = 1 + id % (MIN_TRANSFER + 1);
					int j = 1 + id / (MIN_TRANSFER + 1);
					int pos[3] = { i, j, MIN_TRANSFER + 1 };
					const int gsid_tar = bid * ngsv + lexi2gs(pos, gsVertexReso, gsVertexSetEnd, true);
					auto new_offset = offset;
					new_offset[2] = (new_offset[2] + 1) % block_numz;
					pos[2] = 1;
					if (i == MIN_TRANSFER + 1) {
						pos[0] = 1;
						new_offset[0] = (new_offset[0] + 1) % block_numx;
					}
					if (j == MIN_TRANSFER + 1) {
						pos[1] = 1;
						new_offset[1] = (new_offset[1] + 1) % block_numy;
					}
					const int src_bid = new_offset[0] + new_offset[1] * block_numx + new_offset[2] * block_numx * block_numy;
					const int gsid_src = src_bid * ngsv + lexi2gs(pos, gsVertexReso, gsVertexSetEnd, true);
					v[gsid_tar] = v[gsid_src];
				}
				else if (taskid < 3 * tnum1 + tnum2) {
					int id = taskid - 3 * tnum1;
					int j = id % (MIN_TRANSFER + 3);
					int k = id / (MIN_TRANSFER + 3);
					int pos[3] = { 0, j, k };
					int gsid_tar = bid * ngsv + lexi2gs(pos, gsVertexReso, gsVertexSetEnd, true);
					auto offsetx = offset;
					offsetx[0] = (offsetx[0] - 1 + block_numx) % block_numx;
					pos[0] = MIN_TRANSFER;
					offsetx[1] = (j == 0) ?
						((offsetx[1] - 1 + block_numy) % block_numy) :
						(j == MIN_TRANSFER + 2) ? (offsetx[1] + 1) % block_numy : offsetx[1];
					pos[1] = (j == 0) ? MIN_TRANSFER : (j == MIN_TRANSFER + 2) ? 2 : j;
					offsetx[2] = (k == 0) ?
						((offsetx[2] - 1 + block_numz) % block_numz) :
						(k == MIN_TRANSFER + 2) ? (offsetx[2] + 1) % block_numz : offsetx[2];
					pos[2] = (k == 0) ? MIN_TRANSFER : (k == MIN_TRANSFER + 2) ? 2 : k;
					int src_bid = offsetx[0] + offsetx[1] * block_numx + offsetx[2] * block_numx * block_numy;
					int gsid_src = src_bid * ngsv + lexi2gs(pos, gsVertexReso, gsVertexSetEnd, true);
					v[gsid_tar] = v[gsid_src];
				}
				else if (taskid < 3 * tnum1 + 2 * tnum2) {
					int id = taskid - 3 * tnum1 - tnum2;
					int j = id % (MIN_TRANSFER + 3);
					int k = id / (MIN_TRANSFER + 3);
					int pos[3] = { MIN_TRANSFER + 2, j, k };
					int gsid_tar = bid * ngsv + lexi2gs(pos, gsVertexReso, gsVertexSetEnd, true);
					auto offsetx = offset;
					offsetx[0] = (offsetx[0] + 1) % block_numx;
					pos[0] = 2;
					offsetx[1] = (j == 0) ?
						((offsetx[1] - 1 + block_numy) % block_numy) :
						(j == MIN_TRANSFER + 2) ? (offsetx[1] + 1) % block_numy : offsetx[1];
					pos[1] = (j == 0) ? MIN_TRANSFER : (j == MIN_TRANSFER + 2) ? 2 : j;
					offsetx[2] = (k == 0) ?
						((offsetx[2] - 1 + block_numz) % block_numz) :
						(k == MIN_TRANSFER + 2) ? (offsetx[2] + 1) % block_numz : offsetx[2];
					pos[2] = (k == 0) ? MIN_TRANSFER : (k == MIN_TRANSFER + 2) ? 2 : k;
					int src_bid = offsetx[0] + offsetx[1] * block_numx + offsetx[2] * block_numx * block_numy;
					int gsid_src = src_bid * ngsv + lexi2gs(pos, gsVertexReso, gsVertexSetEnd, true);
					v[gsid_tar] = v[gsid_src];
				}
				else if (taskid < 3 * tnum1 + 3 * tnum2) {
					int id = taskid - 3 * tnum1 - 2 * tnum2;
					int i = id % (MIN_TRANSFER + 3);
					int k = id / (MIN_TRANSFER + 3);
					int pos[3] = { i, 0, k };
					int gsid_tar = bid * ngsv + lexi2gs(pos, gsVertexReso, gsVertexSetEnd, true);
					auto offsety = offset;
					offsety[1] = ((offsety[1] - 1 + block_numy) % block_numy);
					pos[1] = MIN_TRANSFER;
					offsety[0] = (i == 0) ?
						((offsety[0] - 1 + block_numx) % block_numx) :
						(i == MIN_TRANSFER + 2) ? (offsety[0] + 1) % block_numx : offsety[0];
					pos[0] = (i == 0) ? MIN_TRANSFER : (i == MIN_TRANSFER + 2) ? 2 : i;
					offsety[2] = (k == 0) ?
						((offsety[2] - 1 + block_numz) % block_numz) :
						(k == MIN_TRANSFER + 2) ? (offsety[2] + 1) % block_numz : offsety[2];
					pos[2] = (k == 0) ? MIN_TRANSFER : (k == MIN_TRANSFER + 2) ? 2 : k;
					const int src_bid = offsety[0] + offsety[1] * block_numx + offsety[2] * (block_numx * block_numy);
					const int gsid_src = src_bid * ngsv + lexi2gs(pos, gsVertexReso, gsVertexSetEnd, true);
					v[gsid_tar] = v[gsid_src];
				}
				else if (taskid < 3 * tnum1 + 4 * tnum2) {
					int id = taskid - 3 * tnum1 - 3 * tnum2;
					int i = id % (MIN_TRANSFER + 3);
					int k = id / (MIN_TRANSFER + 3);
					int pos[3] = { i, MIN_TRANSFER + 2, k };
					int gsid_tar = bid * ngsv + lexi2gs(pos, gsVertexReso, gsVertexSetEnd, true);
					auto offsety = offset;

					offsety[1] = (offsety[1] + 1) % block_numy;
					pos[1] = 2;

					offsety[0] = (i == 0) ?
						((offsety[0] - 1 + block_numx) % block_numx) :
						(i == MIN_TRANSFER + 2) ? (offsety[0] + 1) % block_numx : offsety[0];
					pos[0] = (i == 0) ? MIN_TRANSFER : (i == MIN_TRANSFER + 2) ? 2 : i;

					offsety[2] = (k == 0) ?
						((offsety[2] - 1 + block_numz) % block_numz) :
						(k == MIN_TRANSFER + 2) ? (offsety[2] + 1) % block_numz : offsety[2];
					pos[2] = (k == 0) ? MIN_TRANSFER : (k == MIN_TRANSFER + 2) ? 2 : k;

					const int src_bid = offsety[0] + offsety[1] * block_numx + offsety[2] * (block_numx * block_numy);
					const int gsid_src = src_bid * ngsv + lexi2gs(pos, gsVertexReso, gsVertexSetEnd, true);
					v[gsid_tar] = v[gsid_src];
				}
				else if (taskid < 3 * tnum1 + 5 * tnum2) {
					int id = taskid - 3 * tnum1 - 4 * tnum2;
					int i = id % (MIN_TRANSFER + 3);
					int j = id / (MIN_TRANSFER + 3);
					int pos[3] = { i, j, 0 };
					int gsid_tar = bid * ngsv + lexi2gs(pos, gsVertexReso, gsVertexSetEnd, true);
					auto offsetz = offset;

					offsetz[2] = ((offsetz[2] - 1 + block_numz) % block_numz);
					pos[2] = MIN_TRANSFER;

					offsetz[0] = (i == 0) ?
						((offsetz[0] - 1 + block_numx) % block_numx) :
						(i == MIN_TRANSFER + 2) ? (offsetz[0] + 1) % block_numx : offsetz[0];
					pos[0] = (i == 0) ? MIN_TRANSFER : (i == MIN_TRANSFER + 2) ? 2 : i;

					offsetz[1] = (j == 0) ?
						((offsetz[1] - 1 + block_numy) % block_numy) :
						(j == MIN_TRANSFER + 2) ? (offsetz[1] + 1) % block_numy : offsetz[1];
					pos[1] = (j == 0) ? MIN_TRANSFER : (j == MIN_TRANSFER + 2) ? 2 : j;

					const int src_bid = offsetz[0] + offsetz[1] * block_numx + offsetz[2] * (block_numx * block_numy);
					const int gsid_src = src_bid * ngsv + lexi2gs(pos, gsVertexReso, gsVertexSetEnd, true);
					v[gsid_tar] = v[gsid_src];
				}
				else if (taskid < 3 * tnum1 + 6 * tnum2) {
					int id = taskid - 3 * tnum1 - 5 * tnum2;
					int i = id % (MIN_TRANSFER + 3);
					int j = id / (MIN_TRANSFER + 3);
					int pos[3] = { i, j, MIN_TRANSFER + 2 };
					int gsid_tar = bid * ngsv + lexi2gs(pos, gsVertexReso, gsVertexSetEnd, true);
					auto offsetz = offset;

					offsetz[2] = (offsetz[2] + 1) % block_numz;
					pos[2] = 2;

					offsetz[0] = (i == 0) ?
						((offsetz[0] - 1 + block_numx) % block_numx) :
						(i == MIN_TRANSFER + 2) ? (offsetz[0] + 1) % block_numx : offsetz[0];
					pos[0] = (i == 0) ? MIN_TRANSFER : (i == MIN_TRANSFER + 2) ? 2 : i;

					offsetz[1] = (j == 0) ?
						((offsetz[1] - 1 + block_numy) % block_numy) :
						(j == MIN_TRANSFER + 2) ? (offsetz[1] + 1) % block_numy : offsetz[1];
					pos[1] = (j == 0) ? MIN_TRANSFER : (j == MIN_TRANSFER + 2) ? 2 : j;

					const int src_bid = offsetz[0] + offsetz[1] * block_numx + offsetz[2] * (block_numx * block_numy);
					const int gsid_src = src_bid * ngsv + lexi2gs(pos, gsVertexReso, gsVertexSetEnd, true);
					v[gsid_tar] = v[gsid_src];
				}
			}
			});
	}
}

void homo::Grid_H::joint_vertex_boundary_block() {
	for (auto& th : asp.workers) {
		if (th.joinable()) th.join();
	}
	asp.workers.clear();
}
void Grid_H::reset_displacement(void)
{
	v_reset(u_g[0]);
}

void Grid_H::reset_residual(void)
{
	v_reset(r_g[0]);
}

void Grid_H::reset_force(void)
{
	v_reset(f_g[0]);
}

void Grid_H::setUchar(int k, VT* uchar)
{
	v_download(uchar_h[k], uchar);
}

static Eigen::Matrix<double, -1, -1> transBase_H;
bool homo::Grid_H::solveHostEquation(void)
{
	Eigen::VectorXd b = v_toMatrix(f_g[0], true).cast<double>();

	// remove translation
	b = b - transBase_H * (transBase_H.transpose() * b);

	Eigen::Matrix<double, -1, 1> x = hostBiCGSolver.solve(b);
	if (hostBiCGSolver.info() != Eigen::Success) {
		printf("\033[31mhost equation failed to solve, code = %d\033[0m\n", int(hostBiCGSolver.info()));
		return false;
	}
	v_fromMatrix(u_g[0], x.cast<float>(), false);
	return true;
}

static Eigen::Matrix<double, -1, -1> transBase;

void homo::Grid_H::assembleHostMatrix(void)
{
	Khost = stencil2matrix();
	//eigen2ConnectedMatlab("Khost", Khost);
	hostBiCGSolver.compute(Khost);
	// init translation base
	transBase_H.resize(Khost.rows(), 6);
	transBase_H.fill(0);
	Eigen::Matrix<double, -1, -1> fk(Khost);
	Eigen::FullPivLU<Eigen::Matrix<double, -1, -1>> dec;
	dec.setThreshold(5e-2);
	dec.compute(fk);
	transBase_H = dec.kernel();
	for (int i = 0; i < transBase_H.cols(); i++) {
		for (int j = 0; j < i; j++) {
			transBase_H.col(i) -= transBase_H.col(i).dot(transBase_H.col(j)) * transBase_H.col(j);
		}
		transBase_H.col(i).normalize();
	}
	printf("Coarse system degenerate rank = %d\n", int(transBase_H.cols()));
}

float homo::Grid_H::diagPrecondition(float strength)
{
	diag_strength = strength;
	return strength;
}

void homo::Grid_H::useUchar(int k)
{
	v_upload(u_g[0], uchar_h[k]);
}

void homo::Grid_H::writeGsVertexPos(const std::string& fname)
{
	std::vector<int> pos[3];
	getGsVertexPos(pos);
	homoutils::writeVectors(fname, pos);
}
void homo::Grid_H::writeStencil() {
	if (assemb_otf == true) {
		printf("no stencil\n");
		return;
	}
	else {
		std::ofstream fout("stencil.txt");
		std::vector<VT> temple_vec(n_gsvertices(), 0);
		for (int i = 0; i < 27; i++) {
			cudaMemcpy(temple_vec.data(), stencil_g[i], sizeof(VT) * n_gsvertices(), cudaMemcpyDeviceToHost);
			for (const auto& x : temple_vec) {
				fout << x << " ";
			}
			fout << "\n";
		}
		fout.close();
	}
}

void homo::Grid_H::writeDensity(const std::string& fname, VoxelIOFormat frmat)
{
	std::vector<int> pos[3];
	getGsElementPos(pos);
	std::vector<float> rho;
	getDensity(rho);
	if (frmat == homo::binary) {
		std::ofstream ofs(fname, std::ios::binary);
		auto eidmap = getCellLexidMap();
		ofs.write((char*)cellReso.data(), sizeof(cellReso));
		for (int i = 0; i < eidmap.size(); i++) {
			float erho = rho[eidmap[i]];
			ofs.write((char*)&erho, sizeof(erho));
		}
		ofs.close();
	}
	else if (frmat == homo::openVDB) {
		std::vector<float> validrho;
		std::vector<int> validpos[3];
		for (int i = 0; i < rho.size(); i++) {
			if (pos[0][i] < 0 || pos[1][i] < 0 || pos[2][i] < 0 ||
				pos[0][i] >= cellReso[0] || pos[1][i] >= cellReso[1] || pos[2][i] >= cellReso[2])
				continue;
			validrho.emplace_back(rho[i]);
			validpos[0].emplace_back(pos[0][i]);
			validpos[1].emplace_back(pos[1][i]);
			validpos[2].emplace_back(pos[2][i]);
		}
		openvdb_wrapper_t<float>::grid2openVDBfile(fname, validpos, validrho);
	}
}

void homo::Grid_H::v_upload(VT* dev, VT* hst) 
{
	cudaMemcpy(dev, hst, sizeof(VT) * n_gsvertices(), cudaMemcpyHostToDevice);
}

void homo::Grid_H::v_download(VT* hst, VT* dev)
{
	cudaMemcpy(hst, dev, sizeof(VT) * n_gsvertices(), cudaMemcpyDeviceToHost);
}

void homo::Grid_H::v_write(const std::string& filename, VT* v, int len /*= -1*/)
{
	if (len == -1) len = n_gsvertices();
	std::vector<VT> arr;
	std::vector<float> arr_f32;
	arr.resize(len);
	arr_f32.resize(len);
	cudaMemcpy(arr.data(), v, sizeof(VT) * len, cudaMemcpyDeviceToHost);
	for (int j = 0; j < arr.size(); j++) {
		arr_f32[j] = arr[j];
	}
	std::vector<float> p_trans[1] = { arr_f32 };
	homoutils::writeVectors(filename, p_trans);
}

Eigen::Matrix<float, -1, 1> homo::Grid_H::v_toMatrix(VT* u, bool removePeriodDof /*= false*/)
{
	int nv;
	if (removePeriodDof) {
		nv = cellReso[0] * cellReso[1] * cellReso[2];
	}
	else {
		nv = (cellReso[0] + 1) * (cellReso[1] + 1) * (cellReso[2] + 1);
	}
	Eigen::Matrix<float, -1, 1> b(nv, 1);
	b.fill(0);
	std::vector<VT> vhost(n_gsvertices());
	std::vector<VertexFlags> vflags(n_gsvertices());
	cudaMemcpy(vflags.data(), vertflag, sizeof(VertexFlags) * n_gsvertices(), cudaMemcpyDeviceToHost);
	cudaMemcpy(vhost.data(), u, sizeof(VT) * n_gsvertices(), cudaMemcpyDeviceToHost);
	for (int k = 0; k < n_gsvertices(); k++) {
		if (vflags[k].is_fiction() || vflags[k].is_period_padding()) continue;
		//int pos[3];
		int id = vgsid2lexid_h(k, removePeriodDof);
		b[id] = vhost[k];
	}
	return b;
}

void homo::Grid_H::v_fromMatrix(VT* u, const Eigen::Matrix<float, -1, 1>& b, bool hasPeriodDof /*= false*/)
{
	std::vector<VT> ui(n_gsvertices());
	std::fill(ui.begin(), ui.end(), 0.);
	int nvlex;
	if (hasPeriodDof) {
		nvlex = (cellReso[0] + 1) * (cellReso[1] + 1) * (cellReso[2] + 1);
	}
	else {
		nvlex = cellReso[0] * cellReso[1] * cellReso[2];
	}
	for (int k = 0; k < nvlex; k++) {
		int gsid = vlexid2gsid(k, hasPeriodDof);
		ui[gsid] = b[k];
	}
	cudaMemcpy(u, ui.data(), sizeof(VT) * n_gsvertices(), cudaMemcpyHostToDevice);
	VT* tran[1] = { u };
	enforce_period_vertex(tran, false);
	pad_vertex_data(tran);
}

Eigen::SparseMatrix<double> homo::Grid_H::stencil2matrix(bool removePeriodDof /*= true*/)
{
	Eigen::SparseMatrix<double> K;
	int ndof;
	if (removePeriodDof) {
		ndof = cellReso[0] * cellReso[1] * cellReso[2];
	}
	else {
		ndof = (cellReso[0] + 1) * (cellReso[1] + 1) * (cellReso[2] + 1);
	}

	K.resize(ndof, ndof);
	std::vector<VT> kij(n_gsvertices());
	using trip = Eigen::Triplet<double>;
	std::vector<trip> trips;
	std::vector<VertexFlags> vflags(n_gsvertices());
	std::vector<CellFlags> eflags(n_gscells());
	cudaMemcpy(eflags.data(), cellflag, sizeof(CellFlags) * n_gscells(), cudaMemcpyDeviceToHost);
	cudaMemcpy(vflags.data(), vertflag, sizeof(VertexFlags) * n_gsvertices(), cudaMemcpyDeviceToHost);

	if (!is_root) {
		for (int i = 0; i < 27; i++) {
			int noff[3] = { i % 3 - 1, i / 3 % 3 - 1, i / 9 - 1 };
			cudaMemcpy(kij.data(), stencil_g[i], sizeof(VT) * n_gsvertices(), cudaMemcpyDeviceToHost);
			for (int k = 0; k < n_gsvertices(); k++) {
				if (vflags[k].is_fiction() || vflags[k].is_period_padding() /*|| vflags[k].is_max_boundary()*/) continue;
				//int gscolor = vflags[k].get_gscolor();
				int vpos[3];
				vgsid2lexpos_h(k, vpos);
				int oldvpos[3] = { vpos[0],vpos[1],vpos[2] };
				if (removePeriodDof) {
					if (vpos[0] >= cellReso[0] || vpos[1] >= cellReso[1] || vpos[2] >= cellReso[2]) continue;
				}
				else {
					if (vpos[0] >= cellReso[0] + 1 || vpos[1] >= cellReso[1] + 1 || vpos[2] >= cellReso[2] + 1) continue;
				}
				int vid = vlexpos2vlexid_h(vpos, removePeriodDof);
				vpos[0] += noff[0]; vpos[1] += noff[1]; vpos[2] += noff[2];
				if (removePeriodDof) {
					for (int kk = 0; kk < 3; kk++) { vpos[kk] = (vpos[kk] + cellReso[kk]) % cellReso[kk]; }
				}
				else {
					bool outBound = false;
					for (int kk = 0; kk < 3; kk++) { outBound = outBound || vpos[kk] < 0 || vpos[kk]>cellReso[kk]; }
					if (outBound) continue;
				}
				int neiid = vlexpos2vlexid_h(vpos, removePeriodDof);
				trips.emplace_back(vid, neiid, kij[k]);
			}
		}
	}
	else {
		using RhoType = std::remove_pointer_t<decltype(rho_g)>;
		std::vector<RhoType> rhohost(n_gscells());
		cudaMemcpy(rhohost.data(), rho_g, sizeof(RhoType) * n_gscells(), cudaMemcpyDeviceToHost);
		Eigen::Matrix<float, 8, 8> ke = getTemplateMatrix_H();
		for (int i = 0; i < eflags.size(); i++) {
			if (eflags[i].is_fiction() || eflags[i].is_period_padding()) continue;
			float rho_p = rhoPenalMin + powf(rhohost[i], 1);
			int epos[3];
			egsid2lexpos_h(i, epos);
			for (int vi = 0; vi < 8; vi++) {
				int vipos[3] = { epos[0] + vi % 2, epos[1] + vi / 2 % 2, epos[2] + vi / 4 };
				// todo check Dirichlet boundary
				int vi_id = vlexpos2vlexid_h(vipos, removePeriodDof);
				for (int vj = 0; vj < 8; vj++) {
					int vjpos[3] = { epos[0] + vj % 2, epos[1] + vj / 2 % 2, epos[2] + vj / 4 };
					// todo check Dirichlet boundary
					int vj_id = vlexpos2vlexid_h(vjpos, removePeriodDof);
					trips.emplace_back(vi_id, vj_id, ke(vi, vj) * rho_p);
				}
			}
		}
	}

	K.setFromTriplets(trips.begin(), trips.end());

	return K;
}

int homo::Grid_H::vgsid2lexid_h(int gsid, bool removePeriodDof /*= false*/)
{
	int lexpos[3];
	vgsid2lexpos_h(gsid, lexpos);
	int lexid = vlexpos2vlexid_h(lexpos, removePeriodDof);
	return lexid;
}

void homo::Grid_H::vgsid2lexpos_h(int gsid, int pos[3])
{
	int color = -1;
	for (int i = 0; i < 8; i++) {
		if (gsid < gsVertexSetEnd[i]) {
			color = i;
			break;
		}
	}
	if (color == -1) throw std::runtime_error("illegal gsid");
	int setid = color == 0 ? gsid : gsid - gsVertexSetEnd[color - 1];
	int gspos[3] = {
		setid % gsVertexReso[0][color],
		setid / gsVertexReso[0][color] % gsVertexReso[1][color],
		setid / (gsVertexReso[0][color] * gsVertexReso[1][color])
	};

	//printf("color = %d  setid = %d  gsvreso = (%d, %d, %d)  gsend = %d gspos = (%d, %d, %d)\n",
	//	color, setid, gsVertexReso[0][color], gsVertexReso[1][color], gsVertexReso[2][color],
	//	gsVertexSetEnd[color - 1], gspos[0], gspos[1], gspos[2]);

	int lexpos[3] = {
		gspos[0] * 2 + color % 2 - 1,
		gspos[1] * 2 + color / 2 % 2 - 1,
		gspos[2] * 2 + color / 4 - 1
	};

	//for (int i = 0; i < 3; i++) lexpos[i] = (lexpos[i] + cellReso[i]) % cellReso[i];

	for (int i = 0; i < 3; i++) pos[i] = lexpos[i];
}

void homo::Grid_H::egsid2lexpos_h(int gsid, int pos[3])
{
	int color = -1;
	for (int i = 0; i < 8; i++) {
		if (gsid < gsCellSetEnd[i]) {
			color = i;
			break;
		}
	}
	if (color == -1) throw std::runtime_error("illegal gsid");
	int setid = color == 0 ? gsid : gsid - gsCellSetEnd[color - 1];
	int gspos[3] = {
		setid % gsCellReso[0][color],
		setid / gsCellReso[0][color] % gsCellReso[1][color],
		setid / (gsCellReso[0][color] * gsCellReso[1][color])
	};
	int lexpos[3] = {
		gspos[0] * 2 + color % 2 - 1,
		gspos[1] * 2 + color / 2 % 2 - 1,
		gspos[2] * 2 + color / 4 - 1
	};

	//for (int i = 0; i < 3; i++) lexpos[i] = (lexpos[i] + cellReso[i]) % cellReso[i];

	for (int i = 0; i < 3; i++) pos[i] = lexpos[i];
}

int homo::Grid_H::vlexpos2vlexid_h(int lexpos[3], bool removePeriodDof/* = false*/)
{
	int vreso[3];
	for (int i = 0; i < 3; i++) lexpos[i] = (lexpos[i] + cellReso[i]) % cellReso[i];
	if (removePeriodDof) {
		for (int i = 0; i < 3; i++) {
			vreso[i] = cellReso[i];
		}
	}
	else {
		for (int i = 0; i < 3; i++) {
			vreso[i] = cellReso[i] + 1;
		}
	}

	for (int i = 0; i < 3; i++) {
		if (lexpos[i] < 0 || lexpos[i] >= vreso[i])
			throw std::runtime_error("illegal lexpos");
	}

	int lexid =
		lexpos[0] +
		lexpos[1] * vreso[0] +
		lexpos[2] * vreso[0] * vreso[1];

	return lexid;
}

int homo::Grid_H::vlexid2gsid(int lexid, bool hasPeriodDof /*= false*/)
{
	int pos[3];
	int vreso[3];
	if (hasPeriodDof) {
		for (int i = 0; i < 3; i++) vreso[i] = cellReso[i] + 1;
	}
	else {
		for (int i = 0; i < 3; i++) vreso[i] = cellReso[i];
	}
	pos[0] = lexid % vreso[0] + 1;
	pos[1] = lexid / vreso[0] % vreso[1] + 1;
	pos[2] = lexid / (vreso[0] * vreso[1]) + 1;
	int gspos[3] = { pos[0] / 2,pos[1] / 2,pos[2] / 2 };
	int color = pos[0] % 2 + pos[1] % 2 * 2 + pos[2] % 2 * 4;
	int setid = gspos[0] +
		gspos[1] * gsVertexReso[0][color] +
		gspos[2] * gsVertexReso[0][color] * gsVertexReso[1][color];
	int base = color == 0 ? 0 : gsVertexSetEnd[color - 1];
	int gsid = base + setid;
	return gsid;
}

int homo::Grid_H::elexid2gsid(int lexid) {
	int pos[3];
	int ereso[3];
	for (int i = 0; i < 3; i++) ereso[i] = cellReso[i];
	pos[0] = lexid % ereso[0] + 1;
	pos[1] = lexid / ereso[0] % ereso[1] + 1;
	pos[2] = lexid / (ereso[0] * ereso[1]) + 1;
	int gspos[3] = { pos[0] / 2,pos[1] / 2,pos[2] / 2 };
	int color = pos[0] % 2 + pos[1] % 2 * 2 + pos[2] % 2 * 4;
	int setid = gspos[0] +
		gspos[1] * gsCellReso[0][color] +
		gspos[2] * gsCellReso[0][color] * gsCellReso[1][color];
	int base = color == 0 ? 0 : gsCellSetEnd[color - 1];
	int gsid = base + setid;
	return gsid;
}

void homo::Grid_H::enforce_period_boundary(VT* v[1], bool additive /*= false*/)
{
	//if (additive) { throw std::runtime_error("additive should never be set"); }
	enforce_period_vertex(v, additive);
	pad_vertex_data(v);
}

void homo::Grid_H::translateForce(int type_, VT* v[1]) {
	VT t_f[1] = {0};
	if (type_ == 1) {
		int gsid = vlexid2gsid(0, true);
		cudaMemcpy(t_f, v[0] + gsid, sizeof(VT), cudaMemcpyDeviceToHost);
	}
	else if (type_ == 2) {
		v_average(v[0], t_f[0], true);
	}
	v_removeT(v[0], t_f);
}

template<typename T>
struct DeviceDataProxy {
	T* pdata;
	template<typename Q>
	operator Q(){
		T data;
		cudaMemcpy(&data, pdata, sizeof(T), cudaMemcpyDeviceToHost);
		return Q(data);
	}
	DeviceDataProxy& operator=(const DeviceDataProxy<T>& q) {
		cudaMemcpy(pdata, q.pdata, sizeof(T), cudaMemcpyDeviceToDevice);
		return *this;
	}
	template<typename Q>
	DeviceDataProxy& operator=(Q data) {
		T tdata = data;
		cudaMemcpy(pdata, &tdata, sizeof(T), cudaMemcpyHostToDevice);
		return *this;
	}
	DeviceDataProxy(T *p) : pdata(p) {}
};

template<typename T>
struct DevicePtr {
	T* pdata;
	DevicePtr(T *p) : pdata(p) {}
	DeviceDataProxy<T> operator[](size_t ind) {
		return DeviceDataProxy<T>(pdata + ind);
	}
	DeviceDataProxy<T> operator*() {
		return DeviceDataProxy<T>(pdata);
	}
	DevicePtr operator+(size_t step) {
		return DevicePtr<T>(pdata + step);
	}
	DevicePtr operator-(size_t step) {
		return DevicePtr<T>(pdata - step);
	}
};

void homo::Grid_H::restrict_stencil_arround_dirichelt_boundary(void) {
	if (!fine->is_root)
		return;
	auto KE = getTemplateMatrix_H();
	auto rholist = DevicePtr(fine->rho_g);
	auto finereso = fine->cellReso;
	std::map<std::array<int, 3>, float> pos2rho;
	// we get the 8 vertex of fine grid
	for (int xc_off = -2 * upCoarse[0]; xc_off < 2 * upCoarse[0]; xc_off++) {
		for (int yc_off = -2 * upCoarse[1]; yc_off < 2 * upCoarse[1]; yc_off++) {
			for (int zc_off = -2 * upCoarse[2]; zc_off < 2 * upCoarse[2]; zc_off++) {
				int epos[3] = {
					(xc_off + finereso[0]) % finereso[0],
					(yc_off + finereso[1]) % finereso[1],
					(zc_off + finereso[2]) % finereso[2]
				};
				int eid = epos[0] + epos[1] * finereso[0] + epos[2] * finereso[0] * finereso[1];
				float prho;
				if (!fine->use_host_memory) {
					int egsid = fine->elexid2gsid(eid);
					prho = rholist[egsid];
				}
				else {
					int block_id = (epos[0] / MIN_TRANSFER) + (epos[1] / MIN_TRANSFER) * (finereso[0] / MIN_TRANSFER) +
						(epos[2] / MIN_TRANSFER) * (finereso[0] / MIN_TRANSFER * finereso[1] / MIN_TRANSFER);
					int pos_id = (epos[0] % MIN_TRANSFER + 1) + (epos[1] % MIN_TRANSFER + 1) * (MIN_TRANSFER + 2) + (epos[2] % MIN_TRANSFER + 1) * (MIN_TRANSFER+2) * (MIN_TRANSFER+2);
					prho = (*fine->rho_h)[block_id * pow(MIN_TRANSFER + 2, 3) + pos_id];
				}
				pos2rho[{xc_off, yc_off, zc_off}] = prho;
				// printf("e(%d, %d, %d) = %4.2e\n", xc_off, yc_off, zc_off, prho);
			}
		}
	}

	double ke[8][8];
	for (int ri = 0; ri < 8; ri++) {
		for (int ci = 0; ci < 8; ci++) {
			ke[ri][ci] = KE(ri, ci);
		}
	}

	double pr = (upCoarse[0] * upCoarse[1] * upCoarse[2]);
	for (int vi = 0; vi < 27; vi++)
	{
		Eigen::Vector3i vi_pos = {
			(vi % 3 - 1) * upCoarse[0],
			(vi / 3 % 3 - 1) * upCoarse[1],
			(vi / 9 - 1) * upCoarse[2] };
		double st[27] = {};
		for (int k = 0; k < 27; k++)
			st[k] = 0.;
		for (int vj = 0; vj < 27; vj++)
		{
			Eigen::Vector3i vj_pos_off_vi = {
				(vj % 3 - 1) * upCoarse[0],
				(vj / 3 % 3 - 1) * upCoarse[1],
				(vj / 9 - 1) * upCoarse[2] };
			Eigen::Vector3i vj_pos = vi_pos + vj_pos_off_vi;
			for (int e_off_vj_x = -upCoarse[0]; e_off_vj_x < upCoarse[0]; e_off_vj_x++)
			{
				for (int e_off_vj_y = -upCoarse[1]; e_off_vj_y < upCoarse[1]; e_off_vj_y++)
				{
					for (int e_off_vj_z = -upCoarse[2]; e_off_vj_z < upCoarse[2]; e_off_vj_z++)
					{
						Eigen::Vector3i epos = vj_pos + Eigen::Vector3i(e_off_vj_x, e_off_vj_y, e_off_vj_z);
						Eigen::Vector3i e_off_vi = epos - vi_pos;
						if (e_off_vi[0] >= upCoarse[0] || e_off_vi[0] < -upCoarse[0] ||
							e_off_vi[1] >= upCoarse[1] || e_off_vi[1] < -upCoarse[1] ||
							e_off_vi[2] >= upCoarse[2] || e_off_vi[2] < -upCoarse[2])
						{
							continue;
						}
						double prho = pos2rho[{epos[0], epos[1], epos[2]}];
						for (int e_vi = 0; e_vi < 8; e_vi++)
						{
							Eigen::Vector3i e_vi_pos = epos + Eigen::Vector3i(e_vi % 2, e_vi / 2 % 2, e_vi / 4);
							Eigen::Vector3i e_vi_off = e_vi_pos - vi_pos;
							// ToDo : this causes errors if resolution is too small
							bool e_vi_d = e_vi_pos[0] == 0 && e_vi_pos[1] == 0 && e_vi_pos[2] == 0;
							if (abs(e_vi_off[0]) >= upCoarse[0] ||
								abs(e_vi_off[1]) >= upCoarse[1] ||
								abs(e_vi_off[2]) >= upCoarse[2])
								continue;
							double wi =
								(upCoarse[0] - abs(e_vi_off[0])) *
								(upCoarse[1] - abs(e_vi_off[1])) *
								(upCoarse[2] - abs(e_vi_off[2])) / pr;
							for (int e_vj = 0; e_vj < 8; e_vj++)
							{
								Eigen::Vector3i e_vj_pos = epos + Eigen::Vector3i(e_vj % 2, e_vj / 2 % 2, e_vj / 4);
								Eigen::Vector3i e_vj_off = e_vj_pos - vj_pos;
								bool e_vj_d = e_vj_pos[0] == 0 && e_vj_pos[1] == 0 && e_vj_pos[2] == 0;
								if (abs(e_vj_off[0]) >= upCoarse[0] ||
									abs(e_vj_off[1]) >= upCoarse[1] ||
									abs(e_vj_off[2]) >= upCoarse[2])
									continue;
								double wj =
									(upCoarse[0] - abs(e_vj_off[0])) *
									(upCoarse[1] - abs(e_vj_off[1])) *
									(upCoarse[2] - abs(e_vj_off[2])) / pr;
								if (e_vi_d || e_vj_d)
								{
									if (e_vi == e_vj)
										st[vj] += wi * wj;
								}
								else {
									st[vj] += wi * wj * ke[e_vi][e_vj] * prho;
								}
							}
						}
					}
				}
			}
		}

		// clamp for fp16 number
		if (abs(st[13]) < 1e-4) {
			st[13] = 1e-4;
		}

		int vi_pos_period[3] = {
			(vi % 3 - 1 + cellReso[0]) % cellReso[0],
			(vi / 3 % 3 - 1 + cellReso[1]) % cellReso[1],
			(vi / 9 - 1 + cellReso[2]) % cellReso[2] };

		int vop[3];
		for (int px = 0; px < 1 + (vi_pos_period[0] == 0); px++) {
			vop[0] = px ? cellReso[0] : vi_pos_period[0];
			for (int py = 0; py < 1 + (vi_pos_period[1] == 0); py++) {
				vop[1] = py ? cellReso[1] : vi_pos_period[1];
				for (int pz = 0; pz < 1 + (vi_pos_period[2] == 0); pz++) {
					vop[2] = pz ? cellReso[2] : vi_pos_period[2];
					int vi_period_id = vlexid2gsid(
						vop[0] +
						vop[1] * (cellReso[0] + 1) +
						vop[2] * (cellReso[0] + 1) * (cellReso[1] + 1),
						true);
					for (int i = 0; i < 27; i++) {
						auto sten = DevicePtr(stencil_g[i]);
						sten[vi_period_id] = st[i];
					}
				}
			}
		}
	}
}
