#include "grid.h"
//#define __CUDACC__
#include "culib/lib.cuh"
#include <vector>
#include "utils.h"
#include <fstream>
#include "homoCommon.cuh"

#define USING_SOR  1
#define DIRICHLET_STRENGTH 1e3

#define USE_LAME_MATRIX 1

//#define DIAG_STRENGTH 1e6 
//#define DIAG_STRENGTH 0 

using namespace homo;
using namespace culib;
__constant__ VT* gU_H[1];
__constant__ VT* gF_H[1];
__constant__ VT* gR_H[1];
__constant__ VT* gUfine_H[1];
__constant__ VT* gFfine_H[1];
__constant__ VT* gRfine_H[1];
__constant__ VT* gUcoarse_H[1];
__constant__ VT* gFcoarse_H[1];
__constant__ VT* gRcoarse_H[1];
__constant__ float gKE_H[8][8];
__constant__ double gKEd_H[8][8];
__constant__ float gfeMu_H[8][3];
__constant__ float gDisp_H[8][3];
__constant__ VT* rxstencil_H[27];
__constant__ VT* rxCoarseStencil_H[27];
__constant__ VT* rxFineStencil_H[27];

__constant__ int gUpCoarse[3];
__constant__ int gDownCoarse[3];
__constant__ int gGridCellReso[3];
__constant__ int gCoarseGridCellReso[3];
__constant__ int gGsCellReso[3][8];
__constant__ int gGsVertexReso[3][8];
__constant__ int gGsVertexEnd[8];
__constant__ int gGsCellEnd[8];
__constant__ int gGsFineVertexReso[3][8];
__constant__ int gGsCoarseVertexReso[3][8];
__constant__ int gGsFineCellReso[3][8];
__constant__ int gGsCoarseCellReso[3][8];
__constant__ int gGsCoarseVertexEnd[8];
__constant__ int gGsFineVertexEnd[8];
__constant__ int gGsFineCellEnd[8];

//__constant__ double* guchar[6][3];
//__constant__ double* gfchar[6][3];
__constant__ float* guchar[3];

__constant__ float exp_penal[1];


template<typename T>
__global__ void restrict_stencil_otf_aos_kernel_1_H(
	int ne, T* rholist, CellFlags* eflags, VertexFlags* vflags
);
template<typename T>
__global__ void restrict_stencil_otf_aos_kernel_alter_H(
	int ne, T* rholist, CellFlags* eflags, VertexFlags* vflags
);
template<typename T>
__global__ void restrict_stencil_otf_aos_kernel_host_H(
	int ne, T* rholist, CellFlags* eflags, VertexFlags* vflags, int bx, int by, int bz, int blocksize
);

__global__ void restrict_stencil_aos_kernel_1_H(
	int nv_coarse, int nv_fine,
	VertexFlags* vflags,
	VertexFlags* vfineflags
);
__device__ void gsid2pos(int gsid, int color, int gsreso[3][8], int gsend[8], int pos[3]) {
	int setid = gsid - (color == 0 ? 0 : gsend[color - 1]);
	int gsorg[3] = { color % 2, color / 2 % 2, color / 4 };
	int gspos[3] = { setid % gsreso[0][color], setid / gsreso[0][color] % gsreso[1][color], setid / (gsreso[0][color] * gsreso[1][color]) };
	for (int i = 0; i < 3; i++) {
		pos[i] = gspos[i] * 2 + gsorg[i] - 1;
	}
}
void homo::Grid_H::useGrid_g(void)
{
	cudaMemcpyToSymbol(gU_H, &u_g[0], sizeof(gU_H));
	cudaMemcpyToSymbol(gF_H, &f_g[0], sizeof(gF_H));
	cudaMemcpyToSymbol(gR_H, &r_g[0], sizeof(gR_H));
	if (fine != nullptr) {
		cudaMemcpyToSymbol(gUfine_H, fine->u_g, sizeof(gUfine_H));
		cudaMemcpyToSymbol(gFfine_H, fine->f_g, sizeof(gFfine_H));
		cudaMemcpyToSymbol(gRfine_H, fine->r_g, sizeof(gRfine_H));

		cudaMemcpyToSymbol(rxFineStencil_H, fine->stencil_g, sizeof(rxFineStencil_H));
		cudaMemcpyToSymbol(gUpCoarse, upCoarse.data(), sizeof(gUpCoarse));
		cudaMemcpyToSymbol(gGsFineVertexEnd, fine->gsVertexSetEnd, sizeof(gGsFineVertexEnd));
		cudaMemcpyToSymbol(gGsFineCellEnd, fine->gsCellSetEnd, sizeof(gGsFineCellEnd));
		cudaMemcpyToSymbol(gGsFineVertexReso, fine->gsVertexReso, sizeof(gGsFineVertexReso));
		cudaMemcpyToSymbol(gGsFineCellReso, fine->gsCellReso, sizeof(gGsFineCellReso));

	}
	if (Coarse != nullptr) {
		cudaMemcpyToSymbol(gUcoarse_H, Coarse->u_g, sizeof(gUcoarse_H));
		cudaMemcpyToSymbol(gFcoarse_H, Coarse->f_g, sizeof(gFcoarse_H));
		cudaMemcpyToSymbol(gRcoarse_H, Coarse->r_g, sizeof(gRcoarse_H));

		cudaMemcpyToSymbol(rxCoarseStencil_H, Coarse->stencil_g, sizeof(rxCoarseStencil_H));
		cudaMemcpyToSymbol(gDownCoarse, downCoarse.data(), sizeof(gDownCoarse));
		cudaMemcpyToSymbol(gGsCoarseVertexEnd, Coarse->gsVertexSetEnd, sizeof(gGsCoarseVertexEnd));
		cudaMemcpyToSymbol(gGsCoarseVertexReso, Coarse->gsVertexReso, sizeof(gGsCoarseVertexReso));
		cudaMemcpyToSymbol(gCoarseGridCellReso, Coarse->cellReso.data(), sizeof(gCoarseGridCellReso));
		cudaMemcpyToSymbol(gGsCoarseCellReso, Coarse->gsCellReso, sizeof(gGsCoarseCellReso));
	}
	cudaMemcpyToSymbol(rxstencil_H, stencil_g, sizeof(rxstencil_H));
	cudaMemcpyToSymbol(gGridCellReso, cellReso.data(), sizeof(gGridCellReso));
	cudaMemcpyToSymbol(gGsVertexEnd, gsVertexSetEnd, sizeof(gGsVertexEnd));
	cudaMemcpyToSymbol(gGsCellEnd, gsCellSetEnd, sizeof(gGsCellEnd));
	cudaMemcpyToSymbol(gGsVertexReso, gsVertexReso, sizeof(gGsVertexReso));

	if (is_root) {
		cudaMemcpyToSymbol(gGsCellReso, gsCellReso, sizeof(gGsCellReso));
	}
}
void homo::Grid_H::useCurrent_g(void) {
	cudaMemcpyToSymbolAsync(gU_H, &u_g[current], sizeof(gU_H), 0, cudaMemcpyDefault, stream[current]);
	cudaMemcpyToSymbolAsync(gF_H, &f_g[current], sizeof(gF_H), 0, cudaMemcpyDefault, stream[current]);
	cudaMemcpyToSymbolAsync(gR_H, &r_g[current], sizeof(gR_H), 0, cudaMemcpyDefault, stream[current]);
}
void homo::Grid_H::useNext_g(void) {
	cudaMemcpyToSymbolAsync(gU_H, &u_g[next], sizeof(gU_H), 0, cudaMemcpyDefault, stream[next]);
	cudaMemcpyToSymbolAsync(gF_H, &f_g[next], sizeof(gF_H), 0, cudaMemcpyDefault, stream[next]);
	cudaMemcpyToSymbolAsync(gR_H, &r_g[next], sizeof(gR_H), 0, cudaMemcpyDefault, stream[next]);
}
__global__ void setVertexFlags_kernel(
	int nv, VertexFlags* pflag,
	devArray_t<int, 3> cellReso,
	devArray_t<int, 8> vGSend, devArray_t<int, 8> vGSvalid
) {
	size_t vid = blockIdx.x * blockDim.x + threadIdx.x;
	if (vid >= nv) return;

	VertexFlags flag = pflag[vid];

	int set_id = -1;
	for (int i = 0; i < 8; i++) {
		if (vid < vGSend[i]) {
			set_id = i;
			break;
		}
	}

	if (set_id == -1) {
		flag.set_fiction();
		pflag[vid] = flag;
		return;
	}

	flag.set_gscolor(set_id);

	int gsid = vid - (set_id >= 1 ? vGSend[set_id - 1] : 0);
	do {
		// check if GS padding
		if (gsid >= vGSvalid[set_id]) {
			flag.set_fiction();
			break;
		}

		// check if periodic boundary padding
		int org[3] = { set_id % 2, set_id / 2 % 2, set_id / 4 };
		int gsvreso[3] = {};
		for (int i = 0; i < 3; i++)
			gsvreso[i] = (cellReso[i] + 2 - org[i]) / 2 + 1;

		int gspos[3] = { gsid % gsvreso[0], gsid / gsvreso[0] % gsvreso[1], gsid / (gsvreso[0] * gsvreso[1]) };
		int pos[3] = { gspos[0] * 2 + org[0], gspos[1] * 2 + org[1], gspos[2] * 2 + org[2] };

		// check if dirichlet boundary
		if ((pos[0] - 1) % cellReso[0] == 0 && 
			(pos[1] - 1) % cellReso[1] == 0 &&
			(pos[2] - 1) % cellReso[2] == 0) {
			printf("pos:(%f,%f,%f)\n", pos[0], pos[1], pos[2]);
			flag.set_dirichlet_boundary();
		}

		// is left padding
		if (pos[0] == 0 || pos[1] == 0 || pos[2] == 0) {
			//flag.set_fiction();
			flag.set_period_padding();
		}

		// is right padding
		if (pos[0] == cellReso[0] + 2 || pos[1] == cellReso[1] + 2 || pos[2] == cellReso[2] + 2) {
			//flag.set_fiction();
			flag.set_period_padding();
		}

		// is boundary
		if (pos[0] == 1) { flag.set_boundary(LEFT_BOUNDARY); }
		if (pos[1] == 1) { flag.set_boundary(NEAR_BOUNDARY); }
		if (pos[2] == 1) { flag.set_boundary(DOWN_BOUNDARY); }
		if (pos[0] == cellReso[0] + 1) { flag.set_boundary(RIGHT_BOUNDARY); }
		if (pos[1] == cellReso[1] + 1) { flag.set_boundary(FAR_BOUNDARY); }
		if (pos[2] == cellReso[2] + 1) { flag.set_boundary(UP_BOUNDARY); }
	} while (0);

	pflag[vid] = flag;
}

__global__ void setCellFlags_kernel(
	int nc, CellFlags* pflag,
	devArray_t<int, 3> cellReso,
	devArray_t<int, 8> vGSend, devArray_t<int, 8> vGSvalid
) {
	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= nc) return;

	CellFlags flag = pflag[tid];

	int set_id = -1;
	for (int i = 0; i < 8; i++) {
		if (tid < vGSend[i]) {
			set_id = i;
			break;
		}
	}

	if (set_id == -1) return;

	flag.set_gscolor(set_id);

	int gsid = tid - (set_id - 1 >= 0 ? vGSend[set_id - 1] : 0);

	do {
		// check if GS padding
		if (gsid >= vGSvalid[set_id]) {
			flag.set_fiction();
			break;
		}

		// check if periodic boundary padding
		int org[3] = { set_id % 2, set_id / 2 % 2, set_id / 4 };
		int gsreso[3] = {};
		for (int i = 0; i < 3; i++)
			gsreso[i] = (cellReso[i] + 1 - org[i]) / 2 + 1;

		int gspos[3] = { gsid % gsreso[0], gsid / gsreso[0] % gsreso[1], gsid / (gsreso[0] * gsreso[1]) };
		int pos[3] = { gspos[0] * 2 + org[0], gspos[1] * 2 + org[1], gspos[2] * 2 + org[2] };

		// check if dirichlet boundary
		if (pos[0] == 1 && pos[1] == 1 && pos[2] == 1) {
			flag.set_dirichlet_boundary();
		}

		// is left padding
		if (pos[0] == 0 || pos[1] == 0 || pos[2] == 0) {
			//flag.set_fiction();
			//if ((pos[0] == 0) + (pos[1] == 0) + (pos[2] == 0) == 1) {
				flag.set_period_padding();
			//}
		}

		// is right padding
		if (pos[0] == cellReso[0] + 1 || pos[1] == cellReso[1] + 1 || pos[2] == cellReso[2] + 1) {
			//flag.set_fiction();
			//if ((pos[0] == cellReso[0] + 1) + (pos[1] == cellReso[1] + 1) + (pos[2] == cellReso[1] + 1) == 1) {
				flag.set_period_padding();
			//}
		}

		// is boundary
		if (pos[0] == 1) { flag.set_boundary(LEFT_BOUNDARY); }
		if (pos[1] == 1) { flag.set_boundary(NEAR_BOUNDARY); }
		if (pos[2] == 1) { flag.set_boundary(DOWN_BOUNDARY); }
		if (pos[0] == cellReso[0]) { flag.set_boundary(RIGHT_BOUNDARY); }
		if (pos[1] == cellReso[1]) { flag.set_boundary(FAR_BOUNDARY); }
		if (pos[2] == cellReso[2]) { flag.set_boundary(UP_BOUNDARY); }
	} while (0);

	pflag[tid] = flag;
}
void homo::Grid_H::setFlags_g(void)
{
	VertexFlags* vflag = vertflag;
	CellFlags* eflag = cellflag;
	cudaMemset(vflag, 0, sizeof(VertexFlags) * n_gsvertices());
	cudaMemset(eflag, 0, sizeof(CellFlags) * n_gscells());
	cuda_error_check;
	devArray_t<int, 3> ereso;
	if (!use_host_memory) {
		for (auto i : { 0, 1, 2 }) {
			ereso[i] = cellReso[i];
		}
	}
	else {
		for (auto i : { 0, 1, 2 }) {
			ereso[i] = MIN_TRANSFER;
		}
	}
	devArray_t<int, 8> vGsend, vGsvalid;
	devArray_t<int, 8> eGsend, eGsvalid;
	for (int i = 0; i < 8; i++) {
		vGsend[i] = gsVertexSetEnd[i];
		vGsvalid[i] = gsVertexSetValid[i];

		eGsend[i] = gsCellSetEnd[i];
		eGsvalid[i] = gsCellSetValid[i];
	}
	size_t grid_size, block_size;
	// set vertex flags
	make_kernel_param(&grid_size, &block_size, n_gsvertices(), 512);
	setVertexFlags_kernel << <grid_size, block_size >> > (n_gsvertices(), vflag, ereso, vGsend, vGsvalid);
	cudaDeviceSynchronize();
	cuda_error_check;

	// set cell flags
	make_kernel_param(&grid_size, &block_size, n_gscells(), 512);
	setCellFlags_kernel << <grid_size, block_size >> > (n_gscells(), eflag, ereso, eGsend, eGsvalid);
	cudaDeviceSynchronize();
	cuda_error_check;
}
void homo::Grid_H::use_block_rho(int blockid) {
	int blockx, blocky, blockz;
	blockx = cellReso[0] / MIN_TRANSFER;
	blocky = cellReso[1] / MIN_TRANSFER;
	blockz = cellReso[2] / MIN_TRANSFER;

	VT* tmp;
	tmp = getMem().getBuffer("temp_rho0")->data<VT>();
	int bx = blockid % blockx;
	int by = (blockid / blockx) % blocky;
	int bz = blockid / (blockx * blocky);

	int offset = (bx + blockx * by + blockx * blocky * bz) * pow(MIN_TRANSFER + 2, 3);
	// set a temp rho for data
	cudaMemcpy(tmp, rho_h->data() + offset, pow(MIN_TRANSFER + 2, 3) * sizeof(VT), cudaMemcpyHostToDevice);
	cuda_error_check;
}
void homo::Grid_H::use_block_rhogs(int blockid) {
	int blockx, blocky, blockz;
	blockx = cellReso[0] / MIN_TRANSFER;
	blocky = cellReso[1] / MIN_TRANSFER;
	blockz = cellReso[2] / MIN_TRANSFER;

	VT* tmp = getMem().getBuffer(
		(blockid == 0) ?
		(current ? "temp_rho0" : "temp_rho1") :
		(current ? "temp_rho1" : "temp_rho0")
	)->data<VT>();
	int bx = blockid % blockx;
	int by = (blockid / blockx) % blocky;
	int bz = blockid / (blockx * blocky);

	int offset = (bx + blockx * by + blockx * blocky * bz) * pow(MIN_TRANSFER + 2, 3);
	// set a temp rho for data
	if (blockid == 0) {
		cudaMemcpyAsync(tmp, rho_h->data() + offset, pow(MIN_TRANSFER + 2, 3) * sizeof(VT), cudaMemcpyHostToDevice, stream[current]);
	}
	else {
		cudaMemcpyAsync(tmp, rho_h->data() + offset, pow(MIN_TRANSFER + 2, 3) * sizeof(VT), cudaMemcpyHostToDevice, stream[next]);
	}
	cuda_error_check;
}
void homo::Grid_H::use_block_u_g(int blockid) {
	int blockx, blocky, blockz;
	blockx = cellReso[0] / MIN_TRANSFER;
	blocky = cellReso[1] / MIN_TRANSFER;
	int bx = blockid % blockx;
	int by = (blockid / blockx) % blocky;
	int bz = blockid / (blockx * blocky);
	int offset = (bx + blockx * by + blockx * blocky * bz) * n_gsvertices();
	cudaMemcpy(u_g[0], u_h.data() + offset, n_gsvertices() * sizeof(VT), cudaMemcpyHostToDevice);
}
void homo::Grid_H::use_block_u_ggs(int blockid) {
	int blockx, blocky, blockz;
	blockx = cellReso[0] / MIN_TRANSFER;
	blocky = cellReso[1] / MIN_TRANSFER;
	int bx = blockid % blockx;
	int by = (blockid / blockx) % blocky;
	int bz = blockid / (blockx * blocky);
	int offset = (bx + blockx * by + blockx * blocky * bz) * n_gsvertices();
	if (blockid == 0) {
		cudaMemcpyAsync(u_g[current], u_h.data() + offset, n_gsvertices() * sizeof(VT), cudaMemcpyHostToDevice, stream[current]);
	}
	else {
		cudaMemcpyAsync(u_g[next], u_h.data() + offset, n_gsvertices() * sizeof(VT), cudaMemcpyHostToDevice, stream[next]);
	}
}

void homo::Grid_H::write_block_u_g(int blockid) {
	int blockx, blocky, blockz;
	blockx = cellReso[0] / MIN_TRANSFER;
	blocky = cellReso[1] / MIN_TRANSFER;
	int bx = blockid % blockx;
	int by = (blockid / blockx) % blocky;
	int bz = blockid / (blockx * blocky);
	size_t offset = (bx + by * blockx + bz * (blockx * blocky)) * n_gsvertices();
	cudaMemcpy(u_h.data() + offset, u_g[0], n_gsvertices() * sizeof(VT), cudaMemcpyDeviceToHost);
}
void homo::Grid_H::write_block_u_ggs(int blockid, bool iscurrent) {
	int blockx, blocky, blockz;
	blockx = cellReso[0] / MIN_TRANSFER;
	blocky = cellReso[1] / MIN_TRANSFER;
	int bx = blockid % blockx;
	int by = (blockid / blockx) % blocky;
	int bz = blockid / (blockx * blocky);
	size_t offset = (bx + by * blockx + bz * (blockx * blocky)) * n_gsvertices();
	if (iscurrent) {
		cudaMemcpyAsync(u_h.data() + offset, u_g[next], n_gsvertices() * sizeof(VT), cudaMemcpyDeviceToHost, stream[next]);
	}
	else {
		cudaMemcpyAsync(u_h.data() + offset, u_g[current], n_gsvertices() * sizeof(VT), cudaMemcpyDeviceToHost, stream[current]);
	}
}

void homo::Grid_H::use_block_r_g(int blockid) {
	int blockx, blocky, blockz;
	blockx = cellReso[0] / MIN_TRANSFER;
	blocky = cellReso[1] / MIN_TRANSFER;
	int bx = blockid % blockx;
	int by = (blockid / blockx) % blocky;
	int bz = blockid / (blockx * blocky);
	int offset = (bx + blockx * by + blockx * blocky * bz) * n_gsvertices();
	cudaMemcpy(r_g[0], r_h.data() + offset, n_gsvertices() * sizeof(VT), cudaMemcpyHostToDevice);
}
void homo::Grid_H::use_block_r_ggs(int blockid) {
	int blockx, blocky, blockz;
	blockx = cellReso[0] / MIN_TRANSFER;
	blocky = cellReso[1] / MIN_TRANSFER;
	int bx = blockid % blockx;
	int by = (blockid / blockx) % blocky;
	int bz = blockid / (blockx * blocky);
	int offset = (bx + blockx * by + blockx * blocky * bz) * n_gsvertices();
	if (blockid == 0) {
		cudaMemcpyAsync(r_g[current], r_h.data() + offset, n_gsvertices() * sizeof(VT), cudaMemcpyHostToDevice, stream[current]);
	}
	else {
		cudaMemcpyAsync(r_g[next], r_h.data() + offset, n_gsvertices() * sizeof(VT), cudaMemcpyHostToDevice, stream[next]);
	}
}

void homo::Grid_H::use_block_f_g(int blockid) {
	int blockx, blocky, blockz;
	blockx = cellReso[0] / MIN_TRANSFER;
	blocky = cellReso[1] / MIN_TRANSFER;
	int bx = blockid % blockx;
	int by = (blockid / blockx) % blocky;
	int bz = blockid / (blockx * blocky);
	int offset = (bx + blockx * by + blockx * blocky * bz) * n_gsvertices();
	cudaMemcpy(f_g[0], f_h.data() + offset, n_gsvertices() * sizeof(VT), cudaMemcpyHostToDevice);
	cuda_error_check;
}
void homo::Grid_H::use_block_f_ggs(int blockid) {
	int blockx, blocky, blockz;
	blockx = cellReso[0] / MIN_TRANSFER;
	blocky = cellReso[1] / MIN_TRANSFER;
	int bx = blockid % blockx;
	int by = (blockid / blockx) % blocky;
	int bz = blockid / (blockx * blocky);
	int offset = (bx + blockx * by + blockx * blocky * bz) * n_gsvertices();
	if (blockid == 0) {
		cudaMemcpyAsync(f_g[current], f_h.data() + offset, n_gsvertices() * sizeof(VT), cudaMemcpyHostToDevice, stream[current]);
	}
	else {
		cudaMemcpyAsync(f_g[next], f_h.data() + offset, n_gsvertices() * sizeof(VT), cudaMemcpyHostToDevice, stream[next]);
	}
	cuda_error_check;
}

void homo::Grid_H::write_block_f_g(int blockid) {
	int blockx, blocky, blockz;
	blockx = cellReso[0] / MIN_TRANSFER;
	blocky = cellReso[1] / MIN_TRANSFER;

	int bx = blockid % blockx;
	int by = (blockid / blockx) % blocky;
	int bz = blockid / (blockx * blocky);
	size_t offset = (bx + by * blockx + bz * (blockx * blocky)) * n_gsvertices();
	cudaMemcpy(f_h.data() + offset, f_g[0], n_gsvertices() * sizeof(VT), cudaMemcpyDeviceToHost);
}
void homo::Grid_H::write_block_f_ggs(int blockid, bool iscurrent) {
	int blockx, blocky, blockz;
	blockx = cellReso[0] / MIN_TRANSFER;
	blocky = cellReso[1] / MIN_TRANSFER;

	int bx = blockid % blockx;
	int by = (blockid / blockx) % blocky;
	int bz = blockid / (blockx * blocky);
	size_t offset = (bx + by * blockx + bz * (blockx * blocky)) * n_gsvertices();
	if (iscurrent) {
		cudaMemcpyAsync(f_h.data() + offset, f_g[next], n_gsvertices() * sizeof(VT), cudaMemcpyDeviceToHost, stream[next]);
	}
	else {
		cudaMemcpyAsync(f_h.data() + offset, f_g[current], n_gsvertices() * sizeof(VT), cudaMemcpyDeviceToHost, stream[current]);
	}
}
void homo::Grid_H::write_block_r_g(int blockid) {
	int blockx, blocky, blockz;
	blockx = cellReso[0] / MIN_TRANSFER;
	blocky = cellReso[1] / MIN_TRANSFER;

	int bx = blockid % blockx;
	int by = (blockid / blockx) % blocky;
	int bz = blockid / (blockx * blocky);
	size_t offset = (bx + by * blockx + bz * (blockx * blocky)) * n_gsvertices();
	cudaMemcpy(r_h.data() + offset, r_g[0], n_gsvertices() * sizeof(VT), cudaMemcpyDeviceToHost);
}
void homo::Grid_H::write_block_r_ggs(int blockid, bool iscurrent) {
	int blockx, blocky, blockz;
	blockx = cellReso[0] / MIN_TRANSFER;
	blocky = cellReso[1] / MIN_TRANSFER;

	int bx = blockid % blockx;
	int by = (blockid / blockx) % blocky;
	int bz = blockid / (blockx * blocky);
	size_t offset = (bx + by * blockx + bz * (blockx * blocky)) * n_gsvertices();
	if (iscurrent) {
		cudaMemcpyAsync(r_h.data() + offset, r_g[next], n_gsvertices() * sizeof(VT), cudaMemcpyDeviceToHost, stream[next]);
	}
	else {
		cudaMemcpyAsync(r_h.data() + offset, r_g[current], n_gsvertices() * sizeof(VT), cudaMemcpyDeviceToHost, stream[current]);
	}
}
// map 32 vertices to 8 warp
template<typename T, int BlockSize = 32 * 8>
__global__ void gs_relaxation_otf_kernel_H(
	int gs_set, T* rholist,
	devArray_t<int, 3> gridCellReso,
	VertexFlags* vflags, CellFlags* eflags,
	// SOR relaxing factor
	float w = 1.f,
	float diag_strength = 0
) {

	__shared__ int gsCellReso[3][8];
	__shared__ int gsVertexReso[3][8];
	__shared__ int gsCellEnd[8];
	__shared__ int gsVertexEnd[8];

	__shared__ float KE[8][8];

	__shared__ float sumKeU[4][32];
	__shared__ float sumKs[4][32];

	initSharedMem(&sumKeU[0][0], sizeof(sumKeU) / sizeof(float));
	initSharedMem(&sumKs[0][0], sizeof(sumKs) / sizeof(float));

	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	//if (tid == 0) {
		for (int i = 0; i < 4; i++)
		{
			for (int j = 0; j < 32; j++)
				sumKeU[i][j] += 1;
		}
	//}
	int warpId = threadIdx.x / 32;
	int laneId = threadIdx.x % 32;


		// local vertex id in gs set
	int vid = blockIdx.x * 32 + laneId;

	bool fiction = false;

    // load cell and vertex reso
	constant2DToShared(gGsCellReso, gsCellReso);
	constant2DToShared(gGsVertexReso, gsVertexReso);
	constantToShared(gGsCellEnd, gsCellEnd);
	constantToShared(gGsVertexEnd, gsVertexEnd);

	loadTemplateMatrix_H(KE);

	// to global vertex id
	vid = gs_set == 0 ? vid : gsVertexEnd[gs_set - 1] + vid;

	if (vid >= gsVertexEnd[gs_set]) fiction = true;

	GridVertexIndex indexer(gridCellReso[0], gridCellReso[1], gridCellReso[2]);
	VertexFlags vflag;
	if (!fiction) {
		vflag = vflags[vid];
		fiction = fiction || vflag.is_fiction();
		indexer.locate(vid, vflag.get_gscolor(), gsVertexEnd);
	}

	float KeU = { 0. };
	float Ks = { 0. };

	if (!fiction && !vflag.is_period_padding()) {
		int elementId = indexer.neighElement(warpId, gsCellEnd, gsCellReso).getId();
		int vselfrow = (7 - warpId);
		float rho_penal = 0;
		CellFlags eflag;
		float penal = exp_penal[0];
		if (elementId != -1) {
			eflag = eflags[elementId];
			if (!eflag.is_fiction()) rho_penal = rhoPenalMin + powf(float(rholist[elementId]), penal);
		}

		if (elementId != -1 && !eflag.is_fiction() /*&& !eflag.is_period_padding()*/) {
#pragma unroll
			for (int i = 0; i < 8; i++) {
				if (i == 7 - warpId) continue;
				//#if 0
				int vneigh =
					(warpId % 2 + i % 2) +
					(warpId / 2 % 2 + i / 2 % 2) * 3 +
					(warpId / 4 + i / 4) * 9;
				int vneighId = indexer.neighVertex(vneigh, gsVertexEnd, gsVertexReso).getId();
				VertexFlags nvflag;
				if (vneighId != -1) {
					nvflag = vflags[vneighId];
					if (!nvflag.is_fiction()) {
						float u = { gU_H[0][vneighId] };
						if (nvflag.is_dirichlet_boundary()) {
							u = 0;
						}
						KeU += KE[vselfrow][i] * u;
					}
				}
			}
			KeU *= rho_penal;
			Ks += float(KE[vselfrow][vselfrow]) * rho_penal;
		}
	}

	if (warpId >= 4) {
		sumKs[warpId - 4][laneId] = Ks;
		sumKeU[warpId - 4][laneId] = KeU;
	}
	__syncthreads();

	if (warpId < 4) {
		sumKs[warpId][laneId] += Ks;
		sumKeU[warpId][laneId] += KeU;
	}
	__syncthreads();

	if (warpId < 2) {
		sumKs[warpId][laneId] += sumKs[warpId + 2][laneId];
		sumKeU[warpId][laneId] += sumKeU[warpId + 2][laneId];
	}
	__syncthreads();

	if (warpId < 1 && !vflag.is_period_padding() && !fiction) {
		Ks = sumKs[warpId][laneId] + sumKs[warpId + 1][laneId];
		KeU = sumKeU[warpId][laneId] + sumKeU[warpId + 1][laneId];

		float u = { gU_H[0][vid] };

		// relax
#if !USING_SOR 
		u = (gF_H[vid] - KeU) / Ks;
#else
		u = w * (gF_H[0][vid] - KeU) / Ks + (float(1) - w) * u;
#endif

		// if dirichlet boundary;
		if (vflag.is_dirichlet_boundary()) { u = 0; }
		// update
		gU_H[0][vid] = u;
	}
}
// map 32 vertices to 8 warp
template<typename T, int BlockSize = 32 * 8>
__global__ void gs_relaxation_otf_kernel_host_H(
	int gs_set, T* rholist, T* ulist, T* flist,
	devArray_t<int, 3> gridCellReso,
	VertexFlags* vflags, CellFlags* eflags,
	// SOR relaxing factor
	int bx, int by, int bz
) {

	__shared__ int gsCellReso[3][8];
	__shared__ int gsVertexReso[3][8];
	__shared__ int gsCellEnd[8];
	__shared__ int gsVertexEnd[8];

	__shared__ float KE[8][8];

	__shared__ float sumKeU[4][32];
	__shared__ float sumKs[4][32];

	initSharedMem(&sumKeU[0][0], sizeof(sumKeU) / sizeof(float));
	initSharedMem(&sumKs[0][0], sizeof(sumKs) / sizeof(float));

	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	int warpId = threadIdx.x / 32;
	int laneId = threadIdx.x % 32;

	// local vertex id in gs set
	int vid = blockIdx.x * 32 + laneId;

	bool fiction = false;

	// load cell and vertex reso
	constant2DToShared(gGsCellReso, gsCellReso);
	constant2DToShared(gGsVertexReso, gsVertexReso);
	constantToShared(gGsCellEnd, gsCellEnd);
	constantToShared(gGsVertexEnd, gsVertexEnd);

	loadTemplateMatrix_H(KE);

	// to global vertex id
	vid = gs_set == 0 ? vid : gsVertexEnd[gs_set - 1] + vid;

	if (vid >= gsVertexEnd[gs_set]) fiction = true;

	GridVertexIndex indexer(gridCellReso[0], gridCellReso[1], gridCellReso[2]);

	VertexFlags vflag;
	if (!fiction) {
		vflag = vflags[vid];
		fiction = fiction || vflag.is_fiction();
		indexer.locate(vid, vflag.get_gscolor(), gsVertexEnd);
	}

	float KeU = { 0. };
	float Ks = { 0. };

	if (!fiction && !vflag.is_period_padding()) {
		int elementId = indexer.neighElement(warpId, gsCellEnd, gsCellReso).getId();
		int vselfrow = (7 - warpId);
		float rho_penal = 0;
		CellFlags eflag;
		if (elementId != -1) {
			eflag = eflags[elementId];
			if (!eflag.is_fiction()) rho_penal = float(rholist[elementId]);
		}

		if (elementId != -1 && !eflag.is_fiction()) {
#pragma unroll
			for (int i = 0; i < 8; i++) {
				if (i == 7 - warpId) continue;
				//#if 0
				int vneigh =
					(warpId % 2 + i % 2) +
					(warpId / 2 % 2 + i / 2 % 2) * 3 +
					(warpId / 4 + i / 4) * 9;
				int vneighId = indexer.neighVertex(vneigh, gsVertexEnd, gsVertexReso).getId();
				VertexFlags nvflag;
				if (vneighId != -1) {
					nvflag = vflags[vneighId];
					if (!nvflag.is_fiction()) {
						float u = { ulist[vneighId] };
						u = ((bx | by | bz) == 0 && vflag.is_dirichlet_boundary()) ? 0.f : u;
						KeU += KE[vselfrow][i] * u;
					}
				}
			}
			KeU *= rho_penal;
			Ks += float(KE[vselfrow][vselfrow]) * rho_penal;
		}
	}

	if (warpId >= 4) {
		sumKs[warpId - 4][laneId] = Ks;
		sumKeU[warpId - 4][laneId] = KeU;
	}
	__syncthreads();

	if (warpId < 4) {
		sumKs[warpId][laneId] += Ks;
		sumKeU[warpId][laneId] += KeU;
	}
	__syncthreads();

	if (warpId < 2) {
		sumKs[warpId][laneId] += sumKs[warpId + 2][laneId];
		sumKeU[warpId][laneId] += sumKeU[warpId + 2][laneId];
	}
	__syncthreads();

	if (warpId < 1 && !vflag.is_period_padding() && !fiction) {
		Ks = sumKs[warpId][laneId] + sumKs[warpId + 1][laneId];
		KeU = sumKeU[warpId][laneId] + sumKeU[warpId + 1][laneId];

		float u = (flist[vid] - KeU) / Ks;

		u = ((bx | by | bz) == 0 && vflag.is_dirichlet_boundary()) ? 0.f : u;
		// update
		ulist[vid] = u;
	}
}

// map 32 vertices to 13 warp
template<int BlockSize = 32 * 13>
__global__ void gs_relaxation_kernel_H(
	int gs_set,
	VertexFlags* vflags,
	// SOR relaxing factor
	VT w = 1.f
) {
	__shared__ float sumAu[1][7][32];
	__shared__ int gsVertexEnd[8];
	__shared__ int gsVertexReso[3][8];

	constantToShared(gGsVertexEnd, gsVertexEnd);
	constant2DToShared(gGsVertexReso, gsVertexReso);
	initSharedMem(&sumAu[0][0][0], sizeof(sumAu) / sizeof(float));
	__syncthreads();

	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	int warpId = threadIdx.x / 32;
	int laneId = threadIdx.x % 32;

	// local vertex id in gs set
	int vid = blockIdx.x * 32 + laneId;
	// global vertex id
	vid = gs_set == 0 ? vid : gsVertexEnd[gs_set - 1] + vid;

	bool fiction = false;
	if (vid >= gsVertexEnd[gs_set]) fiction = true;
	VertexFlags vflag;
	if (!fiction) vflag = vflags[vid];
	fiction = fiction || vflag.is_fiction();

	GridVertexIndex indexer(gGridCellReso[0], gGridCellReso[1], gGridCellReso[2]);

	float Au = 0.;

	if (!fiction && !vflag.is_period_padding()) {
		indexer.locate(vid, vflag.get_gscolor(), gsVertexEnd);

		for (int noff : {0, 14}) {
			int vneigh = warpId + noff;
			int neighId = indexer.neighVertex(vneigh, gsVertexEnd, gsVertexReso).getId();
			if (neighId == -1) continue;
			VertexFlags neiflag = vflags[neighId];
			if (!neiflag.is_fiction()) {
				float u = gU_H[0][neighId];
				Au += rxstencil_H[vneigh][vid] * u;
			}
		}
	}


	if (warpId >= 7) {
		sumAu[0][warpId - 7][laneId] = Au;
	}
	__syncthreads();

	if (warpId < 7) {
		if (warpId < 6) {
			sumAu[0][warpId][laneId] += Au;
		}
		else {
			sumAu[0][6][laneId] = Au;
		}
	}
	__syncthreads();

	if (warpId < 3) {
		sumAu[0][warpId][laneId] += sumAu[0][warpId + 4][laneId];
	}
	__syncthreads();

	if (warpId < 2) {
		sumAu[0][warpId][laneId] += sumAu[0][warpId + 2][laneId];
	}
	__syncthreads();

	if (warpId < 1 && !fiction) {
		Au = sumAu[0][warpId][laneId] + sumAu[0][warpId + 1][laneId];

		if (!vflag.is_period_padding()) {
			VT u = gU_H[0][vid];
			// glm::hmat3 st = rxstencil[13][vid];
			VT st = rxstencil_H[13][vid];
#if !USING_SOR
			u[0] = (gF[0][vid] - Au[0] - rxstencil[13][1][vid] * u[1] - rxstencil[13][2][vid] * u[2]) / rxstencil[13][0][vid];
			u[1] = (gF[1][vid] - Au[1] - rxstencil[13][3][vid] * u[0] - rxstencil[13][5][vid] * u[2]) / rxstencil[13][4][vid];
			u[2] = (gF[2][vid] - Au[2] - rxstencil[13][6][vid] * u[0] - rxstencil[13][7][vid] * u[1]) / rxstencil[13][8][vid];
#else
			VT f = gF_H[0][vid];
			u = w * (f - VT(Au)) / st + (VT(1) - w) * u;
#endif

			//if (rxstencil[13][0][vid] == 0) {
			//	short3 pos = indexer.getPos();
			//	printf("pos = (%d, %d, %d) d = (%e %e %e)\n",
			//		pos.x - 1, pos.y - 1, pos.z - 1,
			//		rxstencil[13][0][vid], rxstencil[13][4][vid], rxstencil[13][8][vid]);
			//}

			gU_H[0][vid] = u;
		}
	}
}

template<int BlockSize = 256>
__global__ void restrict_residual_kernel_H(
	int nv_coarse,
	VertexFlags* vflags,
	VertexFlags* vfineflags,
	devArray_t<int, 8> GsVertexEnd,
	devArray_t<int, 8> GsFineVertexEnd
) {
	__shared__ int gsVertexEnd[8];
	__shared__ int gsFineVertexEnd[8];
	__shared__ int gsFineVertexReso[3][8];

	if (threadIdx.x < 24) {
		gsFineVertexReso[threadIdx.x / 8][threadIdx.x % 8] =
			gGsFineVertexReso[threadIdx.x / 8][threadIdx.x % 8];
		if (threadIdx.x < 8) {
			gsVertexEnd[threadIdx.x] = GsVertexEnd[threadIdx.x];
			gsFineVertexEnd[threadIdx.x] = GsFineVertexEnd[threadIdx.x];
		}
	}
	__syncthreads();

	size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid >= nv_coarse) return;

	VertexFlags vflag = vflags[tid];
	bool fiction = vflag.is_fiction();

	int setid = vflag.get_gscolor();

	GridVertexIndex indexer(gGridCellReso[0], gGridCellReso[1], gGridCellReso[2]);
	indexer.locate(tid, vflag.get_gscolor(), gsVertexEnd);

	int coarseRatio[3] = { gUpCoarse[0], gUpCoarse[1], gUpCoarse[2] };
	float pr = coarseRatio[0] * coarseRatio[1] * coarseRatio[2];

	bool nondyadic = coarseRatio[0] > 2 || coarseRatio[1] > 2 || coarseRatio[2] > 2;

	float r = { 0. };

	if (!fiction && !vflag.is_period_padding()) {
		for (int offx = -coarseRatio[0] + 1; offx < coarseRatio[0]; offx++) {
			for (int offy = -coarseRatio[1] + 1; offy < coarseRatio[1]; offy++) {
				for (int offz = -coarseRatio[2] + 1; offz < coarseRatio[2]; offz++) {
					int off[3] = { offx,offy,offz };
					float w = (coarseRatio[0] - abs(offx)) * (coarseRatio[1] - abs(offy)) * (coarseRatio[2] - abs(offz)) / pr;
					int neighVid = -1;
					// DEBUG
					if (nondyadic)
						neighVid = indexer.neighFineVertex(off, coarseRatio, gsFineVertexEnd, gsFineVertexReso, true).getId();
					else
						neighVid = indexer.neighFineVertex(off, coarseRatio, gsFineVertexEnd, gsFineVertexReso, false).getId();

					VertexFlags vfineflag;
					if (neighVid != -1) {
						vfineflag = vfineflags[neighVid];
						if (!vfineflag.is_fiction()) {
							r += float(gRfine_H[0][neighVid]) * w;
						}
					}
				}
			}
		}
		//if (vflag.is_dirichlet_boundary()) r[0] = r[1] = r[2] = 0;
		gF_H[0][tid] = r;
	}
}

template<int BlockSize = 256>
__global__ void restrict_residual_kernel_host_H(
	int nv_fine,
	VertexFlags* vflags,
	VertexFlags* vfineflags,
	devArray_t<int, 8> GsVertexEnd,
	devArray_t<int, 8> GsFineVertexEnd,
	int bx, int by, int bz
) {
	__shared__ int gsVertexEnd[8];
	__shared__ int gsFineVertexEnd[8];
	__shared__ int gsFineVertexReso[3][8];

	if (threadIdx.x < 24) {
		gsFineVertexReso[threadIdx.x / 8][threadIdx.x % 8] =
			gGsFineVertexReso[threadIdx.x / 8][threadIdx.x % 8];
		if (threadIdx.x < 8) {
			gsVertexEnd[threadIdx.x] = GsVertexEnd[threadIdx.x];
			gsFineVertexEnd[threadIdx.x] = GsFineVertexEnd[threadIdx.x];
		}
	}
	__syncthreads();

	size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid >= nv_fine) return;

	VertexFlags vflag = vfineflags[tid];
	bool fiction = vflag.is_fiction();

	int setid = vflag.get_gscolor();

	GridVertexIndex indexer(MIN_TRANSFER, MIN_TRANSFER, MIN_TRANSFER);
	indexer.locate(tid, vflag.get_gscolor(), gsFineVertexEnd);
	short3 pos = indexer.getPos();

	int coarseRatio[3] = { gUpCoarse[0], gUpCoarse[1], gUpCoarse[2] };
	float pr = coarseRatio[0] * coarseRatio[1] * coarseRatio[2];
	pos.x -= 1;
	pos.y -= 1;
	pos.z -= 1;
	float r = { 0. };
	int bid[3] = { bx, by, bz };
	// if the vertex on the face of a block, it contribution need to * 0.5; 
	float ratio = 1;
	if (pos.x == 0 || pos.x == MIN_TRANSFER) { ratio *= 0.5; }
	if (pos.y == 0 || pos.y == MIN_TRANSFER) { ratio *= 0.5; }
	if (pos.z == 0 || pos.z == MIN_TRANSFER) { ratio *= 0.5; }

	if (!fiction && !vflag.is_period_padding()) {
		for (int offx = -coarseRatio[0] + 1; offx < coarseRatio[0]; offx++) {
			for (int offy = -coarseRatio[1] + 1; offy < coarseRatio[1]; offy++) {
				for (int offz = -coarseRatio[2] + 1; offz < coarseRatio[2]; offz++) {
					// for reso == 32 pos is from 1 to 33
					int off_pos[3] = { pos.x + offx, pos.y + offy, pos.z + offz };
					// check if off_pos reffer to a coarse vertex
					// if no referred skip
					if (off_pos[0] % coarseRatio[0] == 0 && off_pos[1] % coarseRatio[1] == 0 && off_pos[2] % coarseRatio[2] == 0) {
						// weight is still the same
						float w = (coarseRatio[0] - abs(offx)) * (coarseRatio[1] - abs(offy)) * (coarseRatio[2] - abs(offz)) / pr;
						// add to that gF_H
						// coarse pos block:
						int c_pos[3] = { off_pos[0] / coarseRatio[0], off_pos[1] / coarseRatio[1], off_pos[2] / coarseRatio[2] };
						// true coarse pos by adding block
						int c_true_pos[3];
						for (int pos_idx = 0; pos_idx < 3; pos_idx++) {
							c_true_pos[pos_idx] = (MIN_TRANSFER / coarseRatio[pos_idx]) * bid[pos_idx] + c_pos[pos_idx];
						}
						// printf("fine pos: (%d, %d, %d) coarse pos:(%d, %d, %d)\n",pos.x, pos.y, pos.z, c_true_pos[0], c_true_pos[1], c_true_pos[2]);

						// if c_true_pos on the edges do to the opposite pos too
						int xp[2] = {c_true_pos[0], -1};
						int yp[2] = {c_true_pos[1], -1};
						int zp[2] = {c_true_pos[2], -1};
						if (xp[0] == 0) { xp[1] = gGridCellReso[0]; }
						if (xp[0] == gGridCellReso[0]) { xp[1] = 0; }
						if (yp[0] == 0) { yp[1] = gGridCellReso[1]; }
						if (yp[0] == gGridCellReso[1]) { yp[1] = 0; }
						if (zp[0] == 0) { zp[1] = gGridCellReso[2]; }
						if (zp[0] == gGridCellReso[2]) { zp[1] = 0; }
						for (int i = 0; i < 8; i++) {
							int idx = i % 2;
							int idy = i / 2 % 2;
							int idz = i / 4;
							int reflect[3];
							if (xp[idx] != -1 && yp[idy] != -1 && zp[idz] != -1) {
								reflect[0] = xp[idx]; reflect[1] = yp[idy]; reflect[2] = zp[idz];
								int coarse_lexi = lexi2gs(reflect, gGsVertexReso, gGsVertexEnd);
								atomicAdd(&gF_H[0][coarse_lexi], float(gRfine_H[0][tid]) * w * ratio);
							}
						}
					}
				}
			}
		}
	}
}

template<int BlockSize = 256>
__global__ void prolongate_correction_kernel_H(
	bool is_root,
	int nv_fine,
	VertexFlags* vflags,
	VertexFlags* vcoarseflags,
	devArray_t<int, 8> GsVertexEnd,
	devArray_t<int, 8> GsCoarseVertexEnd
) {
	__shared__ int coarseRatio[3];
	__shared__ int gsCoarseVertexReso[3][8];
	__shared__ int gsCoarseVertexEnd[8];

	if (threadIdx.x < 24) {
		gsCoarseVertexReso[threadIdx.x / 8][threadIdx.x % 8] = gGsCoarseVertexReso[threadIdx.x / 8][threadIdx.x % 8];
	}
	if (threadIdx.x < 8) {
		gsCoarseVertexEnd[threadIdx.x] = GsCoarseVertexEnd[threadIdx.x];
	}
	if (threadIdx.x < 3) {
		coarseRatio[threadIdx.x] = gDownCoarse[threadIdx.x];
	}
	__syncthreads();

	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	bool fiction = false;
	if (tid >= nv_fine) {
		return;
	}

	VertexFlags vflag = vflags[tid];
	fiction = fiction || vflag.is_fiction();

	float pr = coarseRatio[0] * coarseRatio[1] * coarseRatio[2];

	bool isRoot = is_root;

	if (!fiction && !vflag.is_period_padding()) {
		GridVertexIndex indexer(gGridCellReso[0], gGridCellReso[1], gGridCellReso[2]);
		indexer.locate(tid, vflag.get_gscolor(), GsVertexEnd._data);

		float u = { 0. };
		int nvCoarse[8];
		float w[8];
		int remainder[3];
		indexer.neighCoarseVertex(nvCoarse, w, coarseRatio, gsCoarseVertexEnd, gsCoarseVertexReso, remainder);
		for (int i = 0; i < 8; i++) {
			int neighId = nvCoarse[i];
			if (neighId != -1) {
				float uc = { gUcoarse_H[0][neighId] };
				u += uc * w[i];
			}
		}

		if (isRoot && vflag.is_dirichlet_boundary()) {
			u = 0;
		}
		gU_H[0][tid] += VT(u);
	}
}
template<int BlockSize = 256>
__global__ void prolongate_correction_kernel_host_H(
	bool is_root,
	int nv_fine,
	VertexFlags* vflags,
	VertexFlags* vcoarseflags,
	devArray_t<int, 8> GsVertexEnd,
	devArray_t<int, 8> GsCoarseVertexEnd,
	int bx, int by, int bz
) {
	__shared__ int coarseRatio[3];
	__shared__ int gsCoarseVertexReso[3][8];
	__shared__ int gsCoarseVertexEnd[8];

	if (threadIdx.x < 24) {
		gsCoarseVertexReso[threadIdx.x / 8][threadIdx.x % 8] = gGsCoarseVertexReso[threadIdx.x / 8][threadIdx.x % 8];
	}
	if (threadIdx.x < 8) {
		gsCoarseVertexEnd[threadIdx.x] = GsCoarseVertexEnd[threadIdx.x];
	}
	if (threadIdx.x < 3) {
		coarseRatio[threadIdx.x] = gDownCoarse[threadIdx.x];
	}
	__syncthreads();
	// tid is for fine vertex index
	// so to change it to cpu version, we need to find the right coarse reference
	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	bool fiction = false;
	if (tid >= nv_fine) {
		return;
	}

	VertexFlags vflag = vflags[tid];
	fiction = fiction || vflag.is_fiction();

	float pr = coarseRatio[0] * coarseRatio[1] * coarseRatio[2];

	bool isRoot = is_root;

	if (!fiction && !vflag.is_period_padding()) {
		GridVertexIndex indexer(MIN_TRANSFER, MIN_TRANSFER, MIN_TRANSFER);
		indexer.locate(tid, vflag.get_gscolor(), GsVertexEnd._data);

		float u = { 0. };

		// originally here use neighCoarseVertex
		// however it's a little tedious to change the interface
		// we copy it and do it by hand
		int nvCoarse[8];
		float w[8];
		int remainder[3];

		short3 pos;
		pos = indexer.getPos();
		pos.x -= 1;
		pos.y -= 1;
		pos.z -= 1;

		if (pos.x < 0 || pos.y < 0 || pos.z < 0) {
			printf("pos is: (%d, %d, %d)", pos.x, pos.y, pos.z);
		};

		remainder[0] = pos.x % coarseRatio[0];
		remainder[1] = pos.y % coarseRatio[1];
		remainder[2] = pos.z % coarseRatio[2];

		short3 coarseBase{ pos.x / coarseRatio[0] + bx * (MIN_TRANSFER / coarseRatio[0]),
			pos.y / coarseRatio[1] + by * (MIN_TRANSFER / coarseRatio[1]),
			pos.z / coarseRatio[2] + bz * (MIN_TRANSFER / coarseRatio[2])};
		// printf("coarseBase is:(%d, %d, %d)", coarseBase.x, coarseBase.y, coarseBase.z);
		float wc = coarseRatio[0] * coarseRatio[1] * coarseRatio[2];

		for (int id = 0; id < 8; id++) {
			short3 idoffCoarse = { id % 2, id / 2 % 2, id / 4 };
			short3 idOff = { idoffCoarse.x * coarseRatio[0], idoffCoarse.y * coarseRatio[1], idoffCoarse.z * coarseRatio[2] };
			w[id] = (coarseRatio[0] - abs(idOff.x - remainder[0])) *
				(coarseRatio[1] - abs(idOff.y - remainder[1])) *
				(coarseRatio[2] - abs(idOff.z - remainder[2])) / wc;
			pos.x = coarseBase.x + idoffCoarse.x + 1;
			pos.y = coarseBase.y + idoffCoarse.y + 1;
			pos.z = coarseBase.z + idoffCoarse.z + 1;
			int color = pos.x % 2 + pos.y % 2 * 2 + pos.z % 2 * 4;
			int base = color == 0 ? 0 : gsCoarseVertexEnd[color - 1];
			int gsid = base +
				pos.x / 2 +
				pos.y / 2 * gsCoarseVertexReso[0][color] +
				pos.z / 2 * gsCoarseVertexReso[0][color] * gsCoarseVertexReso[1][color];
			if (gsid >= gsCoarseVertexEnd[color]) { gsid = -1; }
			nvCoarse[id] = gsid;
		}


		for (int i = 0; i < 8; i++) {
			int neighId = nvCoarse[i];
			if (neighId != -1) {
				float uc = { gUcoarse_H[0][neighId] };
				u += uc * w[i];
			}
		}

		if (isRoot) {
			if (bx == 0 && by == 0 && bz == 0 && vflag.is_dirichlet_boundary()) {
				u = 0;
			}
		}
		gU_H[0][tid] += VT(u);
	}
}

void homo::Grid_H::gs_relaxation(float w_SOR /*= 1.f*/, int times_ /*= 1*/)
{
	AbortErr();
	// change to 8 bytes bank
	use8Bytesbank();
	useGrid_g();
	devArray_t<int, 3>  gridCellReso{};
	devArray_t<int, 8>  gsCellEnd{};
	devArray_t<int, 8>  gsVertexEnd{};
	for (int i = 0; i < 8; i++) {
		gsCellEnd[i] = gsCellSetEnd[i];
		gsVertexEnd[i] = gsVertexSetEnd[i];
		if (i < 3) gridCellReso[i] = cellReso[i];
	}
	for (int itn = 0; itn < times_; itn++) {
		for (int i = 0; i < 8; i++) {
			int set_id = i;
			size_t grid_size, block_size;
			int n_gs = gsVertexEnd[set_id] - (set_id == 0 ? 0 : gsVertexEnd[set_id - 1]);
			if (assemb_otf) {
				make_kernel_param(&grid_size, &block_size, n_gs * 8, 32 * 8);
				gs_relaxation_otf_kernel_H << <grid_size, block_size >> > (set_id, rho_g, gridCellReso, vertflag, cellflag, w_SOR, diag_strength);
			}
			else {
				make_kernel_param(&grid_size, &block_size, n_gs * 13, 32 * 13);
				gs_relaxation_kernel_H << <grid_size, block_size >> > (set_id, vertflag, w_SOR);
			}
			cudaDeviceSynchronize();
			cuda_error_check;
			//enforce_period_boundary(u_g);
		}
		enforce_period_boundary(u_g);
	}
	cudaDeviceSynchronize();
	cuda_error_check;
}

void homo::Grid_H::gs_relaxation_host(int blockid, float w_SOR /*= 1.f*/, int times_ /*= 1*/)
{
	AbortErr();
	// change to 8 bytes bank
	use8Bytesbank();
	devArray_t<int, 3>  gridCellReso{};
	devArray_t<int, 8>  gsCellEnd{};
	devArray_t<int, 8>  gsVertexEnd{};
	for (int i = 0; i < 8; i++) {
		gsCellEnd[i] = gsCellSetEnd[i];
		gsVertexEnd[i] = gsVertexSetEnd[i];
		if (i < 3) gridCellReso[i] = MIN_TRANSFER;
	}
	int blockx, blocky, blockz;
	blockx = cellReso[0] / MIN_TRANSFER;
	blocky = cellReso[1] / MIN_TRANSFER;

	int bx = blockid % blockx;
	int by = (blockid / blockx) % blocky;
	int bz = blockid / (blockx * blocky);
	for (int itn = 0; itn < times_; itn++) {
		for (int i = 0; i < 8; i++) {
			int set_id = i;
			size_t grid_size, block_size;
			int n_gs = gsVertexEnd[set_id] - (set_id == 0 ? 0 : gsVertexEnd[set_id - 1]);
			make_kernel_param(&grid_size, &block_size, n_gs * 8, 32 * 8);
			gs_relaxation_otf_kernel_host_H << <grid_size, block_size, 0, stream[current]>> > (set_id, rho_g, u_g[current], f_g[current], gridCellReso, vertflag, cellflag, bx, by, bz);
		}
	}
}


void homo::Grid_H::prolongate_correction(void)
{
	useGrid_g();
	VertexFlags* vflags = vertflag;
	VertexFlags* vcoarseFlags = Coarse->vertflag;
	devArray_t<int, 8> gsVertexEnd{}, gsCoarseVertexEnd{};
	for (int i = 0; i < 8; i++) {
		gsVertexEnd[i] = gsVertexSetEnd[i];
		gsCoarseVertexEnd[i] = Coarse->gsVertexSetEnd[i];
	}
	int nv_fine = n_gsvertices();
	size_t grid_size, block_size;
	make_kernel_param(&grid_size, &block_size, nv_fine, 256);
	prolongate_correction_kernel_H << <grid_size, block_size >> > (is_root, nv_fine, vflags, vcoarseFlags, gsVertexEnd, gsCoarseVertexEnd);
	cudaDeviceSynchronize();
	cuda_error_check;
	enforce_period_boundary(u_g);
}

void homo::Grid_H::prolongate_correction(int blockid)
{
	VertexFlags* vflags = vertflag;
	VertexFlags* vcoarseFlags = Coarse->vertflag;
	devArray_t<int, 8> gsVertexEnd{}, gsCoarseVertexEnd{};
	for (int i = 0; i < 8; i++) {
		gsVertexEnd[i] = gsVertexSetEnd[i];
		gsCoarseVertexEnd[i] = Coarse->gsVertexSetEnd[i];
	}
	int nv_fine = n_gsvertices();
	size_t grid_size, block_size;
	int blockx, blocky, blockz;
	blockx = cellReso[0] / MIN_TRANSFER;
	blocky = cellReso[1] / MIN_TRANSFER;
	int bx = blockid % blockx;
	int by = (blockid / blockx) % blocky;
	int bz = blockid / (blockx * blocky);

	make_kernel_param(&grid_size, &block_size, nv_fine, 256);
	prolongate_correction_kernel_host_H << <grid_size, block_size >> > (is_root, nv_fine, vflags, vcoarseFlags, gsVertexEnd, gsCoarseVertexEnd, bx, by, bz);
	cudaDeviceSynchronize();
	cuda_error_check;
}

void homo::Grid_H::restrict_residual(void)
{
	useGrid_g();
	VertexFlags* vflags = vertflag;
	VertexFlags* vfineflags = fine->vertflag;
	devArray_t<int, 8> gsVertexEnd{}, gsFineVertexEnd{};
	for (int i = 0; i < 8; i++) {
		gsVertexEnd[i] = gsVertexSetEnd[i];
		gsFineVertexEnd[i] = fine->gsVertexSetEnd[i];
	}
	int nv = n_gsvertices();
	size_t grid_size, block_size;
	make_kernel_param(&grid_size, &block_size, nv, 256);
	restrict_residual_kernel_H << <grid_size, block_size >> > (nv, vflags, vfineflags, gsVertexEnd, gsFineVertexEnd);
	cudaDeviceSynchronize();
	cuda_error_check;
	//std::vector<float> transfer(nv);
	//cudaMemcpy(transfer.data(), f_g[0], nv*sizeof(float), cudaMemcpyDeviceToHost);
	//std::ofstream fout("fg.txt");
	//for (const auto& x : transfer) {
	//	fout << x << " ";
	//}
	//fout.close();
	pad_vertex_data(f_g);
}

void homo::Grid_H::restrict_residual(int blockid)
{
	VertexFlags* vflags = vertflag;
	VertexFlags* vfineflags = fine->vertflag;
	devArray_t<int, 8> gsVertexEnd{}, gsFineVertexEnd{};
	for (int i = 0; i < 8; i++) {
		gsVertexEnd[i] = gsVertexSetEnd[i];
		gsFineVertexEnd[i] = fine->gsVertexSetEnd[i];
	}
	int blockx, blocky, blockz;
	blockx = fine->cellReso[0] / MIN_TRANSFER;
	blocky = fine->cellReso[1] / MIN_TRANSFER;
	blockz = fine->cellReso[2] / MIN_TRANSFER;
	int bx = blockid % blockx;
	int by = (blockid / blockx) % blocky;
	int bz = blockid / (blockx * blocky);
	int nv = fine->n_gsvertices();
	// need to change the order here
	size_t grid_size, block_size;
	make_kernel_param(&grid_size, &block_size, nv, 256);
	restrict_residual_kernel_host_H << <grid_size, block_size >> > (nv, vflags, vfineflags, gsVertexEnd, gsFineVertexEnd, bx, by, bz);
	cudaDeviceSynchronize();
	cuda_error_check;
}

template<typename T>
__global__ void update_residual_otf_kernel_1_host_H(
	int nv, T* rholist, T* ulist, T* rlist, T* flist,
	devArray_t<int, 3> gridCellReso,
	VertexFlags* vflags, CellFlags* eflags,
    int bx, int by, int bz
) {
	__shared__ int gsCellReso[3][8];
	__shared__ int gsVertexReso[3][8];
	__shared__ int gsCellEnd[8];
	__shared__ int gsVertexEnd[8];

	__shared__ float KE[8][8];

	__shared__ float sumKeU[4][32];

	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

	int warpId = threadIdx.x / 32;
	int laneId = threadIdx.x % 32;

	// local vertex id in gs set
	int vid = blockIdx.x * 32 + laneId;

	bool fiction = false;

	fiction = fiction || vid >= nv;
	VertexFlags vflag;
	if (!fiction) {
		vflag = vflags[vid];
		fiction = fiction || vflag.is_fiction() || vflag.is_period_padding();
	}
	int set_id = vflag.get_gscolor();

	// load cell and vertex reso
	constant2DToShared(gGsCellReso, gsCellReso);
	constant2DToShared(gGsVertexReso, gsVertexReso);
	constantToShared(gGsCellEnd, gsCellEnd);
	constantToShared(gGsVertexEnd, gsVertexEnd);
	initSharedMem(&sumKeU[0][0], sizeof(sumKeU) / sizeof(float));
	// load template matrix
	loadTemplateMatrix_H(KE);

	GridVertexIndex indexer(gridCellReso[0], gridCellReso[1], gridCellReso[2]);
	if (!fiction) {
		indexer.locate(vid, vflag.get_gscolor(), gsVertexEnd);
	}

	float KeU = 0.;

	int elementId = -1;
	if (!fiction) elementId = indexer.neighElement(warpId, gsCellEnd, gsCellReso).getId();
	int vselfrow = (7 - warpId);
	float rhop = 0;
	CellFlags eflag;
	if (elementId != -1) {
		eflag = eflags[elementId];
		if (!eflag.is_fiction()) rhop = float(rholist[elementId]);
	}

	if (elementId != -1 && !eflag.is_fiction() && !fiction) {
#pragma unroll
		for (int i = 0; i < 8; i++) {
			int vneigh =
				(warpId % 2 + i % 2) +
				(warpId / 2 % 2 + i / 2 % 2) * 3 +
				(warpId / 4 + i / 4) * 9;
			int vneighId = indexer.neighVertex(vneigh, gsVertexEnd, gsVertexReso).getId();
			VertexFlags nvflag;
			if (vneighId != -1) {
				nvflag = vflags[vneighId];
				if (!nvflag.is_fiction()) {
					float u = ulist[vneighId];
					u = ((bx | by | bz) == 0 && nvflag.is_dirichlet_boundary()) ? 0.f : u;
					KeU += KE[vselfrow][i] * u;
				}
			}
		}
		KeU *= rhop;
	}

	if (warpId >= 4) {
		sumKeU[warpId - 4][laneId] = KeU;
	}
	__syncthreads();

	if (warpId < 4) {
		sumKeU[warpId][laneId] += KeU;
	}
	__syncthreads();

	if (warpId < 2) {
		sumKeU[warpId][laneId] += sumKeU[warpId + 2][laneId];
	}
	__syncthreads();

	if (warpId < 1 && !fiction && !vflag.is_period_padding()) {
		KeU = sumKeU[warpId][laneId] + sumKeU[warpId + 1][laneId];
		float r = float(flist[vid]) - KeU;
		r = ((bx | by | bz) == 0 && vflag.is_dirichlet_boundary()) ? 0.f : r;
		rlist[vid] = r;
	}
}
template<typename T>
__global__ void update_residual_otf_kernel_1_H(
	int nv, T* rholist,
	devArray_t<int, 3> gridCellReso,
	VertexFlags* vflags, CellFlags* eflags,
	float diag_strength
) {
	__shared__ int gsCellReso[3][8];
	__shared__ int gsVertexReso[3][8];
	__shared__ int gsCellEnd[8];
	__shared__ int gsVertexEnd[8];

	__shared__ float KE[8][8];

	__shared__ float sumKeU[4][32];

	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

	int warpId = threadIdx.x / 32;
	int laneId = threadIdx.x % 32;

	// local vertex id in gs set
	int vid = blockIdx.x * 32 + laneId;

	bool fiction = false;

	fiction = fiction || vid >= nv;
	VertexFlags vflag;
	if (!fiction) {
		vflag = vflags[vid];
		fiction = fiction || vflag.is_fiction() || vflag.is_period_padding();
	}

	// load cell and vertex reso
	constant2DToShared(gGsCellReso, gsCellReso);
	constant2DToShared(gGsVertexReso, gsVertexReso);
	constantToShared(gGsCellEnd, gsCellEnd);
	constantToShared(gGsVertexEnd, gsVertexEnd);
	initSharedMem(&sumKeU[0][0], sizeof(sumKeU) / sizeof(float));
	// load template matrix
	loadTemplateMatrix_H(KE);

	GridVertexIndex indexer(gridCellReso[0], gridCellReso[1], gridCellReso[2]);
	if (!fiction) {
		indexer.locate(vid, vflag.get_gscolor(), gsVertexEnd);
	}


	float KeU = 0.;

	// if (!fiction) gR[0][vid] = KE[0][0];

	int elementId = -1;
	if (!fiction) elementId = indexer.neighElement(warpId, gsCellEnd, gsCellReso).getId();
	int vselfrow = (7 - warpId);
	float rhop = 0;
	CellFlags eflag;
	if (elementId != -1) {
		eflag = eflags[elementId];
		if (!eflag.is_fiction()) rhop = float(rholist[elementId]);
	}

	if (elementId != -1 && !eflag.is_fiction() && !fiction) {
#pragma unroll
		for (int i = 0; i < 8; i++) {
			int vneigh =
				(warpId % 2 + i % 2) +
				(warpId / 2 % 2 + i / 2 % 2) * 3 +
				(warpId / 4 + i / 4) * 9;
			int vneighId = indexer.neighVertex(vneigh, gsVertexEnd, gsVertexReso).getId();
			VertexFlags nvflag;
			if (vneighId != -1) {
				nvflag = vflags[vneighId];
				if (!nvflag.is_fiction()) {
					float u = gU_H[0][vneighId];
					if (nvflag.is_dirichlet_boundary()) {
						u = 0;
					}
					KeU += KE[vselfrow][i] * u;
				}
			}
		}
		KeU *= rhop;
	}

	if (warpId >= 4) {
		sumKeU[warpId - 4][laneId] = KeU;
	}
	__syncthreads();

	if (warpId < 4) {
		sumKeU[warpId][laneId] += KeU;
	}
	__syncthreads();

	if (warpId < 2) {
		sumKeU[warpId][laneId] += sumKeU[warpId + 2][laneId];
	}
	__syncthreads();

	if (warpId < 1 && !fiction && !vflag.is_period_padding()) {
		KeU = sumKeU[warpId][laneId] + sumKeU[warpId + 1][laneId];

		float r = float(gF_H[0][vid]) - KeU;

		if (vflag.is_dirichlet_boundary()) {
			r = 0;
		}
		// relax
		gR_H[0][vid] = r;
	}
}

// map 32 vertices to 9 warp
template<int BlockSize = 32 * 9>
__global__ void update_residual_kernel_1_H(
	int nv,
	VertexFlags* vflags
) {

	__shared__ int gsVertexEnd[8];
	__shared__ int gsVertexReso[3][8];
	__shared__ float sumKu[1][5][32];

	constantToShared(gGsVertexEnd, gsVertexEnd);
	constant2DToShared(gGsVertexReso, gsVertexReso);

	initSharedMem(&sumKu[0][0][0], sizeof(sumKu) / sizeof(float));

	__syncthreads();

	bool fiction = false;

	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	int warpId = threadIdx.x / 32;
	int laneId = threadIdx.x % 32;
	int vid = blockIdx.x * 32 + laneId;
	if (vid >= nv) fiction = true;

	VertexFlags vflag;
	if (!fiction) vflag = vflags[vid];
	fiction = fiction || vflag.is_fiction();
	int color = vflag.get_gscolor();


	GridVertexIndex indexer(gGridCellReso[0], gGridCellReso[1], gGridCellReso[2]);
	indexer.locate(vid, vflag.get_gscolor(), gsVertexEnd);

	float KeU = 0.;
	if (!fiction && !vflag.is_period_padding()) {
		for (auto off : { 0,9,18 }) {
			int vneigh = off + warpId;
			int neighId = indexer.neighVertex(vneigh, gsVertexEnd, gsVertexReso).getId();
			if (neighId != -1) {
				VertexFlags neighFlag = vflags[neighId];
				if (!neighFlag.is_fiction()) {
					float u = gU_H[0][neighId];
					KeU += rxstencil_H[vneigh][vid] * u;
				}
			}
		}
	}

	if (warpId >= 4) {
		sumKu[0][warpId - 4][laneId] = KeU;
	}
	__syncthreads();

	if (warpId < 4) {
		sumKu[0][warpId][laneId] += KeU;
	}
	__syncthreads();

	if (warpId < 2) {
		sumKu[0][warpId][laneId] += sumKu[0][warpId + 2][laneId];
	}
	__syncthreads();

	if (warpId < 1 && !fiction) {
		KeU = sumKu[0][warpId][laneId] + sumKu[0][warpId + 1][laneId] + sumKu[0][4][laneId];

		float r = gF_H[0][vid] - VT(KeU);

		gR_H[0][vid] = r;
	}
}


void homo::Grid_H::update_residual(void)
{
	useGrid_g();
	devArray_t<int, 3> gridCellReso{ cellReso[0],cellReso[1],cellReso[2] };
	VertexFlags* vflags = vertflag;
	CellFlags* eflags = cellflag;
	if (assemb_otf) {
		size_t grid_size, block_size;
		make_kernel_param(&grid_size, &block_size, n_gsvertices() * 8, 32 * 8);
		update_residual_otf_kernel_1_H << <grid_size, block_size >> > (n_gsvertices(), rho_g, gridCellReso,
			vflags, eflags, diag_strength);
		cudaDeviceSynchronize();
		cuda_error_check;
	}
	else {
		size_t grid_size, block_size;
		int nv = n_gsvertices();
		make_kernel_param(&grid_size, &block_size, n_gsvertices() * 9, 32 * 9);
		update_residual_kernel_1_H << <grid_size, block_size >> > (nv, vflags);
		cudaDeviceSynchronize();
		cuda_error_check;
	}
	if (!use_host_memory) {
		enforce_period_boundary(r_g);
		pad_vertex_data(r_g);
		//std::ofstream fout("u_g.txt");
		//std::vector<float> temple_vec(n_gsvertices(), 0);
		//cudaMemcpy(temple_vec.data(), u_g[0], sizeof(VT) * n_gsvertices(), cudaMemcpyDeviceToHost);
		//for (const auto& x : temple_vec) {
		//	fout << x << " ";
		//}
		//fout.close();
	}
}

void homo::Grid_H::update_residual_host(int blockid)
{
	devArray_t<int, 3> gridCellReso{ MIN_TRANSFER, MIN_TRANSFER, MIN_TRANSFER };
	VertexFlags* vflags = vertflag;
	CellFlags* eflags = cellflag;
	int blockx, blocky, blockz;
	blockx = cellReso[0] / MIN_TRANSFER;
	blocky = cellReso[1] / MIN_TRANSFER;

	int bx = blockid % blockx;
	int by = (blockid / blockx) % blocky;
	int bz = blockid / (blockx * blocky);
	size_t grid_size, block_size;
	make_kernel_param(&grid_size, &block_size, n_gsvertices() * 8, 32 * 8);
	update_residual_otf_kernel_1_host_H << <grid_size, block_size, 0, stream[current] >> > (n_gsvertices(), rho_g, u_g[current], r_g[current], f_g[current], gridCellReso,
		vflags, eflags, bx, by, bz);
}

__device__ int gsPos2Id(int pos[3], int* gsEnd, int(*gsReso)[8]) {
	int posRem[3] = { pos[0] % 2, pos[1] % 2, pos[2] % 2 };
	int gsPos[3] = { pos[0] / 2,pos[1] / 2,pos[2] / 2 };
	int color = posRem[0] + posRem[1] * 2 + posRem[2] * 4;

	int gsid = (color == 0 ? 0 : gsEnd[color - 1]) +
		gsPos[0] +
		gsPos[1] * gsReso[0][color] +
		gsPos[2] * gsReso[0][color] * gsReso[1][color];
	if (gsid >= gsEnd[color]) {
		return -1;
	}
	return gsid;
}

template<typename T>
__global__ void enforce_unit_macro_strain_kernel_H(
	int nv, int istrain, devArray_t<Grid_H::VT*, 1> fcharlist, VertexFlags* vflags, CellFlags* eflags, T* rholist, int reso
) {
	__shared__ float feMu[8][3];
	loadfeMuMatrix(feMu);
	
	bool fiction = false;
	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= nv) {
		fiction = true;
		return;
	}

	VertexFlags vflag;
	if (!fiction) {
		vflag = vflags[tid];
		fiction = vflag.is_fiction();
	}

	int vid = tid;
	GridVertexIndex indexer(reso, reso, reso);
	indexer.locate(vid, vflag.get_gscolor(), gGsVertexEnd);

	short3 vpos = indexer.getPos();
	float fchar = 0.;
	do {
		if (vflag.is_period_padding() || vflag.is_fiction()) break;
		for (int ei = 0; ei < 8; ei++) {
			int neighEid = indexer.neighElement(ei, gGsCellEnd, gGsCellReso).getId();
			if (neighEid == -1) continue;
			CellFlags eflag = eflags[neighEid];
			float rho_penal = rhoPenalMin + rholist[neighEid];
			int kirow = 7 - ei;
			fchar += rho_penal * feMu[kirow][istrain];
		}
	} while (0);
	fcharlist[0][vid] = fchar;
}

void homo::Grid_H::enforce_unit_macro_strain_host(int istrain)
{
	useGrid_g();
	cuda_error_check;
	VertexFlags* vflags = vertflag;
	CellFlags* eflags = cellflag;
	size_t grid_size, block_size;
	// for blocks do below
	int blockx, blocky, blockz;
	blockx = cellReso[0] / MIN_TRANSFER;
	blocky = cellReso[1] / MIN_TRANSFER;
	blockz = cellReso[2] / MIN_TRANSFER;
	int block_num = blockx * blocky * blockz;
	use_block_rhogs(0);
	for (int i = 0; i < block_num - 1; i++) {
		cudaDeviceSynchronize();
		if (i >= 1) {
			write_block_f_ggs(i - 1);
		}
		use_block_rhogs(i + 1);
		float* tmp;
		if (current) {
			tmp = getMem().getBuffer("temp_rho0")->data<float>();
		}
		else {
			tmp = getMem().getBuffer("temp_rho1")->data<float>();
		}
		update_hostgs(tmp);
		devArray_t<VT*, 1> fcharlist{ f_g[current] };
		make_kernel_param(&grid_size, &block_size, n_gsvertices(), 256);
		enforce_unit_macro_strain_kernel_H << <grid_size, block_size, 0, stream[current] >> > (n_gsvertices(), istrain, fcharlist, vflags, eflags, rho_g, MIN_TRANSFER);
		std::swap(current, next);
	}
	cudaDeviceSynchronize();
	write_block_f_ggs(block_num - 2);

	float* tmp;
	if (current) {
		tmp = getMem().getBuffer("temp_rho0")->data<float>();
	}
	else {
		tmp = getMem().getBuffer("temp_rho1")->data<float>();
	}
	update_hostgs(tmp);
	devArray_t<VT*, 1> fcharlist{ f_g[current] };
	make_kernel_param(&grid_size, &block_size, n_gsvertices(), 256);
	enforce_unit_macro_strain_kernel_H << <grid_size, block_size, 0, stream[current] >> > (n_gsvertices(), istrain, fcharlist, vflags, eflags, rho_g, MIN_TRANSFER);
	write_block_f_ggs(block_num - 1, false);
	cudaDeviceSynchronize();
	std::swap(current, next);
}
void homo::Grid_H::enforce_unit_macro_strain(int istrain)
{
	useGrid_g();
	cuda_error_check;
	VertexFlags* vflags = vertflag;
	CellFlags* eflags = cellflag;
	size_t grid_size, block_size;
	devArray_t<VT*, 1> fcharlist{ f_g[0] };
	make_kernel_param(&grid_size, &block_size, n_gsvertices(), 256);
	enforce_unit_macro_strain_kernel_H << <grid_size, block_size >> > (n_gsvertices(), istrain, fcharlist, vflags, eflags, rho_g, cellReso[0]);
	cudaDeviceSynchronize();
	cuda_error_check;
}

float Grid_H::v_norm(VT* v, bool removePeriodDof /*= false*/, int len /*= -1*/)
{
	if (len < 0) len = n_gsvertices();
	//auto buffer = getTempPool().getBuffer(sizeof(double) * (len / 100));
	if (!removePeriodDof) {
		double nrm = norm(v, len, (float*)(0));
		cuda_error_check;
		return nrm;
	}
	else {
		double n2 = v_dot(v, v, removePeriodDof);
		return sqrt(n2);
	}
}

void Grid_H::v_reset(VT* v, int len)
{
	if (len < 0) len = n_gsvertices();
	cudaMemset(v, 0, sizeof(VT) * len);
	cudaDeviceSynchronize();
}
template<typename T>
struct constVec {
	T val;
	constVec(T val_) : val(val_) {}
	__device__ constVec(const constVec& v2) = default;
	__device__ T operator()(size_t k) {
		return val;
	}
	__device__ T operator[](size_t k) const {
		return val;
	}
};


// scatter per fine element matrix to coarse stencil
// stencil was organized in lexico order(No padding), and should be transferred to gs order
//template<int BlockSize = 256>
//__global__ void restrict_stencil_otf_kernel_1(
//	int ne, float* rholist, CellFlags* eflags,
//	devArray_t<int, 8> gsCellEnd, devArray_t<int, 3> CoarseCellReso
void homo::Grid_H::restrict_stencil(void)
{ 
	if (is_root) return;
	if (fine->assemb_otf && !fine->use_host_memory) {
		useGrid_g();
		size_t grid_size, block_size;
		for (int i = 0; i < 27; i++) {
			cudaMemset(stencil_g[i], 0, sizeof(VT) * n_gsvertices());
		}
		cudaDeviceSynchronize();
		cuda_error_check;
		int nv = (cellReso[0] + 1) * (cellReso[1]+1) * (cellReso[2]+1);
		make_kernel_param(&grid_size, &block_size, nv, 256);
		restrict_stencil_otf_aos_kernel_1_H << <grid_size, block_size >> > (nv, fine->rho_g, cellflag, vertflag);
		cudaDeviceSynchronize();
		cuda_error_check;
		useGrid_g();
		lexiStencil2gsorder();
		enforce_period_stencil(true);
	}
	else if (fine->assemb_otf && fine->use_host_memory)
	{
		// give the true value to gpu vector
		useGrid_g();
		size_t grid_size, block_size;
		for (int i = 0; i < 27; i++) {
			cudaMemset(stencil_g[i], 0, sizeof(VT) * n_gsvertices());
		}
		cudaDeviceSynchronize();
		cuda_error_check;
		int ne_block = MIN_TRANSFER * MIN_TRANSFER * MIN_TRANSFER;

		int blockx, blocky, blockz;
		blockx = fine->cellReso[0] / MIN_TRANSFER;
		blocky = fine->cellReso[1] / MIN_TRANSFER;
		blockz = fine->cellReso[2] / MIN_TRANSFER;
		clock_t start = clock();

		for (int i = 0; i < blockx * blocky * blockz; i++) {
			fine->useGrid_g();
			fine->use_block_rho(i);
			float* tmp = getMem().getBuffer("temp_rho0")->data<VT>();
			fine->update_host(tmp);
			cuda_error_check;
			useGrid_g();
			int bx = i % blockx;
			int by = (i / blockx) % blocky;
			int bz = i / (blockx * blocky);
			// we need one thread one fine->cell
			make_kernel_param(&grid_size, &block_size, ne_block, 256);
			restrict_stencil_otf_aos_kernel_host_H << <grid_size, block_size >> > (ne_block, fine->rho_g, fine->cellflag, fine->vertflag, bx, by, bz, MIN_TRANSFER);
			cudaDeviceSynchronize();
			cuda_error_check;
		}

		clock_t end = clock();
		double duration = (double)(end - start) * 1000.0 / CLOCKS_PER_SEC;
		std::cout << "CPU ??: " << duration << " ms\n";
		useGrid_g();
		lexiStencil2gsorder();
		enforce_period_stencil(true);
	}
	else
	{
		useGrid_g();
		cudaDeviceSynchronize();
		cuda_error_check;
		int nvfine = fine->n_gsvertices();
		//printf("--\n");
		size_t grid_size, block_size;
		make_kernel_param(&grid_size, &block_size, n_gsvertices(), 256);
		//restrict_stencil_kernel_1 << <grid_size, block_size >> > (n_gsvertices(), nvfine, vertflag, fine->vertflag);
		restrict_stencil_aos_kernel_1_H << <grid_size, block_size >> > (n_gsvertices(), nvfine, vertflag, fine->vertflag);
		cudaDeviceSynchronize();
		cuda_error_check;
		//stencil2matlab("Khost");
		enforce_period_stencil(false);
	}
}

void uploadTemplaceMatrix_H(const double* ke, float penal, const float* feMu, const float* disp) {
	float fke[8 * 8];
	for (int i = 0; i < 8 * 8; i++) fke[i] = ke[i];
	for (int i = 0; i < 24; i++)
		std::cout << feMu[i] << std::endl;
	cudaMemcpyToSymbol(gKE_H, fke, sizeof(gKE_H));
	cudaMemcpyToSymbol(gKEd_H, ke, sizeof(gKEd_H));
	cudaMemcpyToSymbol(exp_penal, &penal, sizeof(float));
	cudaMemcpyToSymbol(gfeMu_H, feMu, sizeof(gfeMu_H));
	cudaMemcpyToSymbol(gDisp_H, disp, sizeof(gDisp_H));
}

template<typename T>
__global__ void lexi2gsorder_kernel(T* src, T* dst, 
	devArray_t<int, 3> srcreso, devArray_t<int, 8> gsEnd,
	bool srcpaded = false
) {
	size_t n_src = srcreso[0] * srcreso[1] * srcreso[2];
	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= n_src) return;
	// padded src pos
	int srcpos[3] = { tid % srcreso[0] , tid / srcreso[0] % srcreso[1] , tid / (srcreso[0] * srcreso[1]) };
	// if not padding, add padding
	if (!srcpaded) {
		srcpos[0] += 1; srcpos[1] += 1; srcpos[2] += 1;
	}
	int gsorg[3] = { srcpos[0] % 2, srcpos[1] % 2, srcpos[2] % 2 };
	int gscolor = gsorg[0] + gsorg[1] * 2 + gsorg[2] * 4;
	
	int gsreso[3] = {};
	for (int k = 0; k < 3; k++) {
		// last index - org / 2, should padd 1
		gsreso[k] = (srcreso[k] + 1 - gsorg[k]) / 2 + 1;
	}

	int setpos[3] = { srcpos[0] / 2, srcpos[1] / 2, srcpos[2] / 2 };
	int setid = setpos[0] + setpos[1] * gsreso[0] + setpos[2] * gsreso[0] * gsreso[1];

	int gsid = setid + (gscolor == 0 ? 0 : gsEnd[gscolor - 1]);

	dst[gsid] = src[tid];
}

template <typename T>
void lexi2gsorder_imp(T* src, T* dst, Grid_H::LexiType type_,
	std::array<int, 3> cellReso, int gsVertexSetEnd[8],
	int gsCellSetEnd[8], bool lexipadded /*= false*/)
{
	if (type_ == Grid_H::VERTEX) {
		devArray_t<int, 3> reso{ cellReso[0] + 1, cellReso[1] + 1, cellReso[2] + 1 };
		devArray_t<int, 8>  gsend;
		int nv = reso[0] * reso[1] * reso[2];
		for (int k = 0; k < 8; k++) gsend[k] = gsVertexSetEnd[k];
		size_t grid_size, block_size;
		make_kernel_param(&grid_size, &block_size, nv, 256);
		lexi2gsorder_kernel << <grid_size, block_size >> > (src, dst, reso, gsend, lexipadded);
		cudaDeviceSynchronize();
		cuda_error_check;
	}
	else if (type_ == Grid_H::CELL) {
		devArray_t<int, 3> reso{ cellReso[0] , cellReso[1] , cellReso[2] };
		devArray_t<int, 8> gsend;
		int ne = reso[0] * reso[1] * reso[2];
		for (int k = 0; k < 8; k++) gsend[k] = gsCellSetEnd[k];
		size_t grid_size, block_size;
		make_kernel_param(&grid_size, &block_size, ne, 256);
		lexi2gsorder_kernel << <grid_size, block_size >> > (src, dst, reso, gsend, lexipadded);
		cudaDeviceSynchronize();
		cuda_error_check;
	}
}

void homo::Grid_H::lexi2gsorder(float* src, float* dst, LexiType type_, bool lexipadded /*= false*/)
{
	lexi2gsorder_imp(src, dst, type_, cellReso, gsVertexSetEnd, gsCellSetEnd, lexipadded);
}

void homo::Grid_H::lexi2gsorder(half* src, half* dst, LexiType type_, bool lexipadded /*= false*/)
{
	lexi2gsorder_imp(src, dst, type_, cellReso, gsVertexSetEnd, gsCellSetEnd, lexipadded);
}
void homo::Grid_H::lexi2gsorder(glm::hmat3* src, glm::hmat3* dst, LexiType type_, bool lexipadded /*= false*/)
{
	lexi2gsorder_imp(src, dst, type_, cellReso, gsVertexSetEnd, gsCellSetEnd, lexipadded);
}

void homo::Grid_H::lexiStencil2gsorder(void)
{
	auto tmpname = getMem().addBuffer(n_gsvertices() * sizeof(VT));
	VT* tmp = getMem().getBuffer(tmpname)->data<VT>();
	for (int i = 0; i < 27; i++) {
		cudaMemset(tmp, 0, sizeof(VT) * n_gsvertices());
		cudaDeviceSynchronize();
		cuda_error_check;
		lexi2gsorder(stencil_g[i], tmp, VERTEX);
		cudaMemcpy(stencil_g[i], tmp, sizeof(VT) * n_gsvertices(), cudaMemcpyDeviceToDevice);
	}
	getMem().deleteBuffer(tmpname);
	cuda_error_check;
}

template<int BlockSize = 256>
__global__ void enforce_period_stencil_subst_kernel(void) {
	
	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	int vreso[3] = { gGridCellReso[0] + 1,gGridCellReso[1] + 1, gGridCellReso[2] + 1 };

	int gsid_min = -1;
	int gsid_max = -1;

	do {
		// down - up
		int du_end = vreso[0] * vreso[1];
		if (tid < du_end) {
			int vid = tid;
			int pos[3] = { vid % vreso[0], vid / vreso[0], 0 };
			gsid_min = lexi2gs(pos, gGsVertexReso, gGsVertexEnd);
			pos[2] = vreso[2] - 1;
			gsid_max = lexi2gs(pos, gGsVertexReso, gGsVertexEnd);
			break;
		}

		// left - right
		int lr_end = du_end + vreso[1] * vreso[2];
		if (tid < lr_end) {
			int vid = tid - du_end;
			int pos[3] = { 0, vid % vreso[1], vid / vreso[1] };
			gsid_min = lexi2gs(pos, gGsVertexReso, gGsVertexEnd);
			pos[0] = vreso[0] - 1;
			gsid_max = lexi2gs(pos, gGsVertexReso, gGsVertexEnd);
			break;
		}

		// near - far
		int nf_end = lr_end + vreso[0] * vreso[2];
		if (tid < nf_end) {
			int vid = tid - lr_end;
			int pos[3] = { vid % vreso[0], 0 , vid / vreso[0] };
			gsid_min = lexi2gs(pos, gGsVertexReso, gGsVertexEnd);
			pos[1] = vreso[1] - 1;
			gsid_max = lexi2gs(pos, gGsVertexReso, gGsVertexEnd);
			break;
		}
	} while (0);

	if (gsid_min != -1 && gsid_max != -1) {
		for (int i = 0; i < 27; i++) {
			for (int j = 0; j < 9; j++) {
				rxstencil[i][j][gsid_max] = rxstencil[i][j][gsid_min];
			}
		}
	}
}

template<typename T, int N>
__global__ void pad_vertex_data_kernel(int nvfacepad, int nvedgepadd, devArray_t<T*, N> v, VertexFlags* vflags);

template <typename T, int N>
void pad_vertex_data_imp(T **v, std::array<int, 3> cellReso, VertexFlags* vertflag) {
	int nvpadface = (cellReso[0] + 1) * (cellReso[1] + 1) +
		(cellReso[1] + 1) * (cellReso[2] + 1) +
		(cellReso[0] + 1) * (cellReso[2] + 1);
	int nvpadedge = 2 * (
		(cellReso[0] + 3) * (cellReso[1] + 3) - (cellReso[0] + 1) * (cellReso[1] + 1)) +
		4 * (cellReso[2] + 1);
	devArray_t<T*, N> arr(v);
	size_t grid_size, block_size;
	make_kernel_param(&grid_size, &block_size, nvpadface + nvpadedge, 256);
	pad_vertex_data_kernel << <grid_size, block_size >> > (nvpadface, nvpadedge, arr, vertflag);
	cudaDeviceSynchronize();
	cuda_error_check;
}

void homo::Grid_H::enforce_period_stencil(bool additive)
{
	for (int i = 0; i < 27; i++) {
		float* tr[1] = { stencil_g[i] };
		enforce_period_vertex(tr, additive);
	}
	if (fine->is_root) {
		restrict_stencil_arround_dirichelt_boundary();
	}
	pad_vertex_data_imp<VT, 27>(stencil_g, cellReso, vertflag);
}

template <typename Flag>
__global__ void gsid2pos_kernel(int n, Flag *flags, devArray_t<int *, 3> pos, int off = -1) {
	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= n) { return; }
	constexpr bool isVertexId = std::is_same_v<std::decay_t<Flag>, VertexFlags>;
	constexpr bool isCellId = std::is_same_v<std::decay_t<Flag>, CellFlags>;
	if (isVertexId) {
		Flag vflag = flags[tid];
		if (!vflag.is_fiction()) {
			int p[3];
			gsid2pos(tid, vflag.get_gscolor(), gGsVertexReso, gGsVertexEnd, p);
			if (off < 0 || off > 2)
				for (int i = 0; i < 3; i++) pos[i][tid] = p[i];
			else
				pos[off][tid] = p[off];
		}
	} // vertex id
	else if (isCellId) {
		Flag eflag = flags[tid];
		if (!eflag.is_fiction()) {
			int p[3];
			gsid2pos(tid, eflag.get_gscolor(), gGsCellReso, gGsCellEnd, p);
			if (off < 0 || off > 2)
				for (int i = 0; i < 3; i++) pos[i][tid] = p[i];
			else
				pos[off][tid] = p[off];
		}
	} // element id
	else {
		if (tid == 0) {
			printf("\033[31mno such flags type at grid.cu, line %d\033[0m\n", __LINE__);
		}
	}
}

__global__ void testIndexerNeigh_kernel(int nv, int neigh, VertexFlags* vflags, devArray_t<int*, 3> pos) {
	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= nv) { return; }
	VertexFlags vflag = vflags[tid];
	GridVertexIndex indexer(gGridCellReso[0], gGridCellReso[1], gGridCellReso[2]);
	if (!vflag.is_fiction()) {
		bool nofiction = indexer.locate(tid, vflag.get_gscolor(), gGsVertexEnd);
		int neighid = -2;
		int neicolor = -1;
		if (nofiction) {
			auto idc = indexer.neighVertex(neigh, gGsVertexEnd, gGsVertexReso);
			neighid = idc.getId();
			neicolor = idc.getColor();
		}
		if (neighid >= nv || neighid < -1) {
			printf("error%d\n", __LINE__);
		}
		int p[3] = { -2,-2,-2 };
		if (neighid != -1) {
			gsid2pos(neighid, neicolor, gGsVertexReso, gGsVertexEnd, p);
		}
		for (int i = 0; i < 3; i++) { pos[i][tid] = p[i]; }
	}
}

__global__ void testIndexerNeighElement_kernel(int nv, int ne, int neigh, VertexFlags* vflags, devArray_t<int*, 3> pos) {
	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= nv) { return; }
	VertexFlags vflag = vflags[tid];
	GridVertexIndex indexer(gGridCellReso[0], gGridCellReso[1], gGridCellReso[2]);
	if (!vflag.is_fiction()) {
		bool nofiction = indexer.locate(tid, vflag.get_gscolor(), gGsVertexEnd);
		int neighid = -2;
		int neicolor = -1;
		if (nofiction) {
			auto idc = indexer.neighElement(neigh, gGsCellEnd, gGsCellReso);
			neighid = idc.getId();
			neicolor = idc.getColor();
		}
		if (neighid >= ne || neighid < -1) {
			printf("error%d\n", __LINE__);
		}
		int p[3] = { -2,-2,-2 };
		if (neighid != -1) {
			gsid2pos(neighid, neicolor, gGsCellReso, gGsCellEnd, p);
		}
		for (int i = 0; i < 3; i++) { pos[i][tid] = p[i]; }
	}
}

__global__ void testIndexerNeighCoarseVertex_kernel(int nv, int ne, int neigh, VertexFlags* vflags, devArray_t<int*, 3> pos) {
	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= nv) { return; }
	VertexFlags vflag = vflags[tid];
	GridVertexIndex indexer(gGridCellReso[0], gGridCellReso[1], gGridCellReso[2]);
	if (!vflag.is_fiction() && !vflag.is_period_padding()) {
		bool nofiction = indexer.locate(tid, vflag.get_gscolor(), gGsVertexEnd);
		int neighid = -2;
		int neicolor = -1;
		if (nofiction) {
			int rem[3];
			auto idc = indexer.neighCoarseVertex(neigh, gDownCoarse, gGsCoarseVertexEnd, gGsCoarseVertexReso, rem);
			neighid = idc.getId();
			neicolor = idc.getColor();
			// DEBUG
			//if (tid == 3554) {
			//	printf("neigh = %d, id = %d, coarse = (%d, %d, %d), rem = (%d, %d, %d)\n",
			//		neigh, neighid, gDownCoarse[0], gDownCoarse[1], gDownCoarse[2], rem[0], rem[1], rem[2]);
			//}
		}
		if (neighid >= ne || neighid < -1) {
			printf("error%d\n", __LINE__);
		}
		int p[3] = { -2,-2,-2 };
		if (neighid != -1) {
			gsid2pos(neighid, neicolor, gGsCoarseVertexReso, gGsCoarseVertexEnd, p);
			for (int i = 0; i < 3; i++) p[i] *= gDownCoarse[i];
		}
		for (int i = 0; i < 3; i++) { pos[i][tid] = p[i]; }
		//if (neigh == 2 && p[0] == 0 && p[1] == 0 && p[2] == 0) {
		//	printf("neigh_id = %d \n", neighid);
		//}
	}
	else {
		for (int i = 0; i < 3; i++) { pos[i][tid] = -2; }
	}
}

__global__ void testIndexerNeighFineVertex_kernel(int nv, int neigh, VertexFlags* vflags, devArray_t<int*, 3> pos) {
	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= nv) { return; }
	VertexFlags vflag = vflags[tid];
	GridVertexIndex indexer(gGridCellReso[0], gGridCellReso[1], gGridCellReso[2]);
	if (!vflag.is_fiction() && !vflag.is_period_padding()) {
		bool nofiction = indexer.locate(tid, vflag.get_gscolor(), gGsVertexEnd);
		int neighid = -2;
		int neicolor = -1;
		if (nofiction) {
			int off[3] = { neigh % 3 - 1, neigh / 3 % 3 - 1 , neigh / 9 - 1 };
			bool nondya = gUpCoarse[0] > 2 || gUpCoarse[1] > 2 || gUpCoarse[2] > 2;
			auto idc = indexer.neighFineVertex(off, gUpCoarse, gGsFineVertexEnd, gGsFineVertexReso, nondya);
			neighid = idc.getId();
			neicolor = idc.getColor();
		}
		//if (neighid >= ne || neighid < -1) {
		//	printf("error%d\n", __LINE__);
		//}
		int p[3] = { -2,-2,-2 };
		if (neighid != -1) {
			gsid2pos(neighid, neicolor, gGsFineVertexReso, gGsFineVertexEnd, p);
		}
		for (int i = 0; i < 3; i++) { pos[i][tid] = p[i]; }
		//if (neigh == 2 && p[0] == 0 && p[1] == 0 && p[2] == 0) {
		//	printf("neigh_id = %d \n", neighid);
		//}
	}
	else {
		for (int i = 0; i < 3; i++) { pos[i][tid] = -2; }
	}
}
void homo::Grid_H::testIndexer(void) {
	useGrid_g();
	devArray_t<int*, 3> pos;
	for (int i = 0; i < 3; i++) {
		pos[i] = getMem().getBuffer(getMem().addBuffer(sizeof(int) * n_gsvertices()))->data<int>();
		init_array(pos[i], -2, n_gsvertices());
	}
	std::vector<int> hostpos[3];
	size_t grid_size, block_size;
	make_kernel_param(&grid_size, &block_size, n_gsvertices(), 256);
	VertexFlags* vflags = vertflag;

	// ...
	Coarse->getGsVertexPos(hostpos);
	homoutils::writeVectors(getPath("coarsegspos"), hostpos);

	getGsVertexPos(hostpos);
	homoutils::writeVectors(getPath("gspos"), hostpos);

	// ...
	if (0) {
		for (int i = 0; i < 27; i++) {
			cudaDeviceSynchronize();
			cuda_error_check;
			testIndexerNeigh_kernel << <grid_size, block_size >> > (n_gsvertices(), i, vflags, pos);
			cudaDeviceSynchronize();
			cuda_error_check;
			for (int k = 0; k < 3; k++) {
				hostpos[k].resize(n_gsvertices());
				cudaMemcpy(hostpos[k].data(), pos[k], sizeof(int) * n_gsvertices(), cudaMemcpyDeviceToHost);
			}
			char buf[100];
			sprintf_s(buf, "./neigh%d", i);
			homoutils::writeVectors(getPath(buf), hostpos);
		}
	}
	if (0) {
		for (int i = 0; i < 8; i++) {
			cudaDeviceSynchronize();
			cuda_error_check;
			testIndexerNeighElement_kernel << <grid_size, block_size >> > (n_gsvertices(), n_gscells(), i, vflags, pos);
			cudaDeviceSynchronize();
			cuda_error_check;
			for (int k = 0; k < 3; k++) {
				hostpos[k].resize(n_gsvertices());
				cudaMemcpy(hostpos[k].data(), pos[k], sizeof(int) * n_gsvertices(), cudaMemcpyDeviceToHost);
			}
			char buf[100];
			sprintf_s(buf, "./neigh%d", i);
			homoutils::writeVectors(getPath(buf), hostpos);
		}
	}
	if (1) {
		for (int i = 0; i < 3; i++) cudaMemset(pos[i], 0, sizeof(int) * n_gsvertices());
		for (int i = 0; i < 8; i++) {
			cudaDeviceSynchronize();
			cuda_error_check;
			testIndexerNeighCoarseVertex_kernel << <grid_size, block_size >> > (n_gsvertices(), n_gscells(), i, vflags, pos);
			cudaDeviceSynchronize();
			cuda_error_check;
			for (int k = 0; k < 3; k++) {
				hostpos[k].resize(n_gsvertices());
				cudaMemcpy(hostpos[k].data(), pos[k], sizeof(int) * n_gsvertices(), cudaMemcpyDeviceToHost);
			}
			char buf[100];
			sprintf_s(buf, "./coarseneigh%d", i);
			homoutils::writeVectors(getPath(buf), hostpos);
		}
	}
	if (1) {
		Coarse->useGrid_g();
		for (int i = 0; i < 3; i++) cudaMemset(pos[i], 0, sizeof(int) * n_gsvertices());
		for (int i = 0; i < 27; i++) {
			cudaDeviceSynchronize();
			cuda_error_check;
			testIndexerNeighFineVertex_kernel << <grid_size, block_size >> > (Coarse->n_gsvertices(), i, Coarse->vertflag, pos);
			cudaDeviceSynchronize();
			cuda_error_check;
			for (int k = 0; k < 3; k++) {
				hostpos[k].resize(Coarse->n_gsvertices());
				cudaMemcpy(hostpos[k].data(), pos[k], sizeof(int) * Coarse->n_gsvertices(), cudaMemcpyDeviceToHost);
			}
			char buf[100];
			sprintf_s(buf, "./fineneigh%d", i);
			homoutils::writeVectors(getPath(buf), hostpos);
		}
	}

	for (int i = 0; i < 3; i++) {
		getMem().deleteBuffer(pos[i]);
	}
}

void homo::Grid_H::getGsVertexPos(std::vector<int> hostpos[3])
{
	useGrid_g();
	devArray_t<int*, 3> pos;
	size_t grid_size, block_size;
	make_kernel_param(&grid_size, &block_size, n_gsvertices(), 256);
	for (int i = 0; i < 3; i++) {
		// auto buffer = getTempBuffer(sizeof(int) * n_gsvertices());
		// pos[i] = buffer.data<int>();
		cudaMallocManaged(&pos[i], sizeof(int) * n_gsvertices());
		init_array(pos[i], -2, n_gsvertices());
		// ...
		gsid2pos_kernel << <grid_size, block_size >> > (n_gsvertices(), vertflag, pos, i);
		cudaDeviceSynchronize();
		cuda_error_check;
		hostpos[i].resize(n_gsvertices());
		cudaMemcpy(hostpos[i].data(), pos[i], sizeof(int) * n_gsvertices(), cudaMemcpyDeviceToHost);
		cudaFree(pos[i]);
	}
}

void homo::Grid_H::getGsElementPos(std::vector<int> hostpos[3])
{
	useGrid_g();
	devArray_t<int*, 3> pos;
	size_t grid_size, block_size;
	make_kernel_param(&grid_size, &block_size, n_gscells(), 256);
	for (int i = 0; i < 3; i++) {
		// auto buffer = getTempBuffer(sizeof(int) * n_gscells());
		// pos[i] = buffer.data<int>();
		cudaMallocManaged(&pos[i], sizeof(int) * n_gscells());
		init_array(pos[i], -2, n_gscells());
		// ...
		gsid2pos_kernel << <grid_size, block_size >> > (n_gscells(), cellflag, pos, i);
		cudaDeviceSynchronize();
		cuda_error_check;
		hostpos[i].resize(n_gscells());
		cudaMemcpy(hostpos[i].data(), pos[i], sizeof(int) * n_gscells(), cudaMemcpyDeviceToHost);
		cudaFree(pos[i]);
	}
}

void homo::Grid_H::getDensity(std::vector<float>& rho, bool lexiOrder /*= false*/)
{
	rho.resize(n_gscells());
	if (std::is_same_v<float, RhoT>) {
		cudaMemcpy(rho.data(), rho_g, sizeof(float) * rho.size(), cudaMemcpyDeviceToHost);
	}
	else {
		auto tmpbuf = getTempBuffer(sizeof(float) * n_gscells());
		auto* tmpdata = tmpbuf.data<float>();
		type_cast(tmpdata, rho_g, rho.size());
		cudaMemcpy(rho.data(), tmpdata, sizeof(float) * rho.size(), cudaMemcpyDeviceToHost);
	}
	if (!lexiOrder)
		return;
	else
		throw std::runtime_error("not implemented"); // toDO
}

void homo::Grid_H::test(void)
{
	testIndexer();
	exit(0);
}

template <typename T, int N>
__global__ void enforce_period_boundary_vertex_kernel(int siz, devArray_t<T *, N> v, VertexFlags *vflags, bool additive = false)
{
	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= siz) return;
	int pos[3] = { -2,-2,-2 };
	int ereso[3] = { gGridCellReso[0],gGridCellReso[1],gGridCellReso[2] };

	//bool debug = false;

	do {
		if (tid < ereso[0] * ereso[1]) {
			pos[0] = tid % ereso[0];
			pos[1] = tid / ereso[0];
			pos[2] = 0;
			break;
		}
		tid -= ereso[0] * ereso[1];
		if (tid < ereso[1] *(ereso[2]-1)) {
			pos[0] = 0;
			pos[1] = tid % ereso[1];
			pos[2] = tid / ereso[1] + 1;
			break;
		}
		tid -= ereso[1] * (ereso[2] - 1);
		if (tid < (ereso[0] - 1) * (ereso[2] - 1)) {
			pos[0] = tid % (ereso[0] - 1) + 1;
			pos[1] = 0;
			pos[2] = tid / (ereso[0] - 1) + 1;
			break;
		}
	} while (0);
	if (pos[0] <= -2 || pos[1] <= -2 || pos[2] <= -2) return;

	//if (pos[0] == 0 && pos[1] == 7 && pos[2] == 0) debug = true;

	int gsid = lexi2gs(pos, gGsVertexReso, gGsVertexEnd);
	VertexFlags vflag = vflags[gsid];
	T val[N] = { /*v[0][gsid],v[1][gsid],v[2][gsid]*/ };
	int op_ids[8] = { -1 ,-1,-1,-1, -1 ,-1,-1,-1 };
	{
		// sum opposite 
		//if (debug) printf("vflag = %04x  gsid = %d   edid = %d\n", vflag.flagbits, gsid, gGsVertexEnd[vflag.get_gscolor()]);

		int op_pos[3];
		for (int i = 0; i < vflag.is_set(LEFT_BOUNDARY) + 1; i++) {
			op_pos[0] = pos[0];
			if (i) op_pos[0] += ereso[0];
			for (int j = 0; j < vflag.is_set(NEAR_BOUNDARY) + 1; j++) {
				op_pos[1] = pos[1];
				if (j) op_pos[1] += ereso[1];
				for (int k = 0; k < vflag.is_set(DOWN_BOUNDARY) + 1; k++) {
					op_pos[2] = pos[2];
					if (k) op_pos[2] += ereso[2];
					int op_id = lexi2gs(op_pos, gGsVertexReso, gGsVertexEnd);
					op_ids[i * 4 + j * 2 + k] = op_id;
					if (additive) for (int m = 0; m < N; m++) val[m] += v[m][op_id];
				}
			}
		}

	}

	// enforce period boundary
	for (int i = 0; i < 8; i++) {
		if (op_ids[i] != -1) {
			if (additive)
				for (int j = 0; j < N; j++) v[j][op_ids[i]] = val[j];
			else 
				for (int j = 0; j < N; j++) v[j][op_ids[i]] = v[j][gsid];
		}
	}
	if (additive) { for (int j = 0; j < N; j++) v[j][gsid] = val[j]; }
}

template <typename T, int N>
void enforce_period_vertex_imp(T** v, std::array<int, 3> cellReso, VertexFlags* vertflag, bool additive /*= false*/)
{
	int nvdup = cellReso[0] * cellReso[1]
		+ cellReso[1] * (cellReso[2] - 1)
		+ (cellReso[0] - 1) * (cellReso[2] - 1);
	devArray_t<T*, N> varr(v);
	size_t grid_size, block_size;
	make_kernel_param(&grid_size, &block_size, nvdup, 256);
	enforce_period_boundary_vertex_kernel << <grid_size, block_size >> > (nvdup, varr, vertflag, additive);
	cudaDeviceSynchronize();
	cuda_error_check;
}

void homo::Grid_H::enforce_period_vertex(double* v[1], bool additive /*= false*/)
{
	enforce_period_vertex_imp<double, 1>(v, cellReso, vertflag, additive);
}

void homo::Grid_H::enforce_period_vertex(half* v[1], bool additive /*= false*/)
{
	enforce_period_vertex_imp<half, 1>(v, cellReso, vertflag, additive);
}

void homo::Grid_H::enforce_period_vertex(float* v[1], bool additive /*= false*/) {
	enforce_period_vertex_imp<float, 1>(v, cellReso, vertflag, additive);
}


template<typename T, int N>
__global__ void pad_vertex_data_kernel(int nvfacepad, int nvedgepadd, devArray_t<T*, N> v, VertexFlags* vflags) {
	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= nvfacepad + nvedgepadd) return;
	int ereso[3] = { gGridCellReso[0],gGridCellReso[1],gGridCellReso[2] };

	//bool debug = false;

	int boundaryType = -1;

	if (tid < nvfacepad) {
		int pos[3] = { -2,-2,-2 };
		// padd face
		do {
			if (tid < (ereso[0] + 1) * (ereso[1] + 1)) {
				pos[0] = tid % (ereso[0] + 1);
				pos[1] = tid / (ereso[0] + 1);
				pos[2] = 0;
				boundaryType = 0;
				break;
			}
			tid -= (ereso[0] + 1) * (ereso[1] + 1);
			if (tid < (ereso[1] + 1) * (ereso[2] + 1)) {
				pos[0] = 0;
				pos[1] = tid % (ereso[1] + 1);
				pos[2] = tid / (ereso[1] + 1);
				boundaryType = 1;
				break;
			}
			tid -= (ereso[1] + 1) * (ereso[2] + 1);
			if (tid < (ereso[0] + 1) * (ereso[2] + 1)) {
				pos[0] = tid % (ereso[0] + 1);
				pos[1] = 0;
				pos[2] = tid / (ereso[0] + 1);
				boundaryType = 2;
				break;
			}
		} while (0);
		if (pos[0] <= -2 || pos[1] <= -2 || pos[2] <= -2) return;

		int gsid = lexi2gs(pos, gGsVertexReso, gGsVertexEnd);
		VertexFlags vflag = vflags[gsid];	// padding 
		if (boundaryType == 1) {
			for (int i : {-1, 1}) {
				int p[3] = { pos[0] + i, pos[1], pos[2] };
				int q[3] = { pos[0] + ereso[0] + i, pos[1], pos[2] };
				int pid = lexi2gs(p, gGsVertexReso, gGsVertexEnd);
				int qid = lexi2gs(q, gGsVertexReso, gGsVertexEnd);
				if (i == -1) for (int j = 0; j < N; j++) v[j][pid] = v[j][qid];
				if (i == 1) for (int j = 0; j < N; j++) v[j][qid] = v[j][pid];
			}
		}
		if (boundaryType == 2) {
			for (int i : {-1, 1}) {
				int p[3] = { pos[0] , pos[1] + i, pos[2] };
				int q[3] = { pos[0] , pos[1] + ereso[1] + i, pos[2] };
				int pid = lexi2gs(p, gGsVertexReso, gGsVertexEnd);
				int qid = lexi2gs(q, gGsVertexReso, gGsVertexEnd);
				if (i == -1) for (int j = 0; j < N; j++) v[j][pid] = v[j][qid];
				if (i == 1) for (int j = 0; j < N; j++) v[j][qid] = v[j][pid];
			}
		}
		if (boundaryType == 0) {
			for (int i : {-1, 1}) {
				int p[3] = { pos[0] , pos[1] , pos[2] + i };
				int q[3] = { pos[0] , pos[1] , pos[2] + ereso[2] + i };
				int pid = lexi2gs(p, gGsVertexReso, gGsVertexEnd);
				int qid = lexi2gs(q, gGsVertexReso, gGsVertexEnd);
				//if (debug) {
				//	printf("i = %d  p = (%d %d %d)  q = (%d %d %d)  pid = %d  qid = %d\n", i, p[0], p[1], p[2], q[0], q[1], q[2], pid, qid);
				//}
				if (i == -1) for (int j = 0; j < N; j++) v[j][pid] = v[j][qid];
				if (i == 1) for (int j = 0; j < N; j++) v[j][qid] = v[j][pid];
			}
			//if (debug) {
			//	for (int i : {-1, 1}) {
			//		int p[3] = { pos[0] , pos[1] , pos[2] + i };
			//		int q[3] = { pos[0] , pos[1] , pos[2] + ereso[2] + i };
			//		int pid = lexi2gs(p, gGsVertexReso, gGsVertexEnd);
			//		int qid = lexi2gs(q, gGsVertexReso, gGsVertexEnd);
			//		printf("i = %d  vp = (%e, %e, %e)  vq = (%e, %e, %e)\n", i, v[0][pid], v[1][pid], v[2][pid]
			//			, v[0][qid], v[1][qid], v[2][qid]);
			//	}
			//}
		}
	}
	else if (tid - nvfacepad < nvedgepadd) {
		bool debug = false;
		// padd edge
		int id = tid - nvfacepad;
		int nv_bot = (ereso[0] + 3) * (ereso[1] + 3) - (ereso[0] + 1) * (ereso[1] + 1);
		int po[3] = { 0,0,0 };
		if (id < 2 * nv_bot) {
			po[2] = id / nv_bot * (ereso[2] + 2);
			id = id % nv_bot;
			if (id < 2 * (ereso[0] + 3)) {
				po[0] = id % (ereso[0] + 3);
				po[1] = id / (ereso[0] + 3) * (ereso[1] + 2);
			} else {
				id -= 2 * (ereso[0] + 3);
				po[0] = id / (ereso[1] + 1) * (ereso[0] + 2);
				po[1] = id % (ereso[1] + 1) + 1;
			}
		}
		else {
			id -= 2 * nv_bot;
			int hid = id / (ereso[2] + 1);
			int vid = id % (ereso[2] + 1);
			po[0] = hid % 2 * (ereso[0] + 2);
			po[1] = hid / 2 * (ereso[1] + 2);
			po[2] = vid + 1;
		}
		po[0] -= 1; po[1] -= 1; po[2] -= 1;
		int op_pos[3];
		for (int i = 0; i < 3; i++) {
			op_pos[i] = (po[i] + ereso[i]) % ereso[i];
		}
		int myid = lexi2gs(po, gGsVertexReso, gGsVertexEnd);
		int opid = lexi2gs(op_pos, gGsVertexReso, gGsVertexEnd);
		for (int i = 0; i < N; i++) v[i][myid] = v[i][opid];
	}
	
}

__global__ void testVflags_kernel(int nv, VertexFlags* vflags) {
	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= nv) return;
	VertexFlags vflag = vflags[tid];

	if (vflag.is_fiction()) return;

	GridVertexIndex indexer(gGridCellReso[0], gGridCellReso[1], gGridCellReso[2]);

	indexer.locate(tid, vflag.get_gscolor(), gGsVertexEnd);
	auto pos = indexer.getPos();

	int p[3] = { pos.x,pos.y,pos.z };
	if (lexi2gs(p, gGsVertexReso, gGsVertexEnd, true) != tid) {
		print_exception;
	}

	if (vflag.is_set(LEFT_BOUNDARY)) {
		if (pos.x != 1) { print_exception; }
	}
	if (vflag.is_set(RIGHT_BOUNDARY)) {
		if (pos.x != gGridCellReso[0] + 1) print_exception;
	}
	if (vflag.is_set(NEAR_BOUNDARY)) {
		if (pos.y != 1)print_exception;
	}
	if (vflag.is_set(FAR_BOUNDARY)) {
		if (pos.y != gGridCellReso[1] + 1) print_exception;
	}
	if (vflag.is_set(DOWN_BOUNDARY)) {
		if (pos.z != 1) print_exception;
	}
	if (vflag.is_set(UP_BOUNDARY)) {
		if (pos.z != gGridCellReso[2] + 1) print_exception;
	}
}

void homo::Grid_H::testVflags(void)
{
	useGrid_g();
	size_t grid_size, block_size;
	make_kernel_param(&grid_size, &block_size, n_gsvertices(), 256);
	testVflags_kernel << <grid_size, block_size >> > (n_gsvertices(), vertflag);
	cudaDeviceSynchronize();
	cuda_error_check;
}

__global__ void vertexlexid_kernel(int* plexid) {
	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	int ereso[3] = { gGridCellReso[0], gGridCellReso[1], gGridCellReso[2] };
	int nv = (ereso[0] + 1) * (ereso[1] + 1) * (ereso[2] + 1);
	if (tid >= nv) return;
	int pos[3];
	pos[0] = tid % (ereso[0] + 1);
	pos[1] = tid / (ereso[0] + 1) % (ereso[1] + 1);
	pos[2] = tid / (ereso[0] + 1) / (ereso[1] + 1);
	int gsid = lexi2gs(pos, gGsVertexReso, gGsVertexEnd);
	plexid[tid] = gsid;
}

__global__ void celllexid_kernel(int* plexid) {
	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	int ereso[3] = { gGridCellReso[0], gGridCellReso[1], gGridCellReso[2] };
	int ne = ereso[0] * ereso[1] * ereso[2];
	if (tid >= ne) return;
	int pos[3];
	pos[0] = tid % (ereso[0]);
	pos[1] = tid / (ereso[0]) % (ereso[1]);
	pos[2] = tid / (ereso[0]) / (ereso[1]);
	int gsid = lexi2gs(pos, gGsCellReso, gGsCellEnd);
	plexid[tid] = gsid;
}

std::vector<int> homo::Grid_H::getCellLexidMap(void)
{
	useGrid_g();
	auto tmpname = getMem().addBuffer(n_gscells() * sizeof(float));
	int* tmp = getMem().getBuffer(tmpname)->data<int>();

	int ne = cellReso[0] * cellReso[1] * cellReso[2];
	size_t grid_size, block_size;
	make_kernel_param(&grid_size, &block_size, ne, 256);
	std::vector<int> eidmap(ne);

	celllexid_kernel << <grid_size, block_size >> > (tmp);
	cudaDeviceSynchronize();
	cuda_error_check;

	cudaMemcpy(eidmap.data(), tmp, sizeof(int) * ne, cudaMemcpyDeviceToHost);

	getMem().deleteBuffer(tmp);

	return eidmap;
}


template<typename T, typename Tout, int BlockSize = 256>
__global__ void v_dot_kernel(int nv,
	VertexFlags* vflags,
	devArray_t<T*, 1> vlist, devArray_t<T*, 1> ulist, Tout* p_out, bool removePeriodDof = false
) {

	__shared__  T blocksum[BlockSize / 32];

	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	size_t stride = blockDim.x * gridDim.x;
	bool fiction = false;
	if (tid >= nv) fiction = true;

	int warpId = threadIdx.x / 32;
	int laneId = threadIdx.x % 32;

	size_t baseId = 0;
	Tout bsum = 0;
	for (; baseId + tid < nv; baseId += stride) {
		int vid = baseId + tid;
		VertexFlags vflag = vflags[vid];
		if (vflag.is_fiction()) continue;
		if (removePeriodDof && (vflag.is_period_padding() || vflag.is_max_boundary())) continue;

		T v = { vlist[0][vid] };
		T u = { ulist[0][vid] };
		Tout uv = u*v;
		bsum += uv;
	}

	// warp reduce
	for (int offset = 16; offset > 0; offset /= 2) {
		bsum += shfl_down(bsum, offset);
	}

	if (laneId == 0) {
		blocksum[warpId] = bsum;
	}

	__syncthreads();

	// block reduce
	if (warpId == 0) {
		if (BlockSize / 32 > 32) { print_exception; }

		if (threadIdx.x < BlockSize / 32) {
			bsum = blocksum[threadIdx.x];
		}
		else {
			bsum = 0;
		}
		for (int offset = 16; offset > 0; offset /= 2) {
			bsum += shfl_down(bsum, offset);
		}
		if (laneId == 0) {
			p_out[blockIdx.x] = bsum;
		}
	}
}

float homo::Grid_H::v_dot(VT* v, VT* u, bool removePeriodDof /*= false*/, int len /*= -1*/)
{
	if (len == -1) len = n_gsvertices();
	int szTemp = len * sizeof(float) / 100;
	if (!removePeriodDof) {
		auto buffer = getTempPool().getBuffer(szTemp);
		auto pTemp = buffer.template data<float>();
		double result = dot(v, u, pTemp, len);
		cuda_error_check;
		return result;
	}
	else {
		devArray_t<VT*, 1> vlist{ v };
		devArray_t<VT*, 1> ulist{ u };
		int nv = n_gsvertices();
		auto buffer = getTempBuffer(nv / 100 * sizeof(float));
		float* p_tmp = buffer.template data<float>();
		size_t grid_size, block_size;
		int batch = nv;
		make_kernel_param(&grid_size, &block_size, batch, 256);
		v_dot_kernel << <grid_size, block_size >> > (nv, vertflag, vlist, ulist, p_tmp, removePeriodDof);
		cudaDeviceSynchronize();
		double s = dump_array_sum(p_tmp, grid_size);
		cuda_error_check;
		return s;
	}
}

void homo::Grid_H::pad_vertex_data(float* v[1])
{
	pad_vertex_data_imp<float, 1>(v, cellReso, vertflag);
}
void homo::Grid_H::pad_vertex_data(half* v[1])
{
	pad_vertex_data_imp<half, 1>(v, cellReso, vertflag);
}


template<typename T>
__global__ void v_removeT_kernel(int nv, VertexFlags* vflags, devArray_t<T*, 1> u, devArray_t<T, 1> t) {
	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= nv) return;
	VertexFlags vflag = vflags[tid];
	if (vflag.is_fiction()) return;
	u[0][tid] -= t[0];
}

void homo::Grid_H::v_removeT(VT* u, VT tHost[1])
{
	devArray_t<VT*, 1> uarr{ u };
	devArray_t<VT, 1> tArr{ tHost[0]};
	size_t grid_size, block_size;
	make_kernel_param(&grid_size, &block_size, n_gsvertices(), 256);
	int nv = n_gsvertices();
	v_removeT_kernel << <grid_size, block_size >> > (nv, vertflag, uarr, tArr);
	cudaDeviceSynchronize();
	cuda_error_check;
}

template<typename T, typename Tout, int BlockSize = 256>
__global__ void v_average_kernel(devArray_t<T*, 1> vlist, VertexFlags* vflags, int len, devArray_t<Tout*, 1> outlist,
	bool removePeriodDof, bool firstReduce = false) {
	__shared__ Tout s[1][BlockSize / 32];
	size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
	Tout v[1] = { 0. };

	size_t stride = gridDim.x * blockDim.x;

	int base = 0;
	for (; base + tid < len; base += stride) {
		int vid = base + tid;
		if (firstReduce) {
			VertexFlags vflag = vflags[vid];
			if (vflag.is_fiction()) continue;
			if ((removePeriodDof && vflag.is_max_boundary()) || vflag.is_period_padding()) continue;
		}
		v[0] += Tout(vlist[0][vid]);
	}

	int warpId = threadIdx.x / 32;
	int laneId = threadIdx.x % 32;

	// warp reduce
	for (int offset = 16; offset > 0; offset /= 2) {
		v[0] += shfl_down(v[0], offset);
	}
	if (laneId == 0) {
		s[0][warpId] = v[0];
	}
	// block reduce, do NOT use 1024 or higher blockSize
	if (BlockSize / 32 > 32) { print_exception; }

	__syncthreads();

	// block reduce
	if (warpId == 0) {
		if (threadIdx.x < BlockSize / 32) {
			v[0] = s[0][threadIdx.x];
		}
		else {
			v[0] = 0;
		}

		for (int offset = 16; offset > 0; offset /= 2) {
			v[0] += shfl_down(v[0], offset);
		}

		if (laneId == 0) {
			outlist[0][blockIdx.x] = v[0];
		}
	}
}

void homo::Grid_H::v_average(VT* v, VT& vMean, bool removePeriodDof)
{
	int le = (n_gsvertices() / 100 + 511) / 512 * 512;
	auto buffer = getTempBuffer(sizeof(float) * le * 3);
	float* ptmp = (float*)buffer.template data<>();
	devArray_t<float*, 1> vtmp;
	vtmp[0] = ptmp;

	devArray_t<float*, 1> vout{ vtmp[0] + le / 2 };

	devArray_t<VT*, 1> vlist{ v };
	size_t grid_size, block_size;
	int rest = n_gsvertices();
	make_kernel_param(&grid_size, &block_size, rest, 256);
	if (le / 2 < grid_size) print_exception;
	v_average_kernel << <grid_size, block_size >> > (vlist, vertflag, rest, vtmp, removePeriodDof, true);
	cudaDeviceSynchronize();
	cuda_error_check;
	rest = grid_size;

	while (rest > 1) {
		make_kernel_param(&grid_size, &block_size, rest, 256);
		if (le / 2 < grid_size) print_exception;
		v_average_kernel << <grid_size, block_size >> > (vtmp, vertflag, rest, vout, removePeriodDof, false);
		cudaDeviceSynchronize();
		std::swap(vtmp[0], vout[0]);
		rest = grid_size;
	}

	float vMean_f;
	cudaMemcpy(&vMean_f, vtmp[0], sizeof(float), cudaMemcpyDeviceToHost);

	int nValid;
	if (removePeriodDof) {
		nValid = cellReso[0] * cellReso[1] * cellReso[2];
	}
	else {
		nValid = (cellReso[0] + 1) * (cellReso[1] + 1) * (cellReso[2] + 1);
	}
	vMean = vMean_f / nValid;
	cuda_error_check;
}

template<typename T>
__global__ void update_rho_kernel(
	int nv, VertexFlags* vflags, CellFlags* eflags,
	float* srcrho, int srcPitchT, T* dstrho
) {
	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= nv) return;
	GridVertexIndex indexer(gGridCellReso[0], gGridCellReso[1], gGridCellReso[2]);
	VertexFlags vflag;
	vflag = vflags[tid];
	bool fiction = vflag.is_fiction() || vflag.is_period_padding() || vflag.is_max_boundary();
	if (fiction) return;
	indexer.locate(tid, vflag.get_gscolor(), gGsVertexEnd);
	int eid = indexer.neighElement(7, gGsCellEnd, gGsCellReso).getId();
	fiction = fiction || eid == -1;
	if (fiction) return;
	CellFlags eflag = eflags[eid];
	if (eflag.is_fiction() || eflag.is_period_padding()) return;
	auto p = indexer.getPos();
	// to element pos without padding
	p.x -= 1; p.y -= 1; p.z -= 1;
	int sid;
	if (srcPitchT <= 0)
		sid = p.x + (p.y + p.z * gGridCellReso[1]) * gGridCellReso[0];
	else
		sid = p.x + (p.y + p.z * gGridCellReso[1]) * srcPitchT;
	dstrho[eid] = srcrho[sid];

	//int tid = blockIdx.x * blockDim.x + threadIdx.x;
	//if (tid >= nv) return;
	//// srcrho[tid];
	//int pos[3] = { tid % gGridCellReso[0], tid / gGridCellReso[0] % gGridCellReso[0], tid / (gGridCellReso[0]) / (gGridCellReso[0]) };
	//int gsid = lexi2gs(pos, gGsCellReso, gGsCellEnd, false);
	//int sid;
	//if (srcPitchT <= 0)
	//	sid = pos[0] + (pos[1] + pos[2] * gGridCellReso[1]) * gGridCellReso[0];
	//else
	//	sid = pos[0] + (pos[1] + pos[2] * gGridCellReso[1]) * srcPitchT;
	//CellFlags eflag;

	//eflag = eflags[gsid];

	//if (eflag.is_fiction()) { printf("error\n"); };

	//dstrho[gsid] = srcrho[sid];
}

template<typename T>
__global__ void update_rho_host_kernel(
	int ne, VertexFlags* vflags, CellFlags* eflags,
	float* srcrho, T* dstrho
) {
	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= ne) return;
	// srcrho[tid];
	int pos[3] = { tid % (MIN_TRANSFER + 2), tid / (MIN_TRANSFER + 2) % (MIN_TRANSFER + 2), tid / (MIN_TRANSFER + 2) / (MIN_TRANSFER + 2) };
	int gsid = lexi2gs(pos, gGsCellReso, gGsCellEnd, true);

	CellFlags eflag;

	eflag = eflags[gsid];

	if (eflag.is_fiction()) { printf("error\n"); };

	dstrho[gsid] = srcrho[tid];
}

template<typename T, int N, typename Flag>
__global__ void pad_data_kernel(
	int nsrcpadd, devArray_t<T*, N> v, Flag* flags,
	devArray_t<int, 3> resosrcpadd, devArray_t<int, 3> srcbasepos, devArray_t<int, 3> period,
	devArray_t<int, 3> resolist, devArray_t<devArray_t<int, 8>, 3> gsreso, devArray_t<int, 8> gsend
){
	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

	__shared__ int gsReso[3][8];
	__shared__ int gsEnd[8];
	if (threadIdx.x < 3 * 8) {
		int i = threadIdx.x / 8;
		int j = threadIdx.x % 8;
		gsReso[i][j] = gsreso[i][j];
		if (i == 0) {
			gsEnd[j] = gsend[j];
		}
	}
	__syncthreads();

	if (tid >= nsrcpadd) return;
	int nf[3] = {
		resosrcpadd[0] * resosrcpadd[1],
		resosrcpadd[1] * resosrcpadd[2],
		resosrcpadd[2] * resosrcpadd[0]
	};
	int n_min = nf[0] + nf[1] + nf[2];
	if (tid >= 2 * n_min) return;

	int min_id = tid % n_min;
	int max_id = tid / n_min;

	int pos[3];

	char3 bound = {};

	if (min_id < nf[0]) {
		pos[0] = min_id % resosrcpadd[0];
		pos[1] = min_id / resosrcpadd[0];
		pos[2] = 0;
		bound.z = 1;
	}
	else if (min_id < nf[0] + nf[1]) {
		pos[0] = 0;
		pos[1] = (min_id - nf[0]) % resosrcpadd[1];
		pos[2] = (min_id - nf[0]) / resosrcpadd[1];
		bound.x = 1;
	}
	else if (min_id < nf[0] + nf[1] + nf[2]) {
		pos[0] = (min_id - nf[0] - nf[1]) % resosrcpadd[0];
		pos[1] = 0;
		pos[2] = (min_id - nf[0] - nf[1]) / resosrcpadd[0];
		bound.y = 1;
	}

	if (max_id == 1) {
		if (bound.x) {
			pos[0] += resosrcpadd[0] - 1;
		}
		else if (bound.y) {
			pos[1] += resosrcpadd[1] - 1;
		}
		else if (bound.z) {
			pos[2] += resosrcpadd[2] - 1;
		}
		else {
			print_exception; // DEBUG
		}
	}

	pos[0] += srcbasepos[0];
	pos[1] += srcbasepos[1];
	pos[2] += srcbasepos[2];

	int myid = lexi2gs(pos, gsReso, gsEnd);

	//printf("pos = (%d, %d, %d)\n", pos[0], pos[1], pos[2]);
	// scatter padding data
	int oppos[3];
	for (int offx : { -period[0], 0, period[0]}) {
		oppos[0] = offx + pos[0];
		if (oppos[0] < -1 || oppos[0] > resolist[0]) continue;

		for (int offy : {-period[1], 0, period[1]}) {
			oppos[1] = offy + pos[1];
			if (oppos[1] < -1 || oppos[1] > resolist[1]) continue;

			for (int offz : {-period[2], 0, period[2]}) {
				oppos[2] = offz + pos[2];
				if (oppos[2]<-1 || oppos[2]>resolist[2]) continue;

				if ((oppos[0] == -1 || oppos[0] == resolist[0]) ||
					(oppos[1] == -1 || oppos[1] == resolist[1]) ||
					(oppos[2] == -1 || oppos[2] == resolist[2])) {
					int opid = lexi2gs(oppos, gsReso, gsEnd);
					// debug
					if (!flags[opid].is_period_padding()) {
						printf("\033[31moppos = (%d, %d, %d)\033[0m\n", oppos[0], oppos[1], oppos[2]);
						//print_exception;
					}
					for (int i = 0; i < N; i++) {
						v[i][opid] = v[i][myid];
					}
					//printf("pos = (%d, %d, %d) oppos = (%d, %d, %d)\n",
					//	pos[0], pos[1], pos[2], oppos[0], oppos[1], oppos[2]);
				}
			}
		}
	}
}

template <typename T>
void pad_cell_data_imp(T *e, CellFlags *eflags, std::array<int, 3> cellReso, int gsCellReso[3][8], int gsCellSetEnd[8]) {
	int nsrcpadd = 2 * (cellReso[0] * cellReso[1] + cellReso[1] * cellReso[2] + cellReso[0] * cellReso[2]);
	devArray_t<int, 3> resolist{cellReso[0], cellReso[1], cellReso[2]};
	devArray_t<int, 3> resopad{cellReso[0], cellReso[1], cellReso[2]};
	devArray_t<int, 3> padbase{0, 0, 0};
	devArray_t<int, 3> period{cellReso[0], cellReso[1], cellReso[2]};
	devArray_t<devArray_t<int, 8>, 3> gsreso;
	devArray_t<int, 8> gsend;
	for (int i = 0; i < 3; i++)
		for (int j = 0; j < 8; j++)
			gsreso[i][j] = gsCellReso[i][j];
	for (int i = 0; i < 8; i++) {
		gsend[i] = gsCellSetEnd[i];
	}
	devArray_t<T *, 1> arr{e};
	size_t grid_size, block_size;
	make_kernel_param(&grid_size, &block_size, nsrcpadd, 256);
	pad_data_kernel<<<grid_size, block_size>>>(nsrcpadd, arr, eflags, resopad, padbase, period, resolist, gsreso, gsend);
	cudaDeviceSynchronize();
	cuda_error_check;
}

void homo::Grid_H::pad_cell_data(float* e) {
	pad_cell_data_imp(e, cellflag, cellReso, gsCellReso, gsCellSetEnd);
}
void homo::Grid_H::pad_cell_data(half* e) {
	pad_cell_data_imp(e, cellflag, cellReso, gsCellReso, gsCellSetEnd);
}
void homo::Grid_H::update(float* rho, int pitchT, bool lexiOrder) {
	if (!lexiOrder) {
		type_cast(rho_g, rho, n_gscells());
	}
	else {
		useGrid_g();
		size_t grid_size, block_size;
		int nv = n_gsvertices();
		auto vflags = vertflag;
		auto eflags = cellflag;
		make_kernel_param(&grid_size, &block_size, nv, 256);
		update_rho_kernel << <grid_size, block_size >> > (nv, vflags, eflags, rho, pitchT, rho_g);
		cudaDeviceSynchronize();
		cuda_error_check;
		pad_cell_data(rho_g);
	}
}
void homo::Grid_H::update_host(float* rho) {
	size_t grid_size, block_size;
	int ne = pow(MIN_TRANSFER+2, 3);
	auto vflags = vertflag;
	auto eflags = cellflag;
	make_kernel_param(&grid_size, &block_size, ne, 256);
	update_rho_host_kernel << <grid_size, block_size>> > (ne, vflags, eflags, rho, rho_g);
	cudaDeviceSynchronize();
	cuda_error_check;
}
void homo::Grid_H::update_hostgs(float* rho) {
	size_t grid_size, block_size;
	int ne = pow(MIN_TRANSFER + 2, 3);
	auto vflags = vertflag;
	auto eflags = cellflag;
	make_kernel_param(&grid_size, &block_size, ne, 256);
	update_rho_host_kernel << <grid_size, block_size, 0, stream[current] >> > (ne, vflags, eflags, rho, rho_g);
}
