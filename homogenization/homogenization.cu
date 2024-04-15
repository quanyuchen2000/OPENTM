#include "homogenization.h"
#include "device_launch_parameters.h"
#include "homoCommon.cuh"
#include "tictoc.h"
#include "cuda_fp16.h"
#include "mma.h"

#define USE_LAME_MATRIX 1

using namespace homo;
using namespace culib;

template<typename T>
__global__ void heatMatrix_kernel_opt(
	int nv,
	devArray_t<devArray_t<float*, 1>, 3> ucharlist,
	T* rholist, VertexFlags* vflags, CellFlags* eflags,
	float* elementCompliance, int pitchT
) {
	__shared__ float KE[8][8];
	loadTemplateMatrix_H(KE);

	__shared__ float uChi[3][8];
	__shared__ float uchar[3][8][32];
	__shared__ float gSum[6][4][32];

	int warpId = threadIdx.x / 32;
	int laneId = threadIdx.x % 32;
	if (warpId < 3 && laneId == 0) {
		if (warpId == 0) { elementMacroDisplacement_H<float, 0>(uChi[warpId]); }
		else if (warpId == 1) { elementMacroDisplacement_H<float, 1>(uChi[warpId]); }
		else if (warpId == 2) { elementMacroDisplacement_H<float, 2>(uChi[warpId]); }
	}

	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

	bool is_ghost = false;

	int vid = blockIdx.x * 32 + laneId;

	is_ghost = vid >= nv;

	VertexFlags vflag;

	if (!is_ghost) { vflag = vflags[vid]; }

	is_ghost = is_ghost || vflag.is_fiction() || vflag.is_period_padding() || vflag.is_min_boundary();

	GridVertexIndex indexer(gGridCellReso[0], gGridCellReso[1], gGridCellReso[2]);

	int elementId = -1;

	if (!is_ghost) {
		indexer.locate(vid, vflag.get_gscolor(), gGsVertexEnd);
		elementId = indexer.neighElement(0, gGsCellEnd, gGsCellReso).getId();
	}

	//float vol_inv = 1.f / volume;
	float prho = 0;

	CellFlags eflag;
	int ev[8];
	if (elementId != -1 && !is_ghost) {
		prho = rhoPenalMin + powf(float(rholist[elementId]), exp_penal[0]);
		if (tid == 2566)
			printf("rho = %f\n", rholist[elementId]);
		eflag = eflags[elementId];
		is_ghost = is_ghost || eflag.is_fiction() || eflag.is_period_padding();
		if (!is_ghost) {
			for (int i = 0; i < 8; i++) {
				int id = i % 2 + i / 2 % 2 * 3 + i / 4 * 9;
				int neighVid = indexer.neighVertex(id, gGsVertexEnd, gGsVertexReso).getId();
				ev[i] = neighVid;
			}
			if (warpId < 3) {
#pragma unroll
				for (int i = 0; i < 8; i++) {
					float chipair = ucharlist[warpId][0][ev[i]];
					uchar[warpId][i][laneId] = uChi[warpId][i] - chipair;
				}
			}
		}
	}
	__syncthreads();

	float ce[6] = { 0 };
	if (elementId != -1 && !is_ghost && !eflag.is_fiction() && !eflag.is_period_padding()) {
		//float c = 0;
		int counter = 0;
#pragma unroll
		for (int iStrain = 0; iStrain < 3; iStrain++) {
#pragma unroll
			for (int jStrain = iStrain; jStrain < 3; jStrain++) {
				float kv = { 0. };
#pragma unroll
				for (int kj = 0; kj < 8; kj++) {
					kv += KE[warpId][kj] * uchar[jStrain][kj][laneId];
				}
				kv *= uchar[iStrain][warpId][laneId];
				ce[counter] += kv;
				counter++;
			}
		}
	}
	//if (tid == 2566) {
	//	for (int i = 0; i < 8; i++) {
	//		for (int j = 0; j < 8; j++) {
	//			printf("%f ", KE[i][j]);
	//		}
	//		printf("\n");
	//	}
	//}
	//	printf("2566:%f \n", ce[0]);
	// block reduction
	if (warpId >= 4) {
		for (int i = 0; i < 6; i++) {
			gSum[i][warpId - 4][laneId] = ce[i] * prho;
		}
	}
	__syncthreads();
	if (warpId < 4) {
		for (int i = 0; i < 6; i++) {
			gSum[i][warpId][laneId] += ce[i] * prho;
		}
	}
	__syncthreads();
	if (warpId < 2) {
		for (int i = 0; i < 6; i++) {
			gSum[i][warpId][laneId] += gSum[i][warpId + 2][laneId];
		}
	}
	__syncthreads();
	if (warpId == 0) {
		for (int i = 0; i < 6; i++) {
			ce[i] = gSum[i][0][laneId] + gSum[i][1][laneId];
		}
		// here the number of ce is 512 is right, but it's too small.
		//if (ce[0] > 1e-5)
		//	printf("%f\n", ce[0]);
		// warp reduce
		for (int offset = 16; offset > 0; offset /= 2) {
#pragma unroll
			for (int i = 0; i < 6; i++) {
				ce[i] += shfl_down(ce[i], offset);
			}
		}
		if (laneId == 0) {
			for (int i = 0; i < 6; i++) {
				elementCompliance[blockIdx.x + i * pitchT] = ce[i];
			}
		}
	}
}

template <typename T, int BlockSize = 256>
__global__ void fillTotalVertices_kernel_H(
	int nv, VertexFlags* vflags,
	devArray_t<devArray_t<T*, 1>, 3> uchar,
	devArray_t<devArray_t<float*, 1>, 3> dst)
{
	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= nv) return;
	int ereso[3] = { gGridCellReso[0], gGridCellReso[1], gGridCellReso[2] };
	int vid = tid;
	VertexFlags vflag = vflags[vid];
	bool fiction = vflag.is_period_padding() || vflag.is_fiction();

	GridVertexIndex indexer(gGridCellReso[0], gGridCellReso[1], gGridCellReso[2]);

	if (!fiction) {
		//indexer.locate(vid, vflag.get_gscolor(), gGsVertexEnd);
		//auto pos = indexer.getPos();
		//int p[3] = { pos.x - 1, pos.y - 1, pos.z - 1 };
		//int lexid = p[0] + p[1] * (ereso[0] + 1) + p[2] * (ereso[0] + 1) * (ereso[1] + 1);
		for (int i = 0; i < 3; i++) {
			dst[i][0][vid] = uchar[i][0][vid];
		}
	}
}
void homo::Homogenization_H::heatMatrix(double C[3][3]) {
	mg_->reset_displacement();
	for (int i = 0; i < 3; i++) {
			grid->useFchar(i);
			grid->useUchar(i);
			grid->translateForce(2, grid->u_g);
			mg_->solveEquation(config.femRelThres);
			grid->setUchar(i, grid->getDisplacement());
	}
	// in heat method the vol is the real volume that means 1*1*1
	float vol = 1.;

	printf("n_cell = %d\n", grid->n_cells());

	use4Bytesbank();
	grid->useGrid_g();
	if (config.useManagedMemory) {
		devArray_t<devArray_t<VT*, 1>, 3> ucharlist;
		for (int i = 0; i < 3; i++) {
			ucharlist[i][0] = grid->uchar_h[i];
		}
		auto rho_g = grid->rho_g;
		VertexFlags* vflags = grid->vertflag;
		CellFlags* eflags = grid->cellflag;
		int nv = grid->n_gsvertices();
		size_t grid_size, block_size;
		// prefecth unified memory data to device memory
		devArray_t<devArray_t<float*, 1>, 3> dst;
		dst[0][0] = (grid->f_g[0]);
		dst[1][0] = (grid->u_g[0]);
		dst[2][0] = (grid->r_g[0]);

		make_kernel_param(&grid_size, &block_size, nv, 256);
		fillTotalVertices_kernel_H << <grid_size, block_size >> > (nv, vflags, ucharlist, dst);
		cudaDeviceSynchronize();
		cuda_error_check;
		// compute element energy and sum
		make_kernel_param(&grid_size, &block_size, nv * 8, 256);
		int pitchT = round(grid_size, 128);
		auto buffer = getTempBuffer(pitchT * 6 * sizeof(float));
		init_array(buffer.template data<float>(), 0.f, pitchT * 6);
		//_TIC("ematopt")
		heatMatrix_kernel_opt << <grid_size, block_size >> > (nv, dst,
			rho_g, vflags, eflags,
			buffer.template data<float>(), pitchT);
		cudaDeviceSynchronize();
		//_TOC;
		//printf("elasticMatrix_kernel_opt  time = %4.2f ms\n", tictoc::get_record("ematopt"));
		cuda_error_check;
		int counter = 0;
		for (int i = 0; i < 3; i++) {
			for (int j = i; j < 3; j++) {
				C[i][j] = parallel_sum(buffer.template data<float>() + counter * pitchT, pitchT) / vol;
				counter++;
			}
		}
		for (int i = 0; i < 3; i++) { for (int j = 0; j < i; j++) { C[i][j] = C[j][i]; } }
		cudaMemset(grid->u_g[0], 0, nv * sizeof(float));
		cudaMemset(grid->r_g[0], 0, nv * sizeof(float));
		cudaMemset(grid->f_g[0], 0, nv * sizeof(float));
		cudaDeviceSynchronize();
		cuda_error_check;
	}
	else {
		NO_SUPPORT_ERROR;
	}
	for (int i = 0; i < 3; i++) {
		for (int j = i + 1; j < 3; j++) {
			C[i][j] = C[j][i];
		}
	}
	return;
}

// use vector stored in F(chi_0,chi_1) U(chi_2,chi_3) R(chi_4,chi_5)
template <typename T, int BlockSize = 256>
__global__ void Sensitivity_kernel_opt_2_H(
	int nv, VertexFlags* vflags, CellFlags* eflags,
	devArray_t<devArray_t<float*, 1>, 3> ucharlist,
	T* rholist,
	devArray_t<devArray_t<float, 3>, 3> dc,
	float* sens, float volume,
	int pitchT, bool lexiOrder)
{
	__shared__ float KE[8][8];
	__shared__ float dC[3][3];
	__shared__ float uChi[3][8];

	__shared__ float uchar[3][8][32];
	__shared__ float gSum[3][3][4][32];

	int warpId = threadIdx.x / 32;
	int laneId = threadIdx.x % 32;

	if (warpId < 3 && laneId == 0) {
		if (warpId == 0) {
			elementMacroDisplacement_H<float, 0>(uChi[warpId]);
		}
		else if (warpId == 1) {
			elementMacroDisplacement_H<float, 1>(uChi[warpId]);
		}
		else if (warpId == 2) {
			elementMacroDisplacement_H<float, 2>(uChi[warpId]);
		}
	}

	if (threadIdx.x < 9) {
		dC[threadIdx.x / 3][threadIdx.x % 3] = dc[threadIdx.x / 3][threadIdx.x % 3];
	}

	loadTemplateMatrix_H(KE);

	bool is_ghost = false;

	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;


	int vid = blockIdx.x * 32 + laneId;

	is_ghost = vid >= nv;

	VertexFlags vflag;
	if (!is_ghost) vflag = vflags[vid];
	if (vflag.is_fiction() || vflag.is_period_padding() || vflag.is_min_boundary()) is_ghost = true;

	GridVertexIndex indexer(gGridCellReso[0], gGridCellReso[1], gGridCellReso[2]);

	if (!is_ghost) {
		indexer.locate(vid, vflag.get_gscolor(), gGsVertexEnd);
	}

	int elementId;
	if (!is_ghost) elementId = indexer.neighElement(0, gGsCellEnd, gGsCellReso).getId();

	float vol_inv = 1.f / volume;

	CellFlags eflag;
	int ev[8];
	if (elementId != -1 && !is_ghost) {
		eflag = eflags[elementId];
		is_ghost = is_ghost || eflag.is_fiction() || eflag.is_period_padding();
		if (!is_ghost) {
			for (int i = 0; i < 8; i++) {
				int id = i % 2 + i / 2 % 2 * 3 + i / 4 * 9;
				int neighVid = indexer.neighVertex(id, gGsVertexEnd, gGsVertexReso).getId();
				ev[i] = neighVid;
			}
			if (warpId < 3) {
#pragma unroll
				for (int i = 0; i < 8; i++) {
					float chipair = ucharlist[warpId][0][ev[i]];
					uchar[warpId][i][laneId] = uChi[warpId][i] - chipair;
					if (tid == 2566)
						printf("%f ", uChi[warpId][i] - chipair);
				}
			}
		}
	}
	__syncthreads();

	float dc_lane[3][3] = { 0. };
	float prho = 0;
	// 8 warp to 32 vertices
	if (elementId != -1 && !is_ghost) {
		float pwn = exp_penal[0];
		prho = pwn * powf(float(rholist[elementId]), pwn - 1);
		if (tid == 2566)
			printf("pwn = %f\n rho = %f", pwn, rholist[elementId]);
#pragma unroll
		for (int iStrain = 0; iStrain < 3; iStrain++) {
#pragma unroll
			for (int jStrain = iStrain; jStrain < 3; jStrain++) {
				float c = { 0. };
#pragma unroll
				for (int kj = 0; kj < 8; kj++) {
					c += KE[warpId][kj] * uchar[jStrain][kj][laneId];
				}
				c *= uchar[iStrain][warpId][laneId];
				dc_lane[iStrain][jStrain] = c;
			}
		}
	}
	// block reduce
	if (warpId >= 4) {
		for (int i = 0; i < 3; i++) {
			for (int j = i; j < 3; j++) {
				gSum[i][j][warpId - 4][laneId] = dc_lane[i][j];
			}
		}
	}
	__syncthreads();
	if (warpId < 4) {
		for (int i = 0; i < 3; i++) {
			for (int j = i; j < 3; j++) {
				gSum[i][j][warpId][laneId] += dc_lane[i][j];
			}
		}
	}
	__syncthreads();
	if (warpId < 2) {
		for (int i = 0; i < 3; i++) {
			for (int j = i; j < 3; j++) {
				gSum[i][j][warpId][laneId] += gSum[i][j][warpId + 2][laneId];
			}
		}
	}
	__syncthreads();
	if (warpId < 1 && !is_ghost && elementId != -1) {
		float s = 0;
		float vp = vol_inv * prho;
		if (tid == 2566)
			printf("vp = %f\n", prho);
		for (int i = 0; i < 3; i++) {
			for (int j = i; j < 3; j++) {
				dc_lane[i][j] = gSum[i][j][0][laneId] + gSum[i][j][1][laneId];
				float dclast = dC[i][j];
				if (i != j) dclast += dC[j][i];
				s += dc_lane[i][j] * dclast * vp;
			}
		}
		if (!lexiOrder) {
			sens[elementId] = s;
		}
		else {
			auto p = indexer.getPos();
			// p -> element pos -> element pos without padding 
			p.x -= 2; p.y -= 2; p.z -= 2;
			if (p.x < 0 || p.y < 0 || p.z < 0) print_exception;
			int lexid = p.x + (p.y + p.z * gGridCellReso[1]) * pitchT;
			sens[lexid] = s;
		}
	}
}


void homo::Homogenization_H::Sensitivity(float dC[3][3], float* sens, int pitchT, bool lexiOrder /*= false*/)
{
	grid->useGrid_g();
	devArray_t<devArray_t<float, 3>, 3> dc;
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			dc[i][j] = dC[i][j];
		}
	}

	init_array(sens, 0.f, grid->cellReso[1] * grid->cellReso[2] * pitchT);

	int nv = grid->n_gsvertices();
	auto vflags = grid->vertflag;
	auto eflags = grid->cellflag;
	auto rholist = grid->rho_g;
	// here volume = 1 which is the real scale. 1*1*1
	float volume = 1.0;
	size_t grid_size, block_size;
	make_kernel_param(&grid_size, &block_size, nv, 256);
	if (!config.useManagedMemory) {
		NO_SUPPORT_ERROR;
	}
	else {
		printf("Sensitivity analysis using managed memory...\n");
		devArray_t<devArray_t<VT*, 1>, 3> uchar;
		for (int i = 0; i < 3; i++) {
			uchar[i][0] = grid->uchar_h[i];
		}
		devArray_t<devArray_t<float*, 1>, 3> dst;
		dst[0][0] = (grid->f_g[0]);
		dst[1][0] = (grid->u_g[0]);
		dst[2][0] = (grid->r_g[0]);

		make_kernel_param(&grid_size, &block_size, nv, 256);
		fillTotalVertices_kernel_H << <grid_size, block_size >> > (nv, vflags, uchar, dst);
		cudaDeviceSynchronize();
		cuda_error_check;
		make_kernel_param(&grid_size, &block_size, nv * 8, 256);
		Sensitivity_kernel_opt_2_H << <grid_size, block_size >> > (nv, vflags, eflags,
			dst,
			rholist, dc, sens, volume, pitchT, lexiOrder);
		cudaDeviceSynchronize();
		cuda_error_check;
	}
	cuda_error_check;

	cudaMemset(grid->u_g[0], 0, nv * sizeof(float));
	cudaMemset(grid->r_g[0], 0, nv * sizeof(float));
	cudaMemset(grid->f_g[0], 0, nv * sizeof(float));
	cudaDeviceSynchronize();
	cuda_error_check;

	// DEBUG
	if (0) {
		int slist = grid->cellReso[2] * grid->cellReso[1] * grid->cellReso[2];
		std::vector<float> senslist(slist);
		cudaMemcpy2D(senslist.data(), grid->cellReso[0] * sizeof(float),
			sens, pitchT * sizeof(float),
			grid->cellReso[0] * sizeof(float), grid->cellReso[1] * grid->cellReso[2],
			cudaMemcpyDeviceToHost);
		cuda_error_check;
		//grid->array2matlab("senslist", senslist.data(), senslist.size());
	}
}

