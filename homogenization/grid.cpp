#define _USE_MATH_DEFINES
#include "grid.h"
#include <stdexcept>
#include <algorithm>
#include <numeric>
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

	// compute eight Colored node number
	auto nv_ne = countGS();

	// allocate buffer
	size_t totalMem = allocateBuffer(nv_ne.first, nv_ne.second);

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

size_t Grid_H::allocateBuffer(int nv, int ne) 
{
	size_t total = 0;
	u_g[0] = getMem().addBuffer(homoutils::formated("%s_u_%d", getName().c_str()), nv * sizeof(VT))->data<VT>();
	f_g[0] = getMem().addBuffer(homoutils::formated("%s_f_%d", getName().c_str()), nv * sizeof(VT))->data<VT>();
	r_g[0] = getMem().addBuffer(homoutils::formated("%s_r_%d", getName().c_str()), nv * sizeof(VT))->data<VT>();
	total += nv * 3 * sizeof(VT);
	// warning and todo
	if (!is_root) {
		for (int i = 0; i < 27; i++) {
			stencil_g[i] = getMem().addBuffer(homoutils::formated("%s_st_%d", getName().c_str(), i), nv * sizeof(VT))->data<VT>();
		}
		total += nv * sizeof(VT) * 27;
	}
	if (gridConfig.enableManagedMem) {
		for (int i = 0; i < 3; i++) {
			uchar_h[i] = getMem().addBuffer(homoutils::formated("%s_uchost_%d_%d", getName().c_str(), i), nv * sizeof(VT), Managed)->data<VT>();
			v_reset(uchar_h[i], nv);
		}
	}
	else {
		for (int i = 0; i < 3; i++) {
			uchar_h[i] = getMem().addBuffer(homoutils::formated("%s_uchost_%d_%d", getName().c_str(), i), nv * sizeof(VT), Hostheap)->data<VT>();
			memset(uchar_h[i], 0, sizeof(VT) * nv);
		}
	}
	vertflag = getMem().addBuffer<VertexFlags>(homoutils::formated("%s_vflag", getName().c_str()), nv)->data<VertexFlags>();
	cellflag = getMem().addBuffer<CellFlags>(homoutils::formated("%s_cflag", getName().c_str()), ne)->data<CellFlags>();
	total += nv * sizeof(VertexFlags);
	total += ne * sizeof(CellFlags);

	if (is_root) {
		total += ne * sizeof(float);
		rho_g = getMem().addBuffer(homoutils::formated("%s_rho", getName().c_str()), ne * sizeof(float))->data<float>();
	}

	printf("%s allocated %zd MB GPU memory\n", getName().c_str(), total / 1024 / 1024);

	return total;
}


VT* homo::Grid_H::getDisplacement(void)
{
	return u_g[0];
}


double homo::Grid_H::residual(void)
{
	return v_norm(r_g[0]);
}
void Grid_H::useFchar(int k)
{
	useGrid_g();
	enforce_unit_macro_strain(k);
	pad_vertex_data(f_g);

	if (0) {
		char buf[100];
		sprintf_s(buf, "./fchar%d", k);
		v_write(buf, f_g[0], true);
	}
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
	eigen2ConnectedMatlab("Khost", Khost);
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
	for (int xc_off = -2 * upCoarse[0]; xc_off < 2 * upCoarse[0]; xc_off++) {
		for (int yc_off = -2 * upCoarse[1]; yc_off < 2 * upCoarse[1]; yc_off++) {
			for (int zc_off = -2 * upCoarse[2]; zc_off < 2 * upCoarse[2]; zc_off++) {
				int epos[3] = {
					(xc_off + finereso[0]) % finereso[0],
					(yc_off + finereso[1]) % finereso[1],
					(zc_off + finereso[2]) % finereso[2]
				};
				int eid = epos[0] + epos[1] * finereso[0] + epos[2] * finereso[0] * finereso[1];
				int egsid = fine->elexid2gsid(eid);
				float prho = rhoPenalMin + powf(rholist[egsid], exp_penal);
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

