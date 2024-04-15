#include "templateMatrix.h"
//#include "snippet.h"
#include "utils.h"
#include "math.h"
#include "matlab/matlab_utils.h"

using namespace homo;

/* initialize heat matrix for further computation*/
Scalar he = default_heat_ratio;
Eigen::Matrix<double, 3, 3> heat_matrix;
Eigen::Matrix<double, 8, 8> Kmu;
Eigen::Matrix<float, 8, 8> fKmu;
Eigen::Matrix<float, 8, 3> feMu;
Eigen::Matrix<float, 8, 3> disp;

/*
My computation of heat template matrix is from my matlab code.
The form is different from above but it works.
*/
void initTemplateMatrix_H(Scalar element_len, BufferManager& gm, Scalar hmodu) {
	double a, b, c;
	a = element_len; b = element_len; c = element_len;
	heat_matrix << he, 0, 0,
		0, he, 0,
		0, 0, he;
	std::vector<double> xx = { -sqrt(3. / 5), 0, sqrt(3. / 5) };
	auto yy = xx;
	auto zz = xx;
	double ww[3] = { 5. / 9, 8. / 9, 5. / 9 };
	Kmu.fill(0);
	feMu.fill(0);
	for (int ii = 0; ii < xx.size(); ++ii) {
		for (int jj = 0; jj < yy.size(); ++jj) {
			for (int kk = 0; kk < zz.size(); ++kk) {
				// integration point
				double x = xx[ii], y = yy[jj], z = zz[kk];
				// stress strain displacement matrix
				Eigen::VectorXd qx(8), qy(8), qz(8);
				qx << -((y - 1) * (z - 1)) / 8, ((y - 1) * (z - 1)) / 8, -((y + 1) * (z - 1)) / 8,
					((y + 1) * (z - 1)) / 8, ((y - 1) * (z + 1)) / 8, -((y - 1) * (z + 1)) / 8,
					((y + 1) * (z + 1)) / 8, -((y + 1) * (z + 1)) / 8;
				qy << -((x - 1) * (z - 1)) / 8, ((x + 1) * (z - 1)) / 8, -((x + 1) * (z - 1)) / 8,
					((x - 1) * (z - 1)) / 8, ((x - 1) * (z + 1)) / 8, -((x + 1) * (z + 1)) / 8,
					((x + 1) * (z + 1)) / 8, -((x - 1) * (z + 1)) / 8;
				qz << -((x - 1) * (y - 1)) / 8, ((x + 1) * (y - 1)) / 8, -((x + 1) * (y + 1)) / 8,
					((x - 1) * (y + 1)) / 8, ((x - 1) * (y - 1)) / 8, -((x + 1) * (y - 1)) / 8,
					((x + 1) * (y + 1)) / 8, -((x - 1) * (y + 1)) / 8;
				// Jacobian
				Eigen::Matrix<double, 3, 8> Jl;
				Jl.block<1, 8>(0, 0) = qx;
				Jl.block<1, 8>(1, 0) = qy;
				Jl.block<1, 8>(2, 0) = qz;
				Eigen::Matrix<double, 8, 3> Jr;
				Jr << -a, -b, -c,
					a, -b, -c,
					a, b, -c,
					-a, b, -c,
					-a, -b, c,
					a, -b, c,
					a, b, c,
					-a, b, c;
				auto J = Jl * Jr;
				auto qxyz = J.inverse() * Jl;
				double weight = J.determinant() * ww[ii] * ww[jj] * ww[kk];
				Kmu += weight * qxyz.transpose() * heat_matrix * qxyz;
				feMu += (weight * qxyz.transpose() * heat_matrix).cast<float>();
			}
		}
	}
	// here the order we have is not lesi... order, we need a mapping.
	Eigen::Matrix<double, 8, 8> temp_mat;
	Eigen::Matrix<float, 8, 3> temp_fe;
	std::vector<int> mapping = { 1,3,2,0,5,7,6,4 };
	for (size_t i = 0; i < mapping.size(); ++i) {
		for (size_t j = 0; j < mapping.size(); ++j) {
			temp_mat(mapping[i],mapping[j]) = Kmu(i,j);
		}
		temp_fe.block<1, 3>(mapping[i], 0) = feMu.block<1, 3>(i, 0);
	}
	Kmu = temp_mat;
	feMu = temp_fe;
	fKmu = Kmu.cast<float>();
	disp.fill(0);
	disp.block<7, 3>(1, 0) = fKmu.block<7, 7>(1, 1).colPivHouseholderQr().solve(feMu.block<7, 3>(1, 0));
	std::cout << disp;
}

const Eigen::Matrix<Scalar, 8, 8>& getTemplateMatrix_H(void)
{
	return fKmu;
}

const Eigen::Matrix<double, 8, 8>& getTemplateMatrixFp64_H(void)
{
	return Kmu;
}
const Eigen::Matrix<float, 8, 3>& getFeMatrix(void) {
	return feMu;
}
const Eigen::Matrix<float, 8, 3>& getDispMatrix(void) {
	return disp;
}
const Scalar* getTemplateMatrixElements_H(void)
{
	return fKmu.data();
}
