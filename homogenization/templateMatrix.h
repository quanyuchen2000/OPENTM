#pragma once

#ifndef __TEMPLATE_MATRIX_H
#define __TEMPLATE_MATRIX_H

#include "Eigen/Eigen"
#include "gmem/DeviceBuffer.h"

typedef float Scalar;

constexpr double default_heat_ratio = 1;


void initTemplateMatrix_H(Scalar element_len, homo::BufferManager& gm, Scalar hmodu = default_heat_ratio);

const Eigen::Matrix<Scalar, 8, 8>& getTemplateMatrix_H(void);
const Eigen::Matrix<double, 8, 8>& getTemplateMatrixFp64_H(void);
const Eigen::Matrix<float, 8, 3>& getFeMatrix(void);
const Eigen::Matrix<float, 8, 3>& getDispMatrix(void);

const Scalar* getTemplateMatrixElements_H(void);

#endif

