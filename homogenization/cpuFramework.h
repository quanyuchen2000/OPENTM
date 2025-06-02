#pragma once

#include<vector>
#include <thread>
#include "../cmdline.h"
void initDensity_Host(std::vector<float>& rho, cfg::HomoConfig config);
void caculate_rhop(std::vector<float>& rho, std::vector<float>& rhop, cfg::HomoConfig);
void caculate_sens(std::vector<float>& rhosens, std::vector<float>& rhopsens, std::vector<float>& rho, cfg::HomoConfig config);
void update_density_boundary(std::vector<float>& rho, cfg::HomoConfig config);
void build_filter_block(std::vector<float>& rho, std::vector<float>& padded, int blockid, int filter_radius, int blocksize);
void out_filter_block(std::vector<float>& rho, std::vector<float>& padded, int blockid, int filter_radius, int blocksize);
void subtract_mean_parallel(std::vector<float>& A);
void calboundary(std::vector<float>& rho, std::vector<float>& sens, std::vector<float>& boundary, int blockid, int filter_radius, std::vector<std::thread>& workers, std::atomic<int>& counter);
void reboundary(std::vector<float>& sens, std::vector<float>& boundary, int blockid, int fr);
double norm_host(std::vector<float>& A);
void block2lexi(std::vector<float>& rho, std::vector<float>& lexirho, int reso);
void lexi2block(std::vector<float>& lexirho, std::vector<float>& rho, cfg::HomoConfig config);
float find_max_abs(const std::vector<float>& sens);