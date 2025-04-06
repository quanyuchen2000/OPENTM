#pragma once

#include<vector>
#include "../cmdline.h"
void initDensity_Host(std::vector<float>& rho, cfg::HomoConfig config);
void caculate_rhop(std::vector<float>& rho, std::vector<float>& rhop, cfg::HomoConfig);
void caculate_sens(std::vector<float>& rhosens, std::vector<float>& rhopsens, std::vector<float>& rho, cfg::HomoConfig config);
void update_density_boundary(std::vector<float>& rho, cfg::HomoConfig config);
void subtract_mean_parallel(std::vector<float>& A);
double norm_host(std::vector<float>& A);
void block2lexi(std::vector<float>& rho, std::vector<float>& lexirho, cfg::HomoConfig config);
void lexi2block(std::vector<float>& lexirho, std::vector<float>& rho, cfg::HomoConfig config);