#pragma once

#include<vector>
#include "../cmdline.h"
void initDensity_Host(std::vector<float>& rho, cfg::HomoConfig config);
void caculate_rhop(std::vector<float>& rho, std::vector<float>& rhop, cfg::HomoConfig);
void caculate_sens(std::vector<float>& rhosens, std::vector<float>& rhopsens, std::vector<float>& rho, cfg::HomoConfig config);
void update_density_boundary(std::vector<float>& rho, cfg::HomoConfig config);
void update_vertex_boundary();
