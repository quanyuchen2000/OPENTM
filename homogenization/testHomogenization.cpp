#include "homogenization.h"
#include "cuda_runtime.h"
#include "homogenization/utils.h"
#include "templateMatrix.h"
#include "cmdline.h"
#include "tictoc.h"
#include <set>
#include <tuple>
#include "cuda_profiler_api.h"
#include "matlab/matlab_utils.h"
#include <filesystem>
#include <string>
#include <regex>

using namespace homo;

extern void cudaTest(void);