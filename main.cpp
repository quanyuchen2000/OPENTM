#include <iostream>
#include "cmdline.h"
#include "openvdb/tools/VolumeToMesh.h"
#include "voxelIO/openvdb_wrapper_t.h"
#include <fstream>
#include <sstream>
#include <algorithm>
// include files for write things in main
extern void cuda_test(void);
extern void testAutoDiff(void);
extern void test_MMA(cfg::HomoConfig, int mode);
extern std::vector<float> runCustom(cfg::HomoConfig config, std::vector<float>* rho0 = nullptr);
namespace homo {
	extern std::string setPathPrefix(const std::string& fprefix);
}

class wrapper_homo {
	cfg::HomoConfig config;
	std::vector<float> rho;
public:
	wrapper_homo() { config.init(); };
	~wrapper_homo() {};
	void setDensity(std::vector<float> rho_) {
		rho = rho_;
	}
	void setConfig(int reso, std::vector<double> heat_ratios, std::vector<double> target_ratio, cfg::Model model) {
		config.init();
		config.reso[0] = reso;
		config.reso[1] = reso;
		config.reso[2] = reso;
		if (heat_ratios.size() != 2) {
			std::cout << "Input wrong size of heat_ratios" << std::endl;
			exit(-1);
		}
		for (int i = 0; i < 2; i++)
			config.heatRatio[i] = heat_ratios[i];
		for (int i = 0; i < 6; i++)
			config.target_tensor[i] = target_ratio[i];
		config.model = model;
	}
	std::vector<float> optimize() {
		return runCustom(config, &rho);
	}
};
std::vector<float> runInstance(int reso, std::vector<double> heat_ratios, std::vector<double> target_ratio, cfg::InitWay initway, cfg::Model model) {
	cfg::HomoConfig config;
	config.init();
	config.reso[0] = reso;
	config.reso[1] = reso;
	config.reso[2] = reso;
	if (heat_ratios.size() != 2) {
		std::cout << "Input wrong size of heat_ratios" << std::endl;
		exit(-1);
	}
	for (int i = 0; i < 2; i++)
		config.heatRatio[i] = heat_ratios[i];
	for (int i = 0; i < 6; i++)
		config.target_tensor[i] = target_ratio[i];
	config.winit = initway;
	config.model = model;
	return runCustom(config);
}

int main()
{
	//cfg::HomoConfig config;
	//config.init();

	//std::cout << "Hello World!\n";
	//cuda_test();
	//homo::setPathPrefix(config.outprefix);
	//try {
	//	//testHomogenization(config);
	//	//runInstance(config);
	//test_MMA(config, 2);
	//}
	//catch (std::runtime_error e) {
	//	std::cout << "\033[31m" << "Exception occurred: " << std::endl << e.what() << std::endl << ", aborting..." << "\033[0m" << std::endl;
	//	exit(-1);
	//}
	//catch (...) {
	//	std::cout << "\033[31m" << "Unhandled Exception occurred, aborting..." << "\033[0m" << std::endl;
	//	exit(-1);
	//}
	runInstance(128, { 1, 1e-4 }, { 0.3,0.2,0.1,0.1,0.05,0.05 }, cfg::InitWay::IWP, cfg::Model::oc);
	return 0;
}

/* below is the code for user to bind python .pyd file*/
//#include <pybind11/pybind11.h>
//#include <pybind11/stl.h>
//namespace py = pybind11;
//PYBIND11_MODULE(openTM, m) {
//	m.doc() = "....";
//	py::enum_<cfg::InitWay>(m, "InitWay")
//		.value("random", cfg::InitWay::random)
//		.value("randcenter", cfg::InitWay::randcenter)
//		.value("noise", cfg::InitWay::noise)
//		.value("manual", cfg::InitWay::manual)
//		.value("interp", cfg::InitWay::interp)
//		.value("rep_randcenter", cfg::InitWay::rep_randcenter)
//		.value("P", cfg::InitWay::P)
//		.value("G", cfg::InitWay::G)
//		.value("D", cfg::InitWay::D)
//		.value("IWP", cfg::InitWay::IWP)
//		.value("example", cfg::InitWay::example);
//
//	py::enum_<cfg::Model>(m, "Model")
//		.value("mma", cfg::Model::mma)
//		.value("oc", cfg::Model::oc);
//	m.def("runInstance", &runInstance, py::arg("reso"), py::arg("heat_ratios"), py::arg("target_ratio"),
//		py::arg("initway") = cfg::InitWay::IWP, py::arg("model") = cfg::Model::mma);
//
//	py::class_<wrapper_homo>(m, "homo")
//		.def(py::init<>())
//		.def("setDensity", &wrapper_homo::setDensity, "set density by given rho")
//		.def("setConfig", &wrapper_homo::setConfig, py::arg("reso"), py::arg("heat_ratios"), py::arg("target_ratio"), py::arg("model") = cfg::Model::oc)
//		.def("optimize", &wrapper_homo::optimize, "optimization");
//}

///* below is the code for user to build matlab .mex64w file*/
//#include "mex.hpp"
//#include "mexAdapter.hpp"
//#include "MatlabDataArray/TypedArray.hpp"
//class MexFunction : public matlab::mex::Function {
//    std::shared_ptr<matlab::engine::MATLABEngine> matlabPtr = getEngine();
//    matlab::data::ArrayFactory factory;
//    std::ostringstream stream;
//public:
//    void operator()(matlab::mex::ArgumentList outputs, matlab::mex::ArgumentList inputs) {
//        checkArguments(outputs, inputs);
//        double trran = inputs[0][0];
//        int reso = int(trran);
//        matlab::data::TypedArray<double> heat_ratiom = std::move(inputs[1]);
//       std::vector<double> heat_ratio(2);
//        for (int i = 0; i < 2; i++)
//            heat_ratio[i] = heat_ratiom[i];
//        matlab::data::TypedArray<double> ttm = std::move(inputs[2]);
//        std::vector<double> target_tensor(6);
//        for (int i = 0; i < 6; i++)
//            target_tensor[i] = ttm[i];
//        cfg::Model model;
//        matlab::data::StringArray inArrayRef1(inputs[3]);
//        std::string tstring = (std::string)inArrayRef1[0];
//        if (tstring == "mma")
//            model = cfg::Model::mma;
//        else if (tstring == "oc")
//            model = cfg::Model::oc;
//        std::vector<float> result;
//        if (inputs.size() < 4)
//            result = runInstance(reso,heat_ratio,target_tensor,cfg::InitWay::IWP,model);
//        else {
//            std::vector<float> _rho(reso*reso*reso);
//            if (inputs[4].getNumberOfElements() != pow(reso, 3)) {
//                matlabPtr->feval(u"error",
//                    0, std::vector<matlab::data::Array>({ factory.createScalar("Scale of rho wrong!") }));
//            }
//            else {
//                matlab::data::TypedArray<double> rhom = std::move(inputs[2]);
//                for (int i = 0; i < reso * reso * reso; i++) {
//                    _rho[i] = rhom[i];
//                }
//            }
//            wrapper_homo homo;
//            homo.setConfig(reso, heat_ratio, target_tensor, model);
//            homo.setDensity(_rho);
//            homo.optimize();
//        }
//        const float* start = &result[0];
//        const float* end = &result[reso * reso * reso];
//        unsigned long long t = reso * reso * reso;
//        outputs[0] = factory.createArray<float>({1, t}, start, end);
//    }
//    void displayOnMATLAB(std::ostringstream& stream) {
//        // Pass stream content to MATLAB fprintf function
//        matlabPtr->feval(u"fprintf", 0,
//            std::vector<matlab::data::Array>({ factory.createScalar(stream.str()) }));
//        // Clear stream buffer
//        stream.str("");
//    }
//    void checkArguments(matlab::mex::ArgumentList outputs, matlab::mex::ArgumentList inputs) {
//        std::shared_ptr<matlab::engine::MATLABEngine> matlabPtr = getEngine();
//        matlab::data::ArrayFactory factory;
//
//        if (inputs.size() < 3 || inputs.size() > 5) {
//            matlabPtr->feval(u"error",
//                0, std::vector<matlab::data::Array>({ factory.createScalar("Two inputs required") }));
//        }
//
//        if (inputs[0].getNumberOfElements() != 1) {
//            matlabPtr->feval(u"error",
//                0, std::vector<matlab::data::Array>({ factory.createScalar("Input resolution must be a scalar") }));
//        }
//
//        if (inputs[0].getType() != matlab::data::ArrayType::DOUBLE) {
//            matlabPtr->feval(u"error",
//                0, std::vector<matlab::data::Array>({ factory.createScalar("Input resolution must be a integer") }));
//        }
//
//        if (inputs[1].getNumberOfElements() != 2) {
//            matlabPtr->feval(u"error",
//                0, std::vector<matlab::data::Array>({ factory.createScalar("Input heat ratios of black & white 2 materials") }));
//        }
//        if (inputs[1].getType() != matlab::data::ArrayType::DOUBLE ||
//            inputs[1].getType() == matlab::data::ArrayType::COMPLEX_DOUBLE) {
//            matlabPtr->feval(u"error",
//                0, std::vector<matlab::data::Array>({ factory.createScalar("Heat ratios be type double") }));
//        }
//
//        if (inputs[2].getNumberOfElements() != 6) {
//            matlabPtr->feval(u"error",
//                0, std::vector<matlab::data::Array>({ factory.createScalar("Aimed tensor [xx yy zz xy yz xz] is needed") }));
//        }
//        if (inputs[2].getType() != matlab::data::ArrayType::DOUBLE ||
//            inputs[2].getType() == matlab::data::ArrayType::COMPLEX_DOUBLE) {
//            matlabPtr->feval(u"error",
//                0, std::vector<matlab::data::Array>({ factory.createScalar("Aimed tensor be type double") }));
//        }
//        if (inputs.size() > 3) {
//            if (inputs[3].getNumberOfElements() != 1) {
//                matlabPtr->feval(u"error",
//                    0, std::vector<matlab::data::Array>({ factory.createScalar("Optimizer need to be a string use \"\" may help") }));
//            }
//            matlab::data::StringArray inArrayRef1(inputs[3]);
//            std::string tstring = (std::string)inArrayRef1[0];
//            std::vector<std::string> ienum = { "oc", "mma"};
//            if (inputs[3].getType() != matlab::data::ArrayType::MATLAB_STRING || std::find(ienum.begin(), ienum.end(), tstring) == ienum.end()) {
//                matlabPtr->feval(u"error",
//                    0, std::vector<matlab::data::Array>({ factory.createScalar("Optimizer type error") }));
//            }
//        }
//        if (inputs.size() > 4) {
//            if (inputs[4].getType() != matlab::data::ArrayType::DOUBLE ||
//                inputs[4].getType() == matlab::data::ArrayType::COMPLEX_DOUBLE) {
//                matlabPtr->feval(u"error",
//                    0, std::vector<matlab::data::Array>({ factory.createScalar("Aimed tensor be type double") }));
//            }
//        }
//    }
//};