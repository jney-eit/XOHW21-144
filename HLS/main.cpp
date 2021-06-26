#include "host.hpp"
#define main_translation_unit
#include "profiler.h"


int main(int argc, const char* argv[])
{
	std::string current_file_path =  __FILE__;

	if (argc != 3)
	{
		std::cout << "2 parameters are needed: " << std::endl;
		std::cout << "1 - Input sample directory" << std::endl;
		std::cout << "2 - Ground truth directory" << std::endl;
		return 1;
	}

	std::string input_dir_path(argv[1]);
	std::string gt_dir_path(argv[2]);

	std::cout << "Input images directory path: " << input_dir_path << std::endl;
	std::cout << "Input GT path: " << gt_dir_path << std::endl;
	unsigned int numberOfInputsToProcess = 500;
	std::vector<unsigned int> errors;

	std::cout << "CONV1D_DS_0_IFM_DIM: " << CONV1D_DS_0_IFM_DIM << std::endl;
	std::cout << "RELU1D_0_IFM_DIM: " << RELU1D_0_IFM_DIM << std::endl;
	std::cout << "CONV1D_DS_1_IFM_DIM: " << CONV1D_DS_1_IFM_DIM << std::endl;
	std::cout << "RELU1D_1_IFM_DIM: " << RELU1D_1_IFM_DIM << std::endl;
	std::cout << "CONV1D_DS_2_IFM_DIM: " << CONV1D_DS_2_IFM_DIM << std::endl;
	std::cout << "RELU1D_2_IFM_DIM: " << RELU1D_2_IFM_DIM << std::endl;
	std::cout << "CONV1D_DS_3_IFM_DIM: " << CONV1D_DS_3_IFM_DIM << std::endl;
	std::cout << "RELU1D_3_IFM_DIM: " << RELU1D_3_IFM_DIM << std::endl;
	std::cout << "MAXPOOL1D_0_IFM_DIM: " << MAXPOOL1D_0_IFM_DIM << std::endl;
	std::cout << "MAXPOOL1D_1_IFM_DIM: " << MAXPOOL1D_1_IFM_DIM << std::endl;
	std::cout << "GLOBALAVERAGEPOOL1D_0_IFM_DIM: " << GLOBALAVERAGEPOOL1D_0_IFM_DIM << std::endl;

	float false_positive_rate = 0;
	float false_negative_rate = 0;
	arrhythmia_wrapper(numberOfInputsToProcess, input_dir_path, gt_dir_path, errors, false_positive_rate, false_negative_rate);

	for(unsigned int sinus = 0; sinus < errors.size(); sinus++){
		printf("Sinus %d is recognized %s \n", sinus, errors.at(sinus) ? "WRONG" : "CORRECT");
	}
	std::cout << std::endl;

	std::cout << "False positive rate: " << false_positive_rate << std::endl;
	std::cout << "False negative rate: " << false_negative_rate << std::endl;

	std::size_t hls_dir_pos = current_file_path.find("main.cpp");
	if(USE_PROFILER){
		std::string out_accuracy_json_file = current_file_path.substr(0, hls_dir_pos) + "accuracy_fp.json";
		nlohmann::json j;
		j["false_positive_rate"] = false_positive_rate;
		j["false_negative_rate"] = false_negative_rate;
		std::ofstream accuracy_json;
		accuracy_json.open(out_accuracy_json_file);
		accuracy_json << j.dump(4) << std::endl;
		accuracy_json.close();

		std::string out_act_json_file = current_file_path.substr(0, hls_dir_pos) + "profiled_activations.json";

		std::ofstream act_json;
		act_json.open(out_act_json_file);
		act_json << "[" << std::endl;
		act_json << profiler_conv1d_ds_0.get_json_string("CONV1D_DS_0", 0) << ", " << std::endl;
		act_json << profiler_conv1d_ds_1.get_json_string("CONV1D_DS_1", 1) << ", " << std::endl;
		act_json << profiler_conv1d_ds_2.get_json_string("CONV1D_DS_2", 2) << ", " << std::endl;
		act_json << profiler_conv1d_ds_3.get_json_string("CONV1D_DS_3", 3) << ", " << std::endl;
		act_json << profiler_globalaveragepool1d_0.get_json_string("GLOBALAVERAGEPOOL1D_0", 0) << ", " << std::endl;
		act_json << profiler_fc1d_0.get_json_string("FC1D_0", 0, "FC1D") << std::endl;
		act_json << "]" << std::endl;
		act_json.close();
	}else{
		std::string out_accuracy_json_file = current_file_path.substr(0, hls_dir_pos) + "accuracy_quant.json";
		nlohmann::json j;
		j["false_positive_rate"] = false_positive_rate;
		j["false_negative_rate"] = false_negative_rate;
		std::ofstream accuracy_json;
		accuracy_json.open(out_accuracy_json_file);
		accuracy_json << j.dump(4) << std::endl;
		accuracy_json.close();
	}

	return 0;
}
