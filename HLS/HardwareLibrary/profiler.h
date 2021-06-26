#pragma once

#ifndef __SYNTHESIS__
#include "json.hpp"
#include <iostream>
#endif

#ifdef main_translation_unit
#define EXTERN
#else
#define EXTERN extern
#endif

class Profiler {
private:
	float min_val;
	float max_val;

public:
	Profiler() : min_val(std::numeric_limits<float>::max()), max_val(std::numeric_limits<float>::min()){}

	template<typename val_t> void update(val_t new_val){
		float new_val_float = static_cast<float>(new_val);
		if(min_val > new_val_float){
			min_val = new_val_float;
		}
		if(max_val < new_val_float){
			max_val = new_val_float;
		}
	}

	float get_min(){
		return min_val;
	}

	float get_max(){
		return max_val;
	}

#ifndef __SYNTHESIS__
	void print_profiled_values(std::string layer_name, std::string value_name){
		std::cout << "==================================" << std::endl;
		std::cout << "Min " << layer_name << " " << value_name << ": " << min_val << std::endl;
		std::cout << "Max " << layer_name << " " << value_name << ": " << max_val << std::endl;
		std::cout << "==================================" << std::endl;
	}
#endif

};

class Profiler_MAC {
	
private:
	Profiler profiler_in;
	Profiler profiler_mul;
	Profiler profiler_acc;
	Profiler profiler_out;
	
public:
	
	template<typename val_t> void update_in(val_t new_val){
		profiler_in.update(new_val);
	}

	template<typename val_t> void update_mul(val_t new_val){
		profiler_mul.update(new_val);
	}
	
	template<typename val_t> void update_acc(val_t new_val){
		profiler_acc.update(new_val);
	}
	
	template<typename val_t> void update_out(val_t new_val){
		profiler_out.update(new_val);
	}
	
#ifndef __SYNTHESIS__
	void print_profiled_values(std::string layer_name){
		//profiler_in.print_profiled_values(layer_name, "in");
		profiler_mul.print_profiled_values(layer_name, "mul");
		profiler_acc.print_profiled_values(layer_name, "acc");
		profiler_out.print_profiled_values(layer_name, "out");
		std::cout << "---------------------------------------------------" << std::endl;

	}
#endif

#ifndef __SYNTHESIS__
	std::string get_json_string(std::string layer_name, int index, std::string type, std::string sub_type=""){
		nlohmann::json j;
		j["name"] = layer_name;
		j["type"] = type;
		j["index"] = index;
		if(sub_type != ""){
			j["sub_type"] = sub_type;
		}
		j["MUL"]["min"] = profiler_mul.get_min();
		j["MUL"]["max"] = profiler_mul.get_max();
		j["ACC"]["min"] = profiler_acc.get_min();
		j["ACC"]["max"] = profiler_acc.get_max();
		j["OA"]["min"] = profiler_out.get_min();
		j["OA"]["max"] = profiler_out.get_max();

		return("[ " +  j.dump(4) + " ]");
	}
#endif

};


class Profiler_Conv_DS {
public:
	Profiler_MAC profiler_mac_depthwise;
	Profiler_MAC profiler_mac_pointwise;

#ifndef __SYNTHESIS__
	void print_profiled_values(std::string layer_name){
		std::cout << "Profiled Values Depthwise: " << std::endl;
		profiler_mac_depthwise.print_profiled_values(layer_name);
		std::cout << "Profiled Values Pointwise: " << std::endl;
		profiler_mac_pointwise.print_profiled_values(layer_name);
		std::cout << std::endl;
	}
#endif

#ifndef __SYNTHESIS__
	std::string get_json_string(std::string layer_name, int index){
		std::string json_depthwise = profiler_mac_depthwise.get_json_string(layer_name + "_DEPTHWISE", index, "CONV1D_DS", "DEPTHWISE");
		std::string json_pointwise = profiler_mac_depthwise.get_json_string(layer_name + "_POINTWISE", index, "CONV1D_DS", "POINTWISE");
		return (json_depthwise + ",\n" + json_pointwise);
	}
#endif

};


class Profiler_GAP {

private:
	Profiler profiler_in;
	Profiler profiler_acc;
	Profiler profiler_out;

public:

	template<typename val_t> void update_in(val_t new_val){
		profiler_in.update(new_val);
	}

	template<typename val_t> void update_acc(val_t new_val){
		profiler_acc.update(new_val);
	}

	template<typename val_t> void update_out(val_t new_val){
		profiler_out.update(new_val);
	}

#ifndef __SYNTHESIS__
	void print_profiled_values(std::string layer_name){
		profiler_in.print_profiled_values(layer_name, "IN");
		profiler_acc.print_profiled_values(layer_name, "ACC");
		profiler_out.print_profiled_values(layer_name, "OUT");
		std::cout << "---------------------------------------------------" << std::endl;

	}
#endif

#ifndef __SYNTHESIS__
	std::string get_json_string(std::string layer_name, int index){
		nlohmann::json j;
		j["name"] = layer_name;
		j["type"] = "GLOBALAVERAGEPOOL1D";
		j["index"] = index;
		j["ACC"]["min"] = profiler_acc.get_min();
		j["ACC"]["max"] = profiler_acc.get_max();
		j["OA"]["min"] = profiler_out.get_min();
		j["OA"]["max"] = profiler_out.get_max();

		return("[ " +  j.dump(4) + " ]");
	}
#endif

};

EXTERN Profiler_Conv_DS profiler_conv1d_ds_0;
EXTERN Profiler_Conv_DS profiler_conv1d_ds_1;
EXTERN Profiler_Conv_DS profiler_conv1d_ds_2;
EXTERN Profiler_Conv_DS profiler_conv1d_ds_3;
EXTERN Profiler_Conv_DS profiler_conv1d_ds_4;
EXTERN Profiler_Conv_DS profiler_conv1d_ds_5;
EXTERN Profiler_Conv_DS profiler_conv1d_ds_6;
EXTERN Profiler_Conv_DS profiler_conv1d_ds_7;
EXTERN Profiler_Conv_DS profiler_conv1d_ds_8;
EXTERN Profiler_Conv_DS profiler_conv1d_ds_9;
EXTERN Profiler_Conv_DS profiler_conv1d_ds_10;

EXTERN Profiler_MAC profiler_conv1d_0;
EXTERN Profiler_MAC profiler_conv1d_1;
EXTERN Profiler_MAC profiler_conv1d_2;
EXTERN Profiler_MAC profiler_conv1d_3;
EXTERN Profiler_MAC profiler_conv1d_4;
EXTERN Profiler_MAC profiler_conv1d_5;
EXTERN Profiler_MAC profiler_conv1d_6;
EXTERN Profiler_MAC profiler_conv1d_7;
EXTERN Profiler_MAC profiler_conv1d_8;
EXTERN Profiler_MAC profiler_conv1d_9;
EXTERN Profiler_MAC profiler_conv1d_10;

EXTERN Profiler_GAP profiler_globalaveragepool1d_0;
EXTERN Profiler_MAC profiler_fc1d_0;



