///////////////////////////////////////////////////
// Copyright (C) 2019 University of Kaiserslautern
// Microelectronic Systems Design Research Group
//
// Vladimir Rybalkin (rybalkin@eit.uni-kl.de)
//
// Last review: 06/04/2020
//
//
//
///////////////////////////////////////////////////

#include "host.hpp"
#include <experimental/filesystem>
 
 std::size_t number_of_files_in_directory(std::experimental::filesystem::path path)
{
    using std::experimental::filesystem::directory_iterator;
    return std::distance(directory_iterator(path), directory_iterator{});
}

int main(int argc, const char* argv[])
{
	
	std::string input_dir_path("../sample_dataset/input/");
	std::string gt_dir_path("../sample_dataset/gt/");
	
	std::cout << "Input directory path: " << input_dir_path << std::endl;
	std::cout << "Ground truth directory path: " << gt_dir_path << std::endl;

	float compute_time;

    const int num_input_files = number_of_files_in_directory(input_dir_path);
    const int num_gt_files = number_of_files_in_directory(gt_dir_path);

    // Check if number of gt and input files are the same
    if(num_input_files != num_gt_files){
        throw std::runtime_error("Error: Number of input files is different from number of gt files.");
    }

    const int samples_available = num_input_files;

	int user_input = 0;
	while(user_input == 0){
        
		unsigned int num_errors = 0;
			
        //Get number of samples to process in this run from user
        int number_of_samples;
        bool no_number = true;
        bool wrong_number = true;
        while(no_number || wrong_number){
            std::cout << "Enter Number of Samples to process (from 1 to " << samples_available << "): ";
            std::cin >> number_of_samples;
            if(!std::cin){
                std::cout << "Please enter numbers only." << std::endl;
                std::cin.clear();
                std::cin.ignore(std::numeric_limits<std::streamsize>::max(),'\n'); 
                no_number = true;
            }else if(number_of_samples < 1 || number_of_samples > samples_available){
                std::cout << "Please enter a number in the correct range." << std::endl;
                std::cin.clear();
                std::cin.ignore(std::numeric_limits<std::streamsize>::max(),'\n'); 
                wrong_number = true;                
            }else{
                no_number = false;
                wrong_number = false;
            }
        }
        

		arrhythmia_wrapper(number_of_samples, input_dir_path, gt_dir_path, num_errors, &compute_time, 1, 1);


		std::cout << "The number of errors for the current inputs: " <<  num_errors << std::endl;
		std::cout << std::endl;

        std::cout << "Compute time: " << compute_time << " Milliseconds" << std::endl;
        
		// Show sampel ecg
		no_number = true;
		wrong_number = true;
		int show_sample;
		while(no_number || wrong_number){
			std::cout << "Do you want to see a sample of the detected arrhythmia (enter: 0) or not (enter: -1)? ";
			std::cin >> show_sample;  
			if(!std::cin){
				std::cout << "Please enter numbers only." << std::endl;
				std::cin.clear();
				std::cin.ignore(std::numeric_limits<std::streamsize>::max(),'\n'); 
				no_number = true;
			}else if(user_input < -1 || user_input > 0){
				std::cout << "Please enter a number in the correct range." << std::endl;
				std::cin.clear();
				std::cin.ignore(std::numeric_limits<std::streamsize>::max(),'\n'); 
				wrong_number = true;                
			}else{
				no_number = false;
				wrong_number = false;
			}
		}
		
		if(show_sample == 0){
			system("display ../SW/arrhythmia_sample.png");
		}
		
		
		// Run network again?
		no_number = true;
		wrong_number = true;
		while(no_number || wrong_number){
			std::cout << "Do you want to run the network again (enter: 0) or quit (enter: -1)? ";
			std::cin >> user_input;  
			if(!std::cin){
				std::cout << "Please enter numbers only." << std::endl;
				std::cin.clear();
				std::cin.ignore(std::numeric_limits<std::streamsize>::max(),'\n'); 
				no_number = true;
			}else if(user_input < -1 || user_input > 0){
				std::cout << "Please enter a number in the correct range." << std::endl;
				std::cin.clear();
				std::cin.ignore(std::numeric_limits<std::streamsize>::max(),'\n'); 
				wrong_number = true;                
			}else{
				no_number = false;
				wrong_number = false;
			}
		}
		
    }    

	return 0;
}

