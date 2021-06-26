///////////////////////////////////////////////////
// Copyright (C) 2019 University of Kaiserslautern
// Microelectronic Systems Design Research Group
//
// Vladimir Rybalkin (rybalkin@eit.uni-kl.de)
//
// Last review: 21/11/2019
//
//
//
///////////////////////////////////////////////////

#include "host.hpp"
#include "HardwareDriver.hpp"



std::vector<std::string> open(std::string path)
{
	DIR*    dir;
	dirent* pdir;
	std::vector<std::string> files;

	dir = opendir(path.empty() ? "." : path.c_str());

	while (pdir = readdir(dir)) {
		files.push_back(pdir->d_name);
	}

	closedir(dir);

	std::sort(files.begin(), files.end());
	files.erase(files.begin(), files.begin()+2);

	return files;
}


std::vector<float> ReadImageFromFile(std::string path)
{
	std::vector<float> image;
	std::ifstream input_stream;

	input_stream.open(path, std::ifstream::in);
	if (!input_stream.good()) {
		throw std::runtime_error("No image file found at: " + path);
	}

	float pix;
	unsigned int skip_count = 0;
	unsigned int channel_count = 0;


	while(input_stream >> pix)
	{
		if(skip_count == 0){
			image.push_back(pix);
		}
		channel_count++;
		if(channel_count == 2){
			channel_count = 0;
			skip_count++;
			if(skip_count == INPUT_VAL_SKIP_FAC){
				skip_count = 0;
			}
		}
	}
	input_stream.close();

	return image;
}

unsigned int ReadBinaryFile(std::string path, unsigned int index){
	std::ifstream input_stream;
	input_stream.open(path, std::ifstream::in);

	// Read Header
	for(unsigned int i = 0; i < 16; i++){
		std::string line;
		getline(input_stream, line);
	}

	// Get first "00" byte
	unsigned char temp;
	input_stream.read((char *) &temp, 1);
	while(temp != 0){
		input_stream.read((char *) &temp, 1);
	}

	std::vector<ap_uint<8>> data(SEQUENCE_LENGTH / INPUT_VAL_SKIP_FAC);

	unsigned int skip_count = 0;
	unsigned int channel_count = 0;
	unsigned int data_iter = 0;

	for(int i = 0; i < SEQUENCE_LENGTH; i+=2){
		if(skip_count == 0){

			ap_uint<8> in_0;
			ap_uint<8> in_1;
			ap_uint<8> in_2;
			ap_uint<8> in_3 ;

			input_stream.read((char*) &in_0, 1);
			input_stream.read((char*) &in_1, 1);
			input_stream.read((char*) &in_2, 1);
			input_stream.read((char*) &in_3, 1);

			data[data_iter](3, 0) = in_3(3, 0);
			data[data_iter](7, 4) = in_2(7, 4);
			data[data_iter + 1](3, 0) = in_1(3, 0);
			data[data_iter + 1](7, 4) = in_0(7, 4);

			data_iter+=2;
		}else{
			ap_uint<32> temp;
			input_stream.read((char *) &temp, 4);
		}
		channel_count+=2;
		if(channel_count == 2){
			channel_count = 0;
			skip_count++;
			if(skip_count == INPUT_VAL_SKIP_FAC){
				skip_count = 0;
			}
		}
	}

	std::cout << "End ReadBinaryFile" << std::endl;
	std::cout << "Writing to DRAM" << std::endl;
	
    uint32_t input_address = BASE_INPUT_ADDRESS; //0x30000000;

	uint32_t current_input_address = input_address + data.size() * index;
	std::cout << "Writing image Nr." << index+1 << " to DRAM at address: " << std::hex << current_input_address << std::dec << std::endl;
    writeToDram(data.data(), data.size(), current_input_address);
	std::cout << "Writing done" << std::endl;
	return data.size();
}

template <typename T>
InputImage<T>::InputImage()
{
	image_ = NULL;
	number_of_patches = 0;
}

template <typename T>
InputImage<T>::~InputImage()
{
	if (image_ != NULL)
	{
		delete[] image_;
		image_ = NULL;
	}
}

template <typename T>
template
<
unsigned int TransactionWidth,
unsigned int InputWidth,
typename Pixel_t
>
void InputImage<T>::InitInputImage(std::vector<float> image)
{

	std::cout << "InitInputImage..." << std::endl;

	if (TransactionWidth % InputWidth != 0)
	{
		throw std::runtime_error("ERROR: The transaction width has to be a multiple of an input width");
	}

#if TRANSFORM_INPUTS == 0
	constexpr unsigned int inputs_per_transaction = TransactionWidth / InputWidth;
#else
	constexpr unsigned int inputs_per_transaction = TransactionWidth / INPUT_BITS_PER_VAL;
	constexpr unsigned int input_overhead = TransactionWidth % INPUT_BITS_PER_VAL;
#endif

	std::vector<ap_uint<TransactionWidth> > packed_input;
	unsigned int transactions_per_sequence = image.size() / inputs_per_transaction;

	for(unsigned int tps = 0; tps < transactions_per_sequence; tps++)
	{

		ap_uint<INPUT_BITS_PER_VAL> fpix[inputs_per_transaction];

		for (unsigned int ipt = 0; ipt < inputs_per_transaction; ipt++)
		{
			auto temp = (Pixel_t) image.at(tps * inputs_per_transaction + ipt);
			ap_uint<Pixel_t::width> mask = INPUT_MASK;

			unsigned int out_iter = 0;
			for(unsigned int bit_iter = 0; bit_iter < Pixel_t::width; bit_iter++){
				if(mask[bit_iter] == 1){
					fpix[ipt][out_iter] = temp[bit_iter];
					out_iter++;
				}
			}
		}

		ap_uint<TransactionWidth> pack = Pack<TransactionWidth, ap_uint<INPUT_BITS_PER_VAL>, INPUT_BITS_PER_VAL, inputs_per_transaction>(fpix);

		packed_input.push_back(pack);

	}

	//std::cout << "Allocating a buffer for packed transactions..." << std::endl;

	if (image_ == NULL){
		image_ = new T[packed_input.size()];
		input_image_num_inputs = packed_input.size();
	}
	else
	{
		throw std::runtime_error("ERROR: Internal buffer 'image_' has not a NULL pointer");
	}

	for(int word = 0; word < packed_input.size(); word++){
		image_[word] = packed_input.at(word);
	}

}

template <typename T>
void InputImage<T>::InitGTImage(std::vector<float> image)
{
	if (image.size() != 1)
	{
		throw std::runtime_error("ERROR: Incorrect number of values in the ground truth image");
	}

	std::cout << "Allocating a gt buffer ..." << std::endl;

	if (image_ == NULL)
		image_ = new T[image.size()];
	else
	{
		throw std::runtime_error("ERROR: Internal gt buffer 'image_' has not a NULL pointer");
	}

	for(int i = 0; i < image.size(); i++)
	{
		image_[i] = (T)image.at(i);
	}

	std::cout << "Allocation completed." << std::endl;

}

template <typename T>
template
<
unsigned int TransactionWidth,
unsigned int ClassLabelWidth
>
void InputImage<T>::InitPredictedImage(std::vector<float> image)
{
	unsigned int numberWordsPixels = 1;

	std::cout << "Allocating a buffer for prediction..." << std::endl;

	if (image_ == NULL)
		image_ = new T[numberWordsPixels];
	else
	{
		throw std::runtime_error("ERROR: Internal prediction buffer 'image_' has not a NULL pointer");
	}

	for(int i = 0; i < numberWordsPixels; i++)
		image_[i] = (T)0;

	std::cout << "Allocation completed." << std::endl;
}

void arrhythmia_wrapper(unsigned int numberOfInputsToProcess,
						std::string InputPathDir,
						std::string GTPathDir,
						unsigned int &num_errors,
                        float *compute_time,
						unsigned int numberOfInstances,
						unsigned int iterations)
{
	std::cout << "Preparing inputs ..." << std::endl;

	// Return the list of inputs' file names
	std::vector<std::string> listOfInputs = open(InputPathDir);

	unsigned int number_of_inputs = 0;

	if((numberOfInputsToProcess == 0) || (numberOfInputsToProcess > listOfInputs.size()))
	{
		number_of_inputs = listOfInputs.size();
	}
	else
	{
		number_of_inputs = numberOfInputsToProcess;
	}
	std::cout << "Number of inputs is " << number_of_inputs << std::endl;

	unsigned int input_size;
	for(unsigned int i = 0; i < number_of_inputs; i++)
	{
		std::string path = InputPathDir + listOfInputs.at(i);
		input_size = ReadBinaryFile(path, i);
		std::cout << "ReadBinaryFile done" << std::endl;						   
	}
	//=====================================================================================
	//=====================================================================================

	std::cout << "Preparing GT inputs..." << std::endl;

	// Return the list of images' file names
	std::vector<std::string> listOfGTImages = open(GTPathDir);

	if (listOfInputs.size() != listOfGTImages.size())
	{
		throw std::runtime_error("Number of GT files is not equal to a number of Inputs");
	}

	std::vector<InputImage<unsigned int>> vecGTImages;
	vecGTImages.resize(number_of_inputs);

	for(unsigned int i = 0; i < number_of_inputs; i++)
	{
		std::string path = GTPathDir + listOfGTImages.at(i);
		vecGTImages.at(i).InitGTImage(ReadImageFromFile(path));
		vecGTImages.at(i).image_name = listOfInputs.at(i);

	}

	//=====================================================================================
	//=====================================================================================

	std::cout << "Allocating memory for predicted images..." << std::endl;

	std::vector<InputImage<ap_uint<DATAWIDTH_OUT>>> vecPredictions;
	vecPredictions.resize(number_of_inputs);

	for(unsigned int i = 0; i < number_of_inputs; i++)
	{
		std::string path = GTPathDir + listOfGTImages.at(i);

		vecPredictions.at(i).InitPredictedImage<
												DATAWIDTH_OUT,
												CLASS_LABEL_BITS
												>
												(ReadImageFromFile(path));
												
		vecPredictions.at(i).image_name = listOfInputs.at(i);
	}

	//=====================================================================================
	//=====================================================================================

	// 'vecInputSinus' contains already a desired number of sinus rhythm based on 'numberOfInputsToProcess'
    arrhythmia_on_board<
            DATAWIDTH_IN,
			DATAWIDTH_OUT,
            CLASS_LABEL_BITS>
            (vecPredictions, compute_time, numberOfInstances, iterations, input_size);


	//=====================================================================================
	//=====================================================================================

	CalculateAccuracy
	<
	DATAWIDTH_OUT,
	CLASS_LABEL_BITS
	>
	(vecPredictions, vecGTImages, num_errors);
}



template
<
unsigned int TransactionWidthIn,
unsigned int TransactionWidthOut,
unsigned int ClassLabelWidth
>
void arrhythmia_on_board(   std::vector<InputImage<ap_uint<TransactionWidthOut>>>& vecPrediction,
                            float *computeTime,
                            unsigned int numberOfInstances,
                            unsigned int iterations,
							unsigned int input_size
                            )
{

    assert(!(TransactionWidthIn%8) && "Transaction Width has to be multiple of 8");

	std::cout << "input size: " << input_size << std::endl;

    // Input and output address of images in DRAM
    uint32_t input_address = BASE_INPUT_ADDRESS; 
    uint32_t output_address = BASE_OUTPUT_ADDRESS; 

    uint32_t hw_input_address = BASE_INPUT_ADDRESS;
    uint32_t hw_output_address = BASE_OUTPUT_ADDRESS; 
	
    // Init drivers for all cores
    HardwareDriver* driver_array[numberOfInstances];
    for(unsigned int i=0; i<numberOfInstances; i++){
        driver_array[i] = new HardwareDriver(FIRST_INSTANCE_CTRL_ADDRESS + 0x10000*i);
    }

    unsigned int start_sinus = 0;
    unsigned int finish_sinus;

    if(start_sinus == 0){
        finish_sinus = vecPrediction.size();
    }else{
        finish_sinus = start_sinus + 1;// selective execution of a single image
	}

    if(numberOfInstances > finish_sinus){
        std::cout << "At maximum one instance might be used per Sinus!" << std::endl;
        return;
    }


    //########################################################################
    std::cout << "Start writing to DRAM..." << std::endl;
	std::cout << "TransactionWidthIn: " << TransactionWidthIn << std::endl;
	std::cout << "TransactionWidthOut: " << TransactionWidthOut << std::endl;

    // currently 64 bit are used to store the one output bit
    uint32_t output_width = 1;


    //########################################################################


   //########################################################################
    std::cout << "Start processing..." << std::endl;

    if(iterations > 1){
        int key;
        std::cout << "Press key to continue: " << std::endl;
        std::cin >> key;
    }

    auto start_time = std::chrono::high_resolution_clock::now();

	
    for(unsigned int iter = 0; iter < iterations; iter++){
        bool break_nested = false;
        unsigned int sin_count = 0;

        // Start all instances
        // One instance corresponds to one sinus
        for(int instance=0; instance < numberOfInstances; instance++){
			
            uint32_t current_input_address = hw_input_address+input_size*sin_count;
			uint32_t current_output_address = hw_output_address+output_width*sin_count*(TransactionWidthOut/8);

			driver_array[instance]->setInputs(current_input_address, current_output_address);
            driver_array[instance]->startHardware();
            sin_count++;
        }
		
        // Loop over instances till all patches calculated
        while(!break_nested){
            for(int instance=0; instance < numberOfInstances && !break_nested; instance++){
                if(driver_array[instance]->checkIfFinished()){
					if(sin_count >= finish_sinus){
                        break_nested = true;
                    }else{
						
						uint32_t current_input_address = hw_input_address+input_size*sin_count;
						uint32_t current_output_address = hw_output_address+output_width*sin_count*(TransactionWidthOut/8);

						driver_array[instance]->setInputs(current_input_address, current_output_address);
						driver_array[instance]->startHardware();
						sin_count++;
					}
                }
            }
        }
    }
	
    auto finish_time = std::chrono::high_resolution_clock::now();
    auto ms_compute_time = std::chrono::duration<float, std::chrono::milliseconds::period>(finish_time - start_time);
    *computeTime = ms_compute_time.count();
    std::cout << "Finished processing!" << std::endl;


    //########################################################################
    std::cout << "Start reading from DRAM..." << std::endl;
    for(unsigned int sin = start_sinus; sin < finish_sinus; sin++){
        ap_uint<TransactionWidthOut> *pPrediction = vecPrediction.at(sin).image_;
        readFromDram(pPrediction, output_width, output_address+output_width*sin*(TransactionWidthOut/8));
    }
    std::cout << "End reading from DRAM!" << std::endl;

}



template
<
unsigned int TransactionWidth,
unsigned int ClassLabelWidth
>
void CalculateAccuracy(std::vector<InputImage<ap_uint<TransactionWidth>>>& vecPredictedImages, std::vector<InputImage<unsigned int>>& vecGTImages, unsigned int  &num_errors)
{
	unsigned int start_sinus = 0;
	unsigned int finish_sinus;

	for(unsigned int sin = start_sinus; sin < vecPredictedImages.size(); sin++)
	{

		unsigned int gt = *vecGTImages.at(sin).image_;
		ap_uint<TransactionWidth> guess = *vecPredictedImages.at(sin).image_;

		std::cout << "Checking output for sinus: " << vecGTImages.at(sin).image_name << std::endl;
		std::cout << "Gt: " << gt << std::endl;
		std::cout << "Pred: " << guess << std::endl;

		if(gt != (unsigned int)guess)
		{
			num_errors += 1;
		}
	}
}












