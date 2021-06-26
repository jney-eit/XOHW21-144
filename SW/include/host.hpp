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

#ifndef HOST_HPP
#define HOST_HPP


#include <stdexcept>
#include <string>       // std::string
#include <iostream>     // std::cout, std::cerr
#include <fstream>      // std::ifstream std::ofstream
#include <vector>
#include <math.h>       // tanh, log
#include <dirent.h>
#include <sys/types.h>
#include <algorithm>	// std::sort
#include <ctype.h>		// isspace()
#include <assert.h>

#ifndef __SYNTHESIS__
#include <chrono>       // std::chrono::seconds, std::chrono::duration_cast
#endif

#include <cstdint>		// std::memcpy()
#include <cstring>		// std::memcpy()
#include <unistd.h>
#include <stdio.h>
#include <hls_stream.h>
#include <ap_int.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <iostream>
#include "stdbool.h"
#include "config.hpp"


std::vector<std::string> open(std::string path);
	
template
<
unsigned int DataWidthPacked,
typename t_pixel,
unsigned int PixelWidth,
unsigned int NumberEntries
>
ap_uint<DataWidthPacked> Pack(t_pixel x[NumberEntries])
{
	constexpr unsigned int mask = (1 << PixelWidth) - 1;
	ap_uint<DataWidthPacked> packed = 0;

	for (int count = NumberEntries - 1; count >= 0; count--)
	{
		packed = packed << PixelWidth;
		ap_uint<PixelWidth> uValue = *reinterpret_cast<ap_uint<PixelWidth> *>(&x[count]);
		packed(PixelWidth-1,0) = (uValue&mask);
	}
	return packed;	
}
	
template
<
unsigned int DatawidthPacked,
typename t_pixel,
unsigned int PixelWidth,
unsigned int NumberEntries
>
void UnPack(ap_uint<DatawidthPacked> pack, t_pixel x[NumberEntries])
{
	ap_uint<DatawidthPacked> temp;
	ap_uint<DatawidthPacked> mask = 1;

	for(unsigned int i = 0; i < PixelWidth-1; i++)
	{
		mask = (mask << 1) | mask;
	}

	for(unsigned int i = 0; i < NumberEntries; i++)
	{
		temp = (pack >> (i * PixelWidth));
		temp = temp & mask;
		x[i] = *((t_pixel*)&temp);
	}
}

template <typename T>
class InputImage {
  public:
	unsigned int input_image_num_inputs = 0;

	InputImage();
	~InputImage();
	
	template
	<
	unsigned int TransactionWidth,
	unsigned int InputLength,
	typename Pixel_t
	>
	void InitInputImage(std::vector<float> image);

	void InitGTImage(std::vector<float> image);

	template
	<
	unsigned int TransactionWidth,
	unsigned int ClassLabelWidth
	>
	void InitPredictedImage(std::vector<float> image);

	T *image_;
	std::string image_name; 
	
	unsigned int number_of_patches;

  protected:
  private:
};

std::vector<float> ReadImageFromFile(std::string path);
std::vector<uint8_t> ReadGTFromFile(std::string path);


template
<
unsigned int TransactionWidth,
unsigned int ClassLabelWidth
>
void CalculateAccuracy(std::vector<InputImage<ap_uint<TransactionWidth>>>& vecPredictedImages, std::vector<InputImage<unsigned int>>& vecGTImages, unsigned int &num_errors);


template
<
unsigned int TransactionWidthIn,
unsigned int TransactionWidthOut,
unsigned int ClassLabelWidth
>
void arrhythmia_on_board( 
                            std::vector<InputImage<ap_uint<TransactionWidthOut>>>& vecPrediction,
                            float *computeTime,
                            unsigned int numberOfInstances,
                            unsigned int iterations,
							unsigned int input_size
                            );

						 
						 
void arrhythmia_wrapper(unsigned int numberOfInputsToProcess,
				 	 	std::string InputPathDir,
						std::string GTPathDir,
						unsigned int &num_errors,
						float *compute_time = nullptr,
						unsigned int numberOfInstances = 1,
						unsigned int iterations = 1);

#endif
