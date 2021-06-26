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

#ifndef __SYNTHESIS__
#include <chrono>       // std::chrono::seconds, std::chrono::duration_cast
#include "json.hpp"
#endif

#include <cstdint>		// std::memcpy()
#include <cstring>		// std::memcpy()
#include <hls_stream.h>
#include <ap_int.h>
#include <stdint.h>
#include <iostream>
#include "config_unroll_1.h"


void TopLevel(ap_uint<DATAWIDTH_IN>* in, ap_uint<DATAWIDTH_OUT>* out);

void TopLevel(
		ap_uint<DATAWIDTH_IN> *in0,
		ap_uint<DATAWIDTH_IN> *in1,
		ap_uint<DATAWIDTH_IN> *in2,
		ap_uint<DATAWIDTH_IN> *in3,
		ap_uint<DATAWIDTH_OUT> *out);

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

template<typename dtype>
void printBits(dtype num){
	for(int i=dtype::width-1; i>=0; i--){
		std::cout << num[i];
	}
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

	InputImage();
	~InputImage();

	template
	<
	unsigned int TransactionWidth,
	unsigned int InputLength,
	unsigned int InputChannels,
	unsigned int InputWidth,
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
void CalculateAccuracy(	std::vector<InputImage<ap_uint<TransactionWidth>>>& vecPredictedImages,
						std::vector<InputImage<unsigned int>>& vecGTImages,
						std::vector<unsigned int>& number_errors,
						float &false_positive_rate,
						float &false_negative_rate);

template
<
unsigned int TRANSACTIONWIDTH_IN,
unsigned int TRANSACTIONWIDTH_OUT,
unsigned int ClassLabelWidth
>
void arrhythmia(std::vector<InputImage<ap_uint<TRANSACTIONWIDTH_IN>>>& vecInputSinus, std::vector<InputImage<ap_uint<TRANSACTIONWIDTH_OUT>>>& vecPrediction);


template
<
unsigned int TRANSACTIONWIDTH_IN,
unsigned int TRANSACTIONWIDTH_OUT,
unsigned int ClassLabelWidth
>
void arrhythmia_ecg(std::vector<std::vector<ap_uint<TRANSACTIONWIDTH_IN>>>& vecInputSinus, std::vector<InputImage<ap_uint<TRANSACTIONWIDTH_OUT>>>& vecPrediction);



void arrhythmia_wrapper(unsigned int numberOfInputsToProcess,
				 	 	std::string InputPathDir,
						std::string GTPathDir,
						std::vector<unsigned int>& number_errors,
						float &false_positive_rate,
						float &false_negative_rate);


#endif
