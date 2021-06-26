#pragma once

#include <assert.h>
#include <ap_int.h>

// Compile Time Greatest Common Multiple
template<int N, int M, int K>
class A;

template<int N, int M>
class GCD {
public:
    static const int a = A<N, M, N % M>::a;
};

template<int N, int M, int K>
class A {
public:
    static const int a = A<M, K, M % K>::a;
};

template<int N, int M>
class A<N, M, 0> {
public:
    static const int a = M;
};


template<typename dtype>
void printBits(dtype num){
	for(int i=dtype::width-1; i>=0; i--){
		std::cout << num[i];
	}
}


template<typename dtype,
		 unsigned int mask,
		 unsigned int input_bits_per_val>
ap_uint<dtype::width> TranformVal(ap_uint<input_bits_per_val> masked_val, const ap_fixed<32, 16> factor, const ap_fixed<32, 16> offset){

	const ap_uint<16> mask_ap_uint = mask;
	ap_uint<16> temp_val_little_endian = 0;

	// Demask first value, fill up unset bits with zero
	unsigned int in_iter = 0;
	for(unsigned int bit_iter = 0; bit_iter < 16; bit_iter++){
		if(mask_ap_uint[bit_iter] == 1){
			temp_val_little_endian[bit_iter] = masked_val[in_iter];
			in_iter++;
		}
	}

	// Swap bytes
	ap_uint<16> temp_val_big_endian;
	temp_val_big_endian(15, 8) = temp_val_little_endian(7, 0);
	temp_val_big_endian(7, 0) = temp_val_little_endian(15, 8);
	// Interpret bits as ap_int
	ap_int<16> raw_val = *reinterpret_cast<ap_int<16>*>(&temp_val_big_endian);
	// Cast to ap_fixed to apply normalization
	ap_fixed<32, 16> fixed_val = static_cast<ap_fixed<32, 16>>(raw_val);
	fixed_val = (fixed_val - offset) * factor;
	// Cast to dtype of next layer
	dtype out_val = static_cast<dtype>(fixed_val);
	ap_uint<dtype::width> out_cast_0 = *reinterpret_cast<ap_uint<dtype::width>*>(&out_val);
	return out_cast_0;

}



/*
 *
 * Demask the input bits
 * Normalizes the inputs according to offset and factor
 * Casts the inputs to the type of the next layer
 *
 */
template<
		 unsigned int mask,
		 unsigned int input_bits_per_val,
		 unsigned int NumOutputs,
		 typename out_dtype,
		 int DATAWIDTH_IN,
		 int DATAWIDTH_OUT
		 >
void InputTransform(
		hls::stream<ap_uint<DATAWIDTH_IN>> &in,
		hls::stream<ap_uint<DATAWIDTH_OUT>> &out){

	const ap_fixed<32, 16> factor = (1.0 / 256.0);
	const ap_fixed<32, 16> offset = 2048;

	// Input values are 16 bit, little endian signed integer
	constexpr unsigned int raw_vals_per_in = DATAWIDTH_IN / input_bits_per_val;

	ap_uint<bitsNeeded(NumOutputs)> out_count = 0;
	ap_uint<DATAWIDTH_OUT> output = 0;
	ap_uint<DATAWIDTH_IN> input = 0;

	for(unsigned int iter = 0; iter < NumOutputs; iter++){
		#pragma HLS PIPELINE
		input = in.read();
		for(unsigned int val_iter = 0; val_iter < raw_vals_per_in; val_iter+=2){
			ap_uint<input_bits_per_val> masked_val = input((val_iter + 1) * input_bits_per_val - 1, val_iter * input_bits_per_val);
			ap_uint<out_dtype::width> out_cast_0 = TranformVal<out_dtype, mask, input_bits_per_val>(masked_val, factor, offset);

			masked_val = input(((val_iter+1) + 1) * input_bits_per_val - 1, (val_iter+1) * input_bits_per_val);
			ap_uint<out_dtype::width> out_cast_1 = TranformVal<out_dtype, mask, input_bits_per_val>(masked_val, factor, offset);

			// Swap values
			output(val_iter*out_dtype::width + (out_dtype::width*2 - 1), val_iter*out_dtype::width + out_dtype::width) = out_cast_0;
			output(val_iter*out_dtype::width + (out_dtype::width - 1), val_iter*out_dtype::width) = out_cast_1;
		}
		out.write(output);
	}

}


template<
        unsigned int InWidth,			// width of input stream
        unsigned int OutWidth,	    // width of output stream
		unsigned int OUT_PER_CC=1
        >
void StreamingDataWidthConverterFixed(
        hls::stream<ap_uint<InWidth> > &in,
        hls::stream<ap_uint<OutWidth> > &out,
        const unsigned int NumWords)
{

    if (InWidth > OutWidth)
    {
        // emit multiple output words per read input word
    	constexpr unsigned int IN_PER_CC = (OUT_PER_CC == 1) ? 1 : (OUT_PER_CC / (InWidth/OutWidth));

        // assert on loop boundary to optimize HLS
        for (unsigned int inw = 0; inw < NumWords/IN_PER_CC; inw++)
        {
			#pragma HLS PIPELINE
        	for(unsigned int unroll_iter=0; unroll_iter < IN_PER_CC; unroll_iter++){
				#pragma HLS UNROLL
				ap_uint<InWidth> inWord = in.read();
				for (unsigned int outw = 0; outw < InWidth / OutWidth; outw++){
					#pragma HLS UNROLL
					ap_uint<OutWidth> outWord = inWord(OutWidth - 1, 0);
					out.write(outWord);
					inWord = inWord >> OutWidth;
				}
        	}
        }
    }
    else if (InWidth == OutWidth)
    {
        // straight-through copy
        for (unsigned int w = 0; w < NumWords; w++)
        {
#pragma HLS PIPELINE II=1
            ap_uint<InWidth> val = in.read();
            out.write(val);
        }
    }
    else
    {

    	unsigned out_count = 0;

        // read multiple input words per output word emitted
        //const unsigned int num_packets = NumWords / (OutWidth/MaxInWidth);
        for (unsigned int p = 0; p < NumWords / (OutWidth/InWidth); p++)
        {
			#pragma HLS PIPELINE II=1
            ap_uint<OutWidth> outWord = 0;
            for (unsigned int inw = 0; inw < OutWidth/InWidth; inw++)
            {
                ap_uint<OutWidth> inWord = (ap_uint<OutWidth>) in.read();
                inWord = inWord << (OutWidth - InWidth);
                outWord = outWord >> InWidth;
                outWord = outWord | inWord;
            }
            out_count++;
            out.write(outWord);
        }
        // Discard left inputs if NumWords is not multiple of OutWidth/InWidth
        for(int i=0; i < NumWords % (OutWidth/InWidth); i++){
        	in.read();
        }

        std::cout << "Num out DWC: " << out_count << std::endl;
        exit(10);

    }
}





template<unsigned int NumInputs, unsigned int NumOutputs, int InputWidth, int OutputWidth, bool cut_last=true>
void StreamingDataWidthConverter(hls::stream<ap_uint<InputWidth>> &in_stream, hls::stream<ap_uint<OutputWidth>> &out_stream) {

    // Get greatest common multiple at compile time
    unsigned int constexpr gcd = static_cast<const unsigned int>(GCD<InputWidth, OutputWidth>::a);
    unsigned int output_count = 0;
    unsigned int input_count = 0;
    unsigned int total_output_count = 0;

    ap_uint<InputWidth> input = 0;
    ap_uint<OutputWidth> output = 0;

    for (unsigned int val_count = 0; val_count < (NumInputs * InputWidth) / gcd; val_count++) {

        #pragma HLS PIPELINE

    	if (input_count == 0) {
        	const unsigned int read_occurence = InputWidth / gcd;
            input = in_stream.read();
        }
        output((output_count + 1) * gcd - 1, output_count * gcd) = input((input_count + 1) * gcd - 1,
                                                                         input_count * gcd);
        output_count++;
        input_count++;

        if (output_count >= OutputWidth / gcd && total_output_count < NumOutputs) {
        	const unsigned int write_occurence = OutputWidth / gcd;
            out_stream.write(output);
            total_output_count++;
            output_count = 0;
        }
        if (input_count >= InputWidth / gcd) {
            input_count = 0;
        }
    }

    if(!cut_last){
		if (output_count != 0 && total_output_count < NumOutputs) {
			out_stream.write(output);
            total_output_count++;
		}
    }
}
