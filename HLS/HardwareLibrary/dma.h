/*
    Copyright (c) 2018, Xilinx, Inc.
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:

    1.  Redistributions of source code must retain the above copyright notice,
        this list of conditions and the following disclaimer.

    2.  Redistributions in binary form must reproduce the above copyright
        notice, this list of conditions and the following disclaimer in the
        documentation and/or other materials provided with the distribution.

    3.  Neither the name of the copyright holder nor the names of its
        contributors may be used to endorse or promote products derived from
        this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
    THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
    PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
    CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
    EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
    PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
    OR BUSINESS INTERRUPTION). HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
    WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
    OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
    ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <assert.h>

// essentially small DMA generators, moving data between mem-mapped arrays and
// streams
template<unsigned int DataWidth>
void Mem2Stream(
		ap_uint<DataWidth> *in,
		hls::stream<ap_uint<DataWidth> > &out,
		const unsigned int numBytes) {

	const unsigned int numWords = numBytes / (DataWidth / 8);

	for (unsigned int i = 0; i < numWords; i++)
	{
		#pragma HLS PIPELINE II=1
		ap_uint<DataWidth> e = in[i];
		out.write(e);
	}
}



template
<
short unsigned int SIMDWidth,               // number of SIMD lanes
short unsigned int InputPrecision,          // Precisions for the input activation
short unsigned int InputIntPrecision,       // Input activation int bitwidth
short unsigned int OutputPrecision
>
void Stream2Mem1D(hls::stream<ap_uint<SIMDWidth * InputPrecision> > & in,
				  ap_uint<OutputPrecision> *out)
{
	typedef ap_fixed<InputPrecision, InputIntPrecision, AP_RND_ZERO, AP_WRAP> Input_t;

	ap_int<InputPrecision> input = in.read();
	Input_t data = *reinterpret_cast<Input_t *>(&input);

	std::cout << (float)data << std::endl;

	ap_uint<OutputPrecision> result = 0;

	if(data > (Input_t)0.0){
		result = 1;
	}else{
		result = 0;
	}
	*out = result;
}
