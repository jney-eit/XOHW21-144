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

#pragma once

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

#include <hls_stream.h>
#define AP_INT_MAX_W 8192 //4096
#include "ap_int.h"
#include <iostream>
#include <fstream>
#include <string>
#include <limits>

#define CASSERT_DATAFLOW(x) {if (!(x)) {std::cout<< "CASSERT_DATAFLOW condition is not met " << std::endl; exit(-1);	}}

#ifndef __SYNTHESIS__
template<unsigned int WIDTH, unsigned int CHANNELS, unsigned int PE, typename dtype>
void write_fm_to_file(hls::stream<ap_uint<dtype::width * PE>> &fm, std::string filename){


	std::ofstream f;
	f.open(filename);

	dtype *fm_array = new dtype[CHANNELS * WIDTH];
	f << std::setprecision(6) << std::fixed;

	for(int x = 0; x < WIDTH; x++){
		for(int c = 0; c < CHANNELS/PE; c++){
			auto fm_in = fm.read();
				for(int pe_iter = 0; pe_iter < PE; pe_iter++){
					ap_uint<dtype::width> ch_temp = fm_in((pe_iter+1)*dtype::width - 1, pe_iter*dtype::width);
					dtype ch = *reinterpret_cast<dtype*>(&ch_temp);
					fm_array[pe_iter + c*PE + x*CHANNELS] = ch;
			}
			fm.write(fm_in);
		}
	}


	f << "----------------------------------------------------" << std::endl;
	for(int c = 0; c < CHANNELS; c++){
		f << "Channel " << c << ": ------" << std::endl;
		for(int x = 0; x < WIDTH; x++){
			f << roundf(((float) fm_array[c + x*CHANNELS]) * 1000000) / 1000000 << " ";
		}
		f << std::endl;

	}
	f << "----------------------------------------------------" << std::endl << std::endl;

	delete[] fm_array;
	f.close();
}
#endif


constexpr unsigned int bitsNeeded(unsigned n) {
	return (n<2) ? 1:
		   (n<4) ? 2:
	       (n<8) ? 3:
	       (n<16) ? 4:
		   (n<32) ? 5:
		   (n<64) ? 6:
		   (n<128) ? 7:
		   (n<256) ? 8:
		   (n<512) ? 9:
		   (n<1024) ? 10:
		   (n<2048) ? 11:
		   (n<4096) ? 12:
		   (n<8192) ? 13:
		   (n<16384) ? 14:
		   (n<32768) ? 15:
		   (n<65536) ? 16:
		   (n<131072) ? 17:
		   (n<262144) ? 18:
		   (n<524288) ? 19:
		   (n<1048576) ? 20:
		   (n<2097152) ? 21:
		   (n<4194304) ? 22:
		   (n<8388608) ? 23:
		   (n<16777216) ? 24:
		   (n<33554432) ? 25:
		   (n<61708864) ? 26:
		   (n<134217728) ? 27:
		   (n<268435456) ? 28:
		   (n<536870912) ? 29:
		   (n<1073741824) ? 30:
		   (n<2147483648) ? 31: 32;
}

#define stream_type ap_uint
 
#include "profiler.h"
#include "dma.h"
#include "converters.h"
#include "1d/pool1d.h"
#include "1d/conv1d.h"
#include "1d/fc1d.h"

