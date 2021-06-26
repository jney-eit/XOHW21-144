#pragma once



/*
 * Buffers incoming values and streams SIMDWidth Channels to the next layer.
 * Values are streamed in convolutional fashion with Stride 1 and arbitrary Kernel Size.
 * e.g. KernelSize=3: 0,1,2, 1,2,3, 2,3,4, 3,4,5 ....
 *
 */
template<
        short unsigned int KernelDim,                // e.g 3 for a 1x3 1D conv kernel
        short unsigned int IFMChannels,                // max number of input feature maps
        short unsigned int IFMDim,                // max width of input feature map
        short unsigned int Stride,                    // Stride
        short unsigned int OFMChannels,            // max number of output feature maps
        short unsigned int OFMDim,                // (IFMDim - KernelDim + 2 x Padding) / Stride + 1
        short unsigned int PECount,                    // number of PEs of the following conv layer
        short unsigned int SIMDWidth,                // number of SIMD lanes
        short unsigned int Precision,                // Precisions for the input/output activation
        short unsigned int IntPrecision            // Input/Output activation int bitwidth
>
void Conv1DBufferK(hls::stream <ap_uint<SIMDWidth * Precision>> &in,
                   hls::stream <ap_uint<SIMDWidth * Precision>> &out) {


    CASSERT_DATAFLOW(IFMChannels % SIMDWidth == 0);
    CASSERT_DATAFLOW(OFMChannels % PECount == 0);

    // Number of Output Channels Calculated sequentially
    constexpr unsigned int neuronFold = OFMChannels / PECount;
    // Number of Input Channels Calculated sequentially
    constexpr unsigned int synapseFold = IFMChannels / SIMDWidth;
    // Index to read from input buffer for given kernel position
    unsigned int read_indices[KernelDim][KernelDim];
    for (unsigned int i = 0; i < KernelDim; i++) {
        for (unsigned int j = 0; j < KernelDim; j++) {
            unsigned int curr_read_indice = (j + i) % KernelDim;
            read_indices[i][j] = curr_read_indice;
        }
    }

    // Buffer to read incoming values and write outgoing values
    ap_uint < SIMDWidth * Precision > inputBuf[KernelDim][synapseFold];

    // Read in first kernelDim buffers
    for (unsigned int ptr_k = 0; ptr_k < KernelDim; ptr_k++) {
        for (unsigned int ptr_simd = 0; ptr_simd < synapseFold; ptr_simd++) {
#pragma HLS PIPELINE II=1
            inputBuf[ptr_k][ptr_simd] = in.read();
        }
    }

    ap_uint<bitsNeeded(KernelDim)> read_index = 0;

    for (unsigned int ofm_iter = 0; ofm_iter < OFMDim; ofm_iter++, read_index++) {
        for (unsigned int nm = 0; nm < neuronFold; nm++) {
            for (unsigned int ptr_simd = 0; ptr_simd < synapseFold; ptr_simd++) {
                for (unsigned int read_index_k = 0; read_index_k < KernelDim; read_index_k++) {
#pragma HLS PIPELINE II=1
                    if (read_index == KernelDim) {
                        read_index = 0;
                    }
                    unsigned int ptr_k = read_indices[read_index][read_index_k];
                    out.write(inputBuf[ptr_k][ptr_simd]);

                    if (ofm_iter < IFMDim - KernelDim && read_index_k == 0 && nm == neuronFold - 1) {
                        inputBuf[ptr_k][ptr_simd] = in.read();
                    }
                }
            }
        }
    }
}


/*
 * Streams values in correct order to Conv MAC function.
 * Variable Kernel Size and Stride.
 * Values are received channel wise and streamed kernel wise
 *
 */
template<
        short unsigned int KernelDim,                // e.g 3 for a 1x3 1D conv kernel
        short unsigned int IFMChannels,                // max number of input feature maps
        short unsigned int IFMDim,                // max width of input feature map
        short unsigned int Stride,                    // Stride
        short unsigned int OFMChannels,            // max number of output feature maps
        short unsigned int OFMDim,                // (IFMDim - KernelDim + 2 x Padding) / Stride + 1
        short unsigned int PECount,                    // number of PEs of the following conv layer
        short unsigned int SIMDWidth,                // number of SIMD lanes
        short unsigned int Precision,                // Precisions for the input/output activation
        short unsigned int IntPrecision            // Input/Output activation int bitwidth
>
void Conv1DBuffer_K_S(hls::stream <ap_uint<SIMDWidth * Precision>> &in,
                   hls::stream <ap_uint<SIMDWidth * Precision>> &out) {

    CASSERT_DATAFLOW(IFMChannels % SIMDWidth == 0);
    CASSERT_DATAFLOW(OFMChannels % PECount == 0);


    // Number of Output Channels Calculated sequentially
    constexpr unsigned int neuronFold = OFMChannels / PECount;
    // Number of Input Channels Calculated sequentially
    constexpr unsigned int synapseFold = IFMChannels / SIMDWidth;
    // Size of the PingPong Buffer
    constexpr unsigned int buff_size = MIN(2*KernelDim, KernelDim + Stride);

    // Buffer to read incoming values and write outgoing values
    ap_uint < SIMDWidth * Precision > Buf[buff_size][synapseFold];
	#pragma HLS ARRAY_PARTITION variable=Buf complete dim=1

    // Count reads from IFM
    ap_uint<bitsNeeded(IFMDim * synapseFold)> ifm_count = 0;

    // Read in first buffer
    for (unsigned int ptr_k = 0; ptr_k < KernelDim; ptr_k++) {
        for (unsigned int ptr_simd = 0; ptr_simd < synapseFold; ptr_simd++) {
	    	if(synapseFold != 1){
        		#pragma HLS PIPELINE II=1
	    	}
            Buf[ptr_k][ptr_simd] = in.read();
            ifm_count++;
        }
    }

    constexpr unsigned int discards_per_output = (Stride > KernelDim) ? synapseFold * (Stride - KernelDim) : 0;

    // Discard first values if Stirde > Kernel
    if(discards_per_output > 0){
		for(unsigned int discard_count = 0; discard_count < discards_per_output; discard_count++){
			#pragma HLS PIPELINE II=1
			if(ifm_count < IFMDim * synapseFold){
				in.read();
				ifm_count++;
			}
		}
    }

    // Outputs written from one KernelSize Buffer
    constexpr unsigned int outputs_per_buffer = KernelDim * synapseFold * neuronFold;

    ap_uint<bitsNeeded(KernelDim)> ptr_k_write = 0;
    ap_uint<bitsNeeded(synapseFold)> ptr_simd_write = 0;

    ap_uint<bitsNeeded(MIN(Stride, KernelDim))> ptr_k_read = 0;
    ap_uint<bitsNeeded(synapseFold)> ptr_simd_read = 0;

    ap_uint<bitsNeeded(MIN(Stride, KernelDim) + buff_size)> offset_write = 0;
    ap_uint<bitsNeeded(MIN(Stride, KernelDim) + buff_size)> offset_read = KernelDim;

    ap_uint<bitsNeeded(outputs_per_buffer + discards_per_output)> out_iter = 0;

    // Read and write concurrently
	ofm_loop: for (unsigned int ofm_iter = 0; ofm_iter < OFMDim; ofm_iter++) {
		bool curr_read_done = false;
		if(outputs_per_buffer + discards_per_output == 1){
			#pragma HLS PIPELINE II=1
		}

		out_buff_loop: for(unsigned int out_iter=0; out_iter < outputs_per_buffer + discards_per_output; out_iter++){
			#pragma HLS PIPELINE II=1

			// Discard output
			if(out_iter >= outputs_per_buffer){
				if(ifm_count < IFMDim * synapseFold){
					in.read();
					ifm_count++;
				}
			}else{

				ap_uint<bitsNeeded(MIN(Stride, KernelDim) + buff_size + KernelDim)> index_buff_write = offset_write + ptr_k_write;

				if(index_buff_write >= buff_size){
					index_buff_write = index_buff_write - buff_size;
				}

				// Write value every CC
				out.write(Buf[index_buff_write][ptr_simd_write]);

				ptr_k_write++;
				if(ptr_k_write == KernelDim){
					ptr_k_write = 0;
					ptr_simd_write++;
					if(ptr_simd_write == synapseFold){
						ptr_simd_write = 0;

					}
				}

				// Read in new values as long as values to be written are not overwritten
				if(!curr_read_done && ifm_count < IFMDim * synapseFold){
					unsigned int index_buff_read = offset_read + ptr_k_read;
					if(index_buff_read >= buff_size){
						index_buff_read = index_buff_read - buff_size;
					}

					Buf[index_buff_read][ptr_simd_read] = in.read();
					ifm_count++;

					ptr_simd_read++;
					if(ptr_simd_read == synapseFold){
						ptr_simd_read = 0;
						ptr_k_read++;
						if(ptr_k_read == MIN(Stride, KernelDim)){
							ptr_k_read = 0;
							curr_read_done = true;
						}
					}
				}

			}
		}

		offset_write += MIN(Stride, KernelDim);
		offset_read += MIN(Stride, KernelDim);

		if(offset_write >= buff_size){
			offset_write = offset_write - buff_size;
		}

		if(offset_read >= buff_size){
			offset_read = offset_read - buff_size;
		}
	}

}


template<
        short unsigned int KernelDim,                // e.g 3 for a 1x3 1D conv kernel
        short unsigned int Channels,                // max number of input feature maps
        short unsigned int IFMDim,                // max width of input feature map
        short unsigned int Stride,                    // Stride
        short unsigned int OFMDim,                // (IFMDim - KernelDim + 2 x Padding) / Stride + 1
        short unsigned int SIMDWidth,                // number of SIMD lanes
        short unsigned int Precision,                // Precisions for the input/output activation
        short unsigned int IntPrecision            // Input/Output activation int bitwidth
>
void Conv1DBuffer_K_S_Depthwise(hls::stream <ap_uint<SIMDWidth * Precision>> &in,
                                hls::stream <ap_uint<SIMDWidth * Precision>> &out) {

    CASSERT_DATAFLOW(Channels % SIMDWidth == 0);

    // Number of Input Channels Calculated sequentially
    constexpr unsigned int synapseFold = Channels / SIMDWidth;
    // Size of the PingPong Buffer
    constexpr unsigned int buff_size = MIN(2*KernelDim, KernelDim + Stride);

    // Buffer to read incoming values and write outgoing values
    ap_uint < SIMDWidth * Precision > Buf[buff_size][synapseFold];
    #pragma HLS ARRAY_PARTITION variable=Buf complete dim=1
	//#pragma HLS ARRAY_RESHAPE variable=Buf complete

    // Count reads from IFM
    ap_uint<bitsNeeded(IFMDim * synapseFold)> ifm_count = 0;

    // Read in first buffer
    for (unsigned int ptr_k = 0; ptr_k < KernelDim; ptr_k++) {
        for (unsigned int ptr_simd = 0; ptr_simd < synapseFold; ptr_simd++) {
            if(synapseFold != 1){
                #pragma HLS PIPELINE II=1
            }
            Buf[ptr_k][ptr_simd] = in.read();
            ifm_count++;
        }
    }

    constexpr unsigned int discards_per_output = (Stride > KernelDim) ? synapseFold * (Stride - KernelDim) : 0;

    // Discard first values if Stirde > Kernel
    if(discards_per_output > 0){
        for(unsigned int discard_count = 0; discard_count < discards_per_output; discard_count++){
            #pragma HLS PIPELINE II=1
            if(ifm_count < IFMDim * synapseFold){
                in.read();
                ifm_count++;
            }
        }
    }

    // Outputs written from one KernelSize Buffer
    constexpr unsigned int outputs_per_buffer = KernelDim * synapseFold;

    ap_uint<bitsNeeded(KernelDim)> ptr_k_write = 0;
    ap_uint<bitsNeeded(synapseFold)> ptr_simd_write = 0;

    ap_uint<bitsNeeded(MIN(Stride, KernelDim))> ptr_k_read = 0;
    ap_uint<bitsNeeded(synapseFold)> ptr_simd_read = 0;

    ap_uint<bitsNeeded(MIN(Stride, KernelDim) + buff_size)> offset_write = 0;
    ap_uint<bitsNeeded(MIN(Stride, KernelDim) + buff_size)> offset_read = KernelDim;

    ap_uint<bitsNeeded(outputs_per_buffer + discards_per_output)> out_iter = 0;

    // Read and write concurrently
    ofm_loop: for (unsigned int ofm_iter = 0; ofm_iter < OFMDim; ofm_iter++) {
    bool curr_read_done = false;
    if(outputs_per_buffer + discards_per_output == 1){
		#pragma HLS PIPELINE II=1
    }

    out_buff_loop: for(unsigned int out_iter=0; out_iter < outputs_per_buffer + discards_per_output; out_iter++){
    #pragma HLS PIPELINE II=1

		// Discard output
		if(out_iter >= outputs_per_buffer){
			if(ifm_count < IFMDim * synapseFold){
				in.read();
				ifm_count++;
			}
		}else{

			ap_uint<bitsNeeded(MIN(Stride, KernelDim) + buff_size + KernelDim)> index_buff_write = offset_write + ptr_k_write;

			if(index_buff_write >= buff_size){
				index_buff_write = index_buff_write - buff_size;
			}

			// Write value every CC
			out.write(Buf[index_buff_write][ptr_simd_write]);

			ptr_k_write++;
			if(ptr_k_write == KernelDim){
				ptr_k_write = 0;
				ptr_simd_write++;
				if(ptr_simd_write == synapseFold){
					ptr_simd_write = 0;

				}
			}

			// Read in new values as long as values to be written are not overwritten
			if(!curr_read_done && ifm_count < IFMDim * synapseFold){
				unsigned int index_buff_read = offset_read + ptr_k_read;
				if(index_buff_read >= buff_size){
					index_buff_read = index_buff_read - buff_size;
				}

				Buf[index_buff_read][ptr_simd_read] = in.read();
				ifm_count++;

				ptr_simd_read++;
				if(ptr_simd_read == synapseFold){
					ptr_simd_read = 0;
					ptr_k_read++;
					if(ptr_k_read == MIN(Stride, KernelDim)){
						ptr_k_read = 0;
						curr_read_done = true;
					}
				}
			}
		}
	}

    offset_write += MIN(Stride, KernelDim);
    offset_read += MIN(Stride, KernelDim);

    if(offset_write >= buff_size){
        offset_write = offset_write - buff_size;
    }

    if(offset_read >= buff_size){
        offset_read = offset_read - buff_size;
    }
}

}



/*
 * Buffers incoming values and streams whole Kernel to the next layer.
 * Values are streamed in convolutional fashion with Stride 1 and arbitrary Kernel Size.
 * e.g. KernelSize=3: 0,1,2, 1,2,3, 2,3,4, 3,4,5 ....
 * Conv Layer has to be fully unrolled on channel level to allow kernel level parallelization
 *
 */
template<
        short unsigned int KernelDim,                // e.g 3 for a 1x3 1D conv kernel
        short unsigned int Channels,                // max number of input feature maps
        short unsigned int IFMDim,                // max width of input feature map
        short unsigned int Stride,                    // Stride
        short unsigned int OFMDim,                // (IFMDim - KernelDim + 2 x Padding) / Stride + 1
        short unsigned int SIMDWidth,               // number of SIMD lanes
        short unsigned int Precision,                // Precisions for the input/output activation
        short unsigned int IntPrecision            // Input/Output activation int bitwidth
>
void Conv1DBufferK_Unrolled(hls::stream <ap_uint<SIMDWidth * Precision>> &in,
                            hls::stream <ap_uint<SIMDWidth * Precision * KernelDim>> &out) {


    CASSERT_DATAFLOW(Channels == SIMDWidth);

    // Buffer is bigger than KernelDim to allow concurrent reading and writing
    unsigned int read_indices[KernelDim + 1][KernelDim + 1];
    for (unsigned int i = 0; i < KernelDim + 1; i++) {
        for (unsigned int j = 0; j < KernelDim + 1; j++) {
            unsigned int curr_read_indice = (j + i) % (KernelDim + 1);
            read_indices[i][j] = curr_read_indice;
        }
    }

    ap_uint < SIMDWidth * Precision > inputBuf[KernelDim + 1];
	#pragma HLS ARRAY_PARTITION variable=inputBuf complete


    // Read in first kernelDim buffers
    for (unsigned int ptr_k = 0; ptr_k < KernelDim; ptr_k++) {
		#pragma HLS PIPELINE II=1
        inputBuf[ptr_k] = in.read();
    }

    ap_uint<bitsNeeded(KernelDim+1)> read_index = 0;
    // Read and Write in parallel
    for (unsigned int ofm_iter = 0; ofm_iter < OFMDim * Stride; ofm_iter++, read_index++) {
		#pragma HLS PIPELINE II=1
        if (read_index == KernelDim + 1) {
            read_index = 0;
        }

        ap_uint < SIMDWidth * Precision * KernelDim > output;
        // Write KernelDim Values
        for (unsigned int pixel_iter = 0; pixel_iter < KernelDim; pixel_iter++) {
			#pragma HLS UNROLL
            unsigned int ptr_k = read_indices[read_index][pixel_iter];
            output((pixel_iter + 1) * SIMDWidth * Precision - 1, pixel_iter * SIMDWidth * Precision) = inputBuf[ptr_k];

        }
        out.write(output);

        // Read one value
        if (ofm_iter < IFMDim - KernelDim) {
            unsigned int ptr_k = read_indices[read_index][KernelDim];
            inputBuf[ptr_k] = in.read();
        }

    }
}


/**
 *  Multiplies incoming values with weight and accumulates them till KernelSize values are received.
 *  SIMDWidth input channels are read per clock cycles. So one value will be outputed each
 *  (IFMChannels/SIMD)*(OFMChannels/PECount)*KernelDim clock cycles
 *
 */
template<
        // convolution parameters
        short unsigned int KernelDim,                // e.g 3 for a 1x3 conv kernel
        short unsigned int IFMChannels,                // max number of input feature maps
        short unsigned int IFMDim,                // max width of input feature map
        short unsigned int Stride,                    // Stride
        short unsigned int Padding,                    // Padding
        short unsigned int OFMChannels,            // max number of output feature maps
        short unsigned int OFMDim,                // (IFMDim - KernelDim + 2 x Padding) / Stride + 1
        // parallelization parameters
        short unsigned int PECount,                 // number of PEs
        short unsigned int SIMDWidth,               // number of SIMD lanes
        // matrix-vector unit parameters
        short unsigned int BiasPrecision,            // Precisions for the bias
        short unsigned int BiasIntPrecision,        // Bias int bitwidth
        short unsigned int WeightsPrecision,        // Precisions for the weight
        short unsigned int WeightsIntPrecision,     // Weight int bitwidth
        short unsigned int InputPrecision,          // Precisions for the input activation
        short unsigned int InputIntPrecision,       // Input activation int bitwidth
        short unsigned int MulPrecision,            // Precision for the result of multiplication
        short unsigned int MulIntPrecision,         // Multiplication int bitwidth
        short unsigned int AccPrecision,            // Precision for the result of accumulation
        short unsigned int AccIntPrecision,         // Accumulation int bitwidth
        short unsigned int OutputPrecision,            // Precisions for the output activation
        short unsigned int OutputIntPrecision,        // Output activation int bitwidth
        bool use_profiler = false
>
void Conv1DMac(hls::stream <ap_uint<SIMDWidth * InputPrecision>> &in,
               hls::stream <ap_uint<PECount * OutputPrecision>> &out,
               const ap_uint <WeightsPrecision> weightMem[PECount][SIMDWidth][KernelDim * IFMChannels * OFMChannels /
                                                                              (SIMDWidth * PECount)],
               const ap_uint <BiasPrecision> biasMem[PECount][OFMChannels / PECount],
               Profiler_MAC *profiler = nullptr) {
    //std::cout << "Conv1DMac_new: KernelDim " << KernelDim << " IFMDim " << IFMDim << std::endl;


	if(IFMChannels == SIMDWidth && OFMChannels == PECount){
		#pragma HLS inline
	}

    typedef ap_fixed <InputPrecision, InputIntPrecision, AP_RND_ZERO, AP_WRAP> Input_t;
    typedef ap_fixed <BiasPrecision, BiasIntPrecision, AP_RND_ZERO, AP_WRAP> Bias_t;
    typedef ap_fixed <WeightsPrecision, WeightsIntPrecision, AP_RND_ZERO, AP_WRAP> Weights_t;
    typedef ap_fixed <MulPrecision, MulIntPrecision, AP_RND_ZERO, AP_WRAP> Mul_t;
    typedef ap_fixed <AccPrecision, AccIntPrecision, AP_RND_ZERO, AP_WRAP> Acc_t;
    typedef ap_fixed <OutputPrecision, OutputIntPrecision, AP_RND_ZERO, AP_WRAP> Output_t;

    // Number of Output Channels calculated sequentially
    constexpr unsigned int neuronFold = OFMChannels / PECount;
    // Number of Input Channels calculated sequentially
    constexpr unsigned int synapseFold = KernelDim * IFMChannels / SIMDWidth;

    Acc_t macRegisters[PECount];
	#pragma HLS ARRAY_PARTITION variable=macRegisters complete dim=1

    loop_init:
    for (unsigned int pe = 0; pe < PECount; pe++) {
		#pragma HLS UNROLL
        macRegisters[pe] = 0;
    }

    loop_ofm:
	for (unsigned int ofm_iter = 0; ofm_iter < OFMDim; ofm_iter++) {
		if(synapseFold == 1 && neuronFold == 1){
			#pragma HLS PIPELINE II=1
		}
		loop_neuronFold:
		for (unsigned int nm = 0; nm < neuronFold; nm++) {
			if(synapseFold == 1){
				#pragma HLS PIPELINE II=1
			}
			loop_synapseFold:
			for (unsigned int sf = 0; sf < synapseFold; sf++) {
				#pragma HLS PIPELINE II=1

				ap_uint < InputPrecision * SIMDWidth > input = in.read();


				loop_pe:
				for (unsigned int pe = 0; pe < PECount; pe++) {
					#pragma HLS UNROLL

					Acc_t tmpMac = macRegisters[pe];

					loop_simd:
					for (unsigned int simd = 0; simd < SIMDWidth; simd++) {
						#pragma HLS UNROLL

						Mul_t mul;

						unsigned int lowBit = simd * InputPrecision;
						unsigned int highBit = (simd + 1) * InputPrecision - 1;
						ap_int <InputPrecision> temp_input = input(highBit, lowBit);
						Input_t data = *reinterpret_cast<Input_t *>(&temp_input);

						if (use_profiler) {
							profiler->update_in(data);
						}

						ap_int <WeightsPrecision> temp_weight = weightMem[pe][simd][nm * synapseFold + sf];
						Weights_t weight = *reinterpret_cast<Weights_t *>(&temp_weight);

						mul = data * weight;

						if (use_profiler) {
							profiler->update_mul(mul);
						}

						tmpMac += mul;

						if (use_profiler) {
							profiler->update_acc(tmpMac);
						}
					}

					macRegisters[pe] = tmpMac;
				}

				if (sf == synapseFold - 1) {
					ap_uint < PECount * OutputPrecision > output;

					for (unsigned int pe = 0; pe < PECount; pe++) {
						#pragma HLS UNROLL

						Output_t result;

						// Why ap_int?? Wrong results with ap_uint
						ap_int <BiasPrecision> temp = biasMem[pe][nm];
						Bias_t bias = *reinterpret_cast<Bias_t *>(&temp);
						macRegisters[pe] = macRegisters[pe] + (Acc_t) bias;

						result = (Output_t) macRegisters[pe];

						if (use_profiler) {
							profiler->update_out(result);
						}

						unsigned int lowBit = pe * OutputPrecision;
						unsigned int highBit = (pe + 1) * OutputPrecision - 1;
						ap_uint <OutputPrecision> output_temp = *reinterpret_cast<ap_uint <OutputPrecision> *>(&result);
						output(highBit, lowBit) = output_temp;

						macRegisters[pe] = 0;

					}

					out.write(output);
				}
			}
		}
	}

    //std::cout << "Conv finished ----------------------------------------" << std::endl;

}



template<
        // convolution parameters
        short unsigned int KernelDim,                // e.g 3 for a 1x3 conv kernel
        short unsigned int Channels,                // max number of input feature maps
        short unsigned int IFMDim,                // max width of input feature map
        short unsigned int Stride,                    // Stride
        short unsigned int Padding,                    // Padding
        short unsigned int OFMDim,                // (IFMDim - KernelDim + 2 x Padding) / Stride + 1
        // parallelization parameters
        short unsigned int SIMDWidth,               // number of SIMD lanes
        // matrix-vector unit parameters
        short unsigned int WeightsPrecision,        // Precisions for the weight
        short unsigned int WeightsIntPrecision,     // Weight int bitwidth
        short unsigned int InputPrecision,          // Precisions for the input activation
        short unsigned int InputIntPrecision,       // Input activation int bitwidth
        short unsigned int MulPrecision,            // Precision for the result of multiplication
        short unsigned int MulIntPrecision,         // Multiplication int bitwidth
        short unsigned int AccPrecision,            // Precision for the result of accumulation
        short unsigned int AccIntPrecision,         // Accumulation int bitwidth
        short unsigned int OutputPrecision,            // Precisions for the output activation
        short unsigned int OutputIntPrecision,        // Output activation int bitwidth
        bool use_profiler = false
>
void Conv1DMac_Depthwise(hls::stream <ap_uint<SIMDWidth * InputPrecision>> &in,
                         hls::stream <ap_uint<SIMDWidth * OutputPrecision>> &out,
                         const ap_uint <WeightsPrecision> weightMem[SIMDWidth][KernelDim * Channels /(SIMDWidth)],
                         Profiler_MAC *profiler = nullptr) {

    CASSERT_DATAFLOW(Channels % SIMDWidth == 0);

    typedef ap_fixed <InputPrecision, InputIntPrecision, AP_RND_ZERO, AP_WRAP> Input_t;
    typedef ap_fixed <WeightsPrecision, WeightsIntPrecision, AP_RND_ZERO, AP_WRAP> Weights_t;
    typedef ap_fixed <MulPrecision, MulIntPrecision, AP_RND_ZERO, AP_WRAP> Mul_t;
    // JUST FOR TESTING
    //typedef ap_fixed <AccPrecision, AccIntPrecision, AP_RND_ZERO, AP_SAT> Acc_t;
    //typedef ap_fixed <OutputPrecision, OutputIntPrecision, AP_RND_ZERO, AP_SAT> Output_t;

    typedef ap_fixed <AccPrecision, AccIntPrecision, AP_RND_ZERO, AP_WRAP> Acc_t;
    typedef ap_fixed <OutputPrecision, OutputIntPrecision, AP_RND_ZERO, AP_WRAP> Output_t;

    // Number of Input Channels calculated sequentially
    constexpr unsigned int synapseFold = Channels / SIMDWidth;

    Acc_t macRegisters[SIMDWidth]; //[synapseFold];
    #pragma HLS ARRAY_PARTITION variable=macRegisters complete dim=1
    loop_init:
    for (unsigned int simd = 0; simd < SIMDWidth; simd++) {
		#pragma HLS UNROLL
        macRegisters[simd] = 0;
    }

	//#pragma HLS dependence variable=macRegisters inter false

    loop_ofmChannels:
    for (unsigned int ofm_iter = 0; ofm_iter < OFMDim; ofm_iter++) {
        if(synapseFold == 1 && KernelDim == 1){
            #pragma HLS PIPELINE II=1
        }
        loop_synapseFold:
        for (unsigned int sf = 0; sf < synapseFold; sf++) {
            if(KernelDim == 1){
                #pragma HLS PIPELINE II=1
            }
            for (unsigned int k = 0; k < KernelDim; k++) {
                #pragma HLS PIPELINE II=1

                ap_uint < InputPrecision * SIMDWidth > input = in.read();

                loop_simd:
                for (unsigned int simd = 0; simd < SIMDWidth; simd++) {
                    #pragma HLS UNROLL

                    Mul_t mul;

                    unsigned int lowBit = simd * InputPrecision;
                    unsigned int highBit = (simd + 1) * InputPrecision - 1;
                    ap_int <InputPrecision> temp_input = input(highBit, lowBit);
                    Input_t data = *reinterpret_cast<Input_t *>(&temp_input);

                    if (use_profiler) {
                        profiler->update_in(data);
                    }

                    ap_int <WeightsPrecision> temp_weight = weightMem[simd][sf*KernelDim + k];
                    Weights_t weight = *reinterpret_cast<Weights_t *>(&temp_weight);

                    mul = data * weight;

                    if (use_profiler) {
                        profiler->update_mul(mul);
                    }

                    //if(k == 0){
                    //	macRegisters[simd] = mul;
                    //}else{
                    macRegisters[simd] += mul;
                    //}

                    if (use_profiler) {
                        profiler->update_acc(macRegisters[simd]);
                    }
                }

                if (k == KernelDim - 1) {
					#pragma HLS occurrence cycle=KernelDim
                    ap_uint < SIMDWidth * OutputPrecision > output;

					for (unsigned int simd_out = 0; simd_out < SIMDWidth; simd_out++) {
						#pragma HLS UNROLL

						Output_t result;
						result = (Output_t) macRegisters[simd_out];
						macRegisters[simd_out] = 0;

						if (use_profiler) {
							profiler->update_out(result);
						}

						unsigned int lowBit = simd_out * OutputPrecision;
						unsigned int highBit = (simd_out + 1) * OutputPrecision - 1;
						ap_uint <OutputPrecision> output_temp = *reinterpret_cast<ap_uint <OutputPrecision> *>(&result);
						output(highBit, lowBit) = output_temp;

					}
					out.write(output);

                }
            }
        }
    }

    //std::cout << "Conv finished ----------------------------------------" << std::endl;

}



template<
        // convolution parameters
        short unsigned int KernelDim,                // e.g 3 for a 1x3 conv kernel
        short unsigned int Channels,                // max number of input feature maps
        short unsigned int IFMDim,                // max width of input feature map
        short unsigned int Stride,                    // Stride
        short unsigned int Padding,                    // Padding
        short unsigned int OFMDim,                // (IFMDim - KernelDim + 2 x Padding) / Stride + 1
        short unsigned int WeightsPrecision,        // Precisions for the weight
        short unsigned int WeightsIntPrecision,     // Weight int bitwidth
        short unsigned int InputPrecision,          // Precisions for the input activation
        short unsigned int InputIntPrecision,       // Input activation int bitwidth
        short unsigned int MulPrecision,            // Precision for the result of multiplication
        short unsigned int MulIntPrecision,         // Multiplication int bitwidth
        short unsigned int AccPrecision,            // Precision for the result of accumulation
        short unsigned int AccIntPrecision,         // Accumulation int bitwidth
        short unsigned int OutputPrecision,            // Precisions for the output activation
        short unsigned int OutputIntPrecision,        // Output activation int bitwidth
        bool use_profiler = false
>
void Conv1DMac_Depthwise_Unrolled(hls::stream <ap_uint<KernelDim * Channels * InputPrecision>> &in,
                         hls::stream <ap_uint<Channels * OutputPrecision>> &out,
                         const ap_uint <WeightsPrecision> weightMem[Channels][KernelDim],
                         Profiler_MAC *profiler = nullptr) {


    typedef ap_fixed <InputPrecision, InputIntPrecision, AP_RND_ZERO, AP_WRAP> Input_t;
    typedef ap_fixed <WeightsPrecision, WeightsIntPrecision, AP_RND_ZERO, AP_WRAP> Weights_t;
    typedef ap_fixed <MulPrecision, MulIntPrecision, AP_RND_ZERO, AP_WRAP> Mul_t;
    // JUST FOR TESTING
    //typedef ap_fixed <AccPrecision, AccIntPrecision, AP_RND_ZERO, AP_SAT> Acc_t;
    //typedef ap_fixed <OutputPrecision, OutputIntPrecision, AP_RND_ZERO, AP_SAT> Output_t;

    typedef ap_fixed <AccPrecision, AccIntPrecision, AP_RND_ZERO, AP_WRAP> Acc_t;
    typedef ap_fixed <OutputPrecision, OutputIntPrecision, AP_RND_ZERO, AP_WRAP> Output_t;

    ap_uint<bitsNeeded(Stride)> stride_count = 0;

    Acc_t macRegisters[Channels];
    #pragma HLS ARRAY_PARTITION variable=macRegisters complete dim=1
    loop_init:
    for (unsigned int simd = 0; simd < Channels; simd++) {
		#pragma HLS UNROLL
        macRegisters[simd] = 0;
    }

	//#pragma HLS dependence variable=macRegisters inter false

    loop_ofmChannels:
    for (unsigned int ofm_iter = 0; ofm_iter < OFMDim;) {
    		constexpr unsigned int tripcount = OFMDim*Stride;
			#pragma HLS loop_tripcount avg=tripcount
            #pragma HLS PIPELINE II=1

    		if(stride_count == 0){
				ap_uint <KernelDim * Channels * InputPrecision> input_kernel = in.read();

				for (unsigned int k = 0; k < KernelDim; k++) {
					#pragma HLS UNROLL
					ap_uint < Channels * InputPrecision > input = input_kernel((k + 1) * InputPrecision * Channels - 1, k * InputPrecision * Channels);

					loop_simd:
					for (unsigned int simd = 0; simd < Channels; simd++) {
						#pragma HLS UNROLL

						Mul_t mul;

						unsigned int lowBit = simd * InputPrecision;
						unsigned int highBit = (simd + 1) * InputPrecision - 1;
						ap_int <InputPrecision> temp_input = input(highBit, lowBit);
						Input_t data = *reinterpret_cast<Input_t *>(&temp_input);

						if (use_profiler) {
							profiler->update_in(data);
						}

						ap_int <WeightsPrecision> temp_weight = weightMem[simd][k];
						Weights_t weight = *reinterpret_cast<Weights_t *>(&temp_weight);

						mul = data * weight;

						if (use_profiler) {
							profiler->update_mul(mul);
						}

						macRegisters[simd] += mul;

						if (use_profiler) {
							profiler->update_acc(macRegisters[simd]);
						}
					}
				}

				ap_uint < Channels * OutputPrecision > output;
				for (unsigned int simd_out = 0; simd_out < Channels; simd_out++) {
					#pragma HLS UNROLL

					Output_t result;
					result = (Output_t) macRegisters[simd_out];
					macRegisters[simd_out] = 0;

					if (use_profiler) {
						profiler->update_out(result);
					}

					unsigned int lowBit = simd_out * OutputPrecision;
					unsigned int highBit = (simd_out + 1) * OutputPrecision - 1;
					ap_uint <OutputPrecision> output_temp = *reinterpret_cast<ap_uint <OutputPrecision> *>(&result);
					output(highBit, lowBit) = output_temp;

				}
				out.write(output);
				ofm_iter++;
    		}else{
    			in.read();
    		}
			stride_count++;
			if(stride_count == Stride){
				stride_count = 0;
			}
    }

    for(unsigned int stride_count = 0; stride_count < Stride - 1; stride_count++){
		in.read();
    }
    //std::cout << "Conv finished ----------------------------------------" << std::endl;

}


/**
 *  Multiplies incoming values with weight and accumulates them till KernelSize values are received.
 *  KernelDim * IFMChannels Values are read per clock cycles. So one value will be outputed each
 *  clock cycle
 *
 */
template<
        // convolution parameters
        short unsigned int KernelDim,                // e.g 3 for a 1x3 conv kernel
        short unsigned int IFMChannels,                // max number of input feature maps
        short unsigned int IFMDim,                // max width of input feature map
        short unsigned int Stride,                    // Stride
        short unsigned int Padding,                    // Padding
        short unsigned int OFMChannels,            // max number of output feature maps
        short unsigned int OFMDim,                // (IFMDim - KernelDim + 2 x Padding) / Stride + 1
        // parallelization parameters
        short unsigned int PECount,                 // number of PEs
        short unsigned int SIMDWidth,               // number of SIMD lanes
        // matrix-vector unit parameters
        short unsigned int BiasPrecision,            // Precisions for the bias
        short unsigned int BiasIntPrecision,        // Bias int bitwidth
        short unsigned int WeightsPrecision,        // Precisions for the weight
        short unsigned int WeightsIntPrecision,     // Weight int bitwidth
        short unsigned int InputPrecision,          // Precisions for the input activation
        short unsigned int InputIntPrecision,       // Input activation int bitwidth
        short unsigned int MulPrecision,            // Precision for the result of multiplication
        short unsigned int MulIntPrecision,         // Multiplication int bitwidth
        short unsigned int AccPrecision,            // Precision for the result of accumulation
        short unsigned int AccIntPrecision,         // Accumulation int bitwidth
        short unsigned int OutputPrecision,            // Precisions for the output activation
        short unsigned int OutputIntPrecision,        // Output activation int bitwidth
        bool use_profiler = false
>
void Conv1DMac_Unrolled(hls::stream <ap_uint<SIMDWidth * InputPrecision * KernelDim>> &in,
                        hls::stream <ap_uint<PECount * OutputPrecision>> &out,
                        const ap_uint <WeightsPrecision> weightMem[PECount][SIMDWidth][KernelDim * IFMChannels *
                                                                                       OFMChannels /
                                                                                       (SIMDWidth * PECount)],
                        const ap_uint <BiasPrecision> biasMem[PECount][OFMChannels / PECount],
                        Profiler_MAC *profiler = nullptr) {


    CASSERT_DATAFLOW(IFMChannels == SIMDWidth);
    CASSERT_DATAFLOW(OFMChannels == PECount);

    /*
    typedef ap_fixed <InputPrecision, InputIntPrecision, AP_RND_ZERO, AP_SAT> Input_t;
    typedef ap_fixed <BiasPrecision, BiasIntPrecision, AP_RND_ZERO, AP_SAT> Bias_t;
    typedef ap_fixed <WeightsPrecision, WeightsIntPrecision, AP_RND_ZERO, AP_SAT> Weights_t;
    typedef ap_fixed <MulPrecision, MulIntPrecision, AP_RND_ZERO, AP_SAT> Mul_t;
    typedef ap_fixed <AccPrecision, AccIntPrecision, AP_RND_ZERO, AP_SAT> Acc_t;
    typedef ap_fixed <OutputPrecision, OutputIntPrecision, AP_RND_ZERO, AP_SAT> Output_t;
	*/

    typedef ap_fixed <InputPrecision, InputIntPrecision, AP_RND_ZERO, AP_WRAP> Input_t;
    typedef ap_fixed <BiasPrecision, BiasIntPrecision, AP_RND_ZERO, AP_WRAP> Bias_t;
    typedef ap_fixed <WeightsPrecision, WeightsIntPrecision, AP_RND_ZERO, AP_WRAP> Weights_t;
    typedef ap_fixed <MulPrecision, MulIntPrecision, AP_RND_ZERO, AP_WRAP> Mul_t;
    typedef ap_fixed <AccPrecision, AccIntPrecision, AP_RND_ZERO, AP_WRAP> Acc_t;
    typedef ap_fixed <OutputPrecision, OutputIntPrecision, AP_RND_ZERO, AP_WRAP> Output_t;

    Acc_t macRegisters[PECount];
	#pragma HLS ARRAY_PARTITION variable=macRegisters complete dim=1

    loop_init:
    for (unsigned int pe = 0; pe < PECount; pe++) {
		#pragma HLS UNROLL
        macRegisters[pe] = 0;
    }


    loop_ofmChannels:
    for (unsigned int ofm_iter = 0; ofm_iter < OFMDim; ofm_iter++) {
		#pragma HLS PIPELINE II=1

        ap_uint < InputPrecision * SIMDWidth * KernelDim > input_kernel = in.read();

        for (unsigned int kernel_iter = 0; kernel_iter < KernelDim; kernel_iter++) {
			#pragma HLS UNROLL
            ap_uint < SIMDWidth * InputPrecision > input = input_kernel((kernel_iter + 1) * InputPrecision * SIMDWidth - 1, kernel_iter * InputPrecision * SIMDWidth);

            loop_pe:
            for (unsigned int pe = 0; pe < PECount; pe++) {
				#pragma HLS UNROLL
                Acc_t tmpMac = macRegisters[pe];

                loop_simd:
                for (unsigned int simd = 0; simd < SIMDWidth; simd++) {
					#pragma HLS UNROLL
                    Mul_t mul;

                    unsigned int lowBit = simd * InputPrecision;
                    unsigned int highBit = (simd + 1) * InputPrecision - 1;
                    ap_int <InputPrecision> temp_input = input(highBit, lowBit);
                    Input_t data = *reinterpret_cast<Input_t *>(&temp_input);

                    if (use_profiler) {
                        profiler->update_in(data);
                    }

                    ap_int <WeightsPrecision> temp_weight = weightMem[pe][simd][kernel_iter];
                    Weights_t weight = *reinterpret_cast<Weights_t *>(&temp_weight);

                    mul = data * weight;

                    if (use_profiler) {
                        profiler->update_mul(mul);
                    }

                    tmpMac += mul;

                    if (use_profiler) {
                        profiler->update_acc(tmpMac);
                    }
                }

                macRegisters[pe] = tmpMac;
            }

        }

        ap_uint < PECount * OutputPrecision > output;

        for (unsigned int pe = 0; pe < PECount; pe++) {
			#pragma HLS UNROLL

            Output_t result;

            ap_int <BiasPrecision> temp = biasMem[pe][0];
            Bias_t bias = *reinterpret_cast<Bias_t *>(&temp);
            macRegisters[pe] = macRegisters[pe] + (Acc_t) bias;

            result = (Output_t) macRegisters[pe];

            if (use_profiler) {
                profiler->update_out(result);
            }

            unsigned int lowBit = pe * OutputPrecision;
            unsigned int highBit = (pe + 1) * OutputPrecision - 1;
            ap_uint <OutputPrecision> output_temp = *reinterpret_cast<ap_uint <OutputPrecision> *>(&result);
            output(highBit, lowBit) = output_temp;

            macRegisters[pe] = 0;

        }
        out.write(output);

    }
    //std::cout << "Conv finished ----------------------------------------" << std::endl;

}


template<
        short unsigned int IFMChannels,            // number of input feature maps
        short unsigned int IFMDim,                // length of input sequence
        short unsigned int SIMDWidth,            // number of SIMD lanes
        short unsigned int InputPrecision,         // precisions for the input activation
        short unsigned int InputIntPrecision,      // input activation int bitwidth
        short unsigned int OutputPrecision,        // precisions for the output activation // CAN BE DIFFERENT FROM InputPrecision
        short unsigned int OutputIntPrecision        // output activation int bitwidth // CAN BE DIFFERENT FROM InputIntPrecision
>
void Relu1D(hls::stream <ap_uint<SIMDWidth * InputPrecision>> &in,
            hls::stream <ap_uint<SIMDWidth * OutputPrecision>> &out) {

    CASSERT_DATAFLOW(IFMChannels % SIMDWidth == 0);

    // JUST FOR TESTING
    //typedef ap_fixed <InputPrecision, InputIntPrecision, AP_RND_ZERO, AP_SAT> Input_t;
    //typedef ap_fixed <OutputPrecision, OutputIntPrecision, AP_RND_ZERO, AP_SAT> Output_t;

    constexpr unsigned int MAX_OUT_VAL = (1 << (OutputIntPrecision - 1)) - 1;

    typedef ap_fixed <InputPrecision, InputIntPrecision, AP_RND_ZERO, AP_WRAP> Input_t;
    typedef ap_fixed <OutputPrecision, OutputIntPrecision, AP_RND_ZERO, AP_WRAP> Output_t;

    const unsigned int synapseFold = IFMChannels / SIMDWidth;

    const unsigned int duration = IFMDim * synapseFold;


    loop_dim:
    for (unsigned int i = 0; i < duration; i++) {
#pragma HLS PIPELINE II=1

        ap_uint < SIMDWidth * InputPrecision > input = in.read();
        ap_uint < SIMDWidth * OutputPrecision > output;

        loop_simd:
        for (unsigned int simd = 0; simd < SIMDWidth; simd++) {
#pragma HLS UNROLL

            unsigned int lowBiti = simd * InputPrecision;
            unsigned int highBiti = (simd + 1) * InputPrecision - 1;
            ap_int <InputPrecision> temp_data = input(highBiti, lowBiti);
            Input_t data = *reinterpret_cast<Input_t *>(&temp_data);

            Output_t result;

            if (data < (Input_t) 0.0)
                result = (Output_t) 0.0;
            else
            	if(data > MAX_OUT_VAL){
            		result = (Output_t) MAX_OUT_VAL;
            	}else{
                //result = (Output_t)(data/(Input_t)6.0);
            		result = (Output_t) data;
            	}

            unsigned int lowBito = simd * OutputPrecision;
            unsigned int highBito = (simd + 1) * OutputPrecision - 1;
            ap_uint <OutputPrecision> output_temp = *reinterpret_cast<ap_uint <OutputPrecision> *>(&result);
            output(highBito, lowBito) = output_temp;
        }

        out.write(output);
    }
}





template<
        // convolution parameters
        short unsigned int KernelDim,                // e.g 3 for a 1x3 conv kernel
        short unsigned int Channels,                // max number of input feature maps
        short unsigned int IFMDim,                // max width of input feature map
        short unsigned int Stride,                    // Stride
        short unsigned int Padding,                    // Padding
        short unsigned int OFMDim,                // (IFMDim - KernelDim + 2 x Padding) / Stride + 1
        // parallelization parameters
        short unsigned int SIMDWidth,               // number of SIMD lanes
        // matrix-vector unit parameters
        short unsigned int WeightsPrecision,        // Precisions for the weight
        short unsigned int WeightsIntPrecision,     // Weight int bitwidth
        short unsigned int InputPrecision,          // Precisions for the input activation
        short unsigned int InputIntPrecision,       // Input activation int bitwidth
        short unsigned int MulPrecision,            // Precision for the result of multiplication
        short unsigned int MulIntPrecision,         // Multiplication int bitwidth
        short unsigned int AccPrecision,            // Precision for the result of accumulation
        short unsigned int AccIntPrecision,         // Accumulation int bitwidth
        short unsigned int OutputPrecision,            // Precisions for the output activation
        short unsigned int OutputIntPrecision,        // Output activation int bitwidth
        bool use_profiler = false
>
void Conv1D_Depthwise(hls::stream <ap_uint<SIMDWidth * InputPrecision>> &in,
                         hls::stream <ap_uint<SIMDWidth * OutputPrecision>> &out,
                         const ap_uint <WeightsPrecision> weightMem[SIMDWidth][KernelDim * Channels /(SIMDWidth)],
                         Profiler_MAC *profiler = nullptr)
{

	#pragma HLS DATAFLOW

	hls::stream <ap_uint<SIMDWidth * InputPrecision>> BufferDepthwise_MacDepthwise("BufferDepthwise_MacDepthwise");

    Conv1DBuffer_K_S_Depthwise
            <
			KernelDim,
			Channels,
			IFMDim,
			Stride,
			OFMDim,
			SIMDWidth,
			InputPrecision,
			InputIntPrecision
            >
            (in, BufferDepthwise_MacDepthwise);

    Conv1DMac_Depthwise
            <
                    KernelDim,
                    Channels,
                    IFMDim,
                    Stride,
                    Padding,
                    OFMDim,
                    SIMDWidth,
                    WeightsPrecision,
                    WeightsIntPrecision,
                    InputPrecision,
                    InputIntPrecision,
                    MulPrecision,
                    MulIntPrecision,
                    AccPrecision,
                    AccIntPrecision,
                    OutputPrecision,
                    OutputIntPrecision,
                    use_profiler
            >
            (BufferDepthwise_MacDepthwise, out, weightMem, profiler);
}



template<
        // convolution parameters
        short unsigned int KernelDim,                // e.g 3 for a 1x3 conv kernel
        short unsigned int Channels,                // max number of input feature maps
        short unsigned int IFMDim,                // max width of input feature map
        short unsigned int Stride,                    // Stride
        short unsigned int Padding,                    // Padding
        short unsigned int OFMDim,                // (IFMDim - KernelDim + 2 x Padding) / Stride + 1
        // parallelization parameters
        short unsigned int SIMDWidth,               // number of SIMD lanes
        // matrix-vector unit parameters
        short unsigned int WeightsPrecision,        // Precisions for the weight
        short unsigned int WeightsIntPrecision,     // Weight int bitwidth
        short unsigned int InputPrecision,          // Precisions for the input activation
        short unsigned int InputIntPrecision,       // Input activation int bitwidth
        short unsigned int MulPrecision,            // Precision for the result of multiplication
        short unsigned int MulIntPrecision,         // Multiplication int bitwidth
        short unsigned int AccPrecision,            // Precision for the result of accumulation
        short unsigned int AccIntPrecision,         // Accumulation int bitwidth
        short unsigned int OutputPrecision,            // Precisions for the output activation
        short unsigned int OutputIntPrecision,        // Output activation int bitwidth
        bool use_profiler = false
>
void Conv1D_Depthwise_Unrolled(hls::stream <ap_uint<SIMDWidth * InputPrecision>> &in,
                         hls::stream <ap_uint<SIMDWidth * OutputPrecision>> &out,
                         const ap_uint <WeightsPrecision> weightMem[SIMDWidth][KernelDim * Channels /(SIMDWidth)],
                         Profiler_MAC *profiler = nullptr)
{

	#pragma HLS DATAFLOW

	hls::stream <ap_uint<KernelDim * SIMDWidth * InputPrecision>> BufferDepthwise_MacDepthwise("BufferDepthwise_MacDepthwise");

    Conv1DBufferK_Unrolled
            <
			KernelDim,
			Channels,
			IFMDim,
			Stride,
			OFMDim,
			SIMDWidth,
			InputPrecision,
			InputIntPrecision
            >
            (in, BufferDepthwise_MacDepthwise);



    Conv1DMac_Depthwise_Unrolled
            <
                    KernelDim,
                    Channels,
                    IFMDim,
                    Stride,
                    Padding,
                    OFMDim,
                    WeightsPrecision,
                    WeightsIntPrecision,
                    InputPrecision,
                    InputIntPrecision,
                    MulPrecision,
                    MulIntPrecision,
                    AccPrecision,
                    AccIntPrecision,
                    OutputPrecision,
                    OutputIntPrecision,
                    use_profiler
            >
            (BufferDepthwise_MacDepthwise, out, weightMem, profiler);
}


template<
        // convolution parameters
        short unsigned int KernelDim,                // e.g 3 for a 1x3 conv kernel
        short unsigned int IFMChannels,                // max number of input feature maps
        short unsigned int IFMDim,                // max width of input feature map
        short unsigned int Stride,                    // Stride
        short unsigned int Padding,                    // Padding
        short unsigned int OFMChannels,            // max number of output feature maps
        short unsigned int OFMDim,                // (IFMDim - KernelDim + 2 x Padding) / Stride + 1
        // parallelization parameters
        short unsigned int PECount,                 // number of PEs
        short unsigned int SIMDWidth,               // number of SIMD lanes
        // matrix-vector unit parameters
        short unsigned int BiasPrecision,            // Precisions for the bias
        short unsigned int BiasIntPrecision,        // Bias int bitwidth
        short unsigned int WeightsPrecision,        // Precisions for the weight
        short unsigned int WeightsIntPrecision,     // Weight int bitwidth
        short unsigned int InputPrecision,          // Precisions for the input activation
        short unsigned int InputIntPrecision,       // Input activation int bitwidth
        short unsigned int MulPrecision,            // Precision for the result of multiplication
        short unsigned int MulIntPrecision,         // Multiplication int bitwidth
        short unsigned int AccPrecision,            // Precision for the result of accumulation
        short unsigned int AccIntPrecision,         // Accumulation int bitwidth
        short unsigned int OutputPrecision,            // Precisions for the output activation
        short unsigned int OutputIntPrecision,        // Output activation int bitwidth
        bool use_profiler = false
>
void Conv1D_Pointwise(hls::stream <ap_uint<SIMDWidth * InputPrecision>> &in,
               hls::stream <ap_uint<PECount * OutputPrecision>> &out,
               const ap_uint <WeightsPrecision> weightMem[PECount][SIMDWidth][KernelDim * IFMChannels * OFMChannels /
                                                                              (SIMDWidth * PECount)],
               const ap_uint <BiasPrecision> biasMem[PECount][OFMChannels / PECount],
               Profiler_MAC *profiler = nullptr){

	#pragma HLS DATAFLOW
	if(KernelDim == 1 && IFMChannels == SIMDWidth && OFMChannels == PECount){
		#pragma HLS inline
	}

    hls::stream <ap_uint<SIMDWidth * InputPrecision>> BufferPointwise_MacPointwise("BufferPointwise_MacPointwise");

    Conv1DBuffer_K_S
            <
                    1,
                    IFMChannels,
                    OFMDim,
                    1,
                    OFMChannels,
                    OFMDim,
                    PECount,
                    SIMDWidth,
					InputPrecision,
					InputIntPrecision
            >
            (in, BufferPointwise_MacPointwise);


    Conv1DMac
            <
                    1,
                    IFMChannels,
                    OFMDim,
                    1,
                    0,
                    OFMChannels,
                    OFMDim,
                    PECount,
                    SIMDWidth,
                    BiasPrecision,
                    BiasIntPrecision,
                    WeightsPrecision,
                    WeightsIntPrecision,
					InputPrecision,
					InputIntPrecision,
                    MulPrecision,
                    MulIntPrecision,
                    AccPrecision,
                    AccIntPrecision,
                    OutputPrecision,
                    OutputIntPrecision,
                    use_profiler
            >
            (BufferPointwise_MacPointwise, out, weightMem, biasMem, profiler);

}

template<
        //Conv
        short unsigned int KernelDimConv,
        short unsigned int IFMChannelsConv,
        short unsigned int IFMDimConv,
        short unsigned int StrideConv,
        short unsigned int PaddingConv,
        short unsigned int OFMChannelsConv,
        short unsigned int OFMDimConv,
        short unsigned int PECountConv,
        short unsigned int SIMDWidthConv,
        short unsigned int BiasPrecisionConv,
        short unsigned int BiasIntPrecisionConv,
        short unsigned int WeightsPrecisionConvDepthwise,
        short unsigned int WeightsIntPrecisionConvDepthwise,
        short unsigned int WeightsPrecisionConvPointwise,
        short unsigned int WeightsIntPrecisionConvPointwise,
        short unsigned int InputPrecisionConv,
        short unsigned int InputIntPrecisionConv,
        short unsigned int MulPrecisionConvDepthwise,
        short unsigned int MulIntPrecisionConvDepthwise,
        short unsigned int AccPrecisionConvDepthwise,
        short unsigned int AccIntPrecisionConvDepthwise,
        short unsigned int OutputPrecisionConvDepthwise,
        short unsigned int OutputIntPrecisionConvDepthwise,
        short unsigned int MulPrecisionConvPointwise,
        short unsigned int MulIntPrecisionConvPointwise,
        short unsigned int AccPrecisionConvPointwise,
        short unsigned int AccIntPrecisionConvPointwise,
        short unsigned int OutputPrecisionConvPointwise,
        short unsigned int OutputIntPrecisionConvPointwise,
        //ReLu
        short unsigned int IFMChannelsRelu,
        short unsigned int IFMDimRelu,
        short unsigned int OFMChannelsRelu,
        short unsigned int OFMDimRelu,
        short unsigned int SIMDWidthRelu,
        short unsigned int InputPrecisionRelu,
        short unsigned int InputIntPrecisionRelu,
        short unsigned int OutputPrecisionRelu,
        short unsigned int OutputIntPrecisionRelu,
        bool use_profiler = false
>
void Conv1DReLuBlockDepthwiseSeparable(hls::stream <ap_uint<SIMDWidthConv * InputPrecisionConv>> &in,
                                       hls::stream <ap_uint<SIMDWidthRelu * OutputPrecisionRelu>> &out,
                                       const ap_uint <WeightsPrecisionConvDepthwise> weightMemDepthwise[SIMDWidthConv][KernelDimConv * IFMChannelsConv / SIMDWidthConv],
                                       const ap_uint <WeightsPrecisionConvPointwise> weightMemPointwise[PECountConv][SIMDWidthConv][IFMChannelsConv * OFMChannelsConv / (SIMDWidthConv * PECountConv)],
                                       const ap_uint <BiasPrecisionConv> biasMem[PECountConv][OFMChannelsConv / PECountConv],
									   Profiler_Conv_DS *profiler = nullptr) {
	#pragma HLS DATAFLOW

	if(KernelDimConv == 1 && IFMChannelsConv == SIMDWidthConv && OFMChannelsConv == PECountConv){
		#pragma HLS inline
	}

    hls::stream <ap_uint<SIMDWidthConv * OutputPrecisionConvDepthwise>> MacDepthwise_BufferPointwise("MacDepthwise_BufferPointwise");
    hls::stream <ap_uint<SIMDWidthRelu * InputPrecisionRelu>> MacPointwise_Relu("MacPointwise_Relu");

	constexpr int BUFFER_DEPTHWISE_POINTWISE_DEPTH = IFMChannelsConv/SIMDWidthConv + 2;

	#pragma HLS STREAM variable=MacDepthwise_BufferPointwise depth=BUFFER_DEPTHWISE_POINTWISE_DEPTH	//32

    Conv1D_Depthwise
            <
                    KernelDimConv,
                    IFMChannelsConv,
                    IFMDimConv,
                    StrideConv,
                    PaddingConv,
                    OFMDimConv,
                    SIMDWidthConv,
                    WeightsPrecisionConvDepthwise,
                    WeightsIntPrecisionConvDepthwise,
                    InputPrecisionConv,
                    InputIntPrecisionConv,
                    MulPrecisionConvDepthwise,
                    MulIntPrecisionConvDepthwise,
                    AccPrecisionConvDepthwise,
                    AccIntPrecisionConvDepthwise,
                    OutputPrecisionConvDepthwise,
                    OutputIntPrecisionConvDepthwise,
                    use_profiler
            >
            (in, MacDepthwise_BufferPointwise, weightMemDepthwise, &(profiler->profiler_mac_depthwise));

    Conv1D_Pointwise
            <
                    1,
                    IFMChannelsConv,
                    OFMDimConv,
                    1,
                    0,
                    OFMChannelsConv,
                    OFMDimConv,
                    PECountConv,
                    SIMDWidthConv,
                    BiasPrecisionConv,
                    BiasIntPrecisionConv,
                    WeightsPrecisionConvPointwise,
                    WeightsIntPrecisionConvPointwise,
                    OutputPrecisionConvDepthwise,
                    OutputIntPrecisionConvDepthwise,
                    MulPrecisionConvPointwise,
                    MulIntPrecisionConvPointwise,
                    AccPrecisionConvPointwise,
                    AccIntPrecisionConvPointwise,
                    OutputPrecisionConvPointwise,
                    OutputIntPrecisionConvPointwise,
                    use_profiler
            >
            (MacDepthwise_BufferPointwise, MacPointwise_Relu, weightMemPointwise, biasMem,  &(profiler->profiler_mac_pointwise));


    Relu1D
            <
                    IFMChannelsRelu,
                    IFMDimRelu,
                    SIMDWidthRelu,
                    InputPrecisionRelu,
                    InputIntPrecisionRelu,
                    OutputPrecisionRelu,
                    OutputIntPrecisionRelu
            >
            (MacPointwise_Relu, out);
}


template<
        //Conv
        short unsigned int KernelDimConv,
        short unsigned int IFMChannelsConv,
        short unsigned int IFMDimConv,
        short unsigned int StrideConv,
        short unsigned int PaddingConv,
        short unsigned int OFMChannelsConv,
        short unsigned int OFMDimConv,
        short unsigned int PECountConv,
        short unsigned int SIMDWidthConv,
        short unsigned int BiasPrecisionConv,
        short unsigned int BiasIntPrecisionConv,
        short unsigned int WeightsPrecisionConvDepthwise,
        short unsigned int WeightsIntPrecisionConvDepthwise,
        short unsigned int WeightsPrecisionConvPointwise,
        short unsigned int WeightsIntPrecisionConvPointwise,
        short unsigned int InputPrecisionConv,
        short unsigned int InputIntPrecisionConv,
        short unsigned int MulPrecisionConvDepthwise,
        short unsigned int MulIntPrecisionConvDepthwise,
        short unsigned int AccPrecisionConvDepthwise,
        short unsigned int AccIntPrecisionConvDepthwise,
        short unsigned int OutputPrecisionConvDepthwise,
        short unsigned int OutputIntPrecisionConvDepthwise,
        short unsigned int MulPrecisionConvPointwise,
        short unsigned int MulIntPrecisionConvPointwise,
        short unsigned int AccPrecisionConvPointwise,
        short unsigned int AccIntPrecisionConvPointwise,
        short unsigned int OutputPrecisionConvPointwise,
        short unsigned int OutputIntPrecisionConvPointwise,
        //ReLu
        short unsigned int IFMChannelsRelu,
        short unsigned int IFMDimRelu,
        short unsigned int OFMChannelsRelu,
        short unsigned int OFMDimRelu,
        short unsigned int SIMDWidthRelu,
        short unsigned int InputPrecisionRelu,
        short unsigned int InputIntPrecisionRelu,
        short unsigned int OutputPrecisionRelu,
        short unsigned int OutputIntPrecisionRelu,
        bool use_profiler = false
>
void Conv1DReLuBlockDepthwiseSeparableUnrolled(hls::stream <ap_uint<SIMDWidthConv * InputPrecisionConv>> &in,
                                       hls::stream <ap_uint<SIMDWidthRelu * OutputPrecisionRelu>> &out,
                                       const ap_uint <WeightsPrecisionConvDepthwise> weightMemDepthwise[SIMDWidthConv][KernelDimConv * IFMChannelsConv / SIMDWidthConv],
                                       const ap_uint <WeightsPrecisionConvPointwise> weightMemPointwise[PECountConv][SIMDWidthConv][IFMChannelsConv * OFMChannelsConv / (SIMDWidthConv * PECountConv)],
                                       const ap_uint <BiasPrecisionConv> biasMem[PECountConv][OFMChannelsConv / PECountConv],
									   Profiler_Conv_DS *profiler = nullptr) {

	#pragma HLS DATAFLOW

	if(IFMChannelsConv == SIMDWidthConv && OFMChannelsConv == PECountConv){
		#pragma HLS inline
	}

    hls::stream <ap_uint<SIMDWidthConv * OutputPrecisionConvDepthwise>> MacDepthwise_BufferPointwise("MacDepthwise_BufferPointwise");
    hls::stream <ap_uint<SIMDWidthRelu * InputPrecisionRelu>> MacPointwise_Relu("MacPointwise_Relu");

	#pragma HLS STREAM variable=MacDepthwise_BufferPointwise depth=4	//BUFFER_DEPTHWISE_POINTWISE_DEPTH		//32
	#pragma HLS STREAM variable=MacPointwise_Relu depth=2

    Conv1D_Depthwise_Unrolled
            <
                    KernelDimConv,
                    IFMChannelsConv,
                    IFMDimConv,
                    StrideConv,
                    PaddingConv,
                    OFMDimConv,
					SIMDWidthConv,
                    WeightsPrecisionConvDepthwise,
                    WeightsIntPrecisionConvDepthwise,
                    InputPrecisionConv,
                    InputIntPrecisionConv,
                    MulPrecisionConvDepthwise,
                    MulIntPrecisionConvDepthwise,
                    AccPrecisionConvDepthwise,
                    AccIntPrecisionConvDepthwise,
                    OutputPrecisionConvDepthwise,
                    OutputIntPrecisionConvDepthwise,
                    use_profiler
            >
            (in, MacDepthwise_BufferPointwise, weightMemDepthwise,  &(profiler->profiler_mac_depthwise));

    Conv1D_Pointwise
            <
                    1,
                    IFMChannelsConv,
                    OFMDimConv,
                    1,
                    0,
                    OFMChannelsConv,
                    OFMDimConv,
                    PECountConv,
                    SIMDWidthConv,
                    BiasPrecisionConv,
                    BiasIntPrecisionConv,
                    WeightsPrecisionConvPointwise,
                    WeightsIntPrecisionConvPointwise,
                    OutputPrecisionConvDepthwise,
                    OutputIntPrecisionConvDepthwise,
                    MulPrecisionConvPointwise,
                    MulIntPrecisionConvPointwise,
                    AccPrecisionConvPointwise,
                    AccIntPrecisionConvPointwise,
                    OutputPrecisionConvPointwise,
                    OutputIntPrecisionConvPointwise,
                    use_profiler
            >
            (MacDepthwise_BufferPointwise, MacPointwise_Relu, weightMemPointwise, biasMem,  &(profiler->profiler_mac_pointwise));


    Relu1D
            <
                    IFMChannelsRelu,
                    IFMDimRelu,
                    SIMDWidthRelu,
                    InputPrecisionRelu,
                    InputIntPrecisionRelu,
                    OutputPrecisionRelu,
                    OutputIntPrecisionRelu
            >
            (MacPointwise_Relu, out);
}



template<
        //Conv
        short unsigned int KernelDimConv,
        short unsigned int IFMChannelsConv,
        short unsigned int IFMDimConv,
        short unsigned int StrideConv,
        short unsigned int PaddingConv,
        short unsigned int OFMChannelsConv,
        short unsigned int OFMDimConv,
        short unsigned int PECountConv,
        short unsigned int SIMDWidthConv,
        short unsigned int BiasPrecisionConv,
        short unsigned int BiasIntPrecisionConv,
        short unsigned int WeightsPrecisionConv,
        short unsigned int WeightsIntPrecisionConv,
        short unsigned int InputPrecisionConv,
        short unsigned int InputIntPrecisionConv,
        short unsigned int MulPrecisionConv,
        short unsigned int MulIntPrecisionConv,
        short unsigned int AccPrecisionConv,
        short unsigned int AccIntPrecisionConv,
        short unsigned int OutputPrecisionConv,
        short unsigned int OutputIntPrecisionConv,
        //ReLu
        short unsigned int IFMChannelsRelu,
        short unsigned int IFMDimRelu,
        short unsigned int OFMChannelsReLu,
        short unsigned int OFMDimReLu,
        short unsigned int SIMDWidthRelu,
        short unsigned int InputPrecisionRelu,
        short unsigned int InputIntPrecisionRelu,
        short unsigned int OutputPrecisionRelu,
        short unsigned int OutputIntPrecisionRelu,
        //MaxPool
        short unsigned int KernelDimMaxPool,
        short unsigned int IFMChannelsMaxPool,
        short unsigned int IFMDimMaxPool,
        short unsigned int StrideMaxPool,
        short unsigned int PaddingMaxPool,
        short unsigned int OFMChannelsMaxPool,
        short unsigned int OFMDimMaxPool,
        short unsigned int SIMDWidthMaxPool,
        short unsigned int InputPrecisionMaxPool,
        short unsigned int InputIntPrecisionMaxPool,
        short unsigned int OutputPrecisionMaxPool,
        short unsigned int OutputIntPrecisionMaxPool,
        bool use_profiler = false
>
void Conv1DReLuMaxPoolBlock(hls::stream <ap_uint<SIMDWidthConv * InputPrecisionConv>> &in,
                            hls::stream <ap_uint<SIMDWidthMaxPool * OutputPrecisionMaxPool>> &out,
                            const ap_uint <WeightsPrecisionConv> weightMem[PECountConv][SIMDWidthConv][KernelDimConv *
                                                                                                       IFMChannelsConv *
                                                                                                       OFMChannelsConv /
                                                                                                       (SIMDWidthConv *
                                                                                                        PECountConv)],
                            const ap_uint <BiasPrecisionConv> biasMem[PECountConv][OFMChannelsConv / PECountConv],
                            Profiler_MAC *profiler = nullptr) {

    // set FIFO size on input stream to keep the streams running
    // number of cycles with no reads on the "in" stream
    //const unsigned int inNoReadCycles = KernelDimConv * OFMDimConv;
    // // expected production during the no-read phase
    //const unsigned int inFIFOSize = inNoReadCycles / MinCyclesPerInput;
    // set FIFO size on incoming stream
    //#pragma HLS STREAM variable=in depth=inFIFOSize

#pragma HLS DATAFLOW

    hls::stream <ap_uint<SIMDWidthConv * InputPrecisionConv>> Buffer_Mac("Buffer_Mac");
    hls::stream <ap_uint<SIMDWidthRelu * InputPrecisionRelu>> Mac_Relu("Mac_Relu");
    hls::stream <ap_uint<SIMDWidthMaxPool * InputPrecisionMaxPool>> Relu_MaxPool("Relu_MaxPool");

#pragma HLS STREAM variable=Buffer_Mac depth=2
#pragma HLS STREAM variable=Mac_Relu depth=2
#pragma HLS STREAM variable=Relu_MaxPool depth=2

    //Conv1DBuffer_new
    Conv1DBufferK
            <
                    KernelDimConv,
                    IFMChannelsConv,
                    IFMDimConv,
                    StrideConv,
                    OFMChannelsConv,
                    OFMDimConv,
                    PECountConv,
                    SIMDWidthConv,
                    InputPrecisionConv,
                    InputIntPrecisionConv
            >
            (in, Buffer_Mac);

    Conv1DMac
            <
                    KernelDimConv,
                    IFMChannelsConv,
                    IFMDimConv,
                    StrideConv,
                    PaddingConv,
                    OFMChannelsConv,
                    OFMDimConv,
                    PECountConv,
                    SIMDWidthConv,
                    BiasPrecisionConv,
                    BiasIntPrecisionConv,
                    WeightsPrecisionConv,
                    WeightsIntPrecisionConv,
                    InputPrecisionConv,
                    InputIntPrecisionConv,
                    MulPrecisionConv,
                    MulIntPrecisionConv,
                    AccPrecisionConv,
                    AccIntPrecisionConv,
                    OutputPrecisionConv,
                    OutputIntPrecisionConv,
                    use_profiler
            >
            (Buffer_Mac, Mac_Relu, weightMem, biasMem, profiler);

    Relu1D
            <
                    IFMChannelsRelu,
                    IFMDimRelu,
                    SIMDWidthRelu,
                    InputPrecisionRelu,
                    InputIntPrecisionRelu,
                    OutputPrecisionRelu,
                    OutputIntPrecisionRelu
            >
            (Mac_Relu, Relu_MaxPool);



    //MAXPool1DK_new
    //MAXPool1DK
    //MAXPool1D_new
    MAXPool1DK_Stride
            <
                    KernelDimMaxPool,
                    IFMChannelsMaxPool,
                    IFMDimMaxPool,
                    StrideMaxPool,
                    //PaddingMaxPool,
                    //OFMChannelsMaxPool,
                    OFMDimMaxPool,
                    SIMDWidthMaxPool,
                    InputPrecisionMaxPool,
                    InputIntPrecisionMaxPool,
                    OutputPrecisionMaxPool,
                    OutputIntPrecisionMaxPool
            >
            (Relu_MaxPool, out);
}

template<
        //Conv
        short unsigned int KernelDimConv,
        short unsigned int IFMChannelsConv,
        short unsigned int IFMDimConv,
        short unsigned int StrideConv,
        short unsigned int PaddingConv,
        short unsigned int OFMChannelsConv,
        short unsigned int OFMDimConv,
        short unsigned int PECountConv,
        short unsigned int SIMDWidthConv,
        short unsigned int BiasPrecisionConv,
        short unsigned int BiasIntPrecisionConv,
        short unsigned int WeightsPrecisionConv,
        short unsigned int WeightsIntPrecisionConv,
        short unsigned int InputPrecisionConv,
        short unsigned int InputIntPrecisionConv,
        short unsigned int MulPrecisionConv,
        short unsigned int MulIntPrecisionConv,
        short unsigned int AccPrecisionConv,
        short unsigned int AccIntPrecisionConv,
        short unsigned int OutputPrecisionConv,
        short unsigned int OutputIntPrecisionConv,
        //ReLu
        short unsigned int IFMChannelsRelu,
        short unsigned int IFMDimRelu,
        short unsigned int OFMChannelsRelu,
        short unsigned int OFMDimRelu,
        short unsigned int SIMDWidthRelu,
        short unsigned int InputPrecisionRelu,
        short unsigned int InputIntPrecisionRelu,
        short unsigned int OutputPrecisionRelu,
        short unsigned int OutputIntPrecisionRelu,
        bool use_profiler = false
>
void Conv1DReLuBlock(hls::stream <ap_uint<SIMDWidthConv * InputPrecisionConv>> &in,
                     hls::stream <ap_uint<SIMDWidthRelu * OutputPrecisionRelu>> &out,
                     const ap_uint <WeightsPrecisionConv> weightMem[PECountConv][SIMDWidthConv][KernelDimConv *
                                                                                                IFMChannelsConv *
                                                                                                OFMChannelsConv /
                                                                                                (SIMDWidthConv *
                                                                                                 PECountConv)],
                     const ap_uint <BiasPrecisionConv> biasMem[PECountConv][OFMChannelsConv / PECountConv],
                     Profiler_MAC *profiler = nullptr) {
#pragma HLS DATAFLOW

	if(IFMChannelsConv == SIMDWidthConv && OFMChannelsConv == PECountConv){
		#pragma HLS inline
	}

    hls::stream <ap_uint<SIMDWidthConv * InputPrecisionConv>> Buffer_Mac("Buffer_Mac");
    hls::stream <ap_uint<SIMDWidthRelu * InputPrecisionRelu>> Mac_Relu("Mac_Relu");

#pragma HLS STREAM variable=Buffer_Mac depth=2
#pragma HLS STREAM variable=Mac_Relu depth=2

    //Conv1DBuffer_new
    //Conv1DBufferK
    Conv1DBuffer_K_S
    <
                    KernelDimConv,
                    IFMChannelsConv,
                    IFMDimConv,
                    StrideConv,
                    OFMChannelsConv,
                    OFMDimConv,
                    PECountConv,
                    SIMDWidthConv,
                    InputPrecisionConv,
                    InputIntPrecisionConv
            >
            (in, Buffer_Mac);

    Conv1DMac
            <
                    KernelDimConv,
                    IFMChannelsConv,
                    IFMDimConv,
                    StrideConv,
                    PaddingConv,
                    OFMChannelsConv,
                    OFMDimConv,
                    PECountConv,
                    SIMDWidthConv,
                    BiasPrecisionConv,
                    BiasIntPrecisionConv,
                    WeightsPrecisionConv,
                    WeightsIntPrecisionConv,
                    InputPrecisionConv,
                    InputIntPrecisionConv,
                    MulPrecisionConv,
                    MulIntPrecisionConv,
                    AccPrecisionConv,
                    AccIntPrecisionConv,
                    OutputPrecisionConv,
                    OutputIntPrecisionConv,
                    use_profiler
            >
            (Buffer_Mac, Mac_Relu, weightMem, biasMem, profiler);

    Relu1D
            <
                    IFMChannelsRelu,
                    IFMDimRelu,
                    SIMDWidthRelu,
                    InputPrecisionRelu,
                    InputIntPrecisionRelu,
                    OutputPrecisionRelu,
                    OutputIntPrecisionRelu
            >
            (Mac_Relu, out);
}



template<
        //Conv
        short unsigned int KernelDimConv,
        short unsigned int IFMChannelsConv,
        short unsigned int IFMDimConv,
        short unsigned int StrideConv,
        short unsigned int PaddingConv,
        short unsigned int OFMChannelsConv,
        short unsigned int OFMDimConv,
        short unsigned int PECountConv,
        short unsigned int SIMDWidthConv,
        short unsigned int BiasPrecisionConv,
        short unsigned int BiasIntPrecisionConv,
        short unsigned int WeightsPrecisionConv,
        short unsigned int WeightsIntPrecisionConv,
        short unsigned int InputPrecisionConv,
        short unsigned int InputIntPrecisionConv,
        short unsigned int MulPrecisionConv,
        short unsigned int MulIntPrecisionConv,
        short unsigned int AccPrecisionConv,
        short unsigned int AccIntPrecisionConv,
        short unsigned int OutputPrecisionConv,
        short unsigned int OutputIntPrecisionConv,
        //ReLu
        short unsigned int IFMChannelsRelu,
        short unsigned int IFMDimRelu,
        short unsigned int OFMChannelsRelu,
        short unsigned int OFMDimRelu,
        short unsigned int SIMDWidthRelu,
        short unsigned int InputPrecisionRelu,
        short unsigned int InputIntPrecisionRelu,
        short unsigned int OutputPrecisionRelu,
        short unsigned int OutputIntPrecisionRelu,
        bool use_profiler = false
>
void Conv1DReLuBlockUnrolled(hls::stream <ap_uint<SIMDWidthConv * InputPrecisionConv>> &in,
                             hls::stream <ap_uint<SIMDWidthRelu * OutputPrecisionRelu>> &out,
                             const ap_uint <WeightsPrecisionConv> weightMem[PECountConv][SIMDWidthConv][KernelDimConv *
                                                                                                        IFMChannelsConv *
                                                                                                        OFMChannelsConv /
                                                                                                        (SIMDWidthConv *
                                                                                                         PECountConv)],
                             const ap_uint <BiasPrecisionConv> biasMem[PECountConv][OFMChannelsConv / PECountConv],
                             Profiler_MAC *profiler = nullptr) {

	#pragma HLS DATAFLOW

	if(IFMChannelsConv == SIMDWidthConv && OFMChannelsConv == PECountConv){
		#pragma HLS inline
	}

    CASSERT_DATAFLOW(IFMChannelsConv == SIMDWidthConv);
    CASSERT_DATAFLOW(OFMChannelsConv == PECountConv);

    hls::stream <ap_uint<SIMDWidthConv * InputPrecisionConv * KernelDimConv>> Buffer_Mac("Buffer_Mac");
    hls::stream <ap_uint<SIMDWidthRelu * InputPrecisionRelu>> Mac_Relu("Mac_Relu");

	#pragma HLS STREAM variable=Buffer_Mac depth=2
	#pragma HLS STREAM variable=Mac_Relu depth=2


    Conv1DBufferK_Unrolled
            <
                    KernelDimConv,
                    IFMChannelsConv,
                    IFMDimConv,
                    StrideConv,
                    OFMDimConv,
                    SIMDWidthConv,
                    InputPrecisionConv,
                    InputIntPrecisionConv
            >
            (in, Buffer_Mac);


    Conv1DMac_Unrolled
            <
                    KernelDimConv,
                    IFMChannelsConv,
                    IFMDimConv,
                    StrideConv,
                    PaddingConv,
                    OFMChannelsConv,
                    OFMDimConv,
                    PECountConv,
                    SIMDWidthConv,
                    BiasPrecisionConv,
                    BiasIntPrecisionConv,
                    WeightsPrecisionConv,
                    WeightsIntPrecisionConv,
                    InputPrecisionConv,
                    InputIntPrecisionConv,
                    MulPrecisionConv,
                    MulIntPrecisionConv,
                    AccPrecisionConv,
                    AccIntPrecisionConv,
                    OutputPrecisionConv,
                    OutputIntPrecisionConv,
                    use_profiler
            >
            (Buffer_Mac, Mac_Relu, weightMem, biasMem, profiler);

    Relu1D
            <
                    IFMChannelsRelu,
                    IFMDimRelu,
                    SIMDWidthRelu,
                    InputPrecisionRelu,
                    InputIntPrecisionRelu,
                    OutputPrecisionRelu,
                    OutputIntPrecisionRelu
            >
            (Mac_Relu, out);
}
