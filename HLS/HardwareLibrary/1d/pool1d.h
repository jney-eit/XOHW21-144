template<
        short unsigned int KernelDim,        		// e.g 3 for a 1x3 1D conv kernel
        short unsigned int Channels,			// max number of input feature maps
        short unsigned int IFMDim,               	// max width of input feature map
        short unsigned int Stride,					// Stride
        short unsigned int OFMDim,               	// (IFMDim - KernelDim + 2 x Padding) / Stride + 1
        short unsigned int SIMDWidth,              // number of SIMD lanes
        short unsigned int Precision,         		// Precisions for the input/output activation
        short unsigned int IntPrecision      		// Input/Output activation int bitwidth
>
void Pool1DK_Stride1_Buffer(hls::stream<ap_uint<SIMDWidth * Precision> > & in,
                            hls::stream<ap_uint<SIMDWidth * Precision> > & out)
{

    // Works only with Stride=1 !!!
    CASSERT_DATAFLOW(Channels % SIMDWidth == 0);
    CASSERT_DATAFLOW(Stride == 1);

    const unsigned int synapseFold = Channels / SIMDWidth;
    unsigned int read_indices[KernelDim][KernelDim];
    for(unsigned int i = 0;  i < KernelDim; i++){
        for(unsigned int j = 0; j < KernelDim; j++){
            unsigned int curr_read_indice = (j + i) % KernelDim;
            read_indices[i][j] = curr_read_indice;
        }
    }

    ap_uint<SIMDWidth * Precision> inputBuf[KernelDim][synapseFold];

    // Read in first kernelDim buffers
    for(unsigned int ptr_k = 0; ptr_k < KernelDim; ptr_k++){
        for(unsigned int ptr_simd = 0; ptr_simd < synapseFold; ptr_simd++){
#pragma HLS PIPELINE II=1
            //std::cout << "Reading in into inputBuf[" << ptr_k << "][" << ptr_simd << "]" << std::endl;
            inputBuf[ptr_k][ptr_simd] = in.read();
            //read_count++;
        }
    }

    ap_uint<16> read_index = 0;
    for(unsigned int ofm_iter = 0; ofm_iter < OFMDim; ofm_iter++, read_index+=Stride){
        for (unsigned int read_index_k = 0; read_index_k < KernelDim; read_index_k++) {
            for(unsigned int ptr_simd = 0; ptr_simd < synapseFold; ptr_simd++) {
#pragma HLS PIPELINE II=1
                if (read_index == KernelDim) {
                    read_index = 0;
                }
                unsigned int ptr_k = read_indices[read_index][read_index_k];
                out.write(inputBuf[ptr_k][ptr_simd]);
                //std::cout << "Buffer has written output: " << inputBuf[ptr_k][ptr_simd] << std::endl;


                if (ofm_iter < IFMDim - KernelDim && read_index_k == 0) {
                    //std::cout << "Reading in into inputBuf[" << ptr_k << "][" << ptr_simd << "]" << std::endl;
                    ptr_k = read_indices[read_index][read_index_k];
                    inputBuf[ptr_k][ptr_simd] = in.read();
                    //read_count++;
                }
            }
        }
    }
    //std::cout << "Maxpool total reads: " << read_count << std::endl;
}



template<
        short unsigned int KernelDim,        		// e.g 3 for a 1x3 1D conv kernel
        short unsigned int Channels,			// max number of input feature maps
        short unsigned int IFMDim,               	// max width of input feature map
        short unsigned int Stride,					// Stride
        short unsigned int OFMDim,               	// (IFMDim - KernelDim + 2 x Padding) / Stride + 1
        short unsigned int SIMDWidth,              // number of SIMD lanes
        short unsigned int Precision,         		// Precisions for the input/output activation
        short unsigned int IntPrecision      		// Input/Output activation int bitwidth
>
void Pool1DK_Stride1_Unrolled_Buffer(hls::stream<ap_uint<SIMDWidth * Precision> > & in,
                            hls::stream<ap_uint<SIMDWidth * Precision * KernelDim> > & out)
{

    // Works only with Stride=1 !!!

    CASSERT_DATAFLOW(Channels == SIMDWidth);
    CASSERT_DATAFLOW(Stride == 1);

    const unsigned int synapseFold = Channels / SIMDWidth;
    unsigned int read_indices[KernelDim + 1][KernelDim + 1];
    for(unsigned int i = 0;  i < KernelDim + 1; i++){
        for(unsigned int j = 0; j < KernelDim + 1; j++){
            unsigned int curr_read_indice = (j + i) % (KernelDim + 1);
            read_indices[i][j] = curr_read_indice;
        }
    }

    ap_uint<SIMDWidth * Precision> inputBuf[KernelDim + 1][synapseFold];

    // Read in first kernelDim buffers
    for(unsigned int ptr_k = 0; ptr_k < KernelDim; ptr_k++){
        for(unsigned int ptr_simd = 0; ptr_simd < synapseFold; ptr_simd++){
#pragma HLS PIPELINE II=1
            //std::cout << "Reading in into inputBuf[" << ptr_k << "][" << ptr_simd << "]" << std::endl;
            inputBuf[ptr_k][ptr_simd] = in.read();
            //read_count++;
        }
    }

    ap_uint<16> read_index = 0;
    for(unsigned int ofm_iter = 0; ofm_iter < OFMDim; ofm_iter++, read_index++){
        for(unsigned int ptr_simd = 0; ptr_simd < synapseFold; ptr_simd++) {
#pragma HLS PIPELINE II=1
            if (read_index == KernelDim + 1) {
                read_index = 0;
            }

            ap_uint<SIMDWidth * Precision * KernelDim> output;
            for(unsigned int pixel_iter = 0; pixel_iter < KernelDim; pixel_iter++){
                #pragma HLS UNROLL
                unsigned int ptr_k = read_indices[read_index][pixel_iter];
                output((pixel_iter + 1) * SIMDWidth * Precision - 1, pixel_iter * SIMDWidth * Precision) = inputBuf[ptr_k][ptr_simd];

            }
            out.write(output);
            //std::cout << "Buffer has written output: " << inputBuf[ptr_k][ptr_simd] << std::endl;


            if (ofm_iter < IFMDim - KernelDim) {
                //std::cout << "Reading in into inputBuf[" << ptr_k << "][" << ptr_simd << "]" << std::endl;
                unsigned int ptr_k = read_indices[read_index][KernelDim];
                inputBuf[ptr_k][ptr_simd] = in.read();
                //read_count++;
            }
        }
    }
    //std::cout << "Maxpool total reads: " << read_count << std::endl;
}






//Supports various-size kernels and different stride
template<
        short unsigned int KernelDim,        		// e.g 2 for a 1x2
        short unsigned int Channels,			// number of input feature maps
        short unsigned int IFMDim,               	// length of input sequence
        short unsigned int Stride,					// Stride
        short unsigned int OFMDim,               	// length of output sequence // OFMDim = IFMDim/KernelDim
        short unsigned int SIMDWidth,          	// number of SIMD lanes // NOT USED YET
        short unsigned int InputPrecision,         // precisions for the input activation
        short unsigned int InputIntPrecision,      // input activation int bitwidth
        short unsigned int OutputPrecision,        // precisions for the output activation // THE SAME AS OutputPrecision
        short unsigned int OutputIntPrecision      // output activation int bitwidth // THE SAME AS OutputIntPrecision
>
void MaxPool1DK_Calc(hls::stream<ap_uint<SIMDWidth * InputPrecision> > & in,
                     hls::stream<ap_uint<SIMDWidth * OutputPrecision> > & out)
{
	/*
    typedef ap_fixed<InputPrecision, InputIntPrecision, AP_RND_ZERO, AP_SAT> Input_t;
    typedef ap_fixed<OutputPrecision, OutputIntPrecision, AP_RND_ZERO, AP_SAT> Output_t;
	*/

    typedef ap_fixed<InputPrecision, InputIntPrecision, AP_RND_ZERO, AP_WRAP> Input_t;
    typedef ap_fixed<OutputPrecision, OutputIntPrecision, AP_RND_ZERO, AP_WRAP> Output_t;

    ap_uint<SIMDWidth * InputPrecision> buffer[Channels / SIMDWidth];
	#pragma HLS ARRAY_PARTITION variable=buffer complete

    const unsigned int synapseFold = Channels / SIMDWidth;
    const unsigned int OUTPUT = KernelDim - 2;
    const unsigned int INIT = KernelDim - 1;
    ap_uint<bitsNeeded(synapseFold)> sf = 0;
    ap_uint<bitsNeeded(INIT + 1)> init = 1;
    ap_uint<bitsNeeded(INIT + 1)> k = 0;

    ap_uint<SIMDWidth * OutputPrecision> output;

    //std::cout << "IFM DIM MAXPOOL: " << IFMDim << std::endl;
    //std::cout << "OFM DIM MAXPOOL: " << OFMDim << std::endl;

    loop_dim:for (unsigned int i = 0; i < synapseFold * OFMDim * KernelDim; i++)
{
#pragma HLS PIPELINE II=1

    if(init == 1)
    {
        buffer[sf] = in.read();
        sf++;

        if(sf == synapseFold)
        {
            sf = 0;
            init = 0;
        }
    }
    else
    {
        ap_int<SIMDWidth * InputPrecision> temp_data_0 = in.read();
        ap_int<SIMDWidth * InputPrecision> temp_data_1 = buffer[sf];

        loop_ch_max_pool:for(unsigned int simd = 0; simd < SIMDWidth; simd++)
    {
#pragma HLS UNROLL

        unsigned int lowBit = simd * InputPrecision;
        unsigned int highBit = (simd + 1) * InputPrecision - 1;
        ap_uint<InputPrecision> temp_data_0_in_ch = temp_data_0(highBit, lowBit);
        ap_uint<InputPrecision> temp_data_1_in_ch = temp_data_1(highBit, lowBit);

        Input_t data_0 = *reinterpret_cast<Input_t *>(&temp_data_0_in_ch);
        Input_t data_1 = *reinterpret_cast<Input_t *>(&temp_data_1_in_ch);

        Output_t result;

        if(data_0 > data_1)
        {
            result = data_0;
        }
        else
        {
            result = data_1;
        }

        ap_uint<OutputPrecision> output_temp = *reinterpret_cast< ap_uint<OutputPrecision> *>(&result);
        output(highBit, lowBit) = output_temp;

    }

        if(k == OUTPUT)
        {
            out.write(output);
        }
        else
        {
            buffer[sf] = output;
        }

        sf++;
        if(sf == synapseFold)
        {
            sf = 0;
            k++;
            if(k == INIT)
            {
                k = 0;
                init = 1;
            }
        }

    }
}

    //std::cout << "Maxpool total reads: " << read_count << std::endl;
    //std::cout << "Maxpool total writes: " << write_count << std::endl;

    // Read in values still in stream because IfMDim % KernelSize != 0
    unsigned int overhead = (Stride == 1) ? 0 : synapseFold * (IFMDim % KernelDim);
    for(int i = 0; i < overhead; i++){
        in.read();
    }
}


// Supports various-size kernels and different stride, produces one output in one CC
// Only works for SIMD == Channels!!!!
template<
        short unsigned int KernelDim,        		// e.g 2 for a 1x2
        short unsigned int Channels,			// number of input feature maps
        short unsigned int IFMDim,               	// length of input sequence
        short unsigned int Stride,					// Stride
        short unsigned int OFMDim,               	// length of output sequence // OFMDim = IFMDim/KernelDim
        short unsigned int SIMDWidth,          	// number of SIMD lanes // NOT USED YET
        short unsigned int InputPrecision,         // precisions for the input activation
        short unsigned int InputIntPrecision,      // input activation int bitwidth
        short unsigned int OutputPrecision,        // precisions for the output activation // THE SAME AS OutputPrecision
        short unsigned int OutputIntPrecision      // output activation int bitwidth // THE SAME AS OutputIntPrecision
>
void MaxPool1DK_Unroll_Calc(hls::stream<ap_uint<SIMDWidth * InputPrecision * KernelDim> > & in,
                            hls::stream<ap_uint<SIMDWidth * OutputPrecision> > & out)
{

    const unsigned int synapseFold = Channels / SIMDWidth;

    typedef ap_fixed<InputPrecision, InputIntPrecision, AP_RND_ZERO, AP_WRAP> Input_t;
    typedef ap_fixed<OutputPrecision, OutputIntPrecision, AP_RND_ZERO, AP_WRAP> Output_t;

    unsigned int out_count = 0;

    ap_uint<SIMDWidth * InputPrecision * KernelDim> buffer;

    const unsigned int OUTPUT = KernelDim - 2;
    const unsigned int INIT = KernelDim - 1;
    ap_uint<bitsNeeded(INIT + 1)> init = 1;
    ap_uint<bitsNeeded(INIT + 1)> k = 0;

    ap_uint<SIMDWidth * OutputPrecision> current_max;

    loop_dim:for (unsigned int i = 0; i < OFMDim; i++)
	{
	#pragma HLS PIPELINE II=1


		ap_uint<SIMDWidth * InputPrecision * KernelDim> input = in.read();
		current_max = input(InputPrecision * SIMDWidth - 1, 0);

		for(unsigned int kernel_iter = 1; kernel_iter < KernelDim; kernel_iter++){
	#pragma HLS UNROLL

			ap_uint<SIMDWidth * InputPrecision> input_1 = input((kernel_iter + 1) * InputPrecision * SIMDWidth - 1, kernel_iter * InputPrecision * SIMDWidth);
			loop_ch_max_pool:for(unsigned int simd = 0; simd < SIMDWidth; simd++)
		{
	#pragma HLS UNROLL

			unsigned int lowBit = simd * InputPrecision;
			unsigned int highBit = (simd + 1) * InputPrecision - 1;
			ap_uint<InputPrecision> temp_data_0_in_ch = current_max(highBit, lowBit);
			ap_uint<InputPrecision> temp_data_1_in_ch = input_1(highBit, lowBit);

			Input_t data_0 = *reinterpret_cast<Input_t *>(&temp_data_0_in_ch);
			Input_t data_1 = *reinterpret_cast<Input_t *>(&temp_data_1_in_ch);

			Output_t result;

			if(data_0 > data_1)
			{
				result = data_0;
			}
			else
			{
				result = data_1;
			}


			ap_uint<OutputPrecision> output_temp = *reinterpret_cast< ap_uint<OutputPrecision> *>(&result);
			current_max(highBit, lowBit) = output_temp;
		}
		}
		out.write(current_max);
	}

    //std::cout << "Maxpool total reads: " << read_count << std::endl;


    // Read in values still in stream because IfMDim % KernelSize != 0
    //unsigned int overhead = (Stride == 1) ? 0 : (IFMDim % KernelDim);
    //for(int i = 0; i < overhead; i++){
    //    in.read();
    //}
}


//Supports various-size kernels and stride=1 or stride=KernelDim
template<
        short unsigned int KernelDim,        		// e.g 2 for a 1x2
        short unsigned int Channels,			// number of input feature maps
        short unsigned int IFMDim,               	// length of input sequence
        short unsigned int Stride,					// Stride
        short unsigned int OFMDim,               	// length of output sequence // OFMDim = IFMDim/KernelDim
        short unsigned int SIMDWidth,          	// number of SIMD lanes // NOT USED YET
        short unsigned int InputPrecision,         // precisions for the input activation
        short unsigned int InputIntPrecision,      // input activation int bitwidth
        short unsigned int OutputPrecision,        // precisions for the output activation // THE SAME AS OutputPrecision
        short unsigned int OutputIntPrecision      // output activation int bitwidth // THE SAME AS OutputIntPrecision
>
void MAXPool1DK_Stride(hls::stream<ap_uint<SIMDWidth * InputPrecision> > & in,
                       hls::stream<ap_uint<SIMDWidth * OutputPrecision> > & out){

#pragma HLS DATAFLOW


    // Works only with Stride=1 or Stride = KernelDim !!!
    CASSERT_DATAFLOW(Stride == 1 || Stride == KernelDim);

	hls::stream<ap_uint<SIMDWidth * InputPrecision> > Buffer_Calc("Buffer_Calc");
	if(Stride == 1) {
		Pool1DK_Stride1_Buffer<KernelDim, Channels, IFMDim, Stride, OFMDim, SIMDWidth, InputPrecision, InputIntPrecision>(in, Buffer_Calc);
	}

	auto &pool_calc_in = (Stride == KernelDim) ? in : Buffer_Calc;
	MaxPool1DK_Calc<KernelDim, Channels, IFMDim, Stride, OFMDim, SIMDWidth, InputPrecision, InputIntPrecision, OutputPrecision, OutputIntPrecision>(pool_calc_in, out);


}


//Supports various-size kernels and stride=1 or stride=KernelDim
template<
        short unsigned int KernelDim,        		// e.g 2 for a 1x2
        short unsigned int Channels,			// number of input feature maps
        short unsigned int IFMDim,               	// length of input sequence
        short unsigned int Stride,					// Stride
        short unsigned int OFMDim,               	// length of output sequence // OFMDim = IFMDim/KernelDim
        short unsigned int SIMDWidth,          	// number of SIMD lanes // NOT USED YET
        short unsigned int InputPrecision,         // precisions for the input activation
        short unsigned int InputIntPrecision,      // input activation int bitwidth
        short unsigned int OutputPrecision,        // precisions for the output activation // THE SAME AS OutputPrecision
        short unsigned int OutputIntPrecision     // output activation int bitwidth // THE SAME AS OutputIntPrecision
>
void MAXPool1DK_Unroll_Stride(hls::stream<ap_uint<SIMDWidth * InputPrecision * KernelDim> > & in,
                       	   	  hls::stream<ap_uint<SIMDWidth * OutputPrecision> > & out){

#pragma HLS DATAFLOW


    // Works only with Stride=1 or Stride = KernelDim !!!
    CASSERT_DATAFLOW(Stride == 1 || Stride == KernelDim);

	hls::stream<ap_uint<SIMDWidth * InputPrecision * KernelDim> > Buffer_Calc("Buffer_Calc");
	if(Stride == 1) {
		// Stream multiple pixels to maxpool at once
		hls::stream<ap_uint<SIMDWidth * InputPrecision>> WidthAdjusted("WidthAdjusted");
		StreamingDataWidthConverterFixed<SIMDWidth * InputPrecision * KernelDim, SIMDWidth * InputPrecision> (in, WidthAdjusted, IFMDim / KernelDim);
		Pool1DK_Stride1_Unrolled_Buffer<KernelDim, Channels, IFMDim, Stride, OFMDim, SIMDWidth, InputPrecision, InputIntPrecision>(WidthAdjusted, Buffer_Calc);
	}

	auto &pool_calc_in = (Stride == KernelDim) ? in : Buffer_Calc;
	MaxPool1DK_Unroll_Calc<KernelDim, Channels, IFMDim, Stride, OFMDim, SIMDWidth, InputPrecision, InputIntPrecision, OutputPrecision, OutputIntPrecision>(pool_calc_in, out);


}



//Supports various-size kernels and stride=1 or stride=KernelDim
template<
        short unsigned int KernelDim,        		// e.g 2 for a 1x2
        short unsigned int Channels,			// number of input feature maps
        short unsigned int IFMDim,               	// length of input sequence
        short unsigned int Stride,					// Stride
        short unsigned int OFMDim,               	// length of output sequence // OFMDim = IFMDim/KernelDim
        short unsigned int SIMDWidth,          	// number of SIMD lanes // NOT USED YET
        short unsigned int InputPrecision,         // precisions for the input activation
        short unsigned int InputIntPrecision,      // input activation int bitwidth
        short unsigned int OutputPrecision,        // precisions for the output activation // THE SAME AS OutputPrecision
        short unsigned int OutputIntPrecision      // output activation int bitwidth // THE SAME AS OutputIntPrecision
>
void AveragePool1DK_Calc(hls::stream<ap_uint<SIMDWidth * InputPrecision> > & in,
                         hls::stream<ap_uint<SIMDWidth * OutputPrecision> > & out)
{
	/*
    typedef ap_fixed<InputPrecision, InputIntPrecision, AP_RND_ZERO, AP_SAT> Input_t;
    typedef ap_fixed<OutputPrecision, OutputIntPrecision, AP_RND_ZERO, AP_SAT> Output_t;
	*/

    typedef ap_fixed<InputPrecision, InputIntPrecision, AP_RND_ZERO, AP_WRAP> Input_t;
    typedef ap_fixed<OutputPrecision, OutputIntPrecision, AP_RND_ZERO, AP_WRAP> Output_t;

    ap_uint<SIMDWidth * InputPrecision> buffer[Channels / SIMDWidth];

    const unsigned int synapseFold = Channels / SIMDWidth;
    const unsigned int OUTPUT = KernelDim - 2;
    const unsigned int INIT = KernelDim - 1;
    ap_uint<bitsNeeded(synapseFold)> sf = 0;
    ap_uint<bitsNeeded(INIT + 1)> init = 1;
    ap_uint<bitsNeeded(INIT + 1)> k = 0;

    ap_uint<SIMDWidth * OutputPrecision> output;

    loop_dim:for (unsigned int i = 0; i < synapseFold * OFMDim * KernelDim; i++)
{
#pragma HLS PIPELINE II=1

    if(init == 1)
    {
        buffer[sf] = in.read();
        sf++;

        if(sf == synapseFold)
        {
            sf = 0;
            init = 0;
        }
    }
    else
    {
        ap_int<SIMDWidth * InputPrecision> temp_data_0 = in.read();
        ap_int<SIMDWidth * InputPrecision> temp_data_1 = buffer[sf];

        loop_ch:for(unsigned int simd = 0; simd < SIMDWidth; simd++)
    {
#pragma HLS UNROLL

        unsigned int lowBit = simd * InputPrecision;
        unsigned int highBit = (simd + 1) * InputPrecision - 1;
        ap_uint<InputPrecision> temp_data_0_in_ch = temp_data_0(highBit, lowBit);
        ap_uint<InputPrecision> temp_data_1_in_ch = temp_data_1(highBit, lowBit);

        Input_t data_0 = *reinterpret_cast<Input_t *>(&temp_data_0_in_ch);
        Input_t data_1 = *reinterpret_cast<Input_t *>(&temp_data_1_in_ch);

        data_1 += data_0;

        ap_uint<OutputPrecision> output_temp = *reinterpret_cast< ap_uint<OutputPrecision> *>(&data_1);
        output(highBit, lowBit) = output_temp;

    }

        if(k == OUTPUT)
        {
            loop_ch_avg:for(unsigned int simd = 0; simd < SIMDWidth; simd++)
        {
#pragma HLS UNROLL
            unsigned int lowBit = simd * InputPrecision;
            unsigned int highBit = (simd + 1) * InputPrecision - 1;
            ap_uint<InputPrecision> temp_data = output(highBit, lowBit);
            Input_t temp_data_cast = *reinterpret_cast<Input_t *>(&temp_data);
            Input_t temp_data_avg = temp_data_cast / KernelDim;

            ap_uint<OutputPrecision> output_temp = *reinterpret_cast< ap_uint<OutputPrecision> *>(&temp_data_avg);
            output(highBit, lowBit) = output_temp;
        }

            out.write(output);
        }
        else
        {
            buffer[sf] = output;
        }

        sf++;
        if(sf == synapseFold)
        {
            sf = 0;
            k++;
            if(k == INIT)
            {
                k = 0;
                init = 1;
            }
        }

    }
}

    // Read in values still in stream because IfMDim % KernelSize != 0
    unsigned int overhead = (Stride == 1) ? 0 : synapseFold * (IFMDim % KernelDim);
    for(int i = 0; i < overhead; i++){
        in.read();
    }

}


//Supports various-size kernels and different stride
template<
        short unsigned int KernelDim,        		// e.g 2 for a 1x2
        short unsigned int Channels,			// number of input feature maps
        short unsigned int IFMDim,               	// length of input sequence
        short unsigned int Stride,					// Stride
        short unsigned int OFMDim,               	// length of output sequence // OFMDim = IFMDim/KernelDim
        short unsigned int SIMDWidth,          	// number of SIMD lanes // NOT USED YET
        short unsigned int InputPrecision,         // precisions for the input activation
        short unsigned int InputIntPrecision,      // input activation int bitwidth
        short unsigned int OutputPrecision,        // precisions for the output activation // THE SAME AS OutputPrecision
        short unsigned int OutputIntPrecision      // output activation int bitwidth // THE SAME AS OutputIntPrecision
>
void AveragePool1DK_Stride(hls::stream<ap_uint<SIMDWidth * InputPrecision> > & in,
                           hls::stream<ap_uint<SIMDWidth * OutputPrecision> > & out){

#pragma HLS DATAFLOW

    // Works only with Stride=1 or Stride = KernelDim !!!
    CASSERT_DATAFLOW(Stride == 1 || Stride == KernelDim);

    hls::stream<ap_uint<SIMDWidth * InputPrecision> > Buffer_Calc("Buffer_Calc");
    if(Stride == 1) {
        Pool1DK_Stride1_Buffer<KernelDim, Channels, IFMDim, Stride, OFMDim, SIMDWidth, InputPrecision, InputIntPrecision>(in, Buffer_Calc);
    }

    auto &pool_calc_in = (Stride == KernelDim) ? in : Buffer_Calc;
    AveragePool1DK_Calc<KernelDim, Channels, IFMDim, Stride, OFMDim, SIMDWidth, InputPrecision, InputIntPrecision, OutputPrecision, OutputIntPrecision>(pool_calc_in, out);

}




template<
		 short unsigned int IFMChannels,			// number of input feature maps
		 short unsigned int IFMDim,               	// length of input sequence
		 short unsigned int OFMChannels,			// number of output feature maps // THE SAME AS IFMChannels
		 short unsigned int OFMDim,               	// length of output sequence // OFMDim = IFMDim/KernelDim
		 short unsigned int SIMDWidth,          	// number of SIMD lanes // NOT USED YET
		 short unsigned int InputPrecision,         // precisions for the input activation
		 short unsigned int InputIntPrecision,      // input activation int bitwidth
		 short unsigned int AccPrecision,            // Precision for the result of accumulation
		 short unsigned int AccIntPrecision,         // Accumulation int bitwidth
		 short unsigned int OutputPrecision,        // precisions for the output activation // THE SAME AS OutputPrecision
		 short unsigned int OutputIntPrecision,      // output activation int bitwidth // THE SAME AS OutputIntPrecision
		 bool use_profiler = false
		 >
void GlobalAveragePool1D(hls::stream<ap_uint<SIMDWidth * InputPrecision> > & in,
		        		 hls::stream<ap_uint<SIMDWidth * OutputPrecision> > & out,
						 Profiler_GAP* profiler = nullptr)
{

	typedef ap_fixed<InputPrecision, InputIntPrecision, AP_RND_ZERO, AP_WRAP> Input_t;
	typedef ap_fixed<AccPrecision, AccIntPrecision, AP_RND_ZERO, AP_WRAP> Acc_t;
	typedef ap_fixed<OutputPrecision, OutputIntPrecision, AP_RND_ZERO, AP_WRAP> Output_t;

	const unsigned int synapseFold = IFMChannels / SIMDWidth;

	Acc_t accu[SIMDWidth][synapseFold];
#pragma HLS ARRAY_PARTITION variable=accu complete dim=1

	ap_uint<SIMDWidth * OutputPrecision> output;

	loop_sf_init:for(unsigned int sf = 0; sf < synapseFold; sf++)
	{
	#pragma HLS PIPELINE II=1

		loop_simd_init:for(unsigned int simd = 0; simd < SIMDWidth; simd++)
		{
		#pragma HLS UNROLL

			accu[simd][sf] = (Acc_t)0.0;
		}
	}

	loop_dim:for(unsigned int dim = 0; dim < IFMDim; dim++)
	{
		loop_sf:for(unsigned int sf = 0; sf < synapseFold; sf++)
		{
		#pragma HLS PIPELINE II=1

		#pragma HLS dependence variable=accu inter false

			ap_int<SIMDWidth * InputPrecision> input = in.read();

			loop_simd:for(unsigned int simd = 0; simd < SIMDWidth; simd++)
			{
			#pragma HLS UNROLL

				unsigned int lowBit = simd * InputPrecision;
				unsigned int highBit = (simd + 1) * InputPrecision - 1;
				ap_uint<InputPrecision> temp_data = input(highBit, lowBit);
				Input_t data = *reinterpret_cast<Input_t *>(&temp_data);

				if(use_profiler){
					profiler->update_in(data);
				}

				accu[simd][sf] += (Acc_t)data;

				if(use_profiler){
					profiler->update_acc(accu[simd][sf]);
				}

			}
		}
	}

	loop_sf_out:for(unsigned int sf = 0; sf < synapseFold; sf++)
	{
	#pragma HLS PIPELINE II=1

		loop_simd_out:for(unsigned int simd = 0; simd < SIMDWidth; simd++)
		{
		#pragma HLS UNROLL

			Acc_t div = accu[simd][sf] / (Acc_t)IFMDim;
			Output_t result = (Output_t)div;


			if(use_profiler){
				profiler->update_out(result);
			}

			unsigned int lowBit = simd * OutputPrecision;
			unsigned int highBit = (simd + 1) * OutputPrecision - 1;
			ap_uint<OutputPrecision> output_temp = *reinterpret_cast< ap_uint<OutputPrecision> *>(&result);
			output(highBit, lowBit) = output_temp;

		}

		out.write(output);
	}
}

