#pragma once


template<
        // convolution parameters
        short unsigned int Inputs,                    // number of inputs: dim * ch
        short unsigned int Neurons,                // number of units
        // parallelization parameters
        short unsigned int PECount,                 // number of PEs
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
void FC1DMac(hls::stream <ap_uint<SIMDWidth * InputPrecision>> &in,
           hls::stream <ap_uint<PECount * OutputPrecision>> &out,
           const ap_uint <WeightsPrecision> weightMem[PECount][SIMDWidth][Inputs * Neurons / (SIMDWidth * PECount)],
           Profiler_MAC *profiler = nullptr) {
    CASSERT_DATAFLOW(Inputs % SIMDWidth == 0);
    CASSERT_DATAFLOW(Neurons % PECount == 0);

    typedef ap_fixed <InputPrecision, InputIntPrecision, AP_RND_ZERO, AP_WRAP> Input_t;
    typedef ap_fixed <WeightsPrecision, WeightsIntPrecision, AP_RND_ZERO, AP_WRAP> Weights_t;
    typedef ap_fixed <MulPrecision, MulIntPrecision, AP_RND_ZERO, AP_WRAP> Mul_t;
    typedef ap_fixed <AccPrecision, AccIntPrecision, AP_RND_ZERO, AP_WRAP> Acc_t;
    typedef ap_fixed <OutputPrecision, OutputIntPrecision, AP_RND_ZERO, AP_WRAP> Output_t;

    const unsigned int neuronFold = Neurons / PECount;
    const unsigned int synapseFold = Inputs / SIMDWidth;

    ap_uint < SIMDWidth * InputPrecision > input;
    ap_uint < PECount * OutputPrecision > output;

    Acc_t macRegisters[PECount][neuronFold] = {0};
	#pragma HLS ARRAY_PARTITION variable=macRegisters complete dim=1

    loop_dim:
    for (unsigned int sy = 0; sy < synapseFold; sy++) {

        input = in.read();

        loop_ne:
        for (unsigned int ne = 0; ne < neuronFold; ne++) {
			#pragma HLS PIPELINE II=1

            loop_pe:
            for (unsigned int pe = 0; pe < PECount; pe++) {
                Acc_t tmpMac = macRegisters[pe][ne];

                loop_simd:
                for (unsigned int simd = 0; simd < SIMDWidth; simd++) {
                    Mul_t mul;

                    unsigned int lowBit = simd * InputPrecision;
                    unsigned int highBit = (simd + 1) * InputPrecision - 1;
                    ap_int <InputPrecision> temp_data = input(highBit, lowBit);
                    Input_t data = *reinterpret_cast<Input_t *>(&temp_data);

                    if (use_profiler) {
                        profiler->update_in(data);
                    }

                    ap_int <WeightsPrecision> temp_weight = weightMem[pe][simd][ne * synapseFold + sy];
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

                macRegisters[pe][ne] = tmpMac;
            }
        }
    }

    loop_output_ne:
    for (unsigned int ne = 0; ne < neuronFold; ne++) {
		#pragma HLS PIPELINE II=1

        loop_output_pe:
        for (unsigned int pe = 0; pe < PECount; pe++) {

            Output_t temp_reg = (Output_t) macRegisters[pe][ne];

            if (use_profiler) {
                profiler->update_out(temp_reg);
            }

            ap_uint <OutputPrecision> temp_output = *reinterpret_cast< ap_uint <OutputPrecision> *>(&temp_reg);

            unsigned int lowBit = pe * OutputPrecision;
            unsigned int highBit = (pe + 1) * OutputPrecision - 1;
            output(highBit, lowBit) = temp_output;
        }

        out.write(output);
    }

}


template<
        // convolution parameters
        short unsigned int Inputs,                    // number of inputs: dim * ch
        short unsigned int Neurons,                // number of units
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
void FC1DMac(hls::stream <ap_uint<SIMDWidth * InputPrecision>> &in,
           hls::stream <ap_uint<PECount * OutputPrecision>> &out,
           const ap_uint <WeightsPrecision> weightMem[PECount][SIMDWidth][Inputs * Neurons / (SIMDWidth * PECount)],
           const ap_uint <BiasPrecision> biasMem[PECount][Neurons / PECount],
           Profiler_MAC *profiler = nullptr) {
    CASSERT_DATAFLOW(Inputs % SIMDWidth == 0);
    CASSERT_DATAFLOW(Neurons % PECount == 0);

    typedef ap_fixed <InputPrecision, InputIntPrecision, AP_RND_ZERO, AP_WRAP> Input_t;
    typedef ap_fixed <BiasPrecision, BiasIntPrecision, AP_RND_ZERO, AP_WRAP> Bias_t;
    typedef ap_fixed <WeightsPrecision, WeightsIntPrecision, AP_RND_ZERO, AP_WRAP> Weights_t;
    typedef ap_fixed <MulPrecision, MulIntPrecision, AP_RND_ZERO, AP_WRAP> Mul_t;
    typedef ap_fixed <AccPrecision, AccIntPrecision, AP_RND_ZERO, AP_WRAP> Acc_t;
    typedef ap_fixed <OutputPrecision, OutputIntPrecision, AP_RND_ZERO, AP_WRAP> Output_t;

    const unsigned int neuronFold = Neurons / PECount;
    const unsigned int synapseFold = Inputs / SIMDWidth;

    ap_uint < SIMDWidth * InputPrecision > input;
    ap_uint < PECount * OutputPrecision > output;

    Acc_t macRegisters[PECount][neuronFold];
#pragma HLS ARRAY_PARTITION variable=macRegisters complete dim=1

    loop_init_ne:
    for (unsigned int ne = 0; ne < neuronFold; ne++) {

        loop_init_pe:
        for (unsigned int pe = 0; pe < PECount; pe++) {

            ap_int <BiasPrecision> temp_bias = biasMem[pe][ne];
            Bias_t bias = *reinterpret_cast<Bias_t *>(&temp_bias);

            macRegisters[pe][ne] = (Acc_t) bias;
        }
    }

    loop_dim:
    for (unsigned int sy = 0; sy < synapseFold; sy++) {
        input = in.read();

        loop_ne:
        for (unsigned int ne = 0; ne < neuronFold; ne++) {
#pragma HLS PIPELINE II=1

            loop_pe:
            for (unsigned int pe = 0; pe < PECount; pe++) {
                Acc_t tmpMac = macRegisters[pe][ne];

                loop_simd:
                for (unsigned int simd = 0; simd < SIMDWidth; simd++) {
                    Mul_t mul;

                    unsigned int lowBit = simd * InputPrecision;
                    unsigned int highBit = (simd + 1) * InputPrecision - 1;
                    ap_int <InputPrecision> temp_data = input(highBit, lowBit);
                    Input_t data = *reinterpret_cast<Input_t *>(&temp_data);

                    if (use_profiler) {
                        profiler->update_in(data);
                    }

                    ap_int <WeightsPrecision> temp_weight = weightMem[pe][simd][ne * synapseFold + sy];
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

                macRegisters[pe][ne] = tmpMac;
            }
        }
    }

    loop_output_ne:
    for (unsigned int ne = 0; ne < neuronFold; ne++) {
		#pragma HLS PIPELINE II=1

        loop_output_pe:
        for (unsigned int pe = 0; pe < PECount; pe++) {

            Output_t temp_reg = (Output_t) macRegisters[pe][ne];

            if (use_profiler) {
                profiler->update_out(temp_reg);
            }

            ap_uint <OutputPrecision> temp_output = *reinterpret_cast< ap_uint <OutputPrecision> *>(&temp_reg);

            unsigned int lowBit = pe * OutputPrecision;
            unsigned int highBit = (pe + 1) * OutputPrecision - 1;
            output(highBit, lowBit) = temp_output;
        }

        out.write(output);
    }

}

template<
        //Fc
        short unsigned int InputsFc,
        short unsigned int NeuronsFc,
        short unsigned int PECountFc,
        short unsigned int SIMDWidthFc,
        short unsigned int BiasPrecisionFc,
        short unsigned int BiasIntPrecisionFc,
        short unsigned int WeightsPrecisionFc,
        short unsigned int WeightsIntPrecisionFc,
        short unsigned int InputPrecisionFc,
        short unsigned int InputIntPrecisionFc,
        short unsigned int MulPrecisionFc,
        short unsigned int MulIntPrecisionFc,
        short unsigned int AccPrecisionFc,
        short unsigned int AccIntPrecisionFc,
        short unsigned int OutputPrecisionFc,
        short unsigned int OutputIntPrecisionFc,
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
void FC1DReLuBlock(hls::stream <ap_uint<SIMDWidthFc * InputPrecisionFc>> &in,
                 hls::stream <ap_uint<SIMDWidthRelu * OutputPrecisionRelu>> &out,
                 const ap_uint <WeightsPrecisionFc> weightMem[PECountFc][SIMDWidthFc][InputsFc * NeuronsFc /
                                                                                      (SIMDWidthFc * PECountFc)],
                 const ap_uint <BiasPrecisionFc> biasMem[PECountFc][NeuronsFc / PECountFc],
                 Profiler_MAC *profiler = nullptr) {
	#pragma HLS DATAFLOW

    hls::stream <ap_uint<SIMDWidthRelu * InputPrecisionRelu>> FcMac_Relu("FcMac_Relu");

	#pragma HLS STREAM variable=FcMac_Relu depth=2

    FC1DMac
            <
                    InputsFc,
                    NeuronsFc,
                    PECountFc,
                    SIMDWidthFc,
                    BiasPrecisionFc,
                    BiasIntPrecisionFc,
                    WeightsPrecisionFc,
                    WeightsIntPrecisionFc,
                    InputPrecisionFc,
                    InputIntPrecisionFc,
                    MulPrecisionFc,
                    MulIntPrecisionFc,
                    AccPrecisionFc,
                    AccIntPrecisionFc,
                    OutputPrecisionFc,
                    OutputIntPrecisionFc,
                    use_profiler
            >
            (in, FcMac_Relu, weightMem, biasMem, profiler);

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
            (FcMac_Relu, out);
}
