#define AP_INT_MAX_W                            4096
#include "qnn-library.h"
#include "config_unroll_1.h"
constexpr unsigned num_mem_inputs = (WINDOW_LENGTH + (DATAWIDTH_IN / INPUT_BITS_PER_VAL) - 1) / (DATAWIDTH_IN / INPUT_BITS_PER_VAL);
constexpr unsigned num_mem_outputs = 1;
constexpr unsigned DATAWIDTH_INPUT_TRANSFORM = (DATAWIDTH_IN / INPUT_BITS_PER_VAL) * CONV1D_DS_0_IA_BITS;


void DoCompute(ap_uint<DATAWIDTH_IN> *in, ap_uint<DATAWIDTH_OUT> *out)
{
	#pragma HLS DATAFLOW

	constexpr unsigned CONV1D_DS_0_DEPTH = CONV1D_DS_0_IFM_CH/CONV1D_DS_0_SIMD + 2;
	constexpr unsigned CONV1D_DS_1_DEPTH = CONV1D_DS_1_IFM_CH/CONV1D_DS_1_SIMD + 2;
	constexpr unsigned CONV1D_DS_2_DEPTH = CONV1D_DS_2_IFM_CH/CONV1D_DS_2_SIMD + 2;
	constexpr unsigned CONV1D_DS_3_DEPTH = CONV1D_DS_3_IFM_CH/CONV1D_DS_3_SIMD + 2;

	hls::stream<ap_uint<DATAWIDTH_IN>>Mem_InputTransform_0("Mem_InputTransform_0");
	#pragma HLS STREAM variable=Mem_InputTransform_0 depth=2
	hls::stream<ap_uint<DATAWIDTH_INPUT_TRANSFORM>>InputTransform_0_StreamConverter_0("InputTransform_0_StreamConverter_0");
	#pragma HLS STREAM variable=InputTransform_0_StreamConverter_0 depth=2
	hls::stream<ap_uint<CONV1D_DS_0_SIMD * CONV1D_DS_0_IA_BITS>>StreamConverter_0_Conv1d_DS_ReLuBlock_0("StreamConverter_0_Conv1d_DS_ReLuBlock_0");
	#pragma HLS STREAM variable=StreamConverter_0_Conv1d_DS_ReLuBlock_0 depth=2
	hls::stream<ap_uint<CONV1D_DS_1_SIMD * CONV1D_DS_1_IA_BITS>>Conv1d_DS_ReLuBlock_0_Conv1d_DS_ReLuBlock_1("Conv1d_DS_ReLuBlock_0_Conv1d_DS_ReLuBlock_1");
	#pragma HLS STREAM variable=Conv1d_DS_ReLuBlock_0_Conv1d_DS_ReLuBlock_1 depth=CONV1D_DS_1_DEPTH
	hls::stream<ap_uint<CONV1D_DS_2_SIMD * CONV1D_DS_2_IA_BITS>>Conv1d_DS_ReLuBlock_1_Conv1d_DS_ReLuBlock_2("Conv1d_DS_ReLuBlock_1_Conv1d_DS_ReLuBlock_2");
	#pragma HLS STREAM variable=Conv1d_DS_ReLuBlock_1_Conv1d_DS_ReLuBlock_2 depth=CONV1D_DS_2_DEPTH
	hls::stream<ap_uint<CONV1D_DS_3_SIMD * CONV1D_DS_3_IA_BITS>>Conv1d_DS_ReLuBlock_2_Conv1d_DS_ReLuBlock_3("Conv1d_DS_ReLuBlock_2_Conv1d_DS_ReLuBlock_3");
	#pragma HLS STREAM variable=Conv1d_DS_ReLuBlock_2_Conv1d_DS_ReLuBlock_3 depth=CONV1D_DS_3_DEPTH
	hls::stream<ap_uint<MAXPOOL1D_0_SIMD * MAXPOOL1D_0_IA_BITS>>Conv1d_DS_ReLuBlock_3_MaxPool1D_0("Conv1d_DS_ReLuBlock_3_MaxPool1D_0");
	#pragma HLS STREAM variable=Conv1d_DS_ReLuBlock_3_MaxPool1D_0 depth=2
	hls::stream<ap_uint<MAXPOOL1D_1_SIMD * MAXPOOL1D_1_IA_BITS>>MaxPool1D_0_MaxPool1D_1("MaxPool1D_0_MaxPool1D_1");
	#pragma HLS STREAM variable=MaxPool1D_0_MaxPool1D_1 depth=2
	hls::stream<ap_uint<GLOBALAVERAGEPOOL1D_0_SIMD * GLOBALAVERAGEPOOL1D_0_IA_BITS>>MaxPool1D_1_GlobalAveragePool1D_0("MaxPool1D_1_GlobalAveragePool1D_0");
	#pragma HLS STREAM variable=MaxPool1D_1_GlobalAveragePool1D_0 depth=2
	hls::stream<ap_uint<FC1D_0_SIMD * FC1D_0_IA_BITS>>GlobalAveragePool1D_0_FC1DMac_0("GlobalAveragePool1D_0_FC1DMac_0");
	#pragma HLS STREAM variable=GlobalAveragePool1D_0_FC1DMac_0 depth=2
	hls::stream<ap_uint<FC1D_0_PE * FC1D_0_OA_BITS>>FC1DMac_0_Mem_0("FC1DMac_0_Mem_0");
	#pragma HLS STREAM variable=FC1DMac_0_Mem_0 depth=2

	constexpr unsigned inBits = CONV1D_DS_0_IFM_DIM * CONV1D_DS_0_IFM_CH * CONV1D_DS_0_IA_BITS;

	Mem2Stream<DATAWIDTH_IN>(in, Mem_InputTransform_0, (num_mem_inputs * DATAWIDTH_IN) / 8);

	InputTransform<INPUT_MASK, INPUT_BITS_PER_VAL, num_mem_inputs, ap_fixed<CONV1D_DS_0_IA_BITS, CONV1D_DS_0_IA_INT_BITS>>(Mem_InputTransform_0, InputTransform_0_StreamConverter_0);
	StreamingDataWidthConverter<(inBits + DATAWIDTH_INPUT_TRANSFORM - 1) / DATAWIDTH_INPUT_TRANSFORM, (CONV1D_DS_0_IFM_DIM * CONV1D_DS_0_IFM_CH * CONV1D_DS_0_IA_BITS) / (CONV1D_DS_0_SIMD * CONV1D_DS_0_IA_BITS)> (InputTransform_0_StreamConverter_0, StreamConverter_0_Conv1d_DS_ReLuBlock_0);


	Conv1DReLuBlockDepthwiseSeparable
	<
	CONV1D_DS_0_K,
	CONV1D_DS_0_IFM_CH,
	CONV1D_DS_0_IFM_DIM,
	CONV1D_DS_0_STRIDE,
	CONV1D_DS_0_PADDING,
	CONV1D_DS_0_OFM_CH,
	CONV1D_DS_0_OFM_DIM,
	CONV1D_DS_0_PE,
	CONV1D_DS_0_SIMD,
	CONV1D_DS_0_BIAS_BITS,
	CONV1D_DS_0_BIAS_INT_BITS,
	CONV1D_DS_0_DEPTHWISE_WEIGHT_BITS,
	CONV1D_DS_0_DEPTHWISE_WEIGHT_INT_BITS,
	CONV1D_DS_0_POINTWISE_WEIGHT_BITS,
	CONV1D_DS_0_POINTWISE_WEIGHT_INT_BITS,
	CONV1D_DS_0_IA_BITS,
	CONV1D_DS_0_IA_INT_BITS,
	CONV1D_DS_0_DEPTHWISE_MUL_BITS,
	CONV1D_DS_0_DEPTHWISE_MUL_INT_BITS,
	CONV1D_DS_0_DEPTHWISE_ACC_BITS,
	CONV1D_DS_0_DEPTHWISE_ACC_INT_BITS,
	CONV1D_DS_0_DEPTHWISE_OA_BITS,
	CONV1D_DS_0_DEPTHWISE_OA_INT_BITS,
	CONV1D_DS_0_POINTWISE_MUL_BITS,
	CONV1D_DS_0_POINTWISE_MUL_INT_BITS,
	CONV1D_DS_0_POINTWISE_ACC_BITS,
	CONV1D_DS_0_POINTWISE_ACC_INT_BITS,
	CONV1D_DS_0_POINTWISE_OA_BITS,
	CONV1D_DS_0_POINTWISE_OA_INT_BITS,
	RELU1D_0_IFM_CH,
	RELU1D_0_IFM_DIM,
	RELU1D_0_OFM_CH,
	RELU1D_0_OFM_DIM,
	RELU1D_0_SIMD,
	RELU1D_0_IA_BITS,
	RELU1D_0_IA_INT_BITS,
	RELU1D_0_OA_BITS,
	RELU1D_0_OA_INT_BITS,
	USE_PROFILER
	>
	(StreamConverter_0_Conv1d_DS_ReLuBlock_0, Conv1d_DS_ReLuBlock_0_Conv1d_DS_ReLuBlock_1, conv1d_depthwise_0_weight, conv1d_pointwise_0_weight, conv1d_ds_0_bias, &profiler_conv1d_ds_0);


	Conv1DReLuBlockDepthwiseSeparable
	<
	CONV1D_DS_1_K,
	CONV1D_DS_1_IFM_CH,
	CONV1D_DS_1_IFM_DIM,
	CONV1D_DS_1_STRIDE,
	CONV1D_DS_1_PADDING,
	CONV1D_DS_1_OFM_CH,
	CONV1D_DS_1_OFM_DIM,
	CONV1D_DS_1_PE,
	CONV1D_DS_1_SIMD,
	CONV1D_DS_1_BIAS_BITS,
	CONV1D_DS_1_BIAS_INT_BITS,
	CONV1D_DS_1_DEPTHWISE_WEIGHT_BITS,
	CONV1D_DS_1_DEPTHWISE_WEIGHT_INT_BITS,
	CONV1D_DS_1_POINTWISE_WEIGHT_BITS,
	CONV1D_DS_1_POINTWISE_WEIGHT_INT_BITS,
	CONV1D_DS_1_IA_BITS,
	CONV1D_DS_1_IA_INT_BITS,
	CONV1D_DS_1_DEPTHWISE_MUL_BITS,
	CONV1D_DS_1_DEPTHWISE_MUL_INT_BITS,
	CONV1D_DS_1_DEPTHWISE_ACC_BITS,
	CONV1D_DS_1_DEPTHWISE_ACC_INT_BITS,
	CONV1D_DS_1_DEPTHWISE_OA_BITS,
	CONV1D_DS_1_DEPTHWISE_OA_INT_BITS,
	CONV1D_DS_1_POINTWISE_MUL_BITS,
	CONV1D_DS_1_POINTWISE_MUL_INT_BITS,
	CONV1D_DS_1_POINTWISE_ACC_BITS,
	CONV1D_DS_1_POINTWISE_ACC_INT_BITS,
	CONV1D_DS_1_POINTWISE_OA_BITS,
	CONV1D_DS_1_POINTWISE_OA_INT_BITS,
	RELU1D_1_IFM_CH,
	RELU1D_1_IFM_DIM,
	RELU1D_1_OFM_CH,
	RELU1D_1_OFM_DIM,
	RELU1D_1_SIMD,
	RELU1D_1_IA_BITS,
	RELU1D_1_IA_INT_BITS,
	RELU1D_1_OA_BITS,
	RELU1D_1_OA_INT_BITS,
	USE_PROFILER
	>
	(Conv1d_DS_ReLuBlock_0_Conv1d_DS_ReLuBlock_1, Conv1d_DS_ReLuBlock_1_Conv1d_DS_ReLuBlock_2, conv1d_depthwise_1_weight, conv1d_pointwise_1_weight, conv1d_ds_1_bias, &profiler_conv1d_ds_1);


	Conv1DReLuBlockDepthwiseSeparable
	<
	CONV1D_DS_2_K,
	CONV1D_DS_2_IFM_CH,
	CONV1D_DS_2_IFM_DIM,
	CONV1D_DS_2_STRIDE,
	CONV1D_DS_2_PADDING,
	CONV1D_DS_2_OFM_CH,
	CONV1D_DS_2_OFM_DIM,
	CONV1D_DS_2_PE,
	CONV1D_DS_2_SIMD,
	CONV1D_DS_2_BIAS_BITS,
	CONV1D_DS_2_BIAS_INT_BITS,
	CONV1D_DS_2_DEPTHWISE_WEIGHT_BITS,
	CONV1D_DS_2_DEPTHWISE_WEIGHT_INT_BITS,
	CONV1D_DS_2_POINTWISE_WEIGHT_BITS,
	CONV1D_DS_2_POINTWISE_WEIGHT_INT_BITS,
	CONV1D_DS_2_IA_BITS,
	CONV1D_DS_2_IA_INT_BITS,
	CONV1D_DS_2_DEPTHWISE_MUL_BITS,
	CONV1D_DS_2_DEPTHWISE_MUL_INT_BITS,
	CONV1D_DS_2_DEPTHWISE_ACC_BITS,
	CONV1D_DS_2_DEPTHWISE_ACC_INT_BITS,
	CONV1D_DS_2_DEPTHWISE_OA_BITS,
	CONV1D_DS_2_DEPTHWISE_OA_INT_BITS,
	CONV1D_DS_2_POINTWISE_MUL_BITS,
	CONV1D_DS_2_POINTWISE_MUL_INT_BITS,
	CONV1D_DS_2_POINTWISE_ACC_BITS,
	CONV1D_DS_2_POINTWISE_ACC_INT_BITS,
	CONV1D_DS_2_POINTWISE_OA_BITS,
	CONV1D_DS_2_POINTWISE_OA_INT_BITS,
	RELU1D_2_IFM_CH,
	RELU1D_2_IFM_DIM,
	RELU1D_2_OFM_CH,
	RELU1D_2_OFM_DIM,
	RELU1D_2_SIMD,
	RELU1D_2_IA_BITS,
	RELU1D_2_IA_INT_BITS,
	RELU1D_2_OA_BITS,
	RELU1D_2_OA_INT_BITS,
	USE_PROFILER
	>
	(Conv1d_DS_ReLuBlock_1_Conv1d_DS_ReLuBlock_2, Conv1d_DS_ReLuBlock_2_Conv1d_DS_ReLuBlock_3, conv1d_depthwise_2_weight, conv1d_pointwise_2_weight, conv1d_ds_2_bias, &profiler_conv1d_ds_2);


	Conv1DReLuBlockDepthwiseSeparable
	<
	CONV1D_DS_3_K,
	CONV1D_DS_3_IFM_CH,
	CONV1D_DS_3_IFM_DIM,
	CONV1D_DS_3_STRIDE,
	CONV1D_DS_3_PADDING,
	CONV1D_DS_3_OFM_CH,
	CONV1D_DS_3_OFM_DIM,
	CONV1D_DS_3_PE,
	CONV1D_DS_3_SIMD,
	CONV1D_DS_3_BIAS_BITS,
	CONV1D_DS_3_BIAS_INT_BITS,
	CONV1D_DS_3_DEPTHWISE_WEIGHT_BITS,
	CONV1D_DS_3_DEPTHWISE_WEIGHT_INT_BITS,
	CONV1D_DS_3_POINTWISE_WEIGHT_BITS,
	CONV1D_DS_3_POINTWISE_WEIGHT_INT_BITS,
	CONV1D_DS_3_IA_BITS,
	CONV1D_DS_3_IA_INT_BITS,
	CONV1D_DS_3_DEPTHWISE_MUL_BITS,
	CONV1D_DS_3_DEPTHWISE_MUL_INT_BITS,
	CONV1D_DS_3_DEPTHWISE_ACC_BITS,
	CONV1D_DS_3_DEPTHWISE_ACC_INT_BITS,
	CONV1D_DS_3_DEPTHWISE_OA_BITS,
	CONV1D_DS_3_DEPTHWISE_OA_INT_BITS,
	CONV1D_DS_3_POINTWISE_MUL_BITS,
	CONV1D_DS_3_POINTWISE_MUL_INT_BITS,
	CONV1D_DS_3_POINTWISE_ACC_BITS,
	CONV1D_DS_3_POINTWISE_ACC_INT_BITS,
	CONV1D_DS_3_POINTWISE_OA_BITS,
	CONV1D_DS_3_POINTWISE_OA_INT_BITS,
	RELU1D_3_IFM_CH,
	RELU1D_3_IFM_DIM,
	RELU1D_3_OFM_CH,
	RELU1D_3_OFM_DIM,
	RELU1D_3_SIMD,
	RELU1D_3_IA_BITS,
	RELU1D_3_IA_INT_BITS,
	RELU1D_3_OA_BITS,
	RELU1D_3_OA_INT_BITS,
	USE_PROFILER
	>
	(Conv1d_DS_ReLuBlock_2_Conv1d_DS_ReLuBlock_3, Conv1d_DS_ReLuBlock_3_MaxPool1D_0, conv1d_depthwise_3_weight, conv1d_pointwise_3_weight, conv1d_ds_3_bias, &profiler_conv1d_ds_3);


	MAXPool1DK_Stride
	<
	MAXPOOL1D_0_K,
	MAXPOOL1D_0_IFM_CH,
	MAXPOOL1D_0_IFM_DIM,
	MAXPOOL1D_0_STRIDE,
	MAXPOOL1D_0_OFM_DIM,
	MAXPOOL1D_0_SIMD,
	MAXPOOL1D_0_IA_BITS,
	MAXPOOL1D_0_IA_INT_BITS,
	MAXPOOL1D_0_OA_BITS,
	MAXPOOL1D_0_OA_INT_BITS
	>
	(Conv1d_DS_ReLuBlock_3_MaxPool1D_0, MaxPool1D_0_MaxPool1D_1);


	MAXPool1DK_Stride
	<
	MAXPOOL1D_1_K,
	MAXPOOL1D_1_IFM_CH,
	MAXPOOL1D_1_IFM_DIM,
	MAXPOOL1D_1_STRIDE,
	MAXPOOL1D_1_OFM_DIM,
	MAXPOOL1D_1_SIMD,
	MAXPOOL1D_1_IA_BITS,
	MAXPOOL1D_1_IA_INT_BITS,
	MAXPOOL1D_1_OA_BITS,
	MAXPOOL1D_1_OA_INT_BITS
	>
	(MaxPool1D_0_MaxPool1D_1, MaxPool1D_1_GlobalAveragePool1D_0);


	GlobalAveragePool1D
	<
	GLOBALAVERAGEPOOL1D_0_IFM_CH,
	GLOBALAVERAGEPOOL1D_0_IFM_DIM,
	GLOBALAVERAGEPOOL1D_0_OFM_CH,
	GLOBALAVERAGEPOOL1D_0_OFM_DIM,
	GLOBALAVERAGEPOOL1D_0_SIMD,
	GLOBALAVERAGEPOOL1D_0_IA_BITS,
	GLOBALAVERAGEPOOL1D_0_IA_INT_BITS,
	GLOBALAVERAGEPOOL1D_0_ACC_BITS,
	GLOBALAVERAGEPOOL1D_0_ACC_INT_BITS,
	GLOBALAVERAGEPOOL1D_0_OA_BITS,
	GLOBALAVERAGEPOOL1D_0_OA_INT_BITS,
	USE_PROFILER
	>
	(MaxPool1D_1_GlobalAveragePool1D_0, GlobalAveragePool1D_0_FC1DMac_0, &profiler_globalaveragepool1d_0);


	FC1DMac
	<
	FC1D_0_INPUTS,
	FC1D_0_NEURONS,
	FC1D_0_PE,
	FC1D_0_SIMD,
	FC1D_0_WEIGHT_BITS,
	FC1D_0_WEIGHT_INT_BITS,
	FC1D_0_IA_BITS,
	FC1D_0_IA_INT_BITS,
	FC1D_0_MUL_BITS,
	FC1D_0_MUL_INT_BITS,
	FC1D_0_ACC_BITS,
	FC1D_0_ACC_INT_BITS,
	FC1D_0_OA_BITS,
	FC1D_0_OA_INT_BITS,
	USE_PROFILER
	>
	(GlobalAveragePool1D_0_FC1DMac_0, FC1DMac_0_Mem_0, fc1d_0_weight, &profiler_fc1d_0);


	Stream2Mem1D
	<
	FC1D_0_PE,
	FC1D_0_OA_BITS,
	FC1D_0_OA_INT_BITS,
	DATAWIDTH_OUT
	>
	(FC1DMac_0_Mem_0, out);


}

void TopLevel(ap_uint<DATAWIDTH_IN>* in, ap_uint<DATAWIDTH_OUT>* out)
{

	#pragma HLS INTERFACE s_axilite port=return bundle=control

	#pragma HLS INTERFACE m_axi offset=slave port=in bundle=hostmem_in depth=num_mem_inputs
	#pragma HLS INTERFACE s_axilite port=in bundle=control
	#pragma HLS INTERFACE m_axi offset=slave port=out bundle=hostmem_out depth=num_mem_outputs
	#pragma HLS INTERFACE s_axilite port=out bundle=control

	#pragma HLS ARRAY_RESHAPE variable=conv1d_depthwise_0_weight complete dim=1
	#pragma HLS ARRAY_RESHAPE variable=conv1d_pointwise_0_weight complete dim=1
	#pragma HLS ARRAY_RESHAPE variable=conv1d_pointwise_0_weight complete dim=2
	#pragma HLS ARRAY_RESHAPE variable=conv1d_depthwise_1_weight complete dim=1
	#pragma HLS ARRAY_RESHAPE variable=conv1d_pointwise_1_weight complete dim=1
	#pragma HLS ARRAY_RESHAPE variable=conv1d_pointwise_1_weight complete dim=2
	#pragma HLS ARRAY_RESHAPE variable=conv1d_depthwise_2_weight complete dim=1
	#pragma HLS ARRAY_RESHAPE variable=conv1d_pointwise_2_weight complete dim=1
	#pragma HLS ARRAY_RESHAPE variable=conv1d_pointwise_2_weight complete dim=2
	#pragma HLS ARRAY_RESHAPE variable=conv1d_depthwise_3_weight complete dim=1
	#pragma HLS ARRAY_RESHAPE variable=conv1d_pointwise_3_weight complete dim=1
	#pragma HLS ARRAY_RESHAPE variable=conv1d_pointwise_3_weight complete dim=2
	#pragma HLS ARRAY_RESHAPE variable=fc1d_0_weight complete dim=1
	#pragma HLS ARRAY_RESHAPE variable=fc1d_0_weight complete dim=2

	DoCompute(in, out);
}

