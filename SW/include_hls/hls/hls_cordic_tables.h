/*****************************************************************************
 *
 *     Author: Xilinx, Inc.
 *
 *     This text contains proprietary, confidential information of
 *     Xilinx, Inc. , is distributed by under license from Xilinx,
 *     Inc., and may be used, copied and/or disclosed only pursuant to
 *     the terms of a valid license agreement with Xilinx, Inc.
 *
 *     XILINX IS PROVIDING THIS DESIGN, CODE, OR INFORMATION "AS IS"
 *     AS A COURTESY TO YOU, SOLELY FOR USE IN DEVELOPING PROGRAMS AND
 *     SOLUTIONS FOR XILINX DEVICES.  BY PROVIDING THIS DESIGN, CODE,
 *     OR INFORMATION AS ONE POSSIBLE IMPLEMENTATION OF THIS FEATURE,
 *     APPLICATION OR STANDARD, XILINX IS MAKING NO REPRESENTATION
 *     THAT THIS IMPLEMENTATION IS FREE FROM ANY CLAIMS OF INFRINGEMENT,
 *     AND YOU ARE RESPONSIBLE FOR OBTAINING ANY RIGHTS YOU MAY REQUIRE
 *     FOR YOUR IMPLEMENTATION.  XILINX EXPRESSLY DISCLAIMS ANY
 *     WARRANTY WHATSOEVER WITH RESPECT TO THE ADEQUACY OF THE
 *     IMPLEMENTATION, INCLUDING BUT NOT LIMITED TO ANY WARRANTIES OR
 *     REPRESENTATIONS THAT THIS IMPLEMENTATION IS FREE FROM CLAIMS OF
 *     INFRINGEMENT, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *     FOR A PARTICULAR PURPOSE.
 *
 *     Xilinx products are not intended for use in life support appliances,
 *     devices, or systems. Use in such applications is expressly prohibited.
 *
 *     (c) Copyright 2012 Xilinx Inc.
 *     All rights reserved.
 *
 *****************************************************************************/

#ifndef X_HLS_CORDIC_TABLES_H
#define X_HLS_CORDIC_TABLES_H

#include "../ap_int.h"

const ap_uint<128> cordic_ctab_table_int_128[128] = {
    "0x3243F6A8885A308D313198A2E0370734",
    "0x1DAC670561BB4F68ADFC88BD978751A0",
    "0xFADBAFC96406EB156DC79EF5F7A217E",
    "0x7F56EA6AB0BDB719644BCC4F9F44477",
    "0x3FEAB76E59FBD38DB2C9E4B7038B835",
    "0x1FFD55BBA97624A84EF3AEEDBB518C4",
    "0xFFFAAADDDB94D5BBE78C564015F760",
    "0x7FFF5556EEEA5CB40311A8FDDF3057",
    "0x3FFFEAAAB7776E52EC4ABEDADB53DF",
    "0x1FFFFD5555BBBBA9729AB7AAC08947",
    "0xFFFFFAAAAADDDDDB94B968067EF3A",
    "0x7FFFFF555556EEEEEA5CA5D895892",
    "0x3FFFFFEAAAAAB777776E52E5356F5",
    "0x1FFFFFFD555555BBBBBBA972972D0",
    "0xFFFFFFFAAAAAAADDDDDDDB94B94B",
    "0x7FFFFFFF55555556EEEEEEEA5CA5",
    "0x3FFFFFFFEAAAAAAAB77777776E52",
    "0x1FFFFFFFFD55555555BBBBBBBBA9",
    "0xFFFFFFFFFAAAAAAAAADDDDDDDDD",
    "0x7FFFFFFFFF5555555556EEEEEEE",
    "0x3FFFFFFFFFEAAAAAAAAAB777777",
    "0x1FFFFFFFFFFD5555555555BBBBB",
    "0xFFFFFFFFFFFAAAAAAAAAAADDDD",
    "0x7FFFFFFFFFFF555555555556EE",
    "0x3FFFFFFFFFFFEAAAAAAAAAAAB7",
    "0x1FFFFFFFFFFFFD555555555555",
    "0xFFFFFFFFFFFFFAAAAAAAAAAAA",
    "0x7FFFFFFFFFFFFF55555555555",
    "0x3FFFFFFFFFFFFFEAAAAAAAAAA",
    "0x1FFFFFFFFFFFFFFD555555555",
    "0xFFFFFFFFFFFFFFFAAAAAAAAA",
    "0x7FFFFFFFFFFFFFFF55555555",
    "0x3FFFFFFFFFFFFFFFEAAAAAAA",
    "0x1FFFFFFFFFFFFFFFFD555555",
    "0xFFFFFFFFFFFFFFFFFAAAAAA",
    "0x7FFFFFFFFFFFFFFFFF55555",
    "0x3FFFFFFFFFFFFFFFFFEAAAA",
    "0x1FFFFFFFFFFFFFFFFFFD555",
    "0xFFFFFFFFFFFFFFFFFFFAAA",
    "0x7FFFFFFFFFFFFFFFFFFF55",
    "0x3FFFFFFFFFFFFFFFFFFFEA",
    "0x1FFFFFFFFFFFFFFFFFFFFD",
    "0xFFFFFFFFFFFFFFFFFFFFF",
    "0x7FFFFFFFFFFFFFFFFFFFF",
    "0x3FFFFFFFFFFFFFFFFFFFF",
    "0x1FFFFFFFFFFFFFFFFFFFF",
    "0xFFFFFFFFFFFFFFFFFFFF",
    "0x7FFFFFFFFFFFFFFFFFFF",
    "0x3FFFFFFFFFFFFFFFFFFF",
    "0x1FFFFFFFFFFFFFFFFFFF",
    "0xFFFFFFFFFFFFFFFFFFF",
    "0x7FFFFFFFFFFFFFFFFFF",
    "0x3FFFFFFFFFFFFFFFFFF",
    "0x1FFFFFFFFFFFFFFFFFF",
    "0xFFFFFFFFFFFFFFFFFF",
    "0x7FFFFFFFFFFFFFFFFF",
    "0x3FFFFFFFFFFFFFFFFF",
    "0x1FFFFFFFFFFFFFFFFF",
    "0xFFFFFFFFFFFFFFFFF",
    "0x7FFFFFFFFFFFFFFFF",
    "0x3FFFFFFFFFFFFFFFF",
    "0x1FFFFFFFFFFFFFFFF",
    "0xFFFFFFFFFFFFFFFF",
    "0x7FFFFFFFFFFFFFFF",
    "0x3FFFFFFFFFFFFFFF",
    "0x1FFFFFFFFFFFFFFF",
    "0xFFFFFFFFFFFFFFF",
    "0x7FFFFFFFFFFFFFF",
    "0x3FFFFFFFFFFFFFF",
    "0x1FFFFFFFFFFFFFF",
    "0xFFFFFFFFFFFFFF",
    "0x7FFFFFFFFFFFFF",
    "0x3FFFFFFFFFFFFF",
    "0x1FFFFFFFFFFFFF",
    "0xFFFFFFFFFFFFF",
    "0x7FFFFFFFFFFFF",
    "0x3FFFFFFFFFFFF",
    "0x1FFFFFFFFFFFF",
    "0xFFFFFFFFFFFF",
    "0x7FFFFFFFFFFF",
    "0x3FFFFFFFFFFF",
    "0x1FFFFFFFFFFF",
    "0xFFFFFFFFFFF",
    "0x7FFFFFFFFFF",
    "0x3FFFFFFFFFF",
    "0x1FFFFFFFFFF",
    "0xFFFFFFFFFF",
    "0x7FFFFFFFFF",
    "0x3FFFFFFFFF",
    "0x1FFFFFFFFF",
    "0xFFFFFFFFF",
    "0x7FFFFFFFF",
    "0x3FFFFFFFF",
    "0x1FFFFFFFF",
    "0xFFFFFFFF",
    "0x7FFFFFFF",
    "0x3FFFFFFF",
    "0x1FFFFFFF",
    "0xFFFFFFF",
    "0x7FFFFFF",
    "0x3FFFFFF",
    "0x1FFFFFF",
    "0xFFFFFF",
    "0x7FFFFF",
    "0x3FFFFF",
    "0x1FFFFF",
    "0xFFFFF",
    "0x7FFFF",
    "0x3FFFF",
    "0x1FFFF",
    "0xFFFF",
    "0x7FFF",
    "0x3FFF",
    "0x1FFF",
    "0xFFF",
    "0x7FF",
    "0x3FF",
    "0x1FF",
    "0xFF",
    "0x7F",
    "0x3F",
    "0x1F",
    "0xF",
    "0x7",
    "0x3",
    "0x1",
    "0x0",
    "0x0",
};

const ap_ufixed<128,2> cordic_ctab_table_128[128] = {
    "0x0.C90FDAA22168C234C4C6628B80DC1CD0",
    "0x0.76B19C1586ED3DA2B7F222F65E1D4680",
    "0x0.3EB6EBF25901BAC55B71E7BD7DE885F8",
    "0x0.1FD5BA9AAC2F6DC65912F313E7D111DC",
    "0x0.0FFAADDB967EF4E36CB2792DC0E2E0D4",
    "0x0.07FF556EEA5D892A13BCEBBB6ED46310",
    "0x0.03FFEAAB776E5356EF9E31590057DD80",
    "0x0.01FFFD555BBBA972D00C46A3F77CC15C",
    "0x0.00FFFFAAAADDDDB94BB12AFB6B6D4F7C",
    "0x0.007FFFF55556EEEEA5CA6ADEAB02251C",
    "0x0.003FFFFEAAAAB77776E52E5A019FBCE8",
    "0x0.001FFFFFD55555BBBBBA972976256248",
    "0x0.000FFFFFFAAAAAADDDDDDB94B94D5BD4",
    "0x0.0007FFFFFF5555556EEEEEEA5CA5CB40",
    "0x0.0003FFFFFFEAAAAAAB7777776E52E52C",
    "0x0.0001FFFFFFFD5555555BBBBBBBA97294",
    "0x0.0000FFFFFFFFAAAAAAAADDDDDDDDB948",
    "0x0.00007FFFFFFFF555555556EEEEEEEEA4",
    "0x0.00003FFFFFFFFEAAAAAAAAB777777774",
    "0x0.00001FFFFFFFFFD555555555BBBBBBB8",
    "0x0.00000FFFFFFFFFFAAAAAAAAAADDDDDDC",
    "0x0.000007FFFFFFFFFF55555555556EEEEC",
    "0x0.000003FFFFFFFFFFEAAAAAAAAAAB7774",
    "0x0.000001FFFFFFFFFFFD55555555555BB8",
    "0x0.000000FFFFFFFFFFFFAAAAAAAAAAAADC",
    "0x0.0000007FFFFFFFFFFFF5555555555554",
    "0x0.0000003FFFFFFFFFFFFEAAAAAAAAAAA8",
    "0x0.0000001FFFFFFFFFFFFFD55555555554",
    "0x0.0000000FFFFFFFFFFFFFFAAAAAAAAAA8",
    "0x0.00000007FFFFFFFFFFFFFF5555555554",
    "0x0.00000003FFFFFFFFFFFFFFEAAAAAAAA8",
    "0x0.00000001FFFFFFFFFFFFFFFD55555554",
    "0x0.00000000FFFFFFFFFFFFFFFFAAAAAAA8",
    "0x0.000000007FFFFFFFFFFFFFFFF5555554",
    "0x0.000000003FFFFFFFFFFFFFFFFEAAAAA8",
    "0x0.000000001FFFFFFFFFFFFFFFFFD55554",
    "0x0.000000000FFFFFFFFFFFFFFFFFFAAAA8",
    "0x0.0000000007FFFFFFFFFFFFFFFFFF5554",
    "0x0.0000000003FFFFFFFFFFFFFFFFFFEAA8",
    "0x0.0000000001FFFFFFFFFFFFFFFFFFFD54",
    "0x0.0000000000FFFFFFFFFFFFFFFFFFFFA8",
    "0x0.00000000007FFFFFFFFFFFFFFFFFFFF4",
    "0x0.00000000003FFFFFFFFFFFFFFFFFFFFC",
    "0x0.00000000001FFFFFFFFFFFFFFFFFFFFC",
    "0x0.00000000000FFFFFFFFFFFFFFFFFFFFC",
    "0x0.000000000007FFFFFFFFFFFFFFFFFFFC",
    "0x0.000000000003FFFFFFFFFFFFFFFFFFFC",
    "0x0.000000000001FFFFFFFFFFFFFFFFFFFC",
    "0x0.000000000000FFFFFFFFFFFFFFFFFFFC",
    "0x0.0000000000007FFFFFFFFFFFFFFFFFFC",
    "0x0.0000000000003FFFFFFFFFFFFFFFFFFC",
    "0x0.0000000000001FFFFFFFFFFFFFFFFFFC",
    "0x0.0000000000000FFFFFFFFFFFFFFFFFFC",
    "0x0.00000000000007FFFFFFFFFFFFFFFFFC",
    "0x0.00000000000003FFFFFFFFFFFFFFFFFC",
    "0x0.00000000000001FFFFFFFFFFFFFFFFFC",
    "0x0.00000000000000FFFFFFFFFFFFFFFFFC",
    "0x0.000000000000007FFFFFFFFFFFFFFFFC",
    "0x0.000000000000003FFFFFFFFFFFFFFFFC",
    "0x0.000000000000001FFFFFFFFFFFFFFFFC",
    "0x0.000000000000000FFFFFFFFFFFFFFFFC",
    "0x0.0000000000000007FFFFFFFFFFFFFFFC",
    "0x0.0000000000000003FFFFFFFFFFFFFFFC",
    "0x0.0000000000000001FFFFFFFFFFFFFFFC",
    "0x0.0000000000000000FFFFFFFFFFFFFFFC",
    "0x0.00000000000000007FFFFFFFFFFFFFFC",
    "0x0.00000000000000003FFFFFFFFFFFFFFC",
    "0x0.00000000000000001FFFFFFFFFFFFFFC",
    "0x0.00000000000000000FFFFFFFFFFFFFFC",
    "0x0.000000000000000007FFFFFFFFFFFFFC",
    "0x0.000000000000000003FFFFFFFFFFFFFC",
    "0x0.000000000000000001FFFFFFFFFFFFFC",
    "0x0.000000000000000000FFFFFFFFFFFFFC",
    "0x0.0000000000000000007FFFFFFFFFFFFC",
    "0x0.0000000000000000003FFFFFFFFFFFFC",
    "0x0.0000000000000000001FFFFFFFFFFFFC",
    "0x0.0000000000000000000FFFFFFFFFFFFC",
    "0x0.00000000000000000007FFFFFFFFFFFC",
    "0x0.00000000000000000003FFFFFFFFFFFC",
    "0x0.00000000000000000001FFFFFFFFFFFC",
    "0x0.00000000000000000000FFFFFFFFFFFC",
    "0x0.000000000000000000007FFFFFFFFFFC",
    "0x0.000000000000000000003FFFFFFFFFFC",
    "0x0.000000000000000000001FFFFFFFFFFC",
    "0x0.000000000000000000000FFFFFFFFFFC",
    "0x0.0000000000000000000007FFFFFFFFFC",
    "0x0.0000000000000000000003FFFFFFFFFC",
    "0x0.0000000000000000000001FFFFFFFFFC",
    "0x0.0000000000000000000000FFFFFFFFFC",
    "0x0.00000000000000000000007FFFFFFFFC",
    "0x0.00000000000000000000003FFFFFFFFC",
    "0x0.00000000000000000000001FFFFFFFFC",
    "0x0.00000000000000000000000FFFFFFFFC",
    "0x0.000000000000000000000007FFFFFFFC",
    "0x0.000000000000000000000003FFFFFFFC",
    "0x0.000000000000000000000001FFFFFFFC",
    "0x0.000000000000000000000000FFFFFFFC",
    "0x0.0000000000000000000000007FFFFFFC",
    "0x0.0000000000000000000000003FFFFFFC",
    "0x0.0000000000000000000000001FFFFFFC",
    "0x0.0000000000000000000000000FFFFFFC",
    "0x0.00000000000000000000000007FFFFFC",
    "0x0.00000000000000000000000003FFFFFC",
    "0x0.00000000000000000000000001FFFFFC",
    "0x0.00000000000000000000000000FFFFFC",
    "0x0.000000000000000000000000007FFFFC",
    "0x0.000000000000000000000000003FFFFC",
    "0x0.000000000000000000000000001FFFFC",
    "0x0.000000000000000000000000000FFFFC",
    "0x0.0000000000000000000000000007FFFC",
    "0x0.0000000000000000000000000003FFFC",
    "0x0.0000000000000000000000000001FFFC",
    "0x0.0000000000000000000000000000FFFC",
    "0x0.00000000000000000000000000007FFC",
    "0x0.00000000000000000000000000003FFC",
    "0x0.00000000000000000000000000001FFC",
    "0x0.00000000000000000000000000000FFC",
    "0x0.000000000000000000000000000007FC",
    "0x0.000000000000000000000000000003FC",
    "0x0.000000000000000000000000000001FC",
    "0x0.000000000000000000000000000000FC",
    "0x0.0000000000000000000000000000007C",
    "0x0.0000000000000000000000000000003C",
    "0x0.0000000000000000000000000000001C",
    "0x0.0000000000000000000000000000000C",
    "0x0.00000000000000000000000000000004",
    "0x0.00000000000000000000000000000000",
    "0x0.00000000000000000000000000000000",
};


const float cordic_ctab_table_floatdouble_128[128] = {
7.853982e-01,
4.636476e-01,
2.449787e-01,
1.243550e-01,
6.241881e-02,
3.123983e-02,
1.562373e-02,
7.812341e-03,
3.906230e-03,
1.953123e-03,
9.765622e-04,
4.882812e-04,
2.441406e-04,
1.220703e-04,
6.103516e-05,
3.051758e-05,
1.525879e-05,
7.629395e-06,
3.814697e-06,
1.907349e-06,
9.536743e-07,
4.768372e-07,
2.384186e-07,
1.192093e-07,
5.960464e-08,
2.980232e-08,
1.490116e-08,
7.450581e-09,
3.725290e-09,
1.862645e-09,
9.313226e-10,
4.656613e-10,
2.328306e-10,
1.164153e-10,
5.820766e-11,
2.910383e-11,
1.455192e-11,
7.275958e-12,
3.637979e-12,
1.818989e-12,
9.094947e-13,
4.547474e-13,
2.273737e-13,
1.136868e-13,
5.684342e-14,
2.842171e-14,
1.421085e-14,
7.105427e-15,
3.552714e-15,
1.776357e-15,
8.881784e-16,
4.440892e-16,
2.220446e-16,
1.110223e-16,
5.551115e-17,
2.775558e-17,
1.387779e-17,
6.938894e-18,
3.469447e-18,
1.734723e-18,
8.673617e-19,
4.336809e-19,
2.168404e-19,
1.084202e-19,
5.421011e-20,
2.710505e-20,
1.355253e-20,
6.776264e-21,
3.388132e-21,
1.694066e-21,
8.470329e-22,
4.235165e-22,
2.117582e-22,
1.058791e-22,
5.293956e-23,
2.646978e-23,
1.323489e-23,
6.617445e-24,
3.308722e-24,
1.654361e-24,
8.271806e-25,
4.135903e-25,
2.067952e-25,
1.033976e-25,
5.169879e-26,
2.584939e-26,
1.292470e-26,
6.462349e-27,
3.231174e-27,
1.615587e-27,
8.077936e-28,
4.038968e-28,
2.019484e-28,
1.009742e-28,
5.048710e-29,
2.524355e-29,
1.262177e-29,
6.310887e-30,
3.155444e-30,
1.577722e-30,
7.888609e-31,
3.944305e-31,
1.972152e-31,
9.860761e-32,
4.930381e-32,
2.465190e-32,
1.232595e-32,
6.162976e-33,
3.081488e-33,
1.540744e-33,
7.703720e-34,
3.851860e-34,
1.925930e-34,
9.629650e-35,
4.814825e-35,
2.407412e-35,
1.203706e-35,
6.018531e-36,
3.009266e-36,
1.504633e-36,
7.523164e-37,
3.761582e-37,
1.880791e-37,
9.403955e-38,
4.701977e-38,
2.350989e-38,
1.175494e-38,
5.877472e-39,
};

const float cordic_hyperb_table_floatdouble_128[128] = {
5.493061e-01,
2.554128e-01,
1.256572e-01,
6.258157e-02,
3.126018e-02,
1.562627e-02,
7.812659e-03,
3.906270e-03,
1.953127e-03,
9.765628e-04,
4.882813e-04,
2.441406e-04,
1.220703e-04,
6.103516e-05,
3.051758e-05,
1.525879e-05,
7.629395e-06,
3.814697e-06,
1.907349e-06,
9.536743e-07,
4.768372e-07,
2.384186e-07,
1.192093e-07,
5.960464e-08,
2.980232e-08,
1.490116e-08,
7.450581e-09,
3.725290e-09,
1.862645e-09,
9.313226e-10,
4.656613e-10,
2.328306e-10,
1.164153e-10,
5.820766e-11,
2.910383e-11,
1.455192e-11,
7.275958e-12,
3.637979e-12,
1.818989e-12,
9.094947e-13,
4.547474e-13,
2.273737e-13,
1.136868e-13,
5.684342e-14,
2.842171e-14,
1.421085e-14,
7.105427e-15,
3.552714e-15,
1.776357e-15,
8.881784e-16,
4.440892e-16,
2.220446e-16,
1.110223e-16,
0.000000e+00,
0.000000e+00,
0.000000e+00,
0.000000e+00,
0.000000e+00,
0.000000e+00,
0.000000e+00,
0.000000e+00,
0.000000e+00,
0.000000e+00,
0.000000e+00,
0.000000e+00,
0.000000e+00,
0.000000e+00,
0.000000e+00,
0.000000e+00,
0.000000e+00,
0.000000e+00,
0.000000e+00,
0.000000e+00,
0.000000e+00,
0.000000e+00,
0.000000e+00,
0.000000e+00,
0.000000e+00,
0.000000e+00,
0.000000e+00,
0.000000e+00,
0.000000e+00,
0.000000e+00,
0.000000e+00,
0.000000e+00,
0.000000e+00,
0.000000e+00,
0.000000e+00,
0.000000e+00,
0.000000e+00,
0.000000e+00,
0.000000e+00,
0.000000e+00,
0.000000e+00,
0.000000e+00,
0.000000e+00,
0.000000e+00,
0.000000e+00,
0.000000e+00,
0.000000e+00,
0.000000e+00,
0.000000e+00,
0.000000e+00,
0.000000e+00,
0.000000e+00,
0.000000e+00,
0.000000e+00,
0.000000e+00,
0.000000e+00,
0.000000e+00,
0.000000e+00,
0.000000e+00,
0.000000e+00,
0.000000e+00,
0.000000e+00,
0.000000e+00,
0.000000e+00,
0.000000e+00,
0.000000e+00,
0.000000e+00,
0.000000e+00,
0.000000e+00,
0.000000e+00,
0.000000e+00,
0.000000e+00,
0.000000e+00,
0.000000e+00,
0.000000e+00,
};


const ap_ufixed<128,4> cordic_hyperb_table_128[128] = {
"0x0.8c9f53d5681854bb520cc6aa829dbe5",
"0x0.4162bbea0451469c9daf0be0810eda9",
"0x0.202b12393d5deed328cf41ed722d8c9",
"0x0.1005588ad375acdcb1312a563c68525",
"0x0.0800aac448d77125a4ee9fee2db3774",
"0x0.04001556222b47263834e958ab3b4ca",
"0x0.020002aab111235a6e87a29f88bb425",
"0x0.01000055558888ad1aee1ef93404079",
"0x0.0080000aaaac44448d68e4c64f4d811",
"0x0.004000015555622222b46b4dd0dd6ae",
"0x0.002000002aaaab11111235a35dc3dc4",
"0x0.001000000555555888888ad1ad1c98c",
"0x0.0008000000aaaaaac4444448d68d69b",
"0x0.0004000000155555562222222b46b46",
"0x0.000200000002aaaaaab1111111235a3",
"0x0.0001000000005555555588888888ad1",
"0x0.0000800000000aaaaaaaac444444448",
"0x0.0000400000000155555555622222222",
"0x0.000020000000002aaaaaaaab1111111",
"0x0.0000100000000005555555555888888",
"0x0.0000080000000000aaaaaaaaaac4444",
"0x0.0000040000000000155555555556222",
"0x0.000002000000000002aaaaaaaaaab11",
"0x0.0000010000000000005555555555558",
"0x0.0000008000000000000aaaaaaaaaaaa",
"0x0.0000004000000000000155555555555",
"0x0.000000200000000000002aaaaaaaaaa",
"0x0.0000001000000000000005555555555",
"0x0.0000000800000000000000aaaaaaaaa",
"0x0.0000000400000000000000155555555",
"0x0.000000020000000000000002aaaaaaa",
"0x0.0000000100000000000000005555555",
"0x0.0000000080000000000000000aaaaaa",
"0x0.0000000040000000000000000155555",
"0x0.000000002000000000000000002aaaa",
"0x0.0000000010000000000000000005555",
"0x0.0000000008000000000000000000aaa",
"0x0.0000000004000000000000000000155",
"0x0.000000000200000000000000000002a",
"0x0.0000000001000000000000000000005",
"0x0.0000000000800000000000000000000",
"0x0.0000000000400000000000000000000",
"0x0.00000000001ffffffffffffffffffff",
"0x0.00000000000ffffffffffffffffffff",
"0x0.000000000007fffffffffffffffffff",
"0x0.000000000003fffffffffffffffffff",
"0x0.000000000001fffffffffffffffffff",
"0x0.000000000000fffffffffffffffffff",
"0x0.0000000000007ffffffffffffffffff",
"0x0.0000000000003ffffffffffffffffff",
"0x0.0000000000001ffffffffffffffffff",
"0x0.0000000000000ffffffffffffffffff",
"0x0.00000000000007fffffffffffffffff",
"0x0.00000000000003fffffffffffffffff",
"0x0.00000000000001fffffffffffffffff",
"0x0.00000000000000fffffffffffffffff",
"0x0.000000000000007ffffffffffffffff",
"0x0.000000000000003ffffffffffffffff",
"0x0.000000000000001ffffffffffffffff",
"0x0.000000000000000ffffffffffffffff",
"0x0.0000000000000007fffffffffffffff",
"0x0.0000000000000003fffffffffffffff",
"0x0.0000000000000001fffffffffffffff",
"0x0.0000000000000000fffffffffffffff",
"0x0.00000000000000007ffffffffffffff",
"0x0.00000000000000003ffffffffffffff",
"0x0.00000000000000001ffffffffffffff",
"0x0.00000000000000000ffffffffffffff",
"0x0.000000000000000007fffffffffffff",
"0x0.000000000000000003fffffffffffff",
"0x0.000000000000000001fffffffffffff",
"0x0.000000000000000000fffffffffffff",
"0x0.0000000000000000007ffffffffffff",
"0x0.0000000000000000003ffffffffffff",
"0x0.0000000000000000001ffffffffffff",
"0x0.0000000000000000000ffffffffffff",
"0x0.00000000000000000007fffffffffff",
"0x0.00000000000000000003fffffffffff",
"0x0.00000000000000000001fffffffffff",
"0x0.00000000000000000000fffffffffff",
"0x0.000000000000000000007ffffffffff",
"0x0.000000000000000000003ffffffffff",
"0x0.000000000000000000001ffffffffff",
"0x0.000000000000000000000ffffffffff",
"0x0.0000000000000000000007fffffffff",
"0x0.0000000000000000000003fffffffff",
"0x0.0000000000000000000001fffffffff",
"0x0.0000000000000000000000fffffffff",
"0x0.00000000000000000000007ffffffff",
"0x0.00000000000000000000003ffffffff",
"0x0.00000000000000000000001ffffffff",
"0x0.00000000000000000000000ffffffff",
"0x0.000000000000000000000007fffffff",
"0x0.000000000000000000000003fffffff",
"0x0.000000000000000000000001fffffff",
"0x0.000000000000000000000000fffffff",
"0x0.0000000000000000000000007ffffff",
"0x0.0000000000000000000000003ffffff",
"0x0.0000000000000000000000001ffffff",
"0x0.0000000000000000000000000ffffff",
"0x0.00000000000000000000000007fffff",
"0x0.00000000000000000000000003fffff",
"0x0.00000000000000000000000001fffff",
"0x0.00000000000000000000000000fffff",
"0x0.000000000000000000000000007ffff",
"0x0.000000000000000000000000003ffff",
"0x0.000000000000000000000000001ffff",
"0x0.000000000000000000000000000ffff",
"0x0.0000000000000000000000000007fff",
"0x0.0000000000000000000000000003fff",
"0x0.0000000000000000000000000001fff",
"0x0.0000000000000000000000000000fff",
"0x0.00000000000000000000000000007ff",
"0x0.00000000000000000000000000003ff",
"0x0.00000000000000000000000000001ff",
"0x0.00000000000000000000000000000ff",
"0x0.000000000000000000000000000007f",
"0x0.000000000000000000000000000003f",
"0x0.000000000000000000000000000001f",
"0x0.000000000000000000000000000000f",
"0x0.0000000000000000000000000000007",
"0x0.0000000000000000000000000000003",
"0x0.0000000000000000000000000000001",
"0x0.0000000000000000000000000000000",
"0x0.0000000000000000000000000000000",
"0x0.0000000000000000000000000000000",
"0x0.0000000000000000000000000000000",
"0x0.0000000000000000000000000000000",
};


const ap_ufixed<128,4> cordic_hyperb_table_128_160[170] = {
"0x0.8c9f53d5681854bb520cc6aa829dbe5",
"0x0.4162bbea0451469c9daf0be0810eda9",
"0x0.202b12393d5deed328cf41ed722d8c9",
"0x0.1005588ad375acdcb1312a563c68525",
"0x0.1005588ad375acdcb1312a563c68525",
"0x0.0800aac448d77125a4ee9fee2db3774",
"0x0.04001556222b47263834e958ab3b4ca",
"0x0.020002aab111235a6e87a29f88bb425",
"0x0.020002aab111235a6e87a29f88bb425",
"0x0.01000055558888ad1aee1ef93404079",
"0x0.0080000aaaac44448d68e4c64f4d811",
"0x0.004000015555622222b46b4dd0dd6ae",
"0x0.004000015555622222b46b4dd0dd6ae",
"0x0.002000002aaaab11111235a35dc3dc4",
"0x0.001000000555555888888ad1ad1c98c",
"0x0.0008000000aaaaaac4444448d68d69b",
"0x0.0008000000aaaaaac4444448d68d69b",
"0x0.0004000000155555562222222b46b46",
"0x0.000200000002aaaaaab1111111235a3",
"0x0.0001000000005555555588888888ad1",
"0x0.0001000000005555555588888888ad1",
"0x0.0000800000000aaaaaaaac444444448",
"0x0.0000400000000155555555622222222",
"0x0.000020000000002aaaaaaaab1111111",
"0x0.000020000000002aaaaaaaab1111111",
"0x0.0000100000000005555555555888888",
"0x0.0000080000000000aaaaaaaaaac4444",
"0x0.0000040000000000155555555556222",
"0x0.0000040000000000155555555556222",
"0x0.000002000000000002aaaaaaaaaab11",
"0x0.0000010000000000005555555555558",
"0x0.0000008000000000000aaaaaaaaaaaa",
"0x0.0000008000000000000aaaaaaaaaaaa",
"0x0.0000004000000000000155555555555",
"0x0.000000200000000000002aaaaaaaaaa",
"0x0.0000001000000000000005555555555",
"0x0.0000001000000000000005555555555",
"0x0.0000000800000000000000aaaaaaaaa",
"0x0.0000000400000000000000155555555",
"0x0.000000020000000000000002aaaaaaa",
"0x0.000000020000000000000002aaaaaaa",
"0x0.0000000100000000000000005555555",
"0x0.0000000080000000000000000aaaaaa",
"0x0.0000000040000000000000000155555",
"0x0.0000000040000000000000000155555",
"0x0.000000002000000000000000002aaaa",
"0x0.0000000010000000000000000005555",
"0x0.0000000008000000000000000000aaa",
"0x0.0000000008000000000000000000aaa",
"0x0.0000000004000000000000000000155",
"0x0.000000000200000000000000000002a",
"0x0.0000000001000000000000000000005",
"0x0.0000000001000000000000000000005",
"0x0.0000000000800000000000000000000",
"0x0.0000000000400000000000000000000",
"0x0.00000000001ffffffffffffffffffff",
"0x0.00000000001ffffffffffffffffffff",
"0x0.00000000000ffffffffffffffffffff",
"0x0.000000000007fffffffffffffffffff",
"0x0.000000000003fffffffffffffffffff",
"0x0.000000000003fffffffffffffffffff",
"0x0.000000000001fffffffffffffffffff",
"0x0.000000000000fffffffffffffffffff",
"0x0.0000000000007ffffffffffffffffff",
"0x0.0000000000007ffffffffffffffffff",
"0x0.0000000000003ffffffffffffffffff",
"0x0.0000000000001ffffffffffffffffff",
"0x0.0000000000000ffffffffffffffffff",
"0x0.0000000000000ffffffffffffffffff",
"0x0.00000000000007fffffffffffffffff",
"0x0.00000000000003fffffffffffffffff",
"0x0.00000000000001fffffffffffffffff",
"0x0.00000000000001fffffffffffffffff",
"0x0.00000000000000fffffffffffffffff",
"0x0.000000000000007ffffffffffffffff",
"0x0.000000000000003ffffffffffffffff",
"0x0.000000000000003ffffffffffffffff",
"0x0.000000000000001ffffffffffffffff",
"0x0.000000000000000ffffffffffffffff",
"0x0.0000000000000007fffffffffffffff",
"0x0.0000000000000007fffffffffffffff",
"0x0.0000000000000003fffffffffffffff",
"0x0.0000000000000001fffffffffffffff",
"0x0.0000000000000000fffffffffffffff",
"0x0.0000000000000000fffffffffffffff",
"0x0.00000000000000007ffffffffffffff",
"0x0.00000000000000003ffffffffffffff",
"0x0.00000000000000001ffffffffffffff",
"0x0.00000000000000001ffffffffffffff",
"0x0.00000000000000000ffffffffffffff",
"0x0.000000000000000007fffffffffffff",
"0x0.000000000000000003fffffffffffff",
"0x0.000000000000000003fffffffffffff",
"0x0.000000000000000001fffffffffffff",
"0x0.000000000000000000fffffffffffff",
"0x0.0000000000000000007ffffffffffff",
"0x0.0000000000000000007ffffffffffff",
"0x0.0000000000000000003ffffffffffff",
"0x0.0000000000000000001ffffffffffff",
"0x0.0000000000000000000ffffffffffff",
"0x0.0000000000000000000ffffffffffff",
"0x0.00000000000000000007fffffffffff",
"0x0.00000000000000000003fffffffffff",
"0x0.00000000000000000001fffffffffff",
"0x0.00000000000000000001fffffffffff",
"0x0.00000000000000000000fffffffffff",
"0x0.000000000000000000007ffffffffff",
"0x0.000000000000000000003ffffffffff",
"0x0.000000000000000000003ffffffffff",
"0x0.000000000000000000001ffffffffff",
"0x0.000000000000000000000ffffffffff",
"0x0.0000000000000000000007fffffffff",
"0x0.0000000000000000000007fffffffff",
"0x0.0000000000000000000003fffffffff",
"0x0.0000000000000000000001fffffffff",
"0x0.0000000000000000000000fffffffff",
"0x0.0000000000000000000000fffffffff",
"0x0.00000000000000000000007ffffffff",
"0x0.00000000000000000000003ffffffff",
"0x0.00000000000000000000001ffffffff",
"0x0.00000000000000000000001ffffffff",
"0x0.00000000000000000000000ffffffff",
"0x0.000000000000000000000007fffffff",
"0x0.000000000000000000000003fffffff",
"0x0.000000000000000000000003fffffff",
"0x0.000000000000000000000001fffffff",
"0x0.000000000000000000000000fffffff",
"0x0.0000000000000000000000007ffffff",
"0x0.0000000000000000000000007ffffff",
"0x0.0000000000000000000000003ffffff",
"0x0.0000000000000000000000001ffffff",
"0x0.0000000000000000000000000ffffff",
"0x0.0000000000000000000000000ffffff",
"0x0.00000000000000000000000007fffff",
"0x0.00000000000000000000000003fffff",
"0x0.00000000000000000000000001fffff",
"0x0.00000000000000000000000001fffff",
"0x0.00000000000000000000000000fffff",
"0x0.000000000000000000000000007ffff",
"0x0.000000000000000000000000003ffff",
"0x0.000000000000000000000000003ffff",
"0x0.000000000000000000000000001ffff",
"0x0.000000000000000000000000000ffff",
"0x0.0000000000000000000000000007fff",
"0x0.0000000000000000000000000007fff",
"0x0.0000000000000000000000000003fff",
"0x0.0000000000000000000000000001fff",
"0x0.0000000000000000000000000000fff",
"0x0.0000000000000000000000000000fff",
"0x0.00000000000000000000000000007ff",
"0x0.00000000000000000000000000003ff",
"0x0.00000000000000000000000000001ff",
"0x0.00000000000000000000000000001ff",
"0x0.00000000000000000000000000000ff",
"0x0.000000000000000000000000000007f",
"0x0.000000000000000000000000000003f",
"0x0.000000000000000000000000000003f",
"0x0.000000000000000000000000000001f",
"0x0.000000000000000000000000000000f",
"0x0.0000000000000000000000000000007",
"0x0.0000000000000000000000000000007",
"0x0.0000000000000000000000000000003",
"0x0.0000000000000000000000000000001",
"0x0.0000000000000000000000000000000",
"0x0.0000000000000000000000000000000",
"0x0.0000000000000000000000000000000",
"0x0.0000000000000000000000000000000",
"0x0.0000000000000000000000000000000",
"0x0.0000000000000000000000000000000",
"0x0.0000000000000000000000000000000",
};

#endif
