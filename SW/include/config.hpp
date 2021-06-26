#ifndef CONFIG_H_
#define CONFIG_H_   

#define MAX_NUMBER_INSTANCES			 	1

#define DATAWIDTH_IN	 	 	 	 		16
#define DATAWIDTH_OUT	 	 	 			32
#define CLASS_LABEL_BITS 	 	 	 		1
#define SEQUENCE_LENGTH 	 	 	 		120000
#define INPUT_VAL_SKIP_FAC           		16 

#define INPUT_BITS				     		16
#define INPUT_INT_BITS				 		8

#define TRANSFORM_INPUTS			 		1
#define INPUT_MASK					 		0xf00f
#define INPUT_BITS_PER_VAL 			 		8

#define BASE_INPUT_ADDRESS                  0x16800000
#define BASE_OUTPUT_ADDRESS                 0x17800000

#define FIRST_INSTANCE_CTRL_ADDRESS         0x43C00000

#endif
