#ifndef DEV_HPP
#define DEV_HPP

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <sys/time.h>
#include <time.h>

#define XDMA_DOTPRODUCT_MAX_CONTROL_DMA_ADDR_AP_CTRL            0x00
#define XDMA_DOTPRODUCT_MAX_CONTROL_DMA_ADDR_GIE                0x04
#define XDMA_DOTPRODUCT_MAX_CONTROL_DMA_ADDR_IER                0x08
#define XDMA_DOTPRODUCT_MAX_CONTROL_DMA_ADDR_ISR                0x0c
#define XDMA_DOTPRODUCT_MAX_CONTROL_DMA_ADDR_INPUT_ADDRESS_DATA 0x10
#define XDMA_DOTPRODUCT_MAX_CONTROL_DMA_BITS_INPUT_ADDRESS_DATA 32

#define XDP_MAX_CONTROL_ADDR_AP_CTRL       0x00
#define XDP_MAX_CONTROL_ADDR_GIE           0x04
#define XDP_MAX_CONTROL_ADDR_IER           0x08
#define XDP_MAX_CONTROL_ADDR_ISR           0x0c
#define XDP_MAX_CONTROL_ADDR_OUTPUT_V_DATA 0x10
#define XDP_MAX_CONTROL_BITS_OUTPUT_V_DATA 32
#define XDP_MAX_CONTROL_ADDR_OUTPUT_V_CTRL 0x14

typedef struct
{
	int fd;
	int page_offset;
	void *ptr;
} IO_DEV;
	
IO_DEV open_dev(int base_addr, int addr_size);
void *get_dev_ptr(IO_DEV dev, int addr_offset);
void close_dev(IO_DEV dev, int addr_size);
		
#endif