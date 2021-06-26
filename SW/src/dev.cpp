#include "dev.hpp"

IO_DEV open_dev(int base_addr, int addr_size) {
	int fd;
	unsigned page_addr, page_offset;
	void *ptr;
	//get the size of one Page (number of bytes in a memory page)
	unsigned page_size = sysconf(_SC_PAGESIZE);

	/* Open /dev/mem file O_RDWR -> Open for reading and writing*/
	//dev/mem provides access to phyical memory -> byste addresses are interpreted as physical memory
	fd = open ("/dev/mem", O_RDWR);
	if (fd < 1) {
		perror("Could not open /dev/mem");
		exit(-1);
	}
	
	if(addr_size == -1){
		addr_size = (int) page_size;
	}

	/* mmap the device into memory */
	//(page_size - 1) -> all bits are set that are lower order than the page size bit
	//~ -> all those bits get zero and the higher bits one
	//bitwise and -> all bits that are lower order than page size get set to 0 else stays at it is
	//-> page_addr calculatet
	page_addr = (base_addr & (~(page_size-1)));
	//calculate page_offset 
	page_offset = base_addr - page_addr;
	//mmap maps file or device into memory: void *mmap(void *addr, size_t lengthint " prot ", int " flags, int fd, off_t offset)
	//addr=NULL -> kernel chooses address at which to create the mapping -> address returned to ptr
	//bitwise or of PROT_READ and PROT_WRITE-> page may be read and written
	//MAP_SHARED -> share this mapping to other processes
	ptr = mmap(NULL, addr_size, PROT_READ|PROT_WRITE, MAP_SHARED, fd, page_addr);
	
	
	//safe file descriptor, page_offset and address ptr in struct
	IO_DEV dev = {fd, page_offset, ptr};
	return dev;
}

//addr_offset -> offset from hls of in and outputs of hardware device to base address of the device
void *get_dev_ptr(IO_DEV dev, int addr_offset) {
	return (void*)((char*)dev.ptr + dev.page_offset + addr_offset);
}

void close_dev(IO_DEV dev, int addr_size) {
	unsigned page_size = sysconf(_SC_PAGESIZE);
	if(addr_size == -1){
		addr_size = (int) page_size;
	}	
	munmap(dev.ptr, addr_size);
	close(dev.fd);
}