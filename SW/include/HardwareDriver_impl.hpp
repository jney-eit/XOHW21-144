//
// Created by Jonas on 29.11.2018.
//

#ifndef ULTRASCALE_HARDWAREDRIVER_HWDRIVER_IMPL_H
#define ULTRASCALE_HARDWAREDRIVER_HWDRIVER_IMPL_H

//#define ON_BOARD 1


HardwareDriver::HardwareDriver(const int baseAddress) : _BaseAddress(baseAddress), _CurrentInputBaseAddress(TOPLEVEL_BASE_ADDR_INPUT){
#if ON_BOARD == 1
    _TopDev = open_dev(_BaseAddress, 0xFFFF);
    _TopCntrl = ((unsigned*)get_dev_ptr(_TopDev, TOPLEVEL_ADDR_CTRL));
    //std::cout << "Opened device at: " << std::hex << _BaseAddress << std::dec << std::endl;
#endif
}

void HardwareDriver::printStatus(){
#if ON_BOARD == 1
    std::bitset<sizeof(unsigned)> status(*_TopCntrl);
    std::cout << "HardwareDriver: " << "Status of HW Device: " << status << std::endl;
#endif
}

void HardwareDriver::printStatusUnsigned(){
#if ON_BOARD == 1
    std::cout << "HardwareDriver: " << "Status of HW Device: " << *_TopCntrl << std::endl;
#endif
}

void HardwareDriver::printDebug(){
#if ON_BOARD == 1
    volatile int* debug = ((volatile int*)get_dev_ptr(_TopDev, 0x20));
    std::cout << "Debug value: " << *debug << std::endl;
#endif
}


template<typename T>
void HardwareDriver::setInputs(T input) {
#if ON_BOARD == 1
    *((volatile uint32_t*)get_dev_ptr(_TopDev, _CurrentInputBaseAddress)) = input;
#endif
    //std::cout << "HardwareDriver: " << "Current input address: " << std::hex << _CurrentInputBaseAddress << " | Value: " << input << std::dec << std::endl;
    _CurrentInputBaseAddress = TOPLEVEL_BASE_ADDR_INPUT; //correct??
}

template<typename T, typename ... Args>
void HardwareDriver::setInputs(T first, Args... args){
#if ON_BOARD == 1
    *((volatile uint32_t*)get_dev_ptr(_TopDev, _CurrentInputBaseAddress)) = first;
#endif
    //std::cout << "HardwareDriver: " << "Current input address: " << std::hex << _CurrentInputBaseAddress << " | Value: " << first << std::dec << std::endl;
    _CurrentInputBaseAddress += sizeof(first)*2; //correct??
    setInputs(args...);
}

void HardwareDriver::startHardware() {
    //Check if device is idle
    //std::cout << "HardwareDriver: " << "Checking if device is IDLE" << std::endl;
    while(*_TopCntrl != (uint32_t)(1 << 2)){};

    //Start device
    //std::cout << "HardwareDriver: " << "Device is IDLE, starting Hardware" << std::endl;
    *_TopCntrl |= 0x1;
}

long HardwareDriver::waitForFinish(const bool measureTime) {
 
    std::chrono::high_resolution_clock::time_point startTime;
    if(measureTime) {
        startTime = std::chrono::high_resolution_clock::now();
    }
    
    //Wait for device to finish
    //std::cout << "HardwareDriver: " << "Waiting for Hardware to finish" << std::endl;
    while((*_TopCntrl & 0x0000000E)==0){};
    //std::cout << "HardwareDriver: " << "Hardware finished successfully" << std::endl;   
 
    if(measureTime) {
        auto endTime = std::chrono::high_resolution_clock::now();
        long int duration = static_cast<long int>(std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count());
        return (duration);
    }
    return 1;
 
}


bool HardwareDriver::checkIfFinished() {
 

    if((*_TopCntrl & 0x0000000E)==0){
		return false;
	}
 
    return true;
 
}

long HardwareDriver::calculate(const bool measureTime) {
#if ON_BOARD == 1

    //Check if device is idle
    //std::cout << "HardwareDriver: " << "Checking if device is IDLE" << std::endl;
    while(*_TopCntrl != (uint32_t)(1 << 2)){};

	
    std::chrono::high_resolution_clock::time_point startTime;
    if(measureTime) {
        startTime = std::chrono::high_resolution_clock::now();
    }
	
    //Start device
    //std::cout << "HardwareDriver: " << "Device is IDLE, starting Hardware" << std::endl;
    *_TopCntrl |= 0x1;

    //Wait for device to finish
    //std::cout << "HardwareDriver: " << "Waiting for Hardware to finish" << std::endl;
    while((*_TopCntrl & 0x0000000E)==0){};
    //std::cout << "HardwareDriver: " << "Hardware finished successfully" << std::endl;

    if(measureTime) {
        auto endTime = std::chrono::high_resolution_clock::now();
        long int duration = static_cast<long int>(std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count());
        return (duration);
    }
    return 1;

#endif
}


HardwareDriver::~HardwareDriver(){
#if ON_BOARD == 1
    close_dev(_TopDev, 0xFFFF);
#endif
}

//size: input size in byte
template<typename T>
void writeToDram(const T* const data, const int size, const int baseAddress){
#if ON_BOARD == 1
    IO_DEV input_mem = open_dev(baseAddress, size*sizeof(T)*8);
    for(int i=0; i < size; ++i){
        *((T*)get_dev_ptr(input_mem, i*sizeof(T))) = data[i];
    }
    close_dev(input_mem, size*sizeof(T)*8);
#endif
}

//size: input size in byte
template<typename T>
void readFromDram(T* const data, const int size, const int baseAddress){
#if ON_BOARD == 1
    IO_DEV output_mem = open_dev(baseAddress, size*sizeof(T)*8);
    for(int i=0; i < size; ++i){
        data[i] = *((T*)get_dev_ptr(output_mem, i*sizeof(T)));
    }
    close_dev(output_mem, size*sizeof(T)*8);
#endif
}
#endif //ULTRASCALE_HARDWAREDRIVER_HWDRIVER_IMPL_H
