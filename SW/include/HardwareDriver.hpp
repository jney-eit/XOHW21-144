//
// Created by Jonas on 29.11.2018.
//

#ifndef ULTRASCALE_HARDWAREDRIVER_HARDWAREDRIVER_H
#define ULTRASCALE_HARDWAREDRIVER_HARDWAREDRIVER_H

#include <iostream>
#include <bitset>
#include <cstdarg>
#include <chrono>
#include "dev.hpp"

#define TOPLEVEL_ADDR_CTRL          0x00
#define TOPLEVEL_ADDR_GIE           0x04
#define TOPLEVEL_ADDR_IER           0x08
#define TOPLEVEL_ADDR_ISR           0x0c
#define TOPLEVEL_BASE_ADDR_INPUT    0x10

class HardwareDriver {
public:
    HardwareDriver(int baseAddress = 0xA0000000);
    ~HardwareDriver();

    template<typename T> void                       setInputs(T first);
    template<typename T, typename... Args> void     setInputs(T first, Args... args);
    void                                            startHardware();
    long int                                        waitForFinish(const bool measureTime = false);
	bool											checkIfFinished();
    long int                                        calculate(const bool measureTime = false);
    void                                            printStatus();
	void                                            printStatusUnsigned();
    void                                            printDebug();


protected:
private:
    int                 _BaseAddress;
    IO_DEV              _TopDev;
    volatile unsigned*  _TopCntrl;
    int                 _CurrentInputBaseAddress;
};

template<typename T> void   writeToDram(const T *data, const int size, const int baseAddress);
template<typename T> void   readFromDram(T *data, const int size, const int baseAddress);

#include "HardwareDriver_impl.hpp"

#endif //ULTRASCALE_HARDWAREDRIVER_HWDRIVER_H
