#ifndef ARDUINO_H
#define ARDUINO_H

#include <cstring>
#include <cstdint>
#include <thread>
#include <chrono>

using boolean = bool;
using byte = unsigned char;

using std::memcpy;

inline void delay(std::uint64_t ms) {
	std::this_thread::sleep_for(std::chrono::milliseconds(ms));
}

#endif //ARDUINO_H
