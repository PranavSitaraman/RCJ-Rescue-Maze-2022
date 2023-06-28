#ifndef CAMERA_HPP
#define CAMERA_HPP
#include <atomic>
#include <mutex>
#include <condition_variable>
#include "search.hpp"

enum class ThreadState : std::uint8_t {
	INIT,STARTED,STOP
};

void detect(std::atomic<ThreadState>& state, Search** search, std::mutex& map_lock, std::condition_variable& map_cv);

#endif //CAMERA_HPP
