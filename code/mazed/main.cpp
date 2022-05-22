#include <chrono>
#include <iostream>
#include <thread>
#include <filesystem>
#include <csignal>
#include <unistd.h>
#include <gpiod.hpp>
using namespace std::chrono_literals;
int main()
{
	std::error_code err;
	std::filesystem::remove("map",err);
	constexpr auto BUTTON_PIN = 27;
	gpiod::chip chip("/dev/gpiochip0");
	auto line = chip.get_line(BUTTON_PIN);
	line.request({"mazed",gpiod::line_request::DIRECTION_INPUT,0});
	pid_t pid = 0;
	for (;;)
	{
		if (line.get_value())
		{
			if (pid)
			{
				kill(pid,SIGKILL);
				pid = 0;
				std::cout << "killing\n";
				std::this_thread::sleep_for(1s);
				continue;
			}
			pid = fork();
			if (pid == -1)
			{
				std::perror("fork");
				return 1;
			}
			if (!pid)
			{
				std::cout << "running\n";
				// close(STDOUT_FILENO);
				if (execl("/home/pi/maze-2022/pi/build/maze-2022","maze-2022",NULL) == -1)
				{
					std::perror("execl");
					std::exit(1);
				}
			}
			std::this_thread::sleep_for(1s);
		}
		std::this_thread::sleep_for(10ms);
	}
}