#include "Serial.hpp"

#include <system_error>
#include <cerrno>

namespace POSIX {

#include <unistd.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <poll.h>

}

namespace termios {

#include <termios.h>

}

Serial::Serial(const std::string &port, std::uint32_t baud) {
	if ((fd = POSIX::open(port.c_str(), O_RDWR | O_NOCTTY)) == -1) {
		throw std::system_error({errno, std::system_category()}, "open");
	}
	termios::termios tty{};
	tty.c_cflag = CRTSCTS | CS8 | CLOCAL | CREAD;
	tty.c_iflag = IGNPAR;
	if (termios::cfsetspeed(&tty, baud) == -1) {
		throw std::system_error({errno, std::system_category()}, "cfsetspeed");
	}
	if (termios::tcflush(fd, TCIFLUSH) == -1) {
		throw std::system_error({errno, std::system_category()}, "tcflush");
	}
	if (termios::tcsetattr(fd, TCSANOW, &tty) == -1) {
		throw std::system_error({errno, std::system_category()}, "tcsetattr");
	}
}

std::size_t Serial::write(const void *buf, std::size_t len) const {
	ssize_t bytes_written;
	if ((bytes_written = POSIX::write(fd, buf, len)) == -1) {
		throw std::system_error({errno, std::system_category()}, "write");
	}
	return bytes_written;
}

unsigned char Serial::read() const {
	unsigned char ret;
	read(&ret, 1);
	return ret;
}

std::size_t Serial::read(void *buf, std::size_t len) const {
	ssize_t total_bytes_read = 0;
	static POSIX::pollfd polled_fd{.fd=fd, .events=POLLIN, .revents=0};
	for (ssize_t bytes_read; total_bytes_read != len; total_bytes_read += bytes_read) {
		if (POSIX::poll(&polled_fd, 1, -1) == -1) {
			throw std::system_error({errno, std::system_category()}, "poll");
		}

		if ((bytes_read = POSIX::read(fd, (std::uint8_t *) buf + total_bytes_read, len - total_bytes_read)) == -1) {
			throw std::system_error({errno, std::system_category()}, "read");
		}
	}
	return total_bytes_read;
}

std::size_t Serial::available() const {
	int n;
	if (POSIX::ioctl(fd, FIONREAD, &n) == -1) {
		throw std::system_error({errno, std::system_category()}, "ioctl");
	}
	return n;
}

std::size_t Serial::print(const std::string_view str) const {
	return write(str);
}

std::size_t Serial::println(const std::string_view str) const {
	auto ret = print(str) + 1;
	write('\n');
	return ret;
}

Serial::~Serial() {
	POSIX::close(fd);
}