#include <Wire.h>
#include <system_error>

namespace POSIX {

#include <sys/ioctl.h>
#include <fcntl.h>
#include <unistd.h>

}

namespace I2C {

#include <linux/i2c-dev.h>
#include <linux/i2c.h>

}

TWIBuffer::TWIBuffer() noexcept: buf(nullptr), len(0), capacity(0) {}

void TWIBuffer::reserve(size_t n) {
	if (n > capacity) {
		auto new_buf = std::make_unique<unsigned char[]>(n);
		std::copy(buf.get(), buf.get() + len, new_buf.get());
		buf = std::move(new_buf);
		capacity = n;
	}
}

void TWIBuffer::check_resize(size_t n) {
	if (len + n > capacity) {
		reserve(static_cast<size_t>((len + n) * PHI));
	}
}

void TWIBuffer::push_back(const unsigned char value) {
	check_resize(1);
	buf[len++] = value;
}

void TWIBuffer::push_data_back(const unsigned char *data, size_t data_size) {
	check_resize(data_size);
	std::copy(data, data + data_size, buf.get() + len);
	len += data_size;
}

unsigned char &TWIBuffer::operator[](size_t i) {
	return buf[i];
}

unsigned char *TWIBuffer::data() const {
	return buf.get();
}

size_t TWIBuffer::size() const {
	return len;
}

void TWIBuffer::set_size(size_t sz) {
	len = sz;
}

void TwoWire::begin() {
	if ((fd = POSIX::open("/dev/i2c-3", O_RDWR)) == -1) {
		throw std::system_error({errno, std::system_category()}, "open");
	}
	buf.reserve(256);
	send_addr = 0;
	buf_pos = 0;
}

void TwoWire::beginTransmission(uint16_t address) {
	send_addr = address;
	buf.set_size(0);
}

unsigned char TwoWire::write(unsigned char value) {
	buf.push_back(value);
	return 1;
}

unsigned char TwoWire::write(std::string_view string) {
	buf.push_data_back(reinterpret_cast<const unsigned char *>(string.data()), string.size());
	return string.length();
}

unsigned char TwoWire::write(const unsigned char *data, size_t len) {
	buf.push_data_back(data, len);
	return len;
}

unsigned char TwoWire::endTransmission(bool stop) {
	if (stop || buf.size() > 1) {
		if (POSIX::ioctl(fd, I2C_SLAVE, send_addr) == -1) {
			throw std::system_error({errno, std::system_category()}, "ioctl");
		}

		if (POSIX::write(fd, buf.data(), buf.size()) == -1) {
			throw std::system_error({errno, std::system_category()}, "write");
		}
	} else {
		return 0;
	}
	for (auto fn: recv_callbacks) {
		fn(static_cast<int>(buf.size()));
	}
	buf.set_size(0);
	return 0;
}

unsigned char TwoWire::requestFrom(uint16_t address, unsigned char quantity, bool stop) {
	buf.reserve(quantity);
	if (stop) {
		if (POSIX::ioctl(fd, I2C_SLAVE, send_addr) == -1) {
			throw std::system_error({errno, std::system_category()}, "ioctl");
		}

		if (POSIX::read(fd, buf.data(), quantity) == -1) {
			throw std::system_error({errno, std::system_category()}, "read");
		}
	} else if (buf.size() == 1) {
		I2C::i2c_msg msgs[]{{.addr=address, .flags=0, .len=1, .buf=buf.data()},
		                    {.addr=address, .flags=I2C_M_RD, .len=quantity, .buf=buf.data()}};
		I2C::i2c_rdwr_ioctl_data data{.msgs=msgs, .nmsgs=2};

		if (POSIX::ioctl(fd, I2C_RDWR, &data) == -1) {
			throw std::system_error({errno, std::system_category()}, "ioctl");
		}
	} else {
		return 0;
	}
	buf.set_size(quantity);
	buf_pos = 0;
	for (auto fn: send_callbacks) {
		fn();
	}
	return quantity;
}

unsigned char TwoWire::read() {
	return buf[buf_pos++];
}

void TwoWire::readBytes(unsigned char *data, size_t len) {
	std::copy(buf.data(), buf.data() + len, data);
	buf_pos += len;
}

unsigned char TwoWire::available() {
	return buf.size() - buf_pos;
}

void TwoWire::setClock(int freq) {}

void TwoWire::onReceive(void (*handler)(int)) {
	recv_callbacks.push_back(handler);
}

void TwoWire::onRequest(void (*handler)()) {
	send_callbacks.push_back(handler);
}

TwoWire::~TwoWire() {
	POSIX::close(fd);
}

TwoWire Wire;