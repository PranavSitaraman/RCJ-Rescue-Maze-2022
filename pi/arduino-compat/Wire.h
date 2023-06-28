#ifndef WIREPI_WIRE_H
#define WIREPI_WIRE_H

#include <string_view>
#include <memory>
#include <vector>

class TWIBuffer {
private:
	std::unique_ptr<unsigned char[]> buf;
	size_t len, capacity;
	static constexpr auto PHI = 1.6180339887498949;

	void check_resize(size_t n);

public:
	void push_back(unsigned char value);

	void push_data_back(const unsigned char *data, size_t data_size);

	void reserve(size_t n);

	void set_size(size_t sz);

	unsigned char &operator[](size_t i);

	[[nodiscard]] unsigned char *data() const;

	[[nodiscard]] size_t size() const;

	TWIBuffer() noexcept;
};

class TwoWire {
private:
	int fd;
	uint16_t send_addr;
	TWIBuffer buf;
	size_t buf_pos;
	std::vector<void (*)(int)> recv_callbacks;
	std::vector<void (*)()> send_callbacks;
public:
	void begin();

	//void begin(unsigned char address);
	unsigned char requestFrom(uint16_t address, unsigned char quantity, bool stop = true);

	void beginTransmission(uint16_t address);

	unsigned char endTransmission(bool stop = true);

	unsigned char write(unsigned char value);

	unsigned char write(std::string_view string);

	unsigned char write(const unsigned char *data, size_t len);

	unsigned char available();

	unsigned char read();

	void readBytes(unsigned char *data, size_t len);

	void setClock(int freq);

	void onReceive(void (*handler)(int));

	void onRequest(void (*handler)());

	void end() {}

	~TwoWire();
};

extern TwoWire Wire;

#endif //WIREPI_WIRE_H