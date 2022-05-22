#ifndef SERIAL_HPP
#define SERIAL_HPP
#include <string_view>
#include <sstream>
#include <cstdint>
#include <cstddef>
class Serial
{
private:
	int fd;
	template<class T>
	[[nodiscard]] std::string
	to_str(std::enable_if<std::is_integral_v<T> || std::is_floating_point_v<T>, T> val, std::uint8_t fmt);
public:
	enum : std::uint8_t
	{
		BIN, OCT, DEC, HEX
	};
	Serial(const std::string& port,std::uint32_t baud);
	~Serial();
	std::size_t write(const void *buf, std::size_t len) const;
	template<class T>
	typename std::enable_if<std::is_pointer_v<decltype(std::data(std::declval<T>()))> && std::is_integral_v<decltype(std::size(std::declval<T>()))>, std::size_t>::type
	write(T &&buf) const;
	template<class T>
	typename std::enable_if<std::is_trivial_v<std::remove_reference_t<T>> && std::is_standard_layout_v<std::remove_reference_t<T>>, std::size_t>::type
	write(T &&val) const;
	unsigned char read() const;
	std::size_t read(void *buf, std::size_t len) const;
	template<class T>
	typename std::enable_if<std::is_pointer_v<decltype(std::data(std::declval<T>()))> && std::is_integral_v<decltype(std::size(std::declval<T>()))>, std::size_t>::type
	read(T &buf) const;
	template<class T>
	typename std::enable_if<std::is_trivial_v<std::remove_reference_t<T>> && std::is_standard_layout_v<std::remove_reference_t<T>>, std::size_t>::type
	read(T &val) const;
	std::size_t print(std::string_view str) const;
	template<class T>
	std::size_t print(std::enable_if<std::is_integral_v<T> || std::is_floating_point_v<T>, T> val) const;
	template<class T>
	std::size_t
	print(std::enable_if<std::is_integral_v<T> || std::is_floating_point_v<T>, T> val, std::uint8_t fmt) const;
	std::size_t println(std::string_view str) const;
	template<class T>
	std::size_t println(std::enable_if<std::is_integral_v<T> || std::is_floating_point_v<T>, T> val) const;
	template<class T>
	std::size_t
	println(std::enable_if<std::is_integral_v<T> || std::is_floating_point_v<T>, T> val, std::uint8_t fmt) const;
	[[nodiscard]] std::size_t available() const;
};
template<class T>
typename std::enable_if<std::is_trivial_v<std::remove_reference_t<T>> && std::is_standard_layout_v<std::remove_reference_t<T>>, std::size_t>::type
Serial::write(T &&val) const
{
	return write(&val, sizeof(val));
}
template<class T>
typename std::enable_if<std::is_pointer_v<decltype(std::data(std::declval<T>()))> && std::is_integral_v<decltype(std::size(std::declval<T>()))>, std::size_t>::type
Serial::write(T &&buf) const
{
	return write(std::data(buf), std::size(buf));
}
template<class T>
typename std::enable_if<std::is_trivial_v<std::remove_reference_t<T>> && std::is_standard_layout_v<std::remove_reference_t<T>>, std::size_t>::type
Serial::read(T &val) const
{
	return read(&val, sizeof(val));
}
template<class T>
typename std::enable_if<std::is_pointer_v<decltype(std::data(std::declval<T>()))> && std::is_integral_v<decltype(std::size(std::declval<T>()))>, std::size_t>::type
Serial::read(T &buf) const
{
	return read(std::data(buf), std::size(buf));
}
template<class T>
std::string Serial::to_str(std::enable_if<std::is_integral_v<T> || std::is_floating_point_v<T>, T> val, std::uint8_t fmt)
{
	if (fmt == BIN)
	{
		std::string bin;
		for (std::uint32_t i = sizeof(T) * 8; i >= 0; i--)
		{
			bin.push_back(((val >> i) & 1) + '0');
		}
		return bin;
	}
	std::stringstream ss;
	switch (fmt)
	{
		case OCT:
			ss << std::oct;
			break;
		case HEX:
			ss << std::hex;
			break;
		case DEC:
		default:
			break;
	}
	ss << val;
	return ss.str();
}
template<class T>
std::size_t Serial::print(const std::enable_if<std::is_integral_v<T> || std::is_floating_point_v<T>, T> val) const
{
	return print(val, Serial::DEC);
}
template<class T>
std::size_t Serial::print(const std::enable_if<std::is_integral_v<T> || std::is_floating_point_v<T>, T> val, std::uint8_t fmt) const
{
	return print(to_str(val, fmt));
}
template<class T>
std::size_t Serial::println(const std::enable_if<std::is_integral_v<T> || std::is_floating_point_v<T>, T> val) const
{
	return println(val, DEC);
}
template<class T>
std::size_t Serial::println(std::enable_if<std::is_integral_v<T> || std::is_floating_point_v<T>, T> val, std::uint8_t fmt) const
{
	return print(to_str(val, fmt) + '\n');
}
#endif //SERIAL_HPP