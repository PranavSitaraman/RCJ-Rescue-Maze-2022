#ifndef MATRIX_HPP
#define MATRIX_HPP
#include "tile.hpp"
// y then x
// w then l
class Matrix
{
private:
	static constexpr auto PHI = 1.6180339887498949;
	std::uint32_t l, w;
	std::size_t buf_sz;
	tile *data;
public:
	Matrix() : l(0),w(0),buf_sz(0),data(nullptr){}
	Matrix(std::uint32_t width, std::uint32_t length);
	Matrix(std::uint32_t width, std::uint32_t length, tile* data);
	Matrix(std::uint32_t width, std::uint32_t length, std::size_t buf_size);
	Matrix& operator=(Matrix&& other) noexcept;
	~Matrix();
	tile *operator[](std::uint32_t i) const;
	void resize(std::int32_t x_offset, std::int32_t y_offset, std::uint32_t new_width, std::uint32_t new_length);
	[[nodiscard]] bool adj(std::int32_t y, std::int32_t x, std::uint8_t dir) const;
	void randomize();
	[[nodiscard]] std::uint32_t length() const { return l; }
	[[nodiscard]] std::uint32_t width() const { return w; }
	[[nodiscard]] tile *buf() const { return data; }
};
#endif // MATRIX_HPP