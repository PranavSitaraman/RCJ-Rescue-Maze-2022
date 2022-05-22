#include "matrix.hpp"
#include <cstring>
#include <random>
Matrix::Matrix(std::uint32_t width, std::uint32_t length) : w(width), l(length), buf_sz(length * width * PHI), data(new tile[buf_sz]()) {}
Matrix::Matrix(std::uint32_t width, std::uint32_t length, tile* buf) : w(width), l(length), buf_sz(length * width), data(buf) {}
Matrix::Matrix(std::uint32_t width, std::uint32_t length, std::uint32_t buf_size) : w(width), l(length), buf_sz(buf_size), data(new tile[buf_size]()) {}
Matrix& Matrix::operator =(Matrix&& other) noexcept
{
	std::swap(other.data,data);
	w = other.w;
	l = other.l;
	buf_sz = other.buf_sz;
	return *this;
}
Matrix::~Matrix()
{
	delete[] data;
}
tile *Matrix::operator[](std::uint32_t i) const
{
	return data + i * l;
}
void Matrix::resize(std::int32_t x_offset, std::int32_t y_offset, std::uint32_t new_width, std::uint32_t new_length)
{
	if (new_width == w && new_length == l)
	{
		return;
	}
	if (new_width * new_length > buf_sz)
	{
		buf_sz = new_width * new_length * PHI;
		tile *new_data = new tile[buf_sz]();
		for (std::uint32_t i = 0; i < w; i++)
		{
			std::memcpy(new_data + (i + y_offset) * new_length + x_offset, data + i * l, l * sizeof(*data));
		}
		delete[] data;
		data = new_data;
	}
	else if (y_offset != 0 && x_offset == 0)
	{
		std::memmove(data + y_offset * new_length, data, w * l * sizeof(*data));
		std::memset(data, 0, y_offset * new_length * sizeof(*data));
	}
	else if (y_offset != 0 || x_offset != 0)
	{
		for (std::int32_t i = w - 1; i >= 0; i--)
		{
			//std::swap_ranges(data+i*l,data+i*l+l,data + (i + y_offset) * l + x_offset);
			std::memmove(data + (i + y_offset) * new_length + x_offset, data + i * l, l * sizeof(*data));
		}
		std::memset(data, 0, y_offset * new_length * sizeof(*data));
		for (std::uint32_t i = y_offset; i < w; i++)
		{
			std::memset(data + i * new_length, 0, x_offset * sizeof(*data));
		}
	}
	w = new_width;
	l = new_length;
}
bool Matrix::adj(std::int32_t y, std::int32_t x, std::uint8_t dir) const
{
	auto &map = *this;
	if (dir == Dir::N && (y - 1) >= 0)
	{
		return map[y - 1][x][(dir + 2) % 4];
	}
	else if (dir == Dir::S && (y + 1) < w)
	{
		return map[y + 1][x][(dir + 2) % 4];
	}
	else if (dir == Dir::W && (x - 1) >= 0)
	{
		return map[y][x - 1][(dir + 2) % 4];
	}
	else if (dir == Dir::E && (x + 1) < l)
	{
		return map[y][x + 1][(dir + 2) % 4];
	}
	else
	{
		return false;
		// return map[y][x][dir];
	}
}
void Matrix::randomize()
{
	static std::random_device rd;
	static std::default_random_engine gen(rd());
	static std::uniform_int_distribution<std::uint32_t> dist(0, 1);
	auto &map = *this;
	for (std::int32_t i = 0; i < w; i++)
	{
		for (std::int32_t j = 0; j < l; j++)
		{
			for (std::uint8_t k = 0; k < 4; k++)
			{
				if ((k == Dir::N && i == 0) || (k == Dir::W && j == 0) || (k == Dir::S && i == w - 1) || (k == Dir::E && j == l - 1))
				{
					map[i][j][k] = true;
				}
				else if (k == Dir::N || k == Dir::W)
				{
					map[i][j][k] = map.adj(i, j, k);
				}
				else
				{
					map[i][j][k] = dist(gen);
				}
			}
		}
	}
}