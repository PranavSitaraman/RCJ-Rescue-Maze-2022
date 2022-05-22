#ifndef TILE_HPP
#define TILE_HPP

#include <array>
#include <cstdint>

namespace Dir {
	enum : uint8_t {
		N, E, S, W
	};
}

namespace Prop {
	enum : uint8_t {
		VIS = Dir::W + 1, VIC
	};
}

/**
 * Includes methods to set and get property bits, print properties, and otherwise manipulate said properties. See Tile for more information.
 * @brief Represents a maze tile.
 */
class tile {
private:
	std::array<bool, 6> props;
public:
	tile();

	bool &operator[](std::uint8_t i);

	void print() const;

	void set_vis();

	void set_vic();

	[[nodiscard]] bool vic() const;

	[[nodiscard]] bool vis() const;
};

#endif
