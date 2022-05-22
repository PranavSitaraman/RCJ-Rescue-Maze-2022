#include "tile.hpp"
#include <iostream>
tile::tile() : props{} {}
bool &tile::operator[](std::uint8_t index)
{
	return props[index];
}
/**
 * @brief Sets a \ref tile as visited.
 */
void tile::set_vis()
{
	props[Prop::VIS] = true;
}
/**
 * @brief Checks if the \ref tile is visited.
 * @return The \ref tile's visitation state.
 */
bool tile::vis() const
{
	return props[Prop::VIS];
}
void tile::set_vic()
{
	props[Prop::VIC] = true;
}
bool tile::vic() const
{
	return props[Prop::VIC];
}
/**
 * @brief Prints walls and visited state of a \ref tile.
 */
void tile::print() const
{
	std::cout << "Walls: ";
	for (std::uint32_t i = 0; i < 4; i++)
	{
		std::cout << props[i];
	}
	std::cout << "\nVisited: " << props[Prop::VIS] << '\n';
}