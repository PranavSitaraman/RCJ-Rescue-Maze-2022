#ifndef SEARCH_HPP
#define SEARCH_HPP
#include "matrix.hpp"
#include "Serial.hpp"
#include <stack>
#include <mutex>
#include <condition_variable>
/**
 * Class containing methods used to search the maze, including manipulation of a
 * map of the maze, searching said map, and displaying said map.
 * @brief Class containing methods used for searching the maze.
 */
class Search
{
public:
	Search(const char *map_dump, std::mutex &lock, std::condition_variable &cv, Serial& serial1);
	Search(std::mutex &lock, std::condition_variable &cv,const char* path, Serial& serial1);
	bool move(std::stack<std::uint8_t> &);
	bool move(std::stack<std::uint8_t> &&);
	void check_walls();
	void print_map() const;
	[[nodiscard]] bool get_current_vic() const;
	void set_current_vic();
	[[nodiscard]] bool get_current_vis();
	void dump_map();
	void unmark_start();
	[[nodiscard]] std::stack<std::uint8_t> search() const;
	Matrix map; /**< @brief Dynamically generated map of \ref tile ""s. */
	std::int32_t x; /**< @brief Current x coordinate of robot in #map. */
	std::int32_t y; /**< @brief Current y coordinate of robot in #map. */
private:
	std::uint8_t cd; /**< @brief Current direction robot is facing. Corresponds to
				   directions in #direction. */
	std::mutex &map_lock;
	std::condition_variable &map_cv;
	std::uint32_t init_x; /**< @brief Initial x value of robot. Used to return to starting
				  point. */
	std::uint32_t init_y; /**< @brief Initial y value of robot. Used to return to starting
				  point. */
	Serial& serial;
	const char* filename;
};
#endif