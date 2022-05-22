#include <queue>
#include <unordered_map>
#include <iostream>
#include <filesystem>
#include <fstream>
#include "search.hpp"
#include "tile.hpp"

namespace fs = std::filesystem;

struct Pos {
	std::int32_t x, y;

	bool operator==(const Pos &p) const {
		return x == p.x && y == p.y;
	}
};

struct pos_hash {
private:
	static std::size_t hash_combine(std::size_t hash1, std::size_t hash2) {
		hash1 ^= hash2 + 0x9e3779b9 + (hash1 << 6) + (hash1 >> 2);
		return hash1;
	}

public:
	std::size_t operator()(const Pos &p) const {
		std::hash<std::int32_t> hasher;
		return hash_combine(hasher(p.x), hasher(p.y));
	}
};

void istream_readv(std::istream &is) {}

template<class T, class ...Args>
void istream_readv(std::istream &is, T &val, Args &...args) {
	is.read(reinterpret_cast<char *>(&val), sizeof(val));
	istream_readv(is, args...);
}

/**
 * @brief Initializes length and width of #map, #map, and pos direction.
 */
Search::Search(Serial& ser,std::mutex &lock, std::condition_variable &cv,const char* path) : map(1, 1, 50), init_x(0), init_y(0), x(0), y(0),
                                                                cd(Dir::N), serial(ser),
                                                                map_lock(lock), map_cv(cv),filename(path) {
	constexpr std::uint8_t RESET = Dir::W+1;
	serial.write((std::uint8_t)((1 << 7)|RESET));
	serial.read();
}

Search::Search(Serial& ser, const char *path, std::mutex &lock, std::condition_variable &cv) :
		cd(Dir::N), serial(ser),
		map_lock(lock), map_cv(cv),filename(path) {
	std::ifstream in(path,std::ios::binary);
	std::uint32_t width,length;
	istream_readv(in,x,y,init_x,init_y,cd,width,length);

	tile* data = new tile[width*length];
	in.read(reinterpret_cast<char*>(data),width*length*sizeof(*data));

	map = std::move(Matrix(width,length,data));

	constexpr std::uint8_t RESET = Dir::W+1;
	serial.write((std::uint8_t)((1 << 7)|RESET));
	serial.read();
}

/**
 * @brief Searches the #map for unvisited \ref tile ""s.
 * @return A stack of directions from the pos \ref tile to the target \ref
 * tile, or an empty stack if all \ref tile ""s are visited.
 */
std::stack<std::uint8_t> Search::search() const {
	// hold tiles to search
	std::queue<Pos> q;
	// parent child map for backtracking
	std::unordered_map<Pos, Pos, pos_hash> pc;
	// stack of directions to return
	std::stack<std::uint8_t> dir;
	// mark start tile as visited
	map_lock.lock();
	map[y][x].set_vis();
	map_lock.unlock();
	// enqueue start tile and position
	q.push({x, y});
	// set root node to -1
	pc[{x, y}] = {-1, -1};
	while (!q.empty()) {
		// get tile and coordinates
		auto current_tile = q.front();
		q.pop();

		map_lock.lock();
		// if unvisited get directions and return
		if (!map[current_tile.y][current_tile.x].vis()) {
			map_lock.unlock();
			// get direction of pos tile
			Pos p1 = current_tile;
			Pos p2 = p1;
			// search for root node
			while ((p1 = (*pc.find(p1)).second).x != -1) {
				// obtain direction from coordinates and add to stack
				if ((p2.y - p1.y) == -1) {
					dir.push(Dir::N);
				} else if ((p2.y - p1.y) == 1) {
					dir.push(Dir::S);
				} else if ((p2.x - p1.x) == -1) {
					dir.push(Dir::W);
				} else if ((p2.x - p1.x) == 1) {
					dir.push(Dir::E);
				}
				p2 = p1;
			}
			return dir;
		}

		// iterate through walls
		for (std::uint8_t i = 0; i < 4; i++) {
			Pos p1 = current_tile;
			// if no wall is present, at this tile or the adjacent one
			if (!map[p1.y][p1.x][i] && !map.adj(p1.y, p1.x, i)) {
				// go in direction where no wall is present
				if (i == Dir::S && (p1.y + 1) < map.width()) {
					p1.y++;
				} else if (i == Dir::N && (p1.y - 1) >= 0) {
					p1.y--;
				} else if (i == Dir::E && (p1.x + 1) < map.length()) {
					p1.x++;
				} else if (i == Dir::W && (p1.x - 1) >= 0) {
					p1.x--;
				} else {
					continue;
				}
				// if coords are not present in parent-child map
				if (pc.find(p1) == pc.end()) {
					// add tile + coords to queue, add coords to pc map
					q.push(p1);
					pc[p1] = current_tile;
				}
			}
		}
		map_lock.unlock();
	}
	return dir;
}

bool Search::move(std::stack<std::uint8_t> &&path) {
	return move(path);
}

/**
 * @brief Moves the robot from the pos \ref tile to the target \ref tile
 * using the directions from search().
 * @param path The stack of directions.
 */
bool Search::move(std::stack<std::uint8_t> &path) {
	while (!path.empty()) {
		// convert n/s alignment from search to pos alignment
		std::uint8_t dir = (path.top() - cd + 4) % 4;
		std::cout << "moving " << (int) dir << " orig: " << path.top() << " cd: " << (int) cd << '\n';

		// std::cout << "oldX: " << oldX << " \noldZ: " << oldZ << '\n';

		// move in direction specified in stack - adjust for pos direction
		serial.write((std::uint8_t) (dir | (1 << 7)));
		switch(serial.read()){
			case 0: {
				auto x1 = x, y1 = y;
				switch (path.top()) {
					case Dir::N:
						--y1;
						break;
					case Dir::S:
						++y1;
						break;
					case Dir::E:
						++x1;
						break;
					case Dir::W:
						--x1;
						break;
				}

				for (std::uint8_t i = 0; i < 4; i++) {
					map[y1][x1][i] = true;
				}
				return true;
			}
			case 2:{
				return false;
			}
			default:
				break;
		}
		std::cout << "acked\n";

		// change pos direction if not N/S
		cd = (dir == Dir::S) ? cd : path.top();

		switch (path.top()) {
			case Dir::N:
				--y;
				break;
			case Dir::S:
				++y;
				break;
			case Dir::E:
				++x;
				break;
			case Dir::W:
				--x;
				break;
		}

		map_lock.lock();
		map[y][x].set_vis();
		map_lock.unlock();
		map_cv.notify_one();
		path.pop();
	}
	return true;
}

/**
 * @brief Checks for surrounding walls and uses this data to regenerate and
 * resize #map.
 */
void Search::check_walls() {
	std::lock_guard<std::mutex> guard(map_lock);

	std::uint32_t new_width = map.width(), new_length = map.length();
	// offsets if space must added above or left of array bounds
	std::int32_t x_offset = 0, y_offset = 0;
	for (std::uint8_t i = 0; i < 4; i++) {
		serial.write(i);
		std::uint16_t val = -1;
		serial.read(val);
		std::cout << "sensor " << (int) i << " val: " << val << '\n';
		if (val <= 200) {
			map[y][x][(i + cd) % 4] = true;
		}
	}
	// add to l or w if there is a tile beyond the array bounds
	new_length += (!map[y][x][Dir::E] && x == map.length() - 1);
	new_length += (!map[y][x][Dir::W] && x == 0);
	new_width += (!map[y][x][Dir::S] && y == map.width() - 1);
	new_width += (!map[y][x][Dir::N] && y == 0);
	// add to the offset if there is a tile beyond the left or top array bounds
	x_offset += (!map[y][x][Dir::W] && x == 0);
	y_offset += (!map[y][x][Dir::N] && y == 0);
	// regenerate map to reflect size changes
	map.resize(x_offset, y_offset, new_width, new_length);
	x += x_offset;
	y += y_offset;
	init_x += x_offset;
	init_y += y_offset;
}

/**
 * @brief Prints a visual representation of the maze as known by the robot.
 */
void Search::print_map() const {
	std::lock_guard<std::mutex> guard(map_lock);

	std::cout << "w: " << map.width() << " l: " << map.length() << '\n';
	for (std::uint32_t i = 0; i < map.length(); i++) {
		std::cout << ((map[0][i][Dir::N]) ? " _" : "  ");
	}
	std::cout << '\n';
	for (std::int32_t i = 0; i < map.width(); i++) {
		std::cout << ((map[i][0][Dir::W]) ? "|" : " ");
		for (std::int32_t j = 0; j < map.length(); j++) {
			char c = (i == y && j == x) ? 'X' : 'O';
			if (map[i][j].vis() && (map[i][j][Dir::S] || map.adj(i, j, Dir::S))) {
				std::cout << "\e[4m" << c << "\e[0m";
			} else if ((map[i][j][Dir::S] || map.adj(i, j, Dir::S)) && !map[i][j].vis()) {
				std::cout << '_';
			} else if (map[i][j].vis()) {
				std::cout << c;
			} else {
				std::cout << ' ';
			}
			std::cout << ((map[i][j][Dir::E] || map.adj(i, j, Dir::E)) ? '|' : ' ');
		}
		std::cout << '\n';
	}
}

bool Search::get_current_vic() const {
	return map[y][x].vic();
}

void Search::set_current_vic() {
	map[y][x].set_vic();
}

void Search::unmark_start() {
	map[init_y][init_x][Prop::VIS] = false;
}

void ostream_writev(std::ostream &os) {}

template<class T, class ...Args>
void ostream_writev(std::ostream &os, const T &val, const Args &...args) {
	os.write(reinterpret_cast<const char *>(&val), sizeof(val));
	ostream_writev(os, args...);
}

void Search::dump_map() {
	std::ofstream out(filename, std::ios::binary);
	const auto width = map.width(), length = map.length();
	ostream_writev(out, x, y, init_x, init_y,cd, width, length);
	const auto *buf = map.buf();
	out.write(reinterpret_cast<const char *>(buf), width * length * sizeof(*buf));
}

bool Search::get_current_vis(){
	return map[y][x].vis();
}