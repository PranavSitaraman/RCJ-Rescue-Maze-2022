#include "camera.hpp"
#include <array>
#include <iostream>
#include <chrono>
#include <filesystem>
#include <numeric>
#include <fstream>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>
#include <AS726X.h>
#include "search.hpp"
#include <Wire.h>
#include "Serial.hpp"

namespace fs = std::filesystem;

constexpr auto CANNY_ON = true;
constexpr auto IM_DEBUG = false;

constexpr auto SIZE_THRESH = 60;
constexpr auto SLICE_SIZE_THRESH = 50;
constexpr auto CONT_SIZE_THRESH = 20000;

namespace Letter {
	enum letter : std::uint8_t {
		H, S, U, UNKNOWN
	};
}

namespace Color {
	enum color : std::uint8_t {
		RED, YELLOW, GREEN, UNKNOWN
	};
}

Color::color color_detect(const cv::Mat &frame);

Letter::letter letter_detect(cv::Mat &frame);

bool heat_detect(std::uint16_t address);

void detect(std::atomic<ThreadState> &state, Search** search, std::mutex &map_lock, std::condition_variable &map_cv) {
	Wire.begin();

	std::string port;
	for (const auto &entry: fs::directory_iterator("/sys/class/tty")) {
		const auto &filename = entry.path().filename();
		if (filename.generic_string().rfind("ttyAMA", 0) == 0) {
			port = "/dev/" / filename;
			break;
		}
	}

	Serial serial(port, 9600);

	AS726X color;
	color.begin(Wire, 3, 2);
	color.setIntegrationTime(1);
	color.enableBulb();


	std::array<cv::VideoCapture, 2> caps{cv::VideoCapture(0,cv::CAP_V4L2), cv::VideoCapture(1,cv::CAP_V4L2)};
	//check that opencv version is 3.4+
	for(auto& cap : caps){
		cap.set(cv::CAP_PROP_BUFFERSIZE,1);
		cap.set(cv::CAP_PROP_FRAME_WIDTH,320);
		cap.set(cv::CAP_PROP_FRAME_HEIGHT,240);
	}

	state = ThreadState::STARTED;

	constexpr std::array<std::uint16_t, 2> heat_addrs{0x5a, 0x5b};
	cv::Mat frame;
	while (state != ThreadState::STOP) {
		std::unique_lock<std::mutex> cond_lock(map_lock);
		map_cv.wait(cond_lock,
		            [&search, &state] {
			            return !(*search)->get_current_vic() || state == ThreadState::STOP;
		            });
		cond_lock.unlock();
		for (std::uint8_t i = 0; i < caps.size() && state != ThreadState::STOP; i++) {
			caps[i] >> frame;
			std::cout << "thread running\n";

			std::uint8_t n_kits = 0;
			bool vic = false;

			//auto start = std::chrono::high_resolution_clock::now();
			switch (color_detect(frame)) {
				case Color::RED:
					std::cout << "red\n";
					n_kits = 1;
					break;
				case Color::YELLOW:
					std::cout << "yellow\n";
					n_kits = 1;
					break;
				case Color::GREEN:
					std::cout << "green\n";
					vic = true;
					break;
				case Color::UNKNOWN:
					break;
			}
			/*
			auto elapsed = std::chrono::high_resolution_clock::now() - start;
			std::cout << "color vic detection takes: "
			          << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count() << "ms\n";
			*/

			//start = std::chrono::high_resolution_clock::now();
			switch (letter_detect(frame)) {
				case Letter::H:
					std::cout << "H\n";
					n_kits = 3;
					break;
				case Letter::S:
					std::cout << "S\n";
					n_kits = 2;
					break;
				case Letter::U:
					std::cout << "U\n";
					vic = true;
					break;
				case Letter::UNKNOWN:
					break;
			}
			/*
			elapsed = std::chrono::high_resolution_clock::now() - start;
			std::cout << "letter detection takes: "
			          << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count() << "ms\n";
			*/

			//start = std::chrono::high_resolution_clock::now();
			if (heat_detect(heat_addrs[i])) {
				n_kits++;
				std::cout << "heat\n";
			}
			/*
			elapsed = std::chrono::high_resolution_clock::now() - start;
			std::cout << "heat detection takes: "
			          << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count() << "ms\n";
			*/

			if (n_kits || vic) {
				//start = std::chrono::high_resolution_clock::now();

				cond_lock.lock();
				(*search)->set_current_vic();
				cond_lock.unlock();

				serial.write(static_cast<std::uint8_t>(0));
				serial.write(n_kits);
				serial.write(i);

				/*
				elapsed = std::chrono::high_resolution_clock::now() - start;
				std::cout << "serial takes: " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count()
				          << "ms\n";
				          */
			}


			//start = std::chrono::high_resolution_clock::now();

			try {
				if (color.takeMeasurements()) {
					auto red = color.getCalibratedRed();
					auto green = color.getCalibratedGreen();
					auto blue = color.getCalibratedBlue();

					constexpr auto BLACK_UPPER_R = 5;
					constexpr auto BLACK_UPPER_G = 5;
					constexpr auto BLACK_UPPER_B = 5;

					constexpr auto SILVER_LOWER_R = 10;
					constexpr auto SILVER_LOWER_G = 40;
					constexpr auto SILVER_LOWER_B = 30;

					//if black
					if (red < BLACK_UPPER_R && green < BLACK_UPPER_G && blue < BLACK_UPPER_B) {
						std::cout << "hole\n";
						serial.write(static_cast<std::uint8_t>(1));
						serial.read();
					}

					//if silver
					cond_lock.lock();
					if ((*search)->get_current_vis() &&
					    (red > SILVER_LOWER_R && green > SILVER_LOWER_G && blue > SILVER_LOWER_B)) {
						cond_lock.unlock();
						std::cout << "saving map\n";
						(*search)->dump_map();
					} else {
						cond_lock.unlock();
					}
					/*elapsed = std::chrono::high_resolution_clock::now() - start;
					std::cout << "color tile detection takes: "
					          << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count() << "ms\n";*/
				}
			}catch(std::exception& e){

			}
		}
	}
	color.disableBulb();
}

Color::color color_detect(const cv::Mat &frame) {
	static const std::array<cv::Scalar, 6> bounds{
			cv::Scalar(100, 118, 88), cv::Scalar(200, 255, 255),
			cv::Scalar(14, 107, 129), cv::Scalar(40, 188, 255),
			cv::Scalar(55, 42, 25), cv::Scalar(90, 172, 80)};

	auto color_ratio = 1 / 10.;
	auto color = Color::UNKNOWN;

	cv::cvtColor(frame, frame, cv::COLOR_BGR2HSV);

	cv::Mat filt_frame;
	for (std::uint8_t i = 0; i < bounds.size() && color == Color::UNKNOWN; i += 2) {

		if constexpr (IM_DEBUG) {
			cv::imshow("hsv", frame);
			cv::waitKey(1);
		}

		cv::inRange(frame, bounds[i], bounds[i + 1], filt_frame);

		auto nonzero = cv::countNonZero(filt_frame);
		auto size = filt_frame.cols * filt_frame.rows;

		if constexpr (IM_DEBUG) {
			cv::imshow("bw", filt_frame);
			cv::waitKey(1);
			std::cout << "ratio: " << (nonzero / std::gcd(nonzero, size)) << '/' << (size / std::gcd(nonzero, size))
			          << '\n';
		}
		if (double cur_ratio;(cur_ratio = nonzero / static_cast<double>(size)) > color_ratio) {
			color = static_cast<Color::color>(i / 2);
			color_ratio = cur_ratio;
		}
	}
	cv::cvtColor(frame,frame,cv::COLOR_HSV2BGR);

	return color;
}


Letter::letter letter_detect(cv::Mat &frame) {
	//image is upside down - rotate
	cv::rotate(frame, frame, cv::ROTATE_180);

	std::array<int, 3> letterCount{};
	if constexpr(IM_DEBUG) {
		cv::imshow("win_u", frame);
		cv::waitKey(1);
	}

	cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
	cv::GaussianBlur(frame, frame, cv::Size(5, 5), 0);
//either threshold -> canny -> draw or adaptive -> draw
//adaptive -> draw is slightly more prone to noise
//benchmarked - canny is way faster, more than double the speed
	if constexpr(CANNY_ON) {
		cv::threshold(frame, frame, 0, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);
		cv::Canny(frame, frame, 50, 110);
	} else {
		cv::adaptiveThreshold(frame, frame, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY_INV, 11, 2);
	}
	std::vector<std::vector<cv::Point>> contours;
//get contours
	cv::findContours(frame, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
//fill in contours (necessary as canny/adaptive only does outline)
	cv::drawContours(frame, contours, -1, cv::Scalar(255, 255, 255), -1);
//iterate through contours
	for (const auto &contour: contours) {
		auto rect = cv::boundingRect(contour);
		auto rw = rect.width;
		auto rh = rect.height;
		auto angle = cv::minAreaRect(contour).angle;
//exclude small contours
		if (rw < SIZE_THRESH || rh < SIZE_THRESH) {
			continue;
		}
//get bounding rect of letter
		auto letter = frame(rect);
//rotate letter by specified angle
		auto mat = cv::getRotationMatrix2D(cv::Point(letter.cols / 2, letter.rows / 2), angle, 1);
		cv::warpAffine(letter, letter, mat, cv::Size(letter.cols, letter.rows), cv::INTER_CUBIC);
		auto area = cv::contourArea(contour);

		if(area <= 0){
			continue;
		}

		std::vector<std::vector<cv::Point>> letterContours;
		cv::findContours(letter,letterContours,cv::RETR_EXTERNAL,cv::CHAIN_APPROX_SIMPLE);
		std::vector<cv::Point> largestContour = letterContours[0];

		for(auto& c : letterContours){
			if(cv::contourArea(c) > cv::contourArea(largestContour)){
				largestContour = c;
			}
		}
		letter = letter(cv::boundingRect(largestContour));

		if (letter.cols > letter.rows) {
			cv::rotate(letter, letter, cv::ROTATE_90_CLOCKWISE);
		}

		auto sliceSize = letter.rows/3;
		std::array<cv::Mat,3> slices{letter(cv::Rect(0,0,letter.cols,sliceSize)),
		                             letter(cv::Rect(0,sliceSize,letter.cols,sliceSize)),
		                             letter(cv::Rect(0,2*sliceSize,letter.cols,sliceSize))};
		std::array<int,3> sliceContours{};
		constexpr std::array<int,3> h{2,1,2};
		constexpr std::array<int,3> s{1,1,1};
		constexpr std::array<int,3> u{2,2,1};
		constexpr std::array<int,3> u_rot{1,2,2};

		for(int i = 0;i<slices.size();i++){
			cv::findContours(slices[i],letterContours,cv::RETR_EXTERNAL,cv::CHAIN_APPROX_SIMPLE);
			int count = 0;
			for(auto& sliceContour : letterContours){
				auto sliceRect = cv::minAreaRect(sliceContour);
				if(sliceRect.size.width > SLICE_SIZE_THRESH){
					count++;
				}
			}
			sliceContours[i] = count;
		}

		if(sliceContours == h){
			letterCount[Letter::H]++;
		}else if(sliceContours == s){
			letterCount[Letter::S]++;
		}else if(sliceContours == u || sliceContours == u_rot){
			letterCount[Letter::U]++;
		}
	}

//choose best guess
	auto max = std::max_element(letterCount.begin(), letterCount.end());
	if (*max == 0) {
		return Letter::UNKNOWN;
	}
	return static_cast<Letter::letter>(max - letterCount.begin());
}

bool heat_detect(std::uint16_t address) {
	try {
		Wire.beginTransmission(address);
		Wire.write(0x06);
		Wire.endTransmission(false);
		Wire.requestFrom(address, 3, false);
		double ambient = (Wire.read() | (Wire.read() << 8)) * .02 - 273.15;
		Wire.read();

		Wire.beginTransmission(address);
		Wire.write(0x07);
		Wire.endTransmission(false);

		Wire.requestFrom(address, 3, false);
		double obj = (Wire.read() | (Wire.read() << 8)) * .02 - 273.15;
		Wire.read();

		if(ambient > 50 || obj > 50){
			return false;
		}

		std::cout << "obj: " << obj << " ambient: " << ambient << '\n';
		return (obj - ambient) >= 10;
	}catch(std::exception& e){
		return false;
	}
}
