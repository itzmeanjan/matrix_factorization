CXX = dpcpp
CXXFLAGS = -std=c++17
SYCLFLAGS = -fsycl
SOURCES = $(wildcard *.cpp)
HEADERS = $(wildcard include/*.hpp)
INCLUDES = -I./include
PROG = a.out

$(PROG): $(SOURCES) $(HEADERS)
	$(CXX) $(CXXFLAGS) $(SYCLFLAGS) $^ $(INCLUDES)
