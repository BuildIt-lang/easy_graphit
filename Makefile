-include Makefile.inc
BASE_DIR=$(shell pwd)
SRC_DIR=$(BASE_DIR)/src
BUILD_DIR?=$(BASE_DIR)/build
INCLUDE_DIR=$(BASE_DIR)/include
BUILDIT_DIR?=$(BASE_DIR)/buildit


SAMPLES_DIR=$(BASE_DIR)/samples
APPS_DIR=$(BASE_DIR)/apps

INCLUDES=$(wildcard $(INCLUDE_DIR)/*.h) $(wildcard $(INCLUDE_DIR)/*/*.h) $(wildcard $(BUILDIT_DIR)/include/*.h) $(wildcard $(BUILDIT_DIR)/include/*/*.h)

INCLUDE_FLAG=-I$(INCLUDE_DIR) -I$(BUILDIT_DIR)/include

SRCS=$(wildcard $(SRC_DIR)/*.cpp) $(wildcard $(SRC_DIR)/*/*.cpp)
SAMPLES_SRCS=$(wildcard $(SAMPLES_DIR)/*.cpp)
OBJS=$(subst $(SRC_DIR),$(BUILD_DIR),$(SRCS:.cpp=.o))
SAMPLES=$(subst $(SAMPLES_DIR),$(BUILD_DIR),$(SAMPLES_SRCS:.cpp=))

APPS_SRCS=$(wildcard $(APPS_DIR)/*.cpp)
APPS=$(subst $(APPS_DIR),$(BUILD_DIR)/apps,$(APPS_SRCS:.cpp=))

$(shell mkdir -p $(BUILD_DIR))
$(shell mkdir -p $(BUILD_DIR)/pipeline)
$(shell mkdir -p $(BUILD_DIR)/graphit)
$(shell mkdir -p $(BUILD_DIR)/samples)
$(shell mkdir -p $(BUILD_DIR)/apps)

BUILDIT_LIBRARY_NAME=buildit
BUILDIT_LIBRARY_PATH=$(BUILDIT_DIR)/build

LIBRARY_NAME=graphit
DEBUG ?= 0
ifeq ($(DEBUG),1)
CFLAGS=-g -std=c++11 -O0
LINKER_FLAGS=-rdynamic  -g -L$(BUILDIT_LIBRARY_PATH) -L$(BUILD_DIR) -l$(LIBRARY_NAME) -l$(BUILDIT_LIBRARY_NAME)
else
CFLAGS=-std=c++11 -O3
LINKER_FLAGS=-rdynamic  -L$(BUILDIT_LIBRARY_PATH) -L$(BUILD_DIR) -l$(LIBRARY_NAME) -l$(BUILDIT_LIBRARY_NAME)
endif



LIBRARY=$(BUILD_DIR)/lib$(LIBRARY_NAME).a

CFLAGS+=-Wall -Wextra -Wno-unused-parameter -Wno-missing-field-initializers -Wmissing-declarations -Woverloaded-virtual -pedantic-errors -Wno-deprecated -Wdelete-non-virtual-dtor -Werror

all: executables 

.PHONY: subsystem
subsystem:
	make -C $(BUILDIT_DIR)

.PRECIOUS: $(BUILD_DIR)/samples/%.o
.PRECIOUS: $(BUILD_DIR)/apps/%.o
.PRECIOUS: $(BUILD_DIR)/pipeline/%.o
.PRECIOUS: $(BUILD_DIR)/graphit/%.o

$(BUILD_DIR)/samples/%.o: $(SAMPLES_DIR)/%.cpp $(INCLUDES)
	$(CXX) $(CFLAGS) $< -o $@ $(INCLUDE_FLAG) -c 

$(BUILD_DIR)/apps/%.o: $(APPS_DIR)/%.cpp $(INCLUDES)
	$(CXX) $(CFLAGS) $< -o $@ $(INCLUDE_FLAG) -c 

.PHONY: $(BUILDIT_LIBRARY_PATH)/lib$(BUILDIT_LIBRARY_NAME).a
$(BUILD_DIR)/sample%: $(BUILD_DIR)/samples/sample%.o $(LIBRARY) $(BUILDIT_LIBRARY_PATH)/lib$(BUILDIT_LIBRARY_NAME).a subsystem
	$(CXX) -o $@ $< $(LINKER_FLAGS)

$(BUILD_DIR)/apps/%: $(BUILD_DIR)/apps/%.o $(LIBRARY) $(BUILDIT_LIBRARY_PATH)/lib$(BUILDIT_LIBRARY_NAME).a subsystem
	$(CXX) -o $@ $< $(LINKER_FLAGS)

.PHONY: executables
executables: $(SAMPLES) $(APPS)

$(LIBRARY): $(OBJS)
	ar rv $(LIBRARY) $(OBJS)	
	
$(BUILD_DIR)/pipeline/%.o: $(SRC_DIR)/pipeline/%.cpp $(INCLUDES)
	$(CXX) $(CFLAGS) $< -o $@ $(INCLUDE_FLAG) -c
$(BUILD_DIR)/graphit/%.o: $(SRC_DIR)/graphit/%.cpp $(INCLUDES)
	$(CXX) $(CFLAGS) $< -o $@ $(INCLUDE_FLAG) -c

run: executables
	./build/sample1
	./build/sample2
	./build/sample3
	./build/sample4

clean:
	rm -rf $(BUILD_DIR)
