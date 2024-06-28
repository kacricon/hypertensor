CC = gcc
CFLAGS = -Wall -g

# Targets
TARGET = test_tensor
SOURCES = tensor.c test_tensor.c
HEADERS = tensor.h

# Default target
all: $(TARGET)

# Build the test executable
$(TARGET): $(SOURCES) $(HEADERS)
	$(CC) $(CFLAGS) -o $(TARGET) $(SOURCES)

# Clean up build files
clean:
	rm -f $(TARGET)
	rm -rf $(TARGET).dSYM
	rm -f *.o

