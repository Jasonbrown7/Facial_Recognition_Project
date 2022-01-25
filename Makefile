CFLAGS = -g -Wall -Wno-deprecated

all: clean main

main: image.h image.o ReadImage.o ReadImageHeader.o WriteImage.o \
	main.cpp
	g++ -o main $(CFLAGS) image.o ReadImage.o ReadImageHeader.o \
					WriteImage.o main.cpp

ReadImage.o:	image.h ReadImage.cpp
	g++ -c $(CFLAGS) ReadImage.cpp

ReadImageHeader.o:	image.h ReadImageHeader.cpp
	g++ -c $(CFLAGS) ReadImageHeader.cpp

WriteImage.o:	image.h WriteImage.cpp
	g++ -c $(CFLAGS) WriteImage.cpp

image.o:	image.h image.cpp
	g++ -c $(CFLAGS) image.cpp

clean:
	rm -f main *.o