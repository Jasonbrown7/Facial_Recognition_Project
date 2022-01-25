#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <stdio.h>

using namespace std;

#include "image.h"

void readImageHeader(char fname[], int& N, int& M, int& Q, bool& type)
{
 char header [100], number[2], max[3], *ptr;
 ifstream ifp;

 ifp.open(fname, ios::in | ios::binary);

 if (!ifp) {
   cout << "Can't read image: " << fname << endl;
   exit(1);
 }

 // read header

 type = false; // PGM

 ifp.getline(header,100,'\n');
 if ( (header[0] == 80) &&  /* 'P' */
      (header[1]== 53) ) {  /* '5' */
      type = false;
 }
 else if ( (header[0] == 80) &&  /* 'P' */
      (header[1] == 54) ) {        /* '6' */
      type = true;
 } 
 else {
   cout << "Image " << fname << " is not PGM or PPM" << endl;
   exit(1);
 }

 number[0] = header[3];
 number[1] = header[4];
 M=strtol(number,&ptr,0);
 
 number[0] = header[6];
 number[1] = header[7];
 N=atoi(number);

 //ifp.getline(header,100,'\n');
 max[0] = header[9];
 max[1] = header[10];
 max[2] = header[11];
 Q=strtol(max,&ptr,0);

 ifp.close();
}