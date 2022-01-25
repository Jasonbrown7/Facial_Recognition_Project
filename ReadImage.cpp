#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <stdio.h>

using namespace std;

#include "image.h"

void readImage(char fname[], ImageType& image)
{
 int i, j;
 int N, M, Q;
 unsigned char *charImage;
 char header [100], number[2], max[3], *ptr;
 ifstream ifp;

 ifp.open(fname, ios::in | ios::binary);

 if (!ifp) {
   cout << "Can't read image: " << fname << endl;
   exit(1);
 }

 // read header

 ifp.getline(header,100,'\n');
 if ( (header[0]!=80) ||    /* 'P' */
      (header[1]!=53) ) {   /* '5' */
      cout << "Image " << fname << " is not PGM" << endl;
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

 charImage = (unsigned char *) new unsigned char [M*N];

 ifp.read( reinterpret_cast<char *>(charImage), (M*N)*sizeof(unsigned char));

 if (ifp.fail()) {
   cout << "Image " << fname << " has wrong size" << endl;
   cout << "N: " << N << " M: " << M << endl;
   exit(1);
 }

 ifp.close();

 //
 // Convert the unsigned characters to integers
 //

 int val;

 for(i=0; i<N; i++)
   for(j=0; j<M; j++) {
     val = (int)charImage[i*M+j];
     image.setPixelVal(i, j, val);     
   }

 delete [] charImage;

}