#include <iostream>
#include <fstream>
#include <vector>
#include <math.h>
#include <string>
#include <queue>
#include <sys/stat.h>
#include <dirent.h>
#include <sstream>
#include <limits>	// numeric_limits
#include "eigen3/Eigen/Dense"
#include "eigen3/Eigen/Eigenvalues" // used to decompose matricies
#include "image.h"

using Eigen::VectorXd;
using Eigen::VectorXi;
using Eigen::VectorXcf;
using Eigen::MatrixXd;
using Eigen::MatrixXi;
using Eigen::MatrixXcd;
using Eigen::MatrixXcf;
using Eigen::EigenSolver;
using namespace std;

///////////////////////// THIS IS THE NEW FILE ///////////////////////////////////////////
const int ID_LENGTH = 5;
const int DATE_LENGTH = 6;
const int SET_LENGTH = 2;
const int IMG_W = 16, IMG_H = 20, IMG_VEC_LEN = IMG_H * IMG_W;
const int NUM_SAMPLES = 1204; // 1196
const int NUM_TEST_SAMPLES = 1196;

struct EigenValVecPair
{
	double eigenValue;
	VectorXd eigenVector;
};

struct by_eigenValue
{ 
    bool operator()(EigenValVecPair const &a, EigenValVecPair const &b) const 
    { 
        return abs(a.eigenValue) > (b.eigenValue);
    }
};

class Image
{
public:
	void ExtractEigenVectors(MatrixXd m, int dimension1, int dimension2);

	// Setters
	void SetIdentifier(int input) { identifier = input; };
	void SetDateTaken(int input) { dateTaken = input; };
	void SetDataset(string input) { dataset = input; };
	void SetWearingGlasses(bool input) { wearingGlasses = input; };
	void SetFileName(string input) { fileName = input; };
	void setFaceVector(VectorXi vector) { faceVector = vector; };

	// Getters
	int GetIdentifier() { return identifier; };
	int GetDateTaken() { return dateTaken; };
	string GetFileName() { return fileName; };
	string GetDataset() { return dataset; };
	bool GetWearingGlasses() { return wearingGlasses; };
	VectorXi getFaceVector() const { return faceVector; };

private:
	string fileName;
	int identifier; //nnnnn
	int dateTaken; //yymmdd
	string dataset;
	bool wearingGlasses;

	VectorXi faceVector;
	MatrixXcd eigenVectors;
	MatrixXcd eigenValues;
};

void readImageHeader(char[], int&, int&, int&, bool&);
void readImage(char[], ImageType&);
void writeImage(char[], ImageType&);
void PrintMatrix(MatrixXd m, int dimension1, int dimension2);
vector<Image> obtainTrainingFaces(string directory, int imageWidth, int imageHeight);
VectorXi compAvgFaceVec(const vector<Image> &imageVector);
void getMinMax(VectorXd,double&,double&);

int main()
{
	string inputString;
	string trainingDataset = "Faces_FA_FB/fa_L",
		   testingDataset = "Faces_FA_FB/fb_L",
		   eigenvaluesFile = "eigenValues.txt",
		   eigenvectorsFile = "eigenVectors.txt",
		   imageCoefficientsFile = "imageCoefficients.txt";
	vector<Image> ImageVector;
	vector<Image> TestImageVector;
	VectorXi avgFaceVector;
	vector<VectorXi> phi;
	vector<VectorXi> phiTesting;
	unsigned K = 0;
	MatrixXd A(IMG_VEC_LEN, NUM_SAMPLES);
	MatrixXd C(IMG_VEC_LEN, IMG_VEC_LEN);
	MatrixXd eigenVectors_u, eigenVectors_v;
	VectorXd eigenValues;
	vector<EigenValVecPair> EigenValVecPairs;

	do
	{
		cout << endl
		     << "+==============================================================+\n"
			 << "|Select  0 to obtain training images (I_1...I_M)               |\n"
			 << "|Select  1 to compute average face vector (Psi)                |\n"
			 << "|Select  2 to compute matrix A ([Phi_i...Phi_M])               |\n"
			 << "|Select  3 to compute the eigenvectors/values of A^TA          |\n"
			 << "|Select  4 to project training eigenvalues (req: 0,1,2)        |\n"
			 << "|Select  5 to visualize the 10 largest & smallesteigenvectors  |\n"
		     << "|Select  6 to run facial recognition on fb_H                   |\n"
		     << "|Select  7 to run facial recognition on fb_H against fa2_H     |\n"
		     << "|Select -1 to exit                                             |\n"
		     << "+==============================================================+\n"
		     << endl
		     << "Choice: ";

		cin >> inputString;
		if (inputString == "0") //Initialize images
		{
			ImageVector = obtainTrainingFaces(trainingDataset, IMG_W, IMG_H);
		}
		else if (inputString == "1") //Generate Psi
		{
			avgFaceVector = compAvgFaceVec(ImageVector);
			ImageType avgFaceImg(IMG_H, IMG_W, 255);

			for (int i = 0; i < IMG_H; i++)
			{
				for (int j = 0; j < IMG_W; j++)
				{
					int val = avgFaceVector(i * IMG_W + j);
					avgFaceImg.setPixelVal(i, j, val);
				}
			}

			writeImage((char*) "myAvgFaceImg.pgm", avgFaceImg);

			cout << "Average face vector (Psi) has been computed." << endl;
			cout << "Image of average face can be found in file titled: \"myAvgFaceImg.pgm\"" << endl;
		}
		else if (inputString == "2") // Generate Phi
		{
			for (int j = 0; j < NUM_SAMPLES; j++)
			{
				phi.push_back(ImageVector[j].getFaceVector() - avgFaceVector);

				for (int i = 0; i < IMG_VEC_LEN; i++)
				{
					A(i, j) = phi[j](i);
				}
			}
		}
		else if (inputString == "3") //Generate eigenvalues/vectors
		{
			MatrixXd AT_A(NUM_SAMPLES, NUM_SAMPLES);
			AT_A = A.transpose() * A;

			EigenSolver<MatrixXd> es(AT_A);
			cout << "Computing eigenvalues..." << endl;
			eigenValues = es.eigenvalues().real();
			cout << "Finished computing eigenvalues!" << endl;

			cout << "Computing eigenvectors..." << endl;
			eigenVectors_v = es.eigenvectors().real();
			eigenVectors_u = A * eigenVectors_v;
			eigenVectors_u.colwise().normalize();
			cout << "Finished computing eigenvectors!" << endl;

			for (int i = 0; i < eigenValues.rows(); i++)
			{
				EigenValVecPair pair;
				pair.eigenValue = eigenValues(i);
				pair.eigenVector = eigenVectors_u.col(i);
				EigenValVecPairs.push_back(pair);
			}
			cout << "cp1" << endl;
			//std::sort(EigenValVecPairs.begin(), EigenValVecPairs.end(), by_eigenValue());
			cout << "cp1.5" << endl;
			
			ofstream fout_vals,
				fout_vecs;

			fout_vals.open(eigenvaluesFile.c_str());
			fout_vecs.open(eigenvectorsFile.c_str());
			cout << "cp2" << endl;

			for (unsigned i = 0; i < EigenValVecPairs.size(); i++)
			{
				fout_vals << EigenValVecPairs[i].eigenValue;
				fout_vecs << EigenValVecPairs[i].eigenVector.transpose();

				if (i < EigenValVecPairs.size() - 1) // if we're not on last iteration
				{
					fout_vals << endl;
					fout_vecs << endl;
				}
			}
			cout << "cp3" << endl;
			fout_vals.close();
			fout_vecs.close();
			cout << "cp4" << endl;
		}
		else if (inputString == "4") // Generate Omega based off threshold
		{
			double threshold, currentEigenValueNum = 0, totalEigenValueNum = 0;
			cout << "Select threshold value (0 to 1): ";
			cin >> threshold;
			vector<double> topEigenValues;
			vector<VectorXd> topEigenVectors;
			ifstream fin_vals, fin_vecs;
			double eigenValue;
			VectorXd eigenVector(IMG_VEC_LEN);

			fin_vals.open(eigenvaluesFile.c_str());
			while(!fin_vals.eof())
			{
				fin_vals >> eigenValue;
				topEigenValues.push_back(eigenValue);
			}
			fin_vals.close();

			fin_vecs.open(eigenvectorsFile.c_str());
			while(!fin_vecs.eof())
			{
				for (unsigned i = 0; i < IMG_VEC_LEN; i++)
				{
					fin_vecs >> eigenVector(i);
				}
				
				topEigenVectors.push_back(eigenVector);
			}
			fin_vecs.close();

			for (unsigned i = 0; i < topEigenValues.size(); ++i)
			{
				totalEigenValueNum += topEigenValues[i];
			}

			for (unsigned i = 0; i < topEigenValues.size(); ++i)
			{
				currentEigenValueNum += topEigenValues[i];
				if((currentEigenValueNum/totalEigenValueNum) >= threshold)
				{
					cout << "Found K threshold to save " << threshold << " of info @ K = " << i << endl;
					K = i;
					break;
				}
			}

			topEigenValues.erase(topEigenValues.begin() + K, topEigenValues.end());
			topEigenVectors.erase(topEigenVectors.begin() + K, topEigenVectors.end());

			string omegaVectorsFile = "omegaVectors.txt";
			ofstream fout_omega_vecs(omegaVectorsFile.c_str());

			fout_omega_vecs << K << endl;

			VectorXd phi_hat = VectorXd::Zero(IMG_VEC_LEN);

			for (unsigned i = 0; i < phi.size(); i++)
			{
				for (unsigned j = 0; j < K; j++)
				{
					MatrixXd u_t = ((MatrixXd)topEigenVectors[j]).transpose();
					MatrixXd phi_i = (phi[i]).cast<double>();
					double w = (u_t * phi_i)(0, 0);
					fout_omega_vecs << w;

					if (j < K - 1)	// if not last element
						fout_omega_vecs << " ";

					if (i == 0)
					{
						phi_hat += w * topEigenVectors[j];
					}
				}

				if (i < phi.size() - 1)	// if not last element
					fout_omega_vecs << endl;
			}

			phi_hat += avgFaceVector.cast<double>();
			VectorXi phi_hat_int = phi_hat.cast<int>();

			ImageType reconstructedFaceImg(IMG_H, IMG_W, 255);

			for (int i = 0; i < IMG_H; i++)
			{
				for (int j = 0; j < IMG_W; j++)
				{
					int val = phi_hat_int(i * IMG_W + j);
					reconstructedFaceImg.setPixelVal(i, j, val);
				}
			}
			
			writeImage((char*) "reconstructedFaceImg.pgm", reconstructedFaceImg);

			fout_omega_vecs.close();
		}
		else if (inputString == "5") // Generate 10 largest/smallest eigenvectors
		{
			vector<VectorXd> topEigenVectors;
			ifstream fin_vecs;
			VectorXd eigenVector(IMG_VEC_LEN);

			fin_vecs.open(eigenvectorsFile.c_str());
			while(!fin_vecs.eof())
			{
				for (unsigned i = 0; i < IMG_VEC_LEN; i++)
				{
					fin_vecs >> eigenVector(i);
				}
				
				topEigenVectors.push_back(eigenVector);
			}
			fin_vecs.close();


			for (int i = 0; i < 10; ++i)
			{
				double min,max;
				getMinMax(topEigenVectors[i],min,max);
				cout << "index " << i << " min: " << min << endl;
				cout << "index " << i << " max: " << max << endl;
				ImageType eigenFace(IMG_H, IMG_W, 255);
				for (int j = 0; j < IMG_H; j++)
				{
					for (int k = 0; k < IMG_W; k++)
					{
						
						double val = (topEigenVectors[i](j * IMG_W + k) - min) * (255 / (max - min));
						val += avgFaceVector(j * IMG_W + k);
						//cout << val << endl;
						eigenFace.setPixelVal(j, k, (int)val);
					}
				}
				
				string folder = "largestEigenFaces/";
				cout << "Largest 10 EigenFaces generated in folder: \"largestEigenFaces\"" << endl;
				mkdir(folder.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
				
				stringstream ss;
				ss << i;
				string str = ss.str();
				folder += str;
				folder += ".pgm";
				
				ofstream fout;
				fout.open(folder.c_str());
				fout.close();

				writeImage((char*) folder.c_str(), eigenFace);
			}
			
			vector<VectorXd> temp = topEigenVectors;
			temp.erase(temp.begin(), temp.end()-10);
			for (int i = 0; i < 10; ++i)
			{
				double min,max;
				getMinMax(temp[i],min,max);
				cout << "index " << i << " min: " << min << endl;
				cout << "index " << i << " max: " << max << endl;
				ImageType eigenFace(IMG_H, IMG_W, 255);
				for (int j = 0; j < IMG_H; j++)
				{
					for (int k = 0; k < IMG_W; k++)
					{
						
						double val = (temp[i](j * IMG_W + k) - min) * (255 / (max - min));
						val += avgFaceVector(j * IMG_W + k);
						//cout << val << endl;
						eigenFace.setPixelVal(j, k, (int)val);
					}
				}
				
				string folder = "smallestEigenFaces/";
				cout << "Smallest 10 EigenFaces generated in folder: \"smallestEigenFaces\"" << endl;
				mkdir(folder.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
				
				stringstream ss;
				ss << i;
				string str = ss.str();
				folder += str;
				folder += ".pgm";
				
				ofstream fout;
				fout.open(folder.c_str());
				fout.close();

				writeImage((char*) folder.c_str(), eigenFace);
			}
		}
		else if (inputString == "6")
		{
			TestImageVector = obtainTrainingFaces(testingDataset, IMG_W, IMG_H);

			//cout << "avgFaceVector->rows: "<< avgFaceVector.rows() << " avgFaceVector->cols: "<< avgFaceVector.cols();
			for (int i = 0; i < NUM_TEST_SAMPLES; ++i)
			{
				phiTesting.push_back(TestImageVector[i].getFaceVector() - avgFaceVector);
			}

			//LOAD DATASET'S OMEGA VECTORS
			vector<VectorXd> omegaVectors;
			string omegaVectorsFile = "omegaVectors.txt";
			ifstream fin_omegas(omegaVectorsFile.c_str());

			fin_omegas >> K; // read the K value from the file

			VectorXd readVector(K);

			for (int i = 0; i < NUM_SAMPLES; ++i)
			{
				for (unsigned j = 0; j < K; ++j)
				{
					fin_omegas >> readVector(j);
				}
				omegaVectors.push_back(readVector);
			}
			fin_omegas.close();
			//DONE

			//LOAD TOP K EIGENVECTORS
			vector<VectorXd> topEigenVectors;
			ifstream fin_vecs;
			VectorXd eigenVector(IMG_VEC_LEN);

			fin_vecs.open(eigenvectorsFile.c_str());
			unsigned k = 0;
			while(!fin_vecs.eof() && k < K)
			{
				for (unsigned i = 0; i < IMG_VEC_LEN; i++)
				{
					fin_vecs >> eigenVector(i);
				}
				
				topEigenVectors.push_back(eigenVector);
				k++;
			}
			fin_vecs.close();
			//DONE

			string folder = "Recognition/";
			cout << "Data stored in folder: \"Recognition\"" << endl;

			//cout << "folder->" << folder << endl;
			mkdir(folder.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);

			ofstream fout_results;
			string recognitionFile = folder + "recognitionResults.txt";
			fout_results.open(recognitionFile.c_str());

			ofstream fn1TP, fn1FP;
			string fn1TPfile = "n1TP.txt";
			string fn1FPfile = "n1FP.txt";
			fn1TP.open(fn1TPfile.c_str());
			fn1FP.open(fn1FPfile.c_str());

			//HEADER OF OUTPUT FILE
			fout_results << "N_value CorrectlyIdentified IncorrectlyIdentified Performance" << endl;

			cout << "Beginning face recognition sequence (N from 0-50)..." << endl;
			for (int N = 1; N < 51; ++N)
			{
				cout << "COMPUTING N=" << N << " PERFORMANCE..." << endl;

				int correctlyIdentifiedImages = 0;
				int incorrectlyIdentifiedImages = 0;

				//RUN RECOGNITION ON EVERY TESTING IMAGE
				for (unsigned l = 0; l < phiTesting.size(); ++l)
				{
					//GENERATE IMAGE'S W VECTOR
					VectorXd wVector(K);
					VectorXd phi_hat = VectorXd::Zero(IMG_VEC_LEN);
					for (unsigned j = 0; j < K; j++)
					{
						MatrixXd u_t = ((MatrixXd)topEigenVectors[j]).transpose();
						MatrixXd phi_i = (phiTesting[l]).cast<double>();
						double w = (u_t * phi_i)(0, 0);
						wVector(j) = w;
				 		//cout << w << " ";
					}
					//DONE


					queue<int> identifierQueue;
					queue<double> errorQueue;
					//cout << "calculating e_r..." << endl;
					double e_r;
					//Compare omegaVectors for norm. min of all those is e_r
					for (int i = 0; i < NUM_SAMPLES; ++i)
					{
						double diffSum=0;
						for (unsigned j = 0; j < K; ++j)
						{
							//(image w - database image w)^2
							diffSum += (wVector(j) - omegaVectors[i](j)) * (wVector(j) - omegaVectors[i](j));
						}

						if(i==0)
						{
							e_r = diffSum;
							errorQueue.push(e_r);
							identifierQueue.push(ImageVector[i].GetIdentifier());
						}
						else if(e_r > diffSum)
						{
							//cout << "New min-> " << e_r << endl;
							e_r = diffSum;

							errorQueue.push(e_r);
							identifierQueue.push(ImageVector[i].GetIdentifier());

							if(errorQueue.size() > N)
							{
								errorQueue.pop();
								identifierQueue.pop();
							}
						}
						//cout << "e_r: " << e_r << endl;
					}

					bool foundCorrectImage = false;
					while(!errorQueue.empty()) //(unsigned i = 0; i < errorQueue.size(); ++i)
					{
						int tempInt = identifierQueue.front();
						identifierQueue.pop();
						//int tempDouble = errorQueue.front();
						errorQueue.pop();
						//cout << "QUEUE (e_r, id): " << tempDouble << " | " << tempInt << endl;
						//cout << "identifierQueue[i]: " << tempInt << endl;
						//cout << "TestImageVector[l].GetIdentifier(): " << TestImageVector[l].GetIdentifier() << endl;
						if(tempInt == TestImageVector[l].GetIdentifier() && !foundCorrectImage)
						{
							if(N == 1)
							{
								if(tempInt == ImageVector[l].GetIdentifier())
								{
									fn1TP << "Image id that was correctly identified " << tempInt << endl;
									fn1TP << "Correctly identified training image ID: " << ImageVector[l].GetIdentifier() << endl;
								}
									fn1FP << "Image id that was correctly identified " << tempInt << endl;
									fn1FP << "Correctly identified training image ID: " << ImageVector[l].GetIdentifier() << endl;
							}
							foundCorrectImage = true;
							//cout << "Identified correct image!" << endl;
						}
					}

					//fout_results << foundCorrectImage << endl;

					if(foundCorrectImage)
					{
						correctlyIdentifiedImages++;
					}
					else
					{
						incorrectlyIdentifiedImages++;
					}
				}

				int sumID = correctlyIdentifiedImages+incorrectlyIdentifiedImages;
				float performance = (float)correctlyIdentifiedImages / (float)sumID;

				fout_results << N << " " << correctlyIdentifiedImages << " " << incorrectlyIdentifiedImages << " " << performance << endl;

				cout << "correctlyIdentifiedImages: " << correctlyIdentifiedImages << endl;
				cout << "incorrectlyIdentifiedImages: " << incorrectlyIdentifiedImages << endl;
				//DONE

				cout << "DONE" << endl;
			}
			fn1TP.close();
			fn1FP.close();
				fout_results.close();
		}
		else if (inputString == "7")
		{
			TestImageVector = obtainTrainingFaces(testingDataset, IMG_W, IMG_H);

			for (int i = 0; i < NUM_TEST_SAMPLES; ++i)
			{
				phiTesting.push_back(TestImageVector[i].getFaceVector() - avgFaceVector);
			}

			//LOAD DATASET'S OMEGA VECTORS
			vector<VectorXd> omegaVectors_training, omegaVectors_testing;
			string omegaVectorsFile = "omegaVectors.txt";
			ifstream fin_omegas(omegaVectorsFile.c_str());

			fin_omegas >> K; // read the K value from the file
			
			VectorXd readVector(K);

			for (int i = 0; i < NUM_SAMPLES; ++i)
			{
				for (unsigned j = 0; j < K; ++j)
				{
					fin_omegas >> readVector(j);
				}
				omegaVectors_training.push_back(readVector);
			}
			fin_omegas.close();
			//DONE

			//LOAD TOP K EIGENVECTORS
			vector<VectorXd> topEigenVectors;
			ifstream fin_vecs;
			VectorXd eigenVector(IMG_VEC_LEN);

			fin_vecs.open(eigenvectorsFile.c_str());
			unsigned k = 0;
			while(!fin_vecs.eof() && k < K)
			{
				for (unsigned i = 0; i < IMG_VEC_LEN; i++)
				{
					fin_vecs >> eigenVector(i);
				}
				
				topEigenVectors.push_back(eigenVector);
				k++;
			}
			fin_vecs.close();
			//DONE

			//RUN RECOGNITION ON EVERY TESTING IMAGE
			for (unsigned l = 0; l < phiTesting.size(); ++l)
			{
				//GENERATE TEST IMAGE'S W VECTOR
				VectorXd wVector(K);
				VectorXd phi_hat = VectorXd::Zero(IMG_VEC_LEN);

				for (unsigned j = 0; j < K; j++)
				{
					MatrixXd u_t = ((MatrixXd)topEigenVectors[j]).transpose();
					MatrixXd phi_i = (phiTesting[l]).cast<double>();
					double w = (u_t * phi_i)(0, 0);
					wVector(j) = w;
				}

				omegaVectors_testing.push_back(wVector);
			}

			ofstream fout_results;
			string recognitionFile = "TP_rate_vs_FP_rate.txt";
			fout_results.open(recognitionFile.c_str());

			//HEADER OF OUTPUT FILE
			fout_results << "FP_rate TP_rate threshold" << endl;

			unsigned i = 0, threshold = 0;
			unsigned maxThreshold = 2840000;
			unsigned step = 40000;

			while ((threshold <= maxThreshold))
			{
				int	numFalsePos = 0,
					numIntruders = 0,
					numTruePos = 0,
					numNonIntruders = 0;

				float FP_rate, TP_rate;

				for (unsigned l = 0; l < omegaVectors_testing.size(); ++l)
				{
					double e_r = numeric_limits<double>::max();
					int trainingImageID;

					//Compare omegaVectors for norm. min of all those is e_r
					for (int i = 0; i < NUM_SAMPLES; ++i)
					{
						double diffSum = 0;

						for (unsigned j = 0; j < K; ++j)
						{
							//(image w - database image w)^2
							diffSum += pow(omegaVectors_testing[l](j) - omegaVectors_training[i](j), 2.0);
						}

						if (diffSum < e_r)
						{
							e_r = diffSum;
							trainingImageID = ImageVector[i].GetIdentifier();
						}
					}

					bool weAccept = (e_r < threshold);
					bool idsMatch = (trainingImageID == TestImageVector[l].GetIdentifier());

					if (idsMatch)
					{
						numNonIntruders++;
					}
					else
					{
						numIntruders++;
					}

					if (weAccept && idsMatch)
					{
						numTruePos++;
					}
					else if (weAccept && !idsMatch)
					{
						numFalsePos++;
					}
				}

				FP_rate = (float)numFalsePos / (float)numIntruders;
				TP_rate = (float)numTruePos / (float)numNonIntruders;

				fout_results << FP_rate << " " << TP_rate << " " << threshold << endl;

				i++;
				threshold += step;
			}

			fout_results.close();
		}

		cout << endl;
	} while (inputString != "-1");
}

void getMinMax(VectorXd m, double& min, double& max)
{
	min = 1, max = -1;
	for (int j = 0; j < IMG_H; j++)
	{
		for (int k = 0; k < IMG_W; k++)
		{
			if(m(j * IMG_W + k) < min)
			{
				min = m(j * IMG_W + k);
			}
			if(m(j * IMG_W + k) > max)
			{
				max = m(j * IMG_W + k);
			}
		}
	}
}

void PrintMatrix(MatrixXd m, int dimension1, int dimension2)
{
	for (int i = 0; i < dimension1; ++i)
	{
		for (int j = 0; j < dimension2; ++j)
		{
			cout << m(i, j) << " ";
		}
		cout << endl;
	}
}

vector<Image> obtainTrainingFaces(string directory, int imageWidth, int imageHeight)
{
	vector<Image> returnVector;
	MatrixXd avgFaceMatrix(imageHeight, imageWidth);
	int k, j, M, N, Q;
 	bool type;
	int val;
	DIR *dir;
	struct dirent *ent;

	cout << "Obtaining training faces..." << endl;

	if ((dir = opendir(directory.c_str())) != NULL) 
	{
		while ((ent = readdir(dir)) != NULL) 
		{
			string temp, fileName = ent->d_name;
			int imageID, imageDate, i;
			string imageDataset;
			bool imageGlasses;
			
			for (i = 0; i < ID_LENGTH; ++i)
			{
			    temp += fileName[i];
			}
			imageID = atoi(temp.c_str());
			i++;

			temp = "";
			for (; i < (ID_LENGTH + DATE_LENGTH + 1); ++i)
			{
				temp += fileName[i];
			}
			imageDate = atoi(temp.c_str());
			i++;

			temp = "";
			for (; i < (ID_LENGTH + DATE_LENGTH + SET_LENGTH + 2); ++i)
			{
				temp += fileName[i];
			}
			imageDataset = temp;

			if(fileName[i] == '.')
			{
				imageGlasses = false;
			}
			else
			{
				imageGlasses = true;
			}

			Image currentImage;
			currentImage.SetFileName(ent->d_name);
			currentImage.SetIdentifier(imageID);
			currentImage.SetDateTaken(imageDate);
			currentImage.SetWearingGlasses(imageGlasses);		
			
			string currentFile = directory + '/' + ent->d_name;

			if (fileName != "." && fileName != "..")
			{
				readImageHeader((char*) currentFile.c_str(), N, M, Q, type);

				// allocate memory for the image array
				ImageType tempImage(N, M, Q);

 				readImage((char*) currentFile.c_str(), tempImage);

 				VectorXi imageVector(N * M);
				
				for (k = 0; k < N; k++)
				{
					for (j = 0; j < M; j++)
					{
			            tempImage.getPixelVal(k, j, val);
			            imageVector(k * M + j) = val;
					}
				}

				currentImage.setFaceVector(imageVector);

				returnVector.push_back(currentImage);
			}
		}

		closedir(dir);

		cout << "Finished obtaining training faces." << endl;
	} 
	else
	{
	 	cout << "Error: Could not open directory " << directory << endl;
	}

	return returnVector;
}

VectorXi compAvgFaceVec(const vector<Image> &imageVector)
{
	VectorXi result = VectorXi::Zero(IMG_VEC_LEN);

	for (int i = 0; i < NUM_SAMPLES; i++)
	{
		result += imageVector[i].getFaceVector();
	}

	result /= NUM_SAMPLES;

	return result;
}

void Image::ExtractEigenVectors(MatrixXd m, int dimension1, int dimension2)
{
	MatrixXd m_tm = m.transpose() * m;

	//cout << "m.transpose*m:" << endl;
	//PrintMatrix(m_tm, m_tm.rows(), m_tm.cols());

	EigenSolver<MatrixXd> EigenSolver;
	EigenSolver.compute(m_tm, true); //Initializes eigensolver with something to de-compose
	eigenVectors = EigenSolver.eigenvectors();
	// cout << endl << "EigenVector Matrix: " << endl << eigenVectors << endl;

	eigenValues = EigenSolver.eigenvalues();
	// cout << endl << "EigenValues Matrix: " << endl << eigenValues << endl;
}