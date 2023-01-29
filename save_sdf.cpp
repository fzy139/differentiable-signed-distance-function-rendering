#include <iostream>
#include <vector>
#include <fstream>

using namespace std;
const int N = 256;

int main(int argc, char* argv[]) {
    std::string outname;

    string filename(argv[1]);

    fstream fin(filename);

    vector<float> vol(N * N * N);

    for (int i = 0; i < N * N * N; i++)
        fin >> vol[i];

    outname = filename.substr(0, filename.size()-4) + std::string(".sdf");
    std::cout << "Writing results to: " << outname << "\n";
    
    std::ofstream outfile( outname.c_str());
    outfile << "#sdf 1\n";
    outfile << "dim " << N << " " << N << " " << N << "\n";
    outfile << "data\n";

    char charv[4];
    float *floatv = reinterpret_cast<float*>(charv);
    for (int i = 0; i < N * N * N; i++) {
        *floatv = vol[i];
        outfile << charv[0] << charv[1] << charv[2] << charv[3];
    }
}