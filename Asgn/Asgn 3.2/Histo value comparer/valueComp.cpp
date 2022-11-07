#include <iostream> 
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <iomanip>

int main()
{
  std::ifstream ifs{"./histocdf-nvidia.txt"};

  std::vector<int> bucket(256);
  std::vector<float> cdf(256);

  bool isCDF{};
  int counter{};

  std::string line{};
  while (std::getline(ifs, line))
  {
    if (line == "cdf") 
    {
      isCDF = true;
      counter = 0;
    }

    else if (!isCDF)
    {
      std::stringstream ss{line};
      int val{};
      while (ss >> val)
        bucket[counter++] = val;
    }

    else
    {
      std::stringstream ss{line};
      float val{};
      while (ss >> val)
        cdf[counter++] = val;
    }
  }
  ifs.close();

  // std::ofstream ofs{"./williamVal.txt"};
  // ofs << "Element, histo.bin, histo.cdf\n";
  // for (size_t i{}; i < bucket.size(); ++i)
  // {
  //   ofs << i << ", " << bucket[i] << ",  " << cdf[i] << '\n';
  // }
  // ofs.close();

  for (size_t i{1}; i < bucket.size(); ++i)
  {
    bucket[i] += bucket[i - 1];
  }

  float div = 1.0f / (512.0f * 512.0f);

  std::ofstream ofs{"./cpuCDF.txt"};
  ofs << "Element, histo.bin, histo.cdf\n";
  for (size_t i{}; i < bucket.size(); ++i)
  {
    ofs << i << ", " << bucket[i] << ",  " << (float)bucket[i] * div << '\n';
  }
  ofs.close();
}