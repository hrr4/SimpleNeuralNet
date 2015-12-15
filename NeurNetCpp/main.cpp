/*******************
 Simple NeuralNet
  -- A simple neural net with one hidden layer.
  -- Approximates a given function.
  -- Short and should be easy to follow.
 ******************/
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <fstream>

#define LEARN_LOOP 1
#define WRITE_OUT 0

const unsigned k = 20;
const unsigned n = 20;
const unsigned maxIter = 20000;

double eta = .01;
const double eps = .001;
const double C = 2;

typedef double(*ActivFunc) (double t);
typedef std::vector<double> NNArray;

/* Activation Functions */
double ApproxFunc(double x) {
  return sin(x);
}

double Activ(double t) {
  return tanh(t);
}

/* Gradient Descent Functions */

double Outputs(NNArray weight, ActivFunc phi, unsigned n, NNArray const & interval, double xhat_val) {
  double sum = 0;

  for (unsigned j = 1; j < n-1; ++j)
    sum += weight[j] * phi(xhat_val - interval[j]);

  return sum;
}

double Error(unsigned k, NNArray const & outs, NNArray const & yhat) {
  double sum = 0;

  for (unsigned i = 0; i < k-1; ++i)
    sum += (outs[i] - yhat[i]) * (outs[i] - yhat[i]);

  return sum;
}

double Slope(unsigned k, NNArray const & outs, NNArray const & yhat, ActivFunc phi, NNArray const & xhat, double xinterv) {
  double sum = 0;

  for (unsigned i = 0; i < k-1; ++i)
    sum += (outs[i] - yhat[i]) * phi(xhat[i] - xinterv);

  return sum * 2;
}

void Output(std::ostream & ostr, NNArray const & yhat, NNArray const & y) {
  // Actual Out
  ostr << "Actual Output: \n";
  for (unsigned i = 0; i < yhat.size(); ++i)
    ostr << yhat[i] << std::endl;

  ostr << std::endl;

  // Learned Out
  ostr << "Learned Output: \n";
  for (unsigned i = 0; i < y.size(); ++i)
    ostr << y[i] << std::endl;
}


/**** Variables ****/
NNArray xhat, xinterv, yhat, weights;
NNArray out(k-1, 0), errorweight(k-1, 0), slope(n-1, 0);

int main(int argc, char ** argv) {
  // Seed (for PRNG)
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::mt19937 prng(seed);
  std::uniform_real_distribution<double> dis(0.0, 1.0);

  // Initialize Variables
  for (unsigned i = 0, cnt = 1; i < k; ++i, ++cnt) {
    xhat.push_back(cnt / double(k));
    yhat.push_back(ApproxFunc(xhat[i]));

    // (Step 1) Generate random synaptic weights  
    weights.push_back(dis(prng));
  }

  for (unsigned j = 0; j < (n - 1); ++j)
    xinterv.push_back(j / double(n));


  /* Perform Learning */
  double prevE = 0;
  for (unsigned iter = 0; iter < maxIter; ++iter) {
    // (Step 2) Calculate outputs, given training data & Error
    for (unsigned i = 0; i < k-1; ++i)
      out[i] = Outputs(weights, Activ, n, xinterv, xhat[i]);

    double E = Error(k, out, yhat);

    // (Step 3) Calculate Partial Error w.r.t Synaptic Weight
    for (unsigned j = 0; j < k-1; ++j)
      errorweight[j] = Slope(k, out, yhat, Activ, xhat, xinterv[j]);

    // (Step 4) Update Synaptic Weights
    for (unsigned i = 0; i < n-1; ++i)
      weights[i] = weights[i] - (eta * errorweight[i]);

    // (Step 4.5?) Reduce Learning if increasing
    if (E > prevE)
      eta = eta / C;

    // (Step 5) Check for Early Termination
    if (E < eps)
      break;

    prevE = E;
  }
  
  /* (Step 6) Output */
  NNArray y = out;

  Output(std::cout, yhat, y);

#if WRITE_OUT
  std::ofstream ofs("output.txt", std::ofstream::out);

  Output(ofs, yhat, y);
  
  ofs.close();
#endif

  std::cout << "Press return to close." << std::endl;
  getchar();

  return 0;
}
