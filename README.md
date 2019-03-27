Hi, I am Panagiotis Repouskos, an aspiring student of GSoC 2019! These are the test results for R the project [Geometric sampling, volume and optimization](https://github.com/rstats-gsoc/gsoc2019/wiki/Geometric-sampling,-volume-and-optimization)



The test entailed implementing the probabilistic optimization algorithm from [here](http://www.optimization-online.org/DB_FILE/2008/12/2161.pdf) in c++. The code is in this [repository](https://github.com/panagiotisrep/volume_approximation).

<br>

Bellow folows a quick presentation of the problem, the implementation of the algorithm and a test main. You can find some example test files that were tested here: [tests](https://github.com/panagiotisrep/volume_approximation/tree/master/test/test_inputs).

<br>

### The problem at hand
The goal is to solve the following linear program:


min c*x

s.t. Ax<= b



The idea of the algorithm is to sample the polytope defined by the constraints, pick the "best" candidate from the sample and cut the polytope at that point. Then repeat. 


<br>


The algorithm was implemented in 
### simple_optimization.h

```c++
#ifndef GSOC_SIMPLE_OPTIMIZATION_H
#define GSOC_SIMPLE_OPTIMIZATION_H

#include "polytopes.h"
#include "Eigen"
#include <list>
#include <vector>

namespace simple_optimization {

    /**
     * Computes the dot product of the vector v with point
     *
     * @tparam Point class Point
     * @tparam NT The numeric type
     * @param v A vector of length equal to point's dimension
     * @param point An instance of class Point
     * @return dot product of the vector v with point
     */
    template<class Point, typename NT>
    NT dot_product(std::vector<NT> &v, Point &point) {
        NT sum = NT(0);

        for (typename std::vector<NT>::iterator vit = v.begin(), pit = point.iter_begin(); vit != v.end(); vit++, pit++)
            sum += *pit * *vit;

        return sum;
    }

    /**
     * Compute the cutting plane c (x - point) <= 0  ==>  cx <= c point
     * and add it as a new constraint in the existing polytope, in place of its last one, which will be redundant.
     * If sppedup is true, don't create the cutting plane on top of the point, but a bit further, so
     * we can use this point as an interior point for the next round.
     *
     * @tparam Point class Point
     * @tparam NT The numeric type
     * @param c a vector holding the coefficients of the object function
     * @param polytope An instance of Hpolytope
     * @param point Where to create the cutting plane
     * @param epsilon how much to "move back" the cutting plane
     * @param speedup if we want to move back the cutting plane
     */
    template<class Point, typename NT>
    void cutPolytope(std::vector<NT> c, HPolytope<Point> &polytope, Point point, NT epsilon, bool speedup) {

        unsigned int dim = polytope.dimension();

        //add cx in last row of A
        long j, i;
        typename std::vector<NT>::iterator it;

        for (j = 0, i = polytope.num_of_hyperplanes() - 1, it = c.begin(); j < dim; j++) {
            polytope.put_mat_coeff(i, j, *it);
            it++;
        }

        //add  < c,  point >  in last row of b
        NT _b = dot_product(c, point) + epsilon;
        polytope.put_vec_coeff(polytope.num_of_hyperplanes() - 1, _b);
    }


    /**
     * Find the point that minimizes the object function out of a list of points
     *
     * @tparam Point class Point
     * @tparam NT The numeric type
     * @param c a vector holding the coefficients of the object function
     * @param randPoints a list of points
     * @return point wich minimizes the object function
     */
    template<class Point, typename NT>
    Point getMinimizingPoint(std::vector<NT> c, std::list<Point> randPoints) {
        typename std::list<Point>::iterator it = randPoints.begin();

        NT temp, min;
        Point minPoint = *it;

        min = dot_product<Point, NT>(c, *it);
        it++;

        for (; it != randPoints.end(); it++) {
            temp = dot_product<Point, NT>(c, *it);

            if (temp < min) {
                min = temp;
                minPoint = *it;
            }
        }

        return minPoint;
    }

    /**
     * Adds one more hyperplane in the polytope. The values of the new polytope are not initialized.
     *
     * @tparam Point class Point
     * @tparam NT the numeric type
     * @param polytope an instance of class HPolytope
     */
    template<class Point, typename NT>
    void addRowInPolytope(HPolytope<Point>& polytope) {
        typedef Eigen::Matrix<NT, Eigen::Dynamic, Eigen::Dynamic> MT;
        typedef Eigen::Matrix<NT, Eigen::Dynamic, 1> VT;

        MT A = polytope.get_mat();
        VT b = polytope.get_vec();
        unsigned int dim = polytope.dimension();

        A.conservativeResize(A.rows() + 1, Eigen::NoChange);
        b.conservativeResize(b.rows() + 1);

        polytope.init(dim, A, b);
    }

    /**
     * Prints a message, if verbose is true
     *
     * @param verbose whether or not to print
     * @param msg a message to print
     */
    void print(bool verbose, const char* msg) {
        if (verbose)
            std::cout << msg << std:: endl;
    }

    /**
     * Solve the linear program
     *
     *      min cx
     *      s.t. Ax <= b
     *
     * using the optimization algorithm in http://www.optimization-online.org/DB_FILE/2008/12/2161.pdf
     *
     * @tparam Parameters struct vars
     * @tparam Point class Point
     * @tparam NT The numeric type
     * @param c a vector holding the coefficients of the object function
     * @param polytope An instance of class HPolytope
     * @param parameters An instance of struct vars
     * @param error how much distance between two successive estimations before we stop
     * @param masSteps maximum number of steps
     * @param speedup we speedup, but lose accuracy
     * @return A pair of the point that minimizes the object function and the minimum value
     */
    template<class Parameters, class Point, typename NT>
    std::pair<Point, NT>
    simple_optimization(std::vector<NT> c, HPolytope<Point> polytope, Parameters parameters, const NT error, const unsigned int maxSteps, bool speedup) {

        bool verbose = parameters.verbose;
        unsigned int rnum = parameters.m;
        unsigned int walk_len = parameters.walk_steps;
        bool tillConvergence = maxSteps == 0;
        unsigned int step = 1;

        print(verbose, "Starting Linear Program");

        // get an internal point so you can sample
        std::pair<Point, NT> InnerBall = polytope.ComputeInnerBall();
        Point feasiblePoint = InnerBall.first;

        // sample points from polytope
        std::list<Point> randPoints;
        rand_point_generator(polytope, feasiblePoint, rnum, walk_len, randPoints, parameters);

        // find where to cut the polytope
        feasiblePoint = getMinimizingPoint(c, randPoints);

        // add one more row in polytope, where we will store the current cutting plane
        addRowInPolytope<Point, NT> (polytope);

        // cut the polytope
        cutPolytope<Point, NT>(c, polytope, feasiblePoint, error, speedup);

        NT min = dot_product(c, feasiblePoint);

        do {
            if (!speedup) {
                InnerBall = polytope.ComputeInnerBall();
                feasiblePoint = InnerBall.first;
            }

            // sample points from polytope
            std::list<Point> randPoints;
            rand_point_generator(polytope, feasiblePoint, rnum, walk_len, randPoints, parameters);

            // find where to cut the polytope
            feasiblePoint = getMinimizingPoint(c, randPoints);

            NT newMin = dot_product(c, feasiblePoint);

            // check for distance between successive estimations
            NT distance = newMin - min;
            distance = distance > 0 ? distance : -distance;

            if (distance < error) {
                min = newMin;
                break;
            }
            else
                min = newMin;

            // add the cutting plane
            cutPolytope<Point, NT>(c, polytope, feasiblePoint, error, speedup);

            step++;

        } while (tillConvergence || step <= maxSteps);


        if (verbose) std::cout << "Ended at " << step << " steps" << std::endl;

        return std::pair<Point, NT>(feasiblePoint, min);
    }

}


#endif //GSOC_SIMPLE_OPTIMIZATION_H

```

<br>
<br>

And this main was used to test it: 

### opt.cpp
```c++

#include "Eigen/Eigen"

#define VOLESTI_DEBUG


#include <fstream>
#include "volume.h"
#include "sample_only.h"
#include "exact_vols.h"
#include "simple_optimization.h"

//////////////////////////////////////////////////////////
/**** MAIN *****/
//////////////////////////////////////////////////////////

typedef double NT;
typedef Cartesian<NT> Kernel;
typedef typename Kernel::Point Point;
typedef boost::mt19937 RNGType;
typedef HPolytope<Point> Hpolytope;
typedef std::pair<Point, NT> Result;

void printHelpMessage();

bool
readFromFile(const char *const *argv, bool verbose, HPolytope<point<Cartesian<double>::Self>> &HP, int &n, bool &file,
             int &i, std::vector<NT>& objectFunction);

void printResult(Result result, double time);

bool loadProgramFromStream(std::istream &is, HPolytope<Point> &HP, std::vector<NT>& objectFunction);


int main(const int argc, const char **argv) {

    // the object function is a vector

    std::vector<NT> objectFunction;
    //Deafault values
    int dimensinon, numOfExperinments = 1, walkLength = 10, numOfRandomPoints = 16, nsam = 100, numMaxSteps = 100;
    NT e = 1;
    bool speedup = false;

    bool verbose = false,
            rand_only = false,
            round_only = false,
            file = false,
            round = false,
            NN = false,
            user_walk_len = false,
            linear_extensions = false,
            birk = false,
            rotate = false,
            ball_walk = false,
            ball_rad = false,
            experiments = true,
            annealing = false,
            Vpoly = false,
            Zono = false,
            cdhr = false,
            rdhr = true, // for hit and run
            exact_zono = false,
            gaussian_sam = false;


    Hpolytope HP;

    NT delta = -1.0, error = 0.2;
    NT distance = 0.000001;

    //parse command line input vars
    for (int i = 1; i < argc; ++i) {
        bool correct = false;

        if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) {
            printHelpMessage();
            return 0;
        }

        if (!strcmp(argv[i], "-s")) {
            speedup = true;
            correct = true;
        }

        if (!strcmp(argv[i], "-v") || !strcmp(argv[i], "--verbose")) {
            verbose = true;
            std::cout << "Verbose mode\n";
            correct = true;
        }

        if (!strcmp(argv[i], "-f") || !strcmp(argv[i], "--file")) {
            readFromFile(argv, verbose, HP, dimensinon, file, i, objectFunction);
            correct = true;
        }

        if (!strcmp(argv[i], "-e") || !strcmp(argv[i], "--error")) {
            e = atof(argv[++i]);
            distance = e;
            correct = true;
        }
        if (!strcmp(argv[i], "-w") || !strcmp(argv[i], "--walk_len")) {
            walkLength = atoi(argv[++i]);
            user_walk_len = true;
            correct = true;
        }

        if (!strcmp(argv[i], "-r")) {
            numOfRandomPoints = atoi(argv[++i]);
            correct = true;
        }

        if (!strcmp(argv[i], "-exp")) {
            numOfExperinments = atoi(argv[++i]);
            correct = true;
        }

        if (!strcmp(argv[i], "-k")) {
            numMaxSteps = atoi(argv[++i]);
            correct = true;
        }

        if (!correct) {
            std::cerr << "unknown parameters \'" << argv[i] <<
                      "\', try " << argv[0] << " --help" << std::endl;
            exit(-2);
        }

    }


    // Timings
    double tstart, tstop;

    /* CONSTANTS */
    //error in hit-and-run bisection of P
    const NT err = 0.0000000001;
    const NT err_opt = 0.0000001;


    // If no file specified construct a default polytope
    if (!file) {
        std::cout << "You must specify a file - type -h for help" << std::endl;
        exit(-2);
    }


    /* RANDOM NUMBERS */
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    RNGType rng(seed);
    boost::normal_distribution<> rdist(0, 1);
    boost::random::uniform_real_distribution<>(urdist);
    boost::random::uniform_real_distribution<> urdist1(-1, 1);


    //RUN EXPERIMENTS
    std::vector<double> times;
    std::vector<Result> results;
    NT average, std_dev;
    NT sum = NT(0);
    std::cout.precision(7);


    if (verbose && HP.num_of_hyperplanes() < 100) {
        std::cout << "Input polytope is of dimension: " << dimensinon << std::endl;
        HP.print();
    }

    for (unsigned int i = 0; i < numOfExperinments; ++i) {
        std::cout << "Experiment " << i + 1 << std::endl;
        tstart = (double) clock() / (double) CLOCKS_PER_SEC;


        // Setup the parameters
        vars<NT, RNGType> var(numOfRandomPoints, dimensinon, walkLength, 1, err, e, 0, 0.0, 0, 0, rng,
                              urdist, urdist1, delta, verbose, rand_only, round, NN, birk, ball_walk, cdhr, rdhr);



        Result result = simple_optimization::simple_optimization(objectFunction, HP, var, distance, numMaxSteps, speedup);

        tstop = (double) clock() / (double) CLOCKS_PER_SEC;

        printResult(result, tstop - tstart);

        results.push_back(result);
        times.push_back(tstop - tstart);
        sum += result.second;
    }


    return 0;
}


void printResult(Result result, double time) {
    std::cout << "Min value is: " << result.second << std::endl <<
    "coords: ";
    result.first.print();

    std::cout << "Computed at " << time << " secs" << std::endl << std::endl;
}

bool
readFromFile(const char *const *argv, bool verbose, HPolytope<point<Cartesian<double>::Self>> &HP, int &n, bool &file,
             int &i, std::vector<NT>& objectFunction) {
    file = true;
    std::cout << "Reading input from file..." << std::endl;
    std::ifstream inp;
    inp.open(argv[++i], std::ios_base::in);
    bool retval = loadProgramFromStream(inp, HP, objectFunction);
    n = HP.dimension();
//    HP.print();
    return retval;
}

void printHelpMessage() {
    std::cerr <<
              "Usage: The constraints are passed in a file, as in vol.cpp while the object function is declared in main\n" <<
              "-v, --verbose \n" <<
              "-r [#num]: the number of points to sample at each k (default 16)\n" <<
              "-k [#num]: the number of maximum iterations (default 100) - if 0 runs until convergence\n" <<
              "-f, --file [filename] The file must be of the following format:\n" <<
              "\t #dimension\n" <<
              "\t object function\n" <<
              "\t #num_of_constaints\n" <<
              "\t constraints\n" <<
              "-e, --error epsilon : the goal error of approximation\n" <<
              "-w, --walk_len [walk_len] : the random walk length (default 10)\n" <<
              "-exp [#exps] : number of experiments (default 1)\n" <<
              "-d [#distance] : stop if successive estimations are less than (default 0.000001)\n" <<
              "-s: faster but no guarantees for approximation - depends on -r -w -k \n" <<
              std::endl;
}

bool loadProgramFromStream(std::istream &is, HPolytope<Point> &HP, std::vector<NT>& objectFunction){

    std::string line;
    std::string::size_type sz;

    //read dimension
    if (std::getline(is, line, '\n').eof())
        return false;

    int dim = std::stoi(line.substr(sz));


    //read object function
    if (std::getline(is, line, '\n').eof())
        return false;

    NT num = std::stod(line, &sz);
    objectFunction.push_back(num);

    for (int j=2 ; j<=dim  ; j++) {
        line = line.substr(sz);
        num = std::stod(line, &sz);
        objectFunction.push_back(num);
    }


    if (std::getline(is, line, '\n').eof())
        return false;

    //read number of constraints
    int constraintsNum = std::stoi(line, &sz);

    typedef Eigen::Matrix<NT,Eigen::Dynamic,Eigen::Dynamic> MT;
    typedef Eigen::Matrix<NT,Eigen::Dynamic,1> VT;

    VT b = HP.get_vec();
    MT A = HP.get_mat();

    A.resize(constraintsNum, dim);
    b.resize(constraintsNum);

    // for each constraint
    for (int i=1 ; i<=constraintsNum ; i++) {

        if (std::getline(is, line, '\n').eof())
            return false;

        // read first line of A

        NT num = std::stod(line, &sz);
        A(i - 1, 0) = num;

        for (int j=2 ; j<=dim  ; j++) {
            line = line.substr(sz);
            num = std::stod(line, &sz);
            A(i - 1, j - 1) = num;
        }

        //read first row of b
        line = line.substr(sz);
        num = std::stod(line, &sz);
        b(i - 1) = num;
    }

    HP.init(dim, A, b);
    return true;
}
```
