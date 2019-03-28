Hi, I am Panagiotis Repouskos, an aspiring student of GSoC 2019! These are the test results for R the project [Geometric sampling, volume and optimization](https://github.com/rstats-gsoc/gsoc2019/wiki/Geometric-sampling,-volume-and-optimization)



The test entailed implementing the probabilistic optimization algorithm from [here](http://www.optimization-online.org/DB_FILE/2008/12/2161.pdf) in c++. The code is in this [repository](https://github.com/panagiotisrep/volume_approximation).

<br>

Bellow folows a quick presentation of the problem, the implementation of the algorithm and a test main. You can find some example test files that were tested here: [tests](https://github.com/panagiotisrep/volume_approximation/tree/master/test/test_inputs).

<br>

### The problem at hand
The goal is to solve an optimization problem, to minimize a linear function over a convex set, which is more general than linear programming. Our formulation is:


min c*x

s.t. Ax<= b



The idea of the algorithm is to sample the polytope defined by the constraints, pick the "best" candidate from the sample and cut the polytope at that point. Then repeat. 


The bottleneck for this algorithm is the sampling method, both cost and accuracy wise. Currently, I use hit & run, where to select a direction vector, I select a random point on a hypersphere and then multiply that with a matrix (the computational bottleneck). This is the implicit isotropization technique.


<br>


The algorithm was implemented in 
### simple_optimization.h

```c++
//
// Created by panagiotis on 24/3/2019.
//

#ifndef GSOC_SIMPLE_OPTIMIZATION_H
#define GSOC_SIMPLE_OPTIMIZATION_H

#include "polytopes.h"
#include "Eigen"
#include <list>
#include <vector>
#include <armadillo>
#include <complex>
#include "samplers.h"



namespace simple_optimization {

    typedef arma::mat MAT;
    typedef arma::cx_mat CXMAT;

    typedef double NT_MATRIX;
    typedef Eigen::Matrix<NT_MATRIX, Eigen::Dynamic, Eigen::Dynamic> MT;
    typedef Eigen::Matrix<NT_MATRIX, Eigen::Dynamic, 1> VT;


    ///////////////////////////////////////////////////////////////
    /// Some functions to transfer data - between eigen - armadillo
    //////


    void matFromEigenMatrix(MT &m, MAT &A) {
        long d = m.rows();
        A.set_size(d, d);

        for (int i = 0; i < d; i++) {
            for (int j = 0; j < d; j++) {
                A(i, j) = m(i, j);
            }
        }
    }

    void matrixFromMAT(MT &m, CXMAT A) {
        long d = A.n_rows;
        m.resize(d, d);

        for (int i = 0; i < d; i++) {
            for (int j = 0; j < d; j++) {
                m(i, j) = std::real(A(i, j));
            }
        }

    }


    template<typename NT>
    void vecFromVector(VT &v, std::vector<NT> vector) {
        v.resize(vector.size());
        int i = 0;

        for (typename std::vector<NT>::iterator pit = vector.begin(); pit != vector.end(); pit++)
            v(i++) = *pit;
    }

    template<class Point>
    void pointFromVT(VT &v, Point &point) {
        int i = 0;

        for (typename std::vector<typename Point::FT>::iterator pit = point.iter_begin();
             pit != point.iter_end(); pit++)
            *pit = v(i++);
    }


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
     */
    template<class Point, typename NT>
    void cutPolytope(std::vector<NT> &c, HPolytope<Point> &polytope, Point &point) {

        unsigned int dim = polytope.dimension();

        //add cx in last row of A
        long j, i;
        typename std::vector<NT>::iterator it;

        for (j = 0, i = polytope.num_of_hyperplanes() - 1, it = c.begin(); j < dim; j++) {
            polytope.put_mat_coeff(i, j, *it);
            it++;
        }

        //add  < c,  point >  in last row of b
        NT _b = dot_product(c, point);
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
    Point getMinimizingPoint(std::vector<NT> &c, std::list<Point> &randPoints) {
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
     * Find the point that second minimizes the object function out of a list of points
     *
     * @tparam Point class Point
     * @tparam NT The numeric type
     * @param c a vector holding the coefficients of the object function
     * @param randPoints a list of points
     * @return a pair of points which minimize the object function
     */
    template<class Point, typename NT>
    std::pair<Point, Point> getPairMinimizingPoint(std::vector<NT> &c, std::list<Point> &randPoints) {
        typename std::list<Point>::iterator it = randPoints.begin();

        NT temp, min, min2;
        Point minPoint = *it;

        min = dot_product<Point, NT>(c, *it);

        it++;


        min2 = dot_product<Point, NT>(c, *it);
        Point minPoint2 = *it;

        if (min2 > min)
            minPoint2 = *it;
        else {
            temp = min2;
            min2 = min;
            min = temp;
            Point tempPoint = minPoint;
            minPoint = minPoint2;
            minPoint2 = tempPoint;
        }

        it++;

        for (; it != randPoints.end(); it++) {
            temp = dot_product<Point, NT>(c, *it);

            if (temp < min2) {
                if (temp > min) {
                    min2 = temp;
                    minPoint2 = *it;
                } else {
                    min2 = min;
                    min = temp;
                    minPoint2 = minPoint;
                    minPoint = *it;
                }
            }
        }

        return std::pair<Point, Point>(minPoint, minPoint2);
    }

    /**
     * Adds one more hyperplane in the polytope. The values of the new polytope are not initialized.
     *
     * @tparam Point class Point
     * @tparam NT the numeric type
     * @param polytope an instance of class HPolytope
     */
    template<class Point, typename NT>
    void addRowInPolytope(HPolytope<Point> &polytope) {
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
    void print(bool verbose, const char *msg) {
        if (verbose)
            std::cout << msg << std::endl;
    }


    /**
     * Returns the arithmetic mean of points
     *
     * @tparam Point Class Point
     * @tparam NT The numeric type
     * @param c c a vector holding the coefficients of the object function
     * @param points A collection of points
     * @param pointsPair a pair of point that will participate in the mean
     */
    template<class Point, typename NT>
    Point getArithmeticMean(std::vector<NT> &c, std::list<Point> &points, std::pair<Point, Point> &pointsPair) {
        class std::list<Point>::iterator it = points.begin();

        NT b = dot_product(c, pointsPair.second);

        Point point(it->dimension());
        point = point + pointsPair.first;

        int i = 1;
        for (; it != points.end(); it++)
            if (dot_product(c, *it) <= b) {
                point = point + *it;
                i++;
            }

        point = point * (1.0 / (double) i);

        return point;
    }


    /**
     * Computes the quantites y = (1/N) * Sum {p | p in points}
     * and Y = (1/N) * Sum [(p-y)* (p -y)^T], p in points
     *
     * These quantities will be used to set the direction vector of hit and run
     *
     * @tparam Point class Point
     * @tparam NT The numeric type
     * @param points a collection of points
     * @return sqrt(Y) and the arithmetic mean of the points inside the polytope
     */
    template<class Point, typename NT>
    std::pair<Point, MT>
    getIsotropicQuantities(std::vector<NT> &c, std::list<Point> &points, std::pair<Point, Point> &pointsPair) {
        class std::list<Point>::iterator it = points.begin();
        int dim = it->dimension();

        NT lastEstimation = dot_product(c, pointsPair.second);

        Point point(it->dimension());
        point = point + pointsPair.first;

        VT _point = point.getEigenVector();

        std::vector<VT> _points;
        VT _c;
        vecFromVector(_c, c);

        for (it = points.begin(); it != points.end(); it++)
            _points.push_back(it->getEigenVector());

        VT _rest;
        VT _sum;
        _rest.setZero(dim);
        _sum.setZero(dim);

        int pointsInSum = 1;
        for (std::vector<VT>::iterator vit = _points.begin(); vit != _points.end(); vit++) {
            if (_c.dot(*vit) <= lastEstimation) {
                _sum = _sum + *vit;
                pointsInSum++;
            } else
                _rest = _rest + *vit;
        }

        _point = (_point + _sum) / (double) pointsInSum;
        pointFromVT(_point, point);
        VT _y(dim);

        _y = (_sum + _rest) / (double) _points.size();


        // compute Y
        MT _Y(dim, dim);
        _Y.setZero();

        for (std::vector<VT>::iterator vit = _points.begin(); vit != _points.end(); vit++) {
            *vit = *vit - _y;
            _Y = _Y + (*vit * vit->transpose());
        }

        _Y = _Y / (double) _points.size();

        MAT A;
        matFromEigenMatrix(_Y, A);
        matrixFromMAT(_Y, sqrtmat(A));

        return std::pair<Point, MT>(point, _Y);
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
    simple_optimization(std::vector<NT> c, HPolytope<Point> polytope, Parameters parameters, const NT error,
                        const unsigned int maxSteps, bool speedup) {

        bool verbose = parameters.verbose;
        unsigned int rnum = parameters.m;
        unsigned int walk_len = parameters.walk_steps;
        bool tillConvergence = maxSteps == 0;
        unsigned int step = 1;

        print(verbose, "Starting Linear Program");

        // get an internal point so you can sample
        std::pair<Point, NT> InnerBall = polytope.ComputeInnerBall();
        Point interiorPoint = InnerBall.first;

        std::pair<Point, Point> interiorPoints;

        // sample points from polytope
        std::list<Point> randPoints;

        // the points at the end of the segments of hit and run
        std::list<Point> endPoints;

        rand_point_generator(polytope, interiorPoint, rnum, walk_len, randPoints, endPoints, parameters);

        // find where to cut the polytope
        interiorPoints = getPairMinimizingPoint(c, randPoints);

        // add one more row in polytope, where we will store the current cutting plane
        addRowInPolytope<Point, NT>(polytope);

        // cut the polytope
        cutPolytope<Point, NT>(c, polytope, interiorPoint);

        NT min = dot_product(c, interiorPoint);

        do {
            // delete elements from previous rep
            randPoints.clear();

            // sample points from polytope
            if (speedup) {
                // use this point as the next starting point for sampling
                interiorPoint = getArithmeticMean(c, endPoints, interiorPoints);

                endPoints.clear();
                rand_point_generator(polytope, interiorPoint, rnum, walk_len, randPoints, endPoints, parameters);
            } else {
                // matrix isotropic will be multiplied witch each direction vector of hit and run

                std::pair<Point, MT> isotropic = getIsotropicQuantities<Point, NT>(c, endPoints, interiorPoints);
                endPoints.clear();
                smart_rand_point_generator(polytope, isotropic.first, rnum, walk_len, randPoints, endPoints, parameters,
                                           isotropic.second);
            }

            // find where to cut the polytope
            interiorPoints = getPairMinimizingPoint(c, randPoints);

            NT newMin = dot_product(c, interiorPoints.first);

            // check for distance between successive estimations
            NT distance = newMin - min;
            distance = distance > 0 ? distance : -distance;

            if (distance < error) {
                min = newMin;
                break;
            } else
                min = newMin;

            // add the cutting plane
            cutPolytope<Point, NT>(c, polytope, interiorPoints.second);

            step++;

        } while (tillConvergence || step <= maxSteps);


        if (verbose) std::cout << "Ended at " << step << " steps" << std::endl;

        return std::pair<Point, NT>(interiorPoints.first, min);
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
#include "solve_lp.h"

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
NT solveWithLPSolve(HPolytope<Point>& HP, std::vector<NT> objectFunction);

int main(const int argc, const char **argv) {

    // the object function is a vector

    std::vector<NT> objectFunction;
    //Deafault values
    int dimensinon, numOfExperinments = 1, walkLength = 10, numOfRandomPoints = 16, nsam = 100, numMaxSteps = 100;
    NT e = 1;
    bool speedup = false;
    bool uselpSolve = false;

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
    NT distance = 0.0001;

    //parse command line input vars
    for (int i = 1; i < argc; ++i) {
        bool correct = false;

        if (!strcmp(argv[i], "-lpsolve")) {
            uselpSolve = true;
            correct = true;
        }

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

    if (uselpSolve) {
        std::cout << "Using lp_solve" << std::endl;

        tstart = (double) clock() / (double) CLOCKS_PER_SEC;

        NT min = solveWithLPSolve(HP, objectFunction);

        tstop = (double) clock() / (double) CLOCKS_PER_SEC;

        std::cout << "Min is " << min << std::endl << "Computed at: " << tstop - tstart << " secs" << std::endl;
        return 0;
    }

    /* CONSTANTS */
    //error in hit-and-run bisection of P
    const NT err = 0.0000000001;
    const NT err_opt = 0.0000001;


    // If no file specified
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
    inp.close();
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

    int dim = std::stoi(line);


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

    VT b;
    MT A;

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

NT solveWithLPSolve(HPolytope<Point>& HP, std::vector<NT> objectFunction) {
    lprec *lp;
    unsigned int dim = HP.dimension();

    REAL row[1 + dim]; /* must be 1 more then number of columns ! */

    /* Create a new LP model */
    lp = make_lp(0, dim);

    typedef Eigen::Matrix<NT,Eigen::Dynamic,Eigen::Dynamic> MT;
    typedef Eigen::Matrix<NT,Eigen::Dynamic,1> VT;

    VT b = HP.get_vec();
    MT A = HP.get_mat();
    std::vector<NT>::iterator it = objectFunction.begin();

    for (int j=1 ; j<=dim ; j++)
        row[j] = *(it++); //j must start at 1

    set_obj_fn(lp, row);

    set_add_rowmode(lp, TRUE);

    for (int i=0 ; i<A.rows() ; i++) {
        for (int j=1 ; j<=dim ; j++)
            row[j] = A(i, j-1); //j must start at 1

        add_constraint(lp, row, LE, b(i)); /* constructs the row: +v_1 +2 v_2 >= 3 */
    }

    set_add_rowmode(lp, FALSE);
    set_minim(lp);

    solve(lp);
    NT ret = get_objective(lp);
    delete_lp(lp);
    return ret;
}
```
