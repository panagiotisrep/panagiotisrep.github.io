## Test Results for [Geometric sampling, volume and optimization](https://github.com/rstats-gsoc/gsoc2019/wiki/Geometric-sampling,-volume-and-optimization)


The test entailed implementing the probabilistic optimization algorithm from [here](http://www.optimization-online.org/DB_FILE/2008/12/2161.pdf). The code is in this [repository](https://github.com/panagiotisrep/volume_approximation).


<br>


The algorithm was implemented in simple_optimization.h

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
